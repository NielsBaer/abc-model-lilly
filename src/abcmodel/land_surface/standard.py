from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..models import (
    AbstractLandSurfaceModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_esat, get_qsat


@dataclass
class StandardLandSurfaceInitConds:
    """Data class for standard land surface model initial conditions.

    Args:
        alpha: slope of the light response curve [mol J-1].
        wg: soil moisture content in the root zone [m3 m-3].
        w2: soil moisture content in the deep layer [m3 m-3].
        temp_soil: soil temperature [K].
        temp2: deep soil temperature [K].
        surf_temp: Surface temperature [K].
        wl: liquid water storage on the canopy [m].
        rs: surface resistance [m s-1].
        rssoil: soil resistance [m s-1].
        cliq: wet fraction of the canopy [-].
        temp_soil_tend: soil temperature tendency [K s-1].
        wgtend: soil moisture tendency [m3 m-3 s-1].
        wltend: canopy water storage tendency [m s-1].
        le_veg: latent heat flux from vegetation [W m-2].
        le_liq: latent heat flux from liquid water [W m-2].
        le_soil: latent heat flux from soil [W m-2].
        le: total latent heat flux [W m-2].
        hf: sensible heat flux [W m-2].
        gf: ground heat flux [W m-2].
        le_pot: potential latent heat flux [W m-2].
        le_ref: reference latent heat flux [W m-2].
        ra: aerodynamic resistance [s m-1].
    """

    # the following variables are expected to be initialized by the user
    # here alpha is in a fact a parameter, but it is also used in different models
    alpha: float
    wg: float
    w2: float
    temp_soil: float
    temp2: float
    surf_temp: float
    wl: float

    # the following variables are initialized to high values and
    # are expected to converge during warmup
    rs: float = 1.0e6
    rssoil: float = 1.0e6

    # the following variables are expected to be assigned during warmup
    cliq: float = jnp.nan
    temp_soil_tend: float = jnp.nan
    wgtend: float = jnp.nan
    wltend: float = jnp.nan
    le_veg: float = jnp.nan
    le_liq: float = jnp.nan
    le_soil: float = jnp.nan
    le: float = jnp.nan
    hf: float = jnp.nan
    gf: float = jnp.nan
    le_pot: float = jnp.nan
    le_ref: float = jnp.nan
    ra: float = jnp.nan


class AbstractStandardLandSurfaceModel(AbstractLandSurfaceModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics.

    Args:
        a: Clapp and Hornberger (1978) retention curve parameter.
        b: Clapp and Hornberger (1978) retention curve parameter.
        p: Clapp and Hornberger (1978) retention curve parameter.
        cgsat: Saturated soil heat capacity [J m-3 K-1].
        wsat: Saturated soil moisture content [m3 m-3].
        wfc: Soil moisture content at field capacity [m3 m-3].
        wwilt: Soil moisture content at wilting point [m3 m-3].
        c1sat: saturated soil conductivity parameter [-].
        c2ref: reference soil conductivity parameter [-].
        lai: Leaf area index [m2 m-2].
        gD: Canopy radiation extinction coefficient [-].
        rsmin: Minimum stomatal resistance [s m-1].
        rssoilmin: Minimum soil resistance [s m-1].
        cveg: Vegetation fraction [-].
        wmax: Maximum water storage capacity of the canopy [m].
        lam: Thermal diffusivity of the soil [W m-1 K-1].
    """

    def __init__(
        self,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2ref: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        cveg: float,
        wmax: float,
        lam: float,
    ):
        self.a = a
        self.b = b
        self.p = p
        self.cgsat = cgsat
        self.wsat = wsat
        self.wfc = wfc
        self.wwilt = wwilt
        self.c1sat = c1sat
        self.c2ref = c2ref
        self.lai = lai
        self.gD = gD
        self.rsmin = rsmin
        self.rssoilmin = rssoilmin
        self.cveg = cveg
        self.wmax = wmax
        self.lam = lam
        self.c_beta = 0.0

    @abstractmethod
    def compute_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def compute_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        raise NotImplementedError

    # limamau: this function needs some refactoring
    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
    ):
        """Run the model."""
        # compute aerodynamic resistance
        state.ra = surface_layer.compute_ra(state)

        # first calculate essential thermodynamic variables
        state.esat = get_esat(state.theta)
        state.qsat = get_qsat(state.theta, state.surf_pressure)
        desatdT = state.esat * (
            17.2694 / (state.theta - 35.86)
            - 17.2694 * (state.theta - 273.16) / (state.theta - 35.86) ** 2.0
        )
        state.dqsatdT = 0.622 * desatdT / state.surf_pressure
        state.e = state.q * state.surf_pressure / 0.622

        # sub-model part
        self.compute_surface_resistance(state, const)
        self.compute_co2_flux(state, const)

        # recompute f2 using wg instead of w2
        f2 = jnp.where(
            state.wg > self.wwilt,
            (self.wfc - self.wwilt) / (state.wg - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, Array)
        state.rssoil = self.rssoilmin * f2

        wlmx = self.lai * self.wmax
        state.cliq = jnp.minimum(1.0, state.wl / wlmx)

        # calculate skin temperature implicitly
        state.surf_temp = (
            state.net_rad
            + const.rho * const.cp / state.ra * state.theta
            + self.cveg
            * (1.0 - state.cliq)
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + self.cveg
            * state.cliq
            * const.rho
            * const.lv
            / state.ra
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + self.lam * state.temp_soil
        ) / (
            const.rho * const.cp / state.ra
            + self.cveg
            * (1.0 - state.cliq)
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * state.dqsatdT
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * state.dqsatdT
            + self.cveg * state.cliq * const.rho * const.lv / state.ra * state.dqsatdT
            + self.lam
        )

        # limamau: should eastsurf just be deleted here?
        # or should it rather be updated on mixed layer?
        # esatsurf = get_esat(self.surf_temp)
        state.qsatsurf = get_qsat(state.surf_temp, state.surf_pressure)

        state.le_veg = (
            (1.0 - state.cliq)
            * self.cveg
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * (state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q)
        )
        state.le_liq = (
            state.cliq
            * self.cveg
            * const.rho
            * const.lv
            / state.ra
            * (state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q)
        )
        state.le_soil = (
            (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * (state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q)
        )

        state.wltend = -state.le_liq / (const.rhow * const.lv)

        state.le = state.le_soil + state.le_veg + state.le_liq
        state.hf = const.rho * const.cp / state.ra * (state.surf_temp - state.theta)
        state.gf = self.lam * (state.surf_temp - state.temp_soil)
        state.le_pot = (
            state.dqsatdT * (state.net_rad - state.gf)
            + const.rho * const.cp / state.ra * (state.qsat - state.q)
        ) / (state.dqsatdT + const.cp / const.lv)
        state.le_ref = (
            state.dqsatdT * (state.net_rad - state.gf)
            + const.rho * const.cp / state.ra * (state.qsat - state.q)
        ) / (
            state.dqsatdT
            + const.cp / const.lv * (1.0 + self.rsmin / self.lai / state.ra)
        )

        cg = self.cgsat * (self.wsat / state.w2) ** (self.b / (2.0 * jnp.log(10.0)))

        state.temp_soil_tend = cg * state.gf - 2.0 * jnp.pi / 86400.0 * (
            state.temp_soil - state.temp2
        )

        d1 = 0.1
        c1 = self.c1sat * (self.wsat / state.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (state.w2 / (self.wsat - state.w2))
        wgeq = state.w2 - self.wsat * self.a * (
            (state.w2 / self.wsat) ** self.p
            * (1.0 - (state.w2 / self.wsat) ** (8.0 * self.p))
        )
        state.wgtend = -c1 / (
            const.rhow * d1
        ) * state.le_soil / const.lv - c2 / 86400.0 * (state.wg - wgeq)

        # calculate kinematic heat fluxes
        state.wtheta = state.hf / (const.rho * const.cp)
        state.wq = state.le / (const.rho * const.lv)

        return state

    def integrate(self, state: PyTree, dt: float):
        """Integrate the model."""
        state.temp_soil += dt * state.temp_soil_tend
        state.wg += dt * state.wgtend
        state.wl += dt * state.wltend

        return state
