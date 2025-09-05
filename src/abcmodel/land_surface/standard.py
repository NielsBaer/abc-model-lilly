from abc import abstractmethod

import numpy as np

from ..components import AbstractLandSurfaceModel
from ..mixed_layer import AbstractMixedLayerModel
from ..radiation import AbstractRadiationModel
from ..surface_layer import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants, get_esat, get_qsat


class AbstractStandardLandSurfaceModel(AbstractLandSurfaceModel):
    # wet fraction [-]
    cliq: float
    # soil temperature tendency [K s-1]
    temp_soil_tend: float
    # soil moisture tendency [m3 m-3 s-1]
    wgtend: float
    # equivalent liquid water tendency [m s-1]
    wltend: float

    def __init__(
        self,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
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
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
    ):
        # water content parameters
        # volumetric water content top soil layer [m3 m-3]
        self.wg = wg
        # volumetric water content deeper soil layer [m3 m-3]
        self.w2 = w2
        # saturated volumetric water content ECMWF config [-]
        self.wsat = wsat
        # volumetric water content field capacity [-]
        self.wfc = wfc
        # volumetric water content wilting point [-]
        self.wwilt = wwilt

        # temperature params
        # temperature top soil layer [K]
        self.temp_soil = temp_soil
        # temperature deeper soil layer [K]
        self.temp2 = temp2
        # surface temperature [K]
        self.surf_temp = surf_temp

        # Clapp and Hornberger retention curve parameters
        self.a = a
        self.b = b
        self.p = p
        # saturated soil conductivity for heat
        self.cgsat = cgsat

        # C parameters
        self.c1sat = c1sat
        self.c2ref = c2ref

        # vegetation parameters
        # leaf area index [-]
        self.lai = lai
        # correction factor transpiration for VPD [-]
        self.gD = gD
        # minimum resistance transpiration [s m-1]
        self.rsmin = rsmin
        # minimum resistance soil evaporation [s m-1]
        self.rssoilmin = rssoilmin
        # surface albedo [-]
        self.alpha = alpha

        # resistance parameters (initialized to high values)
        # resistance transpiration [s m-1]
        self.rs = 1.0e6
        # resistance soil [s m-1]
        self.rssoil = 1.0e6

        # vegetation and water layer parameters
        # vegetation fraction [-]
        self.cveg = cveg
        # thickness of water layer on wet vegetation [m]
        self.wmax = wmax
        # equivalent water layer depth for wet vegetation [m]
        self.wl = wl

        # thermal diffusivity
        self.lamb = lam  # thermal diffusivity skin layer [-]

        # old: some sanity checks for valid input
        # limamau: I think this is supposed to be a parameter
        self.c_beta = 0.0  # zero curvature; linear response
        assert self.c_beta >= 0.0 or self.c_beta <= 1.0

    @abstractmethod
    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # compute aerodynamic resistance
        surface_layer.compute_ra(mixed_layer.u, mixed_layer.v, mixed_layer.wstar)

        # first calculate essential thermodynamic variables
        mixed_layer.esat = get_esat(mixed_layer.theta)
        mixed_layer.qsat = get_qsat(mixed_layer.theta, mixed_layer.surf_pressure)
        desatdT = mixed_layer.esat * (
            17.2694 / (mixed_layer.theta - 35.86)
            - 17.2694
            * (mixed_layer.theta - 273.16)
            / (mixed_layer.theta - 35.86) ** 2.0
        )
        mixed_layer.dqsatdT = 0.622 * desatdT / mixed_layer.surf_pressure
        mixed_layer.e = mixed_layer.q * mixed_layer.surf_pressure / 0.622

        # sub-model part
        self.compute_surface_resistance(const, radiation, surface_layer, mixed_layer)
        self.compute_co2_flux(const, surface_layer, mixed_layer)

        # recompute f2 using wg instead of w2
        if self.wg > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
            f2 = 1.0e8
        self.rssoil = self.rssoilmin * f2

        wlmx = self.lai * self.wmax
        self.cliq = min(1.0, self.wl / wlmx)

        # calculate skin temperature implicitly
        self.surf_temp = (
            radiation.net_rad
            + const.rho * const.cp / surface_layer.ra * mixed_layer.theta
            + self.cveg
            * (1.0 - self.cliq)
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rs)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.cveg
            * self.cliq
            * const.rho
            * const.lv
            / surface_layer.ra
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.lamb * self.temp_soil
        ) / (
            const.rho * const.cp / surface_layer.ra
            + self.cveg
            * (1.0 - self.cliq)
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rs)
            * mixed_layer.dqsatdT
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rssoil)
            * mixed_layer.dqsatdT
            + self.cveg
            * self.cliq
            * const.rho
            * const.lv
            / surface_layer.ra
            * mixed_layer.dqsatdT
            + self.lamb
        )

        # limamau: should eastsurf just be deleted here?
        # or should it rather be updated on mixed layer?
        # esatsurf = get_esat(self.surf_temp)
        mixed_layer.qsatsurf = get_qsat(self.surf_temp, mixed_layer.surf_pressure)

        self.le_veg = (
            (1.0 - self.cliq)
            * self.cveg
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rs)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_liq = (
            self.cliq
            * self.cveg
            * const.rho
            * const.lv
            / surface_layer.ra
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_soil = (
            (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (surface_layer.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )

        self.wltend = -self.le_liq / (const.rhow * const.lv)

        self.le = self.le_soil + self.le_veg + self.le_liq
        self.hf = (
            const.rho
            * const.cp
            / surface_layer.ra
            * (self.surf_temp - mixed_layer.theta)
        )
        self.gf = self.lamb * (self.surf_temp - self.temp_soil)
        self.le_pot = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + const.rho
            * const.cp
            / surface_layer.ra
            * (mixed_layer.qsat - mixed_layer.q)
        ) / (mixed_layer.dqsatdT + const.cp / const.lv)
        self.le_ref = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + const.rho
            * const.cp
            / surface_layer.ra
            * (mixed_layer.qsat - mixed_layer.q)
        ) / (
            mixed_layer.dqsatdT
            + const.cp / const.lv * (1.0 + self.rsmin / self.lai / surface_layer.ra)
        )

        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * np.log(10.0)))

        self.temp_soil_tend = cg * self.gf - 2.0 * np.pi / 86400.0 * (
            self.temp_soil - self.temp2
        )

        d1 = 0.1
        c1 = self.c1sat * (self.wsat / self.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        self.wgtend = -c1 / (
            const.rhow * d1
        ) * self.le_soil / const.lv - c2 / 86400.0 * (self.wg - wgeq)

        # calculate kinematic heat fluxes
        mixed_layer.wtheta = self.hf / (const.rho * const.cp)
        mixed_layer.wq = self.le / (const.rho * const.lv)

    def integrate(self, dt: float):
        self.temp_soil += dt * self.temp_soil_tend
        self.wg += dt * self.wgtend
        self.wl += dt * self.wltend
