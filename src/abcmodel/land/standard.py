from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..abstracts import (
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
    AbstractRadiationState,
)
from ..atmosphere.abstracts import AbstractMixedLayerState, AbstractSurfaceLayerState
from ..utils import PhysicalConstants, compute_esat, compute_qsat


@jax.tree_util.register_pytree_node_class
@dataclass
class StandardLandSurfaceState(AbstractLandState):
    """Standard land surface model state."""

    alpha: Array | float
    """Slope of the light response curve [mol J-1]."""
    wg: Array | float
    """Soil moisture content in the root zone [m3 m-3]."""
    temp_soil: Array | float
    """Soil temperature [K]."""
    temp2: Array | float
    """Deep soil temperature [K]."""
    surf_temp: Array | float
    """Surface temperature [K]."""
    wl: Array | float
    """Liquid water storage on the canopy [m]."""

    rs: Array | float = 1.0e6
    """Surface resistance [m s-1]."""
    rssoil: Array | float = 1.0e6
    """Soil resistance [m s-1]."""

    cliq: Array | float = jnp.nan
    """Wet fraction of the canopy [-]."""
    temp_soil_tend: Array | float = jnp.nan
    """Soil temperature tendency [K s-1]."""
    wgtend: Array | float = jnp.nan
    """Soil moisture tendency [m3 m-3 s-1]."""
    wltend: Array | float = jnp.nan
    """Canopy water storage tendency [m s-1]."""
    le_veg: Array | float = jnp.nan
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: Array | float = jnp.nan
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: Array | float = jnp.nan
    """Latent heat flux from soil [W m-2]."""
    le: Array | float = jnp.nan
    """Total latent heat flux [W m-2]."""
    hf: Array | float = jnp.nan
    """Sensible heat flux [W m-2]."""
    gf: Array | float = jnp.nan
    """Ground heat flux [W m-2]."""
    le_pot: Array | float = jnp.nan
    """Potential latent heat flux [W m-2]."""
    le_ref: Array | float = jnp.nan
    """Reference latent heat flux [W m-2]."""
    ra: Array | float = jnp.nan
    """Aerodynamic resistance [s m-1]."""
    esat: Array | float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    qsat: Array | float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array | float = jnp.nan
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array | float = jnp.nan
    """Vapor pressure [Pa]."""
    qsatsurf: Array | float = jnp.nan
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array | float = jnp.nan
    """Kinematic heat flux [K m/s]."""
    wq: Array | float = jnp.nan
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array | float = jnp.nan
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""

    def tree_flatten(self):
        return (
            self.alpha,
            self.wg,
            self.temp_soil,
            self.temp2,
            self.surf_temp,
            self.wl,
            self.rs,
            self.rssoil,
            self.cliq,
            self.temp_soil_tend,
            self.wgtend,
            self.wltend,
            self.le_veg,
            self.le_liq,
            self.le_soil,
            self.le,
            self.hf,
            self.gf,
            self.le_pot,
            self.le_ref,
            self.ra,
            self.esat,
            self.qsat,
            self.dqsatdT,
            self.e,
            self.qsatsurf,
            self.wtheta,
            self.wq,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# Alias for backward compatibility
StandardLandSurfaceInitConds = StandardLandSurfaceState


class AbstractStandardLandSurfaceModel(AbstractLandModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics.

    Args:
        a: Clapp and Hornberger (1978) retention curve parameter.
        b: Clapp and Hornberger (1978) retention curve parameter.
        p: Clapp and Hornberger (1978) retention curve parameter.
        cgsat: saturated soil heat capacity [J m-3 K-1].
        wsat: saturated soil moisture content [m3 m-3].
        wfc: soil moisture content at field capacity [m3 m-3].
        wwilt: soil moisture content at wilting point [m3 m-3].
        w2: soil moisture content in the deep layer [m3 m-3].
        d1: depth of the shallow layer [m].
        c1sat: saturated soil conductivity parameter [-].
        c2ref: reference soil conductivity parameter [-].
        lai: leaf area index [m2 m-2].
        gD: canopy radiation extinction coefficient [-].
        rsmin: minimum stomatal resistance [s m-1].
        rssoilmin: minimum soil resistance [s m-1].
        cveg: vegetation fraction [-].
        wmax: maximum water storage capacity of the canopy [m].
        lam: thermal diffusivity of the soil [W m-1 K-1].
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
        w2: float,
        d1: float,
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
        self.w2 = w2
        self.d1 = d1
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

    def integrate(
        self, state: StandardLandSurfaceState, dt: float
    ) -> StandardLandSurfaceState:
        """Integrate model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        state.temp_soil += dt * state.temp_soil_tend
        state.wg += dt * state.wgtend
        state.wl += dt * state.wltend

        return state

    def run(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> StandardLandSurfaceState:
        """Run the full land surface model for one time step.

        Args:
            state: CoupledState.
            const: the physical constants object.

        Returns:
            The updated land state object.
        """
        # Access components
        land_state = state.land
        ml_state = state.atmosphere.mixed_layer
        sl_state = state.atmosphere.surface_layer
        rad_state = state.radiation

        # compute aerodynamic resistance from state
        ueff = jnp.sqrt(ml_state.u**2.0 + ml_state.v**2.0 + ml_state.wstar**2.0)
        land_state.ra = ueff / jnp.maximum(1.0e-3, sl_state.ustar) ** 2.0

        land_state.esat = compute_esat(ml_state.theta)
        land_state.qsat = compute_qsat(ml_state.theta, ml_state.surf_pressure)
        land_state.dqsatdT = self.compute_dqsatdT(
            land_state, ml_state.theta, ml_state.surf_pressure
        )
        land_state.e = self.compute_e(ml_state.q, ml_state.surf_pressure)

        # update_surface_resistance and update_co2_flux need full state or specific components
        # They are abstract methods, implemented in subclasses.
        # We should pass `state` (CoupledState) to them so they can access what they need.
        # But `update_surface_resistance` in `jarvis_stewart.py` uses `state.in_srad` (radiation), `state.theta` (mixed layer), etc.
        # So passing `state` is correct.
        # However, they should return `land_state` or update it in place (if mutable).
        # Since we are returning `land_state`, we need to make sure they update `land_state`.
        # But `state` passed to them is `CoupledState`.
        # If they modify `state.land`, it's fine.
        # But `run` returns `land_state`.

        # Let's check `jarvis_stewart.py`:
        # state.rs = ...
        # return state

        # If I pass `state` (CoupledState), `state.rs` refers to `state.land.rs`?
        # NO. `CoupledState` has `land` attribute. `state.rs` would fail.
        # So `jarvis_stewart.py` needs update to access `state.land.rs`.

        # So I will pass `state` (CoupledState) to these methods.
        state = self.update_surface_resistance(state, const)
        state = self.update_co2_flux(state, const)

        # Now `state.land` is updated.
        # We need to continue using `state.land` (which is `land_state` reference if mutable, or we re-fetch).
        land_state = state.land

        land_state.rssoil = self.compute_soil_resistance(land_state)
        land_state.cliq = self.compute_cliq(land_state)
        land_state.surf_temp = self.compute_skin_temperature(
            land_state, ml_state, sl_state, rad_state, const
        )
        land_state.qsatsurf = compute_qsat(land_state.surf_temp, ml_state.surf_pressure)
        land_state.le_veg = self.compute_le_veg(land_state, ml_state, const)
        land_state.le_liq = self.compute_le_liq(land_state, ml_state, const)
        land_state.le_soil = self.compute_le_soil(land_state, ml_state, const)
        land_state.wltend = self.compute_wltend(land_state, const)
        land_state.le = self.compute_le(land_state)
        land_state.hf = self.compute_hf(land_state, ml_state, const)
        land_state.gf = self.compute_gf(land_state)
        land_state.le_pot = self.compute_le_pot(land_state, ml_state, rad_state, const)
        land_state.le_ref = self.compute_le_ref(land_state, ml_state, rad_state, const)
        land_state.temp_soil_tend = self.compute_temp_soil_tend(land_state)
        land_state.wgtend = self.compute_wgtend(land_state, const)

        # These update mixed layer fluxes?
        # wtheta and wq are in MixedLayerState.
        # But `compute_wtheta` returns a value.
        # Who uses it?
        # `BulkMixedLayerModel` uses `wtheta` from `MixedLayerState`.
        # `StandardLandSurfaceModel` computes it here.
        # Does it update `MixedLayerState`?
        # In the original code: `state.wtheta = ...`
        # `state` was flat.
        # So `wtheta` was shared.
        # Now `wtheta` is in `MixedLayerState`.
        # `StandardLandSurfaceModel` calculates it.
        # Should it update `MixedLayerState`?
        # `run` returns `LandState`.
        # So it cannot update `MixedLayerState` (unless we return CoupledState).
        # But `integration.py` does:
        # `state.land = coupler.land.run(state, ...)`
        # `state.atmosphere = coupler.atmosphere.run(state, ...)`

        # If `LandModel` calculates `wtheta`, `AtmosphereModel` needs to read it.
        # But `AtmosphereModel` runs AFTER `LandModel` in `timestep`.
        # So `AtmosphereModel` will read `state.land.wtheta`?
        # But `wtheta` is defined in `MixedLayerState`.
        # So `LandModel` needs to write to `state.atmosphere.mixed_layer.wtheta`?
        # Or `MixedLayerModel` should read it from `LandState`?
        # `BulkMixedLayerState` has `wtheta`.
        # `StandardLandSurfaceState` does NOT have `wtheta`.
        # So `LandModel` computes it but where does it put it?
        # It puts it in `state.wtheta` (in original code).
        # Now it must put it in `state.atmosphere.mixed_layer.wtheta`.
        # BUT `run` returns `LandState`.
        # So the update to `MixedLayerState` is LOST if we only return `LandState`.

        # This is a problem with the "return component state" approach when components modify other components' state.
        # Options:
        # 1. `LandModel.run` returns `CoupledState` (or updates it in place).
        # 2. `LandModel` puts fluxes in `LandState`, and `MixedLayerModel` reads them from `LandState`.
        #    - This requires adding `wtheta`, `wq` to `LandState`.
        #    - And updating `MixedLayerModel` to read from `LandState`.
        #    - `BulkMixedLayerModel` uses `state.wtheta`. I mapped it to `ml_state.wtheta`.
        #    - If I change `BulkMixedLayerModel` to read `state.land.wtheta` (if available), that works.
        #    - But `BulkMixedLayerState` has `wtheta`.
        #    - We should probably sync them or move ownership.
        #    - Fluxes `wtheta`, `wq` are surface fluxes. They originate from Land.
        #    - So they belong to Land.
        #    - `MixedLayer` uses them as boundary conditions.
        #    - So `MixedLayer` should read them from `Land`.
        #    - So I should ADD `wtheta`, `wq` (and `wCO2`?) to `StandardLandSurfaceState`.
        #    - And REMOVE them from `BulkMixedLayerState`? Or keep them as "current BCs"?
        #    - If I keep them in `MixedLayerState`, `LandModel` needs to update them.

        # Given the constraint of `run` returning `LandState`, option 2 is better:
        # Add `wtheta`, `wq` to `LandState`.
        # Update `MixedLayerModel` to read from `LandState`.

        # Let's add `wtheta`, `wq` to `StandardLandSurfaceState`.

        land_state.wtheta = self.compute_wtheta(land_state, const)
        land_state.wq = self.compute_wq(land_state, const)

        return land_state

    def compute_dqsatdT(
        self, state: StandardLandSurfaceState, theta: float, surf_pressure: float
    ) -> Array:
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :meth:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :meth:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = state.esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return q * surf_pressure / 0.622

    @abstractmethod
    def update_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Abstract method to update surface resistance."""
        raise NotImplementedError

    @abstractmethod
    def update_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Abstract method to update CO2 flux."""
        raise NotImplementedError

    def compute_soil_resistance(self, state: PyTree) -> Array:
        """Compute the soil resistance ``rssoil``.

        Notes:
            The soil resistance is calculated as

            .. math::
                r_\\text{soil} = r_\\text{soil,min} \\cdot f_2,

            where the parameter :math:`r_\\text{soil,min}` is the minimum surface resistance and
            the correction function :math:`f_2` is given by

            .. math::
                f_2 =
                    \\begin{cases}
                        \\frac{w_\\text{fc} - w_\\text{wilt}}{w_g - w_\\text{wilt}}, & \\text{if } w_g > w_\\text{wilt} \\\\
                        10^8, & \\text{otherwise},
                    \\end{cases}

            where the model parameters :math:`w_\\text{fc}` and :math:`w_\\text{wilt}`
            are the field capacity and wilting point, respectively,
            and the variable :math:`w_g` is the soil water content.

        References:
            Equations 9.28 and 9.31 from the CLASS book.
        """
        f2 = jnp.where(
            state.wg > self.wwilt,
            (self.wfc - self.wwilt) / (state.wg - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, Array)
        return self.rssoilmin * f2

    def compute_cliq(self, state: PyTree) -> Array:
        """Compute the wet fraction ``cliq``.

        Notes:
            The wet fraction is defined as

            .. math::
                c_{\\text{liq}} = \\frac{W_l}{\\text{LAI}\\cdot W_{\\text{max}}},

            where :math:`W_l` is the water layer depth,
            :math:`\\text{LAI}` is the leaf area index and
            :math:`W_{\\text{max}}` is the thickness of the water layer on wet vegetation.
            In case :math:`W_l > \\text{LAI}\\cdot W_{\\text{max}}`, the wet fraction is set to 1.

        References:
            Equation 9.19 from the CLASS book.
        """
        wlmx = self.lai * self.wmax
        return jnp.minimum(1.0, state.wl / wlmx)

    def compute_skin_temperature(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        sl_state: AbstractSurfaceLayerState,
        rad_state: AbstractRadiationState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the skin temperature ``surf_temp``.

        Notes:
            The skin temperature is obtained by solving the surface energy balance

            .. math::
                R_n = H + LE_{\\text{veg}} + LE_{\\text{liq}} + LE_{\\text{soil}} + G

            where :math:`R_n` is the net radiation,
            :math:`H` is the sensible heat flux (see :meth:`compute_hf`),
            :math:`LE_{\\text{veg}}` is the latent heat flux from vegetation (see :meth:`compute_le_veg`),
            :math:`LE_{\\text{liq}}` is the latent heat flux from dew on leaves (see :meth:`compute_le_liq`),
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil (see :meth:`compute_le_soil`)
            and :math:`G` is the ground heat flux (see :meth:`compute_gf`).

            The equation is solved for the skin temperature :math:`T_s`
            by factoring out :math:`T_s` from the above, giving us

            .. math::
                T_s = \\frac{
                    R_n + \\frac{\\rho c_p}{r_a} \\theta
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + \\Lambda T_{\\text{soil}}
                }{
                    \\frac{\\rho c_p}{r_a}
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + \\Lambda
                }.

            The terms computed in the equation above in each related function energy flux related method.
            This approach ensures that the computed skin temperature is consistent with the partitioning of
            energy fluxes as calculated by the other methods in this class.
        """
        return (
            rad_state.net_rad
            + const.rho * const.cp / state.ra * ml_state.theta
            + self.cveg
            * (1.0 - state.cliq)
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * (state.dqsatdT * ml_state.theta - state.qsat + ml_state.q)
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * (state.dqsatdT * ml_state.theta - state.qsat + ml_state.q)
            + self.cveg
            * state.cliq
            * const.rho
            * const.lv
            / state.ra
            * (state.dqsatdT * ml_state.theta - state.qsat + ml_state.q)
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

    def compute_le_veg(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the latent heat flux (transpiration) from vegetation ``le_veg``.

        Notes:
            The latent heat flux is given by

            .. math::
                LE_{\\text{veg}} = \\frac{\\rho L_v}{r_a+r_s}(q_{\\text{sat}}(T_s)-⟨q⟩),

            where :math:`\\rho` is the density of air, :math:`L_v` is the latent heat of vaporization,
            :math:`r_a` is the aerodynamic resistance, :math:`r_s` is the soil resistance,
            :math:`q_{\\text{sat}}(T_s)` is the saturation specific humidity at surface temperature,
            :math:`⟨q⟩` is the specific humidity at the surface.

            :math:`q_{\\text{sat}}(T_s)` has very short time-scales because of the small heat capacity
            (excluding vegetation) of the surface layer and is hard to measure. Consequently,
            we get :math:`q_{\\text{sat}}(T_s)` implicitly using

            .. math::
                q_{\\text{sat}}(T_s) = \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}(\\theta_s-\\theta),

            where :math:`\\theta_s` and :math:`\\theta` are the potential temperature of the surface layer and mixed layer, respectively.

            In the end, we scale the latent heat flux by the vegetation cover fraction :math:`c_{\\text{veg}}`
            and the liquid water content :math:`c_{\\text{liq}}` and return

            .. math::
                c_{\\text{veg}}(1-c_{\\text{liq}})LE_{\\text{veg}}.

        References:
            Equation 9.15 from the CLASS book.
        """
        term = (
            state.dqsatdT * (state.surf_temp - ml_state.theta) + state.qsat - ml_state.q
        )
        le_veg = const.rho * const.lv / (state.ra + state.rs) * term
        frac = (1.0 - state.cliq) * self.cveg
        return frac * le_veg

    def compute_le_liq(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the latent heat flux on the leaf (dew) ``le_liq``.

        Notes:
            We proceed just like in :meth:`compute_le_veg`, but omitting vegetation's resistance :math:`r_s`,
            with the assumption that water at the leaf is ready to be evaporated, giving us

        .. math::
            LE_{\\text{liq}} = \\frac{\\rho L_v}{r_a}(q_{\\text{sat}}(T_s)-⟨q⟩).

        In the end, we scale the result by the fraction of liquid water content :math:`c_{\\text{liq}}`
        and the fraction of vegetation :math:`c_{\\text{veg}}`.

        References:
            Equation 9.18 from the CLASS book.
        """
        term = (
            state.dqsatdT * (state.surf_temp - ml_state.theta) + state.qsat - ml_state.q
        )
        le_liq = const.rho * const.lv / state.ra * term
        frac = state.cliq * self.cveg
        return frac * le_liq

    def compute_le_soil(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the latent heat flux on the soil (evaporation) ``le_soil``.

        Notes:
            We proceed just like in :meth:`compute_le_veg`, but instead of considering resistance from
            the vegetation, we consider the resistance from the soil :math:`r_{soil}`, giving us

        .. math::
            LE_{\\text{soil}} = \\frac{\\rho L_v}{r_a + r_{soil}}(q_{\\text{sat}}(T_s)-⟨q⟩)

        In the end, we scale the result by the fraction of soil :math:`c_{\\text{soil}} = 1 - c_{\\text{veg}}`.

        References:
            Equation 9.21 from the CLASS book.
        """
        term = (
            state.dqsatdT * (state.surf_temp - ml_state.theta) + state.qsat - ml_state.q
        )
        le_soil = const.rho * const.lv / (state.ra + state.rssoil) * term
        frac = 1.0 - self.cveg
        return frac * le_soil

    def compute_wltend(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the water layer depth tendency tendency ``wltend``.

        Notes:
            The water layer depth tendency is the rate at which water is added to or taken from the vegetation,
            described by

            .. math::
                \\frac{\\text{d} w}{\\text{d} t} = -\\frac{LE_{\\text{liq}}}{\\rho_w L_v},

            where :math:`LE_{\\text{liq}}` is dew, :math:`\\rho_w` is water density and :math:`L_v` is the latent heat of vaporization.

        References:
            Equation 9.20 from the CLASS book, with sign convention.
        """
        return -state.le_liq / (const.rhow * const.lv)

    def compute_le(self, state: PyTree) -> Array:
        """Compute the evapotranspiration (latent heat flux) ``le``.

        Notes:
            The sum of

            - transpiration from vegetation in :meth:`compute_le_veg`;
            - evaporation from bare soil in :meth:`compute_le_soil`;
            - evaporation from wet leaves (dew) in :meth:`compute_le_liq`.
        """
        return state.le_soil + state.le_veg + state.le_liq

    def compute_hf(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the sensible heat flux ``hf``.

        Notes:
            The sensible heat flux is given by

            .. math::

                H = \\frac{\\rho c_p}{r_a} (T_s - \\theta),

            where :math:`\\rho` is the air density, :math:`c_p` is the specific heat capacity of air,
            :math:`r_a` is the aerodynamic resistance, :math:`T_s` is the surface temperature and
            :math:`\\theta` is the mixed layer air potential temperature.

        References:
            Equation 9.13 from the CLASS book, but why are we using :math:`T_s` instead of :math:`\\theta_s`?
            Probably because the variations of pressure are not significant enough.
        """
        return const.rho * const.cp / state.ra * (state.surf_temp - ml_state.theta)

    def compute_gf(self, state: PyTree) -> Array:
        """Compute the ground heat flux ``gf``.

        Notes:
            The ground heat flux is given by

            .. math::

                G = \\Lambda (T_s - T_{soil}),

            where :math:`\\Lambda` is the conductivity of the skin layer,
            :math:`T_s` is the surface temperature and
            :math:`T_{soil}` is the soil temperature.

        References:
            Equation 9.33 from the CLASS book.
        """
        return self.lam * (state.surf_temp - state.temp_soil)

    def compute_le_pot(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        rad_state: AbstractRadiationState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the potential latent heat flux ``le_pot``.

        Notes:
            The potential latent heat flux is given by

            .. math::

                LE_{\\text{pot}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}
                },

            which is the Penman-Monteith equation assuming no soil resistance.

        References:
            Equation 9.16 from the CLASS book.
        """
        rad_term = state.dqsatdT * (rad_state.net_rad - state.gf)
        aerodynamic_term = const.rho * const.cp / state.ra * (state.qsat - ml_state.q)
        denominator = state.dqsatdT + const.cp / const.lv
        return (rad_term + aerodynamic_term) / denominator

    def compute_le_ref(
        self,
        state: StandardLandSurfaceState,
        ml_state: AbstractMixedLayerState,
        rad_state: AbstractRadiationState,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the reference latent heat flux ``le_ref``.

        Notes:
            The reference latent heat flux is given by

            .. math::

                LE_{\\text{ref}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}(
                1 + \\frac{r_{s,\\text{min}}}{\\text{LAI} \\cdot r_a}
                )
                },

            which is the Penman-Monteith equation assuming that the soil resistance is given by
            :math:`r_{s,\\text{min}} / \\text{LAI}`, i.e., no correction functions are applied.

        References:
            Equation 9.16 from the CLASS book.
        """
        rad_term = state.dqsatdT * (rad_state.net_rad - state.gf)
        aerodynamic_term = const.rho * const.cp / state.ra * (state.qsat - ml_state.q)
        den1 = state.dqsatdT
        den2 = const.cp / const.lv * (1.0 + self.rsmin / self.lai / state.ra)
        return (rad_term + aerodynamic_term) / (den1 + den2)

    def compute_temp_soil_tend(self, state: PyTree) -> Array:
        """Compute the soil temperature tendency ``temp_soil_tend``.

        Notes:
            The dynamics of heat transport in the soil is given by

            .. math::
                \\frac{\\text{d}T_s}{\\text{d}t}
                =
                C_T G - \\frac{2\\pi}{\\tau} (T_s - T_2),

            :math:`T_2` is the temperature of the second layer in the soil,
            :math:`T_s` is the soil temperature,
            where :math:`\\tau` is the time constant of one day (86400s),
            :math:`G` is the ground the heat flux and
            and :math:`C_T` is the surface soil/vegetation heat capacity, which can be parametrized as

            .. math::
                C_T = C_{T,\\text{sat}} \\left(\\frac{w_{\\text{sat}}}{w_2}\\right)^{\\frac{b}{2\\log(10)}}

            where :math:`C_{T,\\text{sat}}` is the saturated heat capacity,
            :math:`w_{\\text{sat}}` is the saturation water content,
            :math:`w_2` is the water content at the second layer
            and :math:`b` is a parameter from Clapp and Hornberger (1978).
            I have no idea where this log comes from.

        References:
            Equation 9.32 of the CLASS book.
        """
        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * jnp.log(10.0)))
        return cg * state.gf - 2.0 * jnp.pi / 86400.0 * (state.temp_soil - state.temp2)

    def compute_wgtend(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the soil moisture tendency ``wgtend``.

        Notes:
            The dynamics of soil moisture in the top soil layer is described by

            .. math::
                \\frac{\\mathrm{d}w_g}{\\mathrm{d}t}
                =
                -\\frac{C_1}{\\rho_w d_1} \\frac{LE_{\\text{soil}}}{L_v}
                - \\frac{C_2}{\\tau} (w_g - w_{eq}),
                :label: dynamics

            where the coefficients :math:`C_1` and :math:`C_2` are calculated as

            .. math::
                C_1 = C_{1,\\text{sat}} \\left(\\frac{w_{sat}}{w_g}\\right)^{b/2 + 1},
            .. math::
                C_2 = C_{2,\\text{ref}} \\left(\\frac{w_2}{w_{sat} - w_2}\\right),

            where :math:`C_{1,\\text{sat}}` and :math:`C_{2,\\text{sat}}` are parameters from Clapp-Hornberger (1978)
            and the equilibrium soil moisture is given by

            .. math::
                w_{eq} = w_2 - w_{sat} a \\left(\\left(\\frac{w_2}{w_{sat}}\\right)^p
                \\left[1 - \\left(\\frac{w_2}{w_{sat}}\\right)^{8p}\\right]\\right).

            In these equations, :math:`w_g` is the volumetric soil moisture in the top layer,
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil,
            :math:`L_v` is the latent heat of vaporization,
            :math:`\\rho_w` is the density of water,
            :math:`d_1` is the depth of the first soil layer,
            :math:`\\tau` is a time constant (here, 86400 s = 1 day),
            :math:`w_2` is the soil moisture in the second layer,
            :math:`w_{sat}` is the saturated soil moisture, and
            :math:`a`, :math:`b` and :math:`p` are parameters from Clapp-Hornberger (1978).

            In :eq:`dynamics`, the first term represents the loss of soil moisture due to evaporation,
            and the second term represents the relaxation of soil moisture toward equilibrium with the lower layer.

        References:
            - (9.34)–(9.37) in the CLASS book.
            - Clapp, R. B., & Hornberger, G. M. (1978). Empirical equations for some soil hydraulic properties. Water resources research, 14(4), 601-604.

        """
        c1 = self.c1sat * (self.wsat / state.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        evap_loss = -c1 / (const.rhow * self.d1) * state.le_soil / const.lv
        deep_grad = c2 / 86400.0 * (state.wg - wgeq)
        return evap_loss + deep_grad

    def compute_wtheta(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the kinematic heat flux ``wtheta``.

        Notes:
            The kinematic heat flux :math:`\\overline{(w'\\theta')}_s` is directly related to the
            sensible heat flux :math:`H` through

            .. math::
                \\overline{(w'\\theta')}_s = \\frac{H}{\\rho c_p},

            where :math:`\\rho` is the density of air and
            :math:`c_p` is the specific heat capacity of air at constant pressure.
        """
        return state.hf / (const.rho * const.cp)

    def compute_wq(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the kinematic moisture flux ``wq``.

        Notes:
            The kinematic moisture flux :math:`\\overline{(w'q')}_s` is directly related to the
            latent heat flux :math:`LE` through

            .. math::
                \\overline{(w'q')}_s = \\frac{LE}{\\rho L_v},

            where :math:`\\rho` is the density of air and
            :math:`L_v` is the latent heat of vaporization.
        """
        return state.le / (const.rho * const.lv)
