"""I like pizza."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import exp1
from jaxtyping import Array, PyTree

from ..utils import PhysicalConstants, get_esat
from .standard import AbstractStandardLandSurfaceModel, StandardLandSurfaceInitConds


@dataclass
class AquaCropInitConds(StandardLandSurfaceInitConds):
    """AquaCrop model initial state."""

    rsCO2: float = jnp.nan
    """Stomatal resistance to CO2."""
    gcco2: float = jnp.nan
    """Conductance to CO2."""
    ci: float = jnp.nan
    """Intercellular CO2 concentration."""
    co2abs: float = jnp.nan
    """CO2 assimilation rate."""


class AquaCropModel(AbstractStandardLandSurfaceModel):
    """AquaCrop land surface model with coupled photosynthesis and stomatal conductance.

    A bit more advanced land surface model implementing the AquaCrop approach with coupled
    photosynthesis-stomatal conductance calculations. Includes detailed biochemical
    processes for both C3 and C4 vegetation types, soil moisture stress effects,
    and explicit CO2 flux calculations.

    1. Inherit all standard land surface processes from parent class.
    2. Calculate CO2 compensation concentration based on temperature.
    3. Compute mesophyll conductance with temperature response functions.
    4. Determine internal CO2 concentration using stomatal optimization.
    5. Calculate gross primary productivity with light and moisture limitations.
    6. Scale from leaf-level to canopy-level fluxes using extinction functions.
    7. Compute surface resistance from canopy conductance.
    8. Calculate net CO2 fluxes including plant assimilation and soil respiration.

    Args:
        c3c4: plant type, either "c3" or "c4".
        co2comp298: CO2 compensation concentration [mg m-3].
        net_rad10CO2: function parameter to calculate CO2 compensation concentration [-].
        gm298: mesophyill conductance at 298 K [mm s-1].
        ammax298: CO2 maximal primary productivity [mg m-2 s-1].
        net_rad10gm: function parameter to calculate mesophyll conductance [-].
        temp1gm: reference temperature to calculate mesophyll conductance gm [K].
        temp2gm: reference temperature to calculate mesophyll conductance gm [K].
        net_rad10Am: function parameter to calculate maximal primary productivity Ammax.
        temp1Am: reference temperature to calculate maximal primary productivity Ammax [K].
        temp2Am: reference temperature to calculate maximal primary productivity Ammax [K].
        f0: maximum value Cfrac [-].
        ad: regression coefficient to calculate Cfrac [kPa-1].
        alpha0: initial low light conditions [mg J-1].
        kx: extinction coefficient PAR [-].
        gmin: cuticular (minimum) conductance [mm s-1].
        nuco2q: ratio molecular viscosity water to carbon dioxide.
        cw: constant water stress correction (eq. 13 Jacobs et al. 2007) [-].
        wmax: upper reference value soil water [-].
        wmin: lower reference value soil water [-].
        r10: respiration at 10 C [mg CO2 m-2 s-1].
        e0: activation energy [53.3 kJ kmol-1].
    """

    def __init__(self, c3c4: str, **kwargs):
        super().__init__(**kwargs)
        if c3c4 == "c3":
            self.c3c4 = 0
        elif c3c4 == "c4":
            self.c3c4 = 1
        else:
            raise ValueError(f'''Invalid option "{c3c4}" for "c3c4".''')

        self.co2comp298 = jnp.array([68.5, 4.3])
        self.net_rad10CO2 = jnp.array([1.5, 1.5])
        self.gm298 = jnp.array([7.0, 17.5])
        self.ammax298 = jnp.array([2.2, 1.7])
        self.net_rad10gm = jnp.array([2.0, 2.0])
        self.temp1gm = jnp.array([278.0, 286.0])
        self.temp2gm = jnp.array([301.0, 309.0])
        self.net_rad10Am = jnp.array([2.0, 2.0])
        self.temp1Am = jnp.array([281.0, 286.0])
        self.temp2Am = jnp.array([311.0, 311.0])
        self.f0 = jnp.array([0.89, 0.85])
        self.ad = jnp.array([0.07, 0.15])
        self.alpha0 = jnp.array([0.017, 0.014])
        self.kx = jnp.array([0.7, 0.7])
        self.gmin = jnp.array([0.25e-3, 0.25e-3])
        self.nuco2q = 1.6
        self.cw = 0.0016
        self.wmax = 0.55
        self.wmin = 0.005
        self.r10 = 0.23
        self.e0 = 53.3e3

    def calculate_co2_compensation_concentration(
        self,
        thetasurf: Array,
        const: PhysicalConstants,
    ) -> Array:
        """Calculate CO2 compensation concentration."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.pow(self.net_rad10CO2[self.c3c4], temp_diff)
        return self.co2comp298[self.c3c4] * const.rho * exp_term

    def calculate_mesophyll_conductance(
        self,
        thetasurf: Array,
    ) -> Array:
        """Calculate mesophyll conductance."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.pow(self.net_rad10gm[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1gm[self.c3c4] - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2gm[self.c3c4]))
        gm = self.gm298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return gm / 1000.0

    def calculate_internal_co2(
        self,
        surf_temp: Array,
        e: Array,
        co2: Array,
        co2comp: Array,
        gm: Array,
        const: PhysicalConstants,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Calculate internal CO2 concentration and related parameters."""
        fmin0 = self.gmin[self.c3c4] / self.nuco2q - 1.0 / 9.0 * gm
        fmin_sq_term = jnp.pow(fmin0, 2.0) + 4 * self.gmin[self.c3c4] / self.nuco2q * gm
        fmin = -fmin0 + jnp.pow(fmin_sq_term, 0.5) / (2.0 * gm)
        ds = (get_esat(surf_temp) - e) / 1000.0  # kPa

        d0 = (self.f0[self.c3c4] - fmin) / self.ad[self.c3c4]

        cfrac = self.f0[self.c3c4] * (1.0 - (ds / d0)) + fmin * (ds / d0)
        co2abs = co2 * (const.mco2 / const.mair) * const.rho
        ci = cfrac * (co2abs - co2comp) + co2comp
        return ci, co2abs, fmin, ds, d0

    def calculate_max_gross_primary_production(self, thetasurf: Array) -> Array:
        """Calculate maximal gross primary production in high light conditions (Ag)."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10Am[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1Am[self.c3c4] - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2Am[self.c3c4]))
        ammax = self.ammax298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return ammax

    def calculate_soil_moisture_stress_factor(self, w2: Array) -> Array:
        """Calculate effect of soil moisture stress on gross assimilation rate."""
        # soil moisture ratio
        soil_moisture_ratio = (w2 - self.wwilt) / (self.wfc - self.wwilt)
        betaw = jnp.clip(soil_moisture_ratio, 1e-3, 1.0)

        # branch functions for different c_beta ranges
        def case_zero(_):
            """c_beta == 0: return betaw directly"""
            return betaw

        def case_low(_):
            """c_beta < 0.25: p = 6.4 * c_beta"""
            p = 6.4 * self.c_beta
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        def case_medium(_):
            """0.25 <= c_beta < 0.50: p = 7.6 * c_beta - 0.3"""
            p = 7.6 * self.c_beta - 0.3
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        def case_high(_):
            """c_beta >= 0.50: p = 2^(3.66 * c_beta + 0.34) - 1"""
            p = 2.0 ** (3.66 * self.c_beta + 0.34) - 1.0
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        # determine which case to use based on c_beta value
        branch_index = jnp.where(
            self.c_beta == 0,
            0,
            jnp.where(self.c_beta < 0.25, 1, jnp.where(self.c_beta < 0.50, 2, 3)),
        )

        # select the appropriate function
        result = jax.lax.switch(
            branch_index, [case_zero, case_low, case_medium, case_high], None
        )

        return result

    def calculate_gross_assimilation_and_light_use(
        self,
        in_srad: Array,
        ammax: Array,
        gm: Array,
        ci: Array,
        co2comp: Array,
        co2abs: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Calculate gross assimilation rate, dark respiration, PAR, and light use efficiency."""
        assimilation_factor = -(gm * (ci - co2comp) / ammax)
        am = ammax * (1.0 - jnp.exp(assimilation_factor))
        rdark = (1.0 / 9.0) * am
        par = 0.5 * jnp.maximum(1e-1, in_srad * self.cveg)
        co2_ratio = (co2abs - co2comp) / (co2abs + 2.0 * co2comp)
        alphac = self.alpha0[self.c3c4] * co2_ratio
        return am, rdark, par, alphac

    def calculate_canopy_co2_conductance(
        self,
        alphac: Array,
        par: Array,
        am: Array,
        rdark: Array,
        fstr: Array,
        co2abs: Array,
        co2comp: Array,
        ds: Array,
        d0: Array,
        fmin: Array,
    ) -> Array:
        """Calculate upscaling from leaf to canopy and CO2 conductance at canopy level."""
        y = alphac * self.kx[self.c3c4] * par / (am + rdark)
        exp1_arg1 = y * jnp.exp(-self.kx[self.c3c4] * self.lai)
        exp1_arg2 = y
        exp1_term = exp1(exp1_arg1) - exp1(exp1_arg2)
        an = (am + rdark) * (1.0 - (1.0 / (self.kx[self.c3c4] * self.lai)) * exp1_term)
        a1 = 1.0 / (1.0 - self.f0[self.c3c4])
        dstar = d0 / (a1 * (self.f0[self.c3c4] - fmin))
        conductance_factor = a1 * fstr * an / ((co2abs - co2comp) * (1.0 + ds / dstar))
        gcco2 = self.lai * (self.gmin[self.c3c4] / self.nuco2q + conductance_factor)
        return gcco2

    def compute_surface_resistance(self, state: PyTree, const: PhysicalConstants):
        """Compute surface resistance using AquaCrop photosynthesis-conductance model."""
        co2comp = self.calculate_co2_compensation_concentration(state.thetasurf, const)
        gm = self.calculate_mesophyll_conductance(state.thetasurf)

        state.ci, state.co2abs, fmin, ds, d0 = self.calculate_internal_co2(
            state.thetasurf, state.e, state.co2, co2comp, gm, const
        )

        ammax = self.calculate_max_gross_primary_production(state.thetasurf)
        fstr = self.calculate_soil_moisture_stress_factor(state.w2)

        am, rdark, par, alphac = self.calculate_gross_assimilation_and_light_use(
            state.in_srad,
            ammax,
            gm,
            state.ci,
            co2comp,
            state.co2abs,
        )

        state.gcco2 = self.calculate_canopy_co2_conductance(
            alphac, par, am, rdark, fstr, state.co2abs, co2comp, ds, d0, fmin
        )

        state.rs = 1.0 / (1.6 * state.gcco2)

    def compute_co2_flux(self, state: PyTree, const: PhysicalConstants):
        """Compute the CO2 flux."""
        state.rsCO2 = 1.0 / state.gcco2
        an = -(state.co2abs - state.ci) / (state.ra + state.rsCO2)
        fw = self.cw * self.wmax / (state.wg + self.wmin)
        temp_ratio = 1.0 - 283.15 / state.temp_soil
        resp_factor = jnp.exp(self.e0 / (283.15 * 8.314) * temp_ratio)
        resp = self.r10 * (1.0 - fw) * resp_factor
        state.wCO2A = an * (const.mair / (const.rho * const.mco2))
        state.wCO2R = resp * (const.mair / (const.rho * const.mco2))
        state.wCO2 = state.wCO2A + state.wCO2R
