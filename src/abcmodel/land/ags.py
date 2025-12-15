from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import exp1

from ..abstracts import AbstractCoupledState
from ..utils import PhysicalConstants, compute_esat
from .standard import AbstractStandardLandModel, StandardLandState


@dataclass
class AgsState(StandardLandState):
    """A-gs model state."""

    rsCO2: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Stomatal resistance to CO2."""
    gcco2: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Conductance to CO2."""
    ci: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Intercellular CO2 concentration."""
    co2abs: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """CO2 assimilation rate."""
    wCO2A: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Net assimilation flux [mol m-2 s-1]."""
    wCO2R: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Respiration flux [mol m-2 s-1]."""
    wCO2: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Total CO2 flux [mol m-2 s-1]."""


AgsInitConds = AgsState


class AgsModel(AbstractStandardLandModel):
    """Ags land surface model with coupled photosynthesis and stomatal conductance.

    ... (docstring omitted for brevity) ...
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

    def compute_co2comp(
        self,
        thetasurf: Array,
        rho: float,
    ) -> Array:
        """Compute the CO₂ compensation concentration.

        Args:
            thetasurf: surface potential temperature :math:`\\theta_s` [K].
            rho: air density [kg m⁻³].

        Returns:
            The CO₂ compensation concentration :math:`\\gamma` [ppmv].

        Notes:
            The CO₂ compensation point is computed as

            .. math::
                \\gamma = \\rho \\gamma_{298} Q_{10} e^{(T_s-298)/10}

            where :math:`\\gamma_{298}` and :math:`Q_{10}` are parameters depending on the plant type (C3 or C4).
            Here, :math:`\\theta_s` is used instead of the skin temperature :math:`T_s` to compute the exponential term,
            probably because this comes before in the order of updates for stability?

        References:
            Equation E.2 of the CLASS book.
        """
        # limamau: why are we using thetasurf here instead of surf_temp?
        # where is this rho coming from?
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.pow(self.net_rad10CO2[self.c3c4], temp_diff)
        return self.co2comp298[self.c3c4] * rho * exp_term

    def compute_gm(self, thetasurf: Array) -> Array:
        """Compute the mesophyll conductance.

        Args:
            thetasurf: surface potential temperature :math:`\\theta_s` [K].

        Returns:
            Mesophyll conductance :math:`g_m` [mm s⁻¹].

        Notes:
            This is given by

            .. math::
                g_m
                = \\frac{g_{m298} \\cdot \\exp((T_s - 298)/10)}
                {[1 + \\exp(0.3 \\cdot (T_{1g_m} - T_s))] \\cdot [1 + \\exp(0.3 \\cdot (T_s - T_{2g_m}))]},

            where :math:`g_{m298}`, :math:`T_{1g_m}` and :math:`T_{2g_m}` are parameters depending on the plant type (C3 or C4).
            Here again, instead of using the skin temperature :math:`T_s`, we use :math:`\\theta_s`.

        References:
            Equation E.7 from the CLASS book.

        """
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.pow(self.net_rad10gm[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1gm[self.c3c4] - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2gm[self.c3c4]))
        gm = self.gm298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return gm / 1000.0

    def compute_fmin(
        self,
        gm: Array,
    ) -> Array:
        """Compute minimum stomatal conductance factor (fmin).

        Notes:
            The minimum stomatal conductance factor is computed by solving the quadratic equation:

            .. math::
                f_{min}^2 + f_{min} \\left( \\frac{g_{min}}{\\nu_{CO2}} - \\frac{1}{9} g_m \\right) - \\frac{g_{min}}{\\nu_{CO2}} g_m = 0

            which leads to

            .. math::
                f_{min} = -f_{min,0} + \\frac{\\sqrt{f_{min,0}^2 + 4 \\frac{g_{min}}{\\nu_{CO2}} g_m}}{2 g_m}

            where :math:`f_{min,0} = \\frac{g_{min}}{\\nu_{CO2}} - \\frac{1}{9} g_m`.
        """
        fmin0 = self.gmin[self.c3c4] / self.nuco2q - 1.0 / 9.0 * gm
        fmin_sq_term = (
            jnp.power(fmin0, 2.0) + 4 * self.gmin[self.c3c4] / self.nuco2q * gm
        )
        fmin = -fmin0 + jnp.power(fmin_sq_term, 0.5) / (2.0 * gm)
        return fmin

    def compute_ds(
        self,
        surf_temp: Array,
        e: Array,
    ) -> Array:
        """Compute vapor pressure deficit (ds) in kPa.

        Notes:
            The vapor pressure deficit is given by

            .. math::
                d_s = \\frac{e_{sat}(T_s) - e}{1000}

            where :math:`e_{sat}(T_s)` is the saturation vapor pressure at surface temperature
            and :math:`e` is the actual vapor pressure.
        """
        ds = (compute_esat(surf_temp) - e) / 1000.0  # kPa
        return ds

    def compute_d0(
        self,
        fmin: Array,
    ) -> Array:
        """Compute reference vapor pressure deficit (d0) in kPa.

        Notes:
            The reference vapor pressure deficit is given by

            .. math::
                d_0 = \\frac{f_0 - f_{min}}{a_d}

            where :math:`f_0` and :math:`a_d` are empirical parameters.
        """
        d0 = (self.f0[self.c3c4] - fmin) / self.ad[self.c3c4]
        return d0

    def compute_internal_co2(
        self,
        ds: Array,
        d0: Array,
        fmin: Array,
        co2: Array,
        co2comp: Array,
        const: PhysicalConstants,
    ) -> tuple[Array, Array]:
        """Compute cfrac, co2abs, and ci (internal CO2 concentration).

        Notes:
            The fraction of internal to ambient CO2 concentration :math:`c_{frac}` is given by

            .. math::
                c_{frac} = f_0 (1 - d_s/d_0) + f_{min} (d_s/d_0)

            The ambient CO2 concentration in partial pressure units :math:`C_a` is

            .. math::
                C_a = [CO_2] \\frac{M_{CO2}}{M_{air}} \\rho

            The internal CO2 concentration :math:`C_i` is then

            .. math::
                C_i = c_{frac} (C_a - \\gamma) + \\gamma

            where :math:`\\gamma` is the CO2 compensation point.
        """
        cfrac = self.f0[self.c3c4] * (1.0 - (ds / d0)) + fmin * (ds / d0)
        co2abs = co2 * (const.mco2 / const.mair) * const.rho
        ci = cfrac * (co2abs - co2comp) + co2comp
        return ci, co2abs

    def compute_max_gross_primary_production(self, thetasurf: Array) -> Array:
        """Compute maximal gross primary production in high light conditions ``ammax``.

        Notes:
            The maximal gross primary production is given by

            .. math::
                A_{m,max} = A_{m,max,298} Q_{10} e^{(T_s-298)/10} f_Q(T_s),

            where :math:`A_{m,max,298}` is the value at 298 K, :math:`Q_{10}` is the temperature coefficient,
            and :math:`f_Q(T_s)` is a temperature correction function given by

            .. math::
                f_Q(T_s) = \\frac{1}{[1 + \\exp(0.3(T_{1Am}-T_s))] [1 + \\exp(0.3(T_s-T_{2Am}))]}.

        References:
            Equation E.3 from the CLASS book.
        """
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10Am[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1Am[self.c3c4] - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2Am[self.c3c4]))
        ammax = self.ammax298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return ammax

    def compute_soil_moisture_stress_factor(self, w2: float) -> Array:
        """Compute effect of soil moisture stress on gross assimilation rate ``fstr``.

        Notes:
            The soil moisture stress factor is calculated based on the relative soil moisture content
            and the parameter :math:`c_{\\beta}`.

            .. math::
                f_{str} = \\frac{1 - e^{-p \\beta_w}}{1 - e^{-p}}

            where :math:`\\beta_w` is the relative soil moisture and :math:`p` depends on :math:`c_{\\beta}`.

        References:
            Equation E.19 from the CLASS book.
        """
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

    def compute_gross_assimilation(
        self,
        ammax: Array,
        gm: Array,
        ci: Array,
        co2comp: Array,
    ) -> Array:
        """Compute gross assimilation rate (am).

        Notes:
            The gross assimilation rate is given by

            .. math::
                A_m = A_{m,max} \\left[ 1 - \\exp\\left( -\\frac{g_m(C_i - \\gamma)}{A_{m,max}} \\right) \\right]
        """
        assimilation_factor = -(gm * (ci - co2comp) / ammax)
        am = ammax * (1.0 - jnp.exp(assimilation_factor))
        return am

    def compute_dark_respiration(self, am: Array) -> Array:
        """Compute dark respiration (rdark) as a fraction of gross assimilation.

        Notes:
            Dark respiration is assumed to be proportional to gross assimilation:

            .. math::
                R_{dark} = \\frac{1}{9} A_m
        """
        rdark = (1.0 / 9.0) * am
        return rdark

    def compute_absorbed_par(
        self,
        in_srad: Array,
    ) -> Array:
        """Compute absorbed photosynthetically active rad (PAR).

        Notes:
            Absorbed PAR is estimated as 50% of the incoming shortwave rad scaled by vegetation cover:

            .. math::
                PAR = 0.5 \\cdot S_{\\downarrow} \\cdot c_{veg}
        """
        par = 0.5 * jnp.maximum(1e-1, in_srad * self.cveg)
        return par

    def compute_light_use_efficiency(
        self,
        co2abs: Array,
        co2comp: Array,
    ) -> Array:
        """Compute light use efficiency (alphac).

        Notes:
            The light use efficiency is given by

            .. math::
                \\alpha_c = \\alpha_0 \\frac{C_a - \\gamma}{C_a + 2\\gamma}
        """
        co2_ratio = (co2abs - co2comp) / (co2abs + 2.0 * co2comp)
        alphac = self.alpha0[self.c3c4] * co2_ratio
        return alphac

    def compute_canopy_co2_conductance(
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
        """Compute upscaling from leaf to canopy and CO2 conductance at canopy level ``gcco2``.

        Notes:
            The canopy conductance is obtained by integrating the leaf conductance over the canopy depth,
            assuming an exponential decay of rad and photosynthetic capacity.

        References:
            Equations E.13, E.14, E.15 from the CLASS book.
        """
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

    def compute_rs(self, gcco2: Array):
        """Compute surface resistance from canopy CO2 conductance.

        Notes:
            The surface resistance is related to the canopy CO2 conductance by

            .. math::
                r_s = \\frac{1}{1.6 g_{c,CO2}}

            where the factor 1.6 accounts for the ratio of diffusivities of water vapor and CO2.
        """
        return 1.0 / (1.6 * gcco2)

    def update_surface_resistance(
        self, state: AbstractCoupledState, const: PhysicalConstants
    ) -> AbstractCoupledState:
        """Compute surface resistance using Ags photosynthesis-conductance model."""
        land_state = state.land
        ml_state = state.atmos.mixed
        sl_state = state.atmos.surface
        thetasurf = sl_state.thetasurf
        co2comp = self.compute_co2comp(thetasurf, const.rho)
        gm = self.compute_gm(thetasurf)
        fmin = self.compute_fmin(gm)
        ds = self.compute_ds(thetasurf, land_state.e)
        d0 = self.compute_d0(fmin)
        ci, co2abs = self.compute_internal_co2(
            ds,
            d0,
            fmin,
            ml_state.co2,
            co2comp,
            const,
        )
        ammax = self.compute_max_gross_primary_production(thetasurf)
        fstr = self.compute_soil_moisture_stress_factor(self.w2)
        am = self.compute_gross_assimilation(ammax, gm, ci, co2comp)
        rdark = self.compute_dark_respiration(am)
        par = self.compute_absorbed_par(state.in_srad)
        alphac = self.compute_light_use_efficiency(co2abs, co2comp)
        gcco2 = self.compute_canopy_co2_conductance(
            alphac,
            par,
            am,
            rdark,
            fstr,
            co2abs,
            co2comp,
            ds,
            d0,
            fmin,
        )
        rs = self.compute_rs(gcco2)
        new_land = replace(land_state, ci=ci, co2abs=co2abs, gcco2=gcco2, rs=rs)
        return state.replace(land=new_land)

    def compute_surface_co2_resistance(self, gcco2: Array) -> Array:
        """Compute surface resistance to CO₂ (rsCO2) from canopy conductance.

        Notes:
            .. math::
                r_{s,CO2} = \\frac{1}{g_{c,CO2}}
        """
        return 1.0 / gcco2

    def compute_net_assimilation(
        self, co2abs: Array, ci: Array, ra: Array, rsCO2: Array
    ) -> Array:
        """Compute net CO₂ assimilation rate (an).

        Notes:
            The net assimilation rate is given by the diffusion equation:

            .. math::
                A_n = -\\frac{C_a - C_i}{r_a + r_{s,CO2}}
        """
        return -(co2abs - ci) / (ra + rsCO2)

    def compute_soil_water_fraction(self, wg: Array) -> Array:
        """Compute soil water fraction (fw) for respiration scaling.

        Notes:
            .. math::
                f_w = \\frac{c_w W_{max}}{w_g + w_{min}}
        """
        return self.cw * self.wmax / (wg + self.wmin)

    def compute_respiration(
        self,
        temp_soil: Array,
        fw: Array,
    ) -> Array:
        """Compute soil respiration (resp) as a function of temperature and soil water.

        Notes:
            Soil respiration is given by

            .. math::
                R_{soil} = R_{10} (1 - f_w) \\exp\\left( \\frac{E_0}{R T_{ref}} (1 - T_{ref}/T_{soil}) \\right)

            where :math:`R_{10}` is the reference respiration at 10°C, :math:`E_0` is the activation energy,
            and :math:`T_{ref} = 283.15` K.
        """
        temp_ratio = 1.0 - 283.15 / temp_soil
        resp_factor = jnp.exp(self.e0 / (283.15 * 8.314) * temp_ratio)
        resp = self.r10 * (1.0 - fw) * resp_factor
        return resp

    def scale_flux_to_mol(
        self,
        flux: Array,
        const: PhysicalConstants,
    ) -> Array:
        """Scale a flux to mol m⁻² s⁻¹ using physical constants.

        Notes:
            The scaling is given by

            .. math::
                F_{mol} = F \\frac{M_{air}}{\\rho M_{CO2}}
        """
        return flux * (const.mair / (const.rho * const.mco2))

    def update_co2_flux(
        self, state: AbstractCoupledState, const: PhysicalConstants
    ) -> AbstractCoupledState:
        """Compute the CO₂ flux and update the state.

        Notes:
            This method updates the CO2 flux variables in the state:
            - ``rsCO2``: Surface resistance to CO2
            - ``wCO2A``: Net assimilation flux (scaled to mol)
            - ``wCO2R``: Respiration flux (scaled to mol)
            - ``wCO2``: Total CO2 flux
        """
        land_state = state.land
        sl_state = state.atmos.surface
        rsCO2 = self.compute_surface_co2_resistance(land_state.gcco2)
        an = self.compute_net_assimilation(
            land_state.co2abs, land_state.ci, sl_state.ra, rsCO2
        )
        fw = self.compute_soil_water_fraction(land_state.wg)
        resp = self.compute_respiration(land_state.temp_soil, fw)
        wCO2A = self.scale_flux_to_mol(an, const)
        wCO2R = self.scale_flux_to_mol(resp, const)
        wCO2 = wCO2A + wCO2R
        new_land = replace(land_state, rsCO2=rsCO2, wCO2A=wCO2A, wCO2R=wCO2R, wCO2=wCO2)
        return state.replace(land=new_land)
