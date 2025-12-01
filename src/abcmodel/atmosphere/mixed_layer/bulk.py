from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...utils import PhysicalConstants
from .stats import AbstractStandardStatsModel

# conversion factor mgC m-2 s-1 to ppm m s-1
# limamau: this conversion could be done in a post-processing
# function after jax.lax.scan just like in neuralgcm/dinosaur
# FAC = const.mair / (const.rho * const.mco2)


@dataclass
class BulkMixedLayerInitConds:
    """Data class for bulk mixed layer model initial state."""

    # initialized by the user
    abl_height: float
    """Initial ABL height [m]."""
    theta: float
    """Initial mixed-layer potential temperature [K]."""
    dtheta: float
    """Initial temperature jump at h [K]."""
    wtheta: float
    """Surface kinematic heat flux [K m/s]."""
    q: float
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: float
    """Initial specific humidity jump at h [kg/kg]."""
    wq: float
    """Surface kinematic moisture flux [kg/kg m/s]."""
    co2: float
    """Initial mixed-layer CO2 [ppm]."""
    dCO2: float
    """Initial CO2 jump at h [ppm]."""
    wCO2: float
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: float
    """Initial mixed-layer u-wind speed [m/s]."""
    du: float
    """Initial u-wind jump at h [m/s]."""
    v: float
    """Initial mixed-layer v-wind speed [m/s]."""
    dv: float
    """Initial v-wind jump at h [m/s]."""
    dz_h: float
    """Transition layer thickness [m]."""
    surf_pressure: float
    """Surface pressure, which is actually not updated (not a state), it's only here for simplicity [Pa]."""

    # initialized to zero by default
    wstar: float = 0.0
    """Convective velocity scale [m s-1]."""
    we: float = -1.0
    """Entrainment velocity [m s-1]."""
    wCO2A: float = 0.0
    """Surface assimulation CO2 flux [mgC/m²/s]."""
    wCO2R: float = 0.0
    """Surface respiration CO2 flux [mgC/m²/s]."""
    wCO2M: float = 0.0
    """CO2 mass flux [mgC/m²/s]."""

    # should be initialized during warmup
    thetav: float = jnp.nan
    """Mixed-layer potential temperature [K]."""
    dthetav: float = jnp.nan
    """Virtual temperature jump at h [K]."""
    wthetav: float = jnp.nan
    """Surface kinematic virtual heat flux [K m s-1]."""
    wqe: float = jnp.nan
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    qsat: float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    e: float = jnp.nan
    """Vapor pressure [Pa]."""
    esat: float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    wCO2e: float = jnp.nan
    """Entrainment CO2 flux [mgC/m²/s]."""
    wthetae: float = jnp.nan
    """Entrainment potential temperature flux [K m s-1]."""
    wthetave: float = jnp.nan
    """Entrainment virtual heat flux [K m s-1]."""
    lcl: float = jnp.nan
    """Lifting condensation level [m]."""
    top_rh: float = jnp.nan
    """Top of mixed layer relative humidity [%]."""
    top_p: float = jnp.nan
    """Pressure at top of mixed layer [Pa]."""
    top_T: float = jnp.nan
    """Temperature at top of mixed layer [K]."""
    utend: float = jnp.nan
    """Zonal wind velocity tendency [m s-2]."""
    dutend: float = jnp.nan
    """Zonal wind velocity tendency at the ABL height [m s-2]."""
    vtend: float = jnp.nan
    """Meridional wind velocity tendency [m s-2]."""
    dvtend: float = jnp.nan
    """Meridional wind velocity tendency at the ABL height [m/s²]."""
    htend: float = jnp.nan
    """Tendency of CBL [m s-1]."""
    thetatend: float = jnp.nan
    """Tendency of mixed-layer potential temperature [K s-1]."""
    dthetatend: float = jnp.nan
    """Tendency of mixed-layer potential temperature at the ABL height [K s-1]."""
    qtend: float = jnp.nan
    """Tendency of mixed-layer specific humidity [kg/kg s-1]."""
    dqtend: float = jnp.nan
    """Tendency of mixed-layer specific humidity at the ABL height [kg/kg s-1]."""
    co2tend: float = jnp.nan
    """Tendency of CO2 concentration [ppm s-1]."""
    dCO2tend: float = jnp.nan
    """Tendency of CO2 concentration at the ABL height [ppm s-1]."""
    dztend: float = jnp.nan
    """Tendency of transition layer thickness [m s-1]."""
    ws: float = jnp.nan
    """Large-scale vertical velocity (subsidence) [m s-1]."""
    wf: float = jnp.nan
    """Mixed-layer growth due to cloud top radiative divergence [m s-1]."""


class BulkMixedLayerModel(AbstractStandardStatsModel):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    1. Calculate large-scale vertical motions and compensating effects.
    2. Determine convective velocity scale and entrainment parameters.
    3. Compute all tendency terms for mixed layer variables.
    4. Integrate prognostic equations forward in time.

    Args:
        divU: horizontal large-scale divergence of wind [s-1].
        coriolis_param: Coriolis parameter [s-1].
        gammatheta: free atmosphere potential temperature lapse rate [K m-1].
        advtheta: advection of heat [K s-1].
        beta: entrainment ratio for virtual heat [-].
        gammaq: free atmosphere specific humidity lapse rate [kg/kg m-1].
        advq: advection of moisture [kg/kg s-1].
        gammaCO2: free atmosphere CO2 lapse rate [ppm m-1].
        advCO2: advection of CO2 [ppm s-1].
        gammau: free atmosphere u-wind speed lapse rate [s-1].
        advu: advection of u-wind [m s-2].
        gammav: free atmosphere v-wind speed lapse rate [s-1].
        advv: advection of v-wind [m s-2].
        dFz: cloud top radiative divergence [W m-2].
        is_shear_growing: shear growth mixed-layer switch.
        is_fix_free_trop: fix the free-troposphere switch.
        is_wind_prog: prognostic wind switch.
    """

    def __init__(
        self,
        divU: float,
        coriolis_param: float,
        gammatheta: float,
        advtheta: float,
        beta: float,
        gammaq: float,
        advq: float,
        gammaCO2: float,
        advCO2: float,
        gammau: float,
        advu: float,
        gammav: float,
        advv: float,
        dFz: float,
        is_shear_growing: bool = True,
        is_fix_free_trop: bool = True,
        is_wind_prog: bool = True,
    ):
        self.divU = divU
        self.coriolis_param = coriolis_param
        self.gammatheta = gammatheta
        self.advtheta = advtheta
        self.beta = beta
        self.gammaq = gammaq
        self.advq = advq
        self.gammaCO2 = gammaCO2
        self.advCO2 = advCO2
        self.gammau = gammau
        self.advu = advu
        self.gammav = gammav
        self.advv = advv
        self.dFz = dFz
        self.is_shear_growing = is_shear_growing
        self.is_fix_free_trop = is_fix_free_trop
        self.is_wind_prog = is_wind_prog

    def run(self, state: PyTree, const: PhysicalConstants):
        """Run the model."""
        state.ws = self.compute_subsidence_velocity(state.abl_height)
        state.wf = self.compute_radiative_growth_velocity(state.dtheta, const)

        w_th_ft = self.compute_free_troposphere_theta_compensation(state.ws)
        w_q_ft = self.compute_free_troposphere_q_compensation(state.ws)
        w_CO2_ft = self.compute_free_troposphere_co2_compensation(state.ws)
        state.wstar = self.compute_convective_velocity_scale(
            state.abl_height,
            state.wthetav,
            state.thetav,
            const.g,
        )
        state.wthetave = self.compute_entrainment_virtual_heat_flux(state.wthetav)
        state.we = self.compute_entrainment_velocity(
            state.abl_height,
            state.wthetave,
            state.dthetav,
            state.thetav,
            state.ustar,
            const.g,
        )
        state.wthetae = self.compute_entrainment_heat_flux(state.we, state.dtheta)
        state.wqe = self.compute_entrainment_moisture_flux(state.we, state.dq)
        state.wCO2e = self.compute_entrainment_co2_flux(state.we, state.dCO2)
        state.htend = self.compute_abl_height_tendency(
            state.we, state.ws, state.wf, state.cc_mf
        )
        state.thetatend = self.compute_potential_temperature_tendency(
            state.abl_height, state.wtheta, state.wthetae
        )
        state.dthetatend = self.compute_potential_temperature_jump_tendency(
            state.we, state.wf, state.cc_mf, state.thetatend, w_th_ft
        )
        state.qtend = self.compute_humidity_tendency(
            state.abl_height, state.wq, state.wqe, state.cc_qf
        )
        state.dqtend = self.compute_humidity_jump_tendency(
            state.we, state.wf, state.cc_mf, state.qtend, w_q_ft
        )
        state.co2tend = self.compute_co2_tendency(
            state.abl_height, state.wCO2, state.wCO2e, state.wCO2M
        )
        state.dCO2tend = self.compute_co2_jump_tendency(
            state.we, state.wf, state.cc_mf, state.co2tend, w_CO2_ft
        )
        state.utend = self.compute_u_wind_tendency(
            state.abl_height, state.we, state.uw, state.du, state.dv
        )
        state.vtend = self.compute_v_wind_tendency(
            state.abl_height, state.we, state.vw, state.du, state.dv
        )
        state.dutend = self.compute_u_wind_jump_tendency(
            state.we, state.wf, state.cc_mf, state.utend
        )
        state.dvtend = self.compute_v_wind_jump_tendency(
            state.we, state.wf, state.cc_mf, state.vtend
        )
        state.dztend = self.compute_transition_layer_tendency(
            state.lcl,
            state.abl_height,
            state.cc_frac,
            state.dz_h,
        )

        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        """Integrate mixed layer forward in time."""
        state.abl_height += dt * state.htend
        state.theta += dt * state.thetatend
        state.dtheta += dt * state.dthetatend
        state.q += dt * state.qtend
        state.dq += dt * state.dqtend
        state.co2 += dt * state.co2tend
        state.dCO2 += dt * state.dCO2tend
        state.dz_h += dt * state.dztend

        # limit dz to minimal value
        state.dz_h = jnp.maximum(state.dz_h, 50.0)

        state.u = jnp.where(self.is_wind_prog, state.u + dt * state.utend, state.u)
        state.du = jnp.where(self.is_wind_prog, state.du + dt * state.dutend, state.du)
        state.v = jnp.where(self.is_wind_prog, state.v + dt * state.vtend, state.v)
        state.dv = jnp.where(self.is_wind_prog, state.dv + dt * state.dvtend, state.dv)

        return state

    def compute_subsidence_velocity(self, abl_height: Array) -> Array:
        """Compute large-scale subsidence velocity.

        Notes:
            The large-scale vertical velocity (subsidence) :math:`w_s` is given by

            .. math::
                w_s = -\\text{div}U \\cdot h

            where :math:`\\text{div}U` is the horizontal large-scale divergence of wind and :math:`h` is the ABL height.
        """
        return -self.divU * abl_height

    def compute_radiative_growth_velocity(
        self, dtheta: Array, const: PhysicalConstants
    ) -> Array:
        """Compute mixed-layer growth due to cloud top radiative divergence.

        Notes:
            The mixed-layer growth due to cloud top radiative divergence :math:`w_f` is given by

            .. math::
                w_f = \\frac{\\Delta F_z}{\\rho c_p \\Delta \\theta}

            where :math:`\\Delta F_z` is the cloud top radiative divergence, :math:`\\rho` is air density,
            :math:`c_p` is specific heat capacity, and :math:`\\Delta \\theta` is the temperature jump.
        """
        radiative_denominator = const.rho * const.cp * dtheta
        return self.dFz / radiative_denominator

    def compute_free_troposphere_theta_compensation(self, ws: Array) -> Array:
        """Compute potential temperature compensation term to fix free troposphere values.

        Notes:
            .. math::
                w_{\\theta,ft} = \\gamma_\\theta w_s
        """
        w_th_ft_active = self.gammatheta * ws
        return jnp.where(self.is_fix_free_trop, w_th_ft_active, 0.0)

    def compute_free_troposphere_q_compensation(self, ws: Array) -> Array:
        """Compute humidity compensation term to fix free troposphere values.

        Notes:
            .. math::
                w_{q,ft} = \\gamma_q w_s
        """
        w_q_ft_active = self.gammaq * ws
        return jnp.where(self.is_fix_free_trop, w_q_ft_active, 0.0)

    def compute_free_troposphere_co2_compensation(self, ws: Array) -> Array:
        """Compute CO2 compensation term to fix free troposphere values.

        Notes:
            .. math::
                w_{CO2,ft} = \\gamma_{CO2} w_s
        """
        w_CO2_ft_active = self.gammaCO2 * ws
        return jnp.where(self.is_fix_free_trop, w_CO2_ft_active, 0.0)

    def compute_convective_velocity_scale(
        self,
        abl_height: Array,
        wthetav: Array,
        thetav: Array,
        g: float,
    ) -> Array:
        """Compute convective velocity scale.

        Notes:
            The convective velocity scale :math:`w_*` is given by

            .. math::
                w_* = \\left( \\frac{g h \\overline{w'\\theta_v'}_s}{\\theta_v} \\right)^{1/3}
        """
        # calculate wstar for positive wthetav case
        buoyancy_term = g * abl_height * wthetav / thetav
        wstar_positive = buoyancy_term ** (1.0 / 3.0)
        return jnp.where(wthetav > 0.0, wstar_positive, 1e-6)

    def compute_entrainment_virtual_heat_flux(self, wthetav: Array) -> Array:
        """Compute entrainment virtual heat flux.

        Notes:
            The entrainment virtual heat flux :math:`\\overline{w'\\theta_v'}_e` is parametrized as

            .. math::
                \\overline{w'\\theta_v'}_e = -\\beta \\overline{w'\\theta_v'}_s
        """
        return -self.beta * wthetav

    def compute_entrainment_velocity(
        self,
        abl_height: Array,
        wthetave: Array,
        dthetav: Array,
        thetav: Array,
        ustar: Array,
        g: float,
    ):
        """Compute entrainment velocity with optional shear effects.

        Notes:
            The entrainment velocity :math:`w_e` is given by

            .. math::
                w_e = -\\frac{\\overline{w'\\theta_v'}_e}{\\Delta \\theta_v}

            If shear effects are included (``is_shear_growing`` is True), an additional term is added:

            .. math::
                w_e = \\frac{-\\overline{w'\\theta_v'}_e + 5 u_*^3 \\theta_v / (g h)}{\\Delta \\theta_v}
        """
        # entrainment velocity with shear effects
        shear_term = 5.0 * ustar**3.0 * thetav / (g * abl_height)
        numerator = -wthetave + shear_term
        we_with_shear = numerator / dthetav

        # entrainment velocity without shear effects
        we_no_shear = -wthetave / dthetav

        # select based on is_shear_growing flag
        we_calculated = jnp.where(self.is_shear_growing, we_with_shear, we_no_shear)

        # don't allow boundary layer shrinking if wtheta < 0
        assert isinstance(we_calculated, jnp.ndarray)  # limmau: this is not good
        we_final = jnp.where(we_calculated < 0.0, 0.0, we_calculated)

        return we_final

    @staticmethod
    def compute_entrainment_heat_flux(we: Array, dtheta: Array) -> Array:
        """Compute entrainment heat flux.

        Notes:
            .. math::
                \\overline{w'\\theta'}_e = -w_e \\Delta \\theta
        """
        return -we * dtheta

    @staticmethod
    def compute_entrainment_moisture_flux(we: Array, dq: Array) -> Array:
        """Compute entrainment moisture flux.

        Notes:
            .. math::
                \\overline{w'q'}_e = -w_e \\Delta q
        """
        return -we * dq

    @staticmethod
    def compute_entrainment_co2_flux(we: Array, dCO2: Array) -> Array:
        """Compute entrainment CO2 flux.

        Notes:
            .. math::
                \\overline{w'CO_2'}_e = -w_e \\Delta CO_2
        """
        return -we * dCO2

    @staticmethod
    def compute_abl_height_tendency(
        we: Array, ws: Array, wf: Array, cc_mf: Array
    ) -> Array:
        """Compute boundary layer height tendency.

        Notes:
            .. math::
                \\frac{dh}{dt} = w_e + w_s + w_f - \\text{cc}_{mf}
        """
        return we + ws + wf - cc_mf

    def compute_potential_temperature_tendency(
        self, abl_height: Array, wtheta: Array, wthetae: Array
    ) -> Array:
        """Compute mixed-layer potential temperature tendency.

        Notes:
            .. math::
                \\frac{d\\theta}{dt} = \\frac{\\overline{w'\\theta'}_s - \\overline{w'\\theta'}_e}{h} + \\text{adv}_\\theta
        """
        surface_heat_flux = (wtheta - wthetae) / abl_height
        return surface_heat_flux + self.advtheta

    def compute_potential_temperature_jump_tendency(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        thetatend: Array,
        w_th_ft: Array,
    ) -> Array:
        """Compute potential temperature jump tendency.

        Notes:
            .. math::
                \\frac{d\\Delta \\theta}{dt} = \\gamma_\\theta (w_e + w_f - \\text{cc}_{mf}) - \\frac{d\\theta}{dt} + w_{\\theta,ft}
        """
        egrowth = we + wf - cc_mf
        return self.gammatheta * egrowth - thetatend + w_th_ft

    def compute_humidity_tendency(
        self, abl_height: Array, wq: Array, wqe: Array, cc_qf: Array
    ) -> Array:
        """Compute mixed-layer specific humidity tendency.

        Notes:
            .. math::
                \\frac{dq}{dt} = \\frac{\\overline{w'q'}_s - \\overline{w'q'}_e - \\text{cc}_{qf}}{h} + \\text{adv}_q
        """
        surface_moisture_flux = (wq - wqe - cc_qf) / abl_height
        return surface_moisture_flux + self.advq

    def compute_humidity_jump_tendency(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        qtend: Array,
        w_q_ft: Array,
    ) -> Array:
        """Compute specific humidity jump tendency.

        Notes:
            .. math::
                \\frac{d\\Delta q}{dt} = \\gamma_q (w_e + w_f - \\text{cc}_{mf}) - \\frac{dq}{dt} + w_{q,ft}
        """
        egrowth = we + wf - cc_mf
        return self.gammaq * egrowth - qtend + w_q_ft

    def compute_co2_tendency(
        self,
        abl_height: Array,
        wCO2: Array,
        wCO2e: Array,
        wCO2M: Array,
    ) -> Array:
        """Compute mixed-layer CO2 tendency.

        Notes:
            .. math::
                \\frac{dCO_2}{dt} = \\frac{\\overline{w'CO_2'}_s - \\overline{w'CO_2'}_e - \\text{cc}_{CO2f}}{h} + \\text{adv}_{CO2}
        """
        surface_co2_flux_term = (wCO2 - wCO2e - wCO2M) / abl_height
        return surface_co2_flux_term + self.advCO2

    def compute_co2_jump_tendency(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        co2tend: Array,
        w_CO2_ft: Array,
    ) -> Array:
        """Compute CO2 jump tendency.

        Notes:
            .. math::
                \\frac{d\\Delta CO_2}{dt} = \\gamma_{CO2} (w_e + w_f - \\text{cc}_{mf}) - \\frac{dCO_2}{dt} + w_{CO2,ft}
        """
        egrowth = we + wf - cc_mf
        return self.gammaCO2 * egrowth - co2tend + w_CO2_ft

    def compute_u_wind_tendency(
        self,
        abl_height: Array,
        we: Array,
        uw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute u-wind tendency.

        Notes:
            .. math::
                \\frac{du}{dt} = -f_c \\Delta v + \\frac{\\overline{u'w'}_s + w_e \\Delta u}{h} + \\text{adv}_u
        """
        coriolis_term_u = -self.coriolis_param * dv
        momentum_flux_term_u = (uw + we * du) / abl_height
        utend_active = coriolis_term_u + momentum_flux_term_u + self.advu
        return jnp.where(self.is_wind_prog, utend_active, 0.0)

    def compute_v_wind_tendency(
        self,
        abl_height: Array,
        we: Array,
        vw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute v-wind tendency.

        Notes:
            .. math::
                \\frac{dv}{dt} = f_c \\Delta u + \\frac{\\overline{v'w'}_s + w_e \\Delta v}{h} + \\text{adv}_v
        """
        coriolis_term_v = self.coriolis_param * du
        momentum_flux_term_v = (vw + we * dv) / abl_height
        vtend_active = coriolis_term_v + momentum_flux_term_v + self.advv
        return jnp.where(self.is_wind_prog, vtend_active, 0.0)

    def compute_u_wind_jump_tendency(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        utend: Array,
    ) -> Array:
        """Compute u-wind jump tendency.

        Notes:
            .. math::
                \\frac{d\\Delta u}{dt} = \\gamma_u (w_e + w_f - \\text{cc}_{mf}) - \\frac{du}{dt}
        """
        entrainment_growth_term = we + wf - cc_mf
        dutend_active = self.gammau * entrainment_growth_term - utend
        return jnp.where(self.is_wind_prog, dutend_active, 0.0)

    def compute_v_wind_jump_tendency(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        vtend: Array,
    ) -> Array:
        """Compute v-wind jump tendency.

        Notes:
            .. math::
                \\frac{d\\Delta v}{dt} = \\gamma_v (w_e + w_f - \\text{cc}_{mf}) - \\frac{dv}{dt}
        """
        entrainment_growth_term = we + wf - cc_mf
        dvtend_active = self.gammav * entrainment_growth_term - vtend
        return jnp.where(self.is_wind_prog, dvtend_active, 0.0)

    def compute_transition_layer_tendency(
        self,
        lcl: Array,
        abl_height: Array,
        cc_frac: Array,
        dz_h: Array,
    ):
        """Compute transition layer thickness tendency.

        Notes:
            Relaxation of the transition layer thickness :math:`\\delta z_h` towards a target value:

            .. math::
                \\frac{d\\delta z_h}{dt} = \\frac{(LCL - h) - \\delta z_h}{\\tau}

            where :math:`\\tau = 7200` s.
        """
        lcl_distance = lcl - abl_height

        # tendency for active case
        target_thickness = lcl_distance - dz_h
        dztend_active = target_thickness / 7200.0
        condition = (cc_frac > 0) | (lcl_distance < 300)
        dztend = jnp.where(condition, dztend_active, 0.0)

        return dztend
