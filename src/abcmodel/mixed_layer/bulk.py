from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..utils import PhysicalConstants
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


class BulkMixedLayerModel(AbstractStandardStatsModel):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    1. Calculate large-scale vertical motions and compensating effects.
    2. Determine convective velocity scale and entrainment parameters.
    3. Compute all tendency terms for mixed layer variables.
    4. Integrate prognostic equations forward in time.

    Args:
        sw_shearwe: shear growth mixed-layer switch.
        sw_fixft: fix the free-troposphere switch.
        sw_wind: prognostic wind switch.
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
        sw_shearwe: bool = True,
        sw_fixft: bool = True,
        sw_wind: bool = True,
    ):
        self.sw_shearwe = sw_shearwe
        self.sw_fixft = sw_fixft
        self.sw_wind = sw_wind
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

    def calculate_vertical_motions(
        self,
        abl_height: Array,
        dtheta: Array,
        const: PhysicalConstants,
    ) -> tuple[Array, Array]:
        """Calculate large-scale subsidence and radiative divergence effects."""
        # calculate large-scale vertical velocity (subsidence)
        ws = -self.divU * abl_height

        # calculate mixed-layer growth due to cloud top radiative divergence
        radiative_denominator = const.rho * const.cp * dtheta
        wf = self.dFz / radiative_denominator

        return ws, wf

    def calculate_free_troposphere_compensation(self, ws: Array):
        """Calculate compensation terms to fix free troposphere values."""
        # compensation terms
        w_th_ft_active = self.gammatheta * ws
        w_q_ft_active = self.gammaq * ws
        w_CO2_ft_active = self.gammaCO2 * ws

        # switch based on sw_fixft flag
        w_th_ft = jnp.where(self.sw_fixft, w_th_ft_active, 0.0)
        w_q_ft = jnp.where(self.sw_fixft, w_q_ft_active, 0.0)
        w_CO2_ft = jnp.where(self.sw_fixft, w_CO2_ft_active, 0.0)

        return w_th_ft, w_q_ft, w_CO2_ft

    def calculate_convective_velocity_scale(
        self,
        abl_height: Array,
        wthetav: Array,
        thetav: Array,
        g: float,
    ):
        """Calculate convective velocity scale and entrainment parameters."""
        # calculate wstar for positive wthetav case
        buoyancy_term = g * abl_height * wthetav / thetav
        wstar_positive = buoyancy_term ** (1.0 / 3.0)
        wstar = jnp.where(wthetav > 0.0, wstar_positive, 1e-6)

        # virtual heat entrainment flux
        wthetave = -self.beta * wthetav

        return wstar, wthetave

    def calculate_entrainment_velocity(
        self,
        abl_height: Array,
        wthetave: Array,
        dthetav: Array,
        thetav: Array,
        ustar: Array,
        g: float,
    ):
        """Calculate entrainment velocity with optional shear effects."""
        # entrainment velocity with shear effects
        shear_term = 5.0 * ustar**3.0 * thetav / (g * abl_height)
        numerator = -wthetave + shear_term
        we_with_shear = numerator / dthetav

        # entrainment velocity without shear effects
        we_no_shear = -wthetave / dthetav

        # select based on sw_shearwe flag
        we_calculated = jnp.where(self.sw_shearwe, we_with_shear, we_no_shear)

        # don't allow boundary layer shrinking if wtheta < 0
        assert isinstance(we_calculated, jnp.ndarray)  # limmau: this is not good
        we_final = jnp.where(we_calculated < 0.0, 0.0, we_calculated)

        return we_final

    @staticmethod
    def calculate_entrainment_fluxes(we: Array, dtheta: Array, dq: Array, dCO2: Array):
        """Calculate all entrainment fluxes."""
        wthetae = -we * dtheta
        wqe = -we * dq
        wCO2e = -we * dCO2
        return wthetae, wqe, wCO2e

    def calculate_mixed_layer_tendencies(
        self,
        ws: Array,
        wf: Array,
        wq: Array,
        wqe: Array,
        we: Array,
        cc_mf: Array,
        cc_qf: Array,
        wtheta: Array,
        wthetae: Array,
        abl_height: Array,
        wCO2: Array,
        wCO2e: Array,
        wCO2M: Array,
        w_th_ft: Array,
        w_q_ft: Array,
        w_CO2_ft: Array,
    ):
        """Calculate tendency terms for mixed layer variables."""
        # boundary layer height tendency
        htend = we + ws + wf - cc_mf

        # mixed layer scalar tendencies
        surface_heat_flux = (wtheta - wthetae) / abl_height
        thetatend = surface_heat_flux + self.advtheta

        surface_moisture_flux = (wq - wqe - cc_qf) / abl_height
        qtend = surface_moisture_flux + self.advq

        surface_co2_flux_term = (wCO2 - wCO2e - wCO2M) / abl_height
        co2tend = surface_co2_flux_term + self.advCO2

        # jump tendencies at boundary layer top
        # (entrainment growth term)
        egrowth = we + wf - cc_mf

        dthetatend = self.gammatheta * egrowth - thetatend + w_th_ft
        dqtend = self.gammaq * egrowth - qtend + w_q_ft
        dCO2tend = self.gammaCO2 * egrowth - co2tend + w_CO2_ft

        return htend, thetatend, dthetatend, qtend, dqtend, co2tend, dCO2tend

    def calculate_wind_tendencies(
        self,
        we: Array,
        wf: Array,
        uw: Array,
        vw: Array,
        cc_mf: Array,
        du: Array,
        dv: Array,
        abl_height: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Calculate wind tendency terms if wind is prognostic."""
        # wind tendencies for sw_wind = True case
        coriolis_term_u = -self.coriolis_param * dv
        momentum_flux_term_u = (uw + we * du) / abl_height
        utend_active = coriolis_term_u + momentum_flux_term_u + self.advu

        coriolis_term_v = self.coriolis_param * du
        momentum_flux_term_v = (vw + we * dv) / abl_height
        vtend_active = coriolis_term_v + momentum_flux_term_v + self.advv

        entrainment_growth_term = we + wf - cc_mf
        dutend_active = self.gammau * entrainment_growth_term - utend_active
        dvtend_active = self.gammav * entrainment_growth_term - vtend_active

        # select based on sw_wind flag
        utend = jnp.where(self.sw_wind, utend_active, 0.0)
        vtend = jnp.where(self.sw_wind, vtend_active, 0.0)
        dutend = jnp.where(self.sw_wind, dutend_active, 0.0)
        dvtend = jnp.where(self.sw_wind, dvtend_active, 0.0)

        return utend, vtend, dutend, dvtend

    def calculate_transition_layer_tendency(
        self,
        lcl: Array,
        abl_height: Array,
        cc_frac: Array,
        dz_h: Array,
    ):
        """Calculate transition layer thickness tendency."""
        lcl_distance = lcl - abl_height

        # tendency for active case
        target_thickness = lcl_distance - dz_h
        dztend_active = target_thickness / 7200.0
        condition = (cc_frac > 0) | (lcl_distance < 300)
        dztend = jnp.where(condition, dztend_active, 0.0)

        return dztend

    def run(self, state: PyTree, const: PhysicalConstants):
        """Calculate mixed layer tendencies and update diagnostic variables."""
        state.ws, state.wf = self.calculate_vertical_motions(
            state.abl_height,
            state.dtheta,
            const,
        )
        w_th_ft, w_q_ft, w_CO2_ft = self.calculate_free_troposphere_compensation(
            state.ws,
        )
        state.wstar, state.wthetave = self.calculate_convective_velocity_scale(
            state.abl_height,
            state.wthetav,
            state.thetav,
            const.g,
        )
        state.we = self.calculate_entrainment_velocity(
            state.abl_height,
            state.wthetave,
            state.dthetav,
            state.thetav,
            state.ustar,
            const.g,
        )
        state.wthetae, state.wqe, state.wCO2e = self.calculate_entrainment_fluxes(
            state.we, state.dtheta, state.dq, state.dCO2
        )
        (
            state.htend,
            state.thetatend,
            state.dthetatend,
            state.qtend,
            state.dqtend,
            state.co2tend,
            state.dCO2tend,
        ) = self.calculate_mixed_layer_tendencies(
            state.ws,
            state.wf,
            state.wq,
            state.wqe,
            state.we,
            state.cc_mf,
            state.cc_qf,
            state.wtheta,
            state.wthetae,
            state.abl_height,
            state.wCO2,
            state.wCO2e,
            state.wCO2M,
            w_th_ft,
            w_q_ft,
            w_CO2_ft,
        )
        state.utend, state.vtend, state.dutend, state.dvtend = (
            self.calculate_wind_tendencies(
                state.we,
                state.wf,
                state.uw,
                state.vw,
                state.cc_mf,
                state.du,
                state.dv,
                state.abl_height,
            )
        )
        state.dztend = self.calculate_transition_layer_tendency(
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

        state.u = jnp.where(self.sw_wind, state.u + dt * state.utend, state.u)
        state.du = jnp.where(self.sw_wind, state.du + dt * state.dutend, state.du)
        state.v = jnp.where(self.sw_wind, state.v + dt * state.vtend, state.v)
        state.dv = jnp.where(self.sw_wind, state.dv + dt * state.dvtend, state.dv)

        return state
