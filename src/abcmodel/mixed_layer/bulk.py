from dataclasses import dataclass

import numpy as np
from jaxtyping import PyTree

from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel

# conversion factor mgC m-2 s-1 to ppm m s-1
# limamau: this conversion could be done in a post-processing
# function after jax.lax.scan just like in neuralgcm/dinosaur
# FAC = const.mair / (const.rho * const.mco2)


@dataclass
class BulkMixedLayerInitConds:
    """Data class for bulk mixed layer model initial conditions.

    Arguments
    ---------
    - ``abl_height``: initial ABL height [m].
    - ``theta``: initial mixed-layer potential temperature [K].
    - ``dtheta``: initial temperature jump at h [K].
    - ``wtheta``: surface kinematic heat flux [K m/s].
    - ``q``: initial mixed-layer specific humidity [kg/kg].
    - ``dq``: initial specific humidity jump at h [kg/kg].
    - ``wq``: surface kinematic moisture flux [kg/kg m/s].
    - ``co2``: initial mixed-layer CO2 [ppm].
    - ``dCO2``: initial CO2 jump at h [ppm].
    - ``wCO2``: surface kinematic CO2 flux [mgC/m²/s].
    - ``u``: initial mixed-layer u-wind speed [m/s].
    - ``du``: initial u-wind jump at h [m/s].
    - ``v``: initial mixed-layer v-wind speed [m/s].
    - ``dv``: initial v-wind jump at h [m/s].
    - ``dz_h``: transition layer thickness [m].
    - ``wstar``: convective velocity scale [m s-1]. Defaults to 0.0.
    - ``we``: entrainment velocity [m s-1]. Defaults to -1.0.
    - ``wCO2A``: surface assimulation CO2 flux [mgC/m²/s]. Defaults to 0.0.
    - ``wCO2R``: surface respiration CO2 flux [mgC/m²/s]. Defaults to 0.0.
    - ``wCO2M``: CO2 mass flux [mgC/m²/s]. Defaults to 0.0.
    """

    # initialized by the user
    abl_height: float
    theta: float
    dtheta: float
    wtheta: float
    q: float
    dq: float
    wq: float
    co2: float
    dCO2: float
    wCO2: float
    u: float
    du: float
    v: float
    dv: float
    dz_h: float
    # this is actually not updated
    surf_pressure: float

    # initialized to zero by default
    wstar: float = 0.0
    we: float = -1.0
    wCO2A: float = 0.0
    wCO2R: float = 0.0
    wCO2M: float = 0.0

    # should be initialized during warmup
    thetav: float = np.nan
    wthetav: float = np.nan
    wqe: float = np.nan
    qsat: float = np.nan
    e: float = np.nan
    esat: float = np.nan
    wCO2e: float = np.nan
    wthetae: float = np.nan
    dthetav: float = np.nan
    wthetave: float = np.nan
    lcl: float = np.nan
    top_rh: float = np.nan
    utend: float = np.nan
    dutend: float = np.nan
    vtend: float = np.nan
    dvtend: float = np.nan
    htend: float = np.nan
    thetatend: float = np.nan
    dthetatend: float = np.nan
    qtend: float = np.nan
    dqtend: float = np.nan
    co2tend: float = np.nan
    dCO2tend: float = np.nan
    dztend: float = np.nan


class BulkMixedLayerModel(AbstractStandardStatsModel):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    Arguments
    ---------
    - ``sw_shearwe``: shear growth mixed-layer switch.
    - ``sw_fixft``: fix the free-troposphere switch.
    - ``sw_wind``: prognostic wind switch.
    - ``surf_pressure``: surface pressure [Pa].
    - ``divU``: horizontal large-scale divergence of wind [s⁻¹].
    - ``coriolis_param``: Coriolis parameter [s⁻¹].
    - ``gammatheta``: free atmosphere potential temperature lapse rate [K/m].
    - ``advtheta``: advection of heat [K/s].
    - ``beta``: entrainment ratio for virtual heat [-].
    - ``gammaq``: free atmosphere specific humidity lapse rate [kg/kg/m].
    - ``advq``: advection of moisture [kg/kg/s].
    - ``gammaCO2``: free atmosphere CO2 lapse rate [ppm/m].
    - ``advCO2``: advection of CO2 [ppm/s].
    - ``gammau``: free atmosphere u-wind speed lapse rate [s⁻¹].
    - ``advu``: advection of u-wind [m/s²].
    - ``gammav``: free atmosphere v-wind speed lapse rate [s⁻¹].
    - ``advv``: advection of v-wind [m/s²].
    - ``dFz``: something I forgot :).

    Processes
    ---------
    1. Calculate large-scale vertical motions and compensating effects.
    2. Determine convective velocity scale and entrainment parameters.
    3. Compute all tendency terms for mixed layer variables.
    4. Integrate prognostic equations forward in time.
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
        abl_height: float,
        dtheta: float,
        const: PhysicalConstants,
    ) -> tuple[float, float]:
        """Calculate large-scale subsidence and radiative divergence effects."""
        # calculate large-scale vertical velocity (subsidence)
        ws = -self.divU * abl_height

        # calculate mixed-layer growth due to cloud top radiative divergence
        radiative_denominator = const.rho * const.cp * dtheta
        wf = self.dFz / radiative_denominator

        return ws, wf

    def calculate_free_troposphere_compensation(self, ws: float):
        """Calculate compensation terms to fix free troposphere values."""
        if self.sw_fixft:
            w_th_ft = self.gammatheta * ws
            w_q_ft = self.gammaq * ws
            w_CO2_ft = self.gammaCO2 * ws
        else:
            w_th_ft = 0.0
            w_q_ft = 0.0
            w_CO2_ft = 0.0
        return w_th_ft, w_q_ft, w_CO2_ft

    def calculate_convective_velocity_scale(
        self,
        abl_height: float,
        wthetav: float,
        thetav: float,
        g: float,
    ):
        """Calculate convective velocity scale and entrainment parameters."""
        if wthetav > 0.0:
            buoyancy_term = g * abl_height * wthetav / thetav
            wstar = buoyancy_term ** (1.0 / 3.0)
        else:
            wstar = 1e-6

        # virtual heat entrainment flux
        wthetave = -self.beta * wthetav

        return wstar, wthetave

    def calculate_entrainment_velocity(
        self,
        abl_height: float,
        wthetave: float,
        dthetav: float,
        thetav: float,
        we: float,
        ustar: float,
        g: float,
    ):
        """Calculate entrainment velocity with optional shear effects."""
        if self.sw_shearwe:
            shear_term = 5.0 * ustar**3.0 * thetav / (g * abl_height)
            numerator = -wthetave + shear_term
            we = numerator / dthetav
        else:
            we = -wthetave / dthetav

        # don't allow boundary layer shrinking if wtheta < 0
        # limamau: we need to change that for nightime?
        if we < 0:
            we = 0.0

        return we

    @staticmethod
    def calculate_entrainment_fluxes(we, dtheta, dq, dCO2):
        """Calculate all entrainment fluxes."""
        wthetae = -we * dtheta
        wqe = -we * dq
        wCO2e = -we * dCO2
        return wthetae, wqe, wCO2e

    def calculate_mixed_layer_tendencies(
        self,
        ws: float,
        wf: float,
        wq: float,
        wqe: float,
        we: float,
        cc_mf: float,
        cc_qf: float,
        wtheta: float,
        wthetae: float,
        abl_height: float,
        wCO2: float,
        wCO2e: float,
        wCO2M: float,
        w_th_ft: float,
        w_q_ft: float,
        w_CO2_ft: float,
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
        we: float,
        wf: float,
        uw: float,
        vw: float,
        cc_mf: float,
        du: float,
        dv: float,
        abl_height: float,
    ) -> tuple[float, float, float, float]:
        """Calculate wind tendency terms if wind is prognostic."""
        # assume u + du = ug, so ug - u = du
        if self.sw_wind:
            coriolis_term_u = -self.coriolis_param * dv
            momentum_flux_term_u = (uw + we * du) / abl_height
            utend = coriolis_term_u + momentum_flux_term_u + self.advu

            coriolis_term_v = self.coriolis_param * du
            momentum_flux_term_v = (vw + we * dv) / abl_height
            vtend = coriolis_term_v + momentum_flux_term_v + self.advv

            entrainment_growth_term = we + wf - cc_mf
            dutend = self.gammau * entrainment_growth_term - utend
            dvtend = self.gammav * entrainment_growth_term - vtend

            return utend, vtend, dutend, dvtend

        else:
            return 0.0, 0.0, 0.0, 0.0

    def calculate_transition_layer_tendency(
        self,
        lcl: float,
        abl_height: float,
        cc_frac: float,
        dz_h: float,
    ):
        """Calculate transition layer thickness tendency."""
        lcl_distance = lcl - abl_height

        if cc_frac > 0 or lcl_distance < 300:
            target_thickness = lcl_distance - dz_h
            dztend = target_thickness / 7200.0
        else:
            dztend = 0.0

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
            state.we,
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
        dz0 = 50
        if state.dz_h < dz0:
            state.dz_h = dz0

        if self.sw_wind:
            state.u += dt * state.utend
            state.du += dt * state.dutend
            state.v += dt * state.vtend
            state.dv += dt * state.dvtend

        return state
