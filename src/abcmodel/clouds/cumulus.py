from dataclasses import dataclass

import numpy as np
from jaxtyping import PyTree

from ..models import AbstractCloudModel
from ..utils import PhysicalConstants, get_qsat


@dataclass
class StandardCumulusInitConds:
    """Standard cumulus model state.

    Variables
    ---------
    - ``cc_frac``: cloud core fraction [-], range 0 to 1.
    - ``cc_mf``: cloud core mass flux [m/s].
    - ``cc_qf``: cloud core moisture flux [kg/kg/s].
    """

    cc_frac: float = 0.0
    cc_mf: float = 0.0
    cc_qf: float = 0.0


class StandardCumulusModel(AbstractCloudModel):
    """
    Standard cumulus cloud model based on Neggers et al. (2006/7).

    This model calculates shallow cumulus convection properties using a variance-based
    approach to determine cloud core fraction and associated mass fluxes. The model
    characterizes turbulent fluctuations in the mixed layer that lead to cloud formation.
    It quantifies the variance of humidity and CO2 at the mixed-layer top and uses this
    to determine what fraction reaches saturation.

    Parameters
    ----------
    None.

    Processes
    ---------
    1. Calculates turbulent variance for humidity and CO2 at the mixed-layer top
    using entrainment fluxes and convective scaling (Neggers et al. 2006/7).
    2. Determines fraction of mixed-layer top that becomes saturated using
    arctangent formulation based on saturation deficit.
    3. Calculates mass flux and moisture flux through cloud cores.
    4. Computes CO2 transport only when CO2 decreases with height.
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_mixed_layer_variance(
        cc_qf: float,
        wthetav: float,
        wqe: float,
        dq: float,
        abl_height: float,
        dz_h: float,
        wstar: float,
        wCO2e: float,
        wCO2M: float,
        dCO2: float,
    ) -> tuple[float, float]:
        """
        Calculate mixed-layer top relative humidity variance and CO2 variance.
        Based on Neggers et. al 2006/7.
        """
        if wthetav > 0.0:
            q2_h = -(wqe + cc_qf) * dq * abl_height / (dz_h * wstar)
            top_CO22 = -(wCO2e + wCO2M) * dCO2 * abl_height / (dz_h * wstar)
        else:
            q2_h = 0.0
            top_CO22 = 0.0

        return q2_h, top_CO22

    @staticmethod
    def calculate_cloud_core_fraction(
        q: float, top_T: float, top_p: float, q2_h: float
    ) -> float:
        """
        Calculate cloud core fraction using the arctangent formulation.
        """
        if q2_h <= 0.0:
            return 0.0

        qsat = get_qsat(top_T, top_p)
        saturation_deficit = (q - qsat) / (q2_h**0.5)
        cc_frac = 0.5 + 0.36 * np.arctan(1.55 * saturation_deficit)
        cc_frac = max(0.0, cc_frac)
        return cc_frac

    @staticmethod
    def calculate_cloud_core_properties(
        cc_frac: float, wstar: float, q2_h: float
    ) -> tuple[float, float]:
        """
        Calculate and update cloud core mass flux and moisture flux.
        No return needed since we're updating self attributes directly.
        """
        cc_mf = cc_frac * wstar
        cc_qf = cc_mf * (q2_h**0.5) if q2_h > 0.0 else 0.0
        return cc_mf, cc_qf

    @staticmethod
    def calculate_co2_mass_flux(cc_mf: float, top_CO22: float, dCO2: float) -> float:
        """
        Calculate CO2 mass flux, only if mixed-layer top jump is negative.
        """
        if dCO2 < 0 and top_CO22 > 0.0:
            return cc_mf * (top_CO22**0.5)
        else:
            return 0.0

    def run(self, state: PyTree, const: PhysicalConstants):
        """
        State requirements
        ------------------
        - ``wthetav`` : float - Virtual potential temperature flux [K m/s]
        - ``wqe`` : float - Moisture flux at entrainment [kg/kg m/s]
        - ``dq`` : float - Moisture jump at mixed-layer top [kg/kg]
        - ``abl_height`` : float - Atmospheric boundary layer height [m]
        - ``dz_h`` : float - Layer thickness at mixed-layer top [m]
        - ``wstar`` : float - Convective velocity scale [m/s]
        - ``wCO2e`` : float - CO2 flux at entrainment [ppm m/s]
        - ``wCO2M`` : float - CO2 mass flux [ppm m/s]
        - ``dCO2`` : float - CO2 jump at mixed-layer top [ppm]
        - ``q`` : float - Specific humidity [kg/kg]
        - ``top_T`` : float - Temperature at mixed-layer top [K]
        - ``top_p`` : float - Pressure at mixed-layer top [Pa]

        Updates
        -------
        - ``cc_frac`` : float
            Cloud core fraction (0 to 1)
        - ``cc_mf`` : float
            Cloud core mass flux [m/s]
        - ``cc_qf`` : float
            Cloud core moisture flux [kg/kg/s]
        - ``q2_h`` : float
            Humidity variance at mixed-layer top
        - ``top_CO22`` : float
            CO2 variance at mixed-layer top
        - ``wCO2M`` : float
            CO2 mass flux [ppm m/s]
        """
        state.q2_h, state.top_CO22 = self.calculate_mixed_layer_variance(
            state.cc_qf,
            state.wthetav,
            state.wqe,
            state.dq,
            state.abl_height,
            state.dz_h,
            state.wstar,
            state.wCO2e,
            state.wCO2M,
            state.dCO2,
        )

        state.cc_frac = self.calculate_cloud_core_fraction(
            state.q,
            state.top_T,
            state.top_p,
            state.q2_h,
        )

        state.cc_mf, state.cc_qf = self.calculate_cloud_core_properties(
            state.cc_frac,
            state.wstar,
            state.q2_h,
        )

        state.wCO2M = self.calculate_co2_mass_flux(
            state.cc_mf,
            state.top_CO22,
            state.dCO2,
        )

        return state
