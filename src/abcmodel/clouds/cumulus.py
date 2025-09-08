import numpy as np

from ..models import (
    AbstractCloudModel,
    AbstractDiagnostics,
    AbstractInitConds,
    AbstractMixedLayerModel,
    AbstractParams,
)
from ..utils import get_qsat


class StandardCumulusParams(AbstractParams["StandardCumulusModel"]):
    """Standard cumulus model parameters."""

    def __init__(self):
        pass


class StandardCumulusInitConds(AbstractInitConds["StandardCumulusModel"]):
    """Standard cumulus model initial conditions."""

    def __init__(self):
        self.cc_frac = 0.0
        self.cc_mf = 0.0
        self.cc_qf = 0.0


class StandardCumulusDiagnostics(AbstractDiagnostics["StandardCumulusModel"]):
    """Class for standard cumulus model diagnostics.

    Variables
    ---------
    - ``cc_frac``: cloud core fraction [-], range 0 to 1.
    - ``cc_mf``: cloud core mass flux [m/s].
    - ``cc_qf``: cloud core moisture flux [kg/kg/s].
    """

    def post_init(self, tsteps: int):
        self.cc_frac = np.zeros(tsteps)
        self.cc_mf = np.zeros(tsteps)
        self.cc_qf = np.zeros(tsteps)

    def store(self, t: int, model: "StandardCumulusModel"):
        self.cc_frac[t] = model.cc_frac
        self.cc_mf[t] = model.cc_mf
        self.cc_qf[t] = model.cc_qf


class StandardCumulusModel(AbstractCloudModel):
    """
    Standard cumulus cloud model based on Neggers et al. (2006/7).

    This model calculates shallow cumulus convection properties using a variance-based
    approach to determine cloud core fraction and associated mass fluxes. The model
    characterizes turbulent fluctuations in the mixed layer that lead to cloud formation.
    It quantifies the variance of humidity and CO2 at the mixed-layer top and uses this
    to determine what fraction reaches saturation.

    Processes
    ---------
    1. Calculates turbulent variance for humidity and CO2 at the mixed-layer top
    using entrainment fluxes and convective scaling (Neggers et al. 2006/7).
    2. Determines fraction of mixed-layer top that becomes saturated using
    arctangent formulation based on saturation deficit.
    3. Calculates mass flux and moisture flux through cloud cores.
    4. Computes CO2 transport only when CO2 decreases with height.
    """

    def __init__(
        self,
        params: StandardCumulusParams,
        init_conds: StandardCumulusInitConds,
        diagnostics: AbstractDiagnostics = StandardCumulusDiagnostics(),
    ):
        self.cc_frac = init_conds.cc_frac
        self.cc_mf = init_conds.cc_mf
        self.cc_qf = init_conds.cc_qf
        self.diagnostics = diagnostics

    def calculate_mixed_layer_variance(
        self,
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
            q2_h = -(wqe + self.cc_qf) * dq * abl_height / (dz_h * wstar)
            top_CO22 = -(wCO2e + wCO2M) * dCO2 * abl_height / (dz_h * wstar)
        else:
            q2_h = 0.0
            top_CO22 = 0.0

        return q2_h, top_CO22

    def calculate_cloud_core_fraction(
        self,
        q: float,
        top_T: float,
        top_p: float,
        q2_h: float,
    ):
        """
        Calculate cloud core fraction using the arctangent formulation.
        """
        if q2_h <= 0.0:
            self.cc_frac = 0.0
            return None

        qsat = get_qsat(top_T, top_p)
        saturation_deficit = (q - qsat) / (q2_h**0.5)
        cc_frac = 0.5 + 0.36 * np.arctan(1.55 * saturation_deficit)
        self.cc_frac = max(0.0, cc_frac)

    def calculate_cloud_core_properties(self, wstar: float, q2_h: float):
        """
        Calculate and update cloud core mass flux and moisture flux.
        No return needed since we're updating self attributes directly.
        """
        self.cc_mf = self.cc_frac * wstar
        self.cc_qf = self.cc_mf * (q2_h**0.5) if q2_h > 0.0 else 0.0

    def calculate_co2_mass_flux(self, top_CO22: float, dCO2: float) -> float:
        """
        Calculate CO2 mass flux, only if mixed-layer top jump is negative.
        """
        if dCO2 < 0 and top_CO22 > 0.0:
            return self.cc_mf * (top_CO22**0.5)
        else:
            return 0.0

    def run(self, mixed_layer: AbstractMixedLayerModel):
        """
        Parameters
        ----------
        ``mixed_layer`` : AbstractMixedLayerModel
        Mixed-layer model containing required thermodynamic and flux variables.

        **Required attributes:**

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
        **self attributes modified:**

        - ``cc_frac`` : float
            Cloud core fraction (0 to 1)
        - ``cc_mf`` : float
            Cloud core mass flux [m/s]
        - ``cc_qf`` : float
            Cloud core moisture flux [kg/kg/s]

        **mixed_layer attributes modified:**

        - ``q2_h`` : float
            Humidity variance at mixed-layer top
        - ``top_CO22`` : float
            CO2 variance at mixed-layer top
        - ``wCO2M`` : float
            CO2 mass flux [ppm m/s]
        """
        mixed_layer.q2_h, mixed_layer.top_CO22 = self.calculate_mixed_layer_variance(
            mixed_layer.wthetav,
            mixed_layer.wqe,
            mixed_layer.dq,
            mixed_layer.abl_height,
            mixed_layer.dz_h,
            mixed_layer.wstar,
            mixed_layer.wCO2e,
            mixed_layer.wCO2M,
            mixed_layer.dCO2,
        )

        self.calculate_cloud_core_fraction(
            mixed_layer.q,
            mixed_layer.top_T,
            mixed_layer.top_p,
            mixed_layer.q2_h,
        )

        self.calculate_cloud_core_properties(mixed_layer.wstar, mixed_layer.q2_h)

        mixed_layer.wCO2M = self.calculate_co2_mass_flux(
            mixed_layer.top_CO22, mixed_layer.dCO2
        )
