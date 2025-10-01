from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

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
    - ``cl_trans``: cloud layer transmittance [-], range 0 to 1.
    """

    cc_frac: float = 0.0
    cc_mf: float = 0.0
    cc_qf: float = 0.0
    cl_trans: float = 1.0


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
    - ``tcc_cc``: ratio of total cloud cove to core cloud fraction [-], greater or equal to 1.
    - ``tcc_trans``: mean transmittance of cloud cover [-], range 0 to 1.

    Processes
    ---------
    1. Calculates turbulent variance for humidity and CO2 at the mixed-layer top
    using entrainment fluxes and convective scaling (Neggers et al. 2006/7).
    2. Determines fraction of mixed-layer top that becomes saturated using
    arctangent formulation based on saturation deficit.
    3. Calculates mass flux and moisture flux through cloud cores.
    4. Computes CO2 transport only when CO2 decreases with height.
    """

    def __init__(self, 
                 tcc_cc: float = 2.0,
                 tcc_trans: float = 0.6,
                 ):
        self.tcc_cc = tcc_cc
        self.tcc_trans = tcc_trans

    @staticmethod
    def calculate_mixed_layer_variance(
        cc_qf: Array,
        wthetav: Array,
        wqe: Array,
        dq: Array,
        abl_height: Array,
        dz_h: Array,
        wstar: Array,
        wCO2e: Array,
        wCO2M: Array,
        dCO2: Array,
    ) -> tuple[Array, Array]:
        """
        Calculate mixed-layer top relative humidity variance and CO2 variance.
        Based on Neggers et. al 2006/7.
        """
        q2_h = jnp.where(
            wthetav > 0.0, -(wqe + cc_qf) * dq * abl_height / (dz_h * wstar), 0.0
        )
        top_CO22 = jnp.where(
            wthetav > 0.0, -(wCO2e + wCO2M) * dCO2 * abl_height / (dz_h * wstar), 0.0
        )
        return q2_h, top_CO22

    @staticmethod
    def calculate_cloud_core_fraction(
        q: Array, top_T: Array, top_p: Array, q2_h: Array
    ) -> Array:
        """
        Calculate cloud core fraction using the arctangent formulation.
        """
        cc_frac = jnp.where(
            q2_h <= 0.0,
            0.0,
            jnp.maximum(
                0.0,
                0.5
                + 0.36 * jnp.arctan(1.55 * (q - get_qsat(top_T, top_p)) / (q2_h**0.5)),
            ),
        )
        return cc_frac

    @staticmethod
    def calculate_cloud_core_properties(
        cc_frac: Array, wstar: Array, q2_h: Array
    ) -> tuple[Array, Array]:
        """
        Calculate and update cloud core mass flux and moisture flux.
        No return needed since we're updating self attributes directly.
        """
        cc_mf = cc_frac * wstar
        cc_qf = jnp.where(q2_h > 0.0, cc_mf * (q2_h**0.5), 0.0)
        return cc_mf, cc_qf

    @staticmethod
    def calculate_co2_mass_flux(cc_mf: Array, top_CO22: Array, dCO2: Array) -> Array:
        """
        Calculate CO2 mass flux, only if mixed-layer top jump is negative.
        """
        # flux value
        flux_value = cc_mf * (top_CO22**0.5)

        # conditions: dCO2 < 0 AND top_CO22 > 0.0
        condition = (dCO2 < 0) & (top_CO22 > 0.0)

        return jnp.where(condition, flux_value, 0.0)

    def calculate_cloud_layer_transmittance(self, cc_frac: Array) -> Array:
        """Calculate cloud layer transmittance, with maximum total cloud cover equal to 1"""
        # get total cloud cover
        tcc = jnp.minimum(cc_frac*self.tcc_cc, 1.0)
        # return cloud layer transmittance
        return 1.0 - tcc * (1.0 - self.tcc_trans)



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
        - ``cl_trans``: float
            cloud layer transmissivity [-]
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
        state.cl_trans = self.calculate_cloud_layer_transmittance(
            state.cc_frac,
        )

        return state
