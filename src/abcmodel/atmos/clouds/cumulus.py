from dataclasses import dataclass, field, replace

import jax.numpy as jnp

from ...abstracts import AbstractCoupledState
from ...utils import Array, compute_qsat
from ..abstracts import AbstractCloudModel, AbstractCloudState


@dataclass
class CumulusState(AbstractCloudState):
    """Standard cumulus state."""

    cc_frac: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core fraction [-], range 0 to 1."""
    cc_mf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core mass flux [m/s]."""
    cc_qf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core moisture flux [kg/kg/s]."""
    cl_trans: Array = field(default_factory=lambda: jnp.array(1.0))
    """Cloud layer transmittance [-], range 0 to 1."""
    q2_h: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Humidity variance at mixed-layer top [kg²/kg²]."""
    top_CO22: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """CO2 variance at mixed-layer top [ppm²]."""
    wCO2M: Array = field(default_factory=lambda: jnp.array(0.0))
    """CO2 mass flux [mgC/m²/s]."""


class CumulusModel(AbstractCloudModel[CumulusState]):
    """Standard cumulus cloud model based on Neggers et al. (2006/7).

    This model calculates shallow cumulus convection properties using a variance-based
    approach to determine cloud core fraction and associated mass fluxes. The model
    characterizes turbulent fluctuations in the mixed layer that lead to cloud formation.
    It quantifies the variance of humidity and CO2 at the mixed-layer top and uses this
    to determine what fraction reaches saturation.

    Args:
        tcc_cc: ratio of total cloud cove to core cloud fraction [-], greater or equal to 1.
        tcc_trans: mean transmittance of cloud cover [-], range 0 to 1.
    """

    def __init__(self, tcc_cc: float = 2.0, tcc_trans: float = 0.6):
        self.tcc_cc = tcc_cc
        self.tcc_trans = tcc_trans

    def init_state(self) -> CumulusState:
        """Initialize the model state.

        Returns:
            The initial cloud state.
        """
        return CumulusState()

    def run(self, state: AbstractCoupledState) -> CumulusState:
        """Run the model.

        Args:
            state:

        Returns:
            The updated cloud state.

        """
        atmos = state.atmos
        cloud_state = atmos.clouds

        q2_h = self.compute_q2_h(
            cloud_state.cc_qf,
            atmos.wthetav,
            atmos.wqe,
            atmos.dq,
            atmos.h_abl,
            atmos.dz_h,
            atmos.wstar,
        )
        top_CO22 = self.compute_top_CO22(
            atmos.wthetav,
            atmos.h_abl,
            atmos.dz_h,
            atmos.wstar,
            atmos.wCO2e,
            cloud_state.wCO2M,
            atmos.deltaCO2,
        )
        cc_frac = self.compute_cc_frac(atmos.q, atmos.top_T, atmos.top_p, q2_h)
        cc_mf = self.compute_cc_mf(cc_frac, atmos.wstar)
        cc_qf = self.compute_cc_qf(cc_mf, q2_h)
        wCO2M = self.compute_wCO2M(cc_mf, top_CO22, atmos.deltaCO2)
        cl_trans = self.compute_cl_trans(cc_frac)

        return replace(
            cloud_state,
            q2_h=q2_h,
            top_CO22=top_CO22,
            cc_frac=cc_frac,
            cc_mf=cc_mf,
            cc_qf=cc_qf,
            wCO2M=wCO2M,
            cl_trans=cl_trans,
        )

    def compute_q2_h(
        self,
        cc_qf: Array,
        wthetav: Array,
        wqe: Array,
        dq: Array,
        h_abl: Array,
        dz_h: Array,
        wstar: Array,
    ) -> Array:
        """Compute mixed-layer top relative humidity variance.

        Notes:
            Based on Neggers et al. (2006/7). The humidity variance :math:`\\sigma_{q,h}^2` is

            .. math::
                \\sigma_{q,h}^2 = -\\frac{(\\overline{w'q'}_e + \\overline{w'q'}_{cc}) \\Delta q h}{\\delta z_h w_*}
        """
        return jnp.where(
            wthetav > 0.0, -(wqe + cc_qf) * dq * h_abl / (dz_h * wstar), 0.0
        )

    def compute_top_CO22(
        self,
        wthetav: Array,
        h_abl: Array,
        dz_h: Array,
        wstar: Array,
        wCO2e: Array,
        wCO2M: Array,
        deltaCO2: Array,
    ) -> Array:
        """Compute mixed-layer top CO2 variance.

        Notes:
            Based on Neggers et al. (2006/7). The CO2 variance :math:`\\sigma_{CO2,h}^2` is

            .. math::
                \\sigma_{CO2,h}^2 = -\\frac{(\\overline{w'CO_2'}_e + \\overline{w'CO_2'}_{M}) \\Delta CO_2 h}{\\delta z_h w_*}
        """
        return jnp.where(
            wthetav > 0.0, -(wCO2e + wCO2M) * deltaCO2 * h_abl / (dz_h * wstar), 0.0
        )

    def compute_cc_frac(
        self, q: Array, top_T: Array, top_p: Array, q2_h: Array
    ) -> Array:
        """Compute cloud core fraction using the arctangent formulation.

        Notes:
            The cloud core fraction :math:`a_{cc}` is given by

            .. math::
                a_{cc} = 0.5 + 0.36 \\arctan\\left( 1.55 \\frac{q - q_{sat}(T_{top}, p_{top})}{\\sigma_{q,h}} \\right)
        """
        cc_frac = jnp.where(
            q2_h <= 0.0,
            0.0,
            jnp.maximum(
                0.0,
                0.5
                + 0.36
                * jnp.arctan(1.55 * (q - compute_qsat(top_T, top_p)) / (q2_h**0.5)),
            ),
        )
        return cc_frac

    def compute_cc_mf(self, cc_frac: Array, wstar: Array) -> Array:
        """Compute cloud core mass flux.

        Notes:
            The cloud core mass flux :math:`M_{cc}` is

            .. math::
                M_{cc} = a_{cc} w_*
        """
        return cc_frac * wstar

    def compute_cc_qf(self, cc_mf: Array, q2_h: Array) -> Array:
        """Compute cloud core moisture flux.

        Notes:
            The cloud core moisture flux :math:`\\overline{w'q'}_{cc}` is

            .. math::
                \\overline{w'q'}_{cc} = M_{cc} \\sigma_{q,h}
        """
        return jnp.where(q2_h > 0.0, cc_mf * (q2_h**0.5), 0.0)

    def compute_wCO2M(self, cc_mf: Array, top_CO22: Array, deltaCO2: Array) -> Array:
        """Compute CO2 mass flux.

        Notes:
            The CO2 mass flux :math:`\\overline{w'CO_2'}_{M}` is

            .. math::
                \\overline{w'CO_2'}_{M} = M_{cc} \\sigma_{CO2,h}

            This is only computed if the mixed-layer top jump is negative.
        """
        flux_value = cc_mf * (top_CO22**0.5)
        condition = (deltaCO2 < 0) & (top_CO22 > 0.0)
        return jnp.where(condition, flux_value, 0.0)

    def compute_cl_trans(self, cc_frac: Array) -> Array:
        """Compute cloud layer transmittance, with maximum total cloud cover equal to 1.

        Notes:
            The cloud layer transmittance :math:`\\tau_{cl}` is

            .. math::
                \\tau_{cl} = 1 - \\text{TCC} (1 - \\tau_{cloud})

            where :math:`\\text{TCC} = \\min(a_{cc} \\cdot \\text{ratio}, 1)` is the total cloud cover.
        """
        tcc = jnp.minimum(cc_frac * self.tcc_cc, 1.0)
        return 1.0 - tcc * (1.0 - self.tcc_trans)
