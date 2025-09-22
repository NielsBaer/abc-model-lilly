from jaxtyping import PyTree

from ..models import (
    AbstractMixedLayerModel,
)
from ..utils import PhysicalConstants, get_qsat


class AbstractStandardStatsModel(AbstractMixedLayerModel):
    """Abstract base class for mixed layer models with standard meteorological statistics.

    Provides a common calculation method for virtual temperature, mixed-layer top
    properties, and lifting condensation level determination.
    """

    surf_pressure: float

    def statistics(self, state: PyTree, t: int, const: PhysicalConstants):
        """Calculate standard meteorological statistics and diagnostics."""
        # calculate virtual temperatures
        state.thetav = state.theta + 0.61 * state.theta * state.q
        state.wthetav = state.wtheta + 0.61 * state.theta * state.wq
        state.dthetav = (state.theta + state.dtheta) * (
            1.0 + 0.61 * (state.q + state.dq)
        ) - state.theta * (1.0 + 0.61 * state.q)

        # mixed-layer top properties
        state.top_p = state.surf_pressure - const.rho * const.g * state.abl_height
        state.top_T = state.theta - const.g / const.cp * state.abl_height
        state.top_rh = state.q / get_qsat(state.top_T, state.top_p)

        # find lifting condensation level iteratively
        if t == 0:
            state.lcl = state.abl_height
            rhlcl = 0.5
        else:
            rhlcl = 0.9998

        itmax = 30
        it = 0
        # limamau: this can be replace by a jax.lax.while_loop
        while ((rhlcl <= 0.9999) or (rhlcl >= 1.0001)) and it < itmax:
            state.lcl += (1.0 - rhlcl) * 1000.0
            p_lcl = state.surf_pressure - const.rho * const.g * state.lcl
            temp_lcl = state.theta - const.g / const.cp * state.lcl
            rhlcl = state.q / get_qsat(temp_lcl, p_lcl)
            it += 1

        if it == itmax:
            print("LCL calculation not converged!!")
            print("RHlcl = %f, zlcl=%f" % (rhlcl, state.lcl))

        return state
