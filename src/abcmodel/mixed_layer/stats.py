import jax
import jax.numpy as jnp
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

    """Surface pressure, which is actually not updated (not a state), it's only here for simplicity [Pa]."""
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

        # find lifting condensation level iteratively using JAX
        # initialize lcl and rhlcl based on timestep
        initial_lcl = jnp.where(t == 0, state.abl_height, state.lcl)
        initial_rhlcl = jnp.where(t == 0, 0.5, 0.9998)

        def lcl_iteration_body(carry):
            lcl, rhlcl, iteration = carry

            # update lcl based on current relative humidity
            lcl_adjustment = (1.0 - rhlcl) * 1000.0
            new_lcl = lcl + lcl_adjustment

            # calculate new relative humidity at updated lcl
            p_lcl = state.surf_pressure - const.rho * const.g * new_lcl
            temp_lcl = state.theta - const.g / const.cp * new_lcl
            new_rhlcl = state.q / get_qsat(temp_lcl, p_lcl)

            return new_lcl, new_rhlcl, iteration + 1

        def lcl_iteration_cond(carry):
            lcl, rhlcl, iteration = carry
            # continue if not converged and under max iterations
            not_converged = (rhlcl <= 0.9999) | (rhlcl >= 1.0001)
            under_max_iter = iteration < 30  # itmax = 30
            return not_converged & under_max_iter

        final_lcl, final_rhlcl, final_iter = jax.lax.while_loop(
            lcl_iteration_cond, lcl_iteration_body, (initial_lcl, initial_rhlcl, 0)
        )

        state.lcl = final_lcl

        return state
