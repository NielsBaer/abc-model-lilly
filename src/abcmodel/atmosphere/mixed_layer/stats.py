import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...utils import PhysicalConstants, compute_qsat
from ..abstracts import (
    AbstractMixedLayerModel,
)


class AbstractStandardStatsModel(AbstractMixedLayerModel):
    """Abstract base class for mixed layer models with standard meteorological statistics.

    Provides a common calculation method for virtual temperature, mixed-layer top
    properties, and lifting condensation level determination.
    """

    """Surface pressure, which is actually not updated (not a state), it's only here for simplicity [Pa]."""
    surf_pressure: float

    def statistics(self, state: PyTree, t: int, const: PhysicalConstants):
        """Compute standard meteorological statistics and diagnostics."""
        state = self.compute_virtual_temperatures(state)
        state = self.compute_mixed_layer_top_properties(state, const)
        state.lcl = self.compute_lifting_condensation_level(state, t, const)
        return state

    def compute_virtual_temperatures(self, state: PyTree) -> PyTree:
        """Compute virtual temperatures and fluxes.

        Notes:
            The virtual potential temperature :math:`\\theta_v` is given by

            .. math::
                \\theta_v = \\theta (1 + 0.61 q)

            The virtual heat flux :math:`\\overline{w'\\theta_v'}` is

            .. math::
                \\overline{w'\\theta_v'} = \\overline{w'\\theta'} + 0.61 \\theta \\overline{w'q'}

            The virtual potential temperature jump :math:`\\Delta \\theta_v` is

            .. math::
                \\Delta \\theta_v = (\\theta + \\Delta \\theta)(1 + 0.61(q + \\Delta q)) - \\theta(1 + 0.61q)
        """
        # calculate virtual temperatures
        state.thetav = state.theta + 0.61 * state.theta * state.q
        state.wthetav = state.wtheta + 0.61 * state.theta * state.wq
        state.deltathetav = (state.theta + state.deltatheta) * (
            1.0 + 0.61 * (state.q + state.dq)
        ) - state.theta * (1.0 + 0.61 * state.q)
        return state

    def compute_mixed_layer_top_properties(
        self, state: PyTree, const: PhysicalConstants
    ) -> PyTree:
        """Compute properties at the top of the mixed layer.

        Notes:
            The pressure at the top of the mixed layer :math:`p_{top}` is

            .. math::
                p_{top} = p_{surf} - \\rho g h

            The temperature at the top of the mixed layer :math:`T_{top}` is

            .. math::
                T_{top} = \\theta - \\frac{g}{c_p} h
        """
        # mixed-layer top properties
        state.top_p = state.surf_pressure - const.rho * const.g * state.h_abl
        state.top_T = state.theta - const.g / const.cp * state.h_abl
        state.top_rh = state.q / compute_qsat(state.top_T, state.top_p)
        return state

    def compute_lifting_condensation_level(
        self, state: PyTree, t: int, const: PhysicalConstants
    ) -> Array:
        """Compute the lifting condensation level (LCL).

        Notes:
            The LCL is found iteratively by finding the height where the relative humidity is 100%.
        """
        # find lifting condensation level iteratively using JAX
        # initialize lcl and rhlcl based on timestep
        initial_lcl = jnp.where(t == 0, state.h_abl, state.lcl)
        initial_rhlcl = jnp.where(t == 0, 0.5, 0.9998)

        def lcl_iteration_body(carry):
            lcl, rhlcl, iteration = carry

            # update lcl based on current relative humidity
            lcl_adjustment = (1.0 - rhlcl) * 1000.0
            new_lcl = lcl + lcl_adjustment

            # calculate new relative humidity at updated lcl
            p_lcl = state.surf_pressure - const.rho * const.g * new_lcl
            temp_lcl = state.theta - const.g / const.cp * new_lcl
            new_rhlcl = state.q / compute_qsat(temp_lcl, p_lcl)

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

        return final_lcl
