import jax
import jax.numpy as jnp
from jax import Array

from ...utils import PhysicalConstants, compute_qsat
from ..abstracts import AbstractCoupledState, AbstractMixedLayerModel


class AbstractStandardStatsModel(AbstractMixedLayerModel):
    """Abstract base class for mixed layer models with standard meteorological statistics.

    Provides a common calculation method for virtual temperature, mixed-layer top
    properties, and lifting condensation level determination.
    """

    def statistics(self, state: AbstractCoupledState, t: int, const: PhysicalConstants):
        """Compute standard meteorological statistics and diagnostics."""
        mixed_state = state.atmos.mixed
        thetav = self.compute_thetav(mixed_state.theta, mixed_state.q)
        wthetav = self.compute_wthetav(
            mixed_state.wtheta, mixed_state.theta, state.land.wq
        )
        deltathetav = self.compute_deltathetav(
            mixed_state.theta,
            mixed_state.deltatheta,
            mixed_state.q,
            mixed_state.dq,
        )
        top_p = self.compute_top_p(
            mixed_state.surf_pressure, const.rho, const.g, mixed_state.h_abl
        )
        top_T = self.compute_top_T(
            mixed_state.theta, const.g, const.cp, mixed_state.h_abl
        )
        top_rh = self.compute_top_rh(mixed_state.q, top_T, top_p)
        lcl = self.compute_lcl(
            mixed_state.h_abl,
            mixed_state.lcl,
            mixed_state.surf_pressure,
            mixed_state.theta,
            mixed_state.q,
            t,
            const,
        )
        ml_state = state.atmos.mixed.replace(
            thetav=thetav,
            wthetav=wthetav,
            deltathetav=deltathetav,
            top_p=top_p,
            top_T=top_T,
            top_rh=top_rh,
            lcl=lcl,
        )
        return ml_state

    def compute_thetav(self, theta: Array, q: Array) -> Array:
        """Computes the virtual potential temperature as

        .. math::
            \\theta_v = \\theta \\left(1 + 0.61\\, q\\right).
        """
        return theta * (1.0 + 0.61 * q)

    def compute_wthetav(self, wtheta: Array, theta: Array, wq: Array) -> Array:
        """Computes the virtual potential temperature flux as

        .. math::
            \\overline{w'\\theta_v'} = \\overline{w'\\theta'} + 0.61\\,\\theta\\,\\overline{w'q'}.
        """
        return wtheta + 0.61 * theta * wq

    def compute_deltathetav(
        self,
        theta: Array,
        deltatheta: Array,
        q: Array,
        dq: Array,
    ) -> Array:
        """Computes the virtual potential temperature jump as

        .. math::
            \\Delta\\theta_v = (\\theta + \\Delta\\theta)\\left(1 + 0.61\\,(q + \\Delta q)\\right)
            - \\theta\\left(1 + 0.61\\,q\\right)
        """
        return (theta + deltatheta) * (1.0 + 0.61 * (q + dq)) - theta * (1.0 + 0.61 * q)

    def compute_top_p(
        self, surf_pressure: Array, rho: float, g: float, h_abl: Array
    ) -> Array:
        """Computes the pressure at the top of the mixed layer as

        .. math::
            p_{top} = p_{surf} - \\rho\\, g\\, h.
        """
        return surf_pressure - rho * g * h_abl

    def compute_top_T(self, theta: Array, g: float, cp: float, h_abl: Array) -> Array:
        """Computes the temperature at the top of the mixed layer as

        .. math::
            T_{top} = \\theta - \\frac{g}{c_p}\\, h.
        """
        return theta - (g / cp) * h_abl

    def compute_top_rh(self, q: Array, top_T: Array, top_p: Array) -> Array:
        """Computes the relative humidity at the mixed-layer top as

        .. math::
            \\mathrm{RH}_{top} = \\frac{q}{q_{sat}(T_{top},\\,p_{top})}.
        """
        return q / compute_qsat(top_T, top_p)

    def compute_lcl(
        self,
        h_abl: Array,
        lcl: Array,
        surf_pressure: Array,
        theta: Array,
        q: Array,
        t: int,
        const: PhysicalConstants,
    ) -> Array:
        """Compute the lifting condensation level (LCL).

        The LCL is found iteratively by finding the height where the relative humidity is 100%.
        """
        # find lifting condensation level iteratively using JAX
        # initialize lcl and rhlcl based on timestep
        initial_lcl = jnp.where(t == 0, h_abl, lcl)
        initial_rhlcl = jnp.where(t == 0, 0.5, 0.9998)

        def lcl_iteration_body(carry):
            lcl, rhlcl, iteration = carry

            # update lcl based on current relative humidity
            lcl_adjustment = (1.0 - rhlcl) * 1000.0
            new_lcl = lcl + lcl_adjustment

            # calculate new relative humidity at updated lcl
            p_lcl = surf_pressure - const.rho * const.g * new_lcl
            temp_lcl = theta - const.g / const.cp * new_lcl
            new_rhlcl = q / compute_qsat(temp_lcl, p_lcl)

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
