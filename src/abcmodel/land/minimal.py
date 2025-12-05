from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..abstracts import AbstractCoupledState, AbstractLandModel, AbstractLandState
from ..utils import PhysicalConstants, compute_esat, compute_qsat


@jax.tree_util.register_pytree_node_class
@dataclass
class MinimalLandSurfaceState(AbstractLandState):
    """Minimal land surface model state."""

    alpha: float
    """surface albedo [-], range 0 to 1."""
    surf_temp: float
    """Surface temperature [K]."""
    rs: float
    """Surface resistance [s m-1]."""
    wg: float = 0.0
    """No moisture content in the root zone [m3 m-3]."""
    wl: float = 0.0
    """No water content in the canopy [m]."""

    # the following variables are assigned during warmup/timestep
    ra: float = jnp.nan
    """Aerodynamic resistance [s/m]."""
    esat: float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    qsat: float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: float = jnp.nan
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: float = jnp.nan
    """Vapor pressure [Pa]."""
    qsatsurf: float = jnp.nan
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: float = jnp.nan
    """Kinematic heat flux [K m/s]."""
    wq: float = jnp.nan
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: float = jnp.nan
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""

    def tree_flatten(self):
        return (
            self.alpha, self.surf_temp, self.rs, self.wg, self.wl,
            self.ra, self.esat, self.qsat, self.dqsatdT, self.e, self.qsatsurf
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# Alias for backward compatibility
MinimalLandSurfaceInitConds = MinimalLandSurfaceState


class MinimalLandSurfaceModel(AbstractLandModel):
    """Minimal land surface model with fixed surface properties."""

    def __init__(self):
        self.d1 = 0.0

    def run(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> MinimalLandSurfaceState:
        """Run the model.

        Args:
            state: CoupledState.
            const: the physical constants object.

        Returns:
            The updated land state object.
        """
        # Access components
        land_state = state.land
        ml_state = state.atmosphere.mixed_layer
        sl_state = state.atmosphere.surface_layer

        # (1) compute aerodynamic resistance from state
        ueff = jnp.sqrt(ml_state.u**2.0 + ml_state.v**2.0 + ml_state.wstar**2.0)
        land_state.ra = ueff / jnp.maximum(1.0e-3, sl_state.ustar) ** 2.0

        # (2) calculate essential thermodynamic variables
        land_state.esat = compute_esat(ml_state.theta)
        land_state.qsat = compute_qsat(ml_state.theta, ml_state.surf_pressure)
        land_state.dqsatdT = self.compute_dqsatdT(land_state, ml_state.theta, ml_state.surf_pressure)
        land_state.e = self.compute_e(ml_state.q, ml_state.surf_pressure)

        return land_state

    def compute_dqsatdT(self, state: MinimalLandSurfaceState, theta: float, surf_pressure: float) -> Array:
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :meth:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :meth:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = state.esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: float, surf_pressure: float) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return q * surf_pressure / 0.622

    def integrate(self, state: MinimalLandSurfaceState, dt: float) -> MinimalLandSurfaceState:
        """Integrate the model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        return state
