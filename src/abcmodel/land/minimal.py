from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ..abstracts import AbstractCoupledState, AbstractLandModel, AbstractLandState
from ..utils import PhysicalConstants, compute_esat, compute_qsat


@dataclass
class MinimalLandSurfaceState(AbstractLandState):
    """Minimal land surface model state."""

    alpha: Array
    """surface albedo [-], range 0 to 1."""
    surf_temp: Array
    """Surface temperature [K]."""
    rs: Array
    """Surface resistance [s m-1]."""
    wg: Array = field(default_factory=lambda: jnp.array(0.0))
    """No moisture content in the root zone [m3 m-3]."""
    wl: Array = field(default_factory=lambda: jnp.array(0.0))
    """No water content in the canopy [m]."""

    # the following variables are assigned during warmup/timestep
    esat: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Saturation vapor pressure [Pa]."""
    qsat: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Vapor pressure [Pa]."""
    qsatsurf: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Kinematic heat flux [K m/s]."""
    wq: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


# alias
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
        land_state = state.land
        ml_state = state.atmosphere.mixed_layer
        esat = compute_esat(ml_state.theta)
        qsat = compute_qsat(ml_state.theta, ml_state.surf_pressure)
        dqsatdT = self.compute_dqsatdT(esat, ml_state.theta, ml_state.surf_pressure)
        e = self.compute_e(ml_state.q, ml_state.surf_pressure)
        return replace(land_state, esat=esat, qsat=qsat, dqsatdT=dqsatdT, e=e)

    def compute_dqsatdT(self, esat: Array, theta: float, surf_pressure: float) -> Array:
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
        desatdT = esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return q * surf_pressure / 0.622

    def integrate(
        self, state: MinimalLandSurfaceState, dt: float
    ) -> MinimalLandSurfaceState:
        """Integrate the model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        return state
