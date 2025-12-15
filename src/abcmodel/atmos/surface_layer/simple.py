from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ...utils import PhysicalConstants
from ..abstracts import (
    AbstractCoupledState,
    AbstractSurfaceLayerModel,
    AbstractSurfaceLayerState,
)


@dataclass
class SimpleSurfaceLayerState(AbstractSurfaceLayerState):
    """Minimal surface layer model initial state."""

    ustar: Array
    """Surface friction velocity [m/s]."""
    uw: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Zonal surface momentum flux [m2 s-2]."""
    vw: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Meridional surface momentum flux [m2 s-2]."""
    ra: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Aerodynamic resistance [s/m]."""


SimpleSurfaceLayerInitConds = SimpleSurfaceLayerState


class SimpleSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Minimal surface layer model with constant friction velocity."""

    def __init__(self):
        pass

    @staticmethod
    def calculate_momentum_fluxes(
        u: Array, v: Array, ustar: Array
    ) -> tuple[Array, Array]:
        """Calculate momentum fluxes from wind components and friction velocity."""
        uw = jnp.where(
            u == 0.0,
            0.0,
            -jnp.sign(u) * (ustar**4.0 / (v**2.0 / u**2.0 + 1.0)) ** (0.5),
        )
        vw = jnp.where(
            v == 0.0,
            0.0,
            -jnp.sign(v) * (ustar**4.0 / (u**2.0 / v**2.0 + 1.0)) ** (0.5),
        )
        return uw, vw

    def run(self, state: AbstractCoupledState, const: PhysicalConstants):
        """Run the model.

        Args:
            state: CoupledState.
            const: PhysicalConstants.

        Returns:
            Updated MinimalSurfaceLayerState.
        """
        sl_state = state.atmos.surface_layer
        ml_state = state.atmos.mixed_layer
        uw, vw = self.calculate_momentum_fluxes(ml_state.u, ml_state.v, sl_state.ustar)
        ra = self.compute_ra(ml_state.u, ml_state.v, ml_state.wstar, sl_state.ustar)
        return replace(sl_state, uw=uw, vw=vw, ra=ra)

    @staticmethod
    def compute_ra(u: Array, v: Array, wstar: Array, ustar: Array) -> Array:
        """Calculate aerodynamic resistance from wind speed and friction velocity."""
        ueff = jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        return ueff / jnp.maximum(1.0e-3, ustar) ** 2.0
