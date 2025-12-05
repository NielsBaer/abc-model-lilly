from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jaxtyping import Array, PyTree
from simple_pytree import Pytree

from ...utils import PhysicalConstants
from ..abstracts import AbstractSurfaceLayerModel


@dataclass
class MinimalSurfaceLayerState(Pytree):
    """Minimal surface layer model initial state."""

    ustar: Array
    """Surface friction velocity [m/s]."""
    uw: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Zonal surface momentum flux [m2 s-2]."""
    vw: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Meridional surface momentum flux [m2 s-2]."""


# alias
MinimalSurfaceLayerInitConds = MinimalSurfaceLayerState


class MinimalSurfaceLayerModel(AbstractSurfaceLayerModel):
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

    def run(self, state: PyTree, const: PhysicalConstants):
        """Calculate momentum fluxes from wind components and friction velocity."""
        uw, vw = self.calculate_momentum_fluxes(state.u, state.v, state.ustar)
        return replace(state, uw=uw, vw=vw)

    @staticmethod
    def compute_ra(state: PyTree) -> Array:
        """Calculate aerodynamic resistance from wind speed and friction velocity."""
        ueff = jnp.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        return ueff / jnp.maximum(1.0e-3, state.ustar) ** 2.0
