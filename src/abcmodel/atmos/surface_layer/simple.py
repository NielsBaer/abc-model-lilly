from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState, LandT, RadT
from ..abstracts import (
    AbstractSurfaceLayerModel,
    AbstractSurfaceLayerState,
    CloudT,
    MixedT,
)
from ..dayonly import DayOnlyAtmosphereState


@dataclass
class SimpleState(AbstractSurfaceLayerState):
    """Minimal surface layer model initial state."""

    ustar: Array
    """Surface friction velocity [m/s]."""
    uw: Array = field(default_factory=lambda: jnp.array(0.0))
    """Zonal surface momentum flux [m2 s-2]."""
    vw: Array = field(default_factory=lambda: jnp.array(0.0))
    """Meridional surface momentum flux [m2 s-2]."""
    ra: Array = field(default_factory=lambda: jnp.array(0.0))
    """Aerodynamic resistance [s/m]."""


# limamau: maybe these type variables could be abstracts...
StateAlias = AbstractCoupledState[
    RadT,
    LandT,
    DayOnlyAtmosphereState[
        SimpleState,
        MixedT,
        CloudT,
    ],
]


class SimpleModel(AbstractSurfaceLayerModel[SimpleState]):
    """Simple surface layer model with constant friction velocity."""

    def __init__(self):
        pass

    def init_state(self, ustar: float) -> SimpleState:
        """Initialize the model state.

        Args:
            ustar: Friction velocity [m/s].

        Returns:
            The initial surface layer state.
        """
        return SimpleState(
            ustar=jnp.array(ustar),
        )

    def run(self, state: StateAlias):
        """Run the model.

        Args:
            state:

        Returns:
            The updated surface layer state.
        """
        atmos = state.atmos
        sl_state = atmos.surface
        uw = compute_uw(atmos.u, atmos.v, sl_state.ustar)
        vw = compute_vw(atmos.u, atmos.v, sl_state.ustar)
        ra = compute_ra(atmos.u, atmos.v, atmos.wstar, sl_state.ustar)
        return replace(sl_state, uw=uw, vw=vw, ra=ra)


def compute_uw(u: Array, v: Array, ustar: Array) -> Array:
    """Calculate the zonal momentum flux from wind components and friction velocity."""
    return jnp.where(
        u == 0.0,
        0.0,
        -jnp.sign(u) * (ustar**4.0 / (v**2.0 / u**2.0 + 1.0)) ** (0.5),
    )


def compute_vw(u: Array, v: Array, ustar: Array) -> Array:
    """Calculate the meridional momentum flux from wind components and friction velocity."""
    return jnp.where(
        v == 0.0,
        0.0,
        -jnp.sign(v) * (ustar**4.0 / (u**2.0 / v**2.0 + 1.0)) ** (0.5),
    )


def compute_ra(u: Array, v: Array, wstar: Array, ustar: Array) -> Array:
    """Calculate aerodynamic resistance from wind speed and friction velocity."""
    ueff = jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0)
    return ueff / jnp.maximum(1.0e-3, ustar) ** 2.0
