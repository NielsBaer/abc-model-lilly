from dataclasses import dataclass, field
from typing import Generic

import jax.numpy as jnp
from flax import nnx
from jax import Array

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    AtmosT,
    LandT,
    RadT,
)


@dataclass
class CoupledState(
    AbstractCoupledState[RadT, LandT, AtmosT], Generic[RadT, LandT, AtmosT]
):
    """Hierarchical coupled state, generic over component types."""

    rad: RadT
    land: LandT
    atmos: AtmosT
    t: Array = field(default_factory=lambda: jnp.array(-1))
    total_water_mass: Array = field(default_factory=lambda: jnp.array(0.0))


class ABCoupler(nnx.Module):
    """Coupling class to bound all the components."""

    def __init__(
        self,
        rad: AbstractRadiationModel,
        land: AbstractLandModel,
        atmos: AbstractAtmosphereModel,
    ):
        self.rad = rad
        self.land = land
        self.atmos = atmos

    @staticmethod
    def init_state(
        rad_state: RadT,
        land_state: LandT,
        atmos_state: AtmosT,
    ) -> CoupledState[RadT, LandT, AtmosT]:
        return CoupledState(
            rad=rad_state,
            land=land_state,
            atmos=atmos_state,
        )

    def compute_diagnostics(self, state: AbstractCoupledState) -> AbstractCoupledState:
        """Compute diagnostic variables for total water budget."""
        # limamau: this needs to be re-implemented
        total_water_mass = jnp.array(0.0)
        return state.replace(total_water_mass=total_water_mass)
