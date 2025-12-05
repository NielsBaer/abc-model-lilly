from dataclasses import dataclass, field
from typing import Generic, TypeVar

from simple_pytree import Pytree

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
    AbstractRadiationModel,
    AbstractRadiationState,
)
from .utils import PhysicalConstants


class DiagnosticsState(Pytree):
    """Diagnostic variables for the coupled system."""

    total_water_mass: float = 0.0
    total_energy: float = 0.0


# type variables for CoupledState
A = TypeVar("A", bound=AbstractAtmosphereState)
L = TypeVar("L", bound=AbstractLandState)
R = TypeVar("R", bound=AbstractRadiationState)


@dataclass
class CoupledState(AbstractCoupledState[A, L, R], Pytree, Generic[A, L, R]):
    """Hierarchical coupled state, generic over component types."""

    radiation: R
    land: L
    atmosphere: A
    diagnostics: DiagnosticsState = field(default_factory=DiagnosticsState)


class ABCoupler:
    """Coupling class to bound all the components."""

    def __init__(
        self,
        radiation: AbstractRadiationModel,
        land: AbstractLandModel,
        atmosphere: AbstractAtmosphereModel,
    ):
        self.radiation = radiation
        self.land = land
        self.atmosphere = atmosphere
        self.const = PhysicalConstants()

    @staticmethod
    def init_state(
        radiation_state: R,
        land_state: L,
        atmosphere_state: A,
    ) -> CoupledState[A, L, R]:
        return CoupledState(
            radiation=radiation_state,
            land=land_state,
            atmosphere=atmosphere_state,
        )

    def compute_diagnostics(self, state: CoupledState) -> CoupledState:
        """Compute diagnostic variables for total water budget."""
        return state
