"""Abstract classes for atmos sub-modules."""

from abc import abstractmethod
from typing import Generic, TypeVar

from ..abstracts import AbstractCoupledState, AbstractModel, AbstractState
from ..utils import Array, PhysicalConstants


class AbstractSurfaceLayerState(AbstractState):
    """Abstract surface layer state."""

    ra: Array
    """Aerodynamic resistance [s/m]."""


class AbstractMixedLayerState(AbstractState):
    """Abstract mixed layer state."""


class AbstractCloudState(AbstractState):
    """Abstract cloud state."""


SurfT = TypeVar("SurfT", bound=AbstractSurfaceLayerState)
MixedT = TypeVar("MixedT", bound=AbstractMixedLayerState)
CloudT = TypeVar("CloudT", bound=AbstractCloudState)


class AbstractSurfaceLayerModel(AbstractModel, Generic[SurfT]):
    """Abstract surface layer model class to define the interface for all surface layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> SurfT:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel, Generic[MixedT]):
    """Abstract mixed layer model class to define the interface for all mixed layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> MixedT:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self, state: AbstractCoupledState, t: int, const: PhysicalConstants
    ) -> MixedT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: MixedT, dt: float) -> MixedT:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel, Generic[CloudT]):
    """Abstract cloud model class to define the interface for all cloud models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> CloudT:
        raise NotImplementedError
