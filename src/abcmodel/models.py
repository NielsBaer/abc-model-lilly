"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod

from jaxtyping import Array, PyTree

from .utils import PhysicalConstants


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel):
    """Abstract radiation model class to define the interface for all radiation models."""

    tstart: float
    """Start time of the model."""

    @abstractmethod
    def run(self, state: PyTree, t: int, dt: float, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError


class AbstractLandSurfaceModel(AbstractModel):
    """Abstract land surface model class to define the interface for all land surface models."""

    @abstractmethod
    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
        surface_layer: "AbstractSurfaceLayerModel",
    ) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: PyTree, dt: float) -> PyTree:
        raise NotImplementedError


class AbstractSurfaceLayerModel(AbstractModel):
    """Abstract surface layer model class to define the interface for all surface layer models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_ra(state: PyTree) -> Array:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel):
    """Abstract mixed layer model class to define the interface for all mixed layer models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, state: PyTree, t: int, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: PyTree, dt: float) -> PyTree:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel):
    """Abstract cloud model class to define the interface for all cloud models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError
