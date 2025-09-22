from abc import abstractmethod

from jaxtyping import PyTree

from .utils import PhysicalConstants


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel):
    # limamau: this attribute should probably go out of here...
    tstart: float

    @abstractmethod
    def run(self, state: PyTree, t: int, dt: float, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError


class AbstractLandSurfaceModel(AbstractModel):
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
    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_ra(state: PyTree) -> float:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel):
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
    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError
