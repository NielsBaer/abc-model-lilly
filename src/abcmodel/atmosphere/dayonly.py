from jaxtyping import PyTree

from ..abstracts import AbstractAtmosphereModel
from ..utils import PhysicalConstants
from .abstracts import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)


class DayOnlyAtmosphereModel(AbstractAtmosphereModel):
    """Atmosphere model aggregating surface layer, mixed layer, and clouds during the day-time."""

    def __init__(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
        clouds: AbstractCloudModel,
    ):
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.clouds = clouds

    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        state = self.surface_layer.run(state, const)
        state = self.clouds.run(state, const)
        state = self.mixed_layer.run(state, const)
        return state

    def statistics(self, state: PyTree, t: int, const: PhysicalConstants) -> PyTree:
        state = self.mixed_layer.statistics(state, t, const)
        return state

    def warmup(self, state: PyTree, const: PhysicalConstants, land) -> PyTree:
        """Warmup the atmosphere by running it for a few timesteps."""
        # iterate surface layer to converge turbulent fluxes
        for _ in range(10):
            state = self.surface_layer.run(state, const)

        # run land surface
        state = land.run(state, const)

        # conditionally run clouds if model is not NoCloudModel
        from .clouds import NoCloudModel

        if not isinstance(self.clouds, NoCloudModel):
            state = self.mixed_layer.run(state, const)
            state = self.clouds.run(state, const)

        # run mixed layer
        state = self.mixed_layer.run(state, const)

        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        state = self.mixed_layer.integrate(state, dt)
        return state
