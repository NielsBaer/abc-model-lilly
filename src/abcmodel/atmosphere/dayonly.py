from dataclasses import dataclass, replace

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
)
from ..utils import PhysicalConstants
from .abstracts import (
    AbstractCloudModel,
    AbstractCloudState,
    AbstractMixedLayerModel,
    AbstractMixedLayerState,
    AbstractSurfaceLayerModel,
    AbstractSurfaceLayerState,
)
from .clouds import NoCloudModel


@dataclass
class DayOnlyAtmosphereState(AbstractAtmosphereState):
    """Atmosphere state aggregating surface layer, mixed layer, and clouds."""

    surface_layer: AbstractSurfaceLayerState
    mixed_layer: AbstractMixedLayerState
    clouds: AbstractCloudState


class DayOnlyAtmosphereModel(AbstractAtmosphereModel[DayOnlyAtmosphereState]):
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
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> DayOnlyAtmosphereState:
        sl_state = self.surface_layer.run(state, const)
        new_atmosphere = replace(state.atmosphere, surface_layer=sl_state)
        state_with_sl = replace(state, atmosphere=new_atmosphere)
        cl_state = self.clouds.run(state_with_sl, const)
        new_atmosphere = replace(new_atmosphere, clouds=cl_state)
        state_with_cl = replace(state_with_sl, atmosphere=new_atmosphere)
        ml_state = self.mixed_layer.run(state_with_cl, const)
        new_atmosphere = replace(new_atmosphere, mixed_layer=ml_state)
        return new_atmosphere

    def statistics(
        self, state: DayOnlyAtmosphereState, t: int, const: PhysicalConstants
    ) -> DayOnlyAtmosphereState:
        """Update statistics."""
        ml_state = self.mixed_layer.statistics(state.mixed_layer, t, const)
        return replace(
            state,
            mixed_layer=ml_state,
        )

    def warmup(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
        land: AbstractLandModel,
    ) -> AbstractCoupledState:
        """Warmup the atmosphere by running it for a few timesteps."""
        for _ in range(10):
            sl_state = self.surface_layer.run(state, const)
            new_atmosphere = replace(state.atmosphere, surface_layer=sl_state)
            state = replace(state, atmosphere=new_atmosphere)
        land_state = land.run(state, const)
        state = replace(state, land=land_state)

        if not isinstance(self.clouds, NoCloudModel):
            ml_state = self.mixed_layer.run(state, const)
            new_atmosphere = replace(state.atmosphere, mixed_layer=ml_state)
            state = replace(state, atmosphere=new_atmosphere)
            cl_state = self.clouds.run(state, const)
            new_atmosphere = replace(state.atmosphere, clouds=cl_state)
            state = replace(state, atmosphere=new_atmosphere)
        ml_state = self.mixed_layer.run(state, const)
        new_atmosphere = replace(state.atmosphere, mixed_layer=ml_state)
        state = replace(state, atmosphere=new_atmosphere)
        return state

    def integrate(
        self, state: DayOnlyAtmosphereState, dt: float
    ) -> DayOnlyAtmosphereState:
        ml_state = self.mixed_layer.integrate(state.mixed_layer, dt)
        return replace(state, mixed_layer=ml_state)
