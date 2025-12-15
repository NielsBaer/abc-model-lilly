from dataclasses import dataclass, replace
from typing import Generic

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    LandT,
    RadT,
)
from ..utils import PhysicalConstants
from .abstracts import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
    CloudT,
    MixedT,
    SurfT,
)
from .clouds import NoCloudModel


@dataclass
class DayOnlyAtmosphereState(AbstractAtmosphereState, Generic[SurfT, MixedT, CloudT]):
    """Atmosphere state aggregating surface layer, mixed layer, and clouds."""

    surface: SurfT
    mixed: MixedT
    clouds: CloudT


# limamau: can we use this?
StateAlias = AbstractCoupledState[
    RadT,
    LandT,
    DayOnlyAtmosphereState[SurfT, MixedT, CloudT],
]


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
        atmostate = replace(state.atmos, surface=sl_state)
        state = state.replace(atmos=atmostate)
        cl_state = self.clouds.run(state, const)
        atmostate = replace(atmostate, clouds=cl_state)
        state = state.replace(atmos=atmostate)
        ml_state = self.mixed_layer.run(state, const)
        atmostate = replace(atmostate, mixed=ml_state)
        return atmostate

    def statistics(
        self, state: AbstractCoupledState, t: int, const: PhysicalConstants
    ) -> DayOnlyAtmosphereState:
        """Update statistics."""
        ml_state = self.mixed_layer.statistics(state, t, const)
        return state.atmos.replace(mixed=ml_state)

    def warmup(
        self,
        radmodel: AbstractRadiationModel,
        landmodel: AbstractLandModel,
        state: AbstractCoupledState,
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> AbstractCoupledState:
        """Warmup the atmos by running it for a few timesteps."""
        state = state.replace(
            atmos=self.statistics(state, t, const),
        )
        state = state.replace(rad=radmodel.run(state, t, dt, const))
        for _ in range(10):
            sl_state = self.surface_layer.run(state, const)
            atmostate = replace(state.atmos, surface=sl_state)
            state = state.replace(atmos=atmostate)
        landstate = landmodel.run(state, const)
        state = state.replace(land=landstate)

        # this is if clause is ok because it's outise the scan!
        if not isinstance(self.clouds, NoCloudModel):
            ml_state = self.mixed_layer.run(state, const)
            atmostate = replace(state.atmos, mixed=ml_state)
            state = state.replace(atmos=atmostate)
            cl_state = self.clouds.run(state, const)
            atmostate = replace(state.atmos, clouds=cl_state)
            state = state.replace(atmos=atmostate)
        ml_state = self.mixed_layer.run(state, const)
        atmostate = replace(state.atmos, mixed=ml_state)
        state = state.replace(atmos=atmostate)
        return state

    def integrate(
        self, state: DayOnlyAtmosphereState, dt: float
    ) -> DayOnlyAtmosphereState:
        ml_state = self.mixed_layer.integrate(state.mixed, dt)
        return replace(state, mixed=ml_state)
