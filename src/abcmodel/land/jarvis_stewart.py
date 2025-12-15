from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ..abstracts import AbstractCoupledState
from ..utils import PhysicalConstants
from .standard import (
    AbstractStandardLandModel,
    StandardLandState,
)


@dataclass
class JarvisStewartState(StandardLandState):
    """Jarvis-Stewart model state."""


JarvisStewartInitConds = JarvisStewartState


class JarvisStewartModel(AbstractStandardLandModel):
    """Jarvis-Stewart land surface model with empirical surface resistance.

    ... (docstring omitted for brevity) ...
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_surface_resistance(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> AbstractCoupledState:
        """Update the surface resistance ``rs`` in the state using the Jarvis-Stewart model.

        Args:
            state: CoupledState.
            const: PhysicalConstants.

        Returns:
            CoupledState (with updated land component).
        """
        f1 = self.compute_f1(state.in_srad)
        f2 = self.compute_f2(state.land.wg)
        f3 = self.compute_f3(state.land.esat, state.land.e)
        f4 = self.compute_f4(state.atmos.mixed.theta)
        rs = self.rsmin / self.lai * f1 * f2 * f3 * f4
        landstate = state.land.replace(rs=rs)
        state = state.replace(land=landstate)
        return state

    def compute_f1(self, in_srad: Array) -> Array:
        """Compute rad factor f1."""
        ratio = (0.004 * in_srad + 0.05) / (0.81 * (0.004 * in_srad + 1.0))
        f1 = 1.0 / jnp.minimum(1.0, ratio)
        return f1

    def compute_f2(self, wg: Array) -> Array:
        """Compute soil moisture factor f2."""
        f2 = jnp.where(
            self.w2 > self.wwilt,
            (self.wfc - self.wwilt) / (wg - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, jnp.ndarray)
        f2 = jnp.maximum(f2, 1.0)
        return f2

    def compute_f3(self, esat: Array, e: Array) -> Array:
        """Compute VPD factor f3."""
        vpd = esat - e
        f3 = 1.0 / jnp.exp(-self.gD * vpd / 100.0)
        return f3

    def compute_f4(self, theta: Array) -> Array:
        """Compute temperature factor f4."""
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - theta) ** 2.0)
        return f4

    def update_co2_flux(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> AbstractCoupledState:
        """No CO2 flux is computed using this model. See :class:`~abcmodel.land.ags.AgsModel`."""
        return state
