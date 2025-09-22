from dataclasses import dataclass

import numpy as np
from jaxtyping import PyTree

from ..models import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants


@dataclass
class MinimalSurfaceLayerInitConds:
    """Data class for minimal surface layer model initial conditions.

    Arguments
    ---------
    - ``ustar``: surface friction velocity [m/s].

    Others
    ------
    - ``uw``: surface momentum flux u [m2 s-2].
    - ``vw``: surface momentum flux v [m2 s-2].
    """

    ustar: float
    uw: float = np.nan
    vw: float = np.nan


class MinimalSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Minimal surface layer model with constant friction velocity."""

    def __init__(self):
        pass

    @staticmethod
    def calculate_momentum_fluxes(
        u: float, v: float, ustar: float
    ) -> tuple[float, float]:
        """Calculate momentum fluxes from wind components and friction velocity."""
        if u == 0.0:
            uw = 0.0
        else:
            uw = -np.sign(u) * (ustar**4.0 / (v**2.0 / u**2.0 + 1.0)) ** (0.5)

        if v == 0.0:
            vw = 0.0
        else:
            vw = -np.sign(v) * (ustar**4.0 / (u**2.0 / v**2.0 + 1.0)) ** (0.5)
        return uw, vw

    def run(self, state: PyTree, const: PhysicalConstants):
        """Calculate momentum fluxes from wind components and friction velocity."""
        state.uw, state.vw = self.calculate_momentum_fluxes(
            state.u, state.v, state.ustar
        )
        return state

    @staticmethod
    def compute_ra(state: PyTree) -> float:
        """Calculate aerodynamic resistance from wind speed and friction velocity."""
        ueff = np.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        return ueff / max(1.0e-3, state.ustar) ** 2.0
