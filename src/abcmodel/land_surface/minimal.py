from dataclasses import dataclass

from jaxtyping import PyTree

from ..models import (
    AbstractLandSurfaceModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_esat, get_qsat


@dataclass
class MinimalLandSurfaceInitConds:
    """Data class for minimal land surface model initial conditions.

    Arguments
    ---------
    - ``alpha``: surface albedo [-], range 0 to 1.
    - ``surf_temp``: surface temperature [K].
    - ``rs``: surface resistance [s m-1].
    """

    alpha: float
    surf_temp: float
    rs: float


class MinimalLandSurfaceModel(AbstractLandSurfaceModel):
    """Minimal land surface model with fixed surface properties.

    Parameters
    ----------
    None.

    """

    def __init__(self):
        pass

    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
    ):
        """Execute land surface model calculations."""

        # limamau: the following two blocks are also computed by
        # the standard class - should we refactor some things here?
        # (1) compute aerodynamic resistance
        state.ra = surface_layer.compute_ra(state)

        # (2) calculate essential thermodynamic variables
        state.esat = get_esat(state.theta)
        state.qsat = get_qsat(state.theta, state.surf_pressure)
        desatdT = state.esat * (
            17.2694 / (state.theta - 35.86)
            - 17.2694 * (state.theta - 273.16) / (state.theta - 35.86) ** 2.0
        )
        state.dqsatdT = 0.622 * desatdT / state.surf_pressure
        state.e = state.q * state.surf_pressure / 0.622

        return state

    def integrate(self, state: PyTree, dt: float):
        """
        Integrate model forward in time.
        """
        return state
