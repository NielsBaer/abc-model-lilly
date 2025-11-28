from dataclasses import dataclass

from jaxtyping import Array, PyTree

from ..abstracts import (
    AbstractLandSurfaceModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, compute_esat, compute_qsat


@dataclass
class MinimalLandSurfaceInitConds:
    """Minimal land surface model initial state."""

    alpha: float
    """surface albedo [-], range 0 to 1."""
    surf_temp: float
    """Surface temperature [K]."""
    rs: float
    """Surface resistance [s m-1]."""
    wg: float = 0.0
    """No moisture content in the root zone [m3 m-3]."""
    wl: float = 0.0
    """No water content in the canopy [m]."""


class MinimalLandSurfaceModel(AbstractLandSurfaceModel):
    """Minimal land surface model with fixed surface properties."""

    def __init__(self):
        self.d1 = 0.0

    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
    ):
        """Run the model.

        Args:
            state: the state object carrying all variables.
            const: the physical constants object.
            surface_layer: the surface layer model.

        Returns:
            The updated state object.
        """
        # (1) compute aerodynamic resistance
        state.ra = surface_layer.compute_ra(state)

        # (2) calculate essential thermodynamic variables
        state.esat = compute_esat(state.theta)
        state.qsat = compute_qsat(state.theta, state.surf_pressure)
        state.dqsatdT = self.compute_dqsatdT(state)
        state.e = self.compute_e(state)

        return state

    def compute_dqsatdT(self, state: PyTree) -> Array:
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :meth:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :meth:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (state.theta - 273.16)
        den = (state.theta - 35.86) ** 2.0
        mult = num / den
        desatdT = state.esat * mult
        return 0.622 * desatdT / state.surf_pressure

    def compute_e(self, state: PyTree) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return state.q * state.surf_pressure / 0.622

    def integrate(self, state: PyTree, dt: float):
        """Integrate the model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        return state
