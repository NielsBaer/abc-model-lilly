import numpy as np
from jaxtyping import PyTree

from ..utils import PhysicalConstants
from .standard import (
    AbstractStandardLandSurfaceModel,
    StandardLandSurfaceInitConds,
)


class JarvisStewartInitConds(StandardLandSurfaceInitConds):
    """Data class for Jarvis-Stewart model initial conditions.

    Arguments
    ---------
    - all arguments from StandardLandSurfaceInitConds.
    """

    pass


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    """Jarvis-Stewart land surface model with empirical surface resistance.

    Implementation of the Jarvis-Stewart approach for calculating surface resistance
    based on environmental stress factors. Uses multiplicative stress functions
    for radiation, soil moisture, vapor pressure deficit, and temperature effects
    on stomatal conductance.

    Processes
    ---------
    1. Inherit all standard land surface processes from parent class.
    2. Calculate surface resistance using four environmental stress factors.
    3. Apply Jarvis-Stewart multiplicative stress function approach.
    4. No CO2 flux calculations (simple implementation).

    Updates
    --------
    - ``rs``: surface resistance for transpiration [s m-1].
    - all updates from ``AbstractStandardLandSurfaceModel``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_f1(self, state: PyTree):
        """Calculate radiation-dependent scaling factor for surface processes.

        Returns correction factor based on incoming solar radiation that typically
        ranges from 1.0 to higher values, used to scale surface flux calculations.
        """
        ratio = (0.004 * state.in_srad + 0.05) / (0.81 * (0.004 * state.in_srad + 1.0))
        f1 = 1.0 / min(1.0, ratio)
        return f1

    def compute_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ):
        """
        Compute surface resistance using Jarvis-Stewart approach.

        Parameters
        ----------
        - ``const``: physical constants (currently unused).
        - ``radiation``: radiation model. Uses ``get_f1()`` method.
        - ``surface_layer``: surface layer model (currently unused).
        - ``mixed_layer``: mixed layer model. Uses ``esat``, ``e``, and ``theta``.

        Updates
        -------
        Updates ``self.rs`` using multiplicative stress factors for radiation (f1),
        soil moisture (f2), vapor pressure deficit (f3), and temperature (f4).
        """
        # calculate surface resistances using Jarvis-Stewart model
        f1 = self.get_f1(state)

        if state.w2 > self.wwilt:
            f2 = (self.wfc - self.wwilt) / (state.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(-self.gD * (state.esat - state.e) / 100.0)
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - state.theta) ** 2.0)

        state.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

        return state

    def compute_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ):
        """
        Pass.
        """
        return state
