import numpy as np

from ..components import (
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants
from .standard import AbstractStandardLandSurfaceModel


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    """Jarvis-Stewart land surface model with empirical surface resistance.

    Implementation of the Jarvis-Stewart approach for calculating surface resistance
    based on environmental stress factors. Uses multiplicative stress functions
    for radiation, soil moisture, vapor pressure deficit, and temperature effects
    on stomatal conductance.

    **Processes:**
    1. Inherit all standard land surface processes from parent class.
    2. Calculate surface resistance using four environmental stress factors.
    3. Apply Jarvis-Stewart multiplicative stress function approach.
    4. No CO2 flux calculations (simple implementation).

    Arguments
    ----------
    - ``wg``: volumetric water content top soil layer [m3 m-3].
    - ``w2``: volumetric water content deeper soil layer [m3 m-3].
    - ``temp_soil``: temperature top soil layer [K].
    - ``temp2``: temperature deeper soil layer [K].
    - ``a``: Clapp-Hornberger retention curve parameter [-].
    - ``b``: Clapp-Hornberger retention curve parameter [-].
    - ``p``: Clapp-Hornberger retention curve parameter [-].
    - ``cgsat``: saturated soil conductivity for heat [W m-1 K-1].
    - ``wsat``: saturated volumetric water content [-].
    - ``wfc``: volumetric water content field capacity [-].
    - ``wwilt``: volumetric water content wilting point [-].
    - ``c1sat``: saturated soil conductivity parameter [-].
    - ``c2sat``: reference soil conductivity parameter [-].
    - ``lai``: leaf area index [-].
    - ``gD``: correction factor transpiration for VPD [-].
    - ``rsmin``: minimum resistance transpiration [s m-1].
    - ``rssoilmin``: minimum resistance soil evaporation [s m-1].
    - ``alpha``: surface albedo [-], range 0 to 1.
    - ``surf_temp``: surface temperature [K].
    - ``cveg``: vegetation fraction [-], range 0 to 1.
    - ``wmax``: thickness of water layer on wet vegetation [m].
    - ``wl``: equivalent water layer depth for wet vegetation [m].
    - ``lam``: thermal diffusivity skin layer [-].

    Updates
    --------
    - ``rs``: surface resistance for transpiration [s m-1].
    - All updates from ``AbstractStandardLandSurfaceModel``.
    """

    def __init__(
        self,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
    ):
        super().__init__(
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
        )

    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
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
        f1 = radiation.get_f1()

        if self.w2 > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(-self.gD * (mixed_layer.esat - mixed_layer.e) / 100.0)
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - mixed_layer.theta) ** 2.0)

        self.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Compute CO2 flux (no-op implementation).

        Parameters
        ----------
        - ``const``: physical constants (unused).
        - ``surface_layer``: surface layer model (unused).
        - ``mixed_layer``: mixed layer model (unused).

        Updates
        -------
        No updates performed - this model does not calculate CO2 fluxes.
        """
        pass
