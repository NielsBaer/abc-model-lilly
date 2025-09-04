import numpy as np

from .components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
)
from .utils import PhysicalConstants


class MinimalRadiationModel(AbstractRadiationModel):
    """Minimal radiation model with constant net radiation.

    Simple radiation model that maintains fixed net radiation values without
    any atmospheric or solar calculations.

    **Processes:**
    1. Maintains constant net radiation.

    Arguments
    ----------
    * ``net_rad``: net surface radiation [W/m²].
    * ``dFz``: cloud top radiative divergence [W/m²].

    Updates
    --------
    * No updates - ``net_rad`` remains constant.
    """

    def __init__(
        self,
        net_rad: float,
        dFz: float,
    ):
        self.net_rad = net_rad
        self.dFz = dFz

    def run(
        self,
        t: float,
        dt: float,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """No calculations - maintains constant radiation values."""
        pass

    def get_f1(self):
        """No scaling factor for surface processes."""
        return 1.0


class StandardRadiationModel(AbstractRadiationModel):
    """Standard radiation model with solar position and atmospheric effects.

    Calculates time-varying solar radiation based on geographic location and
    atmospheric conditions. Includes both shortwave (solar) and longwave (thermal)
    radiation components.

    **Processes:**
    1. Calculate solar declination and elevation angles.
    2. Determine air temperature and atmospheric transmission.
    3. Compute all radiation components and net surface radiation.

    Arguments
    ----------
    * ``lat``: latitude [degrees], range -90 to +90.
    * ``lon``: longitude [degrees], range -180 to +180.
    * ``doy``: day of year [-], range 1 to 365.
    * ``tstart``: start time of day [hours UTC], range 0 to 24.
    * ``cc``: cloud cover fraction [-], range 0 to 1.
    * ``net_rad``: net surface radiation [W/m²] (also updated).
    * ``dFz``: cloud top radiative divergence [W/m²].

    Updates
    --------
    * ``net_rad``: net surface radiation [W/m²].
    * ``in_srad``: incoming solar radiation [W/m²].
    * ``out_srad``: outgoing solar radiation [W/m²].
    * ``in_lrad``: incoming longwave radiation [W/m²].
    * ``out_lrad``: outgoing longwave radiation [W/m²].
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
        net_rad: float,
        dFz: float,
    ):
        self.lat = lat
        self.lon = lon
        self.doy = doy
        self.tstart = tstart
        self.cc = cc
        self.net_rad = net_rad
        self.dFz = dFz

    def _calculate_solar_declination(self, doy: float) -> float:
        """Calculate solar declination angle based on day of year."""
        return 0.409 * np.cos(2.0 * np.pi * (doy - 173.0) / 365.0)

    def _calculate_solar_elevation(
        self, t: float, dt: float, solar_declination: float
    ) -> float:
        """Calculate solar elevation angle (sine of elevation)."""
        lat_rad = 2.0 * np.pi * self.lat / 360.0
        lon_rad = 2.0 * np.pi * self.lon / 360.0
        time_rad = 2.0 * np.pi * (t * dt + self.tstart * 3600.0) / 86400.0

        sinlea = np.sin(lat_rad) * np.sin(solar_declination) - np.cos(lat_rad) * np.cos(
            solar_declination
        ) * np.cos(time_rad + lon_rad)

        return max(sinlea, 0.0001)

    def _calculate_air_temperature(
        self, mixed_layer: AbstractMixedLayerModel, const: PhysicalConstants
    ) -> float:
        """Calculate air temperature at reference level using potential temperature."""
        # calculate pressure at reference level (10% reduction from surface)
        ref_pressure = (
            mixed_layer.surf_pressure
            - 0.1 * mixed_layer.abl_height * const.rho * const.g
        )

        # convert potential temperature to actual temperature
        pressure_ratio = ref_pressure / mixed_layer.surf_pressure
        air_temp = mixed_layer.theta * (pressure_ratio ** (const.rd / const.cp))

        return air_temp

    def _calculate_atmospheric_transmission(self, solar_elevation: float) -> float:
        """
        Calculate atmospheric transmission coefficient for solar radiation.
        """
        # clear-sky transmission increases with solar elevation
        clear_sky_trans = 0.6 + 0.2 * solar_elevation

        # cloud cover reduces transmission (40% reduction per unit cloud cover)
        cloud_reduction = 1.0 - 0.4 * self.cc

        return clear_sky_trans * cloud_reduction

    def _calculate_radiation_components(
        self,
        solar_elevation: float,
        atmospheric_transmission: float,
        air_temp: float,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
    ):
        """Calculate all radiation components and update attributes."""
        # shortwave radiation components
        self.in_srad = const.solar_in * atmospheric_transmission * solar_elevation
        self.out_srad = (
            land_surface.alpha
            * const.solar_in
            * atmospheric_transmission
            * solar_elevation
        )

        # longwave radiation components
        self.in_lrad = 0.8 * const.bolz * air_temp**4.0
        self.out_lrad = const.bolz * land_surface.surf_temp**4.0

        # net radiation
        self.net_rad = self.in_srad - self.out_srad + self.in_lrad - self.out_lrad

    def run(
        self,
        t: float,
        dt: float,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Calculate radiation components and net surface radiation.

        Parameters
        ----------
        * ``t``: current time step.
        * ``dt``: time step size [s].
        * ``const``: physical constants.
        Uses ``solar_in``, ``bolz``, ``rd``, ``cp``, ``rho`` and ``g``.
        * ``land_surface``: land surface model. Uses ``alpha`` and ``surf_temp``.
        * ``mixed_layer`` : mixed layer model. Uses ``theta``, ``surf_pressure`` and ``abl_height``.

        Updates
        -------
        Updates ``net_rad`` and all radiation components (``in_srad``, ``out_srad``,
        ``in_lrad``, ``out_lrad``) based on current atmospheric and surface conditions.
        """
        # solar position
        solar_declination = self._calculate_solar_declination(self.doy)
        solar_elevation = self._calculate_solar_elevation(t, dt, solar_declination)

        # atmospheric properties
        air_temp = self._calculate_air_temperature(mixed_layer, const)
        atmospheric_transmission = self._calculate_atmospheric_transmission(
            solar_elevation
        )

        # all radiation components
        self._calculate_radiation_components(
            solar_elevation, atmospheric_transmission, air_temp, const, land_surface
        )

    def get_f1(self):
        """Calculate radiation-dependent scaling factor for surface processes.

        Returns correction factor based on incoming solar radiation that typically
        ranges from 1.0 to higher values, used to scale surface flux calculations.
        """
        ratio = (0.004 * self.in_srad + 0.05) / (0.81 * (0.004 * self.in_srad + 1.0))
        f1 = 1.0 / min(1.0, ratio)
        return f1
