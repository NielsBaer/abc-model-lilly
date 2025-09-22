from dataclasses import dataclass

import numpy as np
from jaxtyping import PyTree

from ..models import AbstractRadiationModel
from ..utils import PhysicalConstants


@dataclass
class StandardRadiationInitConds:
    """Data class for standard radiation model initial conditions.

    Arguments
    --------
    - ``net_rad``: net surface radiation [W/m²].

    Others
    ------
    - ``in_srad``: incoming solar radiation [W/m²].
    - ``out_srad``: outgoing solar radiation [W/m²].
    - ``in_lrad``: incoming longwave radiation [W/m²].
    - ``out_lrad``: outgoing longwave radiation [W/m²].
    """

    net_rad: float
    in_srad: float = np.nan
    out_srad: float = np.nan
    in_lrad: float = np.nan
    out_lrad: float = np.nan


class StandardRadiationModel(AbstractRadiationModel):
    """Standard radiation model with solar position and atmospheric effects.

    Calculates time-varying solar radiation based on geographic location and
    atmospheric conditions. Includes both shortwave (solar) and longwave (thermal)
    radiation components.

    Parameters
    ----------
    - ``lat``: latitude [degrees], range -90 to +90.
    - ``lon``: longitude [degrees], range -180 to +180.
    - ``doy``: day of year [-], range 1 to 365.
    - ``tstart``: start time of day [hours UTC], range 0 to 24.
    - ``cc``: cloud cover fraction [-], range 0 to 1.

    Processes
    ---------
    1. Calculate solar declination and elevation angles.
    2. Determine air temperature and atmospheric transmission.
    3. Compute all radiation components and net surface radiation.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
    ):
        self.lat = lat
        self.lon = lon
        self.doy = doy
        self.tstart = tstart
        self.cc = cc

    @staticmethod
    def calculate_solar_declination(doy: float) -> float:
        """Calculate solar declination angle based on day of year."""
        return 0.409 * np.cos(2.0 * np.pi * (doy - 173.0) / 365.0)

    def calculate_solar_elevation(
        self, t: int, dt: float, solar_declination: float
    ) -> float:
        """Calculate solar elevation angle (sine of elevation)."""
        lat_rad = 2.0 * np.pi * self.lat / 360.0
        lon_rad = 2.0 * np.pi * self.lon / 360.0
        time_rad = 2.0 * np.pi * (t * dt + self.tstart * 3600.0) / 86400.0

        sinlea = np.sin(lat_rad) * np.sin(solar_declination) - np.cos(lat_rad) * np.cos(
            solar_declination
        ) * np.cos(time_rad + lon_rad)

        return max(sinlea, 0.0001)

    @staticmethod
    def calculate_air_temperature(
        surf_pressure: float,
        abl_height: float,
        theta: float,
        const: PhysicalConstants,
    ) -> float:
        """Calculate air temperature at reference level using potential temperature."""
        # calculate pressure at reference level (10% reduction from surface)
        ref_pressure = surf_pressure - 0.1 * abl_height * const.rho * const.g

        # convert potential temperature to actual temperature
        pressure_ratio = ref_pressure / surf_pressure
        air_temp = theta * (pressure_ratio ** (const.rd / const.cp))

        return air_temp

    def calculate_atmospheric_transmission(self, solar_elevation: float) -> float:
        """
        Calculate atmospheric transmission coefficient for solar radiation.
        """
        # clear-sky transmission increases with solar elevation
        clear_sky_trans = 0.6 + 0.2 * solar_elevation

        # cloud cover reduces transmission (40% reduction per unit cloud cover)
        cloud_reduction = 1.0 - 0.4 * self.cc

        return clear_sky_trans * cloud_reduction

    def calculate_radiation_components(
        self,
        solar_elevation: float,
        atmospheric_transmission: float,
        air_temp: float,
        alpha: float,
        surf_temp: float,
        const: PhysicalConstants,
    ) -> tuple[float, float, float, float, float]:
        """Calculate all radiation components and update attributes."""
        # shortwave radiation components
        in_srad = const.solar_in * atmospheric_transmission * solar_elevation
        out_srad = alpha * const.solar_in * atmospheric_transmission * solar_elevation

        # longwave radiation components
        in_lrad = 0.8 * const.bolz * air_temp**4.0
        out_lrad = const.bolz * surf_temp**4.0

        # net radiation
        net_rad = in_srad - out_srad + in_lrad - out_lrad

        return net_rad, in_srad, out_srad, in_lrad, out_lrad

    def run(
        self,
        state: PyTree,
        t: int,
        dt: float,
        const: PhysicalConstants,
    ):
        """Calculate radiation components and net surface radiation."""
        # solar position
        solar_declination = self.calculate_solar_declination(self.doy)
        solar_elevation = self.calculate_solar_elevation(t, dt, solar_declination)

        # atmospheric properties
        air_temp = self.calculate_air_temperature(
            state.surf_pressure,
            state.abl_height,
            state.theta,
            const,
        )
        atmospheric_transmission = self.calculate_atmospheric_transmission(
            solar_elevation
        )

        # all radiation components
        (
            state.net_rad,
            state.in_srad,
            state.out_srad,
            state.in_lrad,
            state.out_lrad,
        ) = self.calculate_radiation_components(
            solar_elevation,
            atmospheric_transmission,
            air_temp,
            state.alpha,
            state.surf_temp,
            const,
        )

        return state
