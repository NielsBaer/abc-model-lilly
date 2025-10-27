from dataclasses import dataclass

from jaxtyping import Array, PyTree

from ..utils import PhysicalConstants
from .standard import StandardRadiationInitConds, StandardRadiationModel


@dataclass
class StandardRadiationwCloudsInitConds(StandardRadiationInitConds):
    """Standard radiation model with clouds initial state."""


class StandardRadiationwCloudsModel(StandardRadiationModel):
    """Standard radiation model with solar position and atmospheric effects including prognostic cloud transmittance.

    Calculates time-varying solar radiation based on geographic location and
    atmospheric conditions. Includes both shortwave (solar) and longwave (thermal)
    radiation components.

    1. Calculate solar declination and elevation angles.
    2. Determine air temperature and atmospheric transmission.
    3. Compute all radiation components and net surface radiation.

    Args:
        lat: latitude [degrees], range -90 to +90.
        lon: longitude [degrees], range -180 to +180.
        doy: day of year [-], range 1 to 365.
        tstart: start time of day [hours UTC], range 0 to 24.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
    ):
        self.lat = lat
        self.lon = lon
        self.doy = doy
        self.tstart = tstart

    @staticmethod
    def calculate_atmospheric_transmission_w_clouds(
        solar_elevation: Array, cl_trans: Array
    ) -> Array:
        """Calculate atmospheric transmission coefficient for solar radiation based on elevation and cloud layer transmittance."""
        # clear-sky transmission increases with solar elevation
        clear_sky_trans = 0.6 + 0.2 * solar_elevation
        # apply cloud layer transmissivity and return
        return clear_sky_trans * cl_trans

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
        atmospheric_transmission = self.calculate_atmospheric_transmission_w_clouds(
            solar_elevation,
            state.cl_trans,
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
