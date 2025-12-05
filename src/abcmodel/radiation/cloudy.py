from dataclasses import dataclass, replace

import jax

from ..abstracts import AbstractCoupledState
from ..utils import PhysicalConstants
from .standard import StandardRadiationModel, StandardRadiationState

Array = jax.Array


@dataclass
class StandardRadiationwCloudsState(StandardRadiationState):
    """Standard radiation model with clouds state."""

    pass


StandardRadiationwCloudsInitConds = StandardRadiationwCloudsState


class StandardRadiationwCloudsModel(StandardRadiationModel):
    """Standard radiation model with solar position and atmospheric effects including prognostic cloud transmittance.

    Calculates time-varying solar radiation based on geographic location and
    atmospheric conditions. Includes both shortwave (solar) and longwave (thermal)
    radiation components.

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

    def run(
        self,
        state: AbstractCoupledState,
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> StandardRadiationwCloudsState:
        """Calculate radiation components and net surface radiation.

        Args:
            state: CoupledState.
            t: Current time step index [-].
            dt: Time step duration [s].
            const: PhysicalConstants object.

        Returns:
            The updated radiation state object.
        """
        # needed components
        rad_state = state.radiation
        ml_state = state.atmosphere.mixed_layer
        land_state = state.land
        cloud_state = state.atmosphere.clouds

        # computations
        solar_declination = self.compute_solar_declination(self.doy)
        solar_elevation = self.compute_solar_elevation(t, dt, solar_declination)
        air_temp = self.compute_air_temperature(
            ml_state.surf_pressure,
            ml_state.h_abl,
            ml_state.theta,
            const,
        )
        atmospheric_transmission = self.compute_atmospheric_transmission_w_clouds(
            solar_elevation,
            cloud_state.cl_trans,
        )
        (
            net_rad,
            in_srad,
            out_srad,
            in_lrad,
            out_lrad,
        ) = self.compute_radiation_components(
            solar_elevation,
            atmospheric_transmission,
            air_temp,
            land_state.alpha,
            land_state.surf_temp,
            const,
        )

        return replace(
            rad_state,
            net_rad=net_rad,
            in_srad=in_srad,
            out_srad=out_srad,
            in_lrad=in_lrad,
            out_lrad=out_lrad,
        )

    @staticmethod
    def compute_atmospheric_transmission_w_clouds(
        solar_elevation: Array, cl_trans: Array
    ) -> Array:
        """Calculate atmospheric transmission coefficient for solar radiation.

        Args:
            solar_elevation: sine of the solar elevation angle [-].
            cl_trans: prognostic cloud layer transmittance [-].

        Returns:
            Atmospheric transmission coefficient [-].

        Notes:
            This is a simplified empirical parameterization (linear model) for
            atmospheric transmission :math:`\\tau`.

            1.  A clear-sky transmission :math:`\\tau_{\\text{clear}}` is
                calculated based on the solar elevation :math:`\\sin(\\alpha)` as

                .. math::
                    \\tau_{\\text{clear}} = 0.6 + 0.2 \\cdot \\sin(\\alpha).

            2.  The prognostic cloud transmittance :math:`\\tau_{\\text{cloud}}`
                is provided directly by the model state (``cl_trans``).

            3.  The final transmission is then the product of these two factors, giving

                .. math::
                    \\tau = \\tau_{\\text{clear}} \\cdot \\tau_{\\text{cloud}}.
        """
        # clear-sky transmission increases with solar elevation
        clear_sky_trans = 0.6 + 0.2 * solar_elevation
        # apply cloud layer transmissivity and return
        return clear_sky_trans * cl_trans
