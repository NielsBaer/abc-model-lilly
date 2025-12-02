from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..abstracts import AbstractRadiationModel
from ..utils import PhysicalConstants


@dataclass
class StandardRadiationInitConds:
    """Standard radiation model initial state."""

    net_rad: float
    """Net surface radiation [W m-2]."""
    in_srad: float = jnp.nan
    """Incoming solar radiation [W m-2]."""
    out_srad: float = jnp.nan
    """Outgoing solar radiation [W m-2]."""
    in_lrad: float = jnp.nan
    """Incoming longwave radiation [W m-2]."""
    out_lrad: float = jnp.nan
    """Outgoing longwave radiation [W m-2]."""


class StandardRadiationModel(AbstractRadiationModel):
    """Standard radiation model with solar position and atmospheric effects.

    Calculates time-varying solar radiation based on geographic location and
    atmospheric conditions. Includes both shortwave (solar) and longwave (thermal)
    radiation components.

    Args:
        lat: latitude [degrees], range -90 to +90.
        lon: longitude [degrees], range -180 to +180.
        doy: day of year [-], range 1 to 365.
        tstart: start time of day [hours UTC], range 0 to 24.
        cc: cloud cover fraction [-], range 0 to 1.
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

    def run(
        self,
        state: PyTree,
        t: int,
        dt: float,
        const: PhysicalConstants,
    ):
        """Calculate radiation components and net surface radiation.

        Args:
            state: The current PyTree state of the model.
            t: Current time step index [-].
            dt: Time step duration [s].
            const: PhysicalConstants object.

        Returns:
            The updated state object with new radiation values.

        Notes:
            1.  Calculates solar position with
                :meth:`~compute_solar_declination` and
                :meth:`~compute_solar_elevation`.
            2.  Determines atmospheric properties with
                :meth:`~compute_air_temperature` and
                :meth:`~compute_atmospheric_transmission`.
            3.  Computes all radiation components and the final
                net radiation with :meth:`~compute_radiation_components`,
                then updates the state object with the results.
        """
        # solar position
        solar_declination = self.compute_solar_declination(self.doy)
        solar_elevation = self.compute_solar_elevation(t, dt, solar_declination)

        # atmospheric properties
        air_temp = self.compute_air_temperature(
            state.surf_pressure,
            state.h_abl,
            state.theta,
            const,
        )
        atmospheric_transmission = self.compute_atmospheric_transmission(
            solar_elevation
        )

        # all radiation components
        (
            state.net_rad,
            state.in_srad,
            state.out_srad,
            state.in_lrad,
            state.out_lrad,
        ) = self.compute_radiation_components(
            solar_elevation,
            atmospheric_transmission,
            air_temp,
            state.alpha,
            state.surf_temp,
            const,
        )

        return state

    def compute_solar_declination(self, doy: float) -> Array:
        """Compute solar declination angle based on day of year.

        Args:
            doy: Day of year [-], range 1 to 365.

        Returns:
            Solar declination angle [radians].

        Notes:
            Calculates the solar declination angle :math:`\\delta` using

            .. math::
                \\delta = 0.409 \\cdot \\cos\\left(
                    \\frac{2\\pi \\cdot (\\text{D} - 173)}{365}
                \\right)

            where :math:`\\text{D}` is the day of the year, :math:`0.409` is the
            approximate Earth's axial tilt in radians (23.45°), and :math:`173`
            is the approximate day of the summer solstice, which acts as the
            phase shift for the cosine wave.
        """
        return 0.409 * jnp.cos(2.0 * jnp.pi * (doy - 173.0) / 365.0)

    def compute_solar_elevation(
        self, t: int, dt: float, solar_declination: Array
    ) -> Array:
        """Compute solar elevation angle (sine of elevation).

        Args:
            t: current time step index [-].
            dt: time step duration [s].
            solar_declination: solar declination angle [radians] from :meth:`~compute_solar_declination`.

        Returns:
            Sine of the solar elevation angle [-].

        Notes:
            First, the latitude :math:`\\phi`, longitude :math:`\\lambda`, and
            time of day :math:`t` are converted to radians with

            .. math::
                \\phi_{\\text{rad}} = \\frac{2\\pi \\cdot \\text{lat}}{360},
            .. math::
                \\lambda_{\\text{rad}} = \\frac{2\\pi \\cdot \\text{lon}}{360},
            .. math::
                t_{\\text{rad}} = \\frac{2\\pi \\cdot (t \\cdot dt + t_{\\text{start}} \\cdot 3600)}{86400}.

            The sine of the solar elevation is then calculated as

            .. math::
                \\sin(\\alpha) = \\sin(\\phi_{\\text{rad}})\\sin(\\delta) -
                                \\cos(\\phi_{\\text{rad}})\\cos(\\delta)
                                \\cos(t_{\\text{rad}} + \\lambda_{\\text{rad}}),

            where :math:`\\delta` is the solar declination. The result is clipped
            at a minimum value of 0.0001 to represent night-time and avoid
            mathematical instability in subsequent calculations.
        """
        lat_rad = 2.0 * jnp.pi * self.lat / 360.0
        lon_rad = 2.0 * jnp.pi * self.lon / 360.0
        time_rad = 2.0 * jnp.pi * (t * dt + self.tstart * 3600.0) / 86400.0

        sinlea = jnp.sin(lat_rad) * jnp.sin(solar_declination) - jnp.cos(
            lat_rad
        ) * jnp.cos(solar_declination) * jnp.cos(time_rad + lon_rad)

        return jnp.maximum(sinlea, 0.0001)

    def compute_air_temperature(
        self,
        surf_pressure: float,
        h_abl: float,
        theta: float,
        const: PhysicalConstants,
    ) -> float:
        """Compute air temperature at reference level using potential temperature.

        Args:
            surf_pressure: surface pressure [Pa].
            h_abl: atmospheric boundary layer height [m].
            theta: potential temperature [K].
            const:

        Returns:
            Air temperature at the reference level [K].

        Notes:
            The calculation is a two-step process:

            1.  First, the pressure at a reference level :math:`P_{\\text{ref}}` is
                estimated from the surface pressure :math:`P_{\\text{surf}}` using
                the simplified hydrostatic approximation

                .. math::
                    P_{\\text{ref}} = P_{\\text{surf}} - 0.1 \\cdot h_{\\text{ABL}} \\cdot \\rho \\cdot g,

                where :math:`h_{\\text{ABL}}` is the atmospheric boundary layer (ABL) height,
                :math:`\\rho` is air density, and :math:`g` is gravity.

            2.  Second, the potential temperature :math:`\\theta` is converted to
                the actual air temperature :math:`T_{\\text{air}}` at the
                reference level using Poisson's equation (for an adiabatic process)

                .. math::
                    T_{\\text{air}} = \\theta\\left(
                        \\frac{P_{\\text{ref}}}{P_{\\text{surf}}}
                    \\right) ^{\\kappa},

                where the exponent :math:`\\kappa = R_d / c_p` is the ratio of the
                gas constant for dry air :math:`R_d` to the specific heat
                capacity of air :math:`c_p`.
        """
        # calculate pressure at reference level (10% reduction from surface)
        ref_pressure = surf_pressure - 0.1 * h_abl * const.rho * const.g

        # convert potential temperature to actual temperature
        pressure_ratio = ref_pressure / surf_pressure
        air_temp = theta * (pressure_ratio ** (const.rd / const.cp))

        return air_temp

    def compute_atmospheric_transmission(self, solar_elevation: Array) -> Array:
        """Compute atmospheric transmission coefficient for solar radiation.

        Args:
            solar_elevation: sine of the solar elevation angle [-].

        Returns:
            Atmospheric transmission coefficient [-].

        Notes:
            This is a simplified empirical parameterization (linear model) for
            atmospheric transmission (:math:`\\tau`).

            1.  A clear-sky transmission :math:`\\tau_{\\text{clear}}` is
                calculated based on the solar elevation :math:`\\sin(\\alpha)` as

                .. math::
                    \\tau_{\\text{clear}} = 0.6 + 0.2 \\cdot \\sin(\\alpha).

            2.  A cloud reduction factor :math:`f_{\\text{cloud}}` is
                calculated based on the cloud cover :math:`\\text{cc}` as

                .. math::
                    f_{\\text{cloud}} = 1.0 - 0.4 \\cdot \\text{cc}.

            3.  The final transmission is then the product of these two factors, giving

                .. math::
                    \\tau = \\tau_{\\text{clear}} \\cdot f_{\\text{cloud}}.
        """
        # clear-sky transmission increases with solar elevation
        clear_sky_trans = 0.6 + 0.2 * solar_elevation

        # cloud cover reduces transmission (40% reduction per unit cloud cover)
        cloud_reduction = 1.0 - 0.4 * self.cc

        return clear_sky_trans * cloud_reduction

    def compute_radiation_components(
        self,
        solar_elevation: Array,
        atmospheric_transmission: Array,
        air_temp: float,
        alpha: Array,
        surf_temp: Array,
        const: PhysicalConstants,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Compute all radiation components and update attributes.

        Args:
            solar_elevation: sine of the solar elevation angle [-].
            atmospheric_transmission: atmospheric transmission coefficient [-].
            air_temp: air temperature [K].
            alpha: surface albedo [-].
            surf_temp: surface temperature [K].
            const:

        Returns:
            A tuple containing net radiation, incoming shortwave radiation, outgoing shortwave
            radiation, incoming longwave radiation and outgoing longwave radiation [W/m^2].

        Notes:
            This function calculates the four components of the surface radiation
            budget and the resulting net radiation.

            **Shortwave radiation:**

            1.  Incoming shortwave :math:`SW_{\\text{in}}` is the solar constant
                :math:`S` attenuated by the atmosphere and projected onto the
                surface, given by

                .. math::
                    SW_{\\text{in}} = S \\cdot \\tau \\cdot \\sin(\\alpha),

                where :math:`\\tau` is the atmospheric transmission and
                :math:`\\sin(\\alpha)` is the sine of the solar elevation.

            2.  Outgoing shortwave :math:`SW_{\\text{out}}` is the fraction of
                incoming radiation reflected by the surface (albedo, :math:`\\alpha`), given by

                .. math::
                    SW_{\\text{out}} = \\alpha \\cdot SW_{\\text{in}}.

            **Longwave radiation:**

            Both longwave components are calculated using the Stefan-Boltzmann
            law :math:`E = \\epsilon \\sigma T^4`.

            3.  Incoming longwave :math:`LW_{\\text{in}}` is the radiation from the
                atmosphere, which is treated as a grey body with an emissivity
                :math:`\\epsilon_{\\text{atm}} = 0.8`, given by

                .. math::
                    LW_{\\text{in}} = 0.8 \\cdot \\sigma \\cdot T_{\\text{air}}^4,

                where :math:`\\sigma` is the Stefan-Boltzmann constant `const.bolz`.

            4.  Outgoing longwave :math:`LW_{\\text{out}}` is the radiation from the
                surface, assuming an emissivity of 1.0, given by

                .. math::
                    LW_{\\text{out}} = \\sigma \\cdot T_{\\text{surf}}^4.

            **Net radiation:**

            Finally, the net radiation :math:`R_{\\text{net}}` is given by the balance

            .. math::
                R_{\\text{net}} = (SW_{\\text{in}} - SW_{\\text{out}}) +
                                 (LW_{\\text{in}} - LW_{\\text{out}}).
        """
        # shortwave radiation components
        in_srad = const.solar_in * atmospheric_transmission * solar_elevation
        out_srad = alpha * const.solar_in * atmospheric_transmission * solar_elevation

        # longwave radiation components
        in_lrad = 0.8 * const.bolz * air_temp**4.0
        out_lrad = const.bolz * surf_temp**4.0

        # net radiation
        net_rad = in_srad - out_srad + in_lrad - out_lrad

        return net_rad, in_srad, out_srad, in_lrad, out_lrad
