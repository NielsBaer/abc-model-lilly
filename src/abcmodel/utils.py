from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


def compute_esat(temp: Array) -> Array:
    """Calculate saturated vapor pressure using the August-Roche-Magnus formula.

    Args:
        temp: temperature [K].

    Returns:
        Saturated vapor pressure [Pa].

    Notes:
        First, the temperature is converted from Kelvin
        (:math:`T_K`) to Celsius (:math:`T_C`) with

        .. math::
            T_C = T_K - 273.16,

        then, the saturated vapor pressure :math:`e_{sat}` is calculated as

        .. math::
            e_{\\text{sat}}(T_C) = 611 \\cdot \\exp\\left( \\frac{17.2694 \\cdot T_C}{T_C + 237.3} \\right),

        where :math:`611` [Pa] is a reference pressure. For more on this, see
        `wikipedia <https://en.wikipedia.org/wiki/Clausius–Clapeyron_relation#Meteorology_and_climatology>`_.
    """
    temp_celsius = temp - 273.16
    denominator = temp - 35.86
    return 0.611e3 * jnp.exp(17.2694 * temp_celsius / denominator)


def compute_qsat(temp: Array, pressure: Array) -> Array:
    """Calculate saturated specific humidity.

    Args:
        temp: temperature [K].
        pressure: pressure [Pa].

    Returns:
        Saturated specific humidity [kg/kg].

    Notes:
        Saturated specific humidity :math:`q_{sat}` is the maximum amount of
        water vapor (as a mass fraction) that a parcel of air can hold at
        a given temperature and pressure.

        The full formula for :math:`q_{sat}` is

        .. math::
            q_{\\text{sat}} = \\frac{\\epsilon \\cdot e_{\\text{sat}}}{p - (1-\\epsilon)e_{\\text{sat}}},

        where :math:`e_{\\text{sat}}` is the saturated vapor pressure [Pa] from :func:`~get_esat`,
        :math:`p` is the total atmospheric pressure [Pa] and
        :math:`\\epsilon \\approx 0.622` is the ratio of the molar mass of water vapor to the molar mass of dry air.
        This formula can be derived from the definition of specific humidity (a ratio of vapour and total air mass),
        and then using the Ideal Gas Law and Dalton's Law of Partial Pressures.

        In the code, this function uses a common approximation where the
        :math:`(1-\\epsilon)e_{\\text{sat}}` term in the denominator is
        negligible compared to :math:`p`, simplifying the formula to

        .. math::
            q_{\\text{sat}} \\approx \\epsilon \\frac{e_{\\text{sat}}}{p}.
    """
    esat = compute_esat(temp)
    return 0.622 * esat / pressure


@dataclass
class PhysicalConstants:
    """Container for physical constants used throughout the model."""

    lv = 2.5e6
    """Heat of vaporization [J kg-1]."""
    cp = 1005.0
    """Specific heat of dry air [J kg-1 K-1]."""
    rho = 1.2
    """Density of air [kg m-3]."""
    g = 9.81
    """Gravity acceleration [m s-2]."""
    rd = 287.0
    """Gas constant for dry air [J kg-1 K-1]."""
    rv = 461.5
    """Gas constant for moist air [J kg-1 K-1]."""
    rhow = 1000.0
    """Density of water [kg m-3]."""
    k = 0.4
    """Von Karman constant [-]."""
    bolz = 5.67e-8
    """Boltzmann constant [-]."""
    solar_in = 1368.0
    """Solar constant [W m-2]"""
    mco2 = 44.0
    """Molecular weight CO2 [g mol-1]."""
    mair = 28.9
    """Molecular weight air [g mol-1]."""
    nuco2q = 1.6
    """Ratio molecular viscosity water to carbon dioxide."""
