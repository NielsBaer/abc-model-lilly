from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


def get_esat(temp: Array) -> Array:
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


def get_qsat(temp: Array, pressure: Array) -> Array:
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
    esat = get_esat(temp)
    return 0.622 * esat / pressure


def get_psim(zeta: Array) -> Array:
    """Calculate momentum stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L [-].

    Returns:
        Momentum stability correction [-].

    Notes:
        This function calculates the integrated stability correction function for
        momentum :math:`\\Psi_m`, which is used to adjust wind profiles based
        on atmospheric stability.

        The function is piecewise, depending on the stability parameter
        :math:`\\zeta = z/L`.

        **1. Unstable conditions (ζ ≤ 0):**

        Based on Businger-Dyer relations, an intermediate variable

        .. math::
            x = (1 - 16\\zeta)^{1/4}

        is used to write the stability function as

        .. math::
            \\Psi_m(\\zeta) = \\ln\\left( \\frac{(1+x)^2 (1+x^2)}{8} \\right)
                             - 2 \\arctan(x) + \\frac{\\pi}{2}.

        **2. Stable conditions (ζ > 0):**

        This uses an empirical formula (e.g., Holtslag and De Bruin, 1988)
        with constants:

        - :math:`\\alpha = 0.35`,
        - :math:`\\beta = 5.0 / \\alpha`,
        - :math:`\\gamma = (10.0 / 3.0) / \\alpha`.

        The stability function is then  given by

        .. math::
            \\Psi_m(\\zeta) = -\\frac{2}{3}(\\zeta - \\beta)e^{-\\alpha \\zeta}
                             - \\zeta - \\gamma.
    """
    # constants for stable conditions
    alpha = 0.35
    beta = 5.0 / alpha
    gamma = (10.0 / 3.0) / alpha
    pi_half = jnp.pi / 2.0

    # unstable conditions (zeta <= 0)
    x = (1.0 - 16.0 * zeta) ** 0.25
    arctan_term = 2.0 * jnp.arctan(x)
    log_numerator = (1.0 + x) ** 2.0 * (1.0 + x**2.0)
    log_term = jnp.log(log_numerator / 8.0)
    psim_unstable = pi_half - arctan_term + log_term

    # stable conditions (zeta > 0)
    exponential_term = (zeta - beta) * jnp.exp(-alpha * zeta)
    psim_stable = -2.0 / 3.0 * exponential_term - zeta - gamma

    # select based on stability condition
    psim = jnp.where(zeta <= 0, psim_unstable, psim_stable)

    return psim


def get_psih(zeta: Array) -> Array:
    """Calculate scalar stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L [-].

    Returns:
        Scalar stability correction [-].

    Notes:
        This function calculates the integrated stability correction function for
        scalars :math:`\\Psi_h`, like heat and humidity, which is used to
        adjust temperature and humidity profiles based on atmospheric stability.

        The function is piecewise, depending on the stability parameter
        :math:`\\zeta = z/L`.

        **1. Unstable conditions (ζ ≤ 0):**

        Based on Businger-Dyer relations, an intermediate variable (same as above)

        .. math::
            x = (1 - 16\\zeta)^{1/4}

        is used to write the integrated stability function

        .. math::
            \\Psi_h(\\zeta) = 2 \\ln\\left( \\frac{1+x^2}{2} \\right).

        **2. Stable conditions (ζ > 0):**

        This uses a corresponding empirical formula with the same constants
        (:math:`\\alpha`, :math:`\\beta`, :math:`\\gamma`) as above to write

        .. math::
            \\Psi_h(\\zeta) = -\\frac{2}{3}(\\zeta - \\beta)e^{-\\alpha \\zeta}
                            - \\left(1 + \\frac{2}{3}\\zeta\\right)^{3/2}
                            - \\gamma + 1.
    """
    # constants for stable conditions
    alpha = 0.35
    beta = 5.0 / alpha
    gamma = (10.0 / 3.0) / alpha

    # unstable conditions (zeta <= 0)
    x = (1.0 - 16.0 * zeta) ** 0.25
    log_argument = (1.0 + x * x) / 2.0
    psih_unstable = 2.0 * jnp.log(log_argument)

    # stable conditions (zeta > 0)
    exponential_term = (zeta - beta) * jnp.exp(-alpha * zeta)
    power_term = (1.0 + (2.0 / 3.0) * zeta) ** 1.5
    psih_stable = -2.0 / 3.0 * exponential_term - power_term - gamma + 1.0

    # select based on stability condition
    psih = jnp.where(zeta <= 0, psih_unstable, psih_stable)

    return psih


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
