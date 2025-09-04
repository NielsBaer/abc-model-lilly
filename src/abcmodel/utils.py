import numpy as np


def get_esat(temp):
    """Calculate saturated vapor pressure using Tetens formula.

    Parameters
    ----------
    - ``temp``: temperature [K].

    Returns
    -------
    - Saturated vapor pressure [Pa].
    """
    temp_celsius = temp - 273.16
    denominator = temp - 35.86
    return 0.611e3 * np.exp(17.2694 * temp_celsius / denominator)


def get_qsat(temp, pressure):
    """Calculate saturated specific humidity.

    Parameters
    ----------
    - ``temp``: temperature [K].
    - ``pressure``: pressure [Pa].

    Returns
    -------
    - Saturated specific humidity [kg/kg].
    """
    esat = get_esat(temp)
    return 0.622 * esat / pressure


def get_psim(zeta):
    """Calculate momentum stability function from Monin-Obukhov similarity theory.

    Parameters
    ----------
    - ``zeta``: stability parameter z/L [-].

    Returns
    -------
    - Momentum stability correction [-].
    """
    if zeta <= 0:
        # unstable conditions
        x = (1.0 - 16.0 * zeta) ** 0.25

        arctan_term = 2.0 * np.arctan(x)
        log_numerator = (1.0 + x) ** 2.0 * (1.0 + x**2.0)
        log_term = np.log(log_numerator / 8.0)

        psim = 3.14159265 / 2.0 - arctan_term + log_term
    else:
        # stable conditions
        exponential_term = (zeta - 5.0 / 0.35) * np.exp(-0.35 * zeta)
        constant_term = (10.0 / 3.0) / 0.35

        psim = -2.0 / 3.0 * exponential_term - zeta - constant_term

    return psim


def get_psih(zeta):
    """Calculate scalar stability function from Monin-Obukhov similarity theory.

    Parameters
    ----------
    - ``zeta``: stability parameter z/L [-].

    Returns
    -------
    - Scalar stability correction [-].
    """
    if zeta <= 0:
        # unstable conditions
        x = (1.0 - 16.0 * zeta) ** 0.25
        log_argument = (1.0 + x * x) / 2.0
        psih = 2.0 * np.log(log_argument)
    else:
        # stable conditions
        exponential_term = (zeta - 5.0 / 0.35) * np.exp(-0.35 * zeta)
        power_term = (1.0 + (2.0 / 3.0) * zeta) ** 1.5
        constant_term = (10.0 / 3.0) / 0.35

        psih = -2.0 / 3.0 * exponential_term - power_term - constant_term + 1.0

    return psih


class PhysicalConstants:
    """Container for physical constants used throughout the model."""

    def __init__(self):
        # 1. thermodynamic constants:
        # heat of vaporization [J kg-1]
        self.lv = 2.5e6
        # specific heat of dry air [J kg-1 K-1]
        self.cp = 1005.0
        # density of air [kg m-3]
        self.rho = 1.2
        # gravity acceleration [m s-2]
        self.g = 9.81
        # gas constant for dry air [J kg-1 K-1]
        self.rd = 287.0
        # gas constant for moist air [J kg-1 K-1]
        self.rv = 461.5
        # density of water [kg m-3]
        self.rhow = 1000.0

        # 2. physical constants:
        # von Karman constant [-]
        self.k = 0.4
        # Boltzmann constant [-]
        self.bolz = 5.67e-8
        # solar constant [W m-2]
        self.solar_in = 1368.0

        # 3. molecular weights:
        # molecular weight CO2 [g mol-1]
        self.mco2 = 44.0
        # molecular weight air [g mol-1]
        self.mair = 28.9
        # ratio molecular viscosity water to carbon dioxide
        self.nuco2q = 1.6
