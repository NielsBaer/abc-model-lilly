from dataclasses import dataclass

import numpy as np
from jaxtyping import PyTree

from ..models import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants, get_psih, get_psim, get_qsat


@dataclass
class StandardSurfaceLayerInitConds:
    """Data class for standard surface layer model initial conditions.

    Arguments
    ---------
    - ``ustar``: surface friction velocity [m/s].
    - ``z0m``: roughness length for momentum [m].
    - ``z0h``: roughness length for scalars [m].
    - ``theta``: surface potential temperature [K].

    Others
    ------
    - ``drag_m``: drag coefficient for momentum [-]. Default: 1e12.
    - ``drag_s``: drag coefficient for scalars [-]. Default: 1e12.
    - ``uw``: # surface momentum flux u [m2 s-2].
    - ``vw``: # surface momentum flux v [m2 s-2].
    - ``temp_2m``: # 2m temperature [K].
    - ``q2m``: # 2m specific humidity [kg kg-1].
    - ``u2m``: 2m u-wind [m s-1].
    - ``v2m``: 2m v-wind [m s-1].
    - ``e2m``: 2m vapor pressure [Pa].
    - ``esat2m``: 2m saturated vapor pressure [Pa].
    - ``thetasurf``: surface potential temperature [K].
    - ``thetavsurf``: surface virtual potential temperature [K].
    - ``qsurf``: surface specific humidity [kg kg-1].
    - ``obukhov_length``: Obukhov length [m].
    - ``rib_number``: bulk Richardson number [-].

    """

    # the following variables should be initialized by the user
    ustar: float
    z0m: float
    z0h: float
    theta: float
    # the following variables are initialized to high values and
    # are expected to converge to realistic values during warmup
    drag_m: float = 1e12
    drag_s: float = 1e12
    # the following variables are initialized as NaNs and should
    # and are expected to be assigned during warmup
    uw: float = np.nan
    vw: float = np.nan
    temp_2m: float = np.nan
    q2m: float = np.nan
    u2m: float = np.nan
    v2m: float = np.nan
    e2m: float = np.nan
    esat2m: float = np.nan
    thetasurf: float = np.nan
    thetavsurf: float = np.nan
    qsurf: float = np.nan
    obukhov_length: float = np.nan
    rib_number: float = np.nan


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Standard surface layer model with atmospheric stability corrections.

    Calculates surface-atmosphere exchange using Monin-Obukhov similarity theory
    with stability functions and iterative solution for Obukhov length.

    Parameters
    ----------
    None.

    Processes
    ---------
    1. Calculate effective wind speed and surface properties.
    2. Determine bulk Richardson number and solve for Obukhov length.
    3. Compute drag coefficients with stability corrections.
    4. Calculate momentum fluxes and 2m diagnostic variables.
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_effective_wind_speed(
        u: float,
        v: float,
        wstar: float,
    ) -> float:
        """Calculate effective wind speed including convective effects."""
        return max(0.01, np.sqrt(u**2.0 + v**2.0 + wstar**2.0))

    @staticmethod
    def calculate_surface_properties(
        ueff: float,
        theta: float,
        wtheta: float,
        q: float,
        surf_pressure: float,
        rs: float,
        drag_s: float,
    ) -> tuple[float, float, float]:
        """Calculate surface temperature and humidity."""
        thetasurf = theta + wtheta / (drag_s * ueff)
        qsatsurf = get_qsat(thetasurf, surf_pressure)
        cq = (1.0 + drag_s * ueff * rs) ** -1.0
        qsurf = (1.0 - cq) * q + cq * qsatsurf
        thetavsurf = thetasurf * (1.0 + 0.61 * qsurf)
        return thetasurf, qsurf, thetavsurf

    @staticmethod
    def calculate_richardson_number(
        ueff: float, zsl: float, g: float, thetav: float, thetavsurf: float
    ) -> float:
        """Calculate bulk Richardson number."""
        rib_number = g / thetav * zsl * (thetav - thetavsurf) / ueff**2.0
        return min(rib_number, 0.2)

    @staticmethod
    def calculate_scalar_correction_term(zsl: float, oblen: float, z0h: float) -> float:
        """Calculate scalar stability correction term."""
        log_term = np.log(zsl / z0h)
        upper_stability = get_psih(zsl / oblen)
        surface_stability = get_psih(z0h / oblen)

        return log_term - upper_stability + surface_stability

    @staticmethod
    def calculate_momentum_correction_term(
        zsl: float, oblen: float, z0m: float
    ) -> float:
        """Calculate momentum stability correction term."""
        log_term = np.log(zsl / z0m)
        upper_stability = get_psim(zsl / oblen)
        surface_stability = get_psim(z0m / oblen)

        return log_term - upper_stability + surface_stability

    @staticmethod
    def calculate_rib_function(
        zsl: float, oblen: float, rib_number: float, z0h: float, z0m: float
    ) -> float:
        """Calculate Richardson number function for iteration."""
        scalar_term = StandardSurfaceLayerModel.calculate_scalar_correction_term(
            zsl, oblen, z0h
        )
        momentum_term = StandardSurfaceLayerModel.calculate_momentum_correction_term(
            zsl, oblen, z0m
        )

        return rib_number - zsl / oblen * scalar_term / momentum_term**2.0

    @staticmethod
    def calculate_rib_function_term(
        zsl: float, oblen: float, z0h: float, z0m: float
    ) -> float:
        """Calculate function term for derivative calculation."""
        scalar_term = StandardSurfaceLayerModel.calculate_scalar_correction_term(
            zsl, oblen, z0h
        )
        momentum_term = StandardSurfaceLayerModel.calculate_momentum_correction_term(
            zsl, oblen, z0m
        )

        return -zsl / oblen * scalar_term / momentum_term**2.0

    @staticmethod
    def ribtol(zsl: float, rib_number: float, z0h: float, z0m: float) -> float:
        """Iterative solution for Obukhov length from Richardson number."""
        # initial guess based on stability
        oblen = 1.0 if rib_number > 0.0 else -1.0
        oblen0 = 2.0 if rib_number > 0.0 else -2.0

        convergence_threshold = 0.001
        perturbation = 0.001

        while abs(oblen - oblen0) > convergence_threshold:
            oblen0 = oblen

            # calculate function value at current estimate
            fx = StandardSurfaceLayerModel.calculate_rib_function(
                zsl, oblen, rib_number, z0h, z0m
            )

            # calculate derivative using finite differences
            oblen_start = oblen - perturbation * oblen
            oblen_end = oblen + perturbation * oblen

            fx_start = StandardSurfaceLayerModel.calculate_rib_function_term(
                zsl, oblen_start, z0h, z0m
            )
            fx_end = StandardSurfaceLayerModel.calculate_rib_function_term(
                zsl, oblen_end, z0h, z0m
            )

            fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

            # Newton-Raphson update
            oblen = oblen - fx / fxdif

            # prevent runaway solutions
            if abs(oblen) > 1e15:
                break

        return oblen

    @staticmethod
    def calculate_drag_coefficients(
        zsl: float, k: float, obukhov_length: float, z0h: float, z0m: float
    ) -> tuple[float, float]:
        """Calculate drag coefficients with stability corrections."""
        # momentum stability correction
        momentum_correction = (
            StandardSurfaceLayerModel.calculate_momentum_correction_term(
                zsl, obukhov_length, z0m
            )
        )

        # scalar stability correction
        scalar_correction = StandardSurfaceLayerModel.calculate_scalar_correction_term(
            zsl, obukhov_length, z0h
        )

        # drag coefficients
        drag_m = k**2.0 / momentum_correction**2.0
        drag_s = k**2.0 / (momentum_correction * scalar_correction)
        return drag_m, drag_s

    @staticmethod
    def calculate_momentum_fluxes(
        ueff: float, u: float, v: float, drag_m: float
    ) -> tuple[float, float, float]:
        """Calculate momentum fluxes and friction velocity."""
        ustar = np.sqrt(drag_m) * ueff
        uw = -drag_m * ueff * u
        vw = -drag_m * ueff * v
        return ustar, uw, vw

    @staticmethod
    def calculate_2m_variables(
        wtheta: float,
        wq: float,
        surf_pressure: float,
        k: float,
        z0h: float,
        z0m: float,
        obukhov_length: float,
        thetasurf: float,
        qsurf: float,
        ustar: float,
        uw: float,
        vw: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate 2m diagnostic meteorological variables."""
        # stability correction terms
        scalar_correction = (
            np.log(2.0 / z0h)
            - get_psih(2.0 / obukhov_length)
            + get_psih(z0h / obukhov_length)
        )
        momentum_correction = (
            np.log(2.0 / z0m)
            - get_psim(2.0 / obukhov_length)
            + get_psim(z0m / obukhov_length)
        )

        # scaling factor for scalar fluxes
        scalar_scale = 1.0 / (ustar * k)
        momentum_scale = 1.0 / (ustar * k)

        # temperature and humidity at 2m
        temp_2m = thetasurf - wtheta * scalar_scale * scalar_correction
        q2m = qsurf - wq * scalar_scale * scalar_correction

        # wind components at 2m
        u2m = -uw * momentum_scale * momentum_correction
        v2m = -vw * momentum_scale * momentum_correction

        # vapor pressures at 2m
        # limamau: name these constants
        esat2m = 0.611e3 * np.exp(17.2694 * (temp_2m - 273.16) / (temp_2m - 35.86))
        e2m = q2m * surf_pressure / 0.622
        return temp_2m, q2m, u2m, v2m, e2m, esat2m

    def run(self, state: PyTree, const: PhysicalConstants):
        """
        Calculate surface layer turbulent exchange and diagnostic variables.

        Updates
        -------
        Updates all surface layer variables including momentum fluxes, drag coefficients,
        Obukhov length, and 2m diagnostic meteorological variables.
        """
        ueff = self.calculate_effective_wind_speed(state.u, state.v, state.wstar)

        (
            state.thetasurf,
            state.qsurf,
            state.thetavsurf,
        ) = self.calculate_surface_properties(
            ueff,
            state.theta,
            state.wtheta,
            state.q,
            state.surf_pressure,
            state.rs,
            state.drag_s,
        )

        zsl = 0.1 * state.abl_height
        state.rib_number = self.calculate_richardson_number(
            ueff, zsl, const.g, state.thetav, state.thetavsurf
        )

        # limamau: the following is rather slow
        # we can probably use a scan when JAX is on
        state.obukhov_length = self.ribtol(zsl, state.rib_number, state.z0h, state.z0m)

        state.drag_m, state.drag_s = self.calculate_drag_coefficients(
            zsl, const.k, state.obukhov_length, state.z0h, state.z0m
        )

        state.ustar, state.uw, state.vw = self.calculate_momentum_fluxes(
            ueff, state.u, state.v, state.drag_m
        )

        (
            state.temp_2m,
            state.q2m,
            state.u2m,
            state.v2m,
            state.e2m,
            state.esat2m,
        ) = self.calculate_2m_variables(
            state.wtheta,
            state.wq,
            state.surf_pressure,
            const.k,
            state.z0h,
            state.z0m,
            state.obukhov_length,
            state.thetasurf,
            state.qsurf,
            state.ustar,
            state.uw,
            state.vw,
        )

        return state

    @staticmethod
    def compute_ra(state: PyTree) -> float:
        """Calculate aerodynamic resistance from wind speed and drag coefficient."""
        ueff = np.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        return (state.drag_s * ueff) ** -1.0
