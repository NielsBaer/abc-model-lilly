from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..models import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants, get_psih, get_psim, get_qsat


# helper functions:
def calculate_effective_wind_speed(u: Array, v: Array, wstar: Array) -> Array:
    """Calculate effective wind speed including convective effects."""
    return jnp.maximum(0.01, jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0))


def calculate_surface_properties(
    ueff: Array,
    theta: Array,
    wtheta: Array,
    q: Array,
    surf_pressure: Array,
    rs: Array,
    drag_s: Array,
) -> tuple[Array, Array, Array]:
    """Calculate surface temperature and humidity."""
    thetasurf = theta + wtheta / (drag_s * ueff)
    qsatsurf = get_qsat(thetasurf, surf_pressure)
    cq = (1.0 + drag_s * ueff * rs) ** -1.0
    qsurf = (1.0 - cq) * q + cq * qsatsurf
    thetavsurf = thetasurf * (1.0 + 0.61 * qsurf)
    return thetasurf, qsurf, thetavsurf


def calculate_richardson_number(
    ueff: Array, zsl: Array, g: float, thetav: Array, thetavsurf: Array
) -> Array:
    """Calculate bulk Richardson number."""
    rib_number = g / thetav * zsl * (thetav - thetavsurf) / ueff**2.0
    return jnp.minimum(rib_number, 0.2)


def calculate_scalar_correction_term(zsl: Array, oblen: Array, z0h: Array) -> Array:
    """Calculate scalar stability correction term."""
    log_term = jnp.log(zsl / z0h)
    upper_stability = get_psih(zsl / oblen)
    surface_stability = get_psih(z0h / oblen)

    return log_term - upper_stability + surface_stability


def calculate_momentum_correction_term(zsl: Array, oblen: Array, z0m: Array) -> Array:
    """Calculate momentum stability correction term."""
    log_term = jnp.log(zsl / z0m)
    upper_stability = get_psim(zsl / oblen)
    surface_stability = get_psim(z0m / oblen)

    return log_term - upper_stability + surface_stability


@dataclass
class StandardSurfaceLayerInitConds:
    """Standard surface layer model initial state."""

    # the following variables should be initialized by the user
    ustar: float
    """Surface friction velocity [m/s]."""
    z0m: float
    """Roughness length for momentum [m]."""
    z0h: float
    """Roughness length for scalars [m]."""
    theta: float
    """Surface potential temperature [K]."""

    # the following variables are initialized to high values and
    # are expected to converge to realistic values during warmup
    drag_m: float = 1e12
    """Drag coefficient for momentum [-]."""
    drag_s: float = 1e12
    """Drag coefficient for scalars [-]."""

    # the following variables are initialized as NaNs and should
    # and are expected to be assigned during warmup
    uw: float = jnp.nan
    """Surface momentum flux u [m2 s-2]."""
    vw: float = jnp.nan
    """Surface momentum flux v [m2 s-2]."""
    temp_2m: float = jnp.nan
    """2m temperature [K]."""
    q2m: float = jnp.nan
    """2m specific humidity [kg kg-1]."""
    u2m: float = jnp.nan
    """2m u-wind [m s-1]."""
    v2m: float = jnp.nan
    """2m v-wind [m s-1]."""
    e2m: float = jnp.nan
    """2m vapor pressure [Pa]."""
    esat2m: float = jnp.nan
    """2m saturated vapor pressure [Pa]."""
    thetasurf: float = jnp.nan
    """Surface potential temperature [K]."""
    thetavsurf: float = jnp.nan
    """Surface virtual potential temperature [K]."""
    qsurf: float = jnp.nan
    """Surface specific humidity [kg kg-1]."""
    obukhov_length: float = jnp.nan
    """Obukhov length [m]."""
    rib_number: float = jnp.nan
    """Bulk Richardson number [-]."""


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Standard surface layer model with atmospheric stability corrections.

    Calculates surface-atmosphere exchange using Monin-Obukhov similarity theory
    with stability functions and iterative solution for Obukhov length.

    1. Calculate effective wind speed and surface properties.
    2. Determine bulk Richardson number and solve for Obukhov length.
    3. Compute drag coefficients with stability corrections.
    4. Calculate momentum fluxes and 2m diagnostic variables.
    """

    def __init__(self):
        pass

    def calculate_rib_function(
        self,
        zsl: Array,
        oblen: Array,
        rib_number: Array,
        z0h: Array,
        z0m: Array,
    ) -> Array:
        """Calculate Richardson number function for iteration."""
        scalar_term = calculate_scalar_correction_term(zsl, oblen, z0h)
        momentum_term = calculate_momentum_correction_term(zsl, oblen, z0m)

        return rib_number - zsl / oblen * scalar_term / momentum_term**2.0

    def calculate_rib_function_term(
        self,
        zsl: Array,
        oblen: Array,
        z0h: Array,
        z0m: Array,
    ) -> Array:
        """Calculate function term for derivative calculation."""
        scalar_term = calculate_scalar_correction_term(zsl, oblen, z0h)
        momentum_term = calculate_momentum_correction_term(zsl, oblen, z0m)

        return -zsl / oblen * scalar_term / momentum_term**2.0

    def ribtol(self, zsl: Array, rib_number: Array, z0h: Array, z0m: Array):
        """Iterative solution for Obukhov length from Richardson number (JAX version)."""

        # initial guess based on stability
        oblen = jnp.where(rib_number > 0.0, 1.0, -1.0)
        oblen0 = jnp.where(rib_number > 0.0, 2.0, -2.0)

        convergence_threshold = 0.001
        perturbation = 0.001
        max_oblen = 1e4

        def cond_fun(carry):
            oblen, oblen0 = carry
            res = jnp.logical_and(
                jnp.abs(oblen - oblen0) > convergence_threshold,
                jnp.abs(oblen) < max_oblen,
            ).squeeze()
            return res

        def body_fun(carry):
            oblen, _ = carry
            oblen0 = oblen

            # calculate function value at current estimate
            fx = self.calculate_rib_function(zsl, oblen, rib_number, z0h, z0m)

            # finite difference derivative
            oblen_start = oblen - perturbation * oblen
            oblen_end = oblen + perturbation * oblen

            fx_start = self.calculate_rib_function_term(zsl, oblen_start, z0h, z0m)
            fx_end = self.calculate_rib_function_term(zsl, oblen_end, z0h, z0m)

            fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

            # Newton–Raphson update
            oblen_new = oblen - fx / fxdif

            return oblen_new, oblen0

        oblen, _ = jax.lax.while_loop(cond_fun, body_fun, (oblen, oblen0))

        return oblen

    def calculate_drag_coefficients(
        self, zsl: Array, k: float, obukhov_length: Array, z0h: Array, z0m: Array
    ) -> tuple[Array, Array]:
        """Calculate drag coefficients with stability corrections."""
        # momentum stability correction
        momentum_correction = calculate_momentum_correction_term(
            zsl, obukhov_length, z0m
        )

        # scalar stability correction
        scalar_correction = calculate_scalar_correction_term(zsl, obukhov_length, z0h)

        # drag coefficients
        drag_m = k**2.0 / momentum_correction**2.0
        drag_s = k**2.0 / (momentum_correction * scalar_correction)
        return drag_m, drag_s

    @staticmethod
    def calculate_momentum_fluxes(
        ueff: Array, u: Array, v: Array, drag_m: Array
    ) -> tuple[Array, Array, Array]:
        """Calculate momentum fluxes and friction velocity."""
        ustar = jnp.sqrt(drag_m) * ueff
        uw = -drag_m * ueff * u
        vw = -drag_m * ueff * v
        return ustar, uw, vw

    @staticmethod
    def calculate_2m_variables(
        wtheta: Array,
        wq: Array,
        surf_pressure: Array,
        k: float,
        z0h: Array,
        z0m: Array,
        obukhov_length: Array,
        thetasurf: Array,
        qsurf: Array,
        ustar: Array,
        uw: Array,
        vw: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Calculate 2m diagnostic meteorological variables."""
        # stability correction terms
        scalar_correction = (
            jnp.log(2.0 / z0h)
            - get_psih(2.0 / obukhov_length)
            + get_psih(z0h / obukhov_length)
        )
        momentum_correction = (
            jnp.log(2.0 / z0m)
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
        esat2m = 0.611e3 * jnp.exp(17.2694 * (temp_2m - 273.16) / (temp_2m - 35.86))
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
        ueff = calculate_effective_wind_speed(state.u, state.v, state.wstar)

        (
            state.thetasurf,
            state.qsurf,
            state.thetavsurf,
        ) = calculate_surface_properties(
            ueff,
            state.theta,
            state.wtheta,
            state.q,
            state.surf_pressure,
            state.rs,
            state.drag_s,
        )

        zsl = 0.1 * state.abl_height
        state.rib_number = calculate_richardson_number(
            ueff, zsl, const.g, state.thetav, state.thetavsurf
        )

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
    def compute_ra(state: PyTree) -> Array:
        """Calculate aerodynamic resistance from wind speed and drag coefficient."""
        ueff = jnp.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        return (state.drag_s * ueff) ** -1.0
