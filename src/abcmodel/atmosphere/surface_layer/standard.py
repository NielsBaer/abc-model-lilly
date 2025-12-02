from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...utils import PhysicalConstants, compute_qsat
from ..abstracts import AbstractSurfaceLayerModel


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
    """

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """Run the model.

        Args:
            state:
            const:

        Returns:
            The updated state.
        """
        ueff = compute_effective_wind_speed(state.u, state.v, state.wstar)
        (
            state.thetasurf,
            state.qsurf,
            state.thetavsurf,
        ) = compute_surface_properties(
            ueff,
            state.theta,
            state.wtheta,
            state.q,
            state.surf_pressure,
            state.rs,
            state.drag_s,
        )

        # this should be a method
        zsl = 0.1 * state.h_abl
        state.rib_number = compute_richardson_number(
            ueff, zsl, const.g, state.thetav, state.thetavsurf
        )
        state.obukhov_length = ribtol(zsl, state.rib_number, state.z0h, state.z0m)
        state.drag_m, state.drag_s = compute_drag_coefficients(
            zsl, const.k, state.obukhov_length, state.z0h, state.z0m
        )
        state.ustar, state.uw, state.vw = compute_momentum_fluxes(
            ueff, state.u, state.v, state.drag_m
        )
        (
            state.temp_2m,
            state.q2m,
            state.u2m,
            state.v2m,
            state.e2m,
            state.esat2m,
        ) = compute_2m_variables(
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
        """Calculate aerodynamic resistance from wind speed and drag coefficient.

        Notes:
            The aerodynamic resistance is given by

            .. math::
                r_a = \\frac{1}{C_s u_{\\text{eff}}}

            where :math:`C_s` is the drag coefficient for scalars and :math:`u_{\\text{eff}}` is the effective wind speed.
        """
        ueff = jnp.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        return 1.0 / (state.drag_s * ueff)


def compute_effective_wind_speed(u: Array, v: Array, wstar: Array) -> Array:
    """Compute effective wind speed ``ueff``.

    Args:
        u: zonal wind speed :math:`u`.
        v: meridional wind speed :math:`v`.
        wstar: convective velocity scale :math:`w_*`.

    Returns:
        Effective wind speed :math:`u_{\\text{eff}}`.

    Notes:
        The effective wind speed is given by

        .. math::
            u_{\\text{eff}} = \\sqrt{u^2 + v^2 + w_*^2}.

        A minimum value of 0.01 m/s is enforced to avoid division by zero afterwards.
    """
    return jnp.maximum(0.01, jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0))


def compute_surface_properties(
    ueff: Array,
    theta: Array,
    wtheta: Array,
    q: Array,
    surf_pressure: Array,
    rs: Array,
    drag_s: Array,
) -> tuple[Array, Array, Array]:
    """Compute surface temperature, specific humidity, and virtual potential temperature.

    Args:
        ueff: effective wind speed :math:`u_{\\text{eff}}`.
        theta: mixed layer potential temperature :math:`\\theta`.
        wtheta: surface kinematic heat flux :math:`w'\\theta'`.
        q: mixed layer specific humidity :math:`q`.
        surf_pressure: surface pressure :math:`p`.
        rs: surface roughness length :math:`r_s`.
        drag_s: surface drag coefficient :math:`C_s`.

    Notes:
        The surface potential temperature is given by

        .. math::
            \\theta_s = \\theta + \\frac{w'\\theta'}{C_s u_{\\text{eff}}}.

        The surface specific humidity is a weighted average between the air and the saturated value at the surface

        .. math::
            q_s = (1 - c_q) q + c_q q_{sat}(\\theta_s, p),

        where :math:`c_q = [1 + C_s u_{\\text{eff}} r_s]^{-1}` and :math:`q_{sat}` is the saturation specific humidity.

        The surface virtual potential temperature is

        .. math::
            \\theta_{v,s} = \\theta_s (1 + 0.61 q_s)
    """
    thetasurf = theta + wtheta / (drag_s * ueff)
    qsatsurf = compute_qsat(thetasurf, surf_pressure)
    cq = (1.0 + drag_s * ueff * rs) ** -1.0
    qsurf = (1.0 - cq) * q + cq * qsatsurf
    thetavsurf = thetasurf * (1.0 + 0.61 * qsurf)
    return thetasurf, qsurf, thetavsurf


def compute_richardson_number(
    ueff: Array, zsl: Array, g: float, thetav: Array, thetavsurf: Array
) -> Array:
    """Compute bulk Richardson number.

    Args:
        ueff: effective wind speed :math:`u_{\\text{eff}}`.
        zsl: surface layer height :math:`z_{sl}`.
        g: gravity :math:`g`.
        thetav: virtual potential temperature at reference height :math:`\\theta_v`.
        thetavsurf: Surface virtual potential temperature :math:`\\theta_{v,s}`.

    Notes:
        The bulk Richardson number is given by

        .. math::
            Ri_b = \\frac{g}{\\theta_v} \\frac{z_{sl} (\\theta_v - \\theta_{v,s})}{u_{\\text{eff}}^2}.

        The value is capped at 0.2 for numerical stability.
    """
    rib_number = g / thetav * zsl * (thetav - thetavsurf) / ueff**2.0
    return jnp.minimum(rib_number, 0.2)


def compute_rib_function(
    zsl: Array,
    oblen: Array,
    rib_number: Array,
    z0h: Array,
    z0m: Array,
) -> Array:
    """Compute the Richardson number function for iterative solution of Obukhov length.

    Notes:
        This function computes the difference between the bulk Richardson number and its
        Monin-Obukhov similarity theory estimate, used in the Newton-Raphson iteration
        for finding the Obukhov length.

        The function is:

        .. math::
            f(L) = Ri_b
            - \\frac{z_{sl}}{L} \\frac{\\psi_h(z_{sl}/L)
            - \\psi_h(z_{0h}/L)
            + \\ln(z_{sl}/z_{0h})}{[\\psi_m(z_{sl}/L)
            - \\psi_m(z_{0m}/L)
            + \\ln(z_{sl}/z_{0m})]^2}

        where :math:`Ri_b` is the bulk Richardson number, :math:`z_{sl}` is the surface layer height,
        :math:`L` is the Obukhov length, :math:`z_{0h}` and :math:`z_{0m}` are roughness lengths for scalars and momentum,
        and :math:`\\psi_h`, :math:`\\psi_m` are stability correction functions.
    """
    scalar_term = compute_scalar_correction_term(zsl, oblen, z0h)
    momentum_term = compute_momentum_correction_term(zsl, oblen, z0m)

    return rib_number - zsl / oblen * scalar_term / momentum_term**2.0


def ribtol(zsl: Array, rib_number: Array, z0h: Array, z0m: Array):
    """Iteratively solve for the Obukhov length given the Richardson number.

    Notes:
        Uses a Newton-Raphson method to find the Obukhov length :math:`L` such that the Monin-Obukhov
        similarity theory estimate matches the bulk Richardson number.

        The iteration continues until the change in :math:`L` is below a threshold or a maximum value is reached.
    """

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

    # limamau: the Rib is really used three times here?
    # or is there a reason for a rib_function_term to be created?
    def body_fun(carry):
        oblen, _ = carry
        oblen0 = oblen

        # calculate function value at current estimate
        fx = compute_rib_function(zsl, oblen, rib_number, z0h, z0m)

        # finite difference derivative
        oblen_start = oblen - perturbation * oblen
        oblen_end = oblen + perturbation * oblen

        fx_start = compute_rib_function(zsl, oblen_start, rib_number, z0h, z0m)
        fx_end = compute_rib_function(zsl, oblen_end, rib_number, z0h, z0m)

        fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

        # Newton–Raphson update
        oblen_new = oblen - fx / fxdif

        return oblen_new, oblen0

    oblen, _ = jax.lax.while_loop(cond_fun, body_fun, (oblen, oblen0))

    return oblen


# limamau: this should also be breaken down into two methods
def compute_drag_coefficients(
    zsl: Array,
    k: float,
    obukhov_length: Array,
    z0h: Array,
    z0m: Array,
) -> tuple[Array, Array]:
    """Compute drag coefficients for momentum and scalars with stability corrections.

    Args:
        zsl: surface layer height :math:`z_{sl}`.
        k: Von Kármán constant :math:`k`.
        obukhov_length: Obukhov length :math:`L`.
        z0h: roughness length for scalars :math:`z_{0h}`.
        z0m: roughness length for momentum :math:`z_{0m}`.

    Returns:
        Friction velocity, zonal and meridional momentum fluxes.

    Notes:
        The drag coefficients are given by

        .. math::
            C_m &= \\frac{k^2}{[\\psi_m(z_{sl}/L)
            - \\psi_m(z_{0m}/L)
            + \\ln(z_{sl}/z_{0m})]^2}

            C_s &= \\frac{k^2}{[\\psi_m(z_{sl}/L)
            - \\psi_m(z_{0m}/L)
            + \\ln(z_{sl}/z_{0m})] [\\psi_h(z_{sl}/L)
            - \\psi_h(z_{0h}/L)
            + \\ln(z_{sl}/z_{0h})]}

        where :math:`\\psi_m` (see :meth:`compute_momentum_correction_term`) and
        :math:`\\psi_h` (see :meth:`compute_scalar_correction_term`)
        are stability correction functions for momentum and scalars.
    """
    # momentum stability correction
    momentum_correction = compute_momentum_correction_term(zsl, obukhov_length, z0m)

    # scalar stability correction
    scalar_correction = compute_scalar_correction_term(zsl, obukhov_length, z0h)

    # drag coefficients
    drag_m = k**2.0 / momentum_correction**2.0
    drag_s = k**2.0 / (momentum_correction * scalar_correction)
    return drag_m, drag_s


# limamau: this should be broken down into three methods
def compute_momentum_fluxes(
    ueff: Array,
    u: Array,
    v: Array,
    drag_m: Array,
) -> tuple[Array, Array, Array]:
    """Compute surface momentum fluxes and friction velocity.

    Args:
        ueff: effective wind speed :math:`u_{\\text{eff}}`.
        u: zonal wind speed :math:`u`.
        v: meridional wind speed :math:`v`.
        drag_m: drag coefficient for momentum :math:`C_m`.

    Returns:
        Friction velocity, zonal and meridional momentum fluxes.

    Notes:
        The friction velocity :math:`u_*` is given by

        .. math::
            u_* = \\sqrt{C_m} u_{\\text{eff}},

        and the momentum fluxes :math:`\\overline{u'w'}` and :math:`\\overline{v'w'}` are given by

        .. math::
            \\overline{u'w'} = -C_m u_{\\text{eff}} u,

            \\overline{v'w'} = -C_m u_{\\text{eff}} v.
    """
    ustar = jnp.sqrt(drag_m) * ueff
    uw = -drag_m * ueff * u
    vw = -drag_m * ueff * v
    return ustar, uw, vw


# limamau: this should be six or three different methods
def compute_2m_variables(
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
    """Compute 2m diagnostic meteorological variables.

    Notes:
        Computes temperature, humidity, wind, and vapor pressures at 2 meters above the surface,
        applying Monin-Obukhov similarity theory with stability corrections.

        The 2m values are calculated using the surface values, fluxes, and stability correction terms.
    """
    # stability correction terms
    # limamau: this should call the method for scalar correction
    scalar_correction = (
        jnp.log(2.0 / z0h)
        - compute_psih(2.0 / obukhov_length)
        + compute_psih(z0h / obukhov_length)
    )
    # limamau: this should call the method for momentum correction
    momentum_correction = (
        jnp.log(2.0 / z0m)
        - compute_psim(2.0 / obukhov_length)
        + compute_psim(z0m / obukhov_length)
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
    esat2m = 0.611e3 * jnp.exp(17.2694 * (temp_2m - 273.16) / (temp_2m - 35.86))
    e2m = q2m * surf_pressure / 0.622
    return temp_2m, q2m, u2m, v2m, e2m, esat2m


def compute_scalar_correction_term(z: Array, oblen: Array, z0h: Array) -> Array:
    """Compute scalar stability correction term.

    Args:
        z: height above ground level :math:`z`.
        oblen: Obukhov length :math:`L`.
        z0h: roughness length for heat :math:`z_{0h}`.

    Returns:
        The scalar stability correction.

    Notes:
        This term is used in Monin-Obukhov similarity theory for scalars, and is given by

        .. math::
            \\ln\\left(\\frac{z}{z_{0h}}\\right)
            - \\psi_h\\left(\\frac{z}{L}\\right)
            + \\psi_h\\left(\\frac{z_{0h}}{L}\\right)

        where :math:`\\psi_h` is the stability correction function for scalars (see :func:`compute_psih`).
    """
    log_term = jnp.log(z / z0h)
    upper_stability = compute_psih(z / oblen)
    surface_stability = compute_psih(z0h / oblen)
    return log_term - upper_stability + surface_stability


def compute_momentum_correction_term(z: Array, oblen: Array, z0m: Array) -> Array:
    """Compute momentum stability correction term.

    Args:
        z: height above ground level :math:`z`.
        oblen: Obukhov length :math:`L`.
        z0m: roughness length for momentum :math:`z_{0m}`.

    Returns:
        The momentum stability correction.

    Notes:
        This term is used in Monin-Obukhov similarity theory for momentum, and is given by

        .. math::
            \\ln\\left(\\frac{z}{z_{0m}}\\right)
            - \\psi_m\\left(\\frac{z}{L}\\right)
            + \\psi_m\\left(\\frac{z_{0m}}{L}\\right)

        where :math:`\\psi_m` is the stability correction function for momentum (see :func:`compute_psim`).
    """
    log_term = jnp.log(z / z0m)
    upper_stability = compute_psim(z / oblen)
    surface_stability = compute_psim(z0m / oblen)
    return log_term - upper_stability + surface_stability


def compute_psim(zeta: Array) -> Array:
    """Compute momentum stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L :math:`\\zeta`.

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


def compute_psih(zeta: Array) -> Array:
    """Compute scalar stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L :math:`\\zeta`.

    Returns:
        The scalar stability correction.

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
