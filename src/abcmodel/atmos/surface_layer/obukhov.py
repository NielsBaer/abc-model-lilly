from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState, LandT, RadT
from ...utils import PhysicalConstants as cst
from ...utils import compute_qsat
from ..abstracts import (
    AbstractSurfaceLayerModel,
    AbstractSurfaceLayerState,
    CloudT,
    MixedT,
)
from ..dayonly import DayOnlyAtmosphereState


@dataclass
class ObukhovState(AbstractSurfaceLayerState):
    """Standard surface layer model state."""

    ustar: Array
    """Surface friction velocity [m/s]."""
    z0m: Array
    """Roughness length for momentum [m]."""
    z0h: Array
    """Roughness length for scalars [m]."""

    # the following variables are initialized to high values and
    # are expected to converge to realistic values during warmup
    drag_m: Array = field(default_factory=lambda: jnp.array(1e12))
    """Drag coefficient for momentum [-]."""
    drag_s: Array = field(default_factory=lambda: jnp.array(1e12))
    """Drag coefficient for scalars [-]."""

    # the following variables are initialized as NaNs and should
    # and are expected to be assigned during warmup
    uw: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface momentum flux u [m2 s-2]."""
    vw: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface momentum flux v [m2 s-2]."""
    temp_2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m temperature [K]."""
    q2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m specific humidity [kg kg-1]."""
    u2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m u-wind [m s-1]."""
    v2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m v-wind [m s-1]."""
    e2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m vapor pressure [Pa]."""
    esat2m: Array = field(default_factory=lambda: jnp.array(0.0))
    """2m saturated vapor pressure [Pa]."""
    thetasurf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface potential temperature [K]."""
    thetavsurf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface virtual potential temperature [K]."""
    qsurf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface specific humidity [kg kg-1]."""
    obukhov_length: Array = field(default_factory=lambda: jnp.array(0.0))
    """Obukhov length [m]."""
    rib_number: Array = field(default_factory=lambda: jnp.array(0.0))
    """Bulk Richardson number [-]."""
    ra: Array = field(default_factory=lambda: jnp.array(0.0))
    """Aerodynamic resistance [s/m]."""


StateAlias = AbstractCoupledState[
    RadT,
    LandT,
    DayOnlyAtmosphereState[
        ObukhovState,
        MixedT,
        CloudT,
    ],
]


class ObukhovModel(AbstractSurfaceLayerModel):
    """Standard surface layer model with atmospheric stability corrections.

    Calculates surface-atmosphere exchange using Monin-Obukhov similarity theory
    with stability functions and iterative solution for Obukhov length.
    """

    def __init__(self):
        pass

    def init_state(
        self,
        ustar: float,
        z0m: float,
        z0h: float,
    ) -> ObukhovState:
        """Initialize the model state.

        Args:
            ustar: Friction velocity [m s-1].
            z0m: Surface roughness length for momentum [m].
            z0h: Surface roughness length for heat [m].

        Returns:
            The initial surface layer state.
        """
        return ObukhovState(
            ustar=jnp.array(ustar),
            z0m=jnp.array(z0m),
            z0h=jnp.array(z0h),
        )

    def run(self, state: StateAlias):
        """Run the model.

        Args:
            state:

        Returns:
            The updated state.
        """
        atmos = state.atmos
        sl_state = atmos.surface

        ueff = self.compute_effective_wind_speed(atmos.u, atmos.v, atmos.wstar)
        thetasurf = self.compute_thetasurf(
            atmos.theta,
            state.land.wtheta,
            sl_state.drag_s,
            ueff,
        )
        qsurf = self.compute_qsurf(
            atmos.q,
            thetasurf,
            atmos.surf_pressure,
            state.land.rs,
            sl_state.drag_s,
            ueff,
        )
        thetavsurf = self.compute_thetavsurf(thetasurf, qsurf)
        zsl = self.compute_zsl(atmos.h_abl)
        rib_number = self.compute_richardson_number(
            ueff, zsl, cst.g, atmos.thetav, thetavsurf
        )
        obukhov_length = self.ribtol(zsl, rib_number, sl_state.z0h, sl_state.z0m)
        drag_m = self.compute_drag_m(zsl, cst.k, obukhov_length, sl_state.z0m)
        drag_s = self.compute_drag_s(
            zsl, cst.k, obukhov_length, sl_state.z0h, sl_state.z0m
        )
        ustar = self.compute_ustar(ueff, drag_m)
        uw = self.compute_uw(ueff, atmos.u, drag_m)
        vw = self.compute_vw(ueff, atmos.v, drag_m)
        temp_2m = self.compute_temp_2m(
            thetasurf,
            state.land.wtheta,
            ustar,
            cst.k,
            sl_state.z0h,
            obukhov_length,
        )
        q2m = self.compute_q2m(
            qsurf,
            state.land.wq,
            ustar,
            cst.k,
            sl_state.z0h,
            obukhov_length,
        )
        u2m = self.compute_u2m(uw, ustar, cst.k, sl_state.z0m, obukhov_length)
        v2m = self.compute_v2m(vw, ustar, cst.k, sl_state.z0m, obukhov_length)
        e2m = self.compute_e2m(q2m, atmos.surf_pressure)
        esat2m = self.compute_esat2m(temp_2m)
        ra = self.compute_ra(atmos.u, atmos.v, atmos.wstar, drag_s)
        return state.atmos.surface.replace(
            ustar=ustar,
            drag_m=drag_m,
            drag_s=drag_s,
            uw=uw,
            vw=vw,
            temp_2m=temp_2m,
            q2m=q2m,
            u2m=u2m,
            v2m=v2m,
            e2m=e2m,
            esat2m=esat2m,
            thetasurf=thetasurf,
            thetavsurf=thetavsurf,
            qsurf=qsurf,
            obukhov_length=obukhov_length,
            rib_number=rib_number,
            ra=ra,
        )

    def compute_ra(self, u: Array, v: Array, wstar: Array, drag_s: Array) -> Array:
        """Calculate aerodynamic resistance from wind speed and drag coefficient.

        Args:
            u: zonal wind speed [m s-1].
            v: meridional wind speed [m s-1].
            wstar: convective velocity scale [m s-1].
            drag_s: drag coefficient for scalars [-].

        Returns:
            Aerodynamic resistance [s m-1].

        Notes:
            The aerodynamic resistance is given by

            .. math::
                r_a = \\frac{1}{C_s u_{\\text{eff}}}

            where :math:`C_s` is the drag coefficient for scalars and :math:`u_{\\text{eff}}` is the effective wind speed.
        """
        ueff = jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        return 1.0 / (drag_s * ueff)

    def compute_effective_wind_speed(self, u: Array, v: Array, wstar: Array) -> Array:
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

    def compute_zsl(self, h_abl: Array) -> Array:
        """Compute surface layer height.

        Args:
            h_abl: Atmospheric boundary layer height [m].

        Returns:
            Surface layer height [m], defined as 10% of ABL height.

        Notes:
            The surface layer height is conventionally taken as 10% of
            the atmospheric boundary layer height.
        """
        return 0.1 * h_abl

    def compute_thetasurf(
        self,
        theta: Array,
        wtheta: Array,
        drag_s: Array,
        ueff: Array,
    ) -> Array:
        """Compute surface potential temperature.

        Args:
            theta: mixed layer potential temperature :math:`\\theta`.
            wtheta: surface kinematic heat flux :math:`w'\\theta'`.
            drag_s: surface drag coefficient :math:`C_s`.
            ueff: effective wind speed :math:`u_{\\text{eff}}`.

        Returns:
            Surface potential temperature :math:`\\theta_s`.

        Notes:
            The surface potential temperature is given by

            .. math::
                \\theta_s = \\theta + \\frac{w'\\theta'}{C_s u_{\\text{eff}}}.
        """
        return theta + wtheta / (drag_s * ueff)

    def compute_qsurf(
        self,
        q: Array,
        thetasurf: Array,
        surf_pressure: Array,
        rs: Array,
        drag_s: Array,
        ueff: Array,
    ) -> Array:
        """Compute surface specific humidity.

        Args:
            q: mixed layer specific humidity :math:`q`.
            thetasurf: surface potential temperature :math:`\\theta_s`.
            surf_pressure: surface pressure :math:`p`.
            rs: surface resistance :math:`r_s`.
            drag_s: surface drag coefficient :math:`C_s`.
            ueff: effective wind speed :math:`u_{\\text{eff}}`.

        Returns:
            Surface specific humidity :math:`q_s`.

        Notes:
            The surface specific humidity is a weighted average between the air and the saturated value at the surface

            .. math::
                q_s = (1 - c_q) q + c_q q_{sat}(\\theta_s, p),

            where :math:`c_q = [1 + C_s u_{\\text{eff}} r_s]^{-1}` and :math:`q_{sat}` is the saturation specific humidity.
        """
        qsatsurf = compute_qsat(thetasurf, surf_pressure)
        cq = (1.0 + drag_s * ueff * rs) ** -1.0
        return (1.0 - cq) * q + cq * qsatsurf

    def compute_thetavsurf(self, thetasurf: Array, qsurf: Array) -> Array:
        """Compute surface virtual potential temperature.

        Args:
            thetasurf: surface potential temperature :math:`\\theta_s`.
            qsurf: surface specific humidity :math:`q_s`.

        Returns:
            Surface virtual potential temperature :math:`\\theta_{v,s}`.

        Notes:
            The surface virtual potential temperature is

            .. math::
                \\theta_{v,s} = \\theta_s (1 + 0.61 q_s)
        """
        return thetasurf * (1.0 + 0.61 * qsurf)

    def compute_richardson_number(
        self, ueff: Array, zsl: Array, g: float, thetav: Array, thetavsurf: Array
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
        self,
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
        scalar_term = self.compute_scalar_correction_term(zsl, oblen, z0h)
        momentum_term = self.compute_momentum_correction_term(zsl, oblen, z0m)

        return rib_number - zsl / oblen * scalar_term / momentum_term**2.0

    def ribtol(self, zsl: Array, rib_number: Array, z0h: Array, z0m: Array):
        """Iteratively solve for the Obukhov length given the Richardson number.

        Notes:
            Uses a Newton-Raphson method to find the Obukhov length :math:`L` such that the Monin-Obukhov
            similarity theory estimate matches the bulk Richardson number.

            The iteration continues until the change in :math:`L` is below a threshold or a maximum value is reached.
        """

        # initial guess based on stability
        oblen = jnp.where(rib_number > 0.0, 1.0, -1.0)
        oblen0 = jnp.where(rib_number > 0.0, 2.0, -2.0)

        perturbation = 0.001
        max_oblen = 1e4

        def body_fun_scan(carry, _):
            oblen, oblen0 = carry

            # calculate function value at current estimate
            fx = self.compute_rib_function(zsl, oblen, rib_number, z0h, z0m)

            # finite difference derivative
            oblen_start = oblen - perturbation * oblen
            oblen_end = oblen + perturbation * oblen

            fx_start = self.compute_rib_function(zsl, oblen_start, rib_number, z0h, z0m)
            fx_end = self.compute_rib_function(zsl, oblen_end, rib_number, z0h, z0m)

            fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

            # Newton–Raphson update
            # clamp update to avoid exploding values
            update = fx / fxdif
            # simple clamping logic if needed, but bounded iterations help safety
            oblen_new = oblen - update

            # limit range
            oblen_new = jnp.clip(oblen_new, -max_oblen, max_oblen)

            return (oblen_new, oblen), None

        # now we have a fixed number of iterations!
        n_iter = 20
        (oblen, _), _ = jax.lax.scan(
            body_fun_scan, (oblen, oblen0), None, length=n_iter
        )

        return oblen

    def compute_drag_m(
        self,
        zsl: Array,
        k: float,
        obukhov_length: Array,
        z0m: Array,
    ) -> Array:
        """Compute drag coefficient for momentum with stability corrections.

        Args:
            zsl: surface layer height :math:`z_{sl}`.
            k: Von Kármán constant :math:`k`.
            obukhov_length: Obukhov length :math:`L`.
            z0m: roughness length for momentum :math:`z_{0m}`.

        Returns:
            Drag coefficient for momentum :math:`C_m`.

        Notes:
            The drag coefficient for momentum is given by

            .. math::
                C_m = \\frac{k^2}{[\\psi_m(z_{sl}/L)
                - \\psi_m(z_{0m}/L)
                + \\ln(z_{sl}/z_{0m})]^2}

            where :math:`\\psi_m` (see :meth:`compute_momentum_correction_term`)
            is the stability correction function for momentum.
        """
        momentum_correction = self.compute_momentum_correction_term(
            zsl, obukhov_length, z0m
        )
        return k**2.0 / momentum_correction**2.0

    def compute_drag_s(
        self,
        zsl: Array,
        k: float,
        obukhov_length: Array,
        z0h: Array,
        z0m: Array,
    ) -> Array:
        """Compute drag coefficient for scalars with stability corrections.

        Args:
            zsl: surface layer height :math:`z_{sl}`.
            k: Von Kármán constant :math:`k`.
            obukhov_length: Obukhov length :math:`L`.
            z0h: roughness length for scalars :math:`z_{0h}`.
            z0m: roughness length for momentum :math:`z_{0m}`.

        Returns:
            Drag coefficient for scalars :math:`C_s`.

        Notes:
            The drag coefficient for scalars is given by

            .. math::
                C_s = \\frac{k^2}{[\\psi_m(z_{sl}/L)
                - \\psi_m(z_{0m}/L)
                + \\ln(z_{sl}/z_{0m})] [\\psi_h(z_{sl}/L)
                - \\psi_h(z_{0h}/L)
                + \\ln(z_{sl}/z_{0h})]}

            where :math:`\\psi_m` (see :meth:`compute_momentum_correction_term`) and
            :math:`\\psi_h` (see :meth:`compute_scalar_correction_term`)
            are stability correction functions for momentum and scalars.
        """
        momentum_correction = self.compute_momentum_correction_term(
            zsl, obukhov_length, z0m
        )
        scalar_correction = self.compute_scalar_correction_term(
            zsl, obukhov_length, z0h
        )
        return k**2.0 / (momentum_correction * scalar_correction)

    def compute_ustar(self, ueff: Array, drag_m: Array) -> Array:
        """Compute surface friction velocity.

        Args:
            ueff: effective wind speed :math:`u_{\\text{eff}}`.
            drag_m: drag coefficient for momentum :math:`C_m`.

        Returns:
            Friction velocity :math:`u_*`.

        Notes:
            The friction velocity :math:`u_*` is given by

            .. math::
                u_* = \\sqrt{C_m} u_{\\text{eff}}.
        """
        return jnp.sqrt(drag_m) * ueff

    def compute_uw(self, ueff: Array, u: Array, drag_m: Array) -> Array:
        """Compute zonal momentum flux.

        Args:
            ueff: effective wind speed :math:`u_{\\text{eff}}`.
            u: zonal wind speed :math:`u`.
            drag_m: drag coefficient for momentum :math:`C_m`.

        Returns:
            Zonal momentum flux :math:`\\overline{u'w'}`.

        Notes:
            The zonal momentum flux is given by

            .. math::
                \\overline{u'w'} = -C_m u_{\\text{eff}} u.
        """
        return -drag_m * ueff * u

    def compute_vw(self, ueff: Array, v: Array, drag_m: Array) -> Array:
        """Compute meridional momentum flux.

        Args:
            ueff: effective wind speed :math:`u_{\\text{eff}}`.
            v: meridional wind speed :math:`v`.
            drag_m: drag coefficient for momentum :math:`C_m`.

        Returns:
            Meridional momentum flux :math:`\\overline{v'w'}`.

        Notes:
            The meridional momentum flux is given by

            .. math::
                \\overline{v'w'} = -C_m u_{\\text{eff}} v.
        """
        return -drag_m * ueff * v

    def compute_temp_2m(
        self,
        thetasurf: Array,
        wtheta: Array,
        ustar: Array,
        k: float,
        z0h: Array,
        obukhov_length: Array,
    ) -> Array:
        """Compute 2m temperature diagnostic.

        Args:
            thetasurf: surface potential temperature :math:`\\theta_s`.
            wtheta: surface kinematic heat flux :math:`w'\\theta'`.
            ustar: friction velocity :math:`u_*`.
            k: Von Kármán constant :math:`k`.
            z0h: roughness length for scalars :math:`z_{0h}`.
            obukhov_length: Obukhov length :math:`L`.

        Returns:
            Temperature at 2 meters above surface [K].

        Notes:
            Uses Monin-Obukhov similarity theory with stability corrections to extrapolate
            surface temperature to 2m height.
        """
        scalar_correction = self.compute_scalar_correction_term(
            2.0, obukhov_length, z0h
        )
        scalar_scale = 1.0 / (ustar * k)
        return thetasurf - wtheta * scalar_scale * scalar_correction

    def compute_q2m(
        self,
        qsurf: Array,
        wq: Array,
        ustar: Array,
        k: float,
        z0h: Array,
        obukhov_length: Array,
    ) -> Array:
        """Compute 2m specific humidity diagnostic.

        Args:
            qsurf: surface specific humidity :math:`q_s`.
            wq: surface kinematic moisture flux :math:`w'q'`.
            ustar: friction velocity :math:`u_*`.
            k: Von Kármán constant :math:`k`.
            z0h: roughness length for scalars :math:`z_{0h}`.
            obukhov_length: Obukhov length :math:`L`.

        Returns:
            Specific humidity at 2 meters above surface [kg kg-1].

        Notes:
            Uses Monin-Obukhov similarity theory with stability corrections to extrapolate
            surface specific humidity to 2m height.
        """
        scalar_correction = self.compute_scalar_correction_term(
            2.0, obukhov_length, z0h
        )
        scalar_scale = 1.0 / (ustar * k)
        return qsurf - wq * scalar_scale * scalar_correction

    def compute_u2m(
        self,
        uw: Array,
        ustar: Array,
        k: float,
        z0m: Array,
        obukhov_length: Array,
    ) -> Array:
        """Compute 2m zonal wind diagnostic.

        Args:
            uw: zonal momentum flux :math:`\\overline{u'w'}`.
            ustar: friction velocity :math:`u_*`.
            k: Von Kármán constant :math:`k`.
            z0m: roughness length for momentum :math:`z_{0m}`.
            obukhov_length: Obukhov length :math:`L`.

        Returns:
            Zonal wind at 2 meters above surface [m s-1].

        Notes:
            Uses Monin-Obukhov similarity theory with stability corrections to extrapolate
            surface momentum flux to 2m wind speed.
        """
        momentum_correction = self.compute_momentum_correction_term(
            2.0, obukhov_length, z0m
        )
        momentum_scale = 1.0 / (ustar * k)
        return -uw * momentum_scale * momentum_correction

    def compute_v2m(
        self,
        vw: Array,
        ustar: Array,
        k: float,
        z0m: Array,
        obukhov_length: Array,
    ) -> Array:
        """Compute 2m meridional wind diagnostic.

        Args:
            vw: meridional momentum flux :math:`\\overline{v'w'}`.
            ustar: friction velocity :math:`u_*`.
            k: Von Kármán constant :math:`k`.
            z0m: roughness length for momentum :math:`z_{0m}`.
            obukhov_length: Obukhov length :math:`L`.

        Returns:
            Meridional wind at 2 meters above surface [m s-1].

        Notes:
            Uses Monin-Obukhov similarity theory with stability corrections to extrapolate
            surface momentum flux to 2m wind speed.
        """
        momentum_correction = self.compute_momentum_correction_term(
            2.0, obukhov_length, z0m
        )
        momentum_scale = 1.0 / (ustar * k)
        return -vw * momentum_scale * momentum_correction

    def compute_e2m(self, q2m: Array, surf_pressure: Array) -> Array:
        """Compute 2m vapor pressure.

        Args:
            q2m: specific humidity at 2m :math:`q_{2m}`.
            surf_pressure: surface pressure :math:`p`.

        Returns:
            Vapor pressure at 2 meters above surface [Pa].

        Notes:
            Converts specific humidity to vapor pressure using:

            .. math::
                e_{2m} = \\frac{q_{2m} \\cdot p}{0.622}
        """
        return q2m * surf_pressure / 0.622

    def compute_esat2m(self, temp_2m: Array) -> Array:
        """Compute 2m saturated vapor pressure.

        Args:
            temp_2m: temperature at 2m [K].

        Returns:
            Saturated vapor pressure at 2 meters above surface [Pa].

        Notes:
            Uses the Tetens formula:

            .. math::
                e_{sat,2m} = 611 \\exp\\left(\\frac{17.2694(T_{2m} - 273.16)}{T_{2m} - 35.86}\\right)
        """
        return 0.611e3 * jnp.exp(17.2694 * (temp_2m - 273.16) / (temp_2m - 35.86))

    def compute_scalar_correction_term(
        self, z: Array | float, oblen: Array, z0h: Array
    ) -> Array:
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
        upper_stability = self.compute_psih(z / oblen)
        surface_stability = self.compute_psih(z0h / oblen)
        return log_term - upper_stability + surface_stability

    def compute_momentum_correction_term(
        self, z: Array | float, oblen: Array, z0m: Array
    ) -> Array:
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
        upper_stability = self.compute_psim(z / oblen)
        surface_stability = self.compute_psim(z0m / oblen)
        return log_term - upper_stability + surface_stability

    def compute_psim(self, zeta: Array) -> Array:
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

    def compute_psih(self, zeta: Array) -> Array:
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
