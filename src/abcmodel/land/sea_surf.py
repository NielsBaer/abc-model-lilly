from dataclasses import dataclass, field, replace

from ..utils import PhysicalConstants as cst
import jax.numpy as jnp
from jax import Array

from ..abstracts import AbstractCoupledState, AbstractLandModel, AbstractLandState
from ..utils import compute_esat, compute_qsat


# limamau: this could be much simpler!
@dataclass
class SeaSurfaceState(AbstractLandState):
    """State dataclass for the sea surface"""

    surf_temp: Array
    """Surface temperature [K]."""
    alpha: Array = field(default_factory=lambda: jnp.array(0.0))
    """surface albedo [-], range 0 to 1."""
    rs: Array = field(default_factory=lambda: jnp.array(1.0))
    """No additional surface resistance [s m-1]."""
    wg: Array = field(default_factory=lambda: jnp.array(0.0))
    """No moisture content in the root zone [m3 m-3]."""
    wl: Array = field(default_factory=lambda: jnp.array(0.0))
    """No water content in the canopy [m]."""

    # the following variables are assigned during warmup/timestep
    esat: Array = field(default_factory=lambda: jnp.array(0.0))
    """Saturation vapor pressure [Pa]."""
    qsat: Array = field(default_factory=lambda: jnp.array(0.0))
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array = field(default_factory=lambda: jnp.array(0.0))
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array = field(default_factory=lambda: jnp.array(0.0))
    """Vapor pressure [Pa]."""
    qsatsurf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic heat flux [K m/s]."""
    wq: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""
    le: Array = field(default_factory=lambda: jnp.array(0.0))
    """Total latent heat flux [W m-2]."""
    hf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Sensible heat flux [W m-2]."""


# alias
class SeaSurfaceModel(AbstractLandModel):
    """Minimal land surface model with fixed surface properties."""

    def __init__(self
                 ):
        self.d1 = 0.0

    def init_state(
        self,
        alpha: float,
        surf_temp: float,
        wg: float = 0.0,
        wl: float = 0.0,
        wtheta: float = 0.0,
    ) -> SeaSurfaceState:
        """Initialize the model state.

        Args:
            alpha: surface albedo [-], range 0 to 1.
            surf_temp: Surface temperature [K].
            rs: Surface resistance [s m-1].
            wg: Volumetric soil moisture [m3 m-3].
            wl: Canopy water content [m].
            wtheta: Kinematic heat flux [K m/s].
            sst: Sea Surface temperature [K].

        Returns:
            The initial land state.
        """
        return SeaSurfaceState(
            alpha=jnp.array(alpha),
            surf_temp=jnp.array(surf_temp),
            wg=jnp.array(wg),
            wl=jnp.array(wl),
            wtheta=jnp.array(wtheta),
        )

    def run(
        self,
        state: AbstractCoupledState,
    ) -> SeaSurfaceState:
        """Run the model.

        Args:
            state: CoupledState.

        Returns:
            The updated land state object.
        """
        land_state = state.land
        atmos = state.atmos
        ra = atmos.ra
        theta = atmos.theta
        q = atmos.q
        qsat = compute_qsat(atmos.theta, atmos.surf_pressure)
        e = self.compute_e(atmos.q, atmos.surf_pressure)
        qsatsurf = compute_qsat(land_state.surf_temp, atmos.surf_pressure)
        le = self.compute_le(q, qsat, ra)
        hf = self.compute_hf(land_state.surf_temp,  theta, ra)
        wtheta = self.compute_wtheta(hf)
        wq = self.compute_wq(le)
        return land_state.replace(
            qsatsurf=qsatsurf,
            le=le,
            hf=hf,
            wtheta=wtheta,
            wq=wq,
            e=e
        )

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return q * surf_pressure / 0.622

    def compute_hf(
        self,
        surf_temp: Array,
        theta: Array,
        ra: Array,
    ) -> Array:
        """Compute the sensible heat flux ``hf``.

        Notes:
            The sensible heat flux is given by

            .. math::

                H = \\frac{\\rho c_p}{r_a} (T_s - \\theta),

            where :math:`\\rho` is the air density, :math:`c_p` is the specific heat capacity of air,
            :math:`r_a` is the aerodynamic resistance, :math:`T_s` is the surface temperature and
            :math:`\\theta` is the mixed layer air potential temperature.

        References:
            Equation 14.5 from the CLASS book, but why are we using :math:`T_s` instead of :math:`\\theta_s`?
            Probably because the variations of pressure are not significant enough.
        """
        return cst.rho * cst.cp / ra * (surf_temp - theta)

    def compute_le(
        self,
        q: Array,
        qsat: Array,
        ra: Array,
    ) -> Array:
        """Compute the latent heat flux over the ocean surface.

        Notes:
            We proceed just like in :meth:`compute_le_veg`, but omitting vegetation's resistance :math:`r_s`,
            with the assumption that water at the leaf is ready to be evaporated, giving us

        .. math::
            LE = \\frac{\\rho L_v}{r_a}(q_{\\text{sat}}(T_s)-⟨q⟩).

        In the end, we scale the result by the fraction of liquid water content :math:`c_{\\text{liq}}`
        and the fraction of vegetation :math:`c_{\\text{veg}}`.

        References:
            Equation 14.6 from the CLASS book.
        """
        le = cst.rho * cst.lv / ra * (qsat - q)
        return le

    def compute_wtheta(self, hf: Array) -> Array:
        """Compute the kinematic heat flux ``wtheta``.

        Notes:
            The kinematic heat flux :math:`\\overline{(w'\\theta')}_s` is directly related to the
            sensible heat flux :math:`H` through

            .. math::
                \\overline{(w'\\theta')}_s = \\frac{H}{\\rho c_p},

            where :math:`\\rho` is the density of air and
            :math:`c_p` is the specific heat capacity of air at constant pressure.
        """
        return hf / (cst.rho * cst.cp)

    def compute_wq(self, le: Array) -> Array:
        """Compute the kinematic moisture flux ``wq``.

        Notes:
            The kinematic moisture flux :math:`\\overline{(w'q')}_s` is directly related to the
            latent heat flux :math:`LE` through

            .. math::
                \\overline{(w'q')}_s = \\frac{LE}{\\rho L_v},

            where :math:`\\rho` is the density of air and
            :math:`L_v` is the latent heat of vaporization.
        """
        return le / (cst.rho * cst.lv)

    def integrate(
        self, state: SeaSurfaceState, dt: float
    ) -> SeaSurfaceState:
        """Integrate the model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        return state
