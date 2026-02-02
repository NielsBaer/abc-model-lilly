from abc import abstractmethod
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ..abstracts import (
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
)
from ..utils import PhysicalConstants as cst
from ..utils import compute_esat, compute_qsat


@dataclass
class StandardLandState(AbstractLandState):
    """Standard land surface model state."""

    alpha: Array
    """Slope of the light response curve [mol J-1]."""
    wg: Array
    """Soil moisture content in the root zone [m3 m-3]."""
    temp_soil: Array
    """Soil temperature [K]."""
    temp2: Array
    """Deep soil temperature [K]."""
    surf_temp: Array
    """Surface temperature [K]."""
    wl: Array
    """No water content in the canopy [m]."""
    wq: Array
    """Kinematic moisture flux [kg/kg m/s]."""
    wtheta: Array
    """Kinematic heat flux [K m/s]."""

    rs: Array = field(default_factory=lambda: jnp.array(1.0e6))
    """Surface resistance [m s-1]."""
    rssoil: Array = field(default_factory=lambda: jnp.array(1.0e6))
    """Soil resistance [m s-1]."""

    # the following variables should be initialized to nan
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
    wCO2: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""
    cliq: Array = field(default_factory=lambda: jnp.array(0.0))
    """Wet fraction of the canopy [-]."""
    temp_soil_tend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Soil temperature tendency [K s-1]."""
    wgtend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Soil moisture tendency [m3 m-3 s-1]."""
    wltend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Canopy water storage tendency [m s-1]."""
    le_veg: Array = field(default_factory=lambda: jnp.array(0.0))
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: Array = field(default_factory=lambda: jnp.array(0.0))
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: Array = field(default_factory=lambda: jnp.array(0.0))
    """Latent heat flux from soil [W m-2]."""
    le: Array = field(default_factory=lambda: jnp.array(0.0))
    """Total latent heat flux [W m-2]."""
    hf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Sensible heat flux [W m-2]."""
    gf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Ground heat flux [W m-2]."""
    le_pot: Array = field(default_factory=lambda: jnp.array(0.0))
    """Potential latent heat flux [W m-2]."""
    le_ref: Array = field(default_factory=lambda: jnp.array(0.0))
    """Reference latent heat flux [W m-2]."""
    wtheta: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic heat flux [K m/s]."""
    wq: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array = field(default_factory=lambda: jnp.array(0.0))
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


class AbstractStandardLandModel(AbstractLandModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics.

    Args:
        a: Clapp and Hornberger (1978) retention curve parameter.
        b: Clapp and Hornberger (1978) retention curve parameter.
        p: Clapp and Hornberger (1978) retention curve parameter.
        cgsat: saturated soil heat capacity [J m-3 K-1].
        wsat: saturated soil moisture content [m3 m-3].
        wfc: soil moisture content at field capacity [m3 m-3].
        wwilt: soil moisture content at wilting point [m3 m-3].
        w2: soil moisture content at the second layer [m3 m-3].
        d1: depth of the top soil layer [m].
        c1sat: saturated soil hydraulic conductivity parameter [-].
        c2ref: reference soil hydraulic conductivity parameter [-].
        lai: leaf area index [m2 m-2].
        gD: canopy rad extinction coefficient [-].
        rsmin: minimum stomatal resistance [s m-1].
        rssoilmin: minimum soil resistance [s m-1].
        cveg: vegetation fraction [-].
        wmax: maximum water storage capacity of the canopy [m].
        lam: thermal diffusivity of the soil [W m-1 K-1].
    """

    def __init__(
        self,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        w2: float,
        d1: float,
        c1sat: float,
        c2ref: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        cveg: float,
        wmax: float,
        lam: float,
    ):
        self.a = a
        self.b = b
        self.p = p
        self.cgsat = cgsat
        self.wsat = wsat
        self.wfc = wfc
        self.wwilt = wwilt
        self.w2 = w2
        self.d1 = d1
        self.c1sat = c1sat
        self.c2ref = c2ref
        self.lai = lai
        self.gD = gD
        self.rsmin = rsmin
        self.rssoilmin = rssoilmin
        self.cveg = cveg
        self.wmax = wmax
        self.lam = lam
        self.c_beta = 0.0

    def init_state(
        self,
        alpha: float,
        wg: float,
        temp_soil: float,
        temp2: float,
        surf_temp: float,
        wl: float,
        wq: float,
        wtheta: float,
        rs: float = 1.0e6,
        rssoil: float = 1.0e6,
    ) -> StandardLandState:
        """Initialize the model state.

        Args:
            alpha: albedo [-].
            wg: Volumetric soil moisture [m3 m-3].
            temp_soil: Soil temperature [K].
            temp2: Deep soil temperature [K].
            surf_temp: Surface temperature [K].
            wl: Canopy water content [m].
            wq: Kinematic moisture flux [kg/kg m/s].
            wtheta: Kinematic heat flux [K m/s].
            rs: Surface resistance [s m-1].
            rssoil: Soil resistance [s m-1].

        Returns:
            The initial land state.
        """
        return StandardLandState(
            alpha=jnp.array(alpha),
            wg=jnp.array(wg),
            temp_soil=jnp.array(temp_soil),
            temp2=jnp.array(temp2),
            surf_temp=jnp.array(surf_temp),
            wl=jnp.array(wl),
            wq=jnp.array(wq),
            wtheta=jnp.array(wtheta),
            rs=jnp.array(rs),
            rssoil=jnp.array(rssoil),
        )

    def integrate(self, state: StandardLandState, dt: float) -> StandardLandState:
        """Integrate model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        temp_soil = state.temp_soil + dt * state.temp_soil_tend
        wg = state.wg + dt * state.wgtend
        wl = state.wl + dt * state.wltend

        return replace(state, temp_soil=temp_soil, wg=wg, wl=wl)

    def run(
        self,
        state: AbstractCoupledState,
    ) -> StandardLandState:
        """Run the full land surface model for one time step.

        Args:
            state: CoupledState.

        Returns:
            The updated land state object.
        """
        land_state = state.land
        atmos = state.atmos
        ra = atmos.ra
        esat = compute_esat(atmos.theta)
        qsat = compute_qsat(atmos.theta, atmos.surf_pressure)
        dqsatdT = self.compute_dqsatdT(esat, atmos.theta, atmos.surf_pressure)
        e = self.compute_e(atmos.q, atmos.surf_pressure)
        land_state = land_state.replace(esat=esat, qsat=qsat, dqsatdT=dqsatdT, e=e)
        state = state.replace(land=land_state)
        state = self.update_surface_resistance(state)
        state = self.update_co2_flux(state)
        land_state = state.land
        rssoil = self.compute_soil_resistance(land_state.wg)
        cliq = self.compute_cliq(land_state.wl)
        surf_temp = self.compute_skin_temperature(
            state.net_rad,
            atmos.theta,
            atmos.q,
            land_state.qsat,
            land_state.dqsatdT,
            ra,
            land_state.rs,
            rssoil,
            cliq,
            land_state.temp_soil,
        )
        qsatsurf = compute_qsat(surf_temp, atmos.surf_pressure)
        le_veg = self.compute_le_veg(
            surf_temp,
            atmos.theta,
            atmos.q,
            land_state.qsat,
            land_state.dqsatdT,
            ra,
            land_state.rs,
            cliq,
        )
        le_liq = self.compute_le_liq(
            surf_temp,
            atmos.theta,
            atmos.q,
            land_state.qsat,
            land_state.dqsatdT,
            ra,
            cliq,
        )
        le_soil = self.compute_le_soil(
            surf_temp,
            atmos.theta,
            atmos.q,
            land_state.qsat,
            land_state.dqsatdT,
            ra,
            rssoil,
        )
        wltend = self.compute_wltend(le_liq)
        le = self.compute_le(le_soil, le_veg, le_liq)
        hf = self.compute_hf(surf_temp, atmos.theta, ra)
        gf = self.compute_gf(surf_temp, land_state.temp_soil)
        le_pot = self.compute_le_pot(
            state.net_rad,
            gf,
            land_state.dqsatdT,
            land_state.qsat,
            atmos.q,
            ra,
        )
        le_ref = self.compute_le_ref(
            state.net_rad,
            gf,
            land_state.dqsatdT,
            land_state.qsat,
            atmos.q,
            ra,
        )
        temp_soil_tend = self.compute_temp_soil_tend(
            gf, land_state.temp_soil, land_state.temp2
        )
        wgtend = self.compute_wgtend(land_state.wg, le_soil)

        wtheta = self.compute_wtheta(hf)
        wq = self.compute_wq(le)
        return land_state.replace(
            rssoil=rssoil,
            cliq=cliq,
            surf_temp=surf_temp,
            qsatsurf=qsatsurf,
            le_veg=le_veg,
            le_liq=le_liq,
            le_soil=le_soil,
            wltend=wltend,
            le=le,
            hf=hf,
            gf=gf,
            le_pot=le_pot,
            le_ref=le_ref,
            temp_soil_tend=temp_soil_tend,
            wgtend=wgtend,
            wtheta=wtheta,
            wq=wq,
        )

    def compute_dqsatdT(self, esat: Array, theta: float, surf_pressure: float) -> Array:
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :meth:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :meth:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = esat * mult
        return 0.622 * desatdT / surf_pressure

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

    @abstractmethod
    def update_surface_resistance(
        self,
        state: AbstractCoupledState,
    ) -> AbstractCoupledState:
        """Abstract method to update surface resistance."""
        raise NotImplementedError

    @abstractmethod
    def update_co2_flux(
        self,
        state: AbstractCoupledState,
    ) -> AbstractCoupledState:
        """Abstract method to update CO2 flux."""
        raise NotImplementedError

    def compute_soil_resistance(self, wg: Array) -> Array:
        """Compute the soil resistance ``rssoil``.

        Notes:
            The soil resistance is calculated as

            .. math::
                r_\\text{soil} = r_\\text{soil,min} \\cdot f_2,

            where the parameter :math:`r_\\text{soil,min}` is the minimum surface resistance and
            the correction function :math:`f_2` is given by

            .. math::
                f_2 =
                    \\begin{cases}
                        \\frac{w_\\text{fc} - w_\\text{wilt}}{w_g - w_\\text{wilt}}, & \\text{if } w_g > w_\\text{wilt} \\\\
                        10^8, & \\text{otherwise},
                    \\end{cases}

            where the model parameters :math:`w_\\text{fc}` and :math:`w_\\text{wilt}`
            are the field capacity and wilting point, respectively,
            and the variable :math:`w_g` is the soil water content.

        References:
            Equations 9.28 and 9.31 from the CLASS book.
        """
        f2 = jnp.where(
            wg > self.wwilt,
            (self.wfc - self.wwilt) / (wg - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, Array)
        return self.rssoilmin * f2

    def compute_cliq(self, wl: Array) -> Array:
        """Compute the wet fraction ``cliq``.

        Notes:
            The wet fraction is defined as

            .. math::
                c_{\\text{liq}} = \\frac{W_l}{\\text{LAI}\\cdot W_{\\text{max}}},

            where :math:`W_l` is the water layer depth,
            :math:`\\text{LAI}` is the leaf area index and
            :math:`W_{\\text{max}}` is the thickness of the water layer on wet vegetation.
            In case :math:`W_l > \\text{LAI}\\cdot W_{\\text{max}}`, the wet fraction is set to 1.

        References:
            Equation 9.19 from the CLASS book.
        """
        wlmx = self.lai * self.wmax
        return jnp.minimum(1.0, wl / wlmx)

    def compute_skin_temperature(
        self,
        net_rad: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rs: Array,
        rssoil: Array,
        cliq: Array,
        temp_soil: Array,
    ) -> Array:
        """Compute the skin temperature ``surf_temp``.

        Notes:
            The skin temperature is obtained by solving the surface energy balance

            .. math::
                R_n = H + LE_{\\text{veg}} + LE_{\\text{liq}} + LE_{\\text{soil}} + G

            where :math:`R_n` is the net rad,
            :math:`H` is the sensible heat flux (see :meth:`compute_hf`),
            :math:`LE_{\\text{veg}}` is the latent heat flux from vegetation (see :meth:`compute_le_veg`),
            :math:`LE_{\\text{liq}}` is the latent heat flux from dew on leaves (see :meth:`compute_le_liq`),
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil (see :meth:`compute_le_soil`)
            and :math:`G` is the ground heat flux (see :meth:`compute_gf`).

            The equation is solved for the skin temperature :math:`T_s`
            by factoring out :math:`T_s` from the above, giving us

            .. math::
                T_s = \\frac{
                    R_n + \\frac{\\rho c_p}{r_a} \\theta
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} (\\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T} \\theta - q_{\\text{sat}} + q)
                    + \\Lambda T_{\\text{soil}}
                }{
                    \\frac{\\rho c_p}{r_a}
                    + c_{\\text{veg}} (1-c_{\\text{liq}}) \\frac{\\rho L_v}{r_a + r_s} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + (1-c_{\\text{veg}}) \\frac{\\rho L_v}{r_a + r_{s,\\text{soil}}} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + c_{\\text{veg}} c_{\\text{liq}} \\frac{\\rho L_v}{r_a} \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}
                    + \\Lambda
                }.

            The terms computed in the equation above in each related function energy flux related method.
            This approach ensures that the computed skin temperature is consistent with the partitioning of
            energy fluxes as calculated by the other methods in this class.
        """
        return (
            net_rad
            + cst.rho * cst.cp / ra * theta
            + self.cveg
            * (1.0 - cliq)
            * cst.rho
            * cst.lv
            / (ra + rs)
            * (dqsatdT * theta - qsat + q)
            + (1.0 - self.cveg)
            * cst.rho
            * cst.lv
            / (ra + rssoil)
            * (dqsatdT * theta - qsat + q)
            + self.cveg * cliq * cst.rho * cst.lv / ra * (dqsatdT * theta - qsat + q)
            + self.lam * temp_soil
        ) / (
            cst.rho * cst.cp / ra
            + self.cveg * (1.0 - cliq) * cst.rho * cst.lv / (ra + rs) * dqsatdT
            + (1.0 - self.cveg) * cst.rho * cst.lv / (ra + rssoil) * dqsatdT
            + self.cveg * cliq * cst.rho * cst.lv / ra * dqsatdT
            + self.lam
        )

    def compute_le_veg(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rs: Array,
        cliq: Array,
    ) -> Array:
        """Compute the latent heat flux (transpiration) from vegetation ``le_veg``.

        Notes:
            The latent heat flux is given by

            .. math::
                LE_{\\text{veg}} = \\frac{\\rho L_v}{r_a+r_s}(q_{\\text{sat}}(T_s)-⟨q⟩),

            where :math:`\\rho` is the density of air, :math:`L_v` is the latent heat of vaporization,
            :math:`r_a` is the aerodynamic resistance, :math:`r_s` is the soil resistance,
            :math:`q_{\\text{sat}}(T_s)` is the saturation specific humidity at surface temperature,
            :math:`⟨q⟩` is the specific humidity at the surface.

            :math:`q_{\\text{sat}}(T_s)` has very short time-scales because of the small heat capacity
            (excluding vegetation) of the surface layer and is hard to measure. Consequently,
            we get :math:`q_{\\text{sat}}(T_s)` implicitly using

            .. math::
                q_{\\text{sat}}(T_s) = \\frac{\\text{d}q_{\\text{sat}}}{\\text{d}T}(\\theta_s-\\theta),

            where :math:`\\theta_s` and :math:`\\theta` are the potential temperature of the surface layer and mixed layer, respectively.

            In the end, we scale the latent heat flux by the vegetation cover fraction :math:`c_{\\text{veg}}`
            and the liquid water content :math:`c_{\\text{liq}}` and return

            .. math::
                c_{\\text{veg}}(1-c_{\\text{liq}})LE_{\\text{veg}}.

        References:
            Equation 9.15 from the CLASS book.
        """
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_veg = cst.rho * cst.lv / (ra + rs) * term
        frac = (1.0 - cliq) * self.cveg
        return frac * le_veg

    def compute_le_liq(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        cliq: Array,
    ) -> Array:
        """Compute the latent heat flux on the leaf (dew) ``le_liq``.

        Notes:
            We proceed just like in :meth:`compute_le_veg`, but omitting vegetation's resistance :math:`r_s`,
            with the assumption that water at the leaf is ready to be evaporated, giving us

        .. math::
            LE_{\\text{liq}} = \\frac{\\rho L_v}{r_a}(q_{\\text{sat}}(T_s)-⟨q⟩).

        In the end, we scale the result by the fraction of liquid water content :math:`c_{\\text{liq}}`
        and the fraction of vegetation :math:`c_{\\text{veg}}`.

        References:
            Equation 9.18 from the CLASS book.
        """
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_liq = cst.rho * cst.lv / ra * term
        frac = cliq * self.cveg
        return frac * le_liq

    def compute_le_soil(
        self,
        surf_temp: Array,
        theta: Array,
        q: Array,
        qsat: Array,
        dqsatdT: Array,
        ra: Array,
        rssoil: Array,
    ) -> Array:
        """Compute the latent heat flux on the soil (evaporation) ``le_soil``.

        Notes:
            We proceed just like in :meth:`compute_le_veg`, but instead of considering resistance from
            the vegetation, we consider the resistance from the soil :math:`r_{soil}`, giving us

        .. math::
            LE_{\\text{soil}} = \\frac{\\rho L_v}{r_a + r_{soil}}(q_{\\text{sat}}(T_s)-⟨q⟩)

        In the end, we scale the result by the fraction of soil :math:`c_{\\text{soil}} = 1 - c_{\\text{veg}}`.

        References:
            Equation 9.21 from the CLASS book.
        """
        term = dqsatdT * (surf_temp - theta) + qsat - q
        le_soil = cst.rho * cst.lv / (ra + rssoil) * term
        frac = 1.0 - self.cveg
        return frac * le_soil

    def compute_wltend(self, le_liq: Array) -> Array:
        """Compute the water layer depth tendency tendency ``wltend``.

        Notes:
            The water layer depth tendency is the rate at which water is added to or taken from the vegetation,
            described by

            .. math::
                \\frac{\\text{d} w}{\\text{d} t} = -\\frac{LE_{\\text{liq}}}{\\rho_w L_v},

            where :math:`LE_{\\text{liq}}` is dew, :math:`\\rho_w` is water density and :math:`L_v` is the latent heat of vaporization.

        References:
            Equation 9.20 from the CLASS book, with sign convention.
        """
        return -le_liq / (cst.rhow * cst.lv)

    def compute_le(self, le_soil: Array, le_veg: Array, le_liq: Array) -> Array:
        """Compute the evapotranspiration (latent heat flux) ``le``.

        Notes:
            The sum of

            - transpiration from vegetation in :meth:`compute_le_veg`;
            - evaporation from bare soil in :meth:`compute_le_soil`;
            - evaporation from wet leaves (dew) in :meth:`compute_le_liq`.
        """
        return jnp.clip(le_soil + le_veg + le_liq, 0.0, None)

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
            Equation 9.13 from the CLASS book, but why are we using :math:`T_s` instead of :math:`\\theta_s`?
            Probably because the variations of pressure are not significant enough.
        """
        return cst.rho * cst.cp / ra * (surf_temp - theta)

    def compute_gf(self, surf_temp: Array, temp_soil: Array) -> Array:
        """Compute the ground heat flux ``gf``.

        Notes:
            The ground heat flux is given by

            .. math::

                G = \\Lambda (T_s - T_{soil}),

            where :math:`\\Lambda` is the conductivity of the skin layer,
            :math:`T_s` is the surface temperature and
            :math:`T_{soil}` is the soil temperature.

        References:
            Equation 9.33 from the CLASS book.
        """
        return self.lam * (surf_temp - temp_soil)

    def compute_le_pot(
        self,
        net_rad: Array,
        gf: Array,
        dqsatdT: Array,
        qsat: Array,
        q: Array,
        ra: Array,
    ) -> Array:
        """Compute the potential latent heat flux ``le_pot``.

        Notes:
            The potential latent heat flux is given by

            .. math::

                LE_{\\text{pot}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}
                },

            which is the Penman-Monteith equation assuming no soil resistance.

        References:
            Equation 9.16 from the CLASS book.
        """
        rad_term = dqsatdT * (net_rad - gf)
        aerodynamic_term = cst.rho * cst.cp / ra * (qsat - q)
        denominator = dqsatdT + cst.cp / cst.lv
        return (rad_term + aerodynamic_term) / denominator

    def compute_le_ref(
        self,
        net_rad: Array,
        gf: Array,
        dqsatdT: Array,
        qsat: Array,
        q: Array,
        ra: Array,
    ) -> Array:
        """Compute the reference latent heat flux ``le_ref``.

        Notes:
            The reference latent heat flux is given by

            .. math::

                LE_{\\text{ref}} = \\frac{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} (R_n - G)
                + \\frac{\\rho c_p}{r_a} (q_{\\text{sat}} - q)
                }{
                \\frac{\\text{d}q_{sat}}{\\text{d} T} + \\frac{\\rho c_p}{L_v}(
                1 + \\frac{r_{s,\\text{min}}}{\\text{LAI} \\cdot r_a}
                )
                },

            which is the Penman-Monteith equation assuming that the soil resistance is given by
            :math:`r_{s,\\text{min}} / \\text{LAI}`, i.e., no correction functions are applied.

        References:
            Equation 9.16 from the CLASS book.
        """
        rad_term = dqsatdT * (net_rad - gf)
        aerodynamic_term = cst.rho * cst.cp / ra * (qsat - q)
        den1 = dqsatdT
        den2 = cst.cp / cst.lv * (1.0 + self.rsmin / self.lai / ra)
        return (rad_term + aerodynamic_term) / (den1 + den2)

    def compute_temp_soil_tend(
        self, gf: Array, temp_soil: Array, temp2: Array
    ) -> Array:
        """Compute the soil temperature tendency ``temp_soil_tend``.

        Notes:
            The dynamics of heat transport in the soil is given by

            .. math::
                \\frac{\\text{d}T_s}{\\text{d}t}
                =
                C_T G - \\frac{2\\pi}{\\tau} (T_s - T_2),

            :math:`T_2` is the temperature of the second layer in the soil,
            :math:`T_s` is the soil temperature,
            where :math:`\\tau` is the time constant of one day (86400s),
            :math:`G` is the ground the heat flux and
            and :math:`C_T` is the surface soil/vegetation heat capacity, which can be parametrized as

            .. math::
                C_T = C_{T,\\text{sat}} \\left(\\frac{w_{\\text{sat}}}{w_2}\\right)^{\\frac{b}{2\\log(10)}}

            where :math:`C_{T,\\text{sat}}` is the saturated heat capacity,
            :math:`w_{\\text{sat}}` is the saturation water content,
            :math:`w_2` is the water content at the second layer
            and :math:`b` is a parameter from Clapp and Hornberger (1978).
            I have no idea where this log comes from.

        References:
            Equation 9.32 of the CLASS book.
        """
        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * jnp.log(10.0)))
        return cg * gf - 2.0 * jnp.pi / 86400.0 * (temp_soil - temp2)

    def compute_wgtend(self, wg: Array, le_soil: Array) -> Array:
        """Compute the soil moisture tendency ``wgtend``.

        Notes:
            The dynamics of soil moisture in the top soil layer is described by

            .. math::
                \\frac{\\mathrm{d}w_g}{\\mathrm{d}t}
                =
                -\\frac{C_1}{\\rho_w d_1} \\frac{LE_{\\text{soil}}}{L_v}
                - \\frac{C_2}{\\tau} (w_g - w_{eq}),
                :label: dynamics

            where the coefficients :math:`C_1` and :math:`C_2` are calculated as

            .. math::
                C_1 = C_{1,\\text{sat}} \\left(\\frac{w_{sat}}{w_g}\\right)^{b/2 + 1},
            .. math::
                C_2 = C_{2,\\text{ref}} \\left(\\frac{w_2}{w_{sat} - w_2}\\right),

            where :math:`C_{1,\\text{sat}}` and :math:`C_{2,\\text{sat}}` are parameters from Clapp-Hornberger (1978)
            and the equilibrium soil moisture is given by

            .. math::
                w_{eq} = w_2 - w_{sat} a \\left(\\left(\\frac{w_2}{w_{sat}}\\right)^p
                \\left[1 - \\left(\\frac{w_2}{w_{sat}}\\right)^{8p}\\right]\\right).

            In these equations, :math:`w_g` is the volumetric soil moisture in the top layer,
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil,
            :math:`L_v` is the latent heat of vaporization,
            :math:`\\rho_w` is the density of water,
            :math:`d_1` is the depth of the first soil layer,
            :math:`\\tau` is a time constant (here, 86400 s = 1 day),
            :math:`w_2` is the soil moisture in the second layer,
            :math:`w_{sat}` is the saturated soil moisture, and
            :math:`a`, :math:`b` and :math:`p` are parameters from Clapp-Hornberger (1978).

            In :eq:`dynamics`, the first term represents the loss of soil moisture due to evaporation,
            and the second term represents the relaxation of soil moisture toward equilibrium with the lower layer.

        References:
            - (9.34)–(9.37) in the CLASS book.
            - Clapp, R. B., & Hornberger, G. M. (1978). Empirical equations for some soil hydraulic properties. Water resources research, 14(4), 601-604.

        """
        c1 = self.c1sat * (self.wsat / wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        evap_loss = -c1 / (cst.rhow * self.d1) * le_soil / cst.lv
        deep_grad = c2 / 86400.0 * (wg - wgeq)
        return evap_loss + deep_grad

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
