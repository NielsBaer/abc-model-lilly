from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..abstracts import AbstractLandModel
from ..utils import PhysicalConstants, compute_esat, compute_qsat


@dataclass
class StandardLandSurfaceInitConds:
    """Standard land surface model initial state."""

    alpha: float
    """Slope of the light response curve [mol J-1]."""
    wg: float
    """Soil moisture content in the root zone [m3 m-3]."""
    temp_soil: float
    """Soil temperature [K]."""
    temp2: float
    """Deep soil temperature [K]."""
    surf_temp: float
    """Surface temperature [K]."""
    wl: float
    """Liquid water storage on the canopy [m]."""

    rs: float = 1.0e6
    """Surface resistance [m s-1]."""
    rssoil: float = 1.0e6
    """Soil resistance [m s-1]."""

    cliq: float = jnp.nan
    """Wet fraction of the canopy [-]."""
    temp_soil_tend: float = jnp.nan
    """Soil temperature tendency [K s-1]."""
    wgtend: float = jnp.nan
    """Soil moisture tendency [m3 m-3 s-1]."""
    wltend: float = jnp.nan
    """Canopy water storage tendency [m s-1]."""
    le_veg: float = jnp.nan
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: float = jnp.nan
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: float = jnp.nan
    """Latent heat flux from soil [W m-2]."""
    le: float = jnp.nan
    """Total latent heat flux [W m-2]."""
    hf: float = jnp.nan
    """Sensible heat flux [W m-2]."""
    gf: float = jnp.nan
    """Ground heat flux [W m-2]."""
    le_pot: float = jnp.nan
    """Potential latent heat flux [W m-2]."""
    le_ref: float = jnp.nan
    """Reference latent heat flux [W m-2]."""
    ra: float = jnp.nan
    """Aerodynamic resistance [s m-1]."""
    esat: float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    qsat: float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: float = jnp.nan
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: float = jnp.nan
    """Vapor pressure [Pa]."""
    qsatsurf: float = jnp.nan
    """Saturation specific humidity at surface temperature [kg/kg]."""


class AbstractStandardLandSurfaceModel(AbstractLandModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics.

    Args:
        a: Clapp and Hornberger (1978) retention curve parameter.
        b: Clapp and Hornberger (1978) retention curve parameter.
        p: Clapp and Hornberger (1978) retention curve parameter.
        cgsat: saturated soil heat capacity [J m-3 K-1].
        wsat: saturated soil moisture content [m3 m-3].
        wfc: soil moisture content at field capacity [m3 m-3].
        wwilt: soil moisture content at wilting point [m3 m-3].
        w2: soil moisture content in the deep layer [m3 m-3].
        d1: depth of the shallow layer [m].
        c1sat: saturated soil conductivity parameter [-].
        c2ref: reference soil conductivity parameter [-].
        lai: leaf area index [m2 m-2].
        gD: canopy radiation extinction coefficient [-].
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

    def integrate(self, state: PyTree, dt: float):
        """Integrate model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        state.temp_soil += dt * state.temp_soil_tend
        state.wg += dt * state.wgtend
        state.wl += dt * state.wltend

        return state

    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ):
        """Run the full land surface model for one time step.

        Args:
            state: the state object carrying all variables.
            const: the physical constants object.

        Returns:
            The updated state object.
        """
        # compute aerodynamic resistance from state
        ueff = jnp.sqrt(state.u**2.0 + state.v**2.0 + state.wstar**2.0)
        state.ra = ueff / jnp.maximum(1.0e-3, state.ustar) ** 2.0
        
        state.esat = compute_esat(state.theta)
        state.qsat = compute_qsat(state.theta, state.surf_pressure)
        state.dqsatdT = self.compute_dqsatdT(state)
        state.e = self.compute_e(state)
        state = self.update_surface_resistance(state, const)
        state = self.update_co2_flux(state, const)
        state.rssoil = self.compute_soil_resistance(state)
        state.cliq = self.compute_cliq(state)
        state.surf_temp = self.compute_skin_temperature(state, const)
        state.qsatsurf = compute_qsat(state.surf_temp, state.surf_pressure)
        state.le_veg = self.compute_le_veg(state, const)
        state.le_liq = self.compute_le_liq(state, const)
        state.le_soil = self.compute_le_soil(state, const)
        state.wltend = self.compute_wltend(state, const)
        state.le = self.compute_le(state)
        state.hf = self.compute_hf(state, const)
        state.gf = self.compute_gf(state)
        state.le_pot = self.compute_le_pot(state, const)
        state.le_ref = self.compute_le_ref(state, const)
        state.temp_soil_tend = self.compute_temp_soil_tend(state)
        state.wgtend = self.compute_wgtend(state, const)
        state.wtheta = self.compute_wtheta(state, const)
        state.wq = self.compute_wq(state, const)
        return state

    def compute_dqsatdT(self, state: PyTree) -> Array:
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
        num = 17.2694 * (state.theta - 273.16)
        den = (state.theta - 35.86) ** 2.0
        mult = num / den
        desatdT = state.esat * mult
        return 0.622 * desatdT / state.surf_pressure

    def compute_e(self, state: PyTree) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return state.q * state.surf_pressure / 0.622

    @abstractmethod
    def update_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Abstract method to update surface resistance."""
        raise NotImplementedError

    @abstractmethod
    def update_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Abstract method to update CO2 flux."""
        raise NotImplementedError

    def compute_soil_resistance(self, state: PyTree) -> Array:
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
            state.wg > self.wwilt,
            (self.wfc - self.wwilt) / (state.wg - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, Array)
        return self.rssoilmin * f2

    def compute_cliq(self, state: PyTree) -> Array:
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
        return jnp.minimum(1.0, state.wl / wlmx)

    def compute_skin_temperature(
        self, state: PyTree, const: PhysicalConstants
    ) -> Array:
        """Compute the skin temperature ``surf_temp``.

        Notes:
            The skin temperature is obtained by solving the surface energy balance

            .. math::
                R_n = H + LE_{\\text{veg}} + LE_{\\text{liq}} + LE_{\\text{soil}} + G

            where :math:`R_n` is the net radiation,
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
            state.net_rad
            + const.rho * const.cp / state.ra * state.theta
            + self.cveg
            * (1.0 - state.cliq)
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + self.cveg
            * state.cliq
            * const.rho
            * const.lv
            / state.ra
            * (state.dqsatdT * state.theta - state.qsat + state.q)
            + self.lam * state.temp_soil
        ) / (
            const.rho * const.cp / state.ra
            + self.cveg
            * (1.0 - state.cliq)
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * state.dqsatdT
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * state.dqsatdT
            + self.cveg * state.cliq * const.rho * const.lv / state.ra * state.dqsatdT
            + self.lam
        )

    def compute_le_veg(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        term = state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q
        le_veg = const.rho * const.lv / (state.ra + state.rs) * term
        frac = (1.0 - state.cliq) * self.cveg
        return frac * le_veg

    def compute_le_liq(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        term = state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q
        le_liq = const.rho * const.lv / state.ra * term
        frac = state.cliq * self.cveg
        return frac * le_liq

    def compute_le_soil(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        term = state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q
        le_soil = const.rho * const.lv / (state.ra + state.rssoil) * term
        frac = 1.0 - self.cveg
        return frac * le_soil

    def compute_wltend(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        return -state.le_liq / (const.rhow * const.lv)

    def compute_le(self, state: PyTree) -> Array:
        """Compute the evapotranspiration (latent heat flux) ``le``.

        Notes:
            The sum of

            - transpiration from vegetation in :meth:`compute_le_veg`;
            - evaporation from bare soil in :meth:`compute_le_soil`;
            - evaporation from wet leaves (dew) in :meth:`compute_le_liq`.
        """
        return state.le_soil + state.le_veg + state.le_liq

    def compute_hf(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        return const.rho * const.cp / state.ra * (state.surf_temp - state.theta)

    def compute_gf(self, state: PyTree) -> Array:
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
        return self.lam * (state.surf_temp - state.temp_soil)

    def compute_le_pot(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        rad_term = state.dqsatdT * (state.net_rad - state.gf)
        aerodynamic_term = const.rho * const.cp / state.ra * (state.qsat - state.q)
        denominator = state.dqsatdT + const.cp / const.lv
        return (rad_term + aerodynamic_term) / denominator

    def compute_le_ref(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        rad_term = state.dqsatdT * (state.net_rad - state.gf)
        aerodynamic_term = const.rho * const.cp / state.ra * (state.qsat - state.q)
        den1 = state.dqsatdT
        den2 = const.cp / const.lv * (1.0 + self.rsmin / self.lai / state.ra)
        return (rad_term + aerodynamic_term) / (den1 + den2)

    def compute_temp_soil_tend(self, state: PyTree) -> Array:
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
        return cg * state.gf - 2.0 * jnp.pi / 86400.0 * (state.temp_soil - state.temp2)

    def compute_wgtend(self, state: PyTree, const: PhysicalConstants) -> Array:
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
        c1 = self.c1sat * (self.wsat / state.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        evap_loss = -c1 / (const.rhow * self.d1) * state.le_soil / const.lv
        deep_grad = c2 / 86400.0 * (state.wg - wgeq)
        return evap_loss + deep_grad

    def compute_wtheta(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the kinematic heat flux ``wtheta``.

        Notes:
            The kinematic heat flux :math:`\\overline{(w'\\theta')}_s` is directly related to the
            sensible heat flux :math:`H` through

            .. math::
                \\overline{(w'\\theta')}_s = \\frac{H}{\\rho c_p},

            where :math:`\\rho` is the density of air and
            :math:`c_p` is the specific heat capacity of air at constant pressure.
        """
        return state.hf / (const.rho * const.cp)

    def compute_wq(self, state: PyTree, const: PhysicalConstants) -> Array:
        """Compute the kinematic moisture flux ``wq``.

        Notes:
            The kinematic moisture flux :math:`\\overline{(w'q')}_s` is directly related to the
            latent heat flux :math:`LE` through

            .. math::
                \\overline{(w'q')}_s = \\frac{LE}{\\rho L_v},

            where :math:`\\rho` is the density of air and
            :math:`L_v` is the latent heat of vaporization.
        """
        return state.le / (const.rho * const.lv)
