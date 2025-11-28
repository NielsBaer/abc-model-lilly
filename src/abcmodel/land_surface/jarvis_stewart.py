import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..utils import PhysicalConstants
from .standard import (
    AbstractStandardLandSurfaceModel,
    StandardLandSurfaceInitConds,
)


class JarvisStewartInitConds(StandardLandSurfaceInitConds):
    """Jarvis-Stewart model initial state."""

    pass


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    """Jarvis-Stewart land surface model with empirical surface resistance.

    Implementation of the Jarvis-Stewart approach for calculating surface resistance
    based on environmental stress factors. Uses multiplicative stress functions
    for radiation, soil moisture, vapor pressure deficit, and temperature effects
    on stomatal conductance.

    1. Inherit all standard land surface processes from parent class.
    2. Calculate surface resistance using four environmental stress factors.
    3. Apply Jarvis-Stewart multiplicative stress function approach.
    4. No CO2 flux calculations (simple implementation).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ):
        """Update the surface resistance ``rs`` in the state using the Jarvis-Stewart model.

        Notes:
            The surface (stomatal) resistance is calculated as

            .. math::
                r_s = \\frac{r_{s,\\min}}{LAI} f_1 f_2 f_3 f_4

            where :math:`r_{s,\\min}` is the minimum surface resistance,
            :math:`LAI` is the leaf area index,
            and :math:`f_1`, :math:`f_2`, :math:`f_3`, and :math:`f_4` are scaling factors that account for the effects of

            - radiation (:meth:`compute_f1`),
            - soil moisture (:meth:`compute_f2`),
            - vapor pressure deficit (:meth:`compute_f3`), and
            - temperature (:meth:`compute_f4`), respectively.

            This approach follows the Jarvis-Stewart multiplicative model for stomatal conductance.

        References:
            Stewart, J.B. (1988). Modelling surface conductance of pine forest. Agricultural and Forest Meteorology, 43(1), 19-35.

            Jarvis, P.G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in
            canopies in the field. Philosophical Transactions of the Royal Society of London. B, Biological Sciences, 273(927), 593-610.
        """
        f1 = self.compute_f1(state)
        f2 = self.compute_f2(state)
        f3 = self.compute_f3(state)
        f4 = self.compute_f4(state)

        state.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

        return state

    def compute_f1(self, state: PyTree) -> Array:
        """Compute the radiation-dependent scaling factor ``f1`` for surface resistance.

        Notes:
            This factor accounts for the effect of incoming shortwave radiation on stomatal conductance,
            given by

            .. math::
                f_1 = \\frac{1}{\\min\\left(1, \\frac{0.004 S + 0.05}{0.81 (0.004 S + 1)}\\right)},

            where :math:`S` is the incoming shortwave radiation.

        References:
            Equation 9.27 in the CLASS book.
        """
        ratio = (0.004 * state.in_srad + 0.05) / (0.81 * (0.004 * state.in_srad + 1.0))
        f1 = 1.0 / jnp.minimum(1.0, ratio)
        return f1

    def compute_f2(self, state: PyTree) -> Array:
        """Compute the soil moisture-dependent scaling factor ``f2`` for surface resistance.

        Notes:
            This factor accounts for the effect of soil moisture in the second layer on stomatal conductance,
            given by

            .. math::
                f_2 =
                    \\begin{cases}
                        \\frac{w_{fc} - w_{wilt}}{w_2 - w_{wilt}}, & \\text{if } w_2 > w_{wilt} \\\\
                        10^8, & \\text{otherwise},
                    \\end{cases}

            where :math:`w_2` is the soil moisture in the second layer,
            :math:`w_{fc}` is soil moisture at field capacity, and
            :math:`w_{wilt}` is soil moisture at the wilting point.

            To avoid unrealistically low values when :math:`w_2 > w_{fc}`, the minimum value of :math:`f_2` is limited to 1.

        References:
            Equation 9.28 in the CLASS book.
        """
        f2 = jnp.where(
            self.w2 > self.wwilt,
            (self.wfc - self.wwilt) / (self.w2 - self.wwilt),
            1.0e8,
        )
        assert isinstance(f2, jnp.ndarray)
        f2 = jnp.maximum(f2, 1.0)
        return f2

    def compute_f3(self, state: PyTree) -> Array:
        """Compute the vapor pressure deficit-dependent scaling factor ``f3`` for surface resistance.

        Notes:
            This factor accounts for the effect of vapor pressure deficit (VPD) on stomatal conductance,
            given by

            .. math::
                f_3 = \\exp\\left(\\gamma_D \\frac{e_{sat} - e}{100}\\right)^{-1},

            where :math:`e_{sat}` is the saturation vapor pressure,
            :math:`e` is the actual vapor pressure, and
            :math:`\\gamma_D` is an empirical parameter.

        References:
            Equation 9.29 in the CLASS book.
        """
        vpd = state.esat - state.e
        f3 = 1.0 / jnp.exp(-self.gD * vpd / 100.0)
        return f3

    def compute_f4(self, state: PyTree) -> Array:
        """Compute the temperature-dependent scaling factor ``f4`` for surface resistance.

        Notes:
            This factor accounts for the effect of temperature on stomatal conductance,
            given by

            .. math::
                f_4 = \\frac{1}{1 - 0.0016 (298 - \\theta)^2},

            where :math:`\\theta` is the air temperature in Kelvin.

        References:
            Equation 9.30 in the CLASS book.
        """
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - state.theta) ** 2.0)
        return f4

    def update_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ):
        """No CO2 flux is computed using this model. See :class:`~abcmodel.land_surface.aquacrop.AquaCropModel`."""
        return state
