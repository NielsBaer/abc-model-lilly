from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState
from ...utils import PhysicalConstants as cst
from ...utils import compute_qsat
from ..abstracts import (
    AbstractMixedLayerModel,
    AbstractMixedLayerState,
)


@dataclass
class BulkState(AbstractMixedLayerState):
    """Data class for bulk mixed layer model state."""

    # initialized by the user
    h_abl: Array
    """Initial atmospheric boundary layer (ABL) height [m]."""
    theta: Array
    """Initial mixed-layer potential temperature [K]."""
    deltatheta: Array
    """Initial temperature jump at the top of the ABL [K]."""
    q: Array
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: Array
    """Initial specific humidity jump at h [kg/kg]."""
    co2: Array
    """Initial mixed-layer CO2 [ppm]."""
    deltaCO2: Array
    """Initial CO2 jump at the top of the ABL [ppm]."""
    wCO2: Array
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: Array
    """Initial mixed-layer u-wind speed [m/s]."""
    du: Array
    """Initial u-wind jump at the top of the ABL [m/s]."""
    v: Array
    """Initial mixed-layer v-wind speed [m/s]."""
    dv: Array
    """Initial v-wind jump at the top of the ABL [m/s]."""
    dz_h: Array
    """Transition layer thickness [m]."""
    surf_pressure: Array
    """Surface pressure, which is actually not updated (not a state), it's only here for simplicity [Pa]."""

    # initialized to zero by default
    wstar: Array = field(default_factory=lambda: jnp.array(0.0))
    """Convective velocity scale [m s-1]."""
    we: Array = field(default_factory=lambda: jnp.array(-1.0))
    """Entrainment velocity [m s-1]."""

    # should be initialized during warmup
    thetav: Array = field(default_factory=lambda: jnp.array(0.0))
    """Mixed-layer potential temperature [K]."""
    deltathetav: Array = field(default_factory=lambda: jnp.array(0.0))
    """Virtual temperature jump at the top of the ABL [K]."""
    wthetav: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface kinematic virtual heat flux [K m s-1]."""
    wqe: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    wCO2e: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment CO2 flux [mgC/m²/s]."""
    wthetae: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment potential temperature flux [K m s-1]."""
    wthetave: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment virtual heat flux [K m s-1]."""
    lcl: Array = field(default_factory=lambda: jnp.array(0.0))
    """Lifting condensation level [m]."""
    top_rh: Array = field(default_factory=lambda: jnp.array(0.0))
    """Top of mixed layer relative humidity [%]."""
    top_p: Array = field(default_factory=lambda: jnp.array(0.0))
    """Pressure at top of mixed layer [Pa]."""
    top_T: Array = field(default_factory=lambda: jnp.array(0.0))
    """Temperature at top of mixed layer [K]."""
    utend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Zonal wind velocity tendency [m s-2]."""
    dutend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Zonal wind velocity tendency at the ABL height [m s-2]."""
    vtend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Meridional wind velocity tendency [m s-2]."""
    dvtend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Meridional wind velocity tendency at the ABL height [m/s²]."""
    h_abl_tend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of CBL [m s-1]."""
    thetatend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of mixed-layer potential temperature [K s-1]."""
    deltathetatend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of mixed-layer potential temperature at the ABL height [K s-1]."""
    qtend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of mixed-layer specific humidity [kg/kg s-1]."""
    dqtend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of mixed-layer specific humidity at the ABL height [kg/kg s-1]."""
    co2tend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of CO2 concentration [ppm s-1]."""
    deltaCO2tend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of CO2 concentration at the ABL height [ppm s-1]."""
    dztend: Array = field(default_factory=lambda: jnp.array(0.0))
    """Tendency of transition layer thickness [m s-1]."""
    ws: Array = field(default_factory=lambda: jnp.array(0.0))
    """Large-scale vertical velocity (subsidence) [m s-1]."""
    wf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Mixed-layer growth due to cloud top radiative divergence [m s-1]."""


class BulkModel(AbstractMixedLayerModel[BulkState]):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    Args:
        divU: horizontal large-scale divergence of wind [s-1].
        coriolis_param: Coriolis parameter [s-1].
        gammatheta: free atmos potential temperature lapse rate [K m-1].
        advtheta: advection of heat [K s-1].
        beta: entrainment ratio for virtual heat [-].
        gammaq: free atmos specific humidity lapse rate [kg/kg m-1].
        advq: advection of moisture [kg/kg s-1].
        gammaCO2: free atmos CO2 lapse rate [ppm m-1].
        advCO2: advection of CO2 [ppm s-1].
        gammau: free atmos u-wind speed lapse rate [s-1].
        advu: advection of u-wind [m s-2].
        gammav: free atmos v-wind speed lapse rate [s-1].
        advv: advection of v-wind [m s-2].
        dFz: cloud top radiative divergence [W m-2].
        is_shear_growing: shear growth mixed-layer switch.
        is_fix_free_trop: fix the free-troposphere switch.
        is_wind_prog: prognostic wind switch.
    """

    def __init__(
        self,
        divU: float,
        coriolis_param: float,
        gammatheta: float,
        advtheta: float,
        beta: float,
        gammaq: float,
        advq: float,
        gammaCO2: float,
        advCO2: float,
        gammau: float,
        advu: float,
        gammav: float,
        advv: float,
        dFz: float,
        is_shear_growing: bool = True,
        is_fix_free_trop: bool = True,
        is_wind_prog: bool = True,
    ):
        self.divU = divU
        self.coriolis_param = coriolis_param
        self.gamma_theta = gammatheta
        self.advtheta = advtheta
        self.beta = beta
        self.gamma_q = gammaq
        self.advq = advq
        self.gammaCO2 = gammaCO2
        self.advCO2 = advCO2
        self.gammau = gammau
        self.advu = advu
        self.gammav = gammav
        self.advv = advv
        self.deltaFz = dFz
        self.is_shear_growing = is_shear_growing
        self.is_fix_free_trop = is_fix_free_trop
        self.is_wind_prog = is_wind_prog

    def init_state(
        self,
        h_abl: float,
        theta: float,
        deltatheta: float,
        q: float,
        dq: float,
        co2: float,
        deltaCO2: float,
        wCO2: float,
        u: float,
        du: float,
        v: float,
        dv: float,
        dz_h: float,
        surf_pressure: float,
    ) -> BulkState:
        """Initialize the model state.

        Args:
            h_abl: atmospheric boundary layer height [m].
            theta: mixed-layer potential temperature [K].
            deltatheta: potential temperature jump at h [K].
            q: mixed-layer specific humidity [kg/kg].
            dq: specific humidity jump at h [kg/kg].
            co2: mixed-layer CO2 [ppm].
            deltaCO2: CO2 jump at h [ppm].
            wCO2: surface kinematic CO2 flux [mgC/m²/s].
            u: mixed-layer u-wind speed [m/s].
            du: u-wind jump at h [m/s].
            v: mixed-layer v-wind speed [m/s].
            dv: v-wind jump at h [m/s].
            dz_h: transition layer thickness [m].
            surf_pressure: surface pressure [Pa].

        Returns:
            The initial mixed layer state.
        """
        return BulkState(
            h_abl=jnp.array(h_abl),
            theta=jnp.array(theta),
            deltatheta=jnp.array(deltatheta),
            q=jnp.array(q),
            dq=jnp.array(dq),
            co2=jnp.array(co2),
            deltaCO2=jnp.array(deltaCO2),
            wCO2=jnp.array(wCO2),
            u=jnp.array(u),
            du=jnp.array(du),
            v=jnp.array(v),
            dv=jnp.array(dv),
            dz_h=jnp.array(dz_h),
            surf_pressure=jnp.array(surf_pressure),
        )

    def run(self, state: AbstractCoupledState) -> BulkState:
        """Run the model.

        Args:
            state:

        Returns:
            The updated mixed layer state.

        """
        land_state = state.land
        atmos = state.atmos
        ml_state = atmos.mixed

        ws = self.compute_ws(ml_state.h_abl)
        wf = self.compute_wf(ml_state.deltatheta)
        w_th_ft = self.compute_w_th_ft(ws)
        w_q_ft = self.compute_w_q_ft(ws)
        w_CO2_ft = self.compute_w_CO2_ft(ws)

        # compute virtual heat flux at surface
        wthetav = (
            land_state.wtheta * (1.0 + 0.61 * ml_state.q)
            + 0.61 * ml_state.theta * land_state.wq
        )

        wstar = self.compute_wstar(
            ml_state.h_abl,
            wthetav,
            ml_state.thetav,
            cst.g,
        )
        wthetave = self.compute_wthetave(wthetav)
        we = self.compute_we(
            ml_state.h_abl,
            wthetave,
            ml_state.deltathetav,
            ml_state.thetav,
            atmos.ustar,
            cst.g,
        )
        wthetae = self.compute_wthetae(we, ml_state.deltatheta)
        wqe = self.compute_wqe(we, ml_state.dq)
        wCO2e = self.compute_wCO2e(we, ml_state.deltaCO2)
        h_abl_tend = self.compute_h_abl_tend(we, ws, wf, atmos.cc_mf)
        thetatend = self.compute_thetatend(ml_state.h_abl, land_state.wtheta, wthetae)
        deltathetatend = self.compute_deltathetatend(
            we, wf, atmos.cc_mf, thetatend, w_th_ft
        )
        qtend = self.compute_qtend(ml_state.h_abl, land_state.wq, wqe, atmos.cc_qf)
        dqtend = self.compute_dqtend(we, wf, atmos.cc_mf, qtend, w_q_ft)
        co2tend = self.compute_co2tend(
            ml_state.h_abl, land_state.wCO2, wCO2e, atmos.wCO2M
        )
        deltaCO2tend = self.compute_deltaCO2tend(we, wf, atmos.cc_mf, co2tend, w_CO2_ft)
        utend = self.compute_utend(
            ml_state.h_abl, we, atmos.uw, ml_state.du, ml_state.dv
        )
        vtend = self.compute_vtend(
            ml_state.h_abl, we, atmos.vw, ml_state.du, ml_state.dv
        )
        dutend = self.compute_dutend(we, wf, atmos.cc_mf, utend)
        dvtend = self.compute_dvtend(we, wf, atmos.cc_mf, vtend)
        dztend = self.compute_dztend(
            ml_state.lcl,
            ml_state.h_abl,
            atmos.cc_frac,
            ml_state.dz_h,
        )
        return ml_state.replace(
            ws=ws,
            wf=wf,
            wstar=wstar,
            wthetav=wthetav,
            wthetave=wthetave,
            we=we,
            wthetae=wthetae,
            wqe=wqe,
            wCO2e=wCO2e,
            h_abl_tend=h_abl_tend,
            thetatend=thetatend,
            deltathetatend=deltathetatend,
            qtend=qtend,
            dqtend=dqtend,
            co2tend=co2tend,
            deltaCO2tend=deltaCO2tend,
            utend=utend,
            vtend=vtend,
            dutend=dutend,
            dvtend=dvtend,
            dztend=dztend,
        )

    def integrate(self, state: BulkState, dt: float) -> BulkState:
        """Integrate mixed layer forward in time.

        Args:
            state: BulkMixedLayerState (component state, not CoupledState).
            dt: Time step.
        """
        h_abl = state.h_abl + dt * state.h_abl_tend
        theta = state.theta + dt * state.thetatend
        deltatheta = state.deltatheta + dt * state.deltathetatend
        q = state.q + dt * state.qtend
        dq = state.dq + dt * state.dqtend
        co2 = state.co2 + dt * state.co2tend
        deltaCO2 = state.deltaCO2 + dt * state.deltaCO2tend
        dz_h = state.dz_h + dt * state.dztend

        # limit dz to minimal value
        dz_h = jnp.maximum(dz_h, 50.0)

        u = jnp.where(self.is_wind_prog, state.u + dt * state.utend, state.u)
        du = jnp.where(self.is_wind_prog, state.du + dt * state.dutend, state.du)
        v = jnp.where(self.is_wind_prog, state.v + dt * state.vtend, state.v)
        dv = jnp.where(self.is_wind_prog, state.dv + dt * state.dvtend, state.dv)

        return state.replace(
            h_abl=h_abl,
            theta=theta,
            deltatheta=deltatheta,
            q=q,
            dq=dq,
            co2=co2,
            deltaCO2=deltaCO2,
            dz_h=dz_h,
            u=u,
            du=du,
            v=v,
            dv=dv,
        )

    def compute_ws(self, h_abl: Array) -> Array:
        """Compute the large-scale subsidence velocity as

        .. math::
            w_s = -\\text{div}U \\cdot h,

        where :math:`\\text{div}U` is the horizontal large-scale divergence of wind and :math:`h` is the ABL height.
        """
        return -self.divU * h_abl

    def compute_wf(self, deltatheta: Array) -> Array:
        """Compute the mixed-layer growth due to cloud top radiative divergence as

        .. math::
            w_f = \\frac{\\Delta F_z}{\\rho c_p \\Delta \\theta},

        where :math:`\\Delta F_z` is the cloud top radiative divergence, :math:`\\rho` is air density,
        :math:`c_p` is specific heat capacity, and :math:`\\Delta \\theta` is the temperature jump at the top of the ABL.
        """
        radiative_denominator = cst.rho * cst.cp * deltatheta
        return self.deltaFz / radiative_denominator

    def compute_w_th_ft(self, ws: Array) -> Array:
        """Compute the potential temperature compensation term to fix free troposphere values as

        .. math::
            w_{\\theta,ft} = \\gamma_\\theta w_s,

        where :math:`\\gamma_\\theta` is the potential temperature compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_th_ft_active = self.gamma_theta * ws
        return jnp.where(self.is_fix_free_trop, w_th_ft_active, 0.0)

    def compute_w_q_ft(self, ws: Array) -> Array:
        """Compute humidity compensation term to fix free troposphere values as

        .. math::
            w_{q,ft} = \\gamma_q w_s,

        where :math:`\\gamma_q` is the humidity compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_q_ft_active = self.gamma_q * ws
        return jnp.where(self.is_fix_free_trop, w_q_ft_active, 0.0)

    def compute_w_CO2_ft(self, ws: Array) -> Array:
        """Compute CO2 compensation term to fix free troposphere values as

        .. math::
            w_{CO2,ft} = \\gamma_{CO2} w_s,

        where :math:`\\gamma_{CO2}` is the CO2 compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_CO2_ft_active = self.gammaCO2 * ws
        return jnp.where(self.is_fix_free_trop, w_CO2_ft_active, 0.0)

    def compute_wstar(
        self,
        h_abl: Array,
        wthetav: Array,
        thetav: Array,
        g: float,
    ) -> Array:
        """Compute the convective velocity scale, defined by

        .. math::
            w_* = \\left( \\frac{g h (\\overline{w'\\theta_v'})_s}{\\theta_v} \\right)^{1/3},

        where :math:`g` is the gravity acceleration, :math:`h` is the height of the atmospheric boundary layer,
        :math:`(\\overline{w'\\theta_v'})_s` is the virtual heat flux at the surface and :math:`\\theta_v` is
        the virtual potential temperature.
        """
        buoyancy_term = g * h_abl * wthetav / thetav
        wstar_positive = buoyancy_term ** (1.0 / 3.0)
        # clip to 1e-6 in case wthetav is negative
        return jnp.where(wthetav > 0.0, wstar_positive, 1e-6)

    def compute_wthetave(self, wthetav: Array) -> Array:
        """Compute the entrainment virtual heat flux as

        .. math::
            (\\overline{w'\\theta_v'})_e = -\\beta (\\overline{w'\\theta_v'})_s,

        where :math:`\\beta` is the entrainment coefficient and
        :math:`(\\overline{w'\\theta_v'})_s` is the virtual heat flux at the surface.
        """
        return -self.beta * wthetav

    def compute_we(
        self,
        h_abl: Array,
        wthetave: Array,
        deltathetav: Array,
        thetav: Array,
        ustar: Array,
        g: float,
    ):
        """Compute the entrainment velocity as

        .. math::
            w_e = -\\frac{(\\overline{w'\\theta_v'})_e}{\\Delta \\theta_v},

        where :math:`(\\overline{w'\\theta_v'})_e` is the entrainment virtual heat flux
        and :math:`\\Delta \\theta_v` is the virtual potential temperature jump at the top of the ABL.

        If shear effects are included (``is_shear_growing=True``), an additional term is added

        .. math::
            w_e = \\frac{-\\overline{w'\\theta_v'}_e + 5 u_*^3 \\theta_v / (g h)}{\\Delta \\theta_v},

        where :math:`u_*` is the friction velocity, and :math:`\\theta_v` is the virtual potential temperature,
        :math:`g` is gravity acceleration, and :math:`h` is the height of the ABL.
        """
        # entrainment velocity with shear effects
        shear_term = 5.0 * ustar**3.0 * thetav / (g * h_abl)
        numerator = -wthetave + shear_term
        we_with_shear = numerator / deltathetav

        # entrainment velocity without shear effects
        we_no_shear = -wthetave / deltathetav

        # select based on is_shear_growing flag
        we_calculated = jnp.where(self.is_shear_growing, we_with_shear, we_no_shear)

        # don't allow boundary layer shrinking if wtheta < 0
        assert isinstance(we_calculated, jnp.ndarray)  # limmau: this is not good
        we_final = jnp.where(we_calculated < 0.0, 0.0, we_calculated)

        return we_final

    def compute_wthetae(self, we: Array, deltatheta: Array) -> Array:
        """Compute the entrainment heat flux as

        .. math::
            (\\overline{w'\\theta'})_e = -w_e \\Delta \\theta,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta \\theta` is the potential temperature jump at the top of the ABL.
        """
        return -we * deltatheta

    def compute_wqe(self, we: Array, dq: Array) -> Array:
        """Compute the entrainment moisture flux as

        .. math::
            (\\overline{w'q'})_e = -w_e \\Delta q,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta q` is the moisture jump at the top of the ABL.
        """
        return -we * dq

    def compute_wCO2e(self, we: Array, deltaCO2: Array) -> Array:
        """Compute the entrainment CO2 flux as

        .. math::
            (\\overline{w'CO_2'})_e = -w_e \\Delta CO_2,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta CO_2` is the CO2 jump at the top of the ABL.
        """
        return -we * deltaCO2

    def compute_h_abl_tend(
        self, we: Array, ws: Array, wf: Array, cc_mf: Array
    ) -> Array:
        """Compute the boundary layer height tendency as

        .. math::
            \\frac{\\text{d}h}{\\text{d}t} = w_e + w_s + w_f - \\text{cc}_{mf},

        where

        - :math:`w_e` comes from :meth:`compute_we`,

        - :math:`w_s` comes from :meth:`compute_ws`,

        - :math:`w_f` comes from :meth:`compute_wf`,

        and :math:`\\text{cc}_{mf}` is the cloud core mass flux from the cloud model.
        """
        return we + ws + wf - cc_mf

    def compute_thetatend(self, h_abl: Array, wtheta: Array, wthetae: Array) -> Array:
        """Compute the mixed-layer potential temperature tendency as

        .. math::
            \\frac{\\text{d}\\theta}{\\text{d}t}
            = \\frac{(\\overline{w'\\theta'})_s - (\\overline{w'\\theta'})_e}{h} + \\text{adv}_\\theta,

        where :math:`\\overline{w'\\theta'}_s` is the surface potential temperature flux,
        :math:`\\overline{w'\\theta'}_e` is the entrainment potential temperature flux,
        and :math:`\\text{adv}_\\theta` is the potential temperature advection.
        """
        surface_heat_flux = (wtheta - wthetae) / h_abl
        return surface_heat_flux + self.advtheta

    def compute_deltathetatend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        thetatend: Array,
        w_th_ft: Array,
    ) -> Array:
        """Compute the potential temperature jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta \\theta}{\\text{d}t}
            = \\gamma_\\theta (w_e + w_f - \\text{cc}_{mf})
            - \\frac{\\text{d}\\theta}{\\text{d}t} + w_{\\theta,ft},

        where
        """
        egrowth = we + wf - cc_mf
        return self.gamma_theta * egrowth - thetatend + w_th_ft

    def compute_qtend(self, h_abl: Array, wq: Array, wqe: Array, cc_qf: Array) -> Array:
        """Compute the mixed-layer specific humidity tendency as

        .. math::
            \\frac{\\text{d}q}{\\text{d}t} = \\frac{\\overline{w'q'}_s - \\overline{w'q'}_e - \\text{cc}_{qf}}{h} + \\text{adv}_q
        """
        surface_moisture_flux = (wq - wqe - cc_qf) / h_abl
        return surface_moisture_flux + self.advq

    def compute_dqtend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        qtend: Array,
        w_q_ft: Array,
    ) -> Array:
        """Compute the specific humidity jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta q}{\\text{d}t} = \\gamma_q (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}q}{\\text{d}t} + w_{q,ft}
        """
        egrowth = we + wf - cc_mf
        return self.gamma_q * egrowth - qtend + w_q_ft

    def compute_co2tend(
        self,
        h_abl: Array,
        wCO2: Array,
        wCO2e: Array,
        wCO2M: Array,
    ) -> Array:
        """Compute the mixed-layer CO2 tendency as

        .. math::
            \\frac{\\text{d}CO_2}{\\text{d}t}
            = \\frac{\\overline{w'CO_2'}_s - \\overline{w'CO_2'}_e - \\text{cc}_{CO2f}}{h} + \\text{adv}_{CO2}
        """
        surface_co2_flux_term = (wCO2 - wCO2e - wCO2M) / h_abl
        return surface_co2_flux_term + self.advCO2

    def compute_deltaCO2tend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        co2tend: Array,
        w_CO2_ft: Array,
    ) -> Array:
        """Compute the CO2 jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta CO_2}{\\text{d}t}
            = \\gamma_{CO2} (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}CO_2}{\\text{d}t} + w_{CO2,ft}
        """
        egrowth = we + wf - cc_mf
        return self.gammaCO2 * egrowth - co2tend + w_CO2_ft

    def compute_utend(
        self,
        h_abl: Array,
        we: Array,
        uw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute the zonal wind tendency as

        .. math::
            \\frac{\\text{d}u}{\\text{d}t}
            = -f_c \\Delta v + \\frac{\\overline{u'w'}_s + w_e \\Delta u}{h} + \\text{adv}_u
        """
        coriolis_term_u = -self.coriolis_param * dv
        momentum_flux_term_u = (uw + we * du) / h_abl
        utend_active = coriolis_term_u + momentum_flux_term_u + self.advu
        return jnp.where(self.is_wind_prog, utend_active, 0.0)

    def compute_vtend(
        self,
        h_abl: Array,
        we: Array,
        vw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute the meridional wind tendency as

        .. math::
            \\frac{\\text{d}v}{\\text{d}t} = f_c \\Delta u + \\frac{\\overline{v'w'}_s + w_e \\Delta v}{h} + \\text{adv}_v
        """
        coriolis_term_v = self.coriolis_param * du
        momentum_flux_term_v = (vw + we * dv) / h_abl
        vtend_active = coriolis_term_v + momentum_flux_term_v + self.advv
        return jnp.where(self.is_wind_prog, vtend_active, 0.0)

    def compute_dutend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        utend: Array,
    ) -> Array:
        """Compute zonal wind jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta u}{\\text{d}t} = \\gamma_u (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}u}{\\text{d}t}
        """
        entrainment_growth_term = we + wf - cc_mf
        dutend_active = self.gammau * entrainment_growth_term - utend
        return jnp.where(self.is_wind_prog, dutend_active, 0.0)

    def compute_dvtend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        vtend: Array,
    ) -> Array:
        """Compute meridional wind jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta v}{\\text{d}t}
            = \\gamma_v (w_e + w_f - \\text{cc}_{mf})
            - \\frac{\\text{d}v}{\\text{d}t}
        """
        entrainment_growth_term = we + wf - cc_mf
        dvtend_active = self.gammav * entrainment_growth_term - vtend
        return jnp.where(self.is_wind_prog, dvtend_active, 0.0)

    def compute_dztend(
        self,
        lcl: Array,
        h_abl: Array,
        cc_frac: Array,
        dz_h: Array,
    ):
        """Compute the transition layer thickness tendency as

        .. math::
            \\frac{\\text{d}\\delta z_h}{\\text{d}t} = \\frac{(LCL - h) - \\delta z_h}{\\tau}

        where :math:`\\tau = 7200` s. This is basically a relaxation term.
        """
        lcl_distance = lcl - h_abl

        # tendency for active case
        target_thickness = lcl_distance - dz_h
        dztend_active = target_thickness / 7200.0
        condition = (cc_frac > 0) | (lcl_distance < 300)
        dztend = jnp.where(condition, dztend_active, 0.0)

        return dztend

    def statistics(self, state: AbstractCoupledState, t: int):
        """Compute standard meteorological statistics and diagnostics."""
        mixed_state = state.atmos.mixed
        land_state = state.land
        thetav = self.compute_thetav(mixed_state.theta, mixed_state.q)
        wthetav = self.compute_wthetav(
            land_state.wtheta, mixed_state.theta, land_state.wq
        )
        deltathetav = self.compute_deltathetav(
            mixed_state.theta,
            mixed_state.deltatheta,
            mixed_state.q,
            mixed_state.dq,
        )
        top_p = self.compute_top_p(
            mixed_state.surf_pressure, cst.rho, cst.g, mixed_state.h_abl
        )
        top_T = self.compute_top_T(mixed_state.theta, cst.g, cst.cp, mixed_state.h_abl)
        top_rh = self.compute_top_rh(mixed_state.q, top_T, top_p)
        lcl = self.compute_lcl(
            mixed_state.h_abl,
            mixed_state.lcl,
            mixed_state.surf_pressure,
            mixed_state.theta,
            mixed_state.q,
            t,
        )
        ml_state = state.atmos.mixed.replace(
            thetav=thetav,
            wthetav=wthetav,
            deltathetav=deltathetav,
            top_p=top_p,
            top_T=top_T,
            top_rh=top_rh,
            lcl=lcl,
        )
        return ml_state

    def compute_thetav(self, theta: Array, q: Array) -> Array:
        """Computes the virtual potential temperature as

        .. math::
            \\theta_v = \\theta \\left(1 + 0.61\\, q\\right).
        """
        return theta * (1.0 + 0.61 * q)

    def compute_wthetav(self, wtheta: Array, theta: Array, wq: Array) -> Array:
        """Computes the virtual potential temperature flux as

        .. math::
            \\overline{w'\\theta_v'} = \\overline{w'\\theta'} + 0.61\\,\\theta\\,\\overline{w'q'}.
        """
        return wtheta + 0.61 * theta * wq

    def compute_deltathetav(
        self,
        theta: Array,
        deltatheta: Array,
        q: Array,
        dq: Array,
    ) -> Array:
        """Computes the virtual potential temperature jump as

        .. math::
            \\Delta\\theta_v = (\\theta + \\Delta\\theta)\\left(1 + 0.61\\,(q + \\Delta q)\\right)
            - \\theta\\left(1 + 0.61\\,q\\right)
        """
        return (theta + deltatheta) * (1.0 + 0.61 * (q + dq)) - theta * (1.0 + 0.61 * q)

    def compute_top_p(
        self, surf_pressure: Array, rho: float, g: float, h_abl: Array
    ) -> Array:
        """Computes the pressure at the top of the mixed layer as

        .. math::
            p_{top} = p_{surf} - \\rho\\, g\\, h.
        """
        return surf_pressure - rho * g * h_abl

    def compute_top_T(self, theta: Array, g: float, cp: float, h_abl: Array) -> Array:
        """Computes the temperature at the top of the mixed layer as

        .. math::
            T_{top} = \\theta - \\frac{g}{c_p}\\, h.
        """
        return theta - (g / cp) * h_abl

    def compute_top_rh(self, q: Array, top_T: Array, top_p: Array) -> Array:
        """Computes the relative humidity at the mixed-layer top as

        .. math::
            \\mathrm{RH}_{top} = \\frac{q}{q_{sat}(T_{top},\\,p_{top})}.
        """
        return q / compute_qsat(top_T, top_p)

    def compute_lcl(
        self,
        h_abl: Array,
        lcl: Array,
        surf_pressure: Array,
        theta: Array,
        q: Array,
        t: int,
    ) -> Array:
        """Compute the lifting condensation level (LCL).

        The LCL is found iteratively by finding the height where the relative humidity is 100%.
        """
        # find lifting condensation level iteratively using JAX
        # initialize lcl and rhlcl based on timestep
        initial_lcl = jnp.where(t == 0, h_abl, lcl)
        initial_rhlcl = jnp.where(t == 0, 0.5, 0.9998)

        def lcl_iteration_body(carry, _):
            lcl, rhlcl = carry

            # update lcl based on current relative humidity
            lcl_adjustment = (1.0 - rhlcl) * 1000.0

            # convergence check could be done here but scan runs all steps
            # we damp the adjustment if already converged to avoid jitter, though simple replacement is fine

            new_lcl = lcl + lcl_adjustment

            # calculate new relative humidity at updated lcl
            p_lcl = surf_pressure - cst.rho * cst.g * new_lcl
            temp_lcl = theta - cst.g / cst.cp * new_lcl
            new_rhlcl = q / compute_qsat(temp_lcl, p_lcl)

            return (new_lcl, new_rhlcl), None

        # now we have a fixed number of iterations
        n_iter = 30
        (final_lcl, final_rhlcl), _ = jax.lax.scan(
            lcl_iteration_body, (initial_lcl, initial_rhlcl), None, length=n_iter
        )

        return final_lcl
