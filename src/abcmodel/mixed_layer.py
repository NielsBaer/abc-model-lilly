from .components import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)


class NoMixedLayerModel(AbstractMixedLayerModel):
    # limamau: this shouldn't need all this arguments
    # to be cleaned up in the future
    def __init__(
        self,
        sw_ml: bool,
        sw_shearwe: bool,
        sw_fixft: bool,
        abl_height: float,
        surf_pressure: float,
        divU: float,
        coriolis_param: float,
        theta: float,
        dtheta: float,
        gammatheta: float,
        advtheta: float,
        beta: float,
        wtheta: float,
        q: float,
        dq: float,
        gammaq: float,
        advq: float,
        wq: float,
        co2: float,
        dCO2: float,
        gammaCO2: float,
        advCO2: float,
        wCO2: float,
        sw_wind: bool,
        u: float,
        du: float,
        gammau: float,
        advu: float,
        v: float,
        dv: float,
        gammav: float,
        advv: float,
        dz_h: float,
    ):
        super().__init__(
            sw_ml,
            sw_shearwe,
            sw_fixft,
            abl_height,
            surf_pressure,
            divU,
            coriolis_param,
            theta,
            dtheta,
            gammatheta,
            advtheta,
            beta,
            wtheta,
            q,
            dq,
            gammaq,
            advq,
            wq,
            co2,
            dCO2,
            gammaCO2,
            advCO2,
            wCO2,
            sw_wind,
            u,
            du,
            gammau,
            advu,
            v,
            dv,
            gammav,
            advv,
            dz_h,
        )

    def integrate(self, dt: float):
        pass


class StandardMixedLayerModel(AbstractMixedLayerModel):
    def __init__(
        self,
        sw_ml: bool,
        sw_shearwe: bool,
        sw_fixft: bool,
        abl_height: float,
        surf_pressure: float,
        divU: float,
        coriolis_param: float,
        theta: float,
        dtheta: float,
        gammatheta: float,
        advtheta: float,
        beta: float,
        wtheta: float,
        q: float,
        dq: float,
        gammaq: float,
        advq: float,
        wq: float,
        co2: float,
        dCO2: float,
        gammaCO2: float,
        advCO2: float,
        wCO2: float,
        sw_wind: bool,
        u: float,
        du: float,
        gammau: float,
        advu: float,
        v: float,
        dv: float,
        gammav: float,
        advv: float,
        dz_h: float,
    ):
        super().__init__(
            sw_ml,
            sw_shearwe,
            sw_fixft,
            abl_height,
            surf_pressure,
            divU,
            coriolis_param,
            theta,
            dtheta,
            gammatheta,
            advtheta,
            beta,
            wtheta,
            q,
            dq,
            gammaq,
            advq,
            wq,
            co2,
            dCO2,
            gammaCO2,
            advCO2,
            wCO2,
            sw_wind,
            u,
            du,
            gammau,
            advu,
            v,
            dv,
            gammav,
            advv,
            dz_h,
        )

    def run(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        clouds: AbstractCloudModel,
    ):
        # calculate large-scale vertical velocity (subsidence)
        self.ws = -self.divU * self.abl_height

        # calculate compensation to fix the free troposphere in case of subsidence
        if self.sw_fixft:
            w_th_ft = self.gammatheta * self.ws
            w_q_ft = self.gammaq * self.ws
            w_CO2_ft = self.gammaco2 * self.ws
        else:
            w_th_ft = 0.0
            w_q_ft = 0.0
            w_CO2_ft = 0.0

        # calculate mixed-layer growth due to cloud top radiative divergence
        self.wf = radiation.dFz / (self.const.rho * self.const.cp * self.dtheta)

        # calculate convective velocity scale w*
        if self.wthetav > 0.0:
            self.wstar = (
                (self.const.g * self.abl_height * self.wthetav) / self.thetav
            ) ** (1.0 / 3.0)
        else:
            self.wstar = 1e-6

        # virtual heat entrainment flux
        self.wthetave = -self.beta * self.wthetav

        # compute mixed-layer tendencies
        if self.sw_shearwe:
            self.we = (
                -self.wthetave
                + 5.0
                * surface_layer.ustar**3.0
                * self.thetav
                / (self.const.g * self.abl_height)
            ) / self.dthetav
        else:
            self.we = -self.wthetave / self.dthetav

        # don't allow boundary layer shrinking if wtheta < 0
        if self.we < 0:
            self.we = 0.0

        # calculate entrainment fluxes
        self.wthetae = -self.we * self.dtheta
        self.wqe = -self.we * self.dq
        self.wCO2e = -self.we * self.dCO2

        self.htend = self.we + self.ws + self.wf - clouds.cc_mf

        self.thetatend = (self.wtheta - self.wthetae) / self.abl_height + self.advtheta
        self.qtend = (self.wq - self.wqe - clouds.cc_qf) / self.abl_height + self.advq
        self.co2tend = (
            self.wCO2 - self.wCO2e - self.wCO2M
        ) / self.abl_height + self.advCO2

        self.dthetatend = (
            self.gammatheta * (self.we + self.wf - clouds.cc_mf)
            - self.thetatend
            + w_th_ft
        )
        self.dqtend = (
            self.gammaq * (self.we + self.wf - clouds.cc_mf) - self.qtend + w_q_ft
        )
        self.dCO2tend = (
            self.gammaco2 * (self.we + self.wf - clouds.cc_mf) - self.co2tend + w_CO2_ft
        )

        # assume u + du = ug, so ug - u = du
        if self.sw_wind:
            self.utend = (
                -self.coriolis_param * self.dv
                + (surface_layer.uw + self.we * self.du) / self.abl_height
                + self.advu
            )
            self.vtend = (
                self.coriolis_param * self.du
                + (surface_layer.vw + self.we * self.dv) / self.abl_height
                + self.advv
            )

            self.dutend = self.gammau * (self.we + self.wf - clouds.cc_mf) - self.utend
            self.dvtend = self.gammav * (self.we + self.wf - clouds.cc_mf) - self.vtend

        # tendency of the transition layer thickness
        if clouds.cc_frac > 0 or self.lcl - self.abl_height < 300:
            self.dztend = ((self.lcl - self.abl_height) - self.dz_h) / 7200.0
        else:
            self.dztend = 0.0

    def integrate(self, dt: float):
        # set values previous time step
        h0 = self.abl_height

        theta0 = self.theta
        dtheta0 = self.dtheta
        q0 = self.q
        dq0 = self.dq
        CO20 = self.co2
        dCO20 = self.dCO2

        u0 = self.u
        du0 = self.du
        v0 = self.v
        dv0 = self.dv

        dz0 = self.dz_h

        # integrate mixed-layer equations
        self.abl_height = h0 + dt * self.htend
        self.theta = theta0 + dt * self.thetatend
        self.dtheta = dtheta0 + dt * self.dthetatend
        self.q = q0 + dt * self.qtend
        self.dq = dq0 + dt * self.dqtend
        self.co2 = CO20 + dt * self.co2tend
        self.dCO2 = dCO20 + dt * self.dCO2tend
        self.dz_h = dz0 + dt * self.dztend

        # limit dz to minimal value
        dz0 = 50
        if self.dz_h < dz0:
            self.dz_h = dz0

        if self.sw_wind:
            self.u = u0 + dt * self.utend
            self.du = du0 + dt * self.dutend
            self.v = v0 + dt * self.vtend
            self.dv = dv0 + dt * self.dvtend
