from abc import abstractmethod

from .utils import PhysicalConstants, get_qsat


class AbstractMixedLayerModel:
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
        # constants
        self.const = PhysicalConstants()

        # 1. mixed layer switches
        # mixed-layer model switch
        self.sw_ml = sw_ml
        # shear growth mixed-layer switch
        self.sw_shearwe = sw_shearwe
        # fix the free-troposphere switch
        self.sw_fixft = sw_fixft
        # 2. large scale parameters
        # initial ABL height [m]
        self.abl_height = abl_height
        # surface pressure [Pa]
        self.surf_pressure = surf_pressure
        # horizontal large-scale divergence of wind [s-1]
        self.divU = divU
        # Coriolis parameter [m s-1]
        self.coriolis_param = coriolis_param
        # 3. temperature parameters
        # initial mixed-layer potential temperature [K]
        self.theta = theta
        # initial temperature jump at h [K]
        self.dtheta = dtheta
        # free atmosphere potential temperature lapse rate [K m-1]
        self.gammatheta = gammatheta
        # advection of heat [K s-1]
        self.advtheta = advtheta
        # entrainment ratio for virtual heat [-]
        self.beta = beta
        # surface kinematic heat flux [K m s-1]
        self.wtheta = wtheta
        # 4. entrainment parameters
        # convective velocity scale [m s-1]
        self.wstar = 0.0
        # large-scale vertical velocity [m s-1]
        self.ws = None
        # mixed-layer growth due to radiative divergence [m s-1]
        self.wf = None
        # entrainment velocity [m s-1]
        self.we = -1.0
        # 5. moisture parameters
        # initial mixed-layer specific humidity [kg kg-1]
        self.q = q
        # initial specific humidity jump at h [kg kg-1]
        self.dq = dq
        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.gammaq = gammaq
        # advection of moisture [kg kg-1 s-1]
        self.advq = advq
        # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wq = wq
        # entrainment moisture flux [kg kg-1 m s-1]
        self.wqe = None
        # mixed-layer saturated specific humidity [kg kg-1]
        self.qsat = None
        # mixed-layer saturated vapor pressure [Pa]
        self.esat = None
        # mixed-layer vapor pressure [Pa]
        self.e = None
        # surface saturated specific humidity [g kg-1]
        self.qsatsurf = None
        # slope saturated specific humidity curve [g kg-1 K-1]
        self.dqsatdT = None
        # 8. mixed-layer top variables
        # mixed-layer top pressure [pa]
        self.top_p = None
        # mixed-layer top absolute temperature [K]
        self.top_T = None
        # mixed-layer top specific humidity variance [kg2 kg-2]
        self.q2_h = None
        # mixed-layer top CO2 variance [ppm2]
        self.top_CO22 = None
        # mixed-layer top relavtive humidity [-]
        self.top_rh = None
        # transition layer thickness [-]
        self.dz_h = dz_h
        # lifting condensation level [m]
        self.lcl = None
        # 9. virtual temperatures and fluxes
        # initial mixed-layer potential temperature [K]
        self.thetav = None
        # initial virtual temperature jump at h [K]
        self.dthetav = None
        # surface kinematic virtual heat flux [K m s-1]
        self.wthetav = None
        # entrainment kinematic heat flux [K m s-1]
        self.wthetae = None
        # entrainment kinematic virtual heat flux [K m s-1]
        self.wthetave = None
        # 10. CO2
        # conversion factor mgC m-2 s-1 to ppm m s-1
        fac = self.const.mair / (self.const.rho * self.const.mco2)
        # initial mixed-layer CO2 [ppm]
        self.co2 = co2
        # initial CO2 jump at h [ppm]
        self.dCO2 = dCO2
        # free atmosphere CO2 lapse rate [ppm m-1]
        self.gammaco2 = gammaCO2
        # advection of CO2 [ppm s-1]
        self.advCO2 = advCO2
        # surface kinematic CO2 flux [ppm m s-1]
        self.wCO2 = wCO2 * fac
        # surface assimulation CO2 flux [ppm m s-1]
        self.wCO2A = 0.0
        # surface respiration CO2 flux [ppm m s-1]
        self.wCO2R = 0.0
        # entrainment CO2 flux [ppm m s-1]
        self.wCO2e = None
        # CO2 mass flux [ppm m s-1]
        self.wCO2M = 0.0
        # 11. wind parameters
        # prognostic wind switch
        self.sw_wind = sw_wind
        # initial mixed-layer u-wind speed [m s-1]
        self.u = u
        # initial u-wind jump at h [m s-1]
        self.du = du
        # free atmosphere u-wind speed lapse rate [s-1]
        self.gammau = gammau
        # advection of u-wind [m s-2]
        self.advu = advu
        # initial mixed-layer v-wind speed [m s-1]
        self.v = v
        # initial v-wind jump at h [m s-1]
        self.dv = dv
        # free atmosphere v-wind speed lapse rate [s-1]
        self.gammav = gammav
        # advection of v-wind [m s-2]
        self.advv = advv
        # 12. tendencies
        # tendency of CBL [m s-1]
        self.htend = None
        # tendency of mixed-layer potential temperature [K s-1]
        self.thetatend = None
        # tendency of potential temperature jump at h [K s-1]
        self.dthetatend = None
        # tendency of mixed-layer specific humidity [kg kg-1 s-1]
        self.qtend = None
        # tendency of specific humidity jump at h [kg kg-1 s-1]
        self.dqtend = None
        # tendency of CO2 humidity [ppm]
        self.co2tend = None
        # tendency of CO2 jump at h [ppm s-1]
        self.dCO2tend = None
        # tendency of u-wind [m s-1 s-1]
        self.utend = None
        # tendency of u-wind jump at h [m s-1 s-1]
        self.dutend = None
        # tendency of v-wind [m s-1 s-1]
        self.vtend = None
        # tendency of v-wind jump at h [m s-1 s-1]
        self.dvtend = None
        # tendency of transition layer thickness [m s-1]
        self.dztend = None

    @abstractmethod
    def run(
        self,
        dFz: float,
        cc_mf: float,
        cc_frac: float,
        cc_qf: float,
        ustar: float,
        uw: float,
        vw: float,
    ):
        pass

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError

    def statistics(self, t: float):
        # calculate virtual temperatures
        self.thetav = self.theta + 0.61 * self.theta * self.q
        self.wthetav = self.wtheta + 0.61 * self.theta * self.wq
        self.dthetav = (self.theta + self.dtheta) * (
            1.0 + 0.61 * (self.q + self.dq)
        ) - self.theta * (1.0 + 0.61 * self.q)

        # mixed-layer top properties
        self.top_p = (
            self.surf_pressure - self.const.rho * self.const.g * self.abl_height
        )
        self.top_T = self.theta - self.const.g / self.const.cp * self.abl_height
        self.top_rh = self.q / get_qsat(self.top_T, self.top_p)

        # find lifting condensation level iteratively
        if t == 0:
            self.lcl = self.abl_height
            RHlcl = 0.5
        else:
            RHlcl = 0.9998

        itmax = 30
        it = 0
        while ((RHlcl <= 0.9999) or (RHlcl >= 1.0001)) and it < itmax:
            self.lcl += (1.0 - RHlcl) * 1000.0
            p_lcl = self.surf_pressure - self.const.rho * self.const.g * self.lcl
            T_lcl = self.theta - self.const.g / self.const.cp * self.lcl
            RHlcl = self.q / get_qsat(T_lcl, p_lcl)
            it += 1

        if it == itmax:
            print("LCL calculation not converged!!")
            print("RHlcl = %f, zlcl=%f" % (RHlcl, self.lcl))


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
        dFz: float,
        cc_mf: float,
        cc_frac: float,
        cc_qf: float,
        ustar: float,
        uw: float,
        vw: float,
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
        self.wf = dFz / (self.const.rho * self.const.cp * self.dtheta)

        # calculate convective velocity scale w*
        if self.wthetav > 0.0:
            self.wstar = (
                (self.const.g * self.abl_height * self.wthetav) / self.thetav
            ) ** (1.0 / 3.0)
        else:
            self.wstar = 1e-6

        # Virtual heat entrainment flux
        self.wthetave = -self.beta * self.wthetav

        # compute mixed-layer tendencies
        if self.sw_shearwe:
            self.we = (
                -self.wthetave
                + 5.0 * ustar**3.0 * self.thetav / (self.const.g * self.abl_height)
            ) / self.dthetav
        else:
            self.we = -self.wthetave / self.dthetav

        # Don't allow boundary layer shrinking if wtheta < 0
        if self.we < 0:
            self.we = 0.0

        # Calculate entrainment fluxes
        self.wthetae = -self.we * self.dtheta
        self.wqe = -self.we * self.dq
        self.wCO2e = -self.we * self.dCO2

        self.htend = self.we + self.ws + self.wf - cc_mf

        self.thetatend = (self.wtheta - self.wthetae) / self.abl_height + self.advtheta
        self.qtend = (self.wq - self.wqe - cc_qf) / self.abl_height + self.advq
        self.co2tend = (
            self.wCO2 - self.wCO2e - self.wCO2M
        ) / self.abl_height + self.advCO2

        self.dthetatend = (
            self.gammatheta * (self.we + self.wf - cc_mf) - self.thetatend + w_th_ft
        )
        self.dqtend = self.gammaq * (self.we + self.wf - cc_mf) - self.qtend + w_q_ft
        self.dCO2tend = (
            self.gammaco2 * (self.we + self.wf - cc_mf) - self.co2tend + w_CO2_ft
        )

        # assume u + du = ug, so ug - u = du
        if self.sw_wind:
            self.utend = (
                -self.coriolis_param * self.dv
                + (uw + self.we * self.du) / self.abl_height
                + self.advu
            )
            self.vtend = (
                self.coriolis_param * self.du
                + (vw + self.we * self.dv) / self.abl_height
                + self.advv
            )

            self.dutend = self.gammau * (self.we + self.wf - cc_mf) - self.utend
            self.dvtend = self.gammav * (self.we + self.wf - cc_mf) - self.vtend

        # tendency of the transition layer thickness
        if cc_frac > 0 or self.lcl - self.abl_height < 300:
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

        # Limit dz to minimal value
        dz0 = 50
        if self.dz_h < dz0:
            self.dz_h = dz0

        if self.sw_wind:
            self.u = u0 + dt * self.utend
            self.du = du0 + dt * self.dutend
            self.v = v0 + dt * self.vtend
            self.dv = dv0 + dt * self.dvtend
