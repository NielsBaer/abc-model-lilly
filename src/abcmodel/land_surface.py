from abc import abstractmethod

import numpy as np
from scipy.special import exp1

from .mixed_layer import AbstractMixedLayerModel
from .radiation import AbstractRadiationModel
from .surface_layer import AbstractSurfaceLayerModel
from .utils import PhysicalConstants, get_esat, get_qsat


class AbstractLandSurfaceModel:
    def __init__(
        self,
        ls_type: str,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2ref: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
        c3c4: str,
    ):
        self.const = PhysicalConstants()

        # ls type
        self.ls_type = ls_type

        # A-Gs constants and settings
        # plant type: [C3, C4]
        # CO2 compensation concentration [mg m-3]
        self.co2comp298 = [68.5, 4.3]
        # function parameter to calculate CO2 compensation concentration [-]
        self.net_rad10CO2 = [1.5, 1.5]
        # mesophyill conductance at 298 K [mm s-1]
        self.gm298 = [7.0, 17.5]
        # CO2 maximal primary productivity [mg m-2 s-1]
        self.ammax298 = [2.2, 1.7]
        # function parameter to calculate mesophyll conductance [-]
        self.net_rad10gm = [2.0, 2.0]
        # reference temperature to calculate mesophyll conductance gm [K]
        self.temp1gm = [278.0, 286.0]
        # reference temperature to calculate mesophyll conductance gm [K]
        self.temp2gm = [301.0, 309.0]
        # function parameter to calculate maximal primary profuctivity Ammax
        self.net_rad10Am = [2.0, 2.0]
        # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.temp1Am = [281.0, 286.0]
        # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.temp2Am = [311.0, 311.0]
        # maximum value Cfrac [-]
        self.f0 = [0.89, 0.85]
        # regression coefficient to calculate Cfrac [kPa-1]
        self.ad = [0.07, 0.15]
        # initial low light conditions [mg J-1]
        self.alpha0 = [0.017, 0.014]
        # extinction coefficient PAR [-]
        self.kx = [0.7, 0.7]
        # cuticular (minimum) conductance [mm s-1]
        self.gmin = [0.25e-3, 0.25e-3]
        # ratio molecular viscosity water to carbon dioxide
        self.nuco2q = 1.6
        # constant water stress correction (eq. 13 Jacobs et al. 2007) [-]
        self.cw = 0.0016
        # upper reference value soil water [-]
        self.wmax = 0.55
        # lower reference value soil water [-]
        self.wmin = 0.005
        # respiration at 10 C [mg CO2 m-2 s-1]
        self.r10 = 0.23
        # activation energy [53.3 kJ kmol-1]
        self.e0 = 53.3e3
        # initialize A-Gs surface scheme
        self.c3c4 = c3c4  # plant type ('c3' or 'c4')

        # water content parameters
        # volumetric water content top soil layer [m3 m-3]
        self.wg = wg
        # volumetric water content deeper soil layer [m3 m-3]
        self.w2 = w2
        # saturated volumetric water content ECMWF config [-]
        self.wsat = wsat
        # volumetric water content field capacity [-]
        self.wfc = wfc
        # volumetric water content wilting point [-]
        self.wwilt = wwilt

        # temperature params
        # temperature top soil layer [K]
        self.temp_soil = temp_soil
        # temperature deeper soil layer [K]
        self.temp2 = temp2
        # surface temperature [K]
        self.surf_temp = surf_temp

        # Clapp and Hornberger retention curve parameters
        self.a = a
        self.b = b
        self.p = p
        # saturated soil conductivity for heat
        self.cgsat = cgsat

        # C parameters
        self.c1sat = c1sat
        self.c2ref = c2ref

        # limamau: was this assigned at any point?
        # # curvature plant water-stress factor (0..1) [-]
        self.c_beta = None

        # vegetation parameters
        # leaf area index [-]
        self.lai = lai
        # correction factor transpiration for VPD [-]
        self.gD = gD
        # minimum resistance transpiration [s m-1]
        self.rsmin = rsmin
        # minimum resistance soil evaporation [s m-1]
        self.rssoilmin = rssoilmin
        # surface albedo [-]
        self.alpha = alpha

        # resistance parameters (initialized to high values)
        # resistance transpiration [s m-1]
        self.rs = 1.0e6
        # resistance soil [s m-1]
        self.rssoil = 1.0e6

        # vegetation and water layer parameters
        # vegetation fraction [-]
        self.cveg = cveg
        # thickness of water layer on wet vegetation [m]
        self.wmax = wmax
        # equivalent water layer depth for wet vegetation [m]
        self.wl = wl
        # wet fraction [-]
        self.cliq = None

        # thermal diffusivity
        self.lamb = lam  # thermal diffusivity skin layer [-]

        # tendencies
        # soil temperature tendency [K s-1]
        self.temp_soil_tend = None
        # soil moisture tendency [m3 m-3 s-1]
        self.wgtend = None
        # equivalent liquid water tendency [m s-1]
        self.wltend = None

        # heat and water fluxes
        # sensible heat flux [W m-2]
        self.hf = None
        # evapotranspiration [W m-2]
        self.le = None
        # open water evaporation [W m-2]
        self.le_liq = None
        # transpiration [W m-2]
        self.le_veg = None
        # soil evaporation [W m-2]
        self.le_soil = None
        # potential evaporation [W m-2]
        self.le_pot = None
        # reference evaporation using rs = rsmin / LAI [W m-2]
        self.le_ref = None
        # ground heat flux [W m-2]
        self.gf = None

        # some sanity checks for valid input
        if self.c_beta is None:
            self.c_beta = 0.0  # zero curvature; linear response
        assert self.c_beta >= 0.0 or self.c_beta <= 1.0

    @abstractmethod
    def run(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    def integrate(self, dt: float) -> None:
        pass


class NoLandSurfaceModel(AbstractLandSurfaceModel):
    # limamau: this shouldn't need all this arguments
    # to be cleaned up in the future
    def __init__(
        self,
        ls_type: str,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
        c3c4: str,
    ):
        super().__init__(
            ls_type,
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
            c3c4,
        )

    def run(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        pass


class AbstractStandardLandSurfaceModel(AbstractLandSurfaceModel):
    def __init__(
        self,
        ls_type: str,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
        c3c4: str,
    ):
        super().__init__(
            ls_type,
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
            c3c4,
        )

    @abstractmethod
    def compute_surface_resistance(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_co2_flux(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        raise NotImplementedError

    def run(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # compute aerodynamic resistance
        surface_layer.compute_ra(mixed_layer.u, mixed_layer.v, mixed_layer.wstar)

        # first calculate essential thermodynamic variables
        mixed_layer.esat = get_esat(mixed_layer.theta)
        mixed_layer.qsat = get_qsat(mixed_layer.theta, mixed_layer.surf_pressure)
        desatdT = mixed_layer.esat * (
            17.2694 / (mixed_layer.theta - 35.86)
            - 17.2694
            * (mixed_layer.theta - 273.16)
            / (mixed_layer.theta - 35.86) ** 2.0
        )
        mixed_layer.dqsatdT = 0.622 * desatdT / mixed_layer.surf_pressure
        mixed_layer.e = mixed_layer.q * mixed_layer.surf_pressure / 0.622

        # sub-model part
        self.compute_surface_resistance(radiation, surface_layer, mixed_layer)
        self.compute_co2_flux(surface_layer, mixed_layer)

        # recompute f2 using wg instead of w2
        if self.wg > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
            f2 = 1.0e8
        self.rssoil = self.rssoilmin * f2

        wlmx = self.lai * self.wmax
        self.cliq = min(1.0, self.wl / wlmx)

        # calculate skin temperature implicitly
        self.surf_temp = (
            radiation.net_rad
            + self.const.rho * self.const.cp / surface_layer.ra * mixed_layer.theta
            + self.cveg
            * (1.0 - self.cliq)
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rs)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.cveg
            * self.cliq
            * self.const.rho
            * self.const.lv
            / surface_layer.ra
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.lamb * self.temp_soil
        ) / (
            self.const.rho * self.const.cp / surface_layer.ra
            + self.cveg
            * (1.0 - self.cliq)
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rs)
            * mixed_layer.dqsatdT
            + (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rssoil)
            * mixed_layer.dqsatdT
            + self.cveg
            * self.cliq
            * self.const.rho
            * self.const.lv
            / surface_layer.ra
            * mixed_layer.dqsatdT
            + self.lamb
        )

        # limamau: should eastsurf just be deleted here?
        esatsurf = get_esat(self.surf_temp)
        mixed_layer.qsatsurf = get_qsat(self.surf_temp, mixed_layer.surf_pressure)

        self.le_veg = (
            (1.0 - self.cliq)
            * self.cveg
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rs)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_liq = (
            self.cliq
            * self.cveg
            * self.const.rho
            * self.const.lv
            / surface_layer.ra
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_soil = (
            (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (surface_layer.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )

        self.wltend = -self.le_liq / (self.const.rhow * self.const.lv)

        self.le = self.le_soil + self.le_veg + self.le_liq
        self.hf = (
            self.const.rho
            * self.const.cp
            / surface_layer.ra
            * (self.surf_temp - mixed_layer.theta)
        )
        self.gf = self.lamb * (self.surf_temp - self.temp_soil)
        self.le_pot = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + self.const.rho
            * self.const.cp
            / surface_layer.ra
            * (mixed_layer.qsat - mixed_layer.q)
        ) / (mixed_layer.dqsatdT + self.const.cp / self.const.lv)
        self.le_ref = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + self.const.rho
            * self.const.cp
            / surface_layer.ra
            * (mixed_layer.qsat - mixed_layer.q)
        ) / (
            mixed_layer.dqsatdT
            + self.const.cp
            / self.const.lv
            * (1.0 + self.rsmin / self.lai / surface_layer.ra)
        )

        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * np.log(10.0)))

        self.temp_soil_tend = cg * self.gf - 2.0 * np.pi / 86400.0 * (
            self.temp_soil - self.temp2
        )

        d1 = 0.1
        c1 = self.c1sat * (self.wsat / self.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        self.wgtend = -c1 / (
            self.const.rhow * d1
        ) * self.le_soil / self.const.lv - c2 / 86400.0 * (self.wg - wgeq)

        # calculate kinematic heat fluxes
        mixed_layer.wtheta = self.hf / (self.const.rho * self.const.cp)
        mixed_layer.wq = self.le / (self.const.rho * self.const.lv)

    def integrate(self, dt: float):
        self.temp_soil = self.temp_soil + dt * self.temp_soil_tend
        self.wg = self.wg + dt * self.wgtend
        self.wl = self.wl + dt * self.wltend


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    def __init__(
        self,
        ls_type: str,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
        c3c4: str,
    ):
        super().__init__(
            ls_type,
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
            c3c4,
        )

    def compute_surface_resistance(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # calculate surface resistances using Jarvis-Stewart model
        f1 = radiation.get_f1()

        if self.w2 > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(-self.gD * (mixed_layer.esat - mixed_layer.e) / 100.0)
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - mixed_layer.theta) ** 2.0)

        self.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

    def compute_co2_flux(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        pass


class AquaCropModel(AbstractStandardLandSurfaceModel):
    def __init__(
        self,
        ls_type: str,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
        c3c4: str,
    ):
        super().__init__(
            ls_type,
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
            c3c4,
        )
        self.rsCO2 = None
        self.gcco2 = None
        self.ci = None

    def compute_surface_resistance(
        self,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # Select index for plant type
        if self.c3c4 == "c3":
            c = 0
        elif self.c3c4 == "c4":
            c = 1
        else:
            raise ValueError(f'Invalid option "{self.c3c4}" for "c3c4".')

        # calculate CO2 compensation concentration
        co2comp = (
            self.co2comp298[c]
            * self.const.rho
            * pow(self.net_rad10CO2[c], (0.1 * (surface_layer.thetasurf - 298.0)))
        )

        # calculate mesophyll conductance
        gm = (
            self.gm298[c]
            * pow(self.net_rad10gm[c], (0.1 * (surface_layer.thetasurf - 298.0)))
            / (
                (1.0 + np.exp(0.3 * (self.temp1gm[c] - surface_layer.thetasurf)))
                * (1.0 + np.exp(0.3 * (surface_layer.thetasurf - self.temp2gm[c])))
            )
        )
        # conversion from mm s-1 to m s-1
        gm = gm / 1000.0

        # calculate CO2 concentration inside the leaf (ci)
        fmin0 = self.gmin[c] / self.nuco2q - 1.0 / 9.0 * gm
        fmin = -fmin0 + pow(
            (pow(fmin0, 2.0) + 4 * self.gmin[c] / self.nuco2q * gm), 0.5
        ) / (2.0 * gm)

        ds = (get_esat(self.surf_temp) - mixed_layer.e) / 1000.0  # kPa
        d0 = (self.f0[c] - fmin) / self.ad[c]

        cfrac = self.f0[c] * (1.0 - (ds / d0)) + fmin * (ds / d0)
        self.co2abs = (
            mixed_layer.co2 * (self.const.mco2 / self.const.mair) * self.const.rho
        )
        # conversion mumol mol-1 (ppm) to mgCO2 m3
        self.ci = cfrac * (self.co2abs - co2comp) + co2comp

        # calculate maximal gross primary production in high light conditions (Ag)
        ammax = (
            self.ammax298[c]
            * pow(self.net_rad10Am[c], (0.1 * (surface_layer.thetasurf - 298.0)))
            / (
                (1.0 + np.exp(0.3 * (self.temp1Am[c] - surface_layer.thetasurf)))
                * (1.0 + np.exp(0.3 * (surface_layer.thetasurf - self.temp2Am[c])))
            )
        )

        # calculate effect of soil moisture stress on gross assimilation rate
        betaw = max(1e-3, min(1.0, (self.w2 - self.wwilt) / (self.wfc - self.wwilt)))

        # calculate stress function
        if self.c_beta == 0:
            fstr = betaw
        else:
            # following Combe et al. (2016)
            if self.c_beta < 0.25:
                p = 6.4 * self.c_beta
            elif self.c_beta < 0.50:
                p = 7.6 * self.c_beta - 0.3
            else:
                p = 2 ** (3.66 * self.c_beta + 0.34) - 1
            fstr = (1.0 - np.exp(-p * betaw)) / (1 - np.exp(-p))

        # calculate gross assimilation rate (Am)
        am = ammax * (1.0 - np.exp(-(gm * (self.ci - co2comp) / ammax)))
        rdark = (1.0 / 9.0) * am
        par = 0.5 * max(1e-1, radiation.in_srad * self.cveg)

        # calculate  light use efficiency
        alphac = (
            self.alpha0[c] * (self.co2abs - co2comp) / (self.co2abs + 2.0 * co2comp)
        )

        # calculate gross primary productivity
        # limamau: this is just not being used?
        ag = (am + rdark) * (1 - np.exp(alphac * par / (am + rdark)))

        # 1.- calculate upscaling from leaf to canopy: net flow CO2 into the plant (An)
        y = alphac * self.kx[c] * par / (am + rdark)
        an = (am + rdark) * (
            1.0
            - 1.0
            / (self.kx[c] * self.lai)
            * (exp1(y * np.exp(-self.kx[c] * self.lai)) - exp1(y))
        )

        # 2.- calculate upscaling from leaf to canopy: CO2 conductance at canopy level
        a1 = 1.0 / (1.0 - self.f0[c])
        dstar = d0 / (a1 * (self.f0[c] - fmin))

        self.gcco2 = self.lai * (
            self.gmin[c] / self.nuco2q
            + a1 * fstr * an / ((self.co2abs - co2comp) * (1.0 + ds / dstar))
        )

        # calculate surface resistance for moisture and carbon dioxide
        self.rs = 1.0 / (1.6 * self.gcco2)

    def compute_co2_flux(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # CO2 soil surface flux
        self.rsCO2 = 1.0 / self.gcco2

        # calculate net flux of CO2 into the plant (An)
        an = -(self.co2abs - self.ci) / (surface_layer.ra + self.rsCO2)

        # CO2 soil surface flux
        fw = self.cw * self.wmax / (self.wg + self.wmin)
        resp = (
            self.r10
            * (1.0 - fw)
            * np.exp(self.e0 / (283.15 * 8.314) * (1.0 - 283.15 / (self.temp_soil)))
        )

        # CO2 flux
        mixed_layer.wCO2A = an * (self.const.mair / (self.const.rho * self.const.mco2))
        mixed_layer.wCO2R = resp * (
            self.const.mair / (self.const.rho * self.const.mco2)
        )
        mixed_layer.wCO2 = mixed_layer.wCO2A + mixed_layer.wCO2R
