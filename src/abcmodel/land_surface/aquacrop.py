import numpy as np
from scipy.special import exp1

from ..mixed_layer import AbstractMixedLayerModel
from ..radiation import AbstractRadiationModel
from ..surface_layer import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants, get_esat
from .standard import AbstractStandardLandSurfaceModel


class AquaCropModel(AbstractStandardLandSurfaceModel):
    rsCO2: float
    gcco2: float
    ci: float

    def __init__(
        self,
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
        # A-Gs constants and settings
        # plant type: [C3, C4]
        if c3c4 == "c3":
            self.c3c4 = 0
        elif c3c4 == "c4":
            self.c3c4 = 1
        else:
            raise ValueError(f'Invalid option "{c3c4}" for "c3c4".')

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

        super().__init__(
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
        )

    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # calculate CO2 compensation concentration
        co2comp = (
            self.co2comp298[self.c3c4]
            * const.rho
            * pow(
                self.net_rad10CO2[self.c3c4], (0.1 * (surface_layer.thetasurf - 298.0))
            )
        )

        # calculate mesophyll conductance
        gm = (
            self.gm298[self.c3c4]
            * pow(
                self.net_rad10gm[self.c3c4], (0.1 * (surface_layer.thetasurf - 298.0))
            )
            / (
                (
                    1.0
                    + np.exp(0.3 * (self.temp1gm[self.c3c4] - surface_layer.thetasurf))
                )
                * (
                    1.0
                    + np.exp(0.3 * (surface_layer.thetasurf - self.temp2gm[self.c3c4]))
                )
            )
        )
        # conversion from mm s-1 to m s-1
        gm = gm / 1000.0

        # calculate CO2 concentration inside the leaf (ci)
        fmin0 = self.gmin[self.c3c4] / self.nuco2q - 1.0 / 9.0 * gm
        fmin = -fmin0 + pow(
            (pow(fmin0, 2.0) + 4 * self.gmin[self.c3c4] / self.nuco2q * gm), 0.5
        ) / (2.0 * gm)

        ds = (get_esat(self.surf_temp) - mixed_layer.e) / 1000.0  # kPa
        d0 = (self.f0[self.c3c4] - fmin) / self.ad[self.c3c4]

        cfrac = self.f0[self.c3c4] * (1.0 - (ds / d0)) + fmin * (ds / d0)
        self.co2abs = mixed_layer.co2 * (const.mco2 / const.mair) * const.rho
        # conversion mumol mol-1 (ppm) to mgCO2 m3
        self.ci = cfrac * (self.co2abs - co2comp) + co2comp

        # calculate maximal gross primary production in high light conditions (Ag)
        ammax = (
            self.ammax298[self.c3c4]
            * pow(
                self.net_rad10Am[self.c3c4], (0.1 * (surface_layer.thetasurf - 298.0))
            )
            / (
                (
                    1.0
                    + np.exp(0.3 * (self.temp1Am[self.c3c4] - surface_layer.thetasurf))
                )
                * (
                    1.0
                    + np.exp(0.3 * (surface_layer.thetasurf - self.temp2Am[self.c3c4]))
                )
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
            self.alpha0[self.c3c4]
            * (self.co2abs - co2comp)
            / (self.co2abs + 2.0 * co2comp)
        )

        # calculate gross primary productivity
        # limamau: this is just not being used?
        # ag = (am + rdark) * (1 - np.exp(alphac * par / (am + rdark)))

        # 1.- calculate upscaling from leaf to canopy: net flow CO2 into the plant (An)
        y = alphac * self.kx[self.c3c4] * par / (am + rdark)
        an = (am + rdark) * (
            1.0
            - 1.0
            / (self.kx[self.c3c4] * self.lai)
            * (exp1(y * np.exp(-self.kx[self.c3c4] * self.lai)) - exp1(y))
        )

        # 2.- calculate upscaling from leaf to canopy: CO2 conductance at canopy level
        a1 = 1.0 / (1.0 - self.f0[self.c3c4])
        dstar = d0 / (a1 * (self.f0[self.c3c4] - fmin))

        self.gcco2 = self.lai * (
            self.gmin[self.c3c4] / self.nuco2q
            + a1 * fstr * an / ((self.co2abs - co2comp) * (1.0 + ds / dstar))
        )

        # calculate surface resistance for moisture and carbon dioxide
        self.rs = 1.0 / (1.6 * self.gcco2)

    def compute_co2_flux(
        self,
        const: PhysicalConstants,
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
        mixed_layer.wCO2A = an * (const.mair / (const.rho * const.mco2))
        mixed_layer.wCO2R = resp * (const.mair / (const.rho * const.mco2))
        mixed_layer.wCO2 = mixed_layer.wCO2A + mixed_layer.wCO2R
