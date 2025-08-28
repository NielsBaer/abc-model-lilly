import copy as cp

import numpy as np

from .mixed_layer import MixedLayerModel
from .surface_layer import AbstractSurfaceLayerModel
from .utils import PhysicalConstants, get_esat, get_qsat


# class for storing mixed-layer model input data
class LandSurfaceInput:
    def __init__(self):
        #########################
        # land surface parameters
        self.sw_ls = None  # land surface switch
        self.ls_type = None  # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
        self.wg = None  # volumetric water content top soil layer [m3 m-3]
        self.w2 = None  # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil = None  # temperature top soil layer [K]
        self.T2 = None  # temperature deeper soil layer [K]

        self.a = None  # Clapp and Hornberger retention curve parameter a
        self.b = None  # Clapp and Hornberger retention curve parameter b
        self.p = None  # Clapp and Hornberger retention curve parameter p
        self.CGsat = None  # saturated soil conductivity for heat

        self.wsat = None  # saturated volumetric water content ECMWF config [-]
        self.wfc = None  # volumetric water content field capacity [-]
        self.wwilt = None  # volumetric water content wilting point [-]

        self.C1sat = None
        self.C2ref = None

        self.c_beta = None  # Curvatur plant water-stress factor (0..1) [-]

        self.LAI = None  # leaf area index [-]
        self.gD = None  # correction factor transpiration for VPD [-]
        self.rsmin = None  # minimum resistance transpiration [s m-1]
        self.rssoilmin = None  # minimum resistance soil evaporation [s m-1]
        self.alpha = None  # surface albedo [-]

        self.Ts = None  # initial surface temperature [K]

        self.cveg = None  # vegetation fraction [-]
        self.Wmax = None  # thickness of water layer on wet vegetation [m]
        self.Wl = None  # equivalent water layer depth for wet vegetation [m]

        self.Lambda = None  # thermal diffusivity skin layer [-]

        # A-Gs parameters
        self.c3c4 = None  # Plant type ('c3' or 'c4')
        #########################


class Model:
    # entrainment kinematic heat flux [K m s-1]
    wthetae: float

    def __init__(
        self,
        # 0. running configuration
        dt: float,
        runtime: float,
        # 1. mixed layer
        mixed_layer: MixedLayerModel,
        # 2. surface layer
        surface_layer: AbstractSurfaceLayerModel,
        # 3. radiation
        sw_rad: bool,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
        net_rad: float,
        dFz: float,
        # 4. land surface is left as it is
        # 5. cumulus parameterization
        sw_cu: bool,
        dz_h: float,
        # old input class
        model_input: LandSurfaceInput,
    ):
        # constants
        self.const = PhysicalConstants()

        # 0. running configuration
        self.dt = dt
        self.runtime = runtime
        self.tsteps = int(np.floor(self.runtime / self.dt))
        self.t = 0

        # 1. define mixed layer model
        self.mixed_layer = mixed_layer

        # 2. surface layer
        self.surface_layer = surface_layer

        # 3. radiation
        # radiation switch
        self.sw_rad = sw_rad
        # latitude [deg]
        self.lat = lat
        # longitude [deg]
        self.lon = lon
        # day of the year [-]
        self.doy = doy
        # time of the day [-]
        self.tstart = tstart
        # cloud cover fraction [-]
        self.cc = cc
        # net radiation [W m-2]
        self.net_rad = net_rad
        # cloud top radiative divergence [W m-2]
        self.dFz = dFz
        # incoming short wave radiation [W m-2]
        self.in_srad = None
        # outgoing short wave radiation [W m-2]
        self.out_srad = None
        # incoming long wave radiation [W m-2]
        self.in_lrad = None
        # outgoing long wave radiation [W m-2]
        self.out_lrad = None

        # 4. land surface is initialized like before
        self.input = cp.deepcopy(model_input)

        # 5. cumulus parameterization
        # cumulus parameterization switch
        self.sw_cu = sw_cu
        # transition layer thickness [m]
        self.mixed_layer.dz_h = dz_h
        # cloud core fraction [-]
        self.cc_frac = 0.0
        # cloud core mass flux [m s-1]
        self.cc_mf = 0.0
        # cloud core moisture flux [kg kg-1 m s-1]
        self.mixed_layer.cc_qf = 0.0

    def run(self):
        # initialize model variables
        self.init()

        # time integrate model
        for self.t in range(self.tsteps):
            # time integrate components
            self.timestep()

    def init(self):
        # A-Gs constants and settings
        # plant type: [C3, C4]
        # CO2 compensation concentration [mg m-3]
        self.CO2comp298 = [68.5, 4.3]
        # function parameter to calculate CO2 compensation concentration [-]
        self.net_rad10CO2 = [1.5, 1.5]
        # mesophyill conductance at 298 K [mm s-1]
        self.gm298 = [7.0, 17.5]
        # CO2 maximal primary productivity [mg m-2 s-1]
        self.Ammax298 = [2.2, 1.7]
        # function parameter to calculate mesophyll conductance [-]
        self.net_rad10gm = [2.0, 2.0]
        # reference temperature to calculate mesophyll conductance gm [K]
        self.T1gm = [278.0, 286.0]
        # reference temperature to calculate mesophyll conductance gm [K]
        self.T2gm = [301.0, 309.0]
        # function parameter to calculate maximal primary profuctivity Ammax
        self.net_rad10Am = [2.0, 2.0]
        # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.T1Am = [281.0, 286.0]
        # reference temperature to calculate maximal primary profuctivity Ammax [K]
        self.T2Am = [311.0, 311.0]
        # maximum value Cfrac [-]
        self.f0 = [0.89, 0.85]
        # regression coefficient to calculate Cfrac [kPa-1]
        self.ad = [0.07, 0.15]
        # initial low light conditions [mg J-1]
        self.alpha0 = [0.017, 0.014]
        # extinction coefficient PAR [-]
        self.Kx = [0.7, 0.7]
        # cuticular (minimum) conductance [mm s-1]
        self.gmin = [0.25e-3, 0.25e-3]
        # ratio molecular viscosity water to carbon dioxide
        self.nuco2q = 1.6
        # constant water stress correction (eq. 13 Jacobs et al. 2007) [-]
        self.Cw = 0.0016
        # upper reference value soil water [-]
        self.wmax = 0.55
        # lower reference value soil water [-]
        self.wmin = 0.005
        # respiration at 10 C [mg CO2 m-2 s-1]
        self.R10 = 0.23
        # activation energy [53.3 kJ kmol-1]
        self.E0 = 53.3e3

        # Read switches
        self.sw_ls = self.input.sw_ls  # land surface switch
        self.ls_type = self.input.ls_type  # land surface paramaterization (js or ags)

        # initialize land surface
        self.wg = self.input.wg  # volumetric water content top soil layer [m3 m-3]
        self.w2 = self.input.w2  # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil = self.input.Tsoil  # temperature top soil layer [K]
        self.T2 = self.input.T2  # temperature deeper soil layer [K]

        self.a = self.input.a  # Clapp and Hornberger retention curve parameter a [-]
        self.b = self.input.b  # Clapp and Hornberger retention curve parameter b [-]
        self.p = self.input.p  # Clapp and Hornberger retention curve parameter p [-]
        self.CGsat = self.input.CGsat  # saturated soil conductivity for heat

        self.wsat = (
            self.input.wsat
        )  # saturated volumetric water content ECMWF config [-]
        self.wfc = self.input.wfc  # volumetric water content field capacity [-]
        self.wwilt = self.input.wwilt  # volumetric water content wilting point [-]

        self.C1sat = self.input.C1sat
        self.C2ref = self.input.C2ref

        self.c_beta = (
            self.input.c_beta
        )  # Curvature plant water-stress factor (0..1) [-]

        self.LAI = self.input.LAI  # leaf area index [-]
        self.gD = self.input.gD  # correction factor transpiration for VPD [-]
        self.rsmin = self.input.rsmin  # minimum resistance transpiration [s m-1]
        self.rssoilmin = (
            self.input.rssoilmin
        )  # minimum resistance soil evaporation [s m-1]
        self.alpha = self.input.alpha  # surface albedo [-]

        self.rs = 1.0e6  # resistance transpiration [s m-1]
        self.rssoil = 1.0e6  # resistance soil [s m-1]

        self.Ts = self.input.Ts  # surface temperature [K]

        self.cveg = self.input.cveg  # vegetation fraction [-]
        self.Wmax = self.input.Wmax  # thickness of water layer on wet vegetation [m]
        self.Wl = self.input.Wl  # equivalent water layer depth for wet vegetation [m]
        self.cliq = None  # wet fraction [-]

        self.Lambda = self.input.Lambda  # thermal diffusivity skin layer [-]

        self.Tsoiltend = None  # soil temperature tendency [K s-1]
        self.wgtend = None  # soil moisture tendency [m3 m-3 s-1]
        self.Wltend = None  # equivalent liquid water tendency [m s-1]

        self.H = None  # sensible heat flux [W m-2]
        self.LE = None  # evapotranspiration [W m-2]
        self.LEliq = None  # open water evaporation [W m-2]
        self.LEveg = None  # transpiration [W m-2]
        self.LEsoil = None  # soil evaporation [W m-2]
        self.LEpot = None  # potential evaporation [W m-2]
        self.LEref = None  # reference evaporation using rs = rsmin / LAI [W m-2]
        self.G = None  # ground heat flux [W m-2]

        # initialize A-Gs surface scheme
        self.c3c4 = self.input.c3c4  # plant type ('c3' or 'c4')

        # Some sanity checks for valid input
        if self.c_beta is None:
            self.c_beta = 0  # Zero curvature; linear response
        assert self.c_beta >= 0 or self.c_beta <= 1

        # initialize output
        self.out = ModelOutput(self.tsteps)

        self.statistics()

        # calculate initial diagnostic variables
        if self.sw_rad:
            self.run_radiation()

        for _ in range(10):
            assert isinstance(self.mixed_layer.thetav, float)
            self.surface_layer.run(
                self.mixed_layer.u,
                self.mixed_layer.v,
                self.mixed_layer.theta,
                self.mixed_layer.thetav,
                self.mixed_layer.wstar,
                self.mixed_layer.wtheta,
                self.mixed_layer.wq,
                self.mixed_layer.surf_pressure,
                self.rs,
                self.mixed_layer.q,
                self.mixed_layer.abl_height,
            )

        if self.sw_ls:
            self.run_land_surface()

        assert isinstance(self.surface_layer.uw, float)
        assert isinstance(self.surface_layer.vw, float)
        if self.sw_cu:
            self.mixed_layer.run(
                self.dFz,
                self.cc_mf,
                self.cc_frac,
                self.surface_layer.ustar,
                self.surface_layer.uw,
                self.surface_layer.vw,
            )
            self.run_cumulus()

        if self.mixed_layer.sw_ml:
            self.mixed_layer.run(
                self.dFz,
                self.cc_mf,
                self.cc_frac,
                self.surface_layer.ustar,
                self.surface_layer.uw,
                self.surface_layer.vw,
            )

    def timestep(self):
        self.statistics()

        # run radiation model
        if self.sw_rad:
            self.run_radiation()

        # run surface layer model
        assert isinstance(self.mixed_layer.thetav, float)
        self.surface_layer.run(
            self.mixed_layer.u,
            self.mixed_layer.v,
            self.mixed_layer.theta,
            self.mixed_layer.thetav,
            self.mixed_layer.wstar,
            self.mixed_layer.wtheta,
            self.mixed_layer.wq,
            self.mixed_layer.surf_pressure,
            self.rs,
            self.mixed_layer.q,
            self.mixed_layer.abl_height,
        )

        # run land surface model
        if self.sw_ls:
            self.run_land_surface()

        # run cumulus parameterization
        if self.sw_cu:
            self.run_cumulus()

        # run mixed-layer model
        if self.mixed_layer.sw_ml:
            self.mixed_layer.run(
                self.dFz,
                self.cc_mf,
                self.cc_frac,
                self.surface_layer.ustar,
                self.surface_layer.uw,
                self.surface_layer.vw,
            )

        # store output before time integration
        self.store()

        # time integrate land surface model
        if self.sw_ls:
            self.integrate_land_surface()

        # time integrate mixed-layer model
        if self.mixed_layer.sw_ml:
            self.mixed_layer.integrate(self.dt)

    def statistics(self):
        # Calculate virtual temperatures
        self.mixed_layer.thetav = (
            self.mixed_layer.theta + 0.61 * self.mixed_layer.theta * self.mixed_layer.q
        )
        self.mixed_layer.wthetav = (
            self.mixed_layer.wtheta
            + 0.61 * self.mixed_layer.theta * self.mixed_layer.wq
        )
        self.mixed_layer.dthetav = (
            self.mixed_layer.theta + self.mixed_layer.dtheta
        ) * (
            1.0 + 0.61 * (self.mixed_layer.q + self.mixed_layer.dq)
        ) - self.mixed_layer.theta * (1.0 + 0.61 * self.mixed_layer.q)

        # Mixed-layer top properties
        self.mixed_layer.top_p = (
            self.mixed_layer.surf_pressure
            - self.const.rho * self.const.g * self.mixed_layer.abl_height
        )
        self.mixed_layer.top_T = (
            self.mixed_layer.theta
            - self.const.g / self.const.cp * self.mixed_layer.abl_height
        )
        self.mixed_layer.top_rh = self.mixed_layer.q / get_qsat(
            self.mixed_layer.top_T, self.mixed_layer.top_p
        )

        # Find lifting condensation level iteratively
        if self.t == 0:
            self.mixed_layer.lcl = self.mixed_layer.abl_height
            RHlcl = 0.5
        else:
            RHlcl = 0.9998

        itmax = 30
        it = 0
        while ((RHlcl <= 0.9999) or (RHlcl >= 1.0001)) and it < itmax:
            self.mixed_layer.lcl += (1.0 - RHlcl) * 1000.0
            p_lcl = (
                self.mixed_layer.surf_pressure
                - self.const.rho * self.const.g * self.mixed_layer.lcl
            )
            T_lcl = (
                self.mixed_layer.theta
                - self.const.g / self.const.cp * self.mixed_layer.lcl
            )
            RHlcl = self.mixed_layer.q / get_qsat(T_lcl, p_lcl)
            it += 1

        if it == itmax:
            print("LCL calculation not converged!!")
            print("RHlcl = %f, zlcl=%f" % (RHlcl, self.mixed_layer.lcl))

    def run_cumulus(self):
        # Calculate mixed-layer top relative humidity variance (Neggers et. al 2006/7)
        if self.mixed_layer.wthetav > 0:
            self.mixed_layer.q2_h = (
                -(self.mixed_layer.wqe + self.mixed_layer.cc_qf)
                * self.mixed_layer.dq
                * self.mixed_layer.abl_height
                / (self.mixed_layer.dz_h * self.mixed_layer.wstar)
            )
            self.mixed_layer.top_CO22 = (
                -(self.mixed_layer.wCO2e + self.mixed_layer.wCO2M)
                * self.mixed_layer.dCO2
                * self.mixed_layer.abl_height
                / (self.mixed_layer.dz_h * self.mixed_layer.wstar)
            )
        else:
            self.mixed_layer.q2_h = 0.0
            self.mixed_layer.top_CO22 = 0.0

        # calculate cloud core fraction (ac), mass flux (M) and moisture flux (wqM)
        self.cc_frac = max(
            0.0,
            0.5
            + (
                0.36
                * np.arctan(
                    1.55
                    * (
                        (
                            self.mixed_layer.q
                            - get_qsat(self.mixed_layer.top_T, self.mixed_layer.top_p)
                        )
                        / self.mixed_layer.q2_h**0.5
                    )
                )
            ),
        )
        self.cc_mf = self.cc_frac * self.mixed_layer.wstar
        self.mixed_layer.cc_qf = self.cc_mf * self.mixed_layer.q2_h**0.5

        # Only calculate CO2 mass-flux if mixed-layer top jump is negative
        if self.mixed_layer.dCO2 < 0:
            self.mixed_layer.wCO2M = self.cc_mf * self.mixed_layer.top_CO22**0.5
        else:
            self.mixed_layer.wCO2M = 0.0

    def run_radiation(self):
        sda = 0.409 * np.cos(2.0 * np.pi * (self.doy - 173.0) / 365.0)
        sinlea = np.sin(2.0 * np.pi * self.lat / 360.0) * np.sin(sda) - np.cos(
            2.0 * np.pi * self.lat / 360.0
        ) * np.cos(sda) * np.cos(
            2.0 * np.pi * (self.t * self.dt + self.tstart * 3600.0) / 86400.0
            + 2.0 * np.pi * self.lon / 360.0
        )
        sinlea = max(sinlea, 0.0001)

        Ta = self.mixed_layer.theta * (
            (
                self.mixed_layer.surf_pressure
                - 0.1 * self.mixed_layer.abl_height * self.const.rho * self.const.g
            )
            / self.mixed_layer.surf_pressure
        ) ** (self.const.rd / self.const.cp)

        Tr = (0.6 + 0.2 * sinlea) * (1.0 - 0.4 * self.cc)

        self.in_srad = self.const.solar_in * Tr * sinlea
        self.out_srad = self.alpha * self.const.solar_in * Tr * sinlea
        self.in_lrad = 0.8 * self.const.bolz * Ta**4.0
        self.out_lrad = self.const.bolz * self.Ts**4.0

        self.net_rad = self.in_srad - self.out_srad + self.in_lrad - self.out_lrad

    def jarvis_stewart(self):
        # calculate surface resistances using Jarvis-Stewart model
        if self.sw_rad:
            f1 = 1.0 / min(
                1.0,
                ((0.004 * self.in_srad + 0.05) / (0.81 * (0.004 * self.in_srad + 1.0))),
            )
        else:
            f1 = 1.0

        if self.w2 > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # Limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(
            -self.gD * (self.mixed_layer.esat - self.mixed_layer.e) / 100.0
        )
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - self.mixed_layer.theta) ** 2.0)

        self.rs = self.rsmin / self.LAI * f1 * f2 * f3 * f4

    def factorial(self, k):
        factorial = 1
        for n in range(2, k + 1):
            factorial = factorial * float(n)
        return factorial

    def E1(self, x):
        E1sum = 0
        for k in range(1, 100):
            E1sum += (
                pow((-1.0), (k + 0.0))
                * pow(x, (k + 0.0))
                / ((k + 0.0) * self.factorial(k))
            )
        return -0.57721566490153286060 - np.log(x) - E1sum

    def ags(self):
        # Select index for plant type
        if self.c3c4 == "c3":
            c = 0
        elif self.c3c4 == "c4":
            c = 1
        else:
            raise ValueError(f'Invalid option "{self.c3c4}" for "c3c4".')

        # calculate CO2 compensation concentration
        CO2comp = (
            self.CO2comp298[c]
            * self.const.rho
            * pow(self.net_rad10CO2[c], (0.1 * (self.mixed_layer.thetasurf - 298.0)))
        )

        # calculate mesophyll conductance
        gm = (
            self.gm298[c]
            * pow(self.net_rad10gm[c], (0.1 * (self.mixed_layer.thetasurf - 298.0)))
            / (
                (1.0 + np.exp(0.3 * (self.T1gm[c] - self.mixed_layer.thetasurf)))
                * (1.0 + np.exp(0.3 * (self.mixed_layer.thetasurf - self.T2gm[c])))
            )
        )
        gm = gm / 1000.0  # conversion from mm s-1 to m s-1

        # calculate CO2 concentration inside the leaf (ci)
        fmin0 = self.gmin[c] / self.nuco2q - 1.0 / 9.0 * gm
        fmin = -fmin0 + pow(
            (pow(fmin0, 2.0) + 4 * self.gmin[c] / self.nuco2q * gm), 0.5
        ) / (2.0 * gm)

        Ds = (get_esat(self.Ts) - self.mixed_layer.e) / 1000.0  # kPa
        D0 = (self.f0[c] - fmin) / self.ad[c]

        cfrac = self.f0[c] * (1.0 - (Ds / D0)) + fmin * (Ds / D0)
        co2abs = (
            self.mixed_layer.co2 * (self.const.mco2 / self.const.mair) * self.const.rho
        )  # conversion mumol mol-1 (ppm) to mgCO2 m3
        ci = cfrac * (co2abs - CO2comp) + CO2comp

        # calculate maximal gross primary production in high light conditions (Ag)
        Ammax = (
            self.Ammax298[c]
            * pow(self.net_rad10Am[c], (0.1 * (self.mixed_layer.thetasurf - 298.0)))
            / (
                (1.0 + np.exp(0.3 * (self.T1Am[c] - self.mixed_layer.thetasurf)))
                * (1.0 + np.exp(0.3 * (self.mixed_layer.thetasurf - self.T2Am[c])))
            )
        )

        # calculate effect of soil moisture stress on gross assimilation rate
        betaw = max(1e-3, min(1.0, (self.w2 - self.wwilt) / (self.wfc - self.wwilt)))

        # calculate stress function
        if self.c_beta == 0:
            fstr = betaw
        else:
            # Following Combe et al (2016)
            if self.c_beta < 0.25:
                P = 6.4 * self.c_beta
            elif self.c_beta < 0.50:
                P = 7.6 * self.c_beta - 0.3
            else:
                P = 2 ** (3.66 * self.c_beta + 0.34) - 1
            fstr = (1.0 - np.exp(-P * betaw)) / (1 - np.exp(-P))

        # calculate gross assimilation rate (Am)
        Am = Ammax * (1.0 - np.exp(-(gm * (ci - CO2comp) / Ammax)))
        Rdark = (1.0 / 9.0) * Am
        PAR = 0.5 * max(1e-1, self.in_srad * self.cveg)

        # calculate  light use efficiency
        alphac = self.alpha0[c] * (co2abs - CO2comp) / (co2abs + 2.0 * CO2comp)

        # calculate gross primary productivity
        Ag = (Am + Rdark) * (1 - np.exp(alphac * PAR / (Am + Rdark)))

        # 1.- calculate upscaling from leaf to canopy: net flow CO2 into the plant (An)
        y = alphac * self.Kx[c] * PAR / (Am + Rdark)
        An = (Am + Rdark) * (
            1.0
            - 1.0
            / (self.Kx[c] * self.LAI)
            * (self.E1(y * np.exp(-self.Kx[c] * self.LAI)) - self.E1(y))
        )

        # 2.- calculate upscaling from leaf to canopy: CO2 conductance at canopy level
        a1 = 1.0 / (1.0 - self.f0[c])
        Dstar = D0 / (a1 * (self.f0[c] - fmin))

        gcco2 = self.LAI * (
            self.gmin[c] / self.nuco2q
            + a1 * fstr * An / ((co2abs - CO2comp) * (1.0 + Ds / Dstar))
        )

        # calculate surface resistance for moisture and carbon dioxide
        self.rs = 1.0 / (1.6 * gcco2)
        rsCO2 = 1.0 / gcco2

        # calculate net flux of CO2 into the plant (An)
        An = -(co2abs - ci) / (self.surface_layer.ra + rsCO2)

        # CO2 soil surface flux
        fw = self.Cw * self.wmax / (self.wg + self.wmin)
        Resp = (
            self.R10
            * (1.0 - fw)
            * np.exp(self.E0 / (283.15 * 8.314) * (1.0 - 283.15 / (self.Tsoil)))
        )

        # CO2 flux
        self.mixed_layer.wCO2A = An * (
            self.const.mair / (self.const.rho * self.const.mco2)
        )
        self.mixed_layer.wCO2R = Resp * (
            self.const.mair / (self.const.rho * self.const.mco2)
        )
        self.mixed_layer.wCO2 = self.mixed_layer.wCO2A + self.mixed_layer.wCO2R

    def run_land_surface(self):
        self.surface_layer.compute_ra(
            self.mixed_layer.u, self.mixed_layer.v, self.mixed_layer.wstar
        )

        # first calculate essential thermodynamic variables
        self.mixed_layer.esat = get_esat(self.mixed_layer.theta)
        self.mixed_layer.qsat = get_qsat(
            self.mixed_layer.theta, self.mixed_layer.surf_pressure
        )
        desatdT = self.mixed_layer.esat * (
            17.2694 / (self.mixed_layer.theta - 35.86)
            - 17.2694
            * (self.mixed_layer.theta - 273.16)
            / (self.mixed_layer.theta - 35.86) ** 2.0
        )
        self.mixed_layer.dqsatdT = 0.622 * desatdT / self.mixed_layer.surf_pressure
        self.mixed_layer.e = self.mixed_layer.q * self.mixed_layer.surf_pressure / 0.622

        if self.ls_type == "js":
            self.jarvis_stewart()
        elif self.ls_type == "ags":
            self.ags()
        else:
            raise ValueError(f'Inavalid option "{self.ls_type}" for "ls_type".')

        # recompute f2 using wg instead of w2
        if self.wg > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
            f2 = 1.0e8
        self.rssoil = self.rssoilmin * f2

        Wlmx = self.LAI * self.Wmax
        self.cliq = min(1.0, self.Wl / Wlmx)

        # calculate skin temperature implictly
        self.Ts = (
            self.net_rad
            + self.const.rho
            * self.const.cp
            / self.surface_layer.ra
            * self.mixed_layer.theta
            + self.cveg
            * (1.0 - self.cliq)
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rs)
            * (
                self.mixed_layer.dqsatdT * self.mixed_layer.theta
                - self.mixed_layer.qsat
                + self.mixed_layer.q
            )
            + (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rssoil)
            * (
                self.mixed_layer.dqsatdT * self.mixed_layer.theta
                - self.mixed_layer.qsat
                + self.mixed_layer.q
            )
            + self.cveg
            * self.cliq
            * self.const.rho
            * self.const.lv
            / self.surface_layer.ra
            * (
                self.mixed_layer.dqsatdT * self.mixed_layer.theta
                - self.mixed_layer.qsat
                + self.mixed_layer.q
            )
            + self.Lambda * self.Tsoil
        ) / (
            self.const.rho * self.const.cp / self.surface_layer.ra
            + self.cveg
            * (1.0 - self.cliq)
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rs)
            * self.mixed_layer.dqsatdT
            + (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rssoil)
            * self.mixed_layer.dqsatdT
            + self.cveg
            * self.cliq
            * self.const.rho
            * self.const.lv
            / self.surface_layer.ra
            * self.mixed_layer.dqsatdT
            + self.Lambda
        )

        esatsurf = get_esat(self.Ts)
        self.mixed_layer.qsatsurf = get_qsat(self.Ts, self.mixed_layer.surf_pressure)

        self.LEveg = (
            (1.0 - self.cliq)
            * self.cveg
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rs)
            * (
                self.mixed_layer.dqsatdT * (self.Ts - self.mixed_layer.theta)
                + self.mixed_layer.qsat
                - self.mixed_layer.q
            )
        )
        self.LEliq = (
            self.cliq
            * self.cveg
            * self.const.rho
            * self.const.lv
            / self.surface_layer.ra
            * (
                self.mixed_layer.dqsatdT * (self.Ts - self.mixed_layer.theta)
                + self.mixed_layer.qsat
                - self.mixed_layer.q
            )
        )
        self.LEsoil = (
            (1.0 - self.cveg)
            * self.const.rho
            * self.const.lv
            / (self.surface_layer.ra + self.rssoil)
            * (
                self.mixed_layer.dqsatdT * (self.Ts - self.mixed_layer.theta)
                + self.mixed_layer.qsat
                - self.mixed_layer.q
            )
        )

        self.Wltend = -self.LEliq / (self.const.rhow * self.const.lv)

        self.LE = self.LEsoil + self.LEveg + self.LEliq
        self.H = (
            self.const.rho
            * self.const.cp
            / self.surface_layer.ra
            * (self.Ts - self.mixed_layer.theta)
        )
        self.G = self.Lambda * (self.Ts - self.Tsoil)
        self.LEpot = (
            self.mixed_layer.dqsatdT * (self.net_rad - self.G)
            + self.const.rho
            * self.const.cp
            / self.surface_layer.ra
            * (self.mixed_layer.qsat - self.mixed_layer.q)
        ) / (self.mixed_layer.dqsatdT + self.const.cp / self.const.lv)
        self.LEref = (
            self.mixed_layer.dqsatdT * (self.net_rad - self.G)
            + self.const.rho
            * self.const.cp
            / self.surface_layer.ra
            * (self.mixed_layer.qsat - self.mixed_layer.q)
        ) / (
            self.mixed_layer.dqsatdT
            + self.const.cp
            / self.const.lv
            * (1.0 + self.rsmin / self.LAI / self.surface_layer.ra)
        )

        CG = self.CGsat * (self.wsat / self.w2) ** (self.b / (2.0 * np.log(10.0)))

        self.Tsoiltend = CG * self.G - 2.0 * np.pi / 86400.0 * (self.Tsoil - self.T2)

        d1 = 0.1
        C1 = self.C1sat * (self.wsat / self.wg) ** (self.b / 2.0 + 1.0)
        C2 = self.C2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        self.wgtend = -C1 / (
            self.const.rhow * d1
        ) * self.LEsoil / self.const.lv - C2 / 86400.0 * (self.wg - wgeq)

        # calculate kinematic heat fluxes
        self.mixed_layer.wtheta = self.H / (self.const.rho * self.const.cp)
        self.mixed_layer.wq = self.LE / (self.const.rho * self.const.lv)

    def integrate_land_surface(self):
        # integrate soil equations
        Tsoil0 = self.Tsoil
        wg0 = self.wg
        Wl0 = self.Wl

        self.Tsoil = Tsoil0 + self.dt * self.Tsoiltend
        self.wg = wg0 + self.dt * self.wgtend
        self.Wl = Wl0 + self.dt * self.Wltend

    # store model output
    def store(self):
        t = self.t
        self.out.t[t] = t * self.dt / 3600.0 + self.tstart
        self.out.h[t] = self.mixed_layer.abl_height

        self.out.theta[t] = self.mixed_layer.theta
        self.out.thetav[t] = self.mixed_layer.thetav
        self.out.dtheta[t] = self.mixed_layer.dtheta
        self.out.dthetav[t] = self.mixed_layer.dthetav
        self.out.wtheta[t] = self.mixed_layer.wtheta
        self.out.wthetav[t] = self.mixed_layer.wthetav
        self.out.wthetae[t] = self.mixed_layer.wthetae
        self.out.wthetave[t] = self.mixed_layer.wthetave

        self.out.q[t] = self.mixed_layer.q
        self.out.dq[t] = self.mixed_layer.dq
        self.out.wq[t] = self.mixed_layer.wq
        self.out.wqe[t] = self.mixed_layer.wqe
        self.out.wqM[t] = self.mixed_layer.cc_qf

        self.out.qsat[t] = self.mixed_layer.qsat
        self.out.e[t] = self.mixed_layer.e
        self.out.esat[t] = self.mixed_layer.esat

        fac = (self.const.rho * self.const.mco2) / self.const.mair
        self.out.CO2[t] = self.mixed_layer.co2
        self.out.dCO2[t] = self.mixed_layer.dCO2
        self.out.wCO2[t] = self.mixed_layer.wCO2 * fac
        self.out.wCO2e[t] = self.mixed_layer.wCO2e * fac
        self.out.wCO2R[t] = self.mixed_layer.wCO2R * fac
        self.out.wCO2A[t] = self.mixed_layer.wCO2A * fac

        self.out.u[t] = self.mixed_layer.u
        self.out.du[t] = self.mixed_layer.du
        self.out.uw[t] = self.surface_layer.uw

        self.out.v[t] = self.mixed_layer.v
        self.out.dv[t] = self.mixed_layer.dv
        self.out.vw[t] = self.surface_layer.vw

        self.out.T2m[t] = self.mixed_layer.temp_2m
        self.out.q2m[t] = self.mixed_layer.q2m
        self.out.u2m[t] = self.mixed_layer.u2m
        self.out.v2m[t] = self.mixed_layer.v2m
        self.out.e2m[t] = self.mixed_layer.e2m
        self.out.esat2m[t] = self.mixed_layer.esat2m

        self.out.thetasurf[t] = self.mixed_layer.thetasurf
        self.out.thetavsurf[t] = self.mixed_layer.thetavsurf
        self.out.qsurf[t] = self.mixed_layer.qsurf
        self.out.ustar[t] = self.surface_layer.ustar
        self.out.Cm[t] = self.surface_layer.drag_m
        self.out.Cs[t] = self.surface_layer.drag_s
        self.out.L[t] = self.surface_layer.obukhov_length
        self.out.Rib[t] = self.surface_layer.rib_number

        self.out.Swin[t] = self.in_srad
        self.out.Swout[t] = self.out_srad
        self.out.Lwin[t] = self.in_lrad
        self.out.Lwout[t] = self.out_lrad
        self.out.net_rad[t] = self.net_rad

        self.out.ra[t] = self.surface_layer.ra
        self.out.rs[t] = self.rs
        self.out.H[t] = self.H
        self.out.LE[t] = self.LE
        self.out.LEliq[t] = self.LEliq
        self.out.LEveg[t] = self.LEveg
        self.out.LEsoil[t] = self.LEsoil
        self.out.LEpot[t] = self.LEpot
        self.out.LEref[t] = self.LEref
        self.out.G[t] = self.G

        self.out.zlcl[t] = self.mixed_layer.lcl
        self.out.RH_h[t] = self.mixed_layer.top_rh

        self.out.ac[t] = self.cc_frac
        self.out.M[t] = self.cc_mf
        self.out.dz[t] = self.mixed_layer.dz_h


# class for storing mixed-layer model output data
class ModelOutput:
    def __init__(self, tsteps):
        self.t = np.zeros(tsteps)  # time [s]

        # mixed-layer variables
        self.h = np.zeros(tsteps)  # ABL height [m]

        self.theta = np.zeros(tsteps)  # initial mixed-layer potential temperature [K]
        self.thetav = np.zeros(
            tsteps
        )  # initial mixed-layer virtual potential temperature [K]
        self.dtheta = np.zeros(tsteps)  # initial potential temperature jump at h [K]
        self.dthetav = np.zeros(
            tsteps
        )  # initial virtual potential temperature jump at h [K]
        self.wtheta = np.zeros(tsteps)  # surface kinematic heat flux [K m s-1]
        self.wthetav = np.zeros(tsteps)  # surface kinematic virtual heat flux [K m s-1]
        self.wthetae = np.zeros(tsteps)  # entrainment kinematic heat flux [K m s-1]
        self.wthetave = np.zeros(
            tsteps
        )  # entrainment kinematic virtual heat flux [K m s-1]

        self.q = np.zeros(tsteps)  # mixed-layer specific humidity [kg kg-1]
        self.dq = np.zeros(tsteps)  # initial specific humidity jump at h [kg kg-1]
        self.wq = np.zeros(tsteps)  # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wqe = np.zeros(
            tsteps
        )  # entrainment kinematic moisture flux [kg kg-1 m s-1]
        self.wqM = np.zeros(
            tsteps
        )  # cumulus mass-flux kinematic moisture flux [kg kg-1 m s-1]

        self.qsat = np.zeros(
            tsteps
        )  # mixed-layer saturated specific humidity [kg kg-1]
        self.e = np.zeros(tsteps)  # mixed-layer vapor pressure [Pa]
        self.esat = np.zeros(tsteps)  # mixed-layer saturated vapor pressure [Pa]

        self.CO2 = np.zeros(tsteps)  # mixed-layer CO2 [ppm]
        self.dCO2 = np.zeros(tsteps)  # initial CO2 jump at h [ppm]
        self.wCO2 = np.zeros(tsteps)  # surface total CO2 flux [mgC m-2 s-1]
        self.wCO2A = np.zeros(tsteps)  # surface assimilation CO2 flux [mgC m-2 s-1]
        self.wCO2R = np.zeros(tsteps)  # surface respiration CO2 flux [mgC m-2 s-1]
        self.wCO2e = np.zeros(tsteps)  # entrainment CO2 flux [mgC m-2 s-1]
        self.wCO2M = np.zeros(tsteps)  # CO2 mass flux [mgC m-2 s-1]

        self.u = np.zeros(tsteps)  # initial mixed-layer u-wind speed [m s-1]
        self.du = np.zeros(tsteps)  # initial u-wind jump at h [m s-1]
        self.uw = np.zeros(tsteps)  # surface momentum flux u [m2 s-2]

        self.v = np.zeros(tsteps)  # initial mixed-layer u-wind speed [m s-1]
        self.dv = np.zeros(tsteps)  # initial u-wind jump at h [m s-1]
        self.vw = np.zeros(tsteps)  # surface momentum flux v [m2 s-2]

        # diagnostic meteorological variables
        self.T2m = np.zeros(tsteps)  # 2m temperature [K]
        self.q2m = np.zeros(tsteps)  # 2m specific humidity [kg kg-1]
        self.u2m = np.zeros(tsteps)  # 2m u-wind [m s-1]
        self.v2m = np.zeros(tsteps)  # 2m v-wind [m s-1]
        self.e2m = np.zeros(tsteps)  # 2m vapor pressure [Pa]
        self.esat2m = np.zeros(tsteps)  # 2m saturated vapor pressure [Pa]

        # surface-layer variables
        self.thetasurf = np.zeros(tsteps)  # surface potential temperature [K]
        self.thetavsurf = np.zeros(tsteps)  # surface virtual potential temperature [K]
        self.qsurf = np.zeros(tsteps)  # surface specific humidity [kg kg-1]
        self.ustar = np.zeros(tsteps)  # surface friction velocity [m s-1]
        self.z0m = np.zeros(tsteps)  # roughness length for momentum [m]
        self.z0h = np.zeros(tsteps)  # roughness length for scalars [m]
        self.Cm = np.zeros(tsteps)  # drag coefficient for momentum []
        self.Cs = np.zeros(tsteps)  # drag coefficient for scalars []
        self.L = np.zeros(tsteps)  # Obukhov length [m]
        self.Rib = np.zeros(tsteps)  # bulk Richardson number [-]

        # radiation variables
        self.Swin = np.zeros(tsteps)  # incoming short wave radiation [W m-2]
        self.Swout = np.zeros(tsteps)  # outgoing short wave radiation [W m-2]
        self.Lwin = np.zeros(tsteps)  # incoming long wave radiation [W m-2]
        self.Lwout = np.zeros(tsteps)  # outgoing long wave radiation [W m-2]
        self.net_rad = np.zeros(tsteps)  # net radiation [W m-2]

        # land surface variables
        self.ra = np.zeros(tsteps)  # aerodynamic resistance [s m-1]
        self.rs = np.zeros(tsteps)  # surface resistance [s m-1]
        self.H = np.zeros(tsteps)  # sensible heat flux [W m-2]
        self.LE = np.zeros(tsteps)  # evapotranspiration [W m-2]
        self.LEliq = np.zeros(tsteps)  # open water evaporation [W m-2]
        self.LEveg = np.zeros(tsteps)  # transpiration [W m-2]
        self.LEsoil = np.zeros(tsteps)  # soil evaporation [W m-2]
        self.LEpot = np.zeros(tsteps)  # potential evaporation [W m-2]
        self.LEref = np.zeros(
            tsteps
        )  # reference evaporation at rs = rsmin / LAI [W m-2]
        self.G = np.zeros(tsteps)  # ground heat flux [W m-2]

        # Mixed-layer top variables
        self.zlcl = np.zeros(tsteps)  # lifting condensation level [m]
        self.RH_h = np.zeros(tsteps)  # mixed-layer top relative humidity [-]

        # cumulus variables
        self.ac = np.zeros(tsteps)  # cloud core fraction [-]
        self.M = np.zeros(tsteps)  # cloud core mass flux [m s-1]
        self.dz = np.zeros(tsteps)  # transition layer thickness [m]
