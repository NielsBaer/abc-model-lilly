# limamau: this module contains the abstract classes for all main components of the ABC
# model. Another architecture possibility is to define each abstract class in their own
# files and extract kwargs from each run method - the coupler would manage integration...
# I'm leaving this idea for the furure in case we need/decide to go for it.
# limamau: in the future, it will be good to define parameter and initial conditions classes
# to simplifiy initializations. We should also somehow check whether all parameters of a component
# are being computed from one of the other components, otherwise throw a coupling error. I think
# bringing initalization out of the abstract classes is a good step in this direction.

from abc import abstractmethod

from .utils import PhysicalConstants, get_qsat


# limamau: redefine init outside abstract
class AbstractRadiationModel:
    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
        net_rad: float,
        dFz: float,
    ):
        # constants
        self.const = PhysicalConstants()
        # latitude [deg]
        self.lat = lat
        # longitude [deg]
        self.lon = lon
        # day of the year [-]
        self.doy = doy
        # time of the day [h UTC]
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

    @abstractmethod
    def run(
        self,
        t: float,
        dt: float,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_f1(self) -> float:
        pass


class AbstractLandSurfaceModel:
    @abstractmethod
    def run(
        self,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    def integrate(self, dt: float) -> None:
        pass


# limamau: redefine init outside abstract
class AbstractSurfaceLayerModel:
    def __init__(
        self,
        ustar: float,
        z0m: float,
        z0h: float,
        theta: float,
    ):
        # constants
        self.const = PhysicalConstants()
        # surface friction velocity [m s-1]
        self.ustar = ustar
        # roughness length for momentum [m]
        self.z0m = z0m
        # roughness length for scalars [m]
        self.z0h = z0h
        # surface momentum flux in u-direction [m2 s-2]
        self.uw = None
        # surface momentum flux in v-direction [m2 s-2]
        self.vw = None
        # drag coefficient for momentum [-]
        self.drag_m = 1e12
        # drag coefficient for scalars [-]
        self.drag_s = 1e12
        # Obukhov length [m]
        self.obukhov_length = None
        # bulk Richardson number [-]
        self.rib_number = None
        # aerodynamic resistance [s m-1]
        self.ra = None
        # 2m diagnostic variables:
        # 2m temperature [K]
        self.temp_2m = None
        # 2m specific humidity [kg kg-1]
        self.q2m = None
        # 2m vapor pressure [Pa]
        self.e2m = None
        # 2m saturated vapor pressure [Pa]
        self.esat2m = None
        # 2m u-wind [m s-1]
        self.u2m = None
        # 2m v-wind [m s-1]
        self.v2m = None
        # surface variables:
        # surface potential temperature [K]
        self.thetasurf = theta
        # surface virtual potential temperature [K]
        self.thetavsurf = None
        # surface specific humidity [g kg-1]
        self.qsurf = None

    @abstractmethod
    def run(
        self,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_ra(self, u: float, v: float, wstar: float) -> None:
        raise NotImplementedError


# limamau: redefine init outside abstract
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
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        clouds: "AbstractCloudModel",
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


# limamau: redefine init outside abstract
class AbstractCloudModel:
    def __init__(self):
        # cloud core fraction [-]
        self.cc_frac = 0.0
        # cloud core mass flux [m s-1]
        self.cc_mf = 0.0
        # cloud core moisture flux [kg kg-1 m s-1]
        self.cc_qf = 0.0

    @abstractmethod
    def run(self, mixed_layer: "AbstractMixedLayerModel") -> None:
        raise NotImplementedError
