# limamau: this module contains the abstract classes for all main components of the ABC
# model. Another architecture possibility is to define each abstract class in their own
# files and extract kwargs from each run method - the coupler would manage integration...
# I'm leaving this idea for the furure in case we need/decide to go for it.
# limamau: in the future, it will be good to define parameter and initial conditions classes
# to simplifiy initializations

from abc import abstractmethod

from .utils import PhysicalConstants


class AbstractRadiationModel:
    # require by minimal:
    # net radiation [W m-2]
    net_rad: float
    # cloud top radiative divergence [W m-2]
    dFz: float
    # used to output:
    # time of the day [h UTC]
    tstart: float
    # incoming short wave radiation [W m-2]
    in_srad: float
    # outgoing short wave radiation [W m-2]
    out_srad: float
    # incoming long wave radiation [W m-2]
    in_lrad: float
    # outgoing long wave radiation [W m-2]
    out_lrad: float

    @abstractmethod
    def run(
        self,
        t: float,
        dt: float,
        const: PhysicalConstants,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_f1(self) -> float:
        raise NotImplementedError


class AbstractLandSurfaceModel:
    # required by minimal:
    # surface albedo [-]
    alpha: float
    # initial surface temperature [K]
    surf_temp: float
    # used to output:
    # resistance transpiration [s m-1]
    rs: float
    # sensible heat flux [W m-2]
    hf: float
    # evapotranspiration [W m-2]
    le: float
    # open water evaporation [W m-2]
    le_liq: float
    # transpiration [W m-2]
    le_veg: float
    # soil evaporation [W m-2]
    le_soil: float
    # potential evaporation [W m-2]
    le_pot: float
    # reference evaporation at rs = rsmin / LAI [W m-2]
    le_ref: float
    # ground heat flux [W m-2]
    gf: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError


class AbstractSurfaceLayerModel:
    # required by minimal:
    # surface friction velocity [m s-1]
    ustar: float
    # used to output:
    # surface momentum flux u [m2 s-2]
    uw: float
    # surface momentum flux v [m2 s-2]
    vw: float
    # 2m temperature [K]
    temp_2m: float
    # 2m specific humidity [kg kg-1]
    q2m: float
    # 2m u-wind [m s-1]
    u2m: float
    # 2m v-wind [m s-1]
    v2m: float
    # 2m vapor pressure [Pa]
    e2m: float
    # 2m saturated vapor pressure [Pa]
    esat2m: float
    # surface potential temperature [K]
    thetasurf: float
    # surface virtual potential temperature [K]
    thetavsurf: float
    # surface specific humidity [g kg-1]
    qsurf: float
    # drag coefficient for momentum [-]
    drag_m: float
    # drag coefficient for scalars [-]
    drag_s: float
    # Obukhov length [m]
    obukhov_length: float
    # bulk Richardson number [-]
    rib_number: float
    # aerodynamic resistance [s m-1]
    ra: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_ra(self, u: float, v: float, wstar: float) -> None:
        raise NotImplementedError


class AbstractMixedLayerModel:
    # required by minimal:
    # initial mixed-layer specific humidity [kg kg-1]
    q: float
    # initial specific humidity jump at h [kg kg-1]
    dq: float
    # mixed-layer saturated vapor pressure [Pa]
    esat: float
    # mixed-layer vapor pressure [Pa]
    e: float
    # mixed-layer saturated specific humidity [kg kg-1]
    qsat: float
    # surface saturated specific humidity [g kg-1]
    qsatsurf: float
    # slope saturated specific humidity curve [g kg-1 K-1]
    dqsatdT: float
    # initial mixed-layer potential temperature [K]
    theta: float
    # surface kinematic heat flux [K m s-1]
    wtheta: float
    # surface kinematic moisture flux [kg kg-1 m s-1]
    wq: float
    # entrainment moisture flux [kg kg-1 m s-1]
    wqe: float
    # initial temperature jump at h [K]
    dtheta: float
    # initial ABL height [m]
    abl_height: float
    # surface pressure [Pa]
    surf_pressure: float
    # initial mixed-layer u-wind speed [m s-1]
    u: float
    # initial mixed-layer v-wind speed [m s-1]
    v: float
    # convective velocity scale [m s-1]
    wstar: float
    # initial mixed-layer potential temperature [K]
    thetav: float
    # surface kinematic virtual heat flux [K m s-1]
    wthetav: float
    # mixed-layer top specific humidity variance [kg2 kg-2]
    q2_h: float
    # transition layer thickness [-]
    dz_h: float
    # mixed-layer top pressure [pa]
    top_p: float
    # mixed-layer top absolute temperature [K]
    top_T: float
    # initial mixed-layer CO2 [ppm]
    co2: float
    # mixed-layer top CO2 variance [ppm2]
    top_CO22: float
    # entrainment CO2 flux [ppm m s-1]
    wCO2e: float
    # CO2 mass flux [ppm m s-1]
    wCO2M: float
    # initial CO2 jump at h [ppm]
    dCO2: float
    # surface kinematic CO2 flux [ppm m s-1]
    wCO2: float
    # surface assimulation CO2 flux [ppm m s-1]
    wCO2A: float
    # surface respiration CO2 flux [ppm m s-1]
    wCO2R: float
    # used to output:
    # entrainment kinematic heat flux [K m s-1]
    wthetae: float
    # initial virtual temperature jump at h [K]
    dthetav: float
    # entrainment kinematic virtual heat flux [K m s-1]
    wthetave: float
    # initial u-wind jump at h [m s-1]
    du: float
    # initial v-wind jump at h [m s-1]
    dv: float
    # lifting condensation level [m]
    lcl: float
    # mixed-layer top relavtive humidity [-]
    top_rh: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        clouds: "AbstractCloudModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, t: float, const: PhysicalConstants) -> None:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError


class AbstractCloudModel:
    """
    Abstract base class for atmospheric cloud models in mixed-layer systems.

    This class defines the interface that all cloud models must implement for
    integration with mixed-layer boundary layer models.
    """

    # cloud core fraction [-]
    cc_frac: float
    # cloud core mass flux [m s-1]
    cc_mf: float
    # cloud core moisture flux [kg kg-1 m s-1]
    cc_qf: float

    @abstractmethod
    def run(self, mixed_layer: "AbstractMixedLayerModel") -> None:
        raise NotImplementedError
