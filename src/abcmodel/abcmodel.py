import numpy as np

from abcmodel.land_surface import MinimalLandSurfaceModel

from .clouds import AbstractCloudModel, NoCloudModel
from .land_surface import AbstractLandSurfaceModel
from .mixed_layer import AbstractMixedLayerModel, NoMixedLayerModel
from .radiation import AbstractRadiationModel, MinimalRadiationModel
from .surface_layer import AbstractSurfaceLayerModel, MinimalSurfaceLayerModel
from .utils import PhysicalConstants


class ABCModel:
    def __init__(
        self,
        dt: float,
        runtime: float,
        mixed_layer: AbstractMixedLayerModel,
        surface_layer: AbstractSurfaceLayerModel,
        radiation: AbstractRadiationModel,
        land_surface: AbstractLandSurfaceModel,
        clouds: AbstractCloudModel,
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
        self.radiation = radiation

        # 4. land surface is initialized like before
        self.land_surface = land_surface

        # 5. clouds
        # limamau: if clouds are part of the mixed layer, this could be avoided
        if not isinstance(clouds, NoCloudModel):
            if isinstance(mixed_layer, NoMixedLayerModel):
                raise ValueError("Cloud models requires a mixed layer model.")
            self.clouds = clouds

        # initialize output
        self.out = ABCOutput(self.tsteps)

    def run(self):
        self.warmup()
        for self.t in range(self.tsteps):
            self.timestep()

    def warmup(self):
        self.mixed_layer.statistics(self.t, self.const)

        # calculate initial diagnostic variables
        self.radiation.run(
            self.t,
            self.dt,
            self.const,
            self.land_surface,
            self.mixed_layer,
        )

        for _ in range(10):
            assert isinstance(self.mixed_layer.thetav, float)
            self.surface_layer.run(self.const, self.land_surface, self.mixed_layer)

        self.land_surface.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.mixed_layer,
        )

        assert isinstance(self.surface_layer.uw, float)
        assert isinstance(self.surface_layer.vw, float)
        if not isinstance(self.clouds, NoCloudModel):
            self.mixed_layer.run(
                self.const,
                self.radiation,
                self.surface_layer,
                self.clouds,
            )
            self.clouds.run(self.mixed_layer)

        self.mixed_layer.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.clouds,
        )

    def timestep(self):
        self.mixed_layer.statistics(self.t, self.const)

        # run radiation model
        self.radiation.run(
            self.t,
            self.dt,
            self.const,
            self.land_surface,
            self.mixed_layer,
        )

        # run surface layer model
        assert isinstance(self.mixed_layer.thetav, float)
        self.surface_layer.run(self.const, self.land_surface, self.mixed_layer)

        # run land surface model
        self.land_surface.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.mixed_layer,
        )

        # run cumulus parameterization
        self.clouds.run(self.mixed_layer)

        # run mixed-layer model
        self.mixed_layer.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.clouds,
        )

        # store output before time integration
        self.store()

        # time integrate land surface model
        self.land_surface.integrate(self.dt)

        # time integrate mixed-layer model
        self.mixed_layer.integrate(self.dt)

    # store model output
    def store(self):
        t = self.t

        if not isinstance(self.radiation, MinimalRadiationModel):
            self.out.t[t] = t * self.dt / 3600.0 + self.radiation.tstart
            self.out.in_srad[t] = self.radiation.in_srad
            self.out.out_srad[t] = self.radiation.out_srad
            self.out.in_lrad[t] = self.radiation.in_lrad
            self.out.out_lrad[t] = self.radiation.out_lrad
            self.out.net_rad[t] = self.radiation.net_rad

        if not isinstance(self.land_surface, MinimalLandSurfaceModel):
            self.out.rs[t] = self.land_surface.rs
            self.out.hf[t] = self.land_surface.hf
            self.out.le[t] = self.land_surface.le
            self.out.le_liq[t] = self.land_surface.le_liq
            self.out.le_veg[t] = self.land_surface.le_veg
            self.out.le_soil[t] = self.land_surface.le_soil
            self.out.le_pot[t] = self.land_surface.le_pot
            self.out.le_ref[t] = self.land_surface.le_ref
            self.out.gf[t] = self.land_surface.gf

        if not isinstance(self.surface_layer, MinimalSurfaceLayerModel):
            self.out.uw[t] = self.surface_layer.uw
            self.out.vw[t] = self.surface_layer.vw
            self.out.temp_2m[t] = self.surface_layer.temp_2m
            self.out.q2m[t] = self.surface_layer.q2m
            self.out.u2m[t] = self.surface_layer.u2m
            self.out.v2m[t] = self.surface_layer.v2m
            self.out.e2m[t] = self.surface_layer.e2m
            self.out.esat2m[t] = self.surface_layer.esat2m
            self.out.thetasurf[t] = self.surface_layer.thetasurf
            self.out.thetavsurf[t] = self.surface_layer.thetavsurf
            self.out.qsurf[t] = self.surface_layer.qsurf
            self.out.ustar[t] = self.surface_layer.ustar
            self.out.drag_m[t] = self.surface_layer.drag_m
            self.out.drag_s[t] = self.surface_layer.drag_s
            self.out.obukhov_length[t] = self.surface_layer.obukhov_length
            self.out.rib_number[t] = self.surface_layer.rib_number
            self.out.ra[t] = self.surface_layer.ra

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
        self.out.wqM[t] = self.clouds.cc_qf

        self.out.qsat[t] = self.mixed_layer.qsat
        self.out.e[t] = self.mixed_layer.e
        self.out.esat[t] = self.mixed_layer.esat

        fac = (self.const.rho * self.const.mco2) / self.const.mair
        self.out.co2[t] = self.mixed_layer.co2
        self.out.dCO2[t] = self.mixed_layer.dCO2
        self.out.wCO2[t] = self.mixed_layer.wCO2 * fac
        # limamau: this was failing when no mixed layer model is defined
        # and I don't feel like adding an if else clause here... let's see
        # self.out.wCO2e[t] = self.mixed_layer.wCO2e * fac
        self.out.wCO2R[t] = self.mixed_layer.wCO2R * fac
        self.out.wCO2A[t] = self.mixed_layer.wCO2A * fac

        self.out.u[t] = self.mixed_layer.u
        self.out.du[t] = self.mixed_layer.du

        self.out.v[t] = self.mixed_layer.v
        self.out.dv[t] = self.mixed_layer.dv

        self.out.zlcl[t] = self.mixed_layer.lcl
        self.out.top_rh[t] = self.mixed_layer.top_rh

        self.out.cc_frac[t] = self.clouds.cc_frac
        self.out.cc_mf[t] = self.clouds.cc_mf
        self.out.dz_h[t] = self.mixed_layer.dz_h


class ABCOutput:
    def __init__(self, tsteps):
        # time [s]
        self.t = np.zeros(tsteps)

        # mixed-layer variables
        # ABL height [m]
        self.h = np.zeros(tsteps)

        # initial mixed-layer potential temperature [K]
        self.theta = np.zeros(tsteps)
        # initial mixed-layer virtual potential temperature [K]
        self.thetav = np.zeros(tsteps)
        # initial potential temperature jump at h [K]
        self.dtheta = np.zeros(tsteps)
        # initial virtual potential temperature jump at h [K]
        self.dthetav = np.zeros(tsteps)
        # surface kinematic heat flux [K m s-1]s
        self.wtheta = np.zeros(tsteps)
        # surface kinematic virtual heat flux [K m s-1]s
        self.wthetav = np.zeros(tsteps)
        # entrainment kinematic heat flux [K m s-1]s
        self.wthetae = np.zeros(tsteps)
        # entrainment kinematic virtual heat flux [K m s-1]
        self.wthetave = np.zeros(tsteps)

        # mixed-layer specific humidity [kg kg-1]
        self.q = np.zeros(tsteps)
        # initial specific humidity jump at h [kg kg-1]
        self.dq = np.zeros(tsteps)
        # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wq = np.zeros(tsteps)
        # entrainment kinematic moisture flux [kg kg-1 m s-1]
        self.wqe = np.zeros(tsteps)
        # cumulus mass-flux kinematic moisture flux [kg kg-1 m s-1]
        self.wqM = np.zeros(tsteps)

        # mixed-layer saturated specific humidity [kg kg-1]
        self.qsat = np.zeros(tsteps)
        # mixed-layer vapor pressure [Pa]
        self.e = np.zeros(tsteps)
        # mixed-layer saturated vapor pressure [Pa]
        self.esat = np.zeros(tsteps)

        # mixed-layer CO2 [ppm]
        self.co2 = np.zeros(tsteps)
        # initial CO2 jump at h [ppm]
        self.dCO2 = np.zeros(tsteps)
        # surface total CO2 flux [mgC m-2 s-1]
        self.wCO2 = np.zeros(tsteps)
        # surface assimilation CO2 flux [mgC m-2 s-1]
        self.wCO2A = np.zeros(tsteps)
        # surface respiration CO2 flux [mgC m-2 s-1]
        self.wCO2R = np.zeros(tsteps)
        # entrainment CO2 flux [mgC m-2 s-1]
        self.wCO2e = np.zeros(tsteps)
        # CO2 mass flux [mgC m-2 s-1]
        self.wCO2M = np.zeros(tsteps)

        # initial mixed-layer u-wind speed [m s-1]
        self.u = np.zeros(tsteps)
        # initial u-wind jump at h [m s-1]
        self.du = np.zeros(tsteps)
        # surface momentum flux u [m2 s-2]
        self.uw = np.zeros(tsteps)
        # initial mixed-layer u-wind speed [m s-1]
        self.v = np.zeros(tsteps)
        # initial u-wind jump at h [m s-1]
        self.dv = np.zeros(tsteps)
        # surface momentum flux v [m2 s-2]
        self.vw = np.zeros(tsteps)

        # diagnostic meteorological variables
        # 2m temperature [K]
        self.temp_2m = np.zeros(tsteps)
        # 2m specific humidity [kg kg-1]
        self.q2m = np.zeros(tsteps)
        # 2m u-wind [m s-1]
        self.u2m = np.zeros(tsteps)
        # 2m v-wind [m s-1]
        self.v2m = np.zeros(tsteps)
        # 2m vapor pressure [Pa]
        self.e2m = np.zeros(tsteps)
        # 2m saturated vapor pressure [Pa]
        self.esat2m = np.zeros(tsteps)

        # surface-layer variables
        # surface potential temperature [K]
        self.thetasurf = np.zeros(tsteps)
        # surface virtual potential temperature [K]
        self.thetavsurf = np.zeros(tsteps)
        # surface specific humidity [kg kg-1]
        self.qsurf = np.zeros(tsteps)
        # surface friction velocity [m s-1]
        self.ustar = np.zeros(tsteps)
        # roughness length for momentum [m]
        self.z0m = np.zeros(tsteps)
        # roughness length for scalars [m]
        self.z0h = np.zeros(tsteps)
        # drag coefficient for momentum []
        self.drag_m = np.zeros(tsteps)
        # drag coefficient for scalars []
        self.drag_s = np.zeros(tsteps)
        # Obukhov length [m]
        self.obukhov_length = np.zeros(tsteps)
        # bulk Richardson number [-]
        self.rib_number = np.zeros(tsteps)

        # radiation variables
        # incoming short wave radiation [W m-2]
        self.in_srad = np.zeros(tsteps)
        # outgoing short wave radiation [W m-2]
        self.out_srad = np.zeros(tsteps)
        # incoming long wave radiation [W m-2]
        self.in_lrad = np.zeros(tsteps)
        # outgoing long wave radiation [W m-2]
        self.out_lrad = np.zeros(tsteps)
        # net radiation [W m-2]
        self.net_rad = np.zeros(tsteps)

        # land surface variables
        # aerodynamic resistance [s m-1]
        self.ra = np.zeros(tsteps)
        # surface resistance [s m-1]
        self.rs = np.zeros(tsteps)
        # sensible heat flux [W m-2]
        self.hf = np.zeros(tsteps)
        # evapotranspiration [W m-2]
        self.le = np.zeros(tsteps)
        # open water evaporation [W m-2]
        self.le_liq = np.zeros(tsteps)
        # transpiration [W m-2]
        self.le_veg = np.zeros(tsteps)
        # soil evaporation [W m-2]
        self.le_soil = np.zeros(tsteps)
        # potential evaporation [W m-2]
        self.le_pot = np.zeros(tsteps)
        # reference evaporation at rs = rsmin / LAI [W m-2]
        self.le_ref = np.zeros(tsteps)
        # ground heat flux [W m-2]
        self.gf = np.zeros(tsteps)

        # mixed-layer top variables
        # lifting condensation level [m]
        self.zlcl = np.zeros(tsteps)
        # mixed-layer top relative humidity [-]
        self.top_rh = np.zeros(tsteps)

        # cumulus variables
        # cloud core fraction [-]
        self.cc_frac = np.zeros(tsteps)
        # cloud core mass flux [m s-1]
        self.cc_mf = np.zeros(tsteps)
        # transition layer thickness [m]
        self.dz_h = np.zeros(tsteps)
