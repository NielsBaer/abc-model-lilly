import matplotlib.pyplot as plt

from abcmodel.abcmodel import ABCModel
from abcmodel.clouds import StandardCumulusModel
from abcmodel.land_surface import JarvisStewartModel
from abcmodel.mixed_layer import StandardMixedLayerModel
from abcmodel.radiation import StandardRadiationModel
from abcmodel.surface_layer import StandardSurfaceLayerModel


def main():
    # 0. running configurations:
    dt = 60.0  # time step [s]
    runtime = 96 * 3600.0  # total run time [s]

    # theta is currently assigned in two classes
    theta = 288.0

    # define mixed layer model
    mixed_layer_model = StandardMixedLayerModel(
        # 1.1. switchs
        # mixed-layer model switch
        sw_ml=True,
        # shear growth mixed-layer switch
        sw_shearwe=True,
        # fix the free-troposphere switch
        sw_fixft=True,
        # 1.2. large scale parameters
        # initial ABL height [m]
        abl_height=200.0,
        # surface pressure [Pa]
        surf_pressure=101300.0,
        # horizontal large-scale divergence of wind [s-1]
        divU=0.0,
        # Coriolis parameter [m s-1]
        coriolis_param=1.0e-4,
        # 1.3 temperature parameters
        # initial mixed-layer potential temperature [K]
        theta=theta,
        # initial temperature jump at h [K]
        dtheta=1.0,
        # free atmosphere potential temperature lapse rate [K m-1]
        gammatheta=0.006,
        # advection of heat [K s-1]
        advtheta=0.0,
        # entrainment ratio for virtual heat [-]
        beta=0.2,
        # surface kinematic heat flux [K m s-1]
        wtheta=0.1,
        # 1.4 moisture parameters
        # initial mixed-layer specific humidity [kg kg-1]
        q=0.008,
        # initial specific humidity jump at h [kg kg-1]
        dq=-0.001,
        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        gammaq=0.0,
        # advection of moisture [kg kg-1 s-1]
        advq=0.0,
        # surface kinematic moisture flux [kg kg-1 m s-1]
        wq=1e-4,
        # 1.5. CO2 parameters
        # initial mixed-layer CO2 [ppm]
        co2=422.0,
        # initial CO2 jump at h [ppm]
        dCO2=-44.0,
        # free atmosphere CO2 lapse rate [ppm m-1]
        gammaCO2=0.0,
        # advection of CO2 [ppm s-1]
        advCO2=0.0,
        # surface kinematic CO2 flux [ppm m s-1]
        wCO2=0.0,
        # 1.6. wind parameters
        # prognostic wind switch
        sw_wind=True,
        # initial mixed-layer u-wind speed [m s-1]
        u=6.0,
        # initial u-wind jump at h [m s-1]
        du=4.0,
        # free atmosphere u-wind speed lapse rate [s-1]
        gammau=0.0,
        # advection of u-wind [m s-2]
        advu=0.0,
        # initial mixed-layer v-wind speed [m s-1]
        v=-4.0,
        # initial v-wind jump at h [m s-1]
        dv=4.0,
        # free atmosphere v-wind speed lapse rate [s-1]
        gammav=0.0,
        # advection of v-wind [m s-2]
        advv=0.0,
        # transition layer thickness [m]
        dz_h=150.0,
    )

    # 2. define surface layer model
    surface_layer_model = StandardSurfaceLayerModel(
        # surface friction velocity [m s-1]
        ustar=0.3,
        # roughness length for momentum [m]
        z0m=0.02,
        # roughness length for scalars [m]
        z0h=0.002,
        # initial mixed-layer potential temperature [K]
        theta=theta,
    )

    # 3. define radiation model
    radiation_model = StandardRadiationModel(
        # latitude [deg]
        lat=51.97,
        # longitude [deg]
        lon=-4.93,
        # day of the year [-]
        doy=268.0,
        # time of the day [h UTC]
        tstart=6.8,
        # cloud cover fraction [-]
        cc=0.0,
        # net radiation [W m-2]
        net_rad=400.0,
        # cloud top radiative divergence [W m-2]
        dFz=0.0,
    )

    # 4. define land surface model
    land_surface_model = JarvisStewartModel(
        # volumetric water content top soil layer [m3 m-3]
        wg=0.21,
        # volumetric water content deeper soil layer [m3 m-3]
        w2=0.21,
        # vegetation fraction [-]
        cveg=0.85,
        # temperature top soil layer [K]
        temp_soil=285.0,
        # temperature deeper soil layer [K]
        temp2=286.0,
        # Clapp and Hornberger retention curve parameter a
        a=0.219,
        # Clapp and Hornberger retention curve parameter b
        b=4.90,
        # Clapp and Hornberger retention curve parameter c
        p=4.0,
        # saturated soil conductivity for heat
        cgsat=3.56e-6,
        # saturated volumetric water content ECMWF config [-]
        wsat=0.472,
        # volumetric water content field capacity [-]
        wfc=0.323,
        # volumetric water content wilting point [-]
        wwilt=0.171,
        # C1 sat?
        c1sat=0.132,
        # C2 sat?
        c2sat=1.8,
        # leaf area index [-]
        lai=2.0,
        # correction factor transpiration for VPD [-]
        gD=0.0,
        # minimum resistance transpiration [s m-1]
        rsmin=110.0,
        # minimun resistance soil evaporation [s m-1]
        rssoilmin=50.0,
        # surface albedo [-]
        alpha=0.25,
        # initial surface temperature [K]
        surf_temp=290.0,
        # thickness of water layer on wet vegetation [m]
        wmax=0.0002,
        # equivalent water layer depth for wet vegetation [m]
        wl=0.0000,
        # thermal diffusivity skin layer [-]
        lam=5.9,
    )

    # 5. clouds
    cloud_model = StandardCumulusModel()

    # init and run the model
    r1 = ABCModel(
        dt=dt,
        runtime=runtime,
        mixed_layer=mixed_layer_model,
        surface_layer=surface_layer_model,
        radiation=radiation_model,
        land_surface=land_surface_model,
        clouds=cloud_model,
    )
    r1.run()

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(r1.out.t, r1.out.h)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(r1.out.t, r1.out.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(r1.out.t, r1.out.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(r1.out.t, r1.out.cc_frac)
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(r1.out.t, r1.out.wCO2)
    plt.xlabel("time [h]")
    plt.ylabel("surface kinematic CO2 flux [ppm m s-1]")

    plt.subplot(236)
    plt.plot(r1.out.t, r1.out.le_veg)
    plt.xlabel("time [h]")
    plt.ylabel("transpiration [W m-2]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
