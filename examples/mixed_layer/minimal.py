import matplotlib.pyplot as plt

import configs.class_model as cm
from abcmodel import ABCModel
from abcmodel.clouds import StandardCumulusModel
from abcmodel.land_surface import JarvisStewartModel
from abcmodel.mixed_layer import MinimalMixedLayerModel
from abcmodel.radiation import StandardRadiationModel
from abcmodel.surface_layer import StandardSurfaceLayerModel


def main():
    # 0. running configurations:
    dt = 60.0  # time step [s]
    runtime = 96 * 3600.0  # total run time [s]

    # theta is currently assigned in two classes
    theta = 288.0

    # define mixed layer model
    mixed_layer_model = MinimalMixedLayerModel(
        # large scale parameters:
        # initial ABL height [m]
        abl_height=200.0,
        # surface pressure [Pa]
        surf_pressure=101300.0,
        # temperature parameters:
        # initial mixed-layer potential temperature [K]
        theta=theta,
        # initial temperature jump at h [K]
        dtheta=1.0,
        # surface kinematic heat flux [K m s-1]
        wtheta=0.1,
        # moisture parameters:
        # initial mixed-layer specific humidity [kg kg-1]
        q=0.008,
        # initial specific humidity jump at h [kg kg-1]
        dq=-0.001,
        # surface kinematic moisture flux [kg kg-1 m s-1]
        wq=1e-4,
        # CO2 parameters:
        # initial mixed-layer CO2 [ppm]
        co2=422.0,
        # initial CO2 jump at h [ppm]
        dCO2=-44.0,
        # surface kinematic CO2 flux [ppm m s-1]
        wCO2=0.0,
        # wind parameters:
        # initial mixed-layer u-wind speed [m s-1]
        u=6.0,
        # initial mixed-layer v-wind speed [m s-1]
        v=-4.0,
        # transition layer thickness [m]
        dz_h=150.0,
    )

    # 2. define surface layer model
    surface_layer_model = StandardSurfaceLayerModel(
        cm.params.surface_layer,
        cm.init_conds.surface_layer,
    )

    # 3. define radiation model
    radiation_model = StandardRadiationModel(
        cm.params.radiation,
        cm.init_conds.radiation,
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
    cloud_model = StandardCumulusModel(
        cm.params.clouds,
        cm.init_conds.clouds,
    )

    # init and run the model
    abc = ABCModel(
        dt=dt,
        runtime=runtime,
        mixed_layer=mixed_layer_model,
        surface_layer=surface_layer_model,
        radiation=radiation_model,
        land_surface=land_surface_model,
        clouds=cloud_model,
    )
    abc.run()

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(abc.out.t, abc.out.h)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(abc.out.t, abc.out.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(abc.out.t, abc.out.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(abc.out.t, abc.clouds.diagnostics.get("cc_frac"))
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(abc.out.t, abc.out.co2)
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer CO2 [ppm]")

    plt.subplot(236)
    plt.plot(abc.out.t, abc.out.u)
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer u-wind speed [m s-1]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
