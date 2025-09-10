import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
from abcmodel import ABCoupler
from abcmodel.clouds import StandardCumulusModel
from abcmodel.land_surface import JarvisStewartModel
from abcmodel.mixed_layer import (
    MinimalMixedLayerInitConds,
    MinimalMixedLayerModel,
    MinimalMixedLayerParams,
)
from abcmodel.radiation import StandardRadiationModel
from abcmodel.surface_layer import StandardSurfaceLayerModel


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # define mixed layer model
    mixed_layer_params = MinimalMixedLayerParams()
    mixed_layer_init_conds = MinimalMixedLayerInitConds(
        # initial ABL height [m]
        abl_height=200.0,
        # surface pressure [Pa]
        surf_pressure=101300.0,
        # initial mixed-layer potential temperature [K]
        theta=288.0,
        # initial temperature jump at h [K]
        dtheta=1.0,
        # surface kinematic heat flux [K m s-1]
        wtheta=0.1,
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
        # initial mixed-layer u-wind speed [m s-1]
        u=6.0,
        # initial mixed-layer v-wind speed [m s-1]
        v=-4.0,
        # transition layer thickness [m]
        dz_h=150.0,
    )
    mixed_layer_model = MinimalMixedLayerModel(
        mixed_layer_params,
        mixed_layer_init_conds,
    )

    # define surface layer model
    surface_layer_model = StandardSurfaceLayerModel(
        cm.standard_surface_layer.params,
        cm.standard_surface_layer.init_conds,
    )

    # define radiation model
    radiation_model = StandardRadiationModel(
        cm.standard_radiation.params,
        cm.standard_radiation.init_conds,
    )

    # define land surface model
    land_surface_model = JarvisStewartModel(
        cm.jarvis_stewart.params,
        cm.jarvis_stewart.init_conds,
    )

    # clouds
    cloud_model = StandardCumulusModel(
        cm.standard_cumulus.params,
        cm.standard_cumulus.init_conds,
    )

    # init and run the model
    abc = ABCoupler(
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
    time = abc.get_t()
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, abc.mixed_layer.diagnostics.get("abl_height"))
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, abc.mixed_layer.diagnostics.get("theta"))
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, abc.mixed_layer.diagnostics.get("q") * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, abc.clouds.diagnostics.get("cc_frac"))
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, abc.mixed_layer.diagnostics.get("co2"))
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer CO2 [ppm]")

    plt.subplot(236)
    plt.plot(time, abc.mixed_layer.diagnostics.get("u"))
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer u-wind speed [m s-1]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
