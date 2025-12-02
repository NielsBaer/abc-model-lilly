import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # radiation
    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    land_surface_init_conds = abcmodel.land.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs,
    )
    land_surface_model = abcmodel.land.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = (
        abcmodel.atmosphere.surface_layer.StandardSurfaceLayerInitConds(
            **cm.standard_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmosphere.surface_layer.StandardSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.atmosphere.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.atmosphere.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.atmosphere.clouds.NoCloudInitConds()
    cloud_model = abcmodel.atmosphere.clouds.NoCloudModel()

    # define coupler and coupled state
    # define atmosphere model
    atmosphere_model = abcmodel.atmosphere.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        radiation=radiation_model,
        land=land_surface_model,
        atmosphere=atmosphere_model,
    )
    state = abcoupler.init_state(
        radiation_init_conds,
        land_surface_init_conds,
        surface_layer_init_conds,
        mixed_layer_init_conds,
        cloud_init_conds,
    )

    # run run run
    time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, trajectory.h_abl)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, trajectory.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, trajectory.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, trajectory.cc_frac)
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, trajectory.wCO2)
    plt.xlabel("time [h]")
    plt.ylabel("surface kinematic CO2 flux [mgC m-2 s-1]")

    plt.subplot(236)
    plt.plot(time, trajectory.le_veg)
    plt.xlabel("time [h]")
    plt.ylabel("transpiration [W m-2]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
