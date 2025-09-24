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
    land_surface_init_conds = abcmodel.land_surface.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs,
    )
    land_surface_model = abcmodel.land_surface.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = abcmodel.surface_layer.StandardSurfaceLayerInitConds(
        **cm.standard_surface_layer.init_conds_kwargs
    )
    surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.clouds.StandardCumulusInitConds()
    cloud_model = abcmodel.clouds.StandardCumulusModel()

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        mixed_layer=mixed_layer_model,
        surface_layer=surface_layer_model,
        radiation=radiation_model,
        land_surface=land_surface_model,
        clouds=cloud_model,
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
    plt.plot(time, trajectory.abl_height)
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
    plt.plot(time, trajectory.gf)
    plt.xlabel("time [h]")
    plt.ylabel("ground heat flux [W m-2]")

    plt.subplot(236)
    plt.plot(time, trajectory.le_veg)
    plt.xlabel("time [h]")
    plt.ylabel("latent heat flux from vegetation [W m-2]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
