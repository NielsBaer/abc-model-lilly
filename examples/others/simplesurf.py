import jax.numpy as jnp
import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # rad
    rad_init_conds = abcmodel.rad.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    land_init_conds = abcmodel.land.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs,
    )
    land_model = abcmodel.land.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = abcmodel.atmos.surface_layer.SimpleSurfaceLayerInitConds(
        ustar=jnp.array(0.3)
    )
    surface_layer_model = abcmodel.atmos.surface_layer.SimpleSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.atmos.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.atmos.clouds.CumulusInitConds()
    cloud_model = abcmodel.atmos.clouds.CumulusModel()

    # define atmos model
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_init_conds = abcmodel.atmos.DayOnlyAtmosphereState(
        surface=surface_layer_init_conds,
        mixed=mixed_layer_init_conds,
        clouds=cloud_init_conds,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(rad_init_conds, land_init_conds, atmos_init_conds)

    # run run run
    time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, trajectory.atmos.mixed.h_abl)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, trajectory.atmos.mixed.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, trajectory.atmos.mixed.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, trajectory.atmos.clouds.cc_frac)
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, trajectory.land.wCO2)
    plt.xlabel("time [h]")
    plt.ylabel("surface kinematic CO2 flux [mgC m-2 s-1]")

    plt.subplot(236)
    plt.plot(time, trajectory.land.le_veg)
    plt.xlabel("time [h]")
    plt.ylabel("transpiration [W m-2]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
