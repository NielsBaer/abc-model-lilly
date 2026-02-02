import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel


def main():
    # time step [s]
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    # total run time [s]
    runtime = 12 * 3600.0
    # start time of the day [h]
    tstart = 6.8

    # rad
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # land surface
    land_model = abcmodel.land.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )
    land_state = land_model.init_state(**cm.jarvis_stewart.state_kwargs)

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.SimpleModel()
    surface_layer_init_conds = surface_layer_model.init_state(ustar=0.3)

    # mixed layer
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    mixed_layer_state = mixed_layer_model.init_state(**cm.bulk_mixed_layer.state_kwargs)

    # clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # define atmos model
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_init_conds,
        mixed=mixed_layer_state,
        clouds=cloud_state,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    # run run run
    time, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

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
