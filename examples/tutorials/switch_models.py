# Import packages
import matplotlib.pyplot as plt
import abcconfigs.class_model as cm
import abcmodel


# define main function
def main():

    ##########################################################
    # first define time steps necessary for the model run
    ##########################################################

    # inner step for model calculations
    inner_dt = 60.0
    # outer step for model diagnostics
    outter_dt = 60.0 * 30
    # total run time [s]
    runtime = 12 * 3600.0
    # start time of the day [h]
    tstart = 6.8

    ##########################################################
    # define all models with respective initial states and 
    # arguments from abcconfigs
    # in this tutorial we run two simulations with different
    # CO2 concentrations to compare them
    ##########################################################

    # rad with clouds
    rad_model = abcmodel.rad.CloudyRadiationModel(
        **cm.cloudy_radiation.model_kwargs,
    )
    rad_state = rad_model.init_state(**cm.cloudy_radiation.state_kwargs)

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer with default
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    bml_state = cm.bulk_mixed_layer.state_kwargs
    mixed_layer_state_default = mixed_layer_model.init_state(
            **bml_state,
    )

    # generate an additional mixed layer initial state with double CO2
    bml_state.update({"co2": 844.0})
    mixed_layer_state_2xCO2 = mixed_layer_model.init_state(
            **bml_state,
    )

    # clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # in this tutorial we can compare two land models using this switch
    use_ags = True


    if use_ags:

        # you can use the ags model, that includes
        # CO2-induced stomatal closure
        land_model = abcmodel.land.AgsModel(
            **cm.ags.model_kwargs,
        )
        land_state = land_model.init_state(
            **cm.ags.state_kwargs,
        )

    else:
        # otherwise use the jarvis stewart model for simpler representation
        land_model = abcmodel.land.JarvisStewartModel(
            **cm.jarvis_stewart.model_kwargs,
        )
        land_state = land_model.init_state(
            **cm.jarvis_stewart.state_kwargs,
        )


    ##########################################################
    # define atmos model with all its components
    ##########################################################

    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state_default = atmos_model.init_state(
        surface=surface_layer_state,
        mixed=mixed_layer_state_default,
        clouds=cloud_state,
    )

    atmos_state_2xCO2 = atmos_model.init_state(
        surface=surface_layer_state,
        mixed=mixed_layer_state_2xCO2,
        clouds=cloud_state,
    )

    ##########################################################
    # define atmos model with all its components
    ##########################################################

    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state_default = abcoupler.init_state(
        rad_state,
        land_state,
        atmos_state_default,
    )

    state_2xCO2 = abcoupler.init_state(
        rad_state,
        land_state,
        atmos_state_2xCO2,
    )

    ##########################################################
    # run the model for both CO2 concentrations
    ##########################################################

    time, trajectory_default = abcmodel.integrate(
        state_default, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

    time, trajectory_2xCO2 = abcmodel.integrate(
        state_2xCO2, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

    ##########################################################
    # Plot the output of both runs
    ##########################################################

    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, trajectory_default.atmos.mixed.h_abl)
    plt.plot(time, trajectory_2xCO2.atmos.mixed.h_abl)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, trajectory_default.atmos.mixed.theta)
    plt.plot(time, trajectory_2xCO2.atmos.mixed.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, trajectory_default.atmos.mixed.q * 1000.0)
    plt.plot(time, trajectory_2xCO2.atmos.mixed.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, trajectory_default.atmos.clouds.cc_frac)
    plt.plot(time, trajectory_2xCO2.atmos.clouds.cc_frac)
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, trajectory_default.land.le)
    plt.plot(time, trajectory_2xCO2.land.le)
    plt.xlabel("time [h]")
    plt.ylabel("LE [W m-2]")

    plt.subplot(236)
    plt.plot(time, trajectory_default.atmos.co2)
    plt.plot(time, trajectory_2xCO2.atmos.co2)
    plt.xlabel("time [h]")
    plt.ylabel("CO2 [ppmv]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
