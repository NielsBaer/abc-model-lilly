import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import PhysicalConstants


def main():
    # time step [s]
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    # total run time [s]
    runtime = 12 * 3600.0
    # start time of the day [h]
    tstart = 6.8

    # define rad model
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    # land surface
    land_model = abcmodel.land.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )
    land_state = land_model.init_state(
        **cm.jarvis_stewart.state_kwargs,
    )

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    # mixed layer
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    mixed_layer_state = mixed_layer_model.init_state(
        **cm.bulk_mixed_layer.state_kwargs,
    )

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
        surface=surface_layer_state,
        mixed=mixed_layer_state,
        clouds=cloud_state,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(
        rad_state,
        land_state,
        atmos_state,
    )

    # run model with diagnostics enabled
    time, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )
    const = PhysicalConstants()

    # plot diagnostic evolution
    _, axes = plt.subplots(1, 3, figsize=(12, 4))

    # row 1: water budget
    axes[0].plot(time, trajectory.total_water_mass)
    axes[0].set_xlabel("time [h]")
    axes[0].set_ylabel("Total water mass [kg m-2]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        time,
        trajectory.atmos.mixed.q * trajectory.atmos.mixed.h_abl * const.rho,
        label="Vapor",
    )
    axes[1].plot(time, trajectory.land.wg * const.rhow * 0.1, label="Soil layer 1")
    axes[1].plot(time, trajectory.land.wl * const.rhow, label="Canopy")
    axes[1].set_xlabel("time [h]")
    axes[1].set_ylabel("Water mass [kg m-2]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, trajectory.land.le_veg, label="LE vegetation")
    axes[2].plot(time, trajectory.land.le_soil, label="LE soil")
    axes[2].plot(time, trajectory.land.le_liq, label="LE liq")
    axes[2].plot(time, trajectory.land.le, label="LE total")
    axes[2].set_xlabel("time [h]")
    axes[2].set_ylabel("Latent heat flux [W m-2]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
