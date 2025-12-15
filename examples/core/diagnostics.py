import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import PhysicalConstants


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
    surface_layer_init_conds = (
        abcmodel.atmos.surface_layer.ObukhovSurfaceLayerInitConds(
            **cm.obukhov_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()

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

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    atmos_state = abcmodel.atmos.DayOnlyAtmosphereState(
        surface=surface_layer_init_conds,
        mixed=mixed_layer_init_conds,
        clouds=cloud_init_conds,
    )
    state = abcoupler.init_state(
        rad_init_conds,
        land_init_conds,
        atmos_state,
    )

    # run model with diagnostics enabled
    time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)
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
