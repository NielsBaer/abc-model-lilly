import jax.numpy as jnp

import abcconfigs.class_model as cm
import abcmodel


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # define rad model
    rad_init_conds = abcmodel.rad.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    land_init_conds = abcmodel.land.MinimalLandSurfaceInitConds(
        alpha=jnp.array(0.25),
        surf_temp=jnp.array(288.8),
        rs=jnp.array(1.0),
    )
    land_model = abcmodel.land.MinimalLandSurfaceModel()

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

    # run run run
    abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)


if __name__ == "__main__":
    main()
