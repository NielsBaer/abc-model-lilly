import jax
import jax.numpy as jnp
from jax import Array

import abcconfigs.class_model as cm
import abcmodel


def run_model(theta0: float) -> Array:
    dt = 60.0
    runtime = 12 * 3600.0

    rad_init_conds = abcmodel.rad.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )

    land_init_conds = abcmodel.land.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs
    )
    land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)

    surface_layer_init_conds = (
        abcmodel.atmos.surface_layer.ObukhovSurfaceLayerInitConds(
            **cm.obukhov_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()

    mixed_layer_init_conds = abcmodel.atmos.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs
    )
    mixed_layer_init_conds = mixed_layer_init_conds.replace(
        theta=jnp.array(theta0)  # <--- perturb initial condition
    )

    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )

    cloud_init_conds = abcmodel.atmos.clouds.CumulusInitConds()
    cloud_model = abcmodel.atmos.clouds.CumulusModel()

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
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(rad_init_conds, land_init_conds, atmos_init_conds)

    _, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # return final boundary layer height as scalar
    return trajectory.atmos.mixed.h_abl[-1]


def main():
    # forward mode
    grad_fn = jax.jacfwd(run_model)
    theta0 = 290.0
    dhf_dtheta0 = grad_fn(theta0)
    assert dhf_dtheta0 > 0.0
    print("∂h_final / ∂theta_0 =", dhf_dtheta0)

    # reverse mode
    grad_fn = jax.jacrev(run_model)
    theta0 = 290.0
    try:
        dhf_dtheta0 = grad_fn(theta0)
    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
