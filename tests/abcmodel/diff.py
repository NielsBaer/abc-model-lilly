import jax
import jax.numpy as jnp

import abcconfigs.class_model as cm
import abcmodel


def run_model(theta0: float) -> float:
    # copy-paste setup from your main(), but modify initial conditions
    dt = 60.0
    runtime = 12 * 3600.0

    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )

    land_surface_init_conds = abcmodel.land_surface.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs
    )
    land_surface_model = abcmodel.land_surface.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs
    )

    surface_layer_init_conds = abcmodel.surface_layer.StandardSurfaceLayerInitConds(
        **cm.standard_surface_layer.init_conds_kwargs
    )
    surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel()

    mixed_layer_init_conds = abcmodel.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs
    )
    mixed_layer_init_conds.theta = theta0  # <--- perturb initial condition

    mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )

    cloud_init_conds = abcmodel.clouds.StandardCumulusInitConds()
    cloud_model = abcmodel.clouds.StandardCumulusModel()

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

    _, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # return final boundary layer height as scalar
    return trajectory.abl_height[-1]


def main():
    # forward mode
    grad_fn = jax.jacfwd(run_model)
    theta0 = 290.0
    dh_dtheta0 = grad_fn(theta0)
    assert jnp.isfinite(dh_dtheta0)
    print("∂h_final / ∂θ_0 =", dh_dtheta0)

    # reverse mode
    grad_fn = jax.jacrev(run_model)
    theta0 = 290.0
    dh_dtheta0 = grad_fn(theta0)
    assert jnp.isfinite(dh_dtheta0)
    print("∂h_final / ∂θ_0 =", dh_dtheta0)


if __name__ == "__main__":
    main()
