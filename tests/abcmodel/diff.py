import jax
import jax.numpy as jnp

import abcconfigs.class_model as cm
import abcmodel


def run_model(theta0: float) -> float:
    dt = 60.0
    runtime = 12 * 3600.0

    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )

    land_surface_init_conds = abcmodel.land.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs
    )
    land_surface_model = abcmodel.land.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs
    )

    surface_layer_init_conds = (
        abcmodel.atmosphere.surface_layer.ObukhovSurfaceLayerInitConds(
            **cm.obukhov_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmosphere.surface_layer.ObukhovSurfaceLayerModel()

    mixed_layer_init_conds = abcmodel.atmosphere.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs
    )
    mixed_layer_init_conds = mixed_layer_init_conds.replace(
        theta=jnp.array(theta0)  # <--- perturb initial condition
    )

    mixed_layer_model = abcmodel.atmosphere.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )

    cloud_init_conds = abcmodel.atmosphere.clouds.CumulusInitConds()
    cloud_model = abcmodel.atmosphere.clouds.CumulusModel()

    atmosphere_model = abcmodel.atmosphere.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmosphere_init_conds = abcmodel.atmosphere.DayOnlyAtmosphereState(
        surface_layer=surface_layer_init_conds,
        mixed_layer=mixed_layer_init_conds,
        clouds=cloud_init_conds,
    )
    atmosphere_model = abcmodel.atmosphere.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    abcoupler = abcmodel.ABCoupler(
        radiation=radiation_model,
        land=land_surface_model,
        atmosphere=atmosphere_model,
    )
    state = abcoupler.init_state(
        radiation_init_conds, land_surface_init_conds, atmosphere_init_conds
    )

    _, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # return final boundary layer height as scalar
    return trajectory.atmosphere.mixed_layer.h_abl[-1]


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
