import jax
import jax.numpy as jnp
from jax import Array

import abcconfigs.class_model as cm
import abcmodel


def run_model(theta0: float) -> Array:
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    runtime = 12 * 3600.0
    tstart = 6.8

    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)
    land_state = land_model.init_state(**cm.jarvis_stewart.state_kwargs)

    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel(
        **cm.bulk_mixed_layer.model_kwargs
    )
    mixed_layer_state = mixed_layer_model.init_state(**cm.bulk_mixed_layer.state_kwargs)
    mixed_layer_state = mixed_layer_state.replace(
        theta=jnp.array(theta0)  # <--- perturb initial condition
    )

    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

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

    abcoupler = abcmodel.ABCoupler(
        rad=rad_model,
        land=land_model,
        atmos=atmos_model,
    )
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    _, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

    # return final boundary layer height as scalar
    return trajectory.atmos.mixed.h_abl[-1]


def main():
    # forward mode
    grad_fn = jax.jacfwd(run_model)
    theta0 = 290.0
    dhf_dtheta0 = grad_fn(theta0)
    assert dhf_dtheta0 < 0.0
    print("forward mode: ∂h_final / ∂theta_0 =", dhf_dtheta0)

    # reverse mode
    grad_fn = jax.jacrev(run_model)
    theta0 = 290.0
    dhf_dtheta0 = grad_fn(theta0)
    print("reverse mode: ∂h_final / ∂theta_0 =", dhf_dtheta0)


if __name__ == "__main__":
    main()
