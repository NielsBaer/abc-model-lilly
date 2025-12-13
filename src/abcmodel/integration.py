from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .coupling import A, ABCoupler, CoupledState, L, R


def warmup(
    state: CoupledState[R, L, A], coupler: ABCoupler, t: int, dt: float
) -> CoupledState[R, L, A]:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmos.warmup(coupler.rad, coupler.land, state, t, dt, coupler.const)
    return state


def timestep(
    state: CoupledState[R, L, A], coupler: ABCoupler, t: int, dt: float
) -> CoupledState[R, L, A]:
    """Run a single timestep of the model."""
    atmos = coupler.atmos.statistics(state.atmos, t, coupler.const)
    state = state.replace(atmos=atmos)
    rad = coupler.rad.run(state, t, dt, coupler.const)
    state = state.replace(rad=rad)
    land = coupler.land.run(state, coupler.const)
    state = state.replace(land=land)
    atmos = coupler.atmos.run(state, coupler.const)
    state = state.replace(atmos=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = replace(state, land=land)
    atmos = coupler.atmos.integrate(state.atmos, dt)
    state = replace(state, atmos=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(
    state: CoupledState[R, L, A], coupler: ABCoupler, dt: float, runtime: float
) -> tuple[Array, CoupledState[R, L, A]]:
    """Integrate the coupler forward in time.

    Args:
        state: Initial coupled state.
        coupler: ABCoupler instance.
        dt: Time step [s].
        runtime: Total runtime [s].

    Returns:
        times: Array of time values [h].
        trajectory: CoupledState containing the full state trajectory.
    """
    tsteps = int(np.floor(runtime / dt))

    # warmup
    state = warmup(state, coupler, 0, dt)
    state = coupler.compute_diagnostics(state)

    def iter_fn(state, t):
        state = timestep(state, coupler, t, dt)
        return state, state

    timesteps = jnp.arange(tsteps)
    state, trajectory = jax.lax.scan(iter_fn, state, timesteps, length=tsteps)

    times = jnp.arange(tsteps) * dt / 3600.0 + coupler.rad.tstart

    return times, trajectory
