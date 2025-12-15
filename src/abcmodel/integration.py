import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .abstracts import AbstractCoupledState, AtmosT, LandT, RadT
from .coupling import ABCoupler


def warmup(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmos.warmup(coupler.rad, coupler.land, state, t, dt, coupler.const)
    return state


def timestep(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Run a single timestep of the model."""
    atmos = coupler.atmos.statistics(state, t, coupler.const)
    state = state.replace(atmos=atmos)
    rad = coupler.rad.run(state, t, dt, coupler.const)
    state = state.replace(rad=rad)
    land = coupler.land.run(state, coupler.const)
    state = state.replace(land=land)
    atmos = coupler.atmos.run(state, coupler.const)
    state = state.replace(atmos=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = state.replace(land=land)
    atmos = coupler.atmos.integrate(state.atmos, dt)
    state = state.replace(atmos=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    dt: float,
    runtime: float,
) -> tuple[Array, AbstractCoupledState[RadT, LandT, AtmosT]]:
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
