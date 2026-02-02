from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .abstracts import AbstractCoupledState, AtmosT, LandT, RadT
from .coupling import ABCoupler

StateT = AbstractCoupledState[RadT, LandT, AtmosT]


def warmup(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    t: int,
    coupler: ABCoupler,
    dt: float,
    tstart: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmos.warmup(coupler.rad, coupler.land, state, t, dt, tstart)
    return state


def inner_step(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    _: None,  # this is here because the function signature requires it for a scan
    coupler: ABCoupler,
    dt: float,
    tstart: float,
) -> tuple[
    AbstractCoupledState[RadT, LandT, AtmosT],
    AbstractCoupledState[RadT, LandT, AtmosT],
]:
    """Run a single timestep of the model."""
    t = state.t
    atmos = coupler.atmos.statistics(state, t)
    state = state.replace(atmos=atmos)
    rad = coupler.rad.run(state, t, dt, tstart)
    state = state.replace(rad=rad)
    land = coupler.land.run(state)
    state = state.replace(land=land)
    atmos = coupler.atmos.run(state)
    state = state.replace(atmos=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = state.replace(land=land)
    atmos = coupler.atmos.integrate(state.atmos, dt)
    state = state.replace(atmos=atmos)
    state = coupler.compute_diagnostics(state)
    state = state.replace(t=t + 1)
    return state, state


def outter_step(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    _: None,  # this is here because the function signature requires it for a scan
    coupler: ABCoupler,
    inner_dt: float,
    inner_tsteps: int,
    tstart: float,
) -> tuple[
    AbstractCoupledState[RadT, LandT, AtmosT],
    AbstractCoupledState[RadT, LandT, AtmosT],
]:
    """A block of inner steps averaging the result."""
    initial_t = state.t
    step_fn_configured = partial(
        inner_step, coupler=coupler, dt=inner_dt, tstart=tstart
    )
    state, inner_traj = jax.lax.scan(
        step_fn_configured, state, None, length=inner_tsteps
    )
    avg_traj = jax.tree.map(lambda x: jnp.mean(x, axis=0), inner_traj)
    # this average block is tagged with the initial time
    avg_traj = avg_traj.replace(t=initial_t)
    return state, avg_traj


def integrate(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    inner_dt: float,
    outter_dt: float,
    runtime: float,
    tstart: float,
) -> tuple[Array, AbstractCoupledState[RadT, LandT, AtmosT]]:
    """Integrate the coupler forward in time."""

    inner_tsteps = int(np.floor(outter_dt / inner_dt))
    outter_tsteps = int(np.floor(runtime / outter_dt))

    # warmup and initial diagnostics (t=0)
    state = warmup(state, 0, coupler, inner_dt, tstart)
    state = coupler.compute_diagnostics(state)

    # configure outter step function
    step_fn = partial(
        outter_step,
        coupler=coupler,
        inner_dt=inner_dt,
        inner_tsteps=inner_tsteps,
        tstart=tstart,
    )

    # this is effectively the integration
    state, trajectory = jax.lax.scan(step_fn, state, length=outter_tsteps)

    # real time as separate output
    times = jnp.arange(outter_tsteps) * outter_dt / 3600.0 + tstart

    return times, trajectory
