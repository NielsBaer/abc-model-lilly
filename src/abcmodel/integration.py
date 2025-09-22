import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .clouds import NoCloudModel
from .coupling import ABCoupler


# limamau: this (or something similar) could be a check to run after
# warmup since the static type checking for the state is now bad
def print_nan_variables(state: PyTree):
    """Print all variables in a CoupledState that have NaN values"""
    nan_vars = []

    for name, value in state.__dict__.items():
        try:
            is_nan = False

            # Check JAX arrays
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                if jnp.issubdtype(value.dtype, jnp.floating):
                    if jnp.any(jnp.isnan(value)):
                        is_nan = True
            # Check numpy arrays
            elif hasattr(value, "dtype") and np.issubdtype(value.dtype, np.floating):
                if np.any(np.isnan(value)):
                    is_nan = True
            # Check regular float values
            elif isinstance(value, float) and math.isnan(value):
                is_nan = True

            if is_nan:
                nan_vars.append((name, value))
                print(f"Variable '{name}' contains NaN: {value}")

        except (TypeError, AttributeError, Exception):
            # Skip variables that can't be checked for NaN
            continue

    if not nan_vars:
        print("No NaN variables found in the state.")

    return nan_vars


def warmup(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    state = coupler.mixed_layer.statistics(state, t, coupler.const)

    # calculate initial diagnostic variables
    state = coupler.radiation.run(state, t, dt, coupler.const)

    for _ in range(10):
        state = coupler.surface_layer.run(state, coupler.const)

    state = coupler.land_surface.run(state, coupler.const, coupler.surface_layer)

    if not isinstance(coupler.clouds, NoCloudModel):
        state = coupler.mixed_layer.run(state, coupler.const)
        state = coupler.clouds.run(state, coupler.const)

    state = coupler.mixed_layer.run(state, coupler.const)

    print_nan_variables(state)

    return state


def timestep(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    state = coupler.mixed_layer.statistics(state, t, coupler.const)

    # run radiation model
    state = coupler.radiation.run(state, t, dt, coupler.const)

    # run surface layer model
    state = coupler.surface_layer.run(state, coupler.const)

    # run land surface model
    state = coupler.land_surface.run(state, coupler.const, coupler.surface_layer)

    # run cumulus parameterization
    state = coupler.clouds.run(state, coupler.const)

    # run mixed-layer model
    state = coupler.mixed_layer.run(state, coupler.const)

    # time integrate land surface model
    state = coupler.land_surface.integrate(state, dt)

    # time integrate mixed-layer model
    state = coupler.mixed_layer.integrate(state, dt)

    return state


def create_trajectory_collector():
    """Create a simple trajectory collector function."""
    data = {}

    def collect(state):
        # Initialize on first call
        if not data:
            for name in state.__dict__.keys():
                data[name] = []

        # Collect current values
        for name, value in state.__dict__.items():
            if name not in data:
                data[name] = [None] * (len(next(iter(data.values()))) if data else 0)

            # Make copy to avoid reference issues
            if hasattr(value, "copy"):
                data[name].append(value.copy())
            else:
                data[name].append(
                    jnp.array(value) if hasattr(value, "__array__") else value
                )

    def to_namespace():
        array_data = {name: jnp.array(values) for name, values in data.items()}
        return SimpleNamespace(**array_data)

    collect.to_namespace = to_namespace
    return collect


def integrate(
    state: PyTree,
    coupler: ABCoupler,
    dt: float,
    runtime: float,
):
    """Integrate the coupler forward in time."""
    tsteps = int(np.floor(runtime / dt))

    # warmup
    state = warmup(state, coupler, 0, dt)

    # let's go
    collect_trajectory = create_trajectory_collector()
    for t in range(tsteps):
        state = timestep(state, coupler, t, dt)
        collect_trajectory(state)

    # # ---------------------------------------------------------------------
    # # limamau: maybe this is an advanced step for now
    # def fun(carry, _):
    #     state, t = carry
    #     state = timestep(state, coupler, t, dt)
    #     return (state, t + 1), state

    # state, trajectory = jax.lax.scan(fun, (state, 0), None, length=tsteps)
    # # ---------------------------------------------------------------------

    times = np.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart
    trajectory = collect_trajectory.to_namespace()

    return times, trajectory
