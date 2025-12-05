import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .coupling import ABCoupler


def print_nan_variables(state: PyTree, prefix: str = ""):
    """Print all variables in a CoupledState that have NaN values, recursively."""
    nan_vars = []

    # If it's a dataclass or SimpleNamespace, iterate over fields
    if hasattr(state, "__dict__"):
        items = state.__dict__.items()
    elif hasattr(state, "_asdict"):  # NamedTuple
        items = state._asdict().items()
    else:
        return nan_vars

    for name, value in items:
        full_name = f"{prefix}.{name}" if prefix else name
        try:
            is_nan = False

            # check JAX arrays
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                if jnp.issubdtype(value.dtype, jnp.floating):
                    if jnp.any(jnp.isnan(value)):
                        is_nan = True
            # check numpy arrays
            elif hasattr(value, "dtype") and np.issubdtype(value.dtype, np.floating):
                if np.any(np.isnan(value)):
                    is_nan = True
            # check regular float values
            elif isinstance(value, float) and math.isnan(value):
                is_nan = True

            # Recursive check for nested objects (like AtmosphereState)
            if hasattr(value, "__dict__") or hasattr(value, "_asdict"):
                nan_vars.extend(print_nan_variables(value, full_name))

            if is_nan:
                nan_vars.append((full_name, value))
                print(f"Variable '{full_name}' contains NaN: {value}")

        except (TypeError, AttributeError, Exception):
            # skip variables that can't be checked for NaN
            continue

    return nan_vars


def warmup(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    """Warmup the model by running it for a few timesteps."""
    # Update atmosphere statistics
    # statistics returns AtmosphereState, so we assign to state.atmosphere
    state = replace(
        state,
        atmosphere=coupler.atmosphere.statistics(state.atmosphere, t, coupler.const),
    )

    # calculate initial diagnostic variables
    # radiation.run returns RadiationState, so we assign to state.radiation
    state = replace(state, radiation=coupler.radiation.run(state, t, dt, coupler.const))

    # warmup atmosphere and land
    # atmosphere.warmup returns CoupledState, so we assign to state
    state = coupler.atmosphere.warmup(state, coupler.const, coupler.land)

    return state


def timestep(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    """Run a single timestep of the model."""
    atmos = coupler.atmosphere.statistics(state.atmosphere, t, coupler.const)
    state = replace(state, atmosphere=atmos)
    rad = coupler.radiation.run(state, t, dt, coupler.const)
    state = replace(state, radiation=rad)
    land = coupler.land.run(state, coupler.const)
    state = replace(state, land=land)
    atmos = coupler.atmosphere.run(state, coupler.const)
    state = replace(state, atmosphere=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = replace(state, land=land)
    atmos = coupler.atmosphere.integrate(state.atmosphere, dt)
    state = replace(state, atmosphere=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(state: PyTree, coupler: ABCoupler, dt: float, runtime: float):
    """Integrate the coupler forward in time.

    Args:
        state: Initial coupled state.
        coupler: ABCoupler instance.
        dt: Time step [s].
        runtime: Total runtime [s].

    Returns:
        times: Array of time values [h].
        trajectory: PyTree containing the full state trajectory.
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

    times = jnp.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart

    return times, trajectory
