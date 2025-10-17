from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import PyTree

from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


@dataclass
class MinimalMixedLayerInitConds:
    """Data class for minimal mixed layer model initial conditions.

    Args:
        abl_height: initial ABL height [m].
        surf_pressure: surface pressure [Pa].
        theta: initial mixed-layer potential temperature [K].
        dtheta: initial temperature jump at h [K].
        wtheta: surface kinematic heat flux [K m/s].
        q: initial mixed-layer specific humidity [kg/kg].
        dq: initial specific humidity jump at h [kg/kg].
        wq: surface kinematic moisture flux [kg/kg m/s].
        co2: initial mixed-layer CO2 [ppm].
        dCO2: initial CO2 jump at h [ppm].
        wCO2: surface kinematic CO2 flux [mgC/m²/s].
        u: initial mixed-layer u-wind speed [m/s].
        v: initial mixed-layer v-wind speed [m/s].
        dz_h: transition layer thickness [-].
        wstar: convective velocity scale [m s-1]. Defaults to 1e-6.
        wCO2A: surface assimulation CO2 flux [mgC/m²/s]. Defaults to 0.0.
        wCO2R: surface respiration CO2 flux [mgC/m²/s]. Defaults to 0.0.
        wCO2M: CO2 mass flux [mgC/m²/s]. Defaults to 0.0.
        wqe: entrainment moisture flux [kg kg-1 m s-1]. Defaults to 0.0.
        wCO2e: entrainment CO2 flux [mgC/m²/s]. Defaults to 0.0.
    """

    # the following variables are expected to be initialized by the user
    abl_height: float
    surf_pressure: float
    theta: float
    dtheta: float
    wtheta: float
    q: float
    dq: float
    wq: float
    co2: float
    dCO2: float
    wCO2: float
    u: float
    v: float
    dz_h: float

    # the following variables are initialized as zero
    wstar: float = 1e-6
    wCO2A: float = 0.0
    wCO2R: float = 0.0
    wCO2M: float = 0.0
    wqe: float = 0.0
    wCO2e: float = 0.0

    # the following variables are expected to be assigned during warmup
    thetav: float = jnp.nan
    wthetav: float = jnp.nan
    qsat: float = jnp.nan
    e: float = jnp.nan
    esat: float = jnp.nan
    lcl: float = jnp.nan


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    """Minimal mixed layer model with constant properties."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """Pass."""
        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        """Pass."""
        return state
