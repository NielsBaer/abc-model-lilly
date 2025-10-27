from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import PyTree

from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


@dataclass
class MinimalMixedLayerInitConds:
    """Minimal mixed layer model initial state."""

    # the following variables are expected to be initialized by the user
    abl_height: float
    """Initial ABL height [m]."""
    surf_pressure: float
    """Surface pressure [Pa]."""
    theta: float
    """Initial mixed-layer potential temperature [K]."""
    dtheta: float
    """Initial temperature jump at h [K]."""
    wtheta: float
    """Surface kinematic heat flux [K m/s]."""
    q: float
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: float
    """Initial specific humidity jump at h [kg/kg]."""
    wq: float
    """Surface kinematic moisture flux [kg/kg m/s]."""
    co2: float
    """Initial mixed-layer CO2 [ppm]."""
    dCO2: float
    """Initial CO2 jump at h [ppm]."""
    wCO2: float
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: float
    """Initial mixed-layer u-wind speed [m/s]."""
    v: float
    """Initial mixed-layer v-wind speed [m/s]."""
    dz_h: float
    """Transition layer thickness [-]."""

    # the following variables are initialized as zero
    wstar: float = 1e-6
    """Convective velocity scale [m s-1]."""
    wqe: float = 0.0
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    wCO2A: float = 0.0
    """Surface assimilation CO2 flux [mgC m-2 s]."""
    wCO2R: float = 0.0
    """Surface respiration CO2 flux [mgC m-2 s]."""
    wCO2M: float = 0.0
    """CO2 mass flux [mgC m-2 s]."""
    wCO2e: float = 0.0
    """Entrainment CO2 flux [mgC m-2 s]."""

    # the following variables are expected to be assigned during warmup
    thetav: float = jnp.nan
    """Mixed-layer potential temperature [K]."""
    wthetav: float = jnp.nan
    """Surface kinematic virtual heat flux [K m s-1]."""
    qsat: float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    e: float = jnp.nan
    """Vapor pressure [Pa]."""
    esat: float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    lcl: float = jnp.nan
    """Lifting condensation level [m]."""


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
