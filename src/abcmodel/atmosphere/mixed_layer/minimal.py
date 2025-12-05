from dataclasses import dataclass, field

import jax.numpy as jnp
from jaxtyping import Array, PyTree
from simple_pytree import Pytree

from ...utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


@dataclass
class MinimalMixedLayerState(Pytree):
    """Minimal mixed layer model initial state."""

    # the following variables are expected to be initialized by the user
    h_abl: Array
    """Initial ABL height [m]."""
    surf_pressure: Array
    """Surface pressure [Pa]."""
    theta: Array
    """Initial mixed-layer potential temperature [K]."""
    deltatheta: Array
    """Initial temperature jump at h [K]."""
    wtheta: Array
    """Surface kinematic heat flux [K m/s]."""
    q: Array
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: Array
    """Initial specific humidity jump at h [kg/kg]."""
    wq: Array
    """Surface kinematic moisture flux [kg/kg m/s]."""
    co2: Array
    """Initial mixed-layer CO2 [ppm]."""
    deltaCO2: Array
    """Initial CO2 jump at h [ppm]."""
    wCO2: Array
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: Array
    """Initial mixed-layer u-wind speed [m/s]."""
    v: Array
    """Initial mixed-layer v-wind speed [m/s]."""
    dz_h: Array
    """Transition layer thickness [-]."""

    # the following variables are initialized as zero
    wstar: Array = field(default_factory=lambda: jnp.array(1e-6))
    """Convective velocity scale [m s-1]."""
    wqe: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    wCO2A: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface assimilation CO2 flux [mgC m-2 s]."""
    wCO2R: Array = field(default_factory=lambda: jnp.array(0.0))
    """Surface respiration CO2 flux [mgC m-2 s]."""
    wCO2M: Array = field(default_factory=lambda: jnp.array(0.0))
    """CO2 mass flux [mgC m-2 s]."""
    wCO2e: Array = field(default_factory=lambda: jnp.array(0.0))
    """Entrainment CO2 flux [mgC m-2 s]."""

    # the following variables are expected to be assigned during warmup
    thetav: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Mixed-layer potential temperature [K]."""
    wthetav: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Surface kinematic virtual heat flux [K m s-1]."""
    qsat: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Saturation specific humidity [kg/kg]."""
    e: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Vapor pressure [Pa]."""
    esat: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Saturation vapor pressure [Pa]."""
    lcl: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Lifting condensation level [m]."""
    deltathetav: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Virtual temperature jump at h [K]."""
    top_p: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Pressure at top of mixed layer [Pa]."""
    top_T: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Temperature at top of mixed layer [K]."""
    top_rh: Array = field(default_factory=lambda: jnp.array(jnp.nan))
    """Relative humidity at top of mixed layer [-]."""


# alias
MinimalMixedLayerInitConds = MinimalMixedLayerState


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    """Minimal mixed layer model with constant properties."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """Pass."""
        return state

    def integrate(
        self, state: MinimalMixedLayerState, dt: float
    ) -> MinimalMixedLayerState:
        """Pass."""
        return state
