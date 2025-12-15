from dataclasses import dataclass, field

import jax.numpy as jnp

from ...coupling import AbstractCoupledState
from ...utils import Array, PhysicalConstants
from ..abstracts import AbstractCloudModel, AbstractCloudState


@dataclass
class NoCloudState(AbstractCloudState):
    """No cloud initial state."""

    cc_frac: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core fraction [-], range 0 to 1."""
    cc_mf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core mass flux [kg/kg/s]."""
    cc_qf: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud core moisture flux [kg/kg/s]."""
    cl_trans: Array = field(default_factory=lambda: jnp.array(0.0))
    """Cloud layer transmittance [-], range 0 to 1."""
    q2_h: Array = field(default_factory=lambda: jnp.array(0.0))
    """Humidity variance at mixed-layer top [kg²/kg²]."""
    top_CO22: Array = field(default_factory=lambda: jnp.array(0.0))
    """CO2 variance at mixed-layer top [ppm²]."""
    wCO2M: Array = field(default_factory=lambda: jnp.array(0.0))
    """CO2 mass flux [mgC/m²/s]."""


NoCloudInitConds = NoCloudState


class NoCloudModel(AbstractCloudModel):
    """No cloud is formed using this model."""

    def __init__(self):
        pass

    def run(
        self, state: AbstractCoupledState, const: PhysicalConstants
    ) -> NoCloudState:
        """No calculations."""
        return state.atmos.clouds
