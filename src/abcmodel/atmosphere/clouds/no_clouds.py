from dataclasses import dataclass

from jaxtyping import PyTree

from ...utils import PhysicalConstants
from ..abstracts import AbstractCloudModel


@dataclass
class NoCloudInitConds:
    """No cloud initial state."""

    cc_frac: float = 0.0
    """Cloud core fraction [-], range 0 to 1."""
    cc_mf: float = 0.0
    """Cloud core mass flux [kg/kg/s]."""
    cc_qf: float = 0.0
    """Cloud core moisture flux [kg/kg/s]."""
    cl_trans: float = 1.0
    """Cloud layer transmittance [-], range 0 to 1."""
    q2_h: float = 0.0
    """Humidity variance at mixed-layer top [kg²/kg²]."""
    top_CO22: float = 0.0
    """CO2 variance at mixed-layer top [ppm²]."""
    wCO2M: float = 0.0
    """CO2 mass flux [mgC/m²/s]."""


class NoCloudModel(AbstractCloudModel):
    """No cloud is formed using this model."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """No calculations."""
        return state
