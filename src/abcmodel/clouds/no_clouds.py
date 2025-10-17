from dataclasses import dataclass

from jaxtyping import PyTree

from ..models import AbstractCloudModel
from ..utils import PhysicalConstants


@dataclass
class NoCloudInitConds:
    """No cloud initial conditions."""

    cc_frac: float = 0.0
    cc_mf: float = 0.0
    cc_qf: float = 0.0


class NoCloudModel(AbstractCloudModel):
    """No cloud is formed using this model."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """No calculations."""
        return state
