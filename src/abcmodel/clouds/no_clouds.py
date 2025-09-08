import numpy as np

from ..models import (
    AbstractCloudModel,
    AbstractDiagnostics,
    AbstractInitConds,
    AbstractMixedLayerModel,
    AbstractParams,
)


class NoCloudParams(AbstractParams["NoCloudModel"]):
    """No cloud parameters."""

    def __init__(self):
        pass


class NoCloudInitConds(AbstractInitConds["NoCloudModel"]):
    """No cloud initial conditions."""

    def __init__(self):
        self.cc_frac = 0.0
        self.cc_mf = 0.0
        self.cc_qf = 0.0


# limamau: maybe we could just use NoDiagnostics instead of this ;)
class NoCloudDiagnostics(AbstractDiagnostics["NoCloudModel"]):
    """Class for no cloud model diagnostics.

    This is more of a software consistency thing than anything actually meaningful.

    Variables
    ---------
    - ``cc_frac``: cloud core fraction [-], range 0 to 1.
    - ``cc_mf``: cloud core mass flux [m/s].
    - ``cc_qf``: cloud core moisture flux [kg/kg/s].
    """

    def post_init(self, tsteps: int):
        self.cc_frac = np.zeros(tsteps)
        self.cc_mf = np.zeros(tsteps)
        self.cc_qf = np.zeros(tsteps)

    def store(self, t: int, model: "NoCloudModel"):
        self.cc_frac[t] = model.cc_frac
        self.cc_mf[t] = model.cc_mf
        self.cc_qf[t] = model.cc_qf


class NoCloudModel(AbstractCloudModel):
    """
    No cloud is formed using this model.
    """

    def __init__(
        self,
        params: NoCloudParams,
        init_cond: NoCloudInitConds,
        diagnostics: AbstractDiagnostics = NoCloudDiagnostics(),
    ):
        self.cc_frac = init_cond.cc_frac
        self.cc_mf = init_cond.cc_mf
        self.cc_qf = init_cond.cc_qf
        self.diagnostics = diagnostics

    def run(self, mixed_layer: AbstractMixedLayerModel):
        """No calculations."""
        pass
