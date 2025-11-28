from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import jax

from .abstracts import (
    AbstractCloudModel,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from .utils import PhysicalConstants


# limamau: is this ""optimized""??
# also, for the static type checking of the code,
# it would be nice to guarantee that all fields
# are at least an instance of jax.Array
@jax.tree_util.register_pytree_node_class
class CoupledState(SimpleNamespace):
    """A pytree-compatible state with attribute access! But not very type friendly..."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tree_flatten(self):
        """:meta private:"""
        # children are the values, aux is the keys
        return list(self.__dict__.values()), list(self.__dict__.keys())

    @classmethod
    def tree_unflatten(cls, aux, children):
        """:meta private:"""
        return cls(**dict(zip(aux, children)))


class ABCoupler:
    """Coupling class to bound all the components."""

    def __init__(
        self,
        radiation: AbstractRadiationModel,
        land_surface: AbstractLandSurfaceModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
        clouds: AbstractCloudModel,
    ):
        # constants
        self.const = PhysicalConstants()

        # models
        self.radiation = radiation
        self.land_surface = land_surface
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.clouds = clouds

    @staticmethod
    def init_state(
        radiation_init_conds: Any,
        land_surface_init_conds: Any,
        surface_layer_init_conds: Any,
        mixed_layer_init_conds: Any,
        clouds_init_conds: Any,
    ):
        state_dict = {}
        state_dict.update(asdict(radiation_init_conds))
        state_dict.update(asdict(land_surface_init_conds))
        state_dict.update(asdict(surface_layer_init_conds))
        state_dict.update(asdict(mixed_layer_init_conds))
        state_dict.update(asdict(clouds_init_conds))

        # diagnostic variables for water and energy budgets
        state_dict["total_water_mass"] = 0.0
        state_dict["total_energy"] = 0.0

        state = CoupledState(**state_dict)
        return state

    def compute_diagnostics(self, state: CoupledState) -> CoupledState:
        """Compute diagnostic variables for total water budget.

        In the future it would be nice to include the energy budget, although
        this is significantly more complicated.

        Notes:
            Total water mass (kg/m²):

            - water vapor in the mixed layer: :math:`q \\rho h`;
            - soil moisture in layer 1: :math:`w_g \\rho_w`;
            - soil moisture in layer 2: :math:`w_2 \\rho_w`.
            - canopy moisture: :math:`w_l \\rho_w`.
        """
        # total water mass (kg/m²)
        vap_w = state.q * self.const.rho * state.abl_height
        s1_w = state.wg * self.const.rhow * self.land_surface.d1
        can_w = state.wl * self.const.rhow
        state.total_water_mass = vap_w + s1_w + can_w

        return state
