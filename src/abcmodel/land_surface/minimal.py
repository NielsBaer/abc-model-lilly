from ..components import AbstractLandSurfaceModel
from ..mixed_layer import AbstractMixedLayerModel
from ..radiation import AbstractRadiationModel
from ..surface_layer import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants, get_esat, get_qsat


class MinimalLandSurfaceModel(AbstractLandSurfaceModel):
    def __init__(self, alpha: float, surf_temp: float, rs: float):
        # surface albedo [-]
        self.alpha = alpha
        # surface temperature [K]
        self.surf_temp = surf_temp
        # surface resistance [s m-1]
        self.rs = rs

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # limamau: the following two blocks are also computed by
        # the standard class - should we refactor some things here?
        # compute aerodynamic resistance
        surface_layer.compute_ra(mixed_layer.u, mixed_layer.v, mixed_layer.wstar)

        # calculate essential thermodynamic variables
        mixed_layer.esat = get_esat(mixed_layer.theta)
        mixed_layer.qsat = get_qsat(mixed_layer.theta, mixed_layer.surf_pressure)
        desatdT = mixed_layer.esat * (
            17.2694 / (mixed_layer.theta - 35.86)
            - 17.2694
            * (mixed_layer.theta - 273.16)
            / (mixed_layer.theta - 35.86) ** 2.0
        )
        mixed_layer.dqsatdT = 0.622 * desatdT / mixed_layer.surf_pressure
        mixed_layer.e = mixed_layer.q * mixed_layer.surf_pressure / 0.622

    def integrate(self, dt: float):
        pass
