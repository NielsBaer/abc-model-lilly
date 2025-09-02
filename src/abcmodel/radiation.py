import numpy as np

from .components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
)


class NoRadiationModel(AbstractRadiationModel):
    # limamau: this shouldn't need all this arguments
    # to be cleaned up in the future
    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
        net_rad: float,
        dFz: float,
    ):
        super().__init__(
            lat,
            lon,
            doy,
            tstart,
            cc,
            net_rad,
            dFz,
        )

    def run(
        self,
        t: float,
        dt: float,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        pass

    def get_f1(self):
        return 1.0


class StandardRadiationModel(AbstractRadiationModel):
    def __init__(
        self,
        lat: float,
        lon: float,
        doy: float,
        tstart: float,
        cc: float,
        net_rad: float,
        dFz: float,
    ):
        super().__init__(
            lat,
            lon,
            doy,
            tstart,
            cc,
            net_rad,
            dFz,
        )

    def run(
        self,
        t: float,
        dt: float,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        sda = 0.409 * np.cos(2.0 * np.pi * (self.doy - 173.0) / 365.0)
        sinlea = np.sin(2.0 * np.pi * self.lat / 360.0) * np.sin(sda) - np.cos(
            2.0 * np.pi * self.lat / 360.0
        ) * np.cos(sda) * np.cos(
            2.0 * np.pi * (t * dt + self.tstart * 3600.0) / 86400.0
            + 2.0 * np.pi * self.lon / 360.0
        )
        sinlea = max(sinlea, 0.0001)

        Ta = mixed_layer.theta * (
            (
                mixed_layer.surf_pressure
                - 0.1 * mixed_layer.abl_height * self.const.rho * self.const.g
            )
            / mixed_layer.surf_pressure
        ) ** (self.const.rd / self.const.cp)

        Tr = (0.6 + 0.2 * sinlea) * (1.0 - 0.4 * self.cc)

        self.in_srad = self.const.solar_in * Tr * sinlea
        self.out_srad = land_surface.alpha * self.const.solar_in * Tr * sinlea
        self.in_lrad = 0.8 * self.const.bolz * Ta**4.0
        self.out_lrad = self.const.bolz * land_surface.surf_temp**4.0

        self.net_rad = self.in_srad - self.out_srad + self.in_lrad - self.out_lrad

    def get_f1(self):
        f1 = 1.0 / min(
            1.0,
            ((0.004 * self.in_srad + 0.05) / (0.81 * (0.004 * self.in_srad + 1.0))),
        )
        return f1
