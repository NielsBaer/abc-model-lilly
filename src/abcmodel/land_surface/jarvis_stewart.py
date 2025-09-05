import numpy as np

from ..mixed_layer import AbstractMixedLayerModel
from ..radiation import AbstractRadiationModel
from ..surface_layer import AbstractSurfaceLayerModel
from ..utils import PhysicalConstants
from .standard import AbstractStandardLandSurfaceModel


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    def __init__(
        self,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2sat: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        surf_temp: float,
        cveg: float,
        wmax: float,
        wl: float,
        lam: float,
    ):
        super().__init__(
            wg,
            w2,
            temp_soil,
            temp2,
            a,
            b,
            p,
            cgsat,
            wsat,
            wfc,
            wwilt,
            c1sat,
            c2sat,
            lai,
            gD,
            rsmin,
            rssoilmin,
            alpha,
            surf_temp,
            cveg,
            wmax,
            wl,
            lam,
        )

    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        # calculate surface resistances using Jarvis-Stewart model
        f1 = radiation.get_f1()

        if self.w2 > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(-self.gD * (mixed_layer.esat - mixed_layer.e) / 100.0)
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - mixed_layer.theta) ** 2.0)

        self.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        pass
