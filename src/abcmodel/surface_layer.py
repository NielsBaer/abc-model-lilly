import numpy as np

from .components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)
from .utils import get_psih, get_psim, get_qsat


class InertSurfaceLayerModel(AbstractSurfaceLayerModel):
    def run(
        self,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        self.uw = -np.sign(mixed_layer.u) * (
            self.ustar**4.0 / (mixed_layer.v**2.0 / mixed_layer.u**2.0 + 1.0)
        ) ** (0.5)
        self.vw = -np.sign(mixed_layer.v) * (
            self.ustar**4.0 / (mixed_layer.u**2.0 / mixed_layer.v**2.0 + 1.0)
        ) ** (0.5)

    def compute_ra(self, u: float, v: float, wstar: float):
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        self.ra = ueff / max(1.0e-3, self.ustar) ** 2.0


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    def get_ribtol(self, zsl: float):
        if self.rib_number > 0.0:
            oblen = 1.0
            oblen0 = 2.0
        else:
            oblen = -1.0
            oblen0 = -2.0

        while abs(oblen - oblen0) > 0.001:
            oblen0 = oblen
            fx = (
                self.rib_number
                - zsl
                / oblen
                * (
                    np.log(zsl / self.z0h)
                    - get_psih(zsl / oblen)
                    + get_psih(self.z0h / oblen)
                )
                / (
                    np.log(zsl / self.z0m)
                    - get_psim(zsl / oblen)
                    + get_psim(self.z0m / oblen)
                )
                ** 2.0
            )
            oblen_start = oblen - 0.001 * oblen
            oblen_end = oblen + 0.001 * oblen
            fxdif = (
                (
                    -zsl
                    / oblen_start
                    * (
                        np.log(zsl / self.z0h)
                        - get_psih(zsl / oblen_start)
                        + get_psih(self.z0h / oblen_start)
                    )
                    / (
                        np.log(zsl / self.z0m)
                        - get_psim(zsl / oblen_start)
                        + get_psim(self.z0m / oblen_start)
                    )
                    ** 2.0
                )
                - (
                    -zsl
                    / oblen_end
                    * (
                        np.log(zsl / self.z0h)
                        - get_psih(zsl / oblen_end)
                        + get_psih(self.z0h / oblen_end)
                    )
                    / (
                        np.log(zsl / self.z0m)
                        - get_psim(zsl / oblen_end)
                        + get_psim(self.z0m / oblen_end)
                    )
                    ** 2.0
                )
            ) / (oblen_start - oblen_end)
            oblen = oblen - fx / fxdif

            if abs(oblen) > 1e15:
                break

        return oblen

    def run(
        self,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        ueff = max(
            0.01,
            np.sqrt(mixed_layer.u**2.0 + mixed_layer.v**2.0 + mixed_layer.wstar**2.0),
        )
        self.thetasurf = mixed_layer.theta + mixed_layer.wtheta / (self.drag_s * ueff)
        qsatsurf = get_qsat(self.thetasurf, mixed_layer.surf_pressure)
        cq = (1.0 + self.drag_s * ueff * land_surface.rs) ** -1.0
        self.qsurf = (1.0 - cq) * mixed_layer.q + cq * qsatsurf

        self.thetavsurf = self.thetasurf * (1.0 + 0.61 * self.qsurf)

        zsl = 0.1 * mixed_layer.abl_height
        self.rib_number = (
            self.const.g
            / mixed_layer.thetav
            * zsl
            * (mixed_layer.thetav - self.thetavsurf)
            / ueff**2.0
        )
        self.rib_number = min(self.rib_number, 0.2)

        # limamau: the following is rather slow
        # before they had the option:
        # self.L    = ribtol.ribtol(self.Rib, zsl, self.z0m, self.z0h) # Fast C++ iteration
        # we could make this faster with a scan or something using jax
        self.obukhov_length = self.get_ribtol(zsl)

        self.drag_m = (
            self.const.k**2.0
            / (
                np.log(zsl / self.z0m)
                - get_psim(zsl / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
            ** 2.0
        )
        self.drag_s = (
            self.const.k**2.0
            / (
                np.log(zsl / self.z0m)
                - get_psim(zsl / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
            / (
                np.log(zsl / self.z0h)
                - get_psih(zsl / self.obukhov_length)
                + get_psih(self.z0h / self.obukhov_length)
            )
        )

        self.ustar = np.sqrt(self.drag_m) * ueff
        self.uw = -self.drag_m * ueff * mixed_layer.u
        self.vw = -self.drag_m * ueff * mixed_layer.v

        # diagnostic meteorological variables
        self.temp_2m = (
            self.thetasurf
            - mixed_layer.wtheta
            / self.ustar
            / self.const.k
            * (
                np.log(2.0 / self.z0h)
                - get_psih(2.0 / self.obukhov_length)
                + get_psih(self.z0h / self.obukhov_length)
            )
        )
        self.q2m = self.qsurf - mixed_layer.wq / self.ustar / self.const.k * (
            np.log(2.0 / self.z0h)
            - get_psih(2.0 / self.obukhov_length)
            + get_psih(self.z0h / self.obukhov_length)
        )
        self.u2m = (
            -self.uw
            / self.ustar
            / self.const.k
            * (
                np.log(2.0 / self.z0m)
                - get_psim(2.0 / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
        )
        self.v2m = (
            -self.vw
            / self.ustar
            / self.const.k
            * (
                np.log(2.0 / self.z0m)
                - get_psim(2.0 / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
        )
        self.esat2m = 0.611e3 * np.exp(
            17.2694 * (self.temp_2m - 273.16) / (self.temp_2m - 35.86)
        )
        self.e2m = self.q2m * mixed_layer.surf_pressure / 0.622

    def compute_ra(self, u: float, v: float, wstar: float):
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        self.ra = (self.drag_s * ueff) ** -1.0
