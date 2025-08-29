from abc import abstractclassmethod

import numpy as np

from .utils import get_qsat


class AbstractCloudModel:
    def __init__(self):
        # cloud core fraction [-]
        self.cc_frac = 0.0
        # cloud core mass flux [m s-1]
        self.cc_mf = 0.0
        # cloud core moisture flux [kg kg-1 m s-1]
        self.cc_qf = 0.0

    @abstractclassmethod
    def run(
        self,
        wthetav: float,
        wqe: float,
        dq: float,
        abl_height: float,
        dz_h: float,
        wstar: float,
        wCO2e: float,
        wCO2M: float,
        dCO2: float,
        q: float,
        top_T: float,
        top_p: float,
        q2_h: float,
        top_CO22: float,
    ) -> tuple[float, float, float]:
        raise NotImplementedError


class NoCloudModel(AbstractCloudModel):
    def __init__(self):
        super().__init__()

    def run(
        self,
        wthetav: float,
        wqe: float,
        dq: float,
        abl_height: float,
        dz_h: float,
        wstar: float,
        wCO2e: float,
        wCO2M: float,
        dCO2: float,
        q: float,
        top_T: float,
        top_p: float,
        q2_h: float,
        top_CO22: float,
    ) -> tuple[float, float, float]:
        return q2_h, top_CO22, wCO2M


class StandardCumulusModel(AbstractCloudModel):
    def __init__(self):
        super().__init__()

    def run(
        self,
        wthetav: float,
        wqe: float,
        dq: float,
        abl_height: float,
        dz_h: float,
        wstar: float,
        wCO2e: float,
        wCO2M: float,
        dCO2: float,
        q: float,
        top_T: float,
        top_p: float,
        q2_h: float,
        top_CO22: float,
    ) -> tuple[float, float, float]:
        # calculate mixed-layer top relative humidity variance (Neggers et. al 2006/7)
        if wthetav > 0:
            q2_h = -(wqe + self.cc_qf) * dq * abl_height / (dz_h * wstar)
            top_CO22 = -(wCO2e + wCO2M) * dCO2 * abl_height / (dz_h * wstar)
        else:
            q2_h = 0.0
            top_CO22 = 0.0

        # calculate cloud core fraction (ac), mass flux (M) and moisture flux (wqM)
        self.cc_frac = max(
            0.0,
            0.5 + (0.36 * np.arctan(1.55 * ((q - get_qsat(top_T, top_p)) / q2_h**0.5))),
        )
        self.cc_mf = self.cc_frac * wstar
        self.cc_qf = self.cc_mf * q2_h**0.5

        # Only calculate CO2 mass-flux if mixed-layer top jump is negative
        if dCO2 < 0:
            wCO2M = self.cc_mf * top_CO22**0.5
        else:
            wCO2M = 0.0

        return q2_h, top_CO22, wCO2M
