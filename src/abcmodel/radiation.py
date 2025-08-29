from abc import abstractmethod

import numpy as np

from .utils import PhysicalConstants


class AbstractRadiationModel:
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
        # constants
        self.const = PhysicalConstants()
        # latitude [deg]
        self.lat = lat
        # longitude [deg]
        self.lon = lon
        # day of the year [-]
        self.doy = doy
        # time of the day [h UTC]
        self.tstart = tstart
        # cloud cover fraction [-]
        self.cc = cc
        # net radiation [W m-2]
        self.net_rad = net_rad
        # cloud top radiative divergence [W m-2]
        self.dFz = dFz
        # incoming short wave radiation [W m-2]
        self.in_srad = None
        # outgoing short wave radiation [W m-2]
        self.out_srad = None
        # incoming long wave radiation [W m-2]
        self.in_lrad = None
        # outgoing long wave radiation [W m-2]
        self.out_lrad = None

    @abstractmethod
    def run(
        self,
        t: float,
        dt: float,
        theta: float,
        surf_pressure: float,
        abl_height: float,
        alpha: float,
        Ts: float,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_f1(self) -> float:
        pass


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
        theta: float,
        surf_pressure: float,
        abl_height: float,
        alpha: float,
        Ts: float,
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
        theta: float,
        surf_pressure: float,
        abl_height: float,
        alpha: float,
        Ts: float,
    ):
        sda = 0.409 * np.cos(2.0 * np.pi * (self.doy - 173.0) / 365.0)
        sinlea = np.sin(2.0 * np.pi * self.lat / 360.0) * np.sin(sda) - np.cos(
            2.0 * np.pi * self.lat / 360.0
        ) * np.cos(sda) * np.cos(
            2.0 * np.pi * (t * dt + self.tstart * 3600.0) / 86400.0
            + 2.0 * np.pi * self.lon / 360.0
        )
        sinlea = max(sinlea, 0.0001)

        Ta = theta * (
            (surf_pressure - 0.1 * abl_height * self.const.rho * self.const.g)
            / surf_pressure
        ) ** (self.const.rd / self.const.cp)

        Tr = (0.6 + 0.2 * sinlea) * (1.0 - 0.4 * self.cc)

        self.in_srad = self.const.solar_in * Tr * sinlea
        self.out_srad = alpha * self.const.solar_in * Tr * sinlea
        self.in_lrad = 0.8 * self.const.bolz * Ta**4.0
        self.out_lrad = self.const.bolz * Ts**4.0

        self.net_rad = self.in_srad - self.out_srad + self.in_lrad - self.out_lrad

    def get_f1(self):
        f1 = 1.0 / min(
            1.0,
            ((0.004 * self.in_srad + 0.05) / (0.81 * (0.004 * self.in_srad + 1.0))),
        )
        return f1
