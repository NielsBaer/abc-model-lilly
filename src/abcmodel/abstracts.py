"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod
from typing import Generic, TypeVar

from flax import nnx
from jax import Array
from simple_pytree import Pytree


class AbstractState(Pytree):
    """Abstract state class to define the interface for all states."""


StateT = TypeVar("StateT", bound=AbstractState)


class AbstractRadiationState(AbstractState):
    """Abstract rad state."""

    net_rad: Array
    """Net surface rad [W m-2]."""
    in_srad: Array
    """Incoming solar rad [W m-2]."""
    out_srad: Array
    """Outgoing solar rad [W m-2]."""
    in_lrad: Array
    """Incoming longwave rad [W m-2]."""
    out_lrad: Array
    """Outgoing longwave rad [W m-2]."""


class AbstractLandState(AbstractState):
    """Abstract land state."""

    alpha: Array
    """surface albedo [-], range 0 to 1."""
    surf_temp: Array
    """Surface temperature [K]."""
    rs: Array
    """Surface resistance [s m-1]."""
    wg: Array
    """No moisture content in the root zone [m3 m-3]."""
    wl: Array
    """No water content in the canopy [m]."""
    esat: Array
    """Saturation vapor pressure [Pa]."""
    qsat: Array
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array
    """Vapor pressure [Pa]."""
    qsatsurf: Array
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array
    """Kinematic heat flux [K m/s]."""
    wq: Array
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


class AbstractAtmosphereState(AbstractState):
    """Abstract atmos state."""

    @property
    @abstractmethod
    def theta(self) -> Array:
        """Potential temperature [K]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def q(self) -> Array:
        """Specific humidity [kg/kg]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def co2(self) -> Array:
        """CO2 concentration [ppmv]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def surf_pressure(self) -> Array:
        """Surface pressure [Pa]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def u(self) -> Array:
        """Zonal wind speed [m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self) -> Array:
        """Meridional wind speed [m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ra(self) -> Array:
        """Aerodynamic resistance [s/m]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def thetasurf(self) -> Array:
        """Surface potential temperature [K]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def h_abl(self) -> Array:
        """Boundary layer height [m]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ustar(self) -> Array:
        """Friction velocity [m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def uw(self) -> Array:
        """Zonal momentum flux [m²/s²]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vw(self) -> Array:
        """Meridional momentum flux [m²/s²]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def wstar(self) -> Array:
        """Convective velocity scale [m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def thetav(self) -> Array:
        """Mixed-layer virtual potential temperature [K]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def top_T(self) -> Array:
        """Temperature at top of mixed layer [K]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def top_p(self) -> Array:
        """Pressure at top of mixed layer [Pa]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cc_mf(self) -> Array:
        """Cloud core mass flux [kg/kg/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cc_qf(self) -> Array:
        """Cloud core moisture flux [kg/kg/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def wCO2M(self) -> Array:
        """Cloud core CO2 mass flux [mgC/m²/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cc_frac(self) -> Array:
        """Cloud core fraction [-]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def wthetav(self) -> Array:
        """Virtual potential temperature flux at surface [K m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def wqe(self) -> Array:
        """Entrainment moisture flux [kg/kg m/s]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dq(self) -> Array:
        """Specific humidity jump at h [kg/kg]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dz_h(self) -> Array:
        """Transition layer thickness [m]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def deltaCO2(self) -> Array:
        """CO2 jump at h [ppm]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def wCO2e(self) -> Array:
        """Entrainment CO2 flux [mgC/m²/s]."""
        raise NotImplementedError


RadT = TypeVar("RadT", bound=AbstractRadiationState)
LandT = TypeVar("LandT", bound=AbstractLandState)
AtmosT = TypeVar("AtmosT", bound=AbstractAtmosphereState)


class AbstractCoupledState(AbstractState, Generic[RadT, LandT, AtmosT]):
    """Abstract coupled state, generic over rad, land and atmos types."""

    rad: RadT
    land: LandT
    atmos: AtmosT
    t: Array
    total_water_mass: Array

    @property
    def net_rad(self) -> Array:
        """Net surface rad [W m-2]."""
        return self.rad.net_rad

    @property
    def in_srad(self) -> Array:
        """Incoming shortwave rad [W m-2]."""
        return self.rad.in_srad


class AbstractModel(nnx.Module, Generic[StateT]):
    """Abstract model class to define the interface for all models."""

    @abstractmethod
    def init_state(self, *args, **kwargs) -> StateT:
        raise NotImplementedError


class AbstractRadiationModel(AbstractModel, Generic[RadT]):
    """Abstract radiation model class to define the interface for all rad models."""

    tstart: Array
    """Start time of the model."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        dt: float,
        tstart: float,
    ) -> RadT:
        raise NotImplementedError


class AbstractLandModel(AbstractModel, Generic[LandT]):
    """Abstract land model class to define the interface for all land models."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
    ) -> LandT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: LandT, dt: float) -> LandT:
        raise NotImplementedError


class AbstractAtmosphereModel(AbstractModel, Generic[AtmosT]):
    """Abstract atmos model class to define the interface for all atmos models."""

    @abstractmethod
    def warmup(
        self,
        radmodel: AbstractRadiationModel[RadT],
        landmodel: AbstractLandModel[LandT],
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        dt: float,
        tstart: float,
    ) -> AbstractCoupledState[RadT, LandT, AtmosT]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
    ) -> AtmosT:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
    ) -> AtmosT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AtmosT, dt: float) -> AtmosT:
        raise NotImplementedError
