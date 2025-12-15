"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod
from typing import Generic, TypeVar

from jax import Array
from simple_pytree import Pytree

from .utils import PhysicalConstants


class AbstractState(Pytree):
    """Abstract state class to define the interface for all states."""


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


RadT = TypeVar("RadT", bound=AbstractRadiationState)
LandT = TypeVar("LandT", bound=AbstractLandState)
AtmosT = TypeVar("AtmosT", bound=AbstractAtmosphereState)


class AbstractCoupledState(AbstractState, Generic[RadT, LandT, AtmosT]):
    """Abstract coupled state, generic over rad, land and atmos types."""

    rad: RadT
    land: LandT
    atmos: AtmosT
    total_water_mass: Array

    @property
    def net_rad(self) -> Array:
        """Net surface rad [W m-2]."""
        return self.rad.net_rad

    @property
    def in_srad(self) -> Array:
        """Incoming shortwave rad [W m-2]."""
        return self.rad.in_srad


# limamau: for now this is not needed,
# but in the future we might say that this is an eqx.Module or something
class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel, Generic[RadT]):
    """Abstract rad model class to define the interface for all rad models."""

    tstart: Array
    """Start time of the model."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> RadT:
        raise NotImplementedError


class AbstractLandModel(AbstractModel, Generic[LandT]):
    """Abstract land model class to define the interface for all land models."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        const: PhysicalConstants,
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
        const: PhysicalConstants,
    ) -> AbstractCoupledState[RadT, LandT, AtmosT]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        const: PhysicalConstants,
    ) -> AtmosT:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        const: PhysicalConstants,
    ) -> AtmosT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AtmosT, dt: float) -> AtmosT:
        raise NotImplementedError
