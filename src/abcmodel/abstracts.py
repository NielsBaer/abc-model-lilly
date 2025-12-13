"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod
from typing import Generic, TypeVar

from jax import Array
from simple_pytree import Pytree

from .utils import PhysicalConstants


class AbstractState(Pytree):
    """Abstract state class to define the interface for all states."""


class AbstractRadiationState(AbstractState):
    """Abstract radiation state."""

    net_rad: Array
    """Net surface radiation [W m-2]."""
    in_srad: Array
    """Incoming solar radiation [W m-2]."""
    out_srad: Array
    """Outgoing solar radiation [W m-2]."""
    in_lrad: Array
    """Incoming longwave radiation [W m-2]."""
    out_lrad: Array
    """Outgoing longwave radiation [W m-2]."""


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
    """Abstract atmosphere state."""


R = TypeVar("R", bound=AbstractRadiationState)
L = TypeVar("L", bound=AbstractLandState)
A = TypeVar("A", bound=AbstractAtmosphereState)


class AbstractCoupledState(AbstractState, Generic[A, L, R]):
    """Abstract coupled state, generic over atmosphere, land, and radiation types."""

    atmosphere: A
    land: L
    radiation: R


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel, Generic[R]):
    """Abstract radiation model class to define the interface for all radiation models."""

    tstart: Array
    """Start time of the model."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[A, L, R],
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> R:
        raise NotImplementedError


class AbstractLandModel(AbstractModel, Generic[L]):
    """Abstract land model class to define the interface for all land models."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[A, L, R],
        const: PhysicalConstants,
    ) -> L:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: L, dt: float) -> L:
        raise NotImplementedError


class AbstractAtmosphereModel(AbstractModel, Generic[A]):
    """Abstract atmosphere model class to define the interface for all atmosphere models."""

    @abstractmethod
    def warmup(
        self,
        state: AbstractCoupledState[A, L, R],
        const: PhysicalConstants,
        land: AbstractLandModel[L],
    ) -> AbstractCoupledState[A, L, R]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[A, L, R],
        const: PhysicalConstants,
    ) -> A:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self, state: AbstractCoupledState[A, L, R], t: int, const: PhysicalConstants
    ) -> AbstractCoupledState[A, L, R]:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: A, dt: float) -> A:
        raise NotImplementedError
