"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

T_BaseForecaster = TypeVar("T_BaseForecaster", bound="BaseForecaster")


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    @abstractmethod
    def fit(
        self: T_BaseForecaster,
        ts: npt.NDArray[np.float64],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> T_BaseForecaster:
        """Fit the model to the time-series. Must be implemented."""

    @abstractmethod
    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Return the forecast."""
