"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T_BaseForecaster = TypeVar("T_BaseForecaster", bound="BaseForecaster")
TSArray = npt.NDArray[np.float64]
TSInput = npt.ArrayLike


class BaseForecaster(ABC):
    """Abstract base class for forecasting models.

    Subclasses must implement the following methods:
        - fit(ts): Train the model on a time series.
        - forecast(start, end): Generate forecasts for the specified range.

    This base class defines a consistent interface for time series forecasters,
    allowing for easy wrapping for different approaches such as ADIDA.

    """

    @abstractmethod
    def fit(
        self: T_BaseForecaster,
        ts: TSInput,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> T_BaseForecaster:
        """Fit the model to the time-series."""

    @abstractmethod
    def forecast(
        self,
        start: int,
        end: int,
    ) -> TSArray:
        """Return the forecast."""
