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
    """Base class for forecasting models."""

    @abstractmethod
    def fit(
        self: T_BaseForecaster,
        ts: TSInput,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> T_BaseForecaster:
        """Fit the model to the time-series. Must be implemented."""

    @abstractmethod
    def forecast(
        self,
        start: int,
        end: int,
    ) -> TSArray:
        """Return the forecast."""
