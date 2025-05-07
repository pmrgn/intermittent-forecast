"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from intermittent_forecast import utils

if TYPE_CHECKING:
    import numpy.typing as npt


class FittedParams(TypedDict):
    """TypedDict for fitted parameters."""


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    def __init__(self) -> None:
        """Initialise the forecaster."""
        self._ts: npt.NDArray[np.float64] | None = None
        self._fitted_params: FittedParams | None = None

    def get_timeseries(self) -> npt.NDArray[np.float64]:
        """Get the time-series."""
        if self._ts is None:
            err_msg = (
                "Time-series not set. Use `set_timeseries` method or call "
                "the `fit` method to with a time-series."
            )
            raise ValueError(err_msg)
        return self._ts

    def set_timeseries(self, ts: list[float] | npt.NDArray[np.float64]) -> None:
        """Set the time-series."""
        self._ts = utils.validate_time_series(ts)

    def fit(
        self,
        ts: npt.NDArray[np.float64] | list[float | int],
        **kwargs: Any,  # noqa: ANN401
    ) -> BaseForecaster:
        """Fit the model to the time-series."""
        self.set_timeseries(ts)
        self._fit(**kwargs)
        return self

    def get_fitted_params(self) -> FittedParams:
        """Get the fitted parameters."""
        if not self._fitted_params:
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise ValueError(err_msg)
        return self._fitted_params

    @abstractmethod
    def _fit(self) -> None:
        """Fit the model to the time-series and cache results."""

    @abstractmethod
    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Returns the forecast."""

    @staticmethod
    def _validate_float_within_inclusive_bounds(
        name: str,
        value: float,
        min_value: float = float("-inf"),
        max_value: float = float("inf"),
    ) -> float:
        """Validate a numeric parameter is within inclusive bounds."""
        if value is None:
            err_msg = (
                f"Parameter '{name}' must be provided and cannot be None.",
            )
            raise ValueError(err_msg)
        if not (min_value <= value <= max_value):
            err_msg = (
                f"Parameter '{name}'={value} is out of bounds. ",
                f"Must be between {min_value} and {max_value}.",
            )
            raise ValueError(err_msg)
        return value
