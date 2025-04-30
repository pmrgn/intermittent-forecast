"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    def __init__(self) -> None:
        """Initialise the forecaster."""
        self._ts: npt.NDArray[np.float64] | None = None

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
        self._ts = self._validate_time_series(ts)

    def fit(
        self,
        ts: npt.NDArray[np.float64],
        **kwargs: Any,  # noqa: ANN401
    ) -> BaseForecaster:
        """Fit the model to the time-series."""
        self.set_timeseries(ts)
        self._fit(**kwargs)
        return self

    @abstractmethod
    def _fit(self) -> None:
        """Fit the model to the time-series."""

    @abstractmethod
    def forecast(
        self,
    ) -> npt.NDArray[np.float64]:
        """Returns the forecast."""

    def _validate_time_series(
        self,
        ts: list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Validate the time-series is a 1-dimensional numpy array."""
        if isinstance(ts, list):
            ts = np.array(ts)

        elif not isinstance(ts, np.ndarray):
            err_msg = "Time-series must be a list or numpy array."
            raise TypeError(err_msg)

        if ts.ndim != 1:
            err_msg = "Time-series must be 1-dimensional."
            raise ValueError(err_msg)

        if not (
            np.issubdtype(ts.dtype, np.integer)
            or np.issubdtype(
                ts.dtype,
                np.floating,
            )
        ):
            err_msg = "Time-series must contain integers or floats."
            raise TypeError(err_msg)

        min_length = 2
        if len(ts[ts != 0]) < min_length:
            err_msg = "Time-series needs at least two non-zero values"
            raise ValueError(err_msg)

        return ts

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
