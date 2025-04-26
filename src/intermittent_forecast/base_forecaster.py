"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    def __init__(self, ts: list[float] | npt.NDArray[np.float64]) -> None:
        """Initialise the forecaster.

        Parameters
        ----------
        ts : list[float] | npt.NDArray[np.float64]
            Time-series to forecast.

        Raises
        ------
        TypeError
            If `ts` is not a list or numpy array.
        ValueError
            If `ts` is not 1-dimensional.
        TypeError
            If `ts` does not contain integers or floats.

        """
        self._ts = self._validate_time_series(ts)

    @property
    def ts(self) -> npt.NDArray[np.float64]:
        """Get the time-series."""
        return self._ts

    @ts.setter
    def ts(self, ts: list[float] | npt.NDArray[np.float64]) -> None:
        """Set the time-series."""
        self._ts = self._validate_time_series(ts)

    def forecast(
        self,
        timeseries_overide: None | npt.NDArray[np.float64] = None,
    ) -> npt.NDArray[np.float64]:
        """Returns the forecast."""
        ts = timeseries_overide if timeseries_overide is not None else self._ts
        return self._generate_forecast(ts)

    @abstractmethod
    def _generate_forecast(
        self,
        ts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Logic to generate the forecast."""

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
