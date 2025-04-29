"""Methods for forecasting intermittent time series using ADIDA method."""

from enum import Enum

import numpy as np
import numpy.typing as npt


class AggregationMode(Enum):
    """Enum for aggregation modes."""

    BLOCK = "block"
    SLIDING = "sliding"


class DisaggregationMode(Enum):
    """Enum for disaggregation modes."""

    SEASONAL = "seasonal"
    UNIFORM = "uniform"


class TimeSeriesResampler:
    """Helper class for aggregation and disaggregation of time series."""

    @staticmethod
    def sliding_aggregation(
        ts: npt.NDArray[np.float64],
        window_size: int,
    ) -> npt.NDArray[np.float64]:
        """Aggregate the time-series using a sliding window."""
        return np.convolve(a=ts, v=np.ones(window_size), mode="valid")

    @staticmethod
    def sliding_disaggregation(
        ts: npt.NDArray[np.float64],
        window_size: int,
    ) -> npt.NDArray[np.float64]:
        """Disaggregate the time-series using a sliding window."""
        return np.concatenate((np.full(window_size - 1, np.nan), ts))

    @staticmethod
    def block_aggregation(
        ts: npt.NDArray[np.float64],
        window_size: int,
    ) -> npt.NDArray[np.float64]:
        """Aggregate the time-series using a fixed window."""
        trim = len(ts) % window_size
        ts_trim = ts[trim:]
        return ts_trim.reshape((-1, window_size)).sum(axis=1)

    @staticmethod
    def block_disaggregation(
        ts: npt.NDArray[np.float64],
        window_size: int,
    ) -> npt.NDArray[np.float64]:
        """Disaggregate the time-series using a fixed size."""
        return np.repeat(ts, window_size) / window_size

    @staticmethod
    def calculate_temporal_weights(
        ts: npt.NDArray[np.float64],
        cycle: int,
    ) -> npt.NDArray[np.float64]:
        """Calculate the distribution for a time series.

        If the time series is not a multiple of the cycle, it will be padded with

        """
        # TODO: Validation - cycle must be > ts
        # TODO: Clean up logic
        idx = np.arange(len(ts)) % cycle
        sums = np.bincount(idx, weights=ts)
        counts = np.bincount(idx)

        # Either average per weekday:
        averages = sums / counts

        dist = averages / averages.sum()
        return dist

    @staticmethod
    def apply_temporal_weights(
        ts: npt.NDArray[np.float64],
        temporal_weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply the temporal distribution to a time series."""
        # TODO: Clean up logic
        s = len(temporal_weights)
        pad = s - (len(ts) % s)
        ts_padded = np.concatenate((ts, [np.nan] * pad))
        res = (ts_padded.reshape((-1, s)) * temporal_weights).flatten()
        return res[:-pad]
