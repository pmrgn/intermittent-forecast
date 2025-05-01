"""Methods for forecasting intermittent time series using ADIDA method."""

from enum import Enum

import numpy as np
import numpy.typing as npt

from intermittent_forecast import utils


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
    def block_aggregation(
        ts: npt.NDArray[np.float64],
        window_size: int,
    ) -> npt.NDArray[np.float64]:
        """Aggregate the time-series using a fixed window."""
        ts_length = len(ts)
        if ts_length < 1 or window_size < 1:
            err_msg = "Time series and window size must be greater than 0."
            raise ValueError(err_msg)

        if ts_length < window_size:
            err_msg = (
                "Time series must be greater than window size for block "
                "aggregation."
            )
            raise ValueError(err_msg)

        # The beginning of the time series is trimmed, else it may introduce a bias.
        trim_size = ts_length % window_size
        ts_trimmed = ts[trim_size:]

        # Ensure aggregated time series is valid
        return utils.validate_array_is_numeric(
            ts_trimmed.reshape((-1, window_size)).sum(axis=1),
        )

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
        cycle_length: int,
    ) -> npt.NDArray[np.float64]:
        """Calculate the distribution for a time series.

        If the time series is not a multiple of the cycle, it will be padded with

        """
        if len(ts) < cycle_length:
            err_msg = "Time series must be greater than cycle size."
            raise ValueError(err_msg)

        # Create an index array that repeats every cycle
        idx = np.arange(len(ts)) % cycle_length

        # Sum the time series for each step in the cycle
        sums = np.bincount(idx, weights=ts)

        # Count the number of occurrences for each step in the cycle, as the
        # time series may not be an exact multiple of the cycle.
        counts = np.bincount(idx)

        # Calculate the average for each step in the cycle
        averages = sums / counts

        # The weights are the proportion for each step in the cycle
        temporal_weights = averages / averages.sum()

        return utils.validate_array_is_numeric(temporal_weights)

    @staticmethod
    def apply_temporal_weights(
        ts: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply the temporal distribution to a time series."""
        # Tile the weights up to the length of the time series
        weights_rpt = utils.validate_array_is_numeric(
            np.tile(weights, len(ts) // len(weights) + 1)[: len(ts)],
        )

        # Apply the weights to the time series, scaling by the number of periods
        return utils.validate_array_is_numeric(ts * weights_rpt * len(weights))
