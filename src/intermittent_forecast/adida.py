"""Methods for forecasting intermittent time series using ADIDA method."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import BaseForecaster

T_BaseForecaster = TypeVar("T_BaseForecaster", bound=BaseForecaster)


class AggregationMode(Enum):
    """Enum for aggregation modes."""

    BLOCK = "block"
    SLIDING = "sliding"


class DisaggregationMode(Enum):
    """Enum for disaggregation modes."""

    SEASONAL = "seasonal"
    UNIFORM = "uniform"


class ADIDA:
    """Aggregate-disaggregate Intermittent Demand Approach."""

    def __init__(
        self,
        model: BaseForecaster,
        aggregation_period: int,
        aggregation_mode: str,
        disaggregation_mode: str,
    ) -> None:
        """Initialise the ADIDA model.

        Parameters
        ----------
        model : BaseForecaster
            Forecasting model to use for aggregation.

        """
        if not isinstance(model, BaseForecaster):
            err_msg = (
                "ADIDA model requires a forecasting model.",
                f"Got: {type(model)}",
            )
            raise TypeError(err_msg)
        self._base_ts: npt.NDArray[np.float64] | None = None
        self._aggregated_model = deepcopy(model)
        self._aggregation_period = aggregation_period
        self._aggregation_mode = utils.get_enum_member_from_str(
            member_str=aggregation_mode,
            enum_class=AggregationMode,
            member_name="aggregation_mode",
        )
        self._disaggregation_mode = utils.get_enum_member_from_str(
            member_str=disaggregation_mode,
            enum_class=DisaggregationMode,
            member_name="disaggregation_mode",
        )
        self._temporal_weights: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        ts: npt.NDArray[np.float64],
        **kwargs: Any,  # noqa: ANN401
    ) -> ADIDA:
        """Fit the model."""
        # Cache time series
        self._base_ts = ts

        # Aggregate the time series
        aggregated_ts = self._aggregate(ts)

        self._aggregated_model.fit(
            ts=aggregated_ts,
            **kwargs,
        )

        return self

    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Forecast the time series using the ADIDA method.

        Returns
        -------
        np.ndarray
            Forecasted values.

        """
        start = utils.validate_non_negative_integer(start, name="start")
        end = utils.validate_positive_integer(end, name="end")
        # Calculate the number of steps to forecast for the aggregated model.
        base_steps = end - len(self._base_ts)
        agg_steps = (base_steps // self._aggregation_period) + 1
        self._aggregated_forecast = self._aggregated_model.forecast(
            start=0,
            end=len(self._base_ts) + agg_steps,
        )
        forecast = self._disaggregate()
        return forecast[start : end + 1]

    def _aggregate(
        self,
        ts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Aggregate the time-series using a specified window size.

        Parameters
        ----------
        ts : npt.NDArray[np.float64]
            Time-series to aggregate.

        """
        if self._aggregation_mode == AggregationMode.SLIDING:
            _aggregated_ts = self.sliding_aggregation(
                ts=ts,
                window_size=self._aggregation_period,
            )
        elif self._aggregation_mode == AggregationMode.BLOCK:
            _aggregated_ts = self.block_aggregation(
                ts=ts,
                window_size=self._aggregation_period,
            )

        return utils.validate_time_series(_aggregated_ts)

    def _disaggregate(
        self,
    ) -> npt.NDArray[np.float64]:
        """Disaggregate the forecasted values.

        Parameters
        ----------
        mode : str, optional
            Disaggregation mode, by default DisaggregationMode.UNIFORM.value.

        Returns
        -------
        np.ndarray
            Disaggregated forecasted values.

        """
        # Prepare for disaggregation, which depends on how the forecast was
        # aggregated.
        if self._aggregation_mode == AggregationMode.SLIDING:
            forecast = self.sliding_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        elif self._aggregation_mode == AggregationMode.BLOCK:
            # Block aggregation will return the series to
            forecast = self.block_disaggregation(
                aggregated_ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
                base_ts_length=len(self._base_ts),
            )

        else:
            err_msg = (
                "ADIDA disaggregation only supports sliding or block "
                "aggregation."
            )
            raise ValueError(err_msg)

        if self._disaggregation_mode == DisaggregationMode.SEASONAL:
            self._temporal_weights = self.calculate_temporal_weights(
                ts=self._base_ts,
                cycle_length=self._aggregation_period,
            )
            # Apply the temporal weights to the forecast
            ret = self.apply_temporal_weights(
                ts=forecast,
                weights=self._temporal_weights,
            )

        elif self._disaggregation_mode == DisaggregationMode.UNIFORM:
            # Implement uniform disaggregation logic here
            # TODO: Cleanup
            ret = forecast

        return ret

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
        window_size = utils.validate_positive_integer(
            value=window_size,
            name="window_size",
        )
        # Pad the beginning of the forecast with NaN values, to match the
        # length of the original time series. A sliding aggregation results
        # in a forecast that is shorter than the original time series by
        # (window_size - 1) values.
        ts_disaggregated_padded = np.concatenate(
            (np.full(window_size - 1, np.nan), ts / window_size),
        )
        return ts_disaggregated_padded

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
        aggregated_ts: npt.NDArray[np.float64],
        window_size: int,
        base_ts_length: int,
    ) -> npt.NDArray[np.float64]:
        """Disaggregate the time-series using a fixed size."""
        # Repeat the aggregated time series to match the length of the original
        # time series.
        ret = np.repeat(aggregated_ts, window_size) / window_size
        # Pad the beginning of the forecast with NaN values, to match the
        # length of the original time series length.
        pad_length = base_ts_length - len(ret)
        if pad_length > 0:
            ret = np.concatenate(
                (np.full(pad_length, np.nan), ret),
            )
        return ret

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
