"""Methods for forecasting intermittent time series using ADIDA method."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import (
    BaseForecaster,
    T_BaseForecaster,
)


class AggregationMode(Enum):
    """Enum for aggregation modes."""

    BLOCK = "block"
    SLIDING = "sliding"


class DisaggregationMode(Enum):
    """Enum for disaggregation modes."""

    SEASONAL = "seasonal"
    UNIFORM = "uniform"


class ADIDAConfig(NamedTuple):
    aggregation_period: int
    aggregation_mode: AggregationMode
    disaggregation_mode: DisaggregationMode


class ADIDAFittedResult(NamedTuple):
    aggregated_model: BaseForecaster
    temporal_weights: npt.NDArray[np.float64]
    ts_base: npt.NDArray[np.float64]


class ADIDA:
    """Aggregate-disaggregate Intermittent Demand Approach."""

    def __init__(
        self,
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
        aggregation_period = utils.validate_non_negative_integer(
            value=aggregation_period,
            name="aggregation_period",
        )

        aggregation_mode_member = utils.get_enum_member_from_str(
            member_str=aggregation_mode,
            enum_class=AggregationMode,
            member_name="aggregation_mode",
        )

        disaggregation_mode_member = utils.get_enum_member_from_str(
            member_str=disaggregation_mode,
            enum_class=DisaggregationMode,
            member_name="disaggregation_mode",
        )

        self._config = ADIDAConfig(
            aggregation_period=aggregation_period,
            aggregation_mode=aggregation_mode_member,
            disaggregation_mode=disaggregation_mode_member,
        )

        self._adida_fitted_result: ADIDAFittedResult | None = None

    def fit(
        self,
        model: T_BaseForecaster,
        ts: npt.NDArray[np.float64],
        **kwargs: Any,  # noqa: ANN401
    ) -> ADIDA:
        """Fit the model."""
        if not isinstance(model, BaseForecaster):
            err_msg = (
                "ADIDA model requires a forecasting model.",
                f"Got: {type(model)}",
            )
            raise TypeError(err_msg)

        # Aggregate the time series
        ts = utils.validate_time_series(ts)
        match self._config.aggregation_mode:
            case AggregationMode.SLIDING:
                aggregated_ts = self.sliding_aggregation(
                    ts=ts,
                    window_size=self._config.aggregation_period,
                )
            case AggregationMode.BLOCK:
                aggregated_ts = self.block_aggregation(
                    ts=ts,
                    window_size=self._config.aggregation_period,
                )
        aggregated_ts = utils.validate_time_series(aggregated_ts)

        # Create a model by copying the original model and fitting it using
        # the aggregated time series.
        aggregated_model = deepcopy(model)
        aggregated_model.fit(
            ts=aggregated_ts,
            **kwargs,
        )

        # If using a seasonal disaggregation method, calculate the seasonal
        # weights, which will be used later to disaggregate the forecast.
        temporal_weights = np.array([])
        if self._config.disaggregation_mode == DisaggregationMode.SEASONAL:
            temporal_weights = self.calculate_temporal_weights(
                ts=ts,
                cycle_length=self._config.aggregation_period,
            )

        # Cache results
        self._adida_fitted_result = ADIDAFittedResult(
            aggregated_model=aggregated_model,
            ts_base=ts,
            temporal_weights=temporal_weights,
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

        forecast = self._disaggregate(
            config=self._config,
            fitted_result=self.get_fitted_model_result(),
            end=end,
        )

        return forecast[start : end + 1]

    def get_fitted_model_result(
        self,
    ) -> ADIDAFittedResult:
        """Get the fitted model."""
        if not self._adida_fitted_result:
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise RuntimeError(err_msg)

        return self._adida_fitted_result

    @staticmethod
    def _disaggregate(
        config: ADIDAConfig,
        fitted_result: ADIDAFittedResult,
        end: int,
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
        # Calculate the number of steps to forecast for the aggregated model.
        base_steps = end - len(fitted_result.ts_base)
        agg_steps = (base_steps // config.aggregation_period) + 1
        aggregated_forecast = fitted_result.aggregated_model.forecast(
            start=0,
            end=len(fitted_result.ts_base) + agg_steps,
        )

        # Prepare for disaggregation, which depends on how the forecast was
        # aggregated.
        match config.aggregation_mode:
            case AggregationMode.SLIDING:
                forecast_disaggregated = ADIDA.sliding_disaggregation(
                    ts=aggregated_forecast,
                    window_size=config.aggregation_period,
                )

            case AggregationMode.BLOCK:
                # Block aggregation will return the series to
                forecast_disaggregated = ADIDA.block_disaggregation(
                    aggregated_ts=aggregated_forecast,
                    window_size=config.aggregation_period,
                    base_ts_length=len(fitted_result.ts_base),
                )

        match config.disaggregation_mode:
            case DisaggregationMode.SEASONAL:
                # Apply the temporal weights to the forecast
                forecast_disaggregated = ADIDA.apply_temporal_weights(
                    ts=forecast_disaggregated,
                    weights=fitted_result.temporal_weights,
                )

            case DisaggregationMode.UNIFORM:
                # A uniform disaggregation requires no further processing.
                pass

        return forecast_disaggregated

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
        return np.concatenate(
            (np.full(window_size - 1, np.nan), ts / window_size),
        )

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
