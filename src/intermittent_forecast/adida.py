"""Methods for forecasting intermittent time series using ADIDA method."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, NamedTuple

import numpy as np

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import (
    BaseForecaster,
    T_BaseForecaster,
    TSArray,
    TSInput,
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
    """Configuration for ADIDA model initilisation."""

    aggregation_period: int
    aggregation_mode: AggregationMode
    disaggregation_mode: DisaggregationMode


class ADIDAFittedResult(NamedTuple):
    """Fitted result for ADIDA model."""

    aggregated_model: BaseForecaster
    temporal_weights: TSArray
    ts_base: TSArray


class ADIDA:
    """Aggregate-Disaggregate Intermittent Demand Approach (ADIDA).

    Args:
        aggregation_period (int): Number of time periods to aggregate.
        aggregation_mode (str): The aggregation mode, either "sliding" or
            "block".
        disaggregation_mode (str): The disaggregation mode, either
            "seasonal" or"uniform".

    Methods:
        fit: Fit the model. forecast: Forecast the time series using the fitted
            parameters.

    Examples:
    ```
    >>> model = ADIDA(
    >>>     aggregation_period=12,
    >>>     aggregation_mode="sliding"
    >>>     disaggregation_mode="seasonal",
    >>> )
    ```

    """

    def __init__(  # noqa: D107
        self,
        aggregation_period: int,
        aggregation_mode: str,
        disaggregation_mode: str,
    ) -> None:
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
        ts: TSInput,
        **kwargs: Any,  # noqa: ANN401
    ) -> ADIDA:
        """Aggregate the time series and fit using the forecasting model.

        Args:
            model (T_BaseForecaster): Forecasting model class to use on the
                aggregated time series. Examples include CRO, SBA, TSB,
                and TripleExponentialSmoothing.
            ts (ArrayLike): Time series to fit.
            **kwargs (Any): Additional keyword arguments to pass to the
                forecasting model. Refer to the documentation for the `fit`
                method of the forecasting model you are using for valid
                keyword arguments.

        """
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
    ) -> TSArray:
        """Forecast the time series using the ADIDA method.

        Args:
            start (int): Start index of the forecast (inclusive).
            end (int): End index of the forecast (inclusive).

        Returns:
            np.ndarray: Forecasted values.

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
        if not self._adida_fitted_result or not isinstance(
            self._adida_fitted_result,
            ADIDAFittedResult,
        ):
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
    ) -> TSArray:
        """Disaggregate the forecasted values."""
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
        ts: TSArray,
        window_size: int,
    ) -> TSArray:
        """Aggregate the time-series using a sliding window."""
        return np.convolve(a=ts, v=np.ones(window_size), mode="valid")

    @staticmethod
    def sliding_disaggregation(
        ts: TSArray,
        window_size: int,
    ) -> TSArray:
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
        ts: TSArray,
        window_size: int,
    ) -> TSArray:
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

        # The beginning of the time series is trimmed to allow for an exact
        # number of windows, else it can introduce a bias.
        ts_trimmed = ts[(ts_length % window_size) :]

        # Ensure aggregated time series is valid
        return utils.validate_array_is_numeric(
            ts_trimmed.reshape((-1, window_size)).sum(axis=1),
        )

    @staticmethod
    def block_disaggregation(
        aggregated_ts: TSArray,
        window_size: int,
        base_ts_length: int,
    ) -> TSArray:
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
        ts: TSArray,
        cycle_length: int,
    ) -> TSArray:
        """Calculate the distribution for a time series.

        This returns the average proportion of demand for each step in the
        cycle, i.e. the array of weights will sum to 1.

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

        # The weights are then the average proportion for each step in the
        # cycle.
        temporal_weights = averages / averages.sum()

        return utils.validate_array_is_numeric(temporal_weights)

    @staticmethod
    def apply_temporal_weights(
        ts: TSArray,
        weights: TSArray,
    ) -> TSArray:
        """Apply the temporal weights to a time series."""
        # Tile the weights up to the length of the time series
        weights_rpt = utils.validate_array_is_numeric(
            np.tile(weights, len(ts) // len(weights) + 1)[: len(ts)],
        )

        # Apply the weights to the time series, scaling by the number of periods
        return utils.validate_array_is_numeric(ts * weights_rpt * len(weights))
