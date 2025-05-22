"""The Aggregate-Disaggregate Intermittent Demand Approach (ADIDA)."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from intermittent_forecast.core import utils
from intermittent_forecast.forecasters._base_forecaster import (
    T_BaseForecaster,
    _BaseForecaster,
)

if TYPE_CHECKING:
    from intermittent_forecast.core._types import TSArray, TSInput


class _AggregationMode(Enum):
    """Enum for aggregation modes."""

    BLOCK = "block"
    SLIDING = "sliding"


class _DisaggregationMode(Enum):
    """Enum for disaggregation modes."""

    SEASONAL = "seasonal"
    UNIFORM = "uniform"


class _ADIDAConfig(NamedTuple):
    """Configuration for ADIDA model initilisation."""

    aggregation_period: int
    aggregation_mode: _AggregationMode
    disaggregation_mode: _DisaggregationMode


class _ADIDAFittedResult(NamedTuple):
    """Fitted result for ADIDA model."""

    aggregated_model: _BaseForecaster
    temporal_weights: TSArray
    ts_base: TSArray


class ADIDA:
    """Aggregate-Disaggregate Intermittent Demand Approach (ADIDA).

    ADIDA is a forecasting methodology designed for handling intermittent time
    series. The approach helps improve forecast accuracy by transforming the
    problem into a more stable one via temporal aggregation.

    The method involves aggregating a high-frequency time series (e.g., daily
    observations) into a lower-frequency series (e.g., weekly) as a means to
    reduce variability. This allows for a range of forecasting models to be
    applied to the aggregated series, e.g. Exponential Smoothing. Once a model
    has been fit to the aggregated series, the disaggregation process is
    performed to return it to the original time series.

    Args:
        aggregation_period (int): Number of time periods to aggregate. E.g,
            aperiod of 7 would be used to aggregate a daily time series into a
            weekly time series.
        aggregation_mode (str): The aggregation mode, either "sliding" or
            "block". A sliding window will aggregate the time series by moving
            the window one time period at a time. A block window will aggregate
            the time series by moving the window one block at a time. the time
            series by moving the window one block at a time.
        disaggregation_mode (str): The disaggregation mode, either
            "seasonal" or"uniform". A seasonal disaggregation will disaggregate
            the time series by calculating the proportion of demand for each
            step in the cycle. A uniform disaggregation will disaggregate the
            time series by evenly distributing the demand across the cycle.

    Example:
        >>> # Example intermittent time series.
        >>> ts = [
        ...     3, 0, 0, 4, 0, 0, 0,
        ...     1, 0, 5, 1, 1, 0, 0,
        ...     0, 0, 0, 8, 3, 0, 1,
        ...     0, 1, 0, 4, 3, 0, 0,
        ... ]

        >>> # Initialise ADIDA model.
        >>> from intermittent_forecast.aggregators import ADIDA
        >>> adida = ADIDA(
        ...     aggregation_period=7,
        ...     aggregation_mode="block",
        ...     disaggregation_mode="seasonal",
        ... )

        >>> # Import a forecasting model to use on the aggregated series.
        >>> from intermittent_forecast import forecasters

        >>> # Fit using ADIDA, passing in an instance of the forecasting model.
        >>> # Any valid keyword arguments used by the model can be passed in,
        >>> # e.g. alpha for SimpleExponentialSmoothing.
        >>> adida = adida.fit(
        ...     model=forecasters.SimpleExponentialSmoothing(),
        ...     ts=ts,
        ...     alpha=0.3,
        ... )

        >>> # Forecast the next 7 periods.
        >>> adida.forecast(start=len(ts), end=len(ts)+7)
        array([0.97108571, 0.24277143, 1.21385714, 4.12711429, 1.6994    ,
               0.        , 0.24277143, 0.97108571])

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
            enum_class=_AggregationMode,
            member_name="aggregation_mode",
        )

        disaggregation_mode_member = utils.get_enum_member_from_str(
            member_str=disaggregation_mode,
            enum_class=_DisaggregationMode,
            member_name="disaggregation_mode",
        )

        self._config = _ADIDAConfig(
            aggregation_period=aggregation_period,
            aggregation_mode=aggregation_mode_member,
            disaggregation_mode=disaggregation_mode_member,
        )

        self._adida_fitted_result: _ADIDAFittedResult | None = None

    def fit(
        self,
        model: T_BaseForecaster,
        ts: TSInput,
        **kwargs: Any,  # noqa: ANN401
    ) -> ADIDA:
        """Aggregate the time series and fit using the forecasting model.

        Args:
            model (T_BaseForecaster): Forecasting model to use on the
                aggregated time series, which can be any of the BaseForecaster
                class instances, e.g. Croston, SimpleExponentialSmoothing.
            ts (ArrayLike): Time series to fit.
            **kwargs (Any): Additional keyword arguments to pass to the
                forecasting model. Refer to the documentation for the `fit`
                method of the forecasting model you are using for valid keyword
                arguments.

        Returns:
            self (ADIDA): Fitted model instance.

        """
        if not isinstance(model, _BaseForecaster):
            err_msg = (
                "ADIDA model requires a forecasting model.",
                f"Got: {type(model)}",
            )
            raise TypeError(err_msg)

        # Aggregate the time series
        ts = utils.validate_time_series(ts)
        match self._config.aggregation_mode:
            case _AggregationMode.SLIDING:
                aggregated_ts = self._sliding_aggregation(
                    ts=ts,
                    window_size=self._config.aggregation_period,
                )
            case _AggregationMode.BLOCK:
                aggregated_ts = self._block_aggregation(
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
        if self._config.disaggregation_mode == _DisaggregationMode.SEASONAL:
            temporal_weights = self._calculate_temporal_weights(
                ts=ts,
                cycle_length=self._config.aggregation_period,
            )

        # Cache results
        self._adida_fitted_result = _ADIDAFittedResult(
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
            fitted_result=self._get_fit_result_if_found(),
            end=end,
        )

        return forecast[start : end + 1]

    def get_fit_result(self) -> dict[str, Any]:
        """Return the a dictionary of results if model has been fit."""
        return self._get_fit_result_if_found()._asdict()

    def _get_fit_result_if_found(
        self,
    ) -> _ADIDAFittedResult:
        """Private method for geting the fitted model."""
        if not self._adida_fitted_result or not isinstance(
            self._adida_fitted_result,
            _ADIDAFittedResult,
        ):
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise RuntimeError(err_msg)

        return self._adida_fitted_result

    @staticmethod
    def _disaggregate(
        config: _ADIDAConfig,
        fitted_result: _ADIDAFittedResult,
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
            case _AggregationMode.SLIDING:
                forecast_disaggregated = ADIDA._sliding_disaggregation(
                    ts=aggregated_forecast,
                    window_size=config.aggregation_period,
                )

            case _AggregationMode.BLOCK:
                # Block aggregation will return the series to
                forecast_disaggregated = ADIDA._block_disaggregation(
                    aggregated_ts=aggregated_forecast,
                    window_size=config.aggregation_period,
                    base_ts_length=len(fitted_result.ts_base),
                )

        match config.disaggregation_mode:
            case _DisaggregationMode.SEASONAL:
                # Apply the temporal weights to the forecast
                forecast_disaggregated = ADIDA._apply_temporal_weights(
                    ts=forecast_disaggregated,
                    weights=fitted_result.temporal_weights,
                )

            case _DisaggregationMode.UNIFORM:
                # A uniform disaggregation requires no further processing.
                pass

        return forecast_disaggregated

    @staticmethod
    def _sliding_aggregation(
        ts: TSArray,
        window_size: int,
    ) -> TSArray:
        """Aggregate the time-series using a sliding window."""
        return np.convolve(a=ts, v=np.ones(window_size), mode="valid")

    @staticmethod
    def _sliding_disaggregation(
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
    def _block_aggregation(
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
    def _block_disaggregation(
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
    def _calculate_temporal_weights(
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
    def _apply_temporal_weights(
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
