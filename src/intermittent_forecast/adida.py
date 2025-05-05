"""Methods for forecasting intermittent time series using ADIDA method."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.resampler import (
    AggregationMode,
    DisaggregationMode,
    TimeSeriesResampler,
)


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
        self._aggregation_mode = utils.get_enum_from_str(
            mode_str=aggregation_mode,
            enum_class=AggregationMode,
            mode_name="aggregation_mode",
        )
        self._disaggregation_mode = utils.get_enum_from_str(
            mode_str=disaggregation_mode,
            enum_class=DisaggregationMode,
            mode_name="disaggregation_mode",
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
        agg_steps = self._aggregation_period // base_steps
        self._aggregated_forecast = self._aggregated_model.forecast(
            start=0,
            end=len(self._base_ts) + agg_steps,
        )
        forecast = self._disaggregate()
        return forecast[start:end]

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
            _aggregated_ts = TimeSeriesResampler.sliding_aggregation(
                ts=ts,
                window_size=self._aggregation_period,
            )
        elif self._aggregation_mode == AggregationMode.BLOCK:
            _aggregated_ts = TimeSeriesResampler.block_aggregation(
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
            forecast = TimeSeriesResampler.sliding_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        elif self._aggregation_mode == AggregationMode.BLOCK:
            # Block aggregation will return the series to
            forecast = TimeSeriesResampler.block_disaggregation(
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
            self._temporal_weights = (
                TimeSeriesResampler.calculate_temporal_weights(
                    ts=self._base_ts,
                    cycle_length=self._aggregation_period,
                )
            )
            # Apply the temporal weights to the forecast
            ret = TimeSeriesResampler.apply_temporal_weights(
                ts=forecast,
                weights=self._temporal_weights,
            )

        elif self._disaggregation_mode == DisaggregationMode.UNIFORM:
            # Implement uniform disaggregation logic here
            # TODO: Cleanup
            ret = forecast

        return ret
