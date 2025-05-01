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
        self._temporal_weights: npt.NDArray | None = None

    def fit(
        self,
        ts: npt.NDArray[np.float64],
        **kwargs: Any,  # noqa: ANN401
    ) -> ADIDA:
        """Fit the model."""
        # TODO: Validate ts? Validated in BaseForecaster?
        # Aggregate the time series
        aggregated_ts = self._aggregate(ts)

        # Set the aggregated time series to the model
        self._aggregated_model.set_timeseries(aggregated_ts)

        self._aggregated_model.fit(
            ts=aggregated_ts,
            **kwargs,
        )

        # If required, caclulate the temporal weights required for seasonal
        # disaggregation
        if self._disaggregation_mode == DisaggregationMode.SEASONAL:
            self._temporal_weights = (
                TimeSeriesResampler.calculate_temporal_weights(
                    ts=ts,
                    cycle_length=self._aggregation_period,
                )
            )
        return self

    def forecast(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> npt.NDArray[np.float64]:
        """Forecast the time series using the ADIDA method.

        Returns
        -------
        np.ndarray
            Forecasted values.

        """
        self._aggregated_forecast = self._aggregated_model.forecast(
            **kwargs,
        )
        return self._disaggregate()

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

        return _aggregated_ts

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
        # TODO: Fix logic so it's clearer.
        # Prepare for disaggregation, which depends on how the forecast was
        # aggregated.
        if self._aggregation_mode == AggregationMode.SLIDING:
            # Prepare the forecast for disaggregation
            # forecast = TimeSeriesResampler.sliding_disaggregation(
            #     ts=self._aggregated_forecast,
            #     window_size=self._aggregation_period,
            # )
            forecast = np.concatenate(
                (
                    np.full(self._aggregation_period - 1, np.nan),
                    self._aggregated_forecast,
                ),
            )

        else:
            forecast = TimeSeriesResampler.block_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        if self._disaggregation_mode == DisaggregationMode.SEASONAL:
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
