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
        self._model = model
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

    def fit(self) -> None:
        """Fit the model."""
        self._aggregated_model.fit()

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
        # Aggregate the time series
        self._aggregated_model.ts = self._aggregate()

        self._aggregated_forecast = self._aggregated_model.forecast(
            **kwargs,
        )
        return self._disaggregate()

    def _aggregate(
        self,
    ) -> npt.NDArray[np.float64]:
        """Aggregate the time-series using a specified window size.

        Parameters
        ----------
        size : int, optional
            Window size for aggregation, by default 1.
        mode : str, optional
            Aggregation mode, by default AggregationMode.BLOCK.value.

        """
        if self._aggregation_mode == AggregationMode.SLIDING:
            _aggregated_ts = TimeSeriesResampler.sliding_aggregation(
                ts=self._model.ts,
                window_size=self._aggregation_period,
            )
        elif self._aggregation_mode == AggregationMode.BLOCK:
            _aggregated_ts = TimeSeriesResampler.block_aggregation(
                ts=self._model.ts,
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
            # Compute the temporal weights of the original time-series
            temporal_weights = TimeSeriesResampler.calculate_temporal_weights(
                ts=self._model.ts,
                cycle=self._aggregation_period,
            )
            # Apply the temporal weights to the forecast
            ret = TimeSeriesResampler.apply_temporal_weights(
                ts=forecast,
                temporal_weights=temporal_weights,
            )

        elif self._disaggregation_mode == DisaggregationMode.UNIFORM:
            # Implement uniform disaggregation logic here
            # TODO: Cleanup
            ret = forecast

        return ret
