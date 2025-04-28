"""Methods for forecasting intermittent time series using ADIDA method."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt

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
            raise TypeError(
                "Not a recognised forecasting model.",
            )
        self._model = model
        self._aggregated_model = deepcopy(model)
        self._aggregation_period = aggregation_period
        self.set_aggregation_mode(aggregation_mode)
        self.set_disaggregation_mode(disaggregation_mode)

    def set_aggregation_mode(self, mode: AggregationMode | str) -> None:
        """Set the aggregation mode."""
        if isinstance(mode, str):
            mode = self._get_aggregation_mode_from_str(mode)
        if not isinstance(mode, AggregationMode):
            err_msg = f"Invalid aggregation mode: {mode}"
            raise TypeError(err_msg)
        self._aggregation_mode = mode

    def set_disaggregation_mode(self, mode: DisaggregationMode | str) -> None:
        """Set the disaggregation mode."""
        if isinstance(mode, str):
            mode = self._get_disaggregation_mode_from_str(mode)
        if not isinstance(mode, DisaggregationMode):
            err_msg = f"Invalid disaggregation mode: {mode}"
            raise TypeError(err_msg)
        self._disaggregation_mode = mode

    @staticmethod
    def _get_aggregation_mode_from_str(mode_str: str) -> AggregationMode:
        try:
            return AggregationMode(mode_str.lower())
        except ValueError:
            err_msg = (
                f"Unknown aggregation mode: '{mode_str}'. ",
                f"Expected one of: {[m.value for m in AggregationMode]}",
            )
            raise ValueError(err_msg) from None

    @staticmethod
    def _get_disaggregation_mode_from_str(mode_str: str) -> DisaggregationMode:
        try:
            return DisaggregationMode(mode_str.lower())
        except ValueError:
            err_msg = (
                f"Unknown disaggregation mode: '{mode_str}'.",
                f"Expected one of: {[m.value for m in DisaggregationMode]}",
            )
            raise ValueError(err_msg) from None

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

        else:
            # TODO: May not need this if validated in the class init.
            err_msg = (
                f"Unknown aggregation mode: {self._aggregation_mode}.",
                f"Valid options are: {[m.value for m in AggregationMode]}",
            )
            raise ValueError(err_msg)

        return _aggregated_ts

    def _disaggregate(
        self,
        mode: str = DisaggregationMode.UNIFORM.value,
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
        # disaggregate based on the mode of aggregation
        if self._aggregation_mode == AggregationMode.SLIDING:
            forecast = TimeSeriesResampler.sliding_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        elif self._aggregation_mode == AggregationMode.BLOCK:
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
            ret = TimeSeriesResampler.block_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )
        else:
            # TODO: May not be needed if validated in the class init.
            raise ValueError(
                f"Unknown disaggregation mode: {mode}. "
                f"Valid options are: {[m.value for m in DisaggregationMode]}.",
            )
        return ret
