"""Methods for forecasting intermittent time series using ADIDA method."""

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
        self._base_ts = model.ts
        self._aggregation_period = aggregation_period
        self._aggregation_mode = aggregation_mode
        self._disaggregation_mode = disaggregation_mode

    def forecast(self) -> npt.NDArray[np.float64]:
        """Forecast the time series using the ADIDA method.

        Returns
        -------
        np.ndarray
            Forecasted values.

        """
        # Aggregate the time series
        self._aggregated_ts = self._aggregate()
        self._aggregated_forecast = self._model.forecast(
            timeseries_overide=self._aggregated_ts,
        )
        result = self._disaggregate()
        return result

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
        if self._aggregation_mode == AggregationMode.SLIDING.value:
            _aggregated_ts = TimeSeriesResampler.sliding_aggregation(
                ts=self._model.ts,
                window_size=self._aggregation_period,
            )
        elif self._aggregation_mode == AggregationMode.BLOCK.value:
            _aggregated_ts = TimeSeriesResampler.block_aggregation(
                ts=self._model.ts,
                window_size=self._aggregation_period,
            )

        else:
            # TODO: May not need this if validated in the class init.
            raise ValueError(
                f"Unknown aggregation mode: {self._aggregation_mode}. "
                f"Valid options are: {[m.value for m in AggregationMode]}.",
            )

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
        if self._aggregation_mode == AggregationMode.SLIDING.value:
            forecast = TimeSeriesResampler.sliding_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        elif self._aggregation_mode == AggregationMode.BLOCK.value:
            forecast = TimeSeriesResampler.block_disaggregation(
                ts=self._aggregated_forecast,
                window_size=self._aggregation_period,
            )

        if self._disaggregation_mode == DisaggregationMode.SEASONAL.value:
            # Compute the temporal weights of the original time-series
            temporal_weights = TimeSeriesResampler.calculate_temporal_weights(
                ts=self._base_ts,
                cycle=self._aggregation_period,
            )
            # Apply the temporal weights to the forecast
            ret = TimeSeriesResampler.apply_temporal_weights(
                ts=forecast,
                temporal_weights=temporal_weights,
            )

        elif self._disaggregation_mode == DisaggregationMode.UNIFORM.value:
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
