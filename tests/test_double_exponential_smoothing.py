"""Tests for Double Exponential Smoothing."""

import numpy as np
import pytest

from intermittent_forecast.base_forecaster import TSArray
from intermittent_forecast.double_exponential_smoothing import (
    DoubleExponentialSmoothing,
)
from intermittent_forecast.error_metrics import (
    ErrorMetricRegistry,
)


@pytest.fixture
def ts_linear() -> TSArray:
    return np.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def ts_random() -> TSArray:
    return np.array([40, 28, 35, 41, 33, 21, 37, 20, 28, 31, 31, 22])


class TestDoubleExponentialSmoothingForecast:
    def test_returns_correct_values_with_alpha_and_beta_set(
        self,
        ts_random: TSArray,
    ) -> None:
        n_obs = len(ts_random)
        model = DoubleExponentialSmoothing().fit(
            ts=ts_random,
            alpha=0.3,
            beta=0.1,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            40.0,
            28.0,
            16.0,
            10.27,
            8.9809,
            6.399103,
            1.429872,
            3.818514,
            0.876008,
            2.029973,
            4.606850,
            7.202458,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [
            6.763310,
            1.884900,
            -2.993510,
            -7.871920,
            -12.7503309,
        ]

        np.testing.assert_allclose(
            forecast_outsample,
            expected_outsample,
            rtol=1e-5,
        )

    def test_calling_forecast_before_fit_raises_error(self) -> None:
        """Test calling the forecast method before fit raises an error."""
        with pytest.raises(
            RuntimeError,
            match="Model has not been fitted yet",
        ):
            DoubleExponentialSmoothing().forecast(start=0, end=1)


class TestDoubleExponentialSmoothingOptimisedForecast:
    @pytest.mark.parametrize(
        "error_metric",
        ErrorMetricRegistry.get_registry().keys(),
    )
    def test_optimised_forecast_error_less_than_non_optimised(
        self,
        ts_random: TSArray,
        error_metric: str,
    ) -> None:
        """Test that an optimised forecast produces minimised error.

        This test checks that when fitting the parameters through optimisation,
        the resulting forecast has a lower error than when using the provided
        parameters.
        """
        ts = ts_random
        len_ts = len(ts)
        forecast_estimated = (
            DoubleExponentialSmoothing()
            .fit(ts=ts, alpha=0.3, beta=0.1)
            .forecast(start=0, end=(len_ts - 1))
        )

        forecast_optimised = (
            DoubleExponentialSmoothing()
            .fit(
                ts=ts,
                optimisation_metric=error_metric,
            )
            .forecast(start=0, end=(len_ts - 1))
        )
        # Get the error metric function from the string
        error_metric_func = ErrorMetricRegistry.get(error_metric)

        # Calculate the error for both forecasts
        err_naive_forecast = error_metric_func(ts, forecast_estimated)
        err_optimised_forecast = error_metric_func(ts, forecast_optimised)

        if err_optimised_forecast > err_naive_forecast:
            err_msg = (
                f"Expected optimised forecast error to be <= default guess. "
                f"Got: {err_optimised_forecast} > {err_naive_forecast}"
            )
            raise ValueError(err_msg)
