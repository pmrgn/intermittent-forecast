"""Tests for Simple Exponential Smoothing."""

import numpy as np
import pytest

from intermittent_forecast.core._types import TSArray
from intermittent_forecast.core.error_metrics import (
    ErrorMetricRegistry,
)
from intermittent_forecast.forecasters import SimpleExponentialSmoothing


@pytest.fixture
def ts_linear() -> TSArray:
    return np.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def ts_random() -> TSArray:
    return np.array([40, 28, 35, 41, 33, 21, 37, 20, 28, 31, 31, 22])


class TestSimpleExponentialSmoothingFit:
    def test_alpha_optimises_to_correct_value(
        self,
        ts_linear: TSArray,
    ) -> None:
        ses = SimpleExponentialSmoothing().fit(ts=ts_linear)
        alpha_optimised = ses.get_fit_result()["alpha"]
        if alpha_optimised != 1:
            err_msg = f"Expected alpha to be 1. Got: {alpha_optimised}"
            raise ValueError(err_msg)


class TestSimpleExponentialSmoothingForecast:
    def test_returns_correct_values_with_alpha_set_to_one(
        self,
        ts_linear: TSArray,
    ) -> None:
        n_obs = len(ts_linear)
        model = SimpleExponentialSmoothing().fit(
            ts=ts_linear,
            alpha=1,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [1, 1, 2, 3, 4, 5]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [6, 6, 6, 6, 6]

        np.testing.assert_allclose(
            forecast_outsample,
            expected_outsample,
            rtol=1e-5,
        )

    def test_returns_correct_values_with_alpha_set_to_float(
        self,
        ts_random: TSArray,
    ) -> None:
        n_obs = len(ts_random)
        model = SimpleExponentialSmoothing().fit(
            ts=ts_random,
            alpha=0.3,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            40.0,
            40.0,
            36.4,
            35.98,
            37.486,
            36.1402,
            31.59814,
            33.218698,
            29.2530886,
            28.8771620,
            29.5140134,
            29.9598093,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 2)
        expected_outsample = [27.571866, 27.571866, 27.571866]

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
            SimpleExponentialSmoothing().forecast(start=0, end=1)


class TestSimpleExponentialSmoothingOptimisedForecast:
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
            SimpleExponentialSmoothing()
            .fit(ts=ts, alpha=0.3)
            .forecast(start=0, end=(len_ts - 1))
        )

        forecast_optimised = (
            SimpleExponentialSmoothing()
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
