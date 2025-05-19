"""Tests for the Croston model and variants."""

import itertools
from typing import Any

import numpy as np
import pytest

from intermittent_forecast.base_forecaster import TSArray
from intermittent_forecast.croston import Croston, _CrostonVariant
from intermittent_forecast.error_metrics import (
    ErrorMetricRegistry,
)


@pytest.fixture
def basic_time_series() -> list[float]:
    return [0, 0, 3, 0, 4, 0, 0, 0, 2, 0]


class TestCROFit:
    def test_raises_on_all_zero_series(self) -> None:
        ts = [0.0, 0.0, 0.0, 0.0]
        with pytest.raises(
            ValueError,
            match="at least two non-zero values",
        ):
            Croston().fit(ts=ts)

    def test_raises_on_single_value_series(self) -> None:
        ts = [0.0, 0.0, 0.0, 2.0]
        with pytest.raises(
            ValueError,
            match="at least two non-zero values",
        ):
            Croston().fit(ts=ts)

    def test_raises_on_two_dimensional_array(self) -> None:
        ts = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(
            ValueError,
            match="must be 1-dimensional",
        ):
            Croston().fit(ts=ts)

    def test_raises_on_invalid_ts_type(self) -> None:
        ts = "foo bar"
        with pytest.raises(
            ValueError,
            match="must be an array of integers or floats",
        ):
            Croston().fit(ts=ts)

    def test_raises_on_invalid_ts_values(self) -> None:
        ts = ["foo", "bar"]
        with pytest.raises(
            ValueError,
            match="must be an array of integers or floats",
        ):
            Croston().fit(ts=ts)

    def test_raises_oninvalid_alpha(
        self,
        basic_time_series: TSArray,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="out of bounds",
        ):
            Croston().fit(ts=basic_time_series, beta=3)

    def test_raises_on_invalid_optimisation_metric_string(
        self,
        basic_time_series: TSArray,
    ) -> None:
        invalid_metric = "Foo bar"
        with pytest.raises(
            ValueError,
            match=f"Error metric '{invalid_metric}' not found",
        ):
            Croston().fit(
                ts=basic_time_series,
                optimisation_metric=invalid_metric,
            )

    def test_raises_on_invalid_optimisation_metric_type(
        self,
        basic_time_series: TSArray,
    ) -> None:
        invalid_metric = 5
        with pytest.raises(
            TypeError,
            match="Error metric must be a string",
        ):
            Croston().fit(
                ts=basic_time_series,
                optimisation_metric=invalid_metric,  # type: ignore[arg-type]
            )

    def test_alpha_can_be_set_with_beta_optimised(
        self,
        basic_time_series: list[float],
    ) -> None:
        """Test alpha can be set while beta is optimised."""
        alpha = 0.35
        model = Croston().fit(ts=basic_time_series, alpha=alpha)
        fitted_alpha = model.get_fitted_model_result().alpha
        if alpha != fitted_alpha:
            err_msg = f"Expected alpha to be {alpha}. Got: {fitted_alpha}"
            raise ValueError(err_msg)


class TestCROForecast:
    def test_returns_correct_values(
        self,
        basic_time_series: list[float],
    ) -> None:
        forecast = (
            Croston()
            .fit(ts=basic_time_series, alpha=0.5, beta=0.2)
            .forecast(start=0, end=len(basic_time_series) + 1)
        )
        expected = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                1.25,
                1.25,
                1.25,
                1.25,
                0.904605,
                0.904605,
            ],
        )
        np.testing.assert_allclose(
            forecast,
            expected,
            rtol=1e-5,
        )

    def test_raises_if_model_not_fit(self) -> None:
        with pytest.raises(
            RuntimeError,
            match="Model has not been fitted yet",
        ):
            Croston().forecast(start=0, end=1)

    def test_raises_on_invalid_forecast_start(
        self,
        basic_time_series: TSArray,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="start must be 0 or greater",
        ):
            Croston().fit(ts=basic_time_series, alpha=1, beta=1).forecast(
                start=-1,
                end=1,
            )


class TestSBAForecast:
    def test_returns_correct_values(
        self,
        basic_time_series: list[float],
    ) -> None:
        forecast = (
            Croston()
            .fit(ts=basic_time_series, variant="SBA", alpha=0.5, beta=0.2)
            .forecast(start=0, end=len(basic_time_series) + 1)
        )
        expected = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                0.9,
                0.9,
                1.125,
                1.125,
                1.125,
                1.125,
                0.8141447,
                0.8141447,
            ],
        )
        np.testing.assert_allclose(
            forecast,
            expected,
            rtol=1e-5,
        )


class TestSBJForecast:
    def test_returns_correct_values(
        self,
        basic_time_series: list[float],
    ) -> None:
        forecast = (
            Croston()
            .fit(ts=basic_time_series, variant="SBJ", alpha=0.5, beta=0.2)
            .forecast(start=0, end=len(basic_time_series) + 1)
        )
        expected = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                0.888888,
                0.888888,
                1.111111,
                1.111111,
                1.111111,
                1.111111,
                0.804093,
                0.804093,
            ],
        )
        np.testing.assert_allclose(
            forecast,
            expected,
            rtol=1e-5,
        )


class TestTSBForecast:
    def test_returns_correct_values(
        self,
        basic_time_series: list[float],
    ) -> None:
        forecast = (
            Croston()
            .fit(ts=basic_time_series, variant="TSB", alpha=0.3, beta=0.1)
            .forecast(start=0, end=len(basic_time_series) + 2)
        )
        expected = np.array(
            [
                np.nan,
                0.9,
                0.81,
                1.029,
                0.9261,
                1.2468390,
                1.1221551,
                1.0099396,
                0.9089456,
                1.0123723,
                0.9111351,
                0.9111351,
            ],
        )
        np.testing.assert_allclose(
            forecast,
            expected,
            rtol=1e-5,
        )


def get_test_cases() -> list[tuple[tuple[float, float], Any, Any]]:
    """Generate test cases for TestCROOptimisedForecast."""
    parameter_grid_search = list(
        itertools.product(
            [0.01, 0.99],
            [0.01, 0.99],
        ),
    )
    error_metrics_str = ErrorMetricRegistry.get_registry().keys()
    variant_str = [variant.value for variant in _CrostonVariant]

    return [
        (params, metric, variant)
        for params in parameter_grid_search
        for metric in error_metrics_str
        for variant in variant_str
    ]


class TestCROOptimisedForecast:
    @pytest.mark.parametrize(
        ("smoothing_params", "error_metric", "variant"),
        get_test_cases(),
    )
    def test_optimised_forecast_error_less_than_non_optimised(
        self,
        smoothing_params: tuple[float, float],
        error_metric: str,
        variant: str,
    ) -> None:
        """Test that an optimised forecast produces minimised error.

        This test checks that when fitting the parameters through optimisation,
        the resulting forecast has a lower error than when using the provided
        parameters.
        """
        ts = np.array([1, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 5, 6])
        len_ts = len(ts)
        forecast_estimated = (
            Croston()
            .fit(
                ts=ts,
                alpha=smoothing_params[0],
                beta=smoothing_params[1],
                variant=variant,
            )
            .forecast(start=0, end=(len_ts))
        )

        forecast_optimised = (
            Croston()
            .fit(ts=ts, optimisation_metric=error_metric, variant=variant)
            .forecast(start=0, end=(len_ts))
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
