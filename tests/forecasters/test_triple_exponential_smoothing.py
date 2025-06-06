"""Tests for Triple Exponential Smoothing."""

import itertools
from typing import Any

import numpy as np
import pytest

from intermittent_forecast.core._types import TSArray
from intermittent_forecast.core.error_metrics import (
    ErrorMetricRegistry,
)
from intermittent_forecast.forecasters import TripleExponentialSmoothing
from intermittent_forecast.forecasters.triple_exponential_smoothing import (
    _SmoothingType,
)


@pytest.fixture
def ts_all_positive() -> TSArray:
    return np.array([26, 28, 35, 36, 31, 33, 37, 40, 35, 39, 42, 43])


@pytest.fixture
def ts_intermittent() -> TSArray:
    return np.array([0, 28, 35, 0, 0, 0, 37])


class TestTripleExponentialSmoothingFit:
    def test_raises_with_multiplicative_trend_and_non_zero_series(
        self,
        ts_intermittent: TSArray,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="must be all greater than 0 for multiplicative smoothing",
        ):
            TripleExponentialSmoothing().fit(
                ts=ts_intermittent,
                period=2,
                trend_type=_SmoothingType.MUL.value,
            )

    def test_raises_with_multiplicative_seasonality_and_non_zero_series(
        self,
        ts_intermittent: TSArray,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="must be all greater than 0 for multiplicative smoothing",
        ):
            TripleExponentialSmoothing().fit(
                ts=ts_intermittent,
                period=2,
                seasonal_type=_SmoothingType.MUL.value,
            )

    def test_alpha_can_be_set_with_beta_and_gamma_optimised(
        self,
        ts_all_positive: TSArray,
    ) -> None:
        """Test a smoothing parameter can be set with optimisation."""
        alpha = 0.35
        model = TripleExponentialSmoothing().fit(
            ts=ts_all_positive,
            period=4,
            alpha=alpha,
        )
        alpha_fitted = model.get_fit_result()["alpha"]
        if alpha != alpha_fitted:
            err_msg = f"Expected alpha to be {alpha}. Got: {alpha_fitted} "
            raise ValueError(err_msg)


class TestTripleExponentialSmoothingForecast:
    def test_returns_correct_values_with_add_trend_add_seasonality(
        self,
        ts_all_positive: TSArray,
    ) -> None:
        """Test forecasts using additive / additive smoothing."""
        n_obs = len(ts_all_positive)
        model = TripleExponentialSmoothing().fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=_SmoothingType.ADD.value,
            seasonal_type=_SmoothingType.ADD.value,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            27.0,
            29.64,
            36.9896,
            38.114944,
            27.975788,
            31.595831,
            39.843152,
            40.668113,
            31.531949,
            35.204797,
            43.969815,
            45.551800,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [
            36.42864506,
            39.05020833,
            45.82886691,
            47.79049033,
            39.68805488,
        ]

        np.testing.assert_allclose(
            forecast_outsample,
            expected_outsample,
            rtol=1e-5,
        )

    def test_returns_correct_values_with_add_trend_mul_seasonality(
        self,
        ts_all_positive: TSArray,
    ) -> None:
        """Test forecasts using additive / multiplicative smoothing."""
        n_obs = len(ts_all_positive)
        model = TripleExponentialSmoothing().fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=_SmoothingType.ADD.value,
            seasonal_type=_SmoothingType.MUL.value,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            26.832,
            29.46944,
            37.228352,
            38.436415,
            27.641080,
            31.541533,
            40.976257,
            41.732173,
            30.754507,
            35.007834,
            45.892068,
            47.372845,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [
            35.036353,
            37.980173,
            46.663069,
            49.070469,
            37.624568,
        ]

        np.testing.assert_allclose(
            forecast_outsample,
            expected_outsample,
            rtol=1e-5,
        )

    def test_returns_correct_values_with_mul_trend_mul_seasonality(
        self,
        ts_all_positive: TSArray,
    ) -> None:
        """Test forecasts using multiplicative / multiplicative smoothing."""
        n_obs = len(ts_all_positive)
        model = TripleExponentialSmoothing().fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=_SmoothingType.MUL.value,
            seasonal_type=_SmoothingType.MUL.value,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            26.794806,
            29.420967,
            37.176763,
            38.399407,
            27.629333,
            31.579427,
            41.106434,
            41.914905,
            30.914529,
            35.233195,
            46.285054,
            47.846334,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [
            35.405082,
            38.512039,
            47.495269,
            50.148919,
            38.663521,
        ]

        np.testing.assert_allclose(
            forecast_outsample,
            expected_outsample,
            rtol=1e-5,
        )

    def test_returns_correct_values_with_mul_trend_add_seasonality(
        self,
        ts_all_positive: TSArray,
    ) -> None:
        """Test forecasts using multiplicative / additive smoothing."""
        n_obs = len(ts_all_positive)
        model = TripleExponentialSmoothing().fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=_SmoothingType.MUL.value,
            seasonal_type=_SmoothingType.ADD.value,
        )
        forecast_insample = model.forecast(start=0, end=n_obs - 1)
        expected_insample = [
            26.955296,
            29.585901,
            36.943539,
            38.082818,
            27.961601,
            31.627615,
            39.935793,
            40.798304,
            31.695027,
            35.418386,
            44.266201,
            45.9056501,
        ]
        np.testing.assert_allclose(
            forecast_insample,
            expected_insample,
            rtol=1e-5,
        )

        forecast_outsample = model.forecast(start=n_obs, end=n_obs + 4)
        expected_outsample = [
            36.811832,
            39.583080,
            46.527776,
            48.683056,
            40.843839,
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
            TripleExponentialSmoothing().forecast(start=0, end=1)


def get_test_cases() -> list[tuple[dict[str, float], dict[str, Any], Any]]:
    """Generate test cases for TestCROOptimisedForecast."""
    # Build test cases for the optimised forecast error test
    default_params = {"alpha": 0.3, "beta": 0.2, "gamma": 0.1}

    smoothing_combinations = list(
        itertools.product(
            [_SmoothingType.ADD, _SmoothingType.MUL],
            [_SmoothingType.ADD, _SmoothingType.MUL],
        ),
    )
    error_metrics_str = ErrorMetricRegistry.get_registry().keys()

    return [
        (default_params, {"trend": trend, "seasonal": seasonal}, metric)
        for trend, seasonal in smoothing_combinations
        for metric in error_metrics_str
    ]


class TestTripleExponentialSmoothingOptimisedForecast:
    @pytest.mark.parametrize(
        ("smoothing_params", "smoothing_type", "error_metric"),
        get_test_cases(),
    )
    def test_optimised_forecast_error_less_than_non_optimised(
        self,
        ts_all_positive: TSArray,
        smoothing_params: dict[str, float],
        smoothing_type: dict[str, _SmoothingType],
        error_metric: str,
    ) -> None:
        """Test that an optimised forecast produces minimised error.

        This test checks that when fitting the parameters through optimisation,
        the resulting forecast has a lower error than when using the provided
        parameters.
        """
        ts = ts_all_positive
        len_ts = len(ts)
        forecast_estimated = (
            TripleExponentialSmoothing()
            .fit(
                ts=ts,
                trend_type=smoothing_type["trend"].value,
                seasonal_type=smoothing_type["seasonal"].value,
                alpha=smoothing_params["alpha"],
                beta=smoothing_params["beta"],
                gamma=smoothing_params["gamma"],
                period=4,
            )
            .forecast(start=0, end=(len_ts - 1))
        )
        forecast_optimised = (
            TripleExponentialSmoothing()
            .fit(
                ts=ts,
                trend_type=smoothing_type["trend"].value,
                seasonal_type=smoothing_type["seasonal"].value,
                period=4,
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
