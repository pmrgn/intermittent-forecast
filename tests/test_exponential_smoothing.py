"""Tests for Exponential Smoothing."""

import itertools

import numpy as np
import pytest

from intermittent_forecast.error_metrics import (
    ErrorMetricFunc,
    ErrorMetricRegistry,
)
from intermittent_forecast.exponential_smoothing import (
    SmoothingType,
    TripleExponentialSmoothing,
)

ts_all_positive = np.array([26, 28, 35, 36, 31, 33, 37, 40, 35, 39, 42, 43])
ts_intermittent = np.array([0, 2, 4, 8, 0, 3, 7, 9, 0, 0, 2, 6, 1, 1.5, 5, 10])


def test_tes_add_add() -> None:
    *_, result = TripleExponentialSmoothing._tes_add_add(
        ts=ts_all_positive,
        alpha=0.3,
        beta=0.2,
        gamma=0.1,
        period=4,
    )
    expected = [
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
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_add_mul() -> None:
    *_, result = TripleExponentialSmoothing._tes_add_mul(
        ts=ts_all_positive,
        alpha=0.3,
        beta=0.2,
        gamma=0.1,
        period=4,
    )
    expected = [
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
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_mul_mul() -> None:
    *_, result = TripleExponentialSmoothing._tes_mul_mul(
        ts=ts_all_positive,
        alpha=0.3,
        beta=0.2,
        gamma=0.1,
        period=4,
    )
    expected = [
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
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_mul_add() -> None:
    *_, result = TripleExponentialSmoothing._tes_mul_add(
        ts=ts_all_positive,
        alpha=0.3,
        beta=0.2,
        gamma=0.1,
        period=4,
    )
    expected = [
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
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_add_add_forecast() -> None:
    result = (
        TripleExponentialSmoothing()
        .fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=SmoothingType.ADD.value,
            seasonal_type=SmoothingType.ADD.value,
        )
        .forecast(start=12, end=16)
    )
    expected = [
        36.42864506,
        39.05020833,
        45.82886691,
        47.79049033,
        39.68805488,
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_add_mul_forecast() -> None:
    result = (
        TripleExponentialSmoothing()
        .fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=SmoothingType.ADD.value,
            seasonal_type=SmoothingType.MUL.value,
        )
        .forecast(start=12, end=16)
    )
    expected = [
        35.036353,
        37.980173,
        46.663069,
        49.070469,
        37.624568,
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_mul_mul_forecast() -> None:
    result = (
        TripleExponentialSmoothing()
        .fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=SmoothingType.MUL.value,
            seasonal_type=SmoothingType.MUL.value,
        )
        .forecast(start=12, end=16)
    )
    expected = [35.405082, 38.512039, 47.495269, 50.148919, 38.663521]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_tes_mul_add_forecast() -> None:
    result = (
        TripleExponentialSmoothing()
        .fit(
            ts=ts_all_positive,
            alpha=0.3,
            beta=0.2,
            gamma=0.1,
            period=4,
            trend_type=SmoothingType.MUL.value,
            seasonal_type=SmoothingType.ADD.value,
        )
        .forecast(start=12, end=16)
    )
    expected = [36.811832, 39.583080, 46.527776, 48.683056, 40.843839]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# Build test cases for the optimised forecast error test
default_params = {"alpha": 0.3, "beta": 0.2, "gamma": 0.1}

smoothing_combinations = list(
    itertools.product(
        [SmoothingType.ADD, SmoothingType.MUL],
        [SmoothingType.ADD, SmoothingType.MUL],
    ),
)
error_metrics_str = ErrorMetricRegistry.get_registry().keys()

test_cases = [
    (default_params, {"trend": trend, "seasonal": seasonal}, metric)
    for trend, seasonal in smoothing_combinations
    for metric in error_metrics_str
]


@pytest.mark.parametrize(
    ("smoothing_params", "smoothing_type", "error_metric"),
    test_cases,
)
def test_optimised_forecast_error_less_than_non_optimised(
    smoothing_params: dict[str, float],
    smoothing_type: dict[str, SmoothingType],
    error_metric: ErrorMetricFunc,
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

    assert err_optimised_forecast < err_naive_forecast, (
        f"Expected {err_optimised_forecast} to be less than {err_naive_forecast}"
    )
