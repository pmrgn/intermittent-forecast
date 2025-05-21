"""Tests for Error Metrics module."""

import numpy as np
import pytest

from intermittent_forecast.core.error_metrics import (
    ErrorMetricRegistry,
    ErrorMetrics,
)
from intermittent_forecast.core.types import TSArray


# Sample arrays for testing
@pytest.fixture
def ts() -> TSArray:
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 0.0, -10.0])


@pytest.fixture
def forecast() -> TSArray:
    return np.array([12.0, 18.0, 33.0, 39.0, 48.0, -1.0, 5.0])


def test_mae(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    result = ErrorMetrics.mae(ts, forecast)
    expected = 3.714286
    np.testing.assert_approx_equal(result, expected)


def test_mse(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    result = ErrorMetrics.mse(ts, forecast)
    expected = 35.428574
    np.testing.assert_approx_equal(result, expected)


def test_msr(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    expected = 229.0
    result = ErrorMetrics.msr(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_mar(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    expected = 13.0
    result = ErrorMetrics.mar(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_pis(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    expected = 20.0
    result = ErrorMetrics.pis(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_get_metric_from_str_for_mae(
    ts: TSArray,
    forecast: TSArray,
) -> None:
    mse_func = ErrorMetricRegistry.get("mae")
    result = mse_func(ts, forecast)
    expected = 3.714286
    np.testing.assert_approx_equal(result, expected)


def test_raises_when_metric_not_found() -> None:
    with pytest.raises(
        ValueError,
        match="'foo bar' not found",
    ):
        ErrorMetricRegistry.get("foo bar")
