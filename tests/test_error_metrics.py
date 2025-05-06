"""Unit tests for ErrorMetrics class."""

import numpy as np
import pytest

from intermittent_forecast.error_metrics import (
    ErrorMetricRegistry,
    ErrorMetrics,
)

# Sample arrays for testing
ts = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 0.0, -10.0])
forecast = np.array([12.0, 18.0, 33.0, 39.0, 48.0, -1.0, 5.0])


def test_mae() -> None:
    result = ErrorMetrics.mae(ts, forecast)
    expected = 3.714286
    np.testing.assert_approx_equal(result, expected)


def test_mse() -> None:
    result = ErrorMetrics.mse(ts, forecast)
    expected = 35.428574
    np.testing.assert_approx_equal(result, expected)


def test_msr() -> None:
    expected = 229.0
    result = ErrorMetrics.msr(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_mar() -> None:
    expected = 13.0
    result = ErrorMetrics.mar(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_pis() -> None:
    expected = 20.0
    result = ErrorMetrics.pis(ts, forecast)
    np.testing.assert_approx_equal(result, expected)


def test_error_metric_registry_retrieves_mae() -> None:
    mse_func = ErrorMetricRegistry.get("mae")
    result = mse_func(ts, forecast)
    expected = 3.714286
    np.testing.assert_approx_equal(result, expected)


def test_error_metric_registry_retrieval_fails() -> None:
    with pytest.raises(
        ValueError,
        match="'foo bar' not found",
    ):
        ErrorMetricRegistry.get("foo bar")
