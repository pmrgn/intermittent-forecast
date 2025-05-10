"""Tests for the CRO class in the croston module."""

import itertools

import numpy as np
import pytest

from intermittent_forecast.croston import CRO, SBA, SBJ, TSB
from intermittent_forecast.error_metrics import (
    ErrorMetricFunc,
    ErrorMetricRegistry,
)


@pytest.fixture
def basic_time_series() -> list[float]:
    """Fixture for a basic time series."""
    return [0, 0, 3, 0, 4, 0, 0, 0, 2, 0]


def test_all_zero_series() -> None:
    """Test a time-series with all zero values raises a ValueError."""
    ts = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(
        ValueError,
        match="at least two non-zero values",
    ):
        CRO().fit(ts=ts)


def test_single_value_series() -> None:
    """Test a time-series with a single non-zero value raises a ValueError."""
    ts = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(
        ValueError,
        match="at least two non-zero values",
    ):
        CRO().fit(ts=ts)


def test_croston_forecast(basic_time_series: list[float]) -> None:
    """Test the forecast method of the CRO class."""
    forecast = (
        CRO()
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


def test_sba_forecast(basic_time_series: list[float]) -> None:
    """Test the forecast method of the SBA class."""
    forecast = (
        SBA()
        .fit(ts=basic_time_series, alpha=0.5, beta=0.2)
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


def test_sbj_forecast(basic_time_series: list[float]) -> None:
    """Test the forecast method of the SBJ class."""
    forecast = (
        SBJ()
        .fit(ts=basic_time_series, alpha=0.5, beta=0.2)
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


def test_tsb_forecast(basic_time_series: list[float]) -> None:
    """Test the forecast method of the TSB class."""
    forecast = (
        TSB()
        .fit(ts=basic_time_series, alpha=0.3, beta=0.1)
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


parameter_grid_search = list(
    itertools.product(
        [0.01, 0.1, 0.99],
        [0.01, 0.1, 0.99],
    ),
)
error_metrics_str = ErrorMetricRegistry.get_registry().keys()
test_cases = [
    (params, metric)
    for params in parameter_grid_search
    for metric in error_metrics_str
]


@pytest.mark.parametrize(
    ("smoothing_params", "error_metric"),
    test_cases,
)
def test_optimised_forecast_error_less_than_non_optimised(
    smoothing_params: tuple[float, float],
    error_metric: ErrorMetricFunc,
) -> None:
    """Test that an optimised forecast produces minimised error.

    This test checks that when fitting the parameters through optimisation,
    the resulting forecast has a lower error than when using the provided
    parameters.
    """
    ts = np.array([1, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 5, 6])
    len_ts = len(ts)
    forecast_estimated = (
        CRO()
        .fit(
            ts=ts,
            alpha=smoothing_params[0],
            beta=smoothing_params[1],
        )
        .forecast(start=0, end=(len_ts))
    )

    forecast_optimised = (
        CRO().fit(ts=ts, metric=error_metric).forecast(start=0, end=(len_ts))
    )
    # Get the error metric function from the string
    error_metric_func = ErrorMetricRegistry.get(error_metric)

    # Calculate the error for both forecasts
    err_naive_forecast = error_metric_func(ts, forecast_estimated)
    err_optimised_forecast = error_metric_func(ts, forecast_optimised)

    if not (err_optimised_forecast <= err_naive_forecast):
        err_msg = (
            f"Expected optimised forecast error to be less than default guess. "
            f"Got: {err_optimised_forecast} greater than {err_naive_forecast}"
        )
        raise ValueError(err_msg)
