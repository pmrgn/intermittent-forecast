"""Tests for the CRO class in the croston module."""

import numpy as np
import pytest

from intermittent_forecast.croston import CRO, SBA, SBJ, TSB


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
    forecast = CRO().fit(ts=basic_time_series, alpha=0.5, beta=0.2).forecast()
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
    forecast = SBA().fit(ts=basic_time_series, alpha=0.5, beta=0.2).forecast()
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
    forecast = SBJ().fit(ts=basic_time_series, alpha=0.5, beta=0.2).forecast()
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
    forecast = TSB().fit(ts=basic_time_series, alpha=0.3, beta=0.1).forecast()
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
        ],
    )
    np.testing.assert_allclose(
        forecast,
        expected,
        rtol=1e-5,
    )


def test_croston_fit() -> None:
    """Test that the fit method calculates the correct parameter values."""
    ts = [1, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 5, 6]
    croston = CRO().fit(ts=ts)
    expected_alpha = 1
    expected_beta = 1
    if croston.alpha != expected_alpha:
        error_message = (
            f"Expected alpha: {expected_alpha}, got: {croston.alpha}"
        )
        raise AssertionError(error_message)
    if croston.beta != expected_beta:
        error_message = f"Expected beta: {expected_beta}, got: {croston.beta}"
        raise AssertionError(error_message)
