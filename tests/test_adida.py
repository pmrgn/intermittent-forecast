"""Tests for the ADIDA class."""

import numpy as np
import numpy.typing as npt
import pytest

from intermittent_forecast.adida import ADIDA
from intermittent_forecast.croston import CRO
from intermittent_forecast.exponential_smoothing import (
    TripleExponentialSmoothing,
)


@pytest.fixture
def time_series_linear() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def time_series_cyclical() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])


@pytest.fixture
def even_intermittent_time_series() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([0, 0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0])


@pytest.fixture
def even_aggregated_time_series() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([4, 7, 0, 9])


@pytest.fixture
def odd_time_series() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0])


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [0, 0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0]),
        (2, [0, 3, 5, 0, 2, 0, 0, 4]),
        (3, [3, 5, 2, 0, 4]),
        (4, [3, 5, 2, 4]),
        (7, [8, 6]),
        (8, [8, 6]),
        (9, [6]),
    ],
)
def test_aggregation_even_block(
    size: int,
    expected: npt.NDArray[np.float64],
    even_intermittent_time_series: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    result = ADIDA.block_aggregation(
        even_intermittent_time_series,
        window_size=size,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0]),
        (2, [3, 5, 0, 2, 0, 0, 4]),
        (3, [3, 5, 2, 0, 4]),
        (4, [5, 2, 4]),
        (7, [8, 6]),
        (8, [6]),
        (9, [6]),
    ],
)
def test_aggregation_odd_block(
    size: int,
    expected: npt.NDArray[np.float64],
    odd_time_series: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    result = ADIDA.block_aggregation(
        odd_time_series,
        window_size=size,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [0, 0, 3, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0]),
        (2, [0, 3, 3, 0, 5, 5, 0, 0, 2, 2, 0, 0, 0, 4, 4]),
        (3, [3, 3, 3, 5, 5, 5, 0, 2, 2, 2, 0, 0, 4, 4]),
        (4, [3, 3, 8, 5, 5, 5, 2, 2, 2, 2, 0, 4, 4]),
        (7, [8, 8, 8, 7, 7, 7, 2, 2, 6, 6]),
        (8, [8, 8, 10, 7, 7, 7, 2, 6, 6]),
        (16, [14]),
    ],
)
def test_aggregation_even_overlapping(
    size: int,
    expected: npt.NDArray[np.float64],
    even_intermittent_time_series: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    result = ADIDA.sliding_aggregation(
        even_intermittent_time_series,
        window_size=size,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("window_size", "base_ts_length", "expected"),
    [
        (1, 4, [4, 8, 0, 9]),
        (2, 8, [2, 2, 4, 4, 0, 0, 4.5, 4.5]),
        (2, 9, [np.nan, 2, 2, 4, 4, 0, 0, 4.5, 4.5]),
        (
            4,
            16,
            [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2.25, 2.25, 2.25, 2.25],
        ),
        (
            4,
            18,
            [
                np.nan,
                np.nan,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                0,
                0,
                0,
                0,
                2.25,
                2.25,
                2.25,
                2.25,
            ],
        ),
    ],
)
def test_disaggregation_block(
    window_size: int,
    base_ts_length: int,
    expected: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    aggregated_ts = np.array([4, 8, 0, 9])
    result = ADIDA.block_disaggregation(
        aggregated_ts=aggregated_ts,
        window_size=window_size,
        base_ts_length=base_ts_length,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [4, 7, 2, 9, 0, 6]),
        (2, [np.nan, 2, 3.5, 1, 4.5, 0, 3]),
        (3, [np.nan, np.nan, 1.333333, 2.333333, 0.666666, 3, 0, 2]),
    ],
)
def test_disaggregation_sliding(
    size: int,
    expected: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    ts = np.array([4, 7, 2, 9, 0, 6])
    result = ADIDA.sliding_disaggregation(
        ts=ts,
        window_size=size,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("ts", "cycle", "expected"),
    [
        (
            np.array([4, 7, 2, 9, 1, 6, 4, 8]),
            4,
            np.array([0.121951, 0.317073, 0.146341, 0.414634]),
        ),
        (
            np.array([4, 7, 2, 9, 1, 6, 4, 8]),
            3,
            np.array([0.377777, 0.355555, 0.266666]),
        ),
        (
            np.array([4, 0, 2, 9, 0, 6, 4, 0]),
            3,
            np.array([0.586206, 0, 0.413793]),
        ),
        (
            np.array([0, 7, 2, 9, 1, 6, 4, 8]),
            1,
            np.array([1]),
        ),
    ],
)
def test_calculate_temporal_weights(
    ts: npt.NDArray[np.float64],
    cycle: int,
    expected: npt.NDArray[np.float64],
) -> None:
    """Test computing temporal distribution."""
    result = ADIDA.calculate_temporal_weights(
        ts=ts,
        cycle_length=cycle,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("ts", "temporal_weights", "expected"),
    [
        (
            np.array([2, 2, 2, 2]),
            np.array([0.1, 0.2, 0.4, 0.3]),
            np.array([0.8, 1.6, 3.2, 2.4]),
        ),
        (
            np.array([4, 7, 2, 9, 1, 6, 4, 8]),
            np.array([0.1, 0.2, 0.4, 0.3]),
            np.array([1.6, 5.6, 3.2, 10.8, 0.4, 4.8, 6.4, 9.6]),
        ),
        (
            np.array([4, 7, 2, 9, 1]),
            np.array([0.1, 0.2, 0.7]),
            np.array([1.2, 4.2, 4.2, 2.7, 0.6]),
        ),
    ],
)
def test_apply_temporal_weights(
    ts: npt.NDArray[np.float64],
    temporal_weights: int,
    expected: npt.NDArray[np.float64],
) -> None:
    """Test applying a temporal distribution to a time-series."""
    result = ADIDA.apply_temporal_weights(
        ts=ts,
        weights=temporal_weights,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("ts", "cycle", "ts_agg", "expected"),
    [
        (
            np.array([4, 1, 2, 3, 4, 1, 2, 3, 4]),
            4,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1.6, 0.4, 0.8, 1.2, 1.6, 0.4, 0.8, 1.2, 1.6]),
        ),
    ],
)
def test_calculate_and_apply_temporal_weights(
    ts: npt.NDArray[np.float64],
    cycle: int,
    ts_agg: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> None:
    """Test applying a temporal distribution to a time-series."""
    temporal_weights = ADIDA.calculate_temporal_weights(
        ts=ts,
        cycle_length=cycle,
    )
    result = ADIDA.apply_temporal_weights(
        ts=ts_agg,
        weights=temporal_weights,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (2, [np.nan, np.nan, 1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 9.5, 9.5]),
        (3, [np.nan, np.nan, np.nan, 3, 3, 3, 6, 6, 6, 9, 9, 9, 9]),
    ],
)
def test_adida_forecast_croston_block_uniform(
    time_series_linear: npt.NDArray[np.float64],
    size: int,
    expected: list[np.float64],
) -> None:
    """Test the ADIDA aggregation method."""
    adida_model = ADIDA(
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="uniform",
    )
    result = adida_model.fit(
        model=CRO(),
        ts=time_series_linear,
        alpha=1,
        beta=1,
    ).forecast(
        start=0,
        end=len(time_series_linear) + size - 1,
    )
    np.testing.assert_allclose(
        result,
        np.array(expected),
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (
            3,
            [
                np.nan,
                np.nan,
                np.nan,
                3,
                2.727272,
                3.272727,
                6,
                5.454545,
                6.545454,
                9,
                8.181818,
                9.818181,
                9,
            ],
        ),
    ],
)
def test_adida_forecast_croston_block_seasonal(
    time_series_linear: npt.NDArray[np.float64],
    size: int,
    expected: list[np.float64],
) -> None:
    """Test the ADIDA aggregation method."""
    adida_model = ADIDA(
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )
    result = adida_model.fit(
        model=CRO(),
        ts=time_series_linear,
        alpha=1,
        beta=1,
    ).forecast(start=0, end=len(time_series_linear) + size - 1)
    np.testing.assert_allclose(
        result,
        np.array(expected),
        rtol=1e-5,
    )


def test_adida_forecast_croston_cyclical_block_seasonal(
    time_series_cyclical: npt.NDArray[np.float64],
) -> None:
    """Test the ADIDA aggregation method."""
    aggregation_period = 5
    adida_model = ADIDA(
        aggregation_period=aggregation_period,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )
    result = adida_model.fit(
        model=CRO(),
        ts=time_series_cyclical,
        alpha=1,
        beta=1,
    ).forecast(start=0, end=len(time_series_cyclical) + aggregation_period - 1)
    np.testing.assert_allclose(
        result,
        np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
            ],
        ),
        rtol=1e-5,
    )


def test_adida_forecast_triple_exponential_smoothing() -> None:
    """Test the ADIDA aggregation method."""
    # Generate an intermittent array from a non-zero cyclical series.
    cyclical_series = np.array(
        [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8],
    )
    period = 5
    zero_series = np.zeros(len(cyclical_series) * period)

    # Assign the cyclical array to every 5th value array, to make it
    # `[1,0,0,0,0,2,0,0,0,0,3,...]`
    intermittent_series = zero_series.copy()
    intermittent_series[::period] = cyclical_series

    adida_model = ADIDA(
        aggregation_period=period,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )

    adida_forecast = adida_model.fit(
        model=TripleExponentialSmoothing(),
        ts=intermittent_series,
        period=4,
    ).forecast(start=0, end=len(intermittent_series) - 1)

    tes_forecast = (
        TripleExponentialSmoothing()
        .fit(
            ts=cyclical_series,
            period=4,
        )
        .forecast(start=0, end=len(cyclical_series) - 1)
    )
    tes_forecast_intermittent = zero_series.copy()
    tes_forecast_intermittent[::period] = tes_forecast

    # The resulting ADIDA forecast should be equivalent to the Exponential
    # Smoothing forecast,
    np.testing.assert_allclose(
        adida_forecast,
        tes_forecast_intermittent,
        rtol=1e-5,
    )
