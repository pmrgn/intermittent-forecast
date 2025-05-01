"""Tests for the ADIDA class and methods."""

import numpy as np
import numpy.typing as npt
import pytest

from intermittent_forecast.resampler import TimeSeriesResampler


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
    result = TimeSeriesResampler.block_aggregation(
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
    result = TimeSeriesResampler.block_aggregation(
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
    result = TimeSeriesResampler.sliding_aggregation(
        even_intermittent_time_series,
        window_size=size,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [4, 7, 0, 9]),
        (2, [2, 2, 3.5, 3.5, 0, 0, 4.5, 4.5]),
        (
            3,
            [
                1.333333,
                1.333333,
                1.333333,
                2.333333,
                2.333333,
                2.333333,
                0,
                0,
                0,
                3,
                3,
                3,
            ],
        ),
    ],
)
def test_disaggregation_block(
    size: int,
    expected: npt.NDArray[np.float64],
    even_aggregated_time_series: npt.NDArray[np.float64],
) -> None:
    """Test non-overlapping aggregation."""
    result = TimeSeriesResampler.block_disaggregation(
        even_aggregated_time_series,
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
    result = TimeSeriesResampler.calculate_temporal_weights(
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
    result = TimeSeriesResampler.apply_temporal_weights(
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
    temporal_weights = TimeSeriesResampler.calculate_temporal_weights(
        ts=ts,
        cycle_length=cycle,
    )
    result = TimeSeriesResampler.apply_temporal_weights(
        ts=ts_agg,
        weights=temporal_weights,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# def test_aggregate(self):
#     ts = np.array([0, 1, 2, 0, 3])
#     agg = _aggregate(ts, size=1, overlapping=False)
#     self.assertIsNone(assert_array_equal(agg, ts))

#     agg = _aggregate(ts, size=2, overlapping=False)
#     self.assertIsNone(assert_array_equal(agg, [3, 3]))

#     agg = _aggregate(ts, size=2, overlapping=True)
#     self.assertIsNone(assert_array_equal(agg, [1, 3, 2, 3]))


# def test_seasonal_cycle(self):
#     ts = np.tile(np.arange(5), 3).astype("float")
#     s = _seasonal_cycle(ts, cycle=5)
#     self.assertIsNone(
#         assert_allclose(s, [0, 0.1, 0.2, 0.3, 0.4]),
#     )

#     ts = np.array([1, 0, 1, 1, 0, 1, 1])
#     s = _seasonal_cycle(ts, cycle=3)
#     self.assertIsNone(
#         assert_allclose(s, [0, 0.5, 0.5]),
#     )


# def test_apply_temporal_weights(self):
#     f = _apply_temporal_weights(np.ones(5), [0, 0.3, 0.7])
#     self.assertIsNone(
#         assert_allclose(f, [0.3, 0.7, 0, 0.3, 0.7]),
#     )


# def test_adida(self):
#     ts = np.arange(1, 11)
#     f = adida.adida(
#         ts,
#         size=1,
#         overlapping=False,
#         method="cro",
#         alpha=1,
#         h=1,
#     )
#     expected = np.insert(ts.astype("float"), 0, [np.nan])
#     self.assertIsNone(assert_array_equal(f, expected))

#     f = adida.adida(
#         ts,
#         size=2,
#         overlapping=False,
#         method="cro",
#         alpha=1,
#         h=2,
#     )
#     expected = np.array([np.nan, 1.5, 3.5, 5.5, 7.5, 9.5]).repeat(2)
#     self.assertIsNone(assert_array_equal(f, expected))

#     f = adida.adida(
#         ts,
#         size=2,
#         overlapping=True,
#         method="cro",
#         alpha=1,
#         h=1,
#     )
#     expected = np.concatenate(([np.nan, np.nan], np.arange(1.5, 10.5, 1)))
#     self.assertIsNone(assert_array_equal(f, expected))

#     # Test for seasonal cycles
#     ts = np.tile(np.arange(7), 5)
#     f = adida.adida(
#         ts,
#         size=7,
#         overlapping=False,
#         method="sba",
#         alpha=1,
#         h=7,
#         cycle=7,
#     )
#     expected = np.insert(ts / 2, 0, [np.nan] * 7)
#     self.assertIsNone(assert_array_equal(f, expected))
