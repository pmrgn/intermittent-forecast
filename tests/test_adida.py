"""Tests for the ADIDA class."""

import numpy as np
import numpy.typing as npt
import pytest

from intermittent_forecast.adida import ADIDA
from intermittent_forecast.croston import CRO


@pytest.fixture
def time_series_linear() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def time_series_cyclical() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])


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
        model=CRO(),
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="uniform",
    )
    result = adida_model.fit(ts=time_series_linear, alpha=1, beta=1).forecast(
        start=0,
        end=len(time_series_linear) + size,
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
        model=CRO(),
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )
    result = adida_model.fit(
        ts=time_series_linear,
        alpha=1,
        beta=1,
    ).forecast(start=0, end=len(time_series_linear) + size)
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
        model=CRO(),
        aggregation_period=aggregation_period,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )
    result = adida_model.fit(
        ts=time_series_cyclical,
        alpha=1,
        beta=1,
    ).forecast(start=0, end=len(time_series_cyclical) + aggregation_period)
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
