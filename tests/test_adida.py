"""Tests for the ADIDA class."""

import numpy as np
import numpy.typing as npt
import pytest

from intermittent_forecast.adida import ADIDA
from intermittent_forecast.croston import CRO


@pytest.fixture
def intermittent_time_series() -> npt.NDArray[np.float64]:
    """Fixture for a basic time series."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def croston_model(intermittent_time_series: npt.NDArray[np.float64]) -> CRO:
    """Fixture for the CRO model."""
    return CRO(intermittent_time_series)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (2, [np.nan, np.nan, 1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5, 9.5, 9.5]),
        (3, [np.nan, np.nan, np.nan, 3, 3, 3, 6, 6, 6, 9, 9, 9]),
    ],
)
def test_adida_forecast_croston_block_uniform(
    croston_model: CRO,
    size: int,
    expected: list[np.float64],
) -> None:
    """Test the ADIDA aggregation method."""
    adida_model = ADIDA(
        model=croston_model,
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="uniform",
    )
    result = adida_model.forecast(alpha=1, beta=1)
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
                1,
                0.909090,
                1.090909,
                2,
                1.818181,
                2.181818,
                3,
                2.727272,
                3.272727,
            ],
        ),
    ],
)
def test_adida_forecast_croston_block_seasonal(
    croston_model: CRO,
    size: int,
    expected: list[np.float64],
) -> None:
    """Test the ADIDA aggregation method."""
    adida_model = ADIDA(
        model=croston_model,
        aggregation_period=size,
        aggregation_mode="block",
        disaggregation_mode="seasonal",
    )
    result = adida_model.forecast(alpha=1, beta=1)
    np.testing.assert_allclose(
        result,
        np.array(expected),
        rtol=1e-5,
    )
