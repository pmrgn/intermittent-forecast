"""Tests for the CRO class in the croston module."""

import numpy as np

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
