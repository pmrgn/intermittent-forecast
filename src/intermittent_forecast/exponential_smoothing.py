"""Exponential Smoothing for Time Series Forecasting."""

from __future__ import annotations

from enum import Enum
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import ErrorMetricRegistry


class SmoothingType(Enum):
    """Enum for aggregation modes."""

    ADD = "additive"
    MUL = "multiplicative"


class FittedValues(NamedTuple):
    """TypedDict for fitted parameters."""

    alpha: float
    beta: float
    gamma: float
    ts_base: npt.NDArray[np.float64]
    ts_fitted: npt.NDArray[np.float64]
    trend_type: SmoothingType
    seasonal_type: SmoothingType
    period: int
    lvl_final: float
    trend_final: float
    seasonal_final: npt.NDArray[np.float64]


class TripleExponentialSmoothing(BaseForecaster):
    """Triple Exponential Smoothing."""

    def __init__(self) -> None:
        """Initialise the forecaster."""
        super().__init__()
        self._fitted_params: FittedValues | None = None

    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Forecast the time series using the fitted parameters."""
        # Unpack the fitted values
        fitted_params = self.get_fitted_params()
        trend_type = fitted_params.trend_type
        seasonal_type = fitted_params.seasonal_type
        period = fitted_params.period
        lvl_final = fitted_params.lvl_final
        trend_final = fitted_params.trend_final
        seasonal_final = fitted_params.seasonal_final
        ts_base = fitted_params.ts_base
        ts_fitted = fitted_params.ts_fitted

        # Determine the forecasting horizon if required
        h = end - len(ts_base) + 1
        if h > 0:
            # Define trend functions, which differ based on additive or
            # multiplicative trend type, i.e. for additive smoothing the trend
            # final value is added to the level at each step, i.
            trend_funcs: dict[
                SmoothingType,
                Callable[[float, float, int], float],
            ] = {
                SmoothingType.ADD: lambda lvl, trend, i: lvl + i * trend,
                SmoothingType.MUL: lambda lvl, trend, i: lvl * (trend**i),
            }

            # A seasonal combination functions will be used to combine the now
            # adjusted level with the seasonal component. The seasonal
            # component is either added or multiplied to the level, depending on
            # the seasonal type.
            seasonal_combine_funcs: dict[
                SmoothingType,
                Callable[[float, float], float],
            ] = {
                SmoothingType.ADD: lambda lvl_adjusted, seasonal: lvl_adjusted
                + seasonal,
                SmoothingType.MUL: lambda lvl_adjusted, seasonal: lvl_adjusted
                * seasonal,
            }

            # Get the correct function based on smoothing type.
            apply_trend = trend_funcs[trend_type]
            apply_seasonal = seasonal_combine_funcs[seasonal_type]

            # Generate forecast
            forecast = [
                apply_seasonal(
                    apply_trend(lvl_final, trend_final, i),
                    seasonal_final[(i - 1) % period],
                )
                for i in range(1, h + 1)
            ]

            # Append the out of sample forecast to the fitted values.
            ts_fitted = np.concatenate((ts_fitted, np.array(forecast)))

        return ts_fitted[start : end + 1]

    def get_fitted_params(
        self,
    ) -> FittedValues:
        """Get the fitted parameters."""
        if not self._fitted_params:
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise ValueError(err_msg)

        return self._fitted_params

    def fit(
        self,
        ts: npt.NDArray[np.float64],
        period: int,
        trend_type: str = SmoothingType.ADD.value,
        seasonal_type: str = SmoothingType.ADD.value,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        optimisation_metric: str | None = None,
    ) -> TripleExponentialSmoothing:
        # Validate trend and seasonal types, and convert to enum members.
        trend_type_member = utils.get_enum_member_from_str(
            member_str=trend_type,
            enum_class=SmoothingType,
            member_name="trend_type",
        )
        seasonal_type_member = utils.get_enum_member_from_str(
            member_str=seasonal_type,
            enum_class=SmoothingType,
            member_name="seasonal_type",
        )

        period = utils.validate_non_negative_integer(
            value=period,
            name="period",
        )

        ts = utils.validate_time_series(ts)

        if alpha is None or beta is None or gamma is None:
            # TODO: Bundle params together
            error_metric_func = ErrorMetricRegistry.get(
                optimisation_metric or "MSE",
            )
            # TODO: Need to validate params if required.
            alpha, beta, gamma = (
                TripleExponentialSmoothing._find_optimal_parameters(
                    ts=ts,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    error_metric_func=error_metric_func,
                    period=period,
                    trend_type=trend_type_member,
                    seasonal_type=seasonal_type_member,
                )
            )

        else:
            alpha = utils.validate_float_within_inclusive_bounds(
                name="alpha",
                value=alpha,
                min_value=0,
                max_value=1,
            )
            beta = utils.validate_float_within_inclusive_bounds(
                name="beta",
                value=beta,
                min_value=0,
                max_value=1,
            )
            gamma = utils.validate_float_within_inclusive_bounds(
                name="gamma",
                value=gamma,
                min_value=0,
                max_value=1,
            )

        lvl_final, trend_final, seasonal_final, ts_fitted = (
            TripleExponentialSmoothing._compute_exponential_smoothing(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                ts=ts,
                period=period,
                trend_type=trend_type_member,
                seasonal_type=seasonal_type_member,
            )
        )
        self._fitted_params = FittedValues(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ts_base=ts,
            ts_fitted=ts_fitted,
            trend_type=trend_type_member,
            seasonal_type=seasonal_type_member,
            period=period,
            lvl_final=lvl_final,
            trend_final=trend_final,
            seasonal_final=seasonal_final,
        )

        return self

    @staticmethod
    def _compute_exponential_smoothing(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
        trend_type: SmoothingType,
        seasonal_type: SmoothingType,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute exponential smoothing."""
        match trend_type, seasonal_type:
            case SmoothingType.ADD, SmoothingType.ADD:
                fn = TripleExponentialSmoothing._calculate_add_add_smoothing
            case SmoothingType.ADD, SmoothingType.MUL:
                fn = TripleExponentialSmoothing._calculate_add_mul_smoothing
            case SmoothingType.MUL, SmoothingType.ADD:
                fn = TripleExponentialSmoothing._calculate_mul_add_smoothing
            case SmoothingType.MUL, SmoothingType.MUL:
                fn = TripleExponentialSmoothing._calculate_mul_mul_smoothing
            case _:
                err_msg = (
                    f"Invalid combination of trend_type and seasonal_type: "
                    f"{trend_type}, {seasonal_type}"
                )
                raise ValueError(err_msg)

        return fn(
            ts=ts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            period=period,
        )

    @staticmethod
    def _calculate_add_add_smoothing(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.ADD,
            seasonal_type=SmoothingType.ADD,
        )
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] - s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] + b[i - 1]
            )
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] - (lvl[i - 1] + b[i - 1]))
                + (1 - gamma) * s[i - 1]
            )
        ts_fitted = lvl[:-1] + b[:-1] + s[:-period]
        return lvl[-1], b[-1], s[-period:], ts_fitted

    @staticmethod
    def _calculate_add_mul_smoothing(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.ADD,
            seasonal_type=SmoothingType.MUL,
        )
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] / s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] + b[i - 1]
            )
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] / (lvl[i - 1] + b[i - 1]))
                + (1 - gamma) * s[i - 1]
            )
        ts_fitted = (lvl[:-1] + b[:-1]) * s[:-period]
        return lvl[-1], b[-1], s[-period:], ts_fitted

    @staticmethod
    def _calculate_mul_add_smoothing(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.MUL,
            seasonal_type=SmoothingType.ADD,
        )

        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] - s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] * b[i - 1]
            )
            b[i] = beta * (lvl[i] / lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] - (lvl[i - 1] * b[i - 1]))
                + (1 - gamma) * s[i - 1]
            )
        ts_fitted = (lvl[:-1] * b[:-1]) + s[:-period]
        return lvl[-1], b[-1], s[-period:], ts_fitted

    @staticmethod
    def _calculate_mul_mul_smoothing(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.MUL,
            seasonal_type=SmoothingType.MUL,
        )
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] / s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] * b[i - 1]
            )
            b[i] = beta * (lvl[i] / lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] / (lvl[i - 1] * b[i - 1]))
                + (1 - gamma) * s[i - 1]
            )
        ts_fitted = lvl[:-1] * b[:-1] * s[:-period]
        return lvl[-1], b[-1], s[-period:], ts_fitted

    @staticmethod
    def _initialise_arrays(
        ts: npt.NDArray[np.float64],
        period: int,
        trend_type: SmoothingType,
        seasonal_type: SmoothingType,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        m = period
        n = len(ts)
        lvl = np.zeros(n + 1)
        b = np.zeros(n + 1)
        s = np.zeros(n + m)
        lvl[0] = ts[:m].mean()
        if trend_type == SmoothingType.ADD:
            b[0] = (ts[m : 2 * m].sum() - ts[:m].sum()) / m**2
        if trend_type == SmoothingType.MUL:
            b[0] = (ts[m : 2 * m].sum() / ts[:m].sum()) ** (1 / m)
        if seasonal_type == SmoothingType.ADD:
            s[:m] = ts[:m] - lvl[0]
        if seasonal_type == SmoothingType.MUL:
            s[:m] = ts[:m] / lvl[0]

        return lvl, b, s

    @staticmethod
    def _find_optimal_parameters(
        ts: npt.NDArray[np.float64],
        error_metric_func: Callable[..., float],
        period: int,
        trend_type: SmoothingType,
        seasonal_type: SmoothingType,
        alpha: float | None,
        beta: float | None,
        gamma: float | None,
    ) -> tuple[float, float, float]:
        """Return squared error between timeseries and smoothed array"""
        # Set the bounds for the smoothing parameters. If values have been
        # passed, then the bounds will be locked at that value. Else they are
        # set at (0,1).
        alpha_bounds = (alpha or 0, beta or 1)
        beta_bounds = (beta or 0, beta or 1)
        gamma_bounds = (gamma or 0, gamma or 1)

        # Set the initial guess as the midpoint of the bounds.
        initial_guess = (
            sum(alpha_bounds) / 2,
            sum(beta_bounds) / 2,
            sum(gamma_bounds) / 2,
        )
        min_err = optimize.minimize(
            TripleExponentialSmoothing._cost_function,
            initial_guess,
            args=(
                ts,
                error_metric_func,
                period,
                trend_type,
                seasonal_type,
            ),
            bounds=[
                alpha_bounds,
                beta_bounds,
                gamma_bounds,
            ],
        )
        alpha, beta, gamma = min_err.x

        return alpha, beta, gamma

    @staticmethod
    def _cost_function(
        params: tuple[float, float, float],
        ts: npt.NDArray[np.float64],
        error_metric_func: Callable[..., float],
        period: int,
        trend_type: SmoothingType,
        seasonal_type: SmoothingType,
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta, gamma = params
        *_, ts_fitted = (
            TripleExponentialSmoothing._compute_exponential_smoothing(
                ts=ts,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                period=period,
                trend_type=trend_type,
                seasonal_type=seasonal_type,
            )
        )
        return error_metric_func(ts, ts_fitted)
