"""Exponential Smoothing for Time Series Forecasting."""

from enum import Enum
from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import ErrorMetricRegistry


class SmoothingType(Enum):
    """Enum for aggregation modes."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class FittedValues(TypedDict):
    """TypedDict for fitted parameters."""

    alpha: float
    beta: float
    gamma: float
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
        trend_type = fitted_params["trend_type"]
        seasonal_type = fitted_params["seasonal_type"]
        period = fitted_params["period"]
        lvl_final = fitted_params["lvl_final"]
        trend_final = fitted_params["trend_final"]
        seasonal_final = fitted_params["seasonal_final"]
        ts_fitted = fitted_params["ts_fitted"]

        # Determine the forecasting horizon if required
        h = end - len(self.get_timeseries()) + 1
        if h > 0:
            # Define trend functions, which differ based on additive or
            # multiplicative trend type, i.e. for additive smoothing the trend
            # final value is added to the level at each step, i.
            trend_funcs = {
                SmoothingType.ADDITIVE: lambda lvl, trend, i: lvl + i * trend,
                SmoothingType.MULTIPLICATIVE: lambda lvl, trend, i: lvl
                * (trend**i),
            }

            # A seasonal combination functions will be used to combine the now
            # adjusted level with the seasonal component. The seasonal
            # component is either added or multiplied to the level, depending on
            # the seasonal type.
            seasonal_combine = {
                SmoothingType.ADDITIVE: lambda lvl_adjusted,
                seasonal: lvl_adjusted + seasonal,
                SmoothingType.MULTIPLICATIVE: lambda lvl_adjusted,
                seasonal: lvl_adjusted * seasonal,
            }

            # Get the correct functions based on types
            apply_trend = trend_funcs[trend_type]
            apply_seasonal = seasonal_combine[seasonal_type]

            # Generate forecast
            forecast = [
                apply_seasonal(
                    apply_trend(lvl_final, trend_final, i),
                    seasonal_final[(i - 1) % period],
                )
                for i in range(1, h + 1)
            ]

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

    def _fit(
        self,
        trend_type: str = SmoothingType.ADDITIVE.value,
        seasonal_type: str = SmoothingType.ADDITIVE.value,
        period: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        # Validate trend and seasonal types, and convert to enum members.
        trend_type_ = utils.get_enum_member_from_str(
            member_str=trend_type,
            enum_class=SmoothingType,
            member_name="trend_type",
        )
        seasonal_type_ = utils.get_enum_member_from_str(
            member_str=seasonal_type,
            enum_class=SmoothingType,
            member_name="seasonal_type",
        )

        lvl_final, trend_final, seasonal_final, ts_fitted = (
            TripleExponentialSmoothing.calc_exp_smoothing(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                ts=self.get_timeseries(),
                period=period,
                trend_type=trend_type_,
                seasonal_type=seasonal_type_,
            )
        )
        self._fitted_params = FittedValues(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ts_fitted=ts_fitted,
            trend_type=trend_type_,
            seasonal_type=seasonal_type_,
            period=period,
            lvl_final=lvl_final,
            trend_final=trend_final,
            seasonal_final=seasonal_final,
        )

    @staticmethod
    def calc_exp_smoothing(
        alpha: int,
        beta: int,
        gamma: int,
        ts: npt.NDArray[np.float64],
        period: int,
        trend_type: SmoothingType,
        seasonal_type: SmoothingType,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the exponential smoothing."""
        match trend_type, seasonal_type:
            case SmoothingType.ADDITIVE, SmoothingType.ADDITIVE:
                model = TripleExponentialSmoothing._tes_add_add
            case SmoothingType.ADDITIVE, SmoothingType.MULTIPLICATIVE:
                model = TripleExponentialSmoothing._tes_add_mul
            case SmoothingType.MULTIPLICATIVE, SmoothingType.ADDITIVE:
                model = TripleExponentialSmoothing._tes_mul_add
            case SmoothingType.MULTIPLICATIVE, SmoothingType.MULTIPLICATIVE:
                model = TripleExponentialSmoothing._tes_mul_mul
            case _:
                err_msg = (
                    f"Invalid combination of trend_type and seasonal_type: "
                    f"{trend_type}, {seasonal_type}"
                )
                raise ValueError(err_msg)

        return model(
            ts=ts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            period=period,
        )

    @staticmethod
    def _tes_add_add(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.ADDITIVE,
            seasonal_type=SmoothingType.ADDITIVE,
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
    def _tes_add_mul(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.ADDITIVE,
            seasonal_type=SmoothingType.MULTIPLICATIVE,
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
    def _tes_mul_add(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.MULTIPLICATIVE,
            seasonal_type=SmoothingType.ADDITIVE,
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
    def _tes_mul_mul(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts=ts,
            period=period,
            trend_type=SmoothingType.MULTIPLICATIVE,
            seasonal_type=SmoothingType.MULTIPLICATIVE,
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
    def initialise_arrays(
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
        if trend_type == SmoothingType.ADDITIVE:
            b[0] = (ts[m : 2 * m].sum() - ts[:m].sum()) / m**2
        if trend_type == SmoothingType.MULTIPLICATIVE:
            b[0] = (ts[m : 2 * m].sum() / ts[:m].sum()) ** (1 / m)
        if seasonal_type == SmoothingType.ADDITIVE:
            s[:m] = ts[:m] - lvl[0]
        if seasonal_type == SmoothingType.MULTIPLICATIVE:
            s[:m] = ts[:m] / lvl[0]

        return lvl, b, s

    @staticmethod
    def _get_optimised_parameters(
        ts: npt.NDArray[np.float64],
        metric: str,
        period: int,
        trend_type: str,
        seasonal_type: str,
    ) -> tuple[float, float, float]:
        """Return squared error between timeseries and smoothed array"""
        error_metric_func = ErrorMetricRegistry.get(metric)
        # TODO: Change to alpha_bnds, store as tuple. Same for Crostons.
        # Set the bounds for alpha and beta.
        alpha_min, alpha_max = (0, 1)
        beta_min, beta_max = (0, 1)
        gamma_min, gamma_max = (0, 1)

        # Set the initial guess as the midpoint of the bounds for alpha and
        # beta.
        initial_guess = (
            (alpha_max - alpha_min) / 2,
            (beta_max - beta_min) / 2,
            (gamma_max - gamma_min) / 2,
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
                (alpha_min, alpha_max),
                (beta_min, beta_max),
                (gamma_min, gamma_max),
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
        trend_type: str,
        seasonal_type: str,
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta, gamma = params
        *_, ts_fitted = TripleExponentialSmoothing.calc_exp_smoothing(
            ts=ts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            period=period,
            trend_type=trend_type,
            seasonal_type=seasonal_type,
        )
        return error_metric_func(ts, ts_fitted)
