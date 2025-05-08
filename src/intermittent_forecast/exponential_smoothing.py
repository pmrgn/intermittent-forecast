"""Exponential Smoothing for Time Series Forecasting."""

from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import ErrorMetricRegistry


class FittedValues(TypedDict):
    """TypedDict for fitted parameters."""

    alpha: float
    beta: float
    gamma: float
    ts_fitted: npt.NDArray[np.float64]
    trend_type: str
    seasonal_type: str
    period: int
    lvl_final: float
    trend_final: float
    seasonal_final: npt.NDArray[np.float64]


class TripleExponentialSmoothing(BaseForecaster):
    """Triple Exponential Smoothing."""

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
            # Define trend functions
            trend_funcs = {
                "add": lambda lvl, trd, i: lvl + i * trd,
                "mul": lambda lvl, trd, i: lvl * (trd**i),
            }

            # Define seasonal combination functions
            seasonal_combine = {
                "add": lambda val, seas: val + seas,
                "mul": lambda val, seas: val * seas,
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

    def _fit(
        self,
        trend_type: str = "add",
        seasonal_type: str = "add",
        period: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        lvl_final, trend_final, seasonal_final, ts_fitted = (
            TripleExponentialSmoothing.calc_exp_smoothing(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                ts=self.get_timeseries(),
                period=period,
                trend_type=trend_type,
                seasonal_type=seasonal_type,
            )
        )
        self._fitted_params = FittedValues(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ts_fitted=ts_fitted,
            trend_type=trend_type,
            seasonal_type=seasonal_type,
            period=period,
            lvl_final=lvl_final,
            trend_final=trend_final,
            seasonal_final=seasonal_final,
        )

    def get_fitted_params(
        self,
    ) -> FittedValues:
        """Get the fitted parameters."""
        if not self._fitted_params:
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise ValueError(err_msg)

        if not isinstance(self._fitted_params, dict):
            err_msg = (
                "Fitted parameters are not of the expected type. ",
                f"Expected {FittedValues}, got {type(self._fitted_params)}.",
            )
            raise TypeError(err_msg)

        return self._fitted_params

    @staticmethod
    def calc_exp_smoothing(
        alpha: int,
        beta: int,
        gamma: int,
        ts: npt.NDArray[np.float64],
        period: int,
        trend_type: str,
        seasonal_type: str,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        match trend_type, seasonal_type:
            case "add", "add":
                model = TripleExponentialSmoothing._tes_add_add
            case "add", "mul":
                model = TripleExponentialSmoothing._tes_add_mul
            case "mul", "add":
                model = TripleExponentialSmoothing._tes_mul_add
            case "mul", "mul":
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
            trend_type="add",
            seasonal_type="add",
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
            trend_type="add",
            seasonal_type="mul",
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
            trend_type="mul",
            seasonal_type="add",
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
            trend_type="mul",
            seasonal_type="mul",
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
        trend_type: str,
        seasonal_type: str,
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
        if trend_type == "add":
            b[0] = (ts[m : 2 * m].sum() - ts[:m].sum()) / m**2
        if trend_type == "mul":
            b[0] = (ts[m : 2 * m].sum() / ts[:m].sum()) ** (1 / m)
        if seasonal_type == "add":
            s[:m] = ts[:m] - lvl[0]
        if seasonal_type == "mul":
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
