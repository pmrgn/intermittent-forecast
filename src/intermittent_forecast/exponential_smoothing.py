"""Exponential Smoothing for Time Series Forecasting."""

from typing import TypedDict, Callable

import numpy as np
import numpy.typing as npt

from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import ErrorMetricRegistry
from scipy import optimize


class FittedParams(TypedDict):
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
        self._fitted_params = FittedParams(
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

    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Forecast the time series using the fitted parameters."""
        # Get the fitted parameters
        fitted_params = self.get_fitted_params()
        trend_type = fitted_params.get("trend_type")
        seasonal_type = fitted_params.get("seasonal_type")
        period = fitted_params.get("period")
        lvl_final = fitted_params.get("lvl_final")
        trend_final = fitted_params.get("trend_final")
        seasonal_final = fitted_params.get("seasonal_final")
        ts_fitted = fitted_params.get("ts_fitted")

        # Determine the forecast horizon
        h = end - len(self.get_timeseries())
        if h >= 1:
            match trend_type, seasonal_type:
                case "add", "add":
                    forecast = np.array(
                        [
                            lvl_final
                            + i * trend_final
                            + seasonal_final[-(-i % period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                case "add", "mul":
                    forecast = np.array(
                        [
                            lvl_final
                            + i
                            * trend_final
                            * seasonal_final[-(-i % period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                case "mul", "add":
                    forecast = np.array(
                        [
                            lvl_final * trend_final**i
                            + seasonal_final[-(-i % period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                case "mul", "mul":
                    forecast = np.array(
                        [
                            lvl_final
                            * trend_final**i
                            * seasonal_final[-(-i % period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                case _:
                    err_msg = (
                        f"Invalid combination of trend_type and seasonal_type:"
                        f" {trend_type}, {seasonal_type}"
                    )
                    raise ValueError(err_msg)
            ts_fitted = np.concatenate((ts_fitted, forecast))

        return ts_fitted[start:end]

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
            trend_type=trend_type,
            seasonal_type=seasonal_type,
            period=period,
        )

    @staticmethod
    def _tes_add_add(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        trend_type: str,
        seasonal_type: str,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts,
            period,
            trend_type,
            seasonal_type,
        )
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] - s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] + b[i - 1]
            )
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] - lvl[i - 1] - b[i - 1])
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
        trend_type: str,
        seasonal_type: str,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts,
            period,
            trend_type,
            seasonal_type,
        )
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] / s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] + b[i - 1]
            )
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] / (lvl[i - 1] - b[i - 1]))
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
        trend_type: str,
        seasonal_type: str,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts,
            period,
            trend_type,
            seasonal_type,
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
        trend_type: str,
        seasonal_type: str,
        period: int,
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lvl, b, s = TripleExponentialSmoothing.initialise_arrays(
            ts,
            period,
            trend_type,
            seasonal_type,
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
