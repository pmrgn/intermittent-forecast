"""Methods for forecasting time series using Double Exponential Smoothing."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import (
    BaseForecaster,
    TSArray,
    TSInput,
)
from intermittent_forecast.error_metrics import ErrorMetricRegistry


class SmoothingType(Enum):
    """Enum for smoothing modes."""

    ADD = "additive"
    MUL = "multiplicative"


class FittedModelResult(NamedTuple):
    """TypedDict for the results after fitting the model."""

    alpha: float
    beta: float
    ts_base: TSArray
    ts_fitted: TSArray
    lvl_final: float
    trend_final: float


class DoubleExponentialSmoothing(BaseForecaster):
    """Double Exponential Smoothing."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._fitted_model_result: FittedModelResult | None = None

    def fit(
        self,
        ts: TSInput,
        alpha: float | None = None,
        beta: float | None = None,
        optimisation_metric: str | None = None,
    ) -> DoubleExponentialSmoothing:
        """Fit the model to the time-series.

        Args:
            ts (ArrayLike): Time series to fit the model to. Must be
                1-dimensional and contain at least two non-zero values. If
                using multiplicative smoothing, the time series must be
                entirely positive.
            alpha (float, optional): Level smoothing factor in the range
                [0,1]. Values closer to 1 will favour recent demand. If not
                set, the value will be optimised.
            beta (float, optional): Trend smoothing factor in the range [0, 1].
                Values closer to 1 will favour recent demand. If not set, the
                value will be optimised.
            optimisation_metric (str, optional): Metric to use when optimising
                for alpha and beta. Options are 'MAR', 'MAE', 'MSE', 'MSR',
                'PIS'. Defaults to 'MSE'. The selected metric is used when
                comparing the error between the time series and the fitted
                in-sample forecast.

        Returns:
        -------
        self : Fitted model instance.

        """
        # Validate the time series.
        ts = utils.validate_time_series(ts)

        # Validate any provided smoothing parameters.
        for param, param_str in zip([alpha, beta], ["alpha", "beta"]):
            if param is not None:
                utils.validate_float_within_inclusive_bounds(
                    name=param_str,
                    value=param,
                    min_value=0,
                    max_value=1,
                )

        # Optimise for any smoothing parameters not povided.
        if alpha is None or beta is None:
            error_metric_func = ErrorMetricRegistry.get(
                optimisation_metric or "MSE",
            )

            alpha, beta = DoubleExponentialSmoothing._find_optimal_parameters(
                ts=ts,
                alpha=alpha,
                beta=beta,
                error_metric_func=error_metric_func,
            )

        lvl_final, trend_final, ts_fitted = (
            DoubleExponentialSmoothing._compute_exponential_smoothing(
                alpha=alpha,
                beta=beta,
                ts=ts,
            )
        )
        self._fitted_model_result = FittedModelResult(
            alpha=alpha,
            beta=beta,
            ts_base=ts,
            ts_fitted=ts_fitted,
            lvl_final=lvl_final,
            trend_final=trend_final,
        )

        return self

    def forecast(
        self,
        start: int,
        end: int,
    ) -> TSArray:
        """Forecast the time series using the fitted parameters.

        Args:
            start (int): Start index of the forecast (inclusive).
            end (int): End index of the forecast (inclusive).

        Returns:
            np.ndarray: Forecasted values.


        """
        # Get the fitted model result
        fitted_params = self._get_fit_result_if_found()
        ts_fitted = fitted_params.ts_fitted

        # Determine the forecasting horizon if required
        h = end - len(fitted_params.ts_base) + 1
        if h > 0:
            forecast = [
                fitted_params.lvl_final + i * fitted_params.trend_final
                for i in range(1, h + 1)
            ]

            # Append the out of sample forecast to the fitted values.
            ts_fitted = np.concatenate((ts_fitted, np.array(forecast)))

        return ts_fitted[start : end + 1]

    def get_fit_result(self) -> dict[str, Any]:
        """Return the a dictionary of results if model has been fit."""
        return self._get_fit_result_if_found()._asdict()

    def _get_fit_result_if_found(
        self,
    ) -> FittedModelResult:
        """Private method for getting the results after fitting the model."""
        if not self._fitted_model_result or not isinstance(
            self._fitted_model_result,
            FittedModelResult,
        ):
            err_msg = (
                "Model has not been fitted yet. Call the `fit` method first."
            )
            raise RuntimeError(err_msg)

        return self._fitted_model_result

    @staticmethod
    def _compute_exponential_smoothing(
        ts: TSArray,
        alpha: float,
        beta: float,
    ) -> tuple[float, float, TSArray]:
        """Map the smoothing types to the appropriate functions and call."""
        n = len(ts) + 1
        lvl = np.zeros(n)
        b = np.zeros(n)
        # Set the trend naively, i.e. the initial trend is the difference
        # between the first two values in the time series.
        b[0] = ts[1] - ts[0]

        # The initial level has the trend removed as it will be added back in
        # when constructing the fitted time series.
        lvl[0] = ts[0] - b[0]

        for i in range(1, n):
            lvl[i] = alpha * ts[i - 1] + (1 - alpha) * (lvl[i - 1] + b[i - 1])
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]

        # Construct the fitted time series.
        ts_fitted = lvl + b

        return lvl[-1], b[-1], ts_fitted[:-1]

    @staticmethod
    def _find_optimal_parameters(
        ts: TSArray,
        error_metric_func: Callable[..., float],
        alpha: float | None,
        beta: float | None,
    ) -> tuple[float, float]:
        """Find the optimal smoothing parameters that minimise the error."""
        # Set the bounds for the smoothing parameters. If values have been
        # passed, then the bounds will be locked at that value. Else they are
        # set at (0,1).
        alpha_bounds = (alpha or 0, alpha or 1)
        beta_bounds = (beta or 0, beta or 1)

        # Set the initial guess as the midpoint of the bounds.
        initial_guess = np.array([sum(alpha_bounds) / 2, sum(beta_bounds) / 2])
        min_err = optimize.minimize(
            DoubleExponentialSmoothing._cost_function,
            initial_guess,
            args=(ts, error_metric_func),
            bounds=[alpha_bounds, beta_bounds],
        )
        optimal_alpha, optimal_beta = min_err.x

        return optimal_alpha, optimal_beta

    @staticmethod
    def _cost_function(
        params: npt.NDArray[np.float64],
        /,
        ts: TSArray,
        error_metric_func: Callable[..., float],
    ) -> float:
        """Calculate the error between actual and fitted time series."""
        alpha, beta = params
        *_, ts_fitted = (
            DoubleExponentialSmoothing._compute_exponential_smoothing(
                ts=ts,
                alpha=alpha,
                beta=beta,
            )
        )
        return error_metric_func(ts, ts_fitted)
