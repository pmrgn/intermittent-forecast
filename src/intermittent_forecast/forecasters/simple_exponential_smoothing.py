"""Methods for forecasting time series using Simple Exponential Smoothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast.core import utils
from intermittent_forecast.core.error_metrics import ErrorMetricRegistry
from intermittent_forecast.forecasters._base_forecaster import (
    _BaseForecaster,
)

if TYPE_CHECKING:
    from intermittent_forecast.core._types import TSArray, TSInput


class _FittedModelResult(NamedTuple):
    """TypedDict for the results after fitting the model."""

    alpha: float
    ts_base: TSArray
    ts_fitted: TSArray
    lvl_final: float


class SimpleExponentialSmoothing(_BaseForecaster):
    """A class for forecasting time series using Simple Exponential Smoothing.

    Simple Exponential Smoothing (`SES`) is a time series forecasting method
    for data that does not exhibit trend or seasonal patterns. It
    applies a weighted average to past observations, where the weights decay
    exponentially with time.

    When fitting, the smoothing factor alpha can be specified manually or be
    automatically optimised. If not provided, alpha is optimised by minimising
    the error between the fitted and actual values of the time series. The
    error minimization is based on a chosen `optimisation_metric`, which
    defaults to the Mean Squared Error (`MSE`). Other available metrics include
    Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), and Mean Squared
    Rate (`MSR`).

    Example:
        >>> # Initialise an instance of SimpleExponentialSmoothing, fit a time
        >>> # series and create a forecast.
        >>> from intermittent_forecast.forecasters import SimpleExponentialSmoothing
        >>> ts = [40, 28, 35, 41, 33, 21, 37, 20]
        >>> ses = SimpleExponentialSmoothing().fit(ts=ts, alpha=0.3)
        >>> ses.forecast(start=0, end=8)
        array([40.       , 40.       , 36.4      , 35.98     , 37.486    ,
               36.1402   , 31.59814  , 33.218698 , 29.2530886])

        >>> # Smoothing parameters can instead be optimised with a chosen
        >>> # error metric.
        >>> ses = SimpleExponentialSmoothing()
        >>> ses = ses.fit(ts=ts, optimisation_metric="MSR")
        >>> ses.forecast(start=0, end=8)
        array([40.        , 40.        , 36.1204974 , 35.75824969, 37.45286502,
               36.0132899 , 31.15961513, 33.04776416, 28.82952791])

        >>> # Access a dict of the fitted values, get smoothing parameter alpha.
        >>> result = ses.get_fit_result()
        >>> result["alpha"]
        0.32329188326949737

    """  # noqa: E501

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._fitted_model_result: _FittedModelResult | None = None

    def fit(
        self,
        ts: TSInput,
        alpha: float | None = None,
        optimisation_metric: str | None = None,
    ) -> SimpleExponentialSmoothing:
        """Fit the model to the time-series.

        Args:
            ts: ArrayLike
                Time series to fit the model to. Must be 1-dimensional and
                contain at least two non-zero values. If using multiplicative
                smoothing, the time series must be entirely positive.
            alpha: float, optional
                Level smoothing factor in the range [0, 1]. Values closer to 1
                will favour recent demand. If not set, the value will be
                optimised.
            optimisation_metric: {'MAR', 'MAE', 'MSE', 'MSR', 'PIS'},
                default='MSE' Metric to use when optimising for alpha and beta.
                The selected metric is used when comparing the error between
                the time series and the fitted in-sample forecast.

        Returns:
            self: SimpleExponentialSmoothing
                Fitted model instance.

        """
        # Validate the time series.
        ts = utils.validate_time_series(ts)

        if alpha is None:
            # Optimise for the smoothing parameter.
            error_metric_func = ErrorMetricRegistry.get(
                optimisation_metric or "MSE",
            )
            alpha = SimpleExponentialSmoothing._find_optimal_alpha(
                ts=ts,
                error_metric_func=error_metric_func,
            )
        else:
            # Validate the provided smoothing parameter.
            alpha = utils.validate_float_within_inclusive_bounds(
                name="alpha",
                value=alpha,
                min_value=0,
                max_value=1,
            )

        lvl_final, ts_fitted = (
            SimpleExponentialSmoothing._compute_exponential_smoothing(
                alpha=alpha,
                ts=ts,
            )
        )

        self._fitted_model_result = _FittedModelResult(
            alpha=alpha,
            ts_base=ts,
            ts_fitted=ts_fitted,
            lvl_final=lvl_final,
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
            # The out of sample forecast is the final value for the level.
            forecast = [fitted_params.lvl_final] * h

            # Append the out of sample forecast to the fitted values.
            ts_fitted = np.concatenate((ts_fitted, np.array(forecast)))

        return ts_fitted[start : end + 1]

    def get_fit_result(self) -> dict[str, Any]:
        """Return the a dictionary of results if model has been fit."""
        return self._get_fit_result_if_found()._asdict()

    def _get_fit_result_if_found(
        self,
    ) -> _FittedModelResult:
        """Private method for getting the results after fitting the model."""
        if not self._fitted_model_result or not isinstance(
            self._fitted_model_result,
            _FittedModelResult,
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
    ) -> tuple[float, TSArray]:
        """Compute Simple Exponential Smoothing on the series."""
        n = len(ts) + 1
        forecast = np.zeros(n)
        forecast[0] = ts[0]
        for i in range(1, n):
            forecast[i] = alpha * ts[i - 1] + (1 - alpha) * forecast[i - 1]

        return forecast[-1], forecast[:-1]

    @staticmethod
    def _find_optimal_alpha(
        ts: TSArray,
        error_metric_func: Callable[..., float],
    ) -> float:
        """Find the optimal smoothing parameters that minimise the error."""
        alpha_bounds = (0, 1)

        # Set the initial guess as the midpoint of the bounds.
        initial_guess = np.array([sum(alpha_bounds) / 2])

        min_err = optimize.minimize(
            SimpleExponentialSmoothing._cost_function,
            initial_guess,
            args=(ts, error_metric_func),
            bounds=[alpha_bounds],
        )

        return utils.validate_float_within_inclusive_bounds(
            name="alpha",
            value=min_err.x[0],
            min_value=0,
            max_value=1,
        )

    @staticmethod
    def _cost_function(
        alpha: npt.NDArray[np.float64],
        /,
        ts: TSArray,
        error_metric_func: Callable[..., float],
    ) -> float:
        """Calculate the error between actual and fitted time series."""
        *_, ts_fitted = (
            SimpleExponentialSmoothing._compute_exponential_smoothing(
                ts=ts,
                alpha=alpha[0],
            )
        )
        return error_metric_func(ts, ts_fitted)
