"""Methods for forecasting intermittent time series using Croston's method."""

from __future__ import annotations

from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast import utils
from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import ErrorMetricRegistry


class FittedParams(TypedDict):
    """TypedDict for fitted parameters."""

    alpha: float
    beta: float
    ts_fitted: npt.NDArray[np.float64]


class CrostonVariant(BaseForecaster):
    """Base class for Croston variants."""

    requires_bias_correction = False

    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Forecast the time series using the fitted parameters."""
        if not isinstance(self._fitted_params, dict):
            err_msg = "Model has not been fitted yet."
            raise TypeError(err_msg)

        start = utils.validate_non_negative_integer(start, name="start")
        end = utils.validate_positive_integer(end, name="end")
        forecast = self._fitted_params.get("ts_fitted")

        if len(forecast) < end:
            # Append with the out of sample forecast
            forecast = np.concatenate(
                (forecast, np.full(end - len(forecast), forecast[-1])),
            )

        return forecast[start:end]

    def _fit(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        metric: str = "MSE",
    ) -> None:
        """Fit the model to the time-series."""
        if alpha is None or beta is None:
            alpha, beta = self._optimise_and_set_parameters(
                self.get_timeseries(),
                metric=metric,
            )
        else:
            alpha = self._validate_float_within_inclusive_bounds(
                name="alpha",
                value=alpha,
                min_value=0,
                max_value=1,
            )
            beta = self._validate_float_within_inclusive_bounds(
                name="beta",
                value=beta,
                min_value=0,
                max_value=1,
            )

        # Get the time series data.
        ts = self.get_timeseries()

        if self.requires_bias_correction:
            bias_correction = self._get_bias_correction_value(beta=beta)
        else:
            bias_correction = 1

        # Compute forecast using Croston's method.
        forecast = self._compute_forecast(
            ts=ts,
            alpha=alpha,
            beta=beta,
            bias_correction=bias_correction,
        )

        # Cache results
        self._fitted_params = FittedParams(
            alpha=alpha,
            beta=beta,
            ts_fitted=forecast,
        )

    @staticmethod
    def _compute_forecast(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        bias_correction: float = 1,
    ) -> npt.NDArray[np.float64]:
        """Compute Croston's method."""
        # Perform croston's method.
        non_zero_demand = CrostonVariant._get_nonzero_demand_array(ts)
        p_idx = CrostonVariant._get_nonzero_demand_indices(ts)
        p_diff = CrostonVariant._get_nonzero_demand_intervals(p_idx)

        # Intialise an array for the demand.
        z = CrostonVariant._initialise_array(
            array_length=len(non_zero_demand),
            initial_value=non_zero_demand[0],
        )

        # Intialise an array for the demand intervals.
        p = CrostonVariant._initialise_array(
            array_length=len(non_zero_demand),
            initial_value=float(np.mean(p_diff)),
        )

        # Apply the smoothing rules to the demand and demand intervals.
        for i in range(1, len(z)):
            z[i] = alpha * non_zero_demand[i] + (1 - alpha) * z[i - 1]
            p[i] = beta * p_diff[i] + (1 - beta) * p[i - 1]

        # Calculate the forecast.
        f = (z / p) * bias_correction

        # Initialize forecast array
        forecast = np.zeros(len(ts))
        forecast[p_idx] = f

        # Forward fill non-zero forecasted demand values
        forecast = CrostonVariant.forward_fill(forecast)

        # Set values before first p_idx to NaN
        forecast[: p_idx[0]] = np.nan

        return np.insert(forecast, 0, np.nan)

    @staticmethod
    def _optimise_and_set_parameters(
        ts: npt.NDArray[np.float64],
        metric: str = "MSE",
    ) -> tuple[float, float]:
        """Optimise the smoothing parameters alpha and beta."""
        error_metric_func = ErrorMetricRegistry.get(metric)

        # Set the bounds for alpha and beta.
        alpha_min, alpha_max = (0, 1)
        beta_min, beta_max = (0, 1)

        # Set the initial guess as the midpoint of the bounds for alpha and
        # beta.
        initial_guess = (
            (alpha_max - alpha_min) / 2,
            (beta_max - beta_min) / 2,
        )
        min_err = optimize.minimize(
            CrostonVariant._cost_function,
            initial_guess,
            args=(ts, error_metric_func),
            bounds=[(alpha_min, alpha_max), (beta_min, beta_max)],
        )
        alpha, beta = min_err.x
        return alpha, beta

    @staticmethod
    def _cost_function(
        params: tuple[float, float],
        ts: npt.NDArray[np.float64],
        error_metric_func: Callable[..., float],
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta = params
        f = CrostonVariant._compute_forecast(
            ts=ts,
            alpha=alpha,
            beta=beta,
        )
        return error_metric_func(ts, f[:-1])

    def _get_bias_correction_value(self, beta: float) -> float:  # noqa: ARG002
        """Return the bias correction value, if applicable."""
        if not self.requires_bias_correction:
            err_msg = "Bias correction is not applicable for this method."
            raise RuntimeError(err_msg)
        err_msg = (
            "Bias correction is not implemented for this method. "
            "Please implement the '_get_bias_correction_value' method."
        )
        raise NotImplementedError(err_msg)

    @staticmethod
    def _get_nonzero_demand_array(
        ts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Get non-zero demand values from the time series."""
        return np.asarray(ts[ts != 0], dtype=np.float64)

    @staticmethod
    def _get_nonzero_demand_indices(
        ts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        """Get indices of non-zero demand values."""
        return np.flatnonzero(ts)

    @staticmethod
    def _get_nonzero_demand_intervals(
        p_idx: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.int_]:
        """Get intervals between non-zero demand values."""
        return np.diff(p_idx, prepend=-1)

    @staticmethod
    def _initialise_array(
        array_length: int,
        initial_value: float,
    ) -> npt.NDArray[np.float64]:
        """Initialise array and set value at the 0th index."""
        array = np.zeros(array_length)
        array[0] = initial_value
        return array

    @staticmethod
    def forward_fill(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward fills zeros in an array with the last non-zero value."""
        mask = arr != 0
        valid = np.where(mask, arr, 0)
        idx = np.where(mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return np.asarray(valid[idx], dtype=np.float64)


class CRO(CrostonVariant):
    """Croston's method."""


class SBA(CrostonVariant):
    """SBA variant of Croston's method."""

    requires_bias_correction = True

    def _get_bias_correction_value(self, beta: float) -> float:
        """Bias correction applicable to the SBA method."""
        return 1 - (beta / 2)


class SBJ(CrostonVariant):
    """SBJ variant of Croston's method."""

    requires_bias_correction = True

    def _get_bias_correction_value(self, beta: float) -> float:
        """Bias correction applicable to the SBJ method."""
        return 1 - (beta / (2 - beta))


class TSB(CrostonVariant):
    """TSB variant of Croston's method."""

    @staticmethod
    def _compute_forecast(
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
        bias_correction: float = 1,
    ) -> npt.NDArray[np.float64]:
        """Perform forecasting using TSB method."""
        n = len(ts)
        p_idx = TSB._get_nonzero_demand_indices(ts)
        z = TSB._initialise_array(
            array_length=n,
            initial_value=ts[p_idx[0]],
        )

        p = TSB._initialise_array(
            array_length=n,
            initial_value=len(p_idx) / n,
        )

        # Update rules are dependent on whether there is a non-zero demand.
        for i in range(1, n):
            if ts[i] > 0:
                z[i] = alpha * ts[i] + (1 - alpha) * z[i - 1]
                p[i] = beta + (1 - beta) * p[i - 1]
            else:
                z[i] = z[i - 1]
                p[i] = (1 - beta) * p[i - 1]

        forecast = (p * z) * bias_correction

        # Offset the forecast by 1
        return np.insert(forecast, 0, np.nan)
