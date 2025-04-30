"""Methods for forecasting intermittent time series using Croston's method."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast import error_metrics
from intermittent_forecast.base_forecaster import BaseForecaster

# Define the mapping dictionary for error metric functions
METRIC_FUNCTIONS = {
    "MAE": error_metrics.mae,
    "MSE": error_metrics.mse,
    "MAR": error_metrics.mar,
    "MSR": error_metrics.msr,
    "PIS": error_metrics.pis,
}


def get_metric_function(metric_name: str) -> Callable[..., float]:
    """Retrieve a metric function by its name."""
    try:
        return METRIC_FUNCTIONS[metric_name.upper()]
    except KeyError:
        error_message = (
            f"Unknown metric '{metric_name}'. Available options: "
            f"{list(METRIC_FUNCTIONS.keys())}"
        )
        raise ValueError(error_message) from None


class CrostonVariant(BaseForecaster):
    """Base class for Croston variants."""

    requires_bias_correction = False

    def __init__(self) -> None:
        """Initialise the Croston variant."""
        super().__init__()
        self.alpha: float | None = None
        self.beta: float | None = None

    def _fit(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        metric: str = "MSE",
    ) -> None:
        """Fit the model to the time-series."""
        if alpha is None or beta is None:
            self._optimise_and_set_parameters(metric=metric)
        else:
            # TODO: Check code duplication
            self.alpha = self._validate_float_within_inclusive_bounds(
                name="alpha",
                value=alpha,
                min_value=0,
                max_value=1,
            )
            self.beta = self._validate_float_within_inclusive_bounds(
                name="beta",
                value=beta,
                min_value=0,
                max_value=1,
            )

    def _optimise_and_set_parameters(self, metric: str = "MSE") -> None:
        """Optimise the smoothing parameters alpha and beta."""
        _metric = get_metric_function(metric)

        # Set the bounds for alpha and beta.
        alpha_min = 0
        alpha_max = 1
        beta_min = 0
        beta_max = 1

        # Set the initial guess as the midpoint of the bounds for alpha and
        # beta.
        initial_guess = (
            (alpha_max - alpha_min) / 2,
            (beta_max - beta_min) / 2,
        )
        min_err = optimize.minimize(
            self._cost_function,
            initial_guess,
            args=(_metric,),
            bounds=[(alpha_min, alpha_max), (beta_min, beta_max)],
        )
        self.alpha, self.beta = min_err.x

    def _cost_function(
        self,
        params: tuple[float, float],
        metric_function: Callable[..., float],
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta = params
        f = self.forecast(
            alpha=alpha,
            beta=beta,
        )
        error = metric_function(self._ts, f[:-1])
        return error

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

    def forecast(
        self,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Perform forecasting for CRO, SBA, and SBJ methods."""
        # Get the time series data.
        ts = self.get_timeseries()

        alpha = alpha or self.alpha
        beta = beta or self.beta
        if alpha is None or beta is None:
            err_msg = (
                "Alpha and beta must be set before calling forecast, or call "
                "the fit() method automatically select values."
            )
            raise ValueError(err_msg)

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

        non_zero_demand = self._get_nonzero_demand_array(ts)
        p_idx = self._get_nonzero_demand_indices(ts)
        p_diff = self._get_nonzero_demand_intervals(p_idx)

        # Intialise an array for the demand.
        z = self._initialise_array(
            array_length=len(non_zero_demand),
            initial_value=non_zero_demand[0],
        )

        # Intialise an array for the demand intervals.
        p = self._initialise_array(
            array_length=len(non_zero_demand),
            initial_value=float(np.mean(p_diff)),
        )

        # Apply the smoothing rules to the demand and demand intervals.
        for i in range(1, len(z)):
            z[i] = alpha * non_zero_demand[i] + (1 - alpha) * z[i - 1]
            p[i] = beta * p_diff[i] + (1 - beta) * p[i - 1]

        # Calculate the forecast.
        f = z / p

        # Apply bias correction if required, e.g., for SBA and SBJ methods. For
        # CRO, the bias correction is 1, so it does not affect the forecast.
        if self.requires_bias_correction:
            f *= self._get_bias_correction_value(beta)

        # Initialize forecast array
        forecast = np.zeros(len(ts))
        forecast[p_idx] = f

        # Forward fill non-zero forecasted demand values
        forecast = self.forward_fill(forecast)

        # Set values before first p_idx to NaN
        forecast[: p_idx[0]] = np.nan

        # Offset the forecast, the first value is set to NaN.
        return np.insert(forecast, 0, np.nan)

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
        return valid[idx]


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

    def forecast(
        self,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Perform forecasting using TSB method."""
        # Get the time series data.
        ts = self.get_timeseries()

        alpha = alpha or self.alpha
        beta = beta or self.beta
        if alpha is None or beta is None:
            err_msg = (
                "Alpha and beta must be set before calling forecast, or call"
                "the fit() method automatically select values."
            )
            raise ValueError(err_msg)

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
        n = len(ts)
        p_idx = self._get_nonzero_demand_indices(ts)
        z = self._initialise_array(
            array_length=n,
            initial_value=ts[p_idx[0]],
        )

        p = self._initialise_array(
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

        forecast = p * z

        # Offset the forecast to match the original time series.
        return np.insert(forecast, 0, np.nan)
