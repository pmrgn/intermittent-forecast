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

    def __init__(
        self,
        ts: list[float] | npt.NDArray[np.float64],
        alpha: float = 0.1,
        beta: float = 0.1,
    ) -> None:
        """Initialise the Croston variant."""
        super().__init__(ts)
        # TODO: Remove parameters from constructor.
        self.alpha = self._validate_smoothing_parameter(
            value=alpha,
            name="alpha",
        )
        self.beta = self._validate_smoothing_parameter(
            value=beta,
            name="beta",
        )

    def forecast(self) -> npt.NDArray[np.float64]:
        """Forecast the time series using Croston's method."""
        # TODO: Allow parameters here, instead of being passed in the constructor.
        return self._forecast(
            ts=self.ts,
            alpha=self.alpha,
            beta=self.beta,
        )

    def fit(self, metric: str = "MSE") -> None:
        """Optimise the smoothing parameters alpha and beta."""
        _metric = get_metric_function(metric)
        initial_guess = self.alpha, self.beta  # Initial guess for alpha, beta
        min_err = optimize.minimize(
            self._cost_function,
            initial_guess,
            args=(_metric,),
            bounds=[(0, 1), (0, 1)],
        )
        print(min_err.x)
        self.alpha, self.beta = min_err.x

    def _cost_function(
        self,
        params: tuple[float, float],
        metric_function: Callable[..., float],
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta = params
        f = self._forecast(
            ts=self._ts,
            alpha=alpha,
            beta=beta,
        )
        error = metric_function(self._ts, f[:-1])
        print(f"Alpha: {alpha}, Beta: {beta}, Error: {error}")
        return error

    def _get_bias_correction_value(self) -> float:
        """Apply bias correction to forecast when required.

        Croston adaptations such as SBA and SBJ require bias correction, which
        is a value multiplied with the forecast.
        """
        return 1

    def _forecast(
        self,
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.float64]:
        """Perform forecasting for CRO, SBA, and SBJ methods."""
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
        f *= self._get_bias_correction_value()

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

    @staticmethod
    def _validate_smoothing_parameter(
        value: float,
        name: str,
    ) -> float:
        """Validate the smoothing parameter."""
        if not isinstance(value, (float, int)):
            err_msg = (
                f"Invalid value set for parameter: `{name}`. Must be type"
                f" int or float, instead got type: `{type(value)}`"
            )
            raise TypeError(err_msg)

        if not 0 <= value <= 1:
            err_msg = (
                f"Invalid value set for parameter: `{name}`. Must be in"
                f" the range (0, 1). Instead got value: `{value}`"
            )
            raise ValueError(err_msg)

        return value


class CRO(CrostonVariant):
    """Croston's method."""


class SBA(CrostonVariant):
    """SBA variant of Croston's method."""

    def __init__(
        self,
        ts: list[float] | npt.NDArray[np.float64],
        alpha: float = 0.1,
        beta: float = 0.05,
    ) -> None:
        """Initialise the SBA variant of Croston's method."""
        super().__init__(ts, alpha, beta)

    def _get_bias_correction_value(self) -> float:
        """Bias correction applicable to the SBA method."""
        return 1 - (self.beta / 2)


class SBJ(CrostonVariant):
    """SBJ variant of Croston's method."""

    def _get_bias_correction_value(self) -> float:
        """Bias correction applicable to the SBJ method."""
        return 1 - (self.beta / (2 - self.beta))


class TSB(CrostonVariant):
    """TSB variant of Croston's method."""

    def _forecast(
        self,
        ts: npt.NDArray[np.float64],
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.float64]:
        """Perform forecasting using TSB method."""
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
