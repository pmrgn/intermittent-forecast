# type: ignore
from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast.base_forecaster import BaseForecaster
from intermittent_forecast.error_metrics import mae, mar, mse, msr, pis


class DefaultParams(NamedTuple):
    name: str
    default_alpha: float | None
    default_beta: float | None


class CrostonVariant(Enum):
    CRO = DefaultParams(name="CRO", default_alpha=0.1, default_beta=0.1)
    SBA = DefaultParams(name="SBA", default_alpha=0.1, default_beta=0.05)
    SBJ = DefaultParams(name="SBJ", default_alpha=0.1, default_beta=0.1)
    TSB = DefaultParams(name="TSB", default_alpha=0.1, default_beta=0.1)

    @classmethod
    def from_str(cls, variant: str) -> CrostonVariant:
        """Retrieve an enum member by its name string."""
        try:
            return cls[variant]
        except KeyError:
            msg = (
                f"Invalid Croston Variant: {variant}. Must be one of "
                f"{list(cls.__members__.keys())}"
            )
            raise ValueError(msg) from None


class ErrorMetricNames(NamedTuple):
    name: str
    function: callable


class ErrorMetrics(Enum):
    MAE = ErrorMetricNames(name="MAE", function=mae)
    MSE = ErrorMetricNames(name="MSE", function=mse)
    MAR = ErrorMetricNames(name="MAR", function=mar)
    MSR = ErrorMetricNames(name="MSR", function=msr)
    PIS = ErrorMetricNames(name="PIS", function=pis)

    @classmethod
    def from_str(cls, metric: str) -> ErrorMetrics:
        """Retrieve an enum member by its name string."""
        try:
            return cls[metric]
        except KeyError:
            msg = (
                f"Invalid Error Metric: {metric}. Must be one of "
                f"{list(cls.__members__.keys())}"
            )
            raise ValueError(msg) from None


CrostonVariant.from_str("CRO")


class Croston(BaseForecaster):
    """Forecast demand using Croston's method.

    Parameters
    ----------
    ts : list[float] | npt.NDArray[np.float64]
        Time-series to forecast.

    Raises
    ------
    ValueError
        If `ts` does not contain at least two non-zero values.

    """

    def __init__(
        self,
        ts: list[float] | npt.NDArray[np.float64],
        variant: str = CrostonVariant.CRO.name,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        """Initialise the forecaster.

        Parameters
        ----------
        ts : list[float] | npt.NDArray[np.float64]
            Time-series to forecast.

        Raises
        ------
        ValueError
            If `ts` does not contain at least two non-zero values.

        """
        super().__init__(ts)
        _variant_obj = CrostonVariant.from_str(variant)
        self.variant = _variant_obj.name

        if alpha:
            self.alpha = self._validate_smoothing_parameter("alpha", alpha)
        else:
            self.alpha = _variant_obj.value.default_alpha

        if beta:
            self.beta = self._validate_smoothing_parameter("beta", beta)
        else:
            self.beta = _variant_obj.value.default_beta

    def optimise_variant(self) -> None:
        """Select the forecasting method using squared covariance of non-zero
        demand and mean demand interval
        """
        # Calculate the squared covariance of non-zero demand and mean demand.
        ts_nonzero = self.ts[self.ts != 0]
        p_mean = len(self.ts) / len(ts_nonzero)
        cv2 = (np.std(ts_nonzero, ddof=1) / np.mean(ts_nonzero)) ** 2

        # Set the boundary values for the selection of the optimal variant.
        cv2_boundary = 0.49
        p_mean_boundary = 1.34

        # Select the optimal variant based on the boundary values.
        if cv2 <= cv2_boundary and p_mean <= p_mean_boundary:
            optimimal_variant = CrostonVariant.CRO.name
        else:
            optimimal_variant = CrostonVariant.SBA.name

        self.variant = optimimal_variant

    def optimise_parameters(self, metric: str) -> None:
        """Optimise the smoothing parameters alpha and beta."""
        # TODO validate metric
        _metric = ErrorMetrics.from_str(metric)
        initial_guess = self.alpha, self.beta  # Initial guess for alpha, beta
        min_err = optimize.minimize(
            self._cost_function,
            initial_guess,
            args=(_metric,),
            bounds=[(0, 1), (0, 1)],
        )
        # TODO Validate alpha beta?
        self.alpha, self.beta = min_err.x

    def _cost_function(self, params: tuple[float, float], metric: ErrorMetrics):
        """Cost function used for optimisation of alpha and beta"""
        alpha, beta = params
        f = self._forecast(
            ts=self.ts,
            variant=CrostonVariant.from_str(self.variant),
            alpha=alpha,
            beta=beta,
        )

        return metric.value.function(self.ts, f[:-1])

    def forecast(self):
        """Wrapper."""
        return self._forecast(
            ts=self.ts,
            variant=CrostonVariant.from_str(self.variant),
            alpha=self.alpha,
            beta=self.beta,
        )

    @staticmethod
    def _forecast(
        ts: npt.NDArray[np.float64],
        variant: CrostonVariant,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Perform smoothing on an intermittent time series a forecast array.

        Returns
        -------
        forecast : (N+1,) ndarray
            1-D array of forecasted values

        """
        if variant == CrostonVariant.TSB:
            # Initialise demand array, z, and demand probability, p. The starting
            # value for z is the first non-zero demand value, starting value for p
            # is the inverse of the mean of all intervals
            n = len(ts)
            z = np.zeros(n)
            p = np.zeros(n)
            p_idx = np.flatnonzero(ts)
            p_diff = np.diff(p_idx, prepend=-1)
            z[0] = ts[p_idx[0]]
            p[0] = 1 / np.mean(p_diff)  # Probability of demand occurence

            # Perform TSB
            for i in range(1, n):
                if ts[i] > 0:
                    z[i] = alpha * ts[i] + (1 - alpha) * z[i - 1]
                    p[i] = beta + (1 - beta) * p[i - 1]
                else:
                    z[i] = z[i - 1]
                    p[i] = (1 - beta) * p[i - 1]
            forecast = p * z
            forecast = np.insert(forecast, 0, np.nan)
            return forecast

        # CRO, SBA, SBJ:
        # Initialise arrays for demand, z, and period, p. Starting
        # demand is first non-zero demand value, starting period is
        # mean of all demand intervals
        nz = ts[ts != 0]
        p_idx = np.flatnonzero(ts)
        p_diff = np.diff(p_idx, prepend=-1)
        n = len(nz)
        z = np.zeros(n)
        p = np.zeros(n)
        z[0] = nz[0]
        p[0] = np.mean(p_diff)

        # Perform smoothing on demand and interval arrays
        for i in range(1, n):
            z[i] = alpha * nz[i] + (1 - alpha) * z[i - 1]
            p[i] = beta * p_diff[i] + (1 - beta) * p[i - 1]

        # Create forecast array, apply bias correction if required
        f = z / p
        if variant == CrostonVariant.SBA:
            f *= 1 - (beta / 2)

        elif variant == CrostonVariant.SBJ:
            f *= 1 - (beta / (2 - beta))

        # Return to original time scale by forward filling
        z_idx = np.zeros(len(ts))
        z_idx[p_idx] = p_idx
        z_idx = np.maximum.accumulate(z_idx).astype("int")
        forecast = np.zeros(len(ts))
        forecast[p_idx] = f
        forecast = forecast[z_idx]

        # Starting forecast values up to and including first demand occurence
        # will be np.nan
        forecast[: p_idx[0]] = np.nan
        forecast = np.insert(forecast, 0, np.nan)
        return forecast

    def _validate_smoothing_parameter(
        self,
        parameter: str,
        value: float,
    ) -> None:
        """Validate the smoothing parameter."""
        if not isinstance(value, (float, int)):
            err_msg = (
                f"Invalid value set for parameter: `{parameter}`. Must be type"
                f" int, float or None. Instead got type: `{type(value)}`"
            )
            raise TypeError(err_msg)

        if not 0 <= value <= 1:
            err_msg = (
                f"Invalid value set for parameter: `{parameter}`. Must be in"
                f" the range (0, 1). Instead got value: `{value}`"
            )
            raise ValueError(err_msg)

        return value
