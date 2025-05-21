"""Croston's method for forecasting intermittent time series."""

from __future__ import annotations

import itertools
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from intermittent_forecast.core import error_metrics, utils
from intermittent_forecast.forecasters._base_forecaster import (
    _BaseForecaster,
)

if TYPE_CHECKING:
    from intermittent_forecast.core._types import TSArray, TSInput


class _CrostonVariant(Enum):
    """Enum for Croston variants."""

    CRO = "cro"
    SBA = "sba"
    SBJ = "sbj"
    TSB = "tsb"


class _FitOptimisationConfig(NamedTuple):
    """Config to use when fitting the model with optimisation."""

    ts: TSArray
    variant: _CrostonVariant
    optimisation_metric: error_metrics.ErrorMetricFunc
    alpha: float | None
    beta: float | None


class _FittedModelResult(NamedTuple):
    """TypedDict for results after fitting the model."""

    alpha: float
    beta: float
    ts_base: TSArray
    ts_fitted: TSArray


class Croston(_BaseForecaster):
    """A class for fitting and forecasting intermittent time series.

    Croston's method deconstructs a time series into separate demand and
    interval series. It applies a Simple Exponential Smoothing (`SES`)
    algorithm to the both series, then reconstructs this back into a forecast.
    This class provides a convenient way to use Croston's method for
    forecasting. It provides a simple interface to fit the model to a time
    series, and to generate forecasts.

    When fitting, variants of Croston's can be selected, including
    Syntetos-Boylan Approximation (`SBA`), Shale-Boylan-Johnston (`SBJ`), and
    Teunter-Syntetos-Babai (`TSB`). Each allows the selection of the smoothing
    factors, alpha and beta. If not specified, alpha and beta will be optimised
    through minimising the error between the fitted time series and the
    original time series. The error is specified by the `optimisation_metric`,
    which defaults to the Mean Squared Error (`MSE`), but can also be set to
    the Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), Mean Squared
    Rate (`MSR`), or Periods in Stock (`PIS`).

    Example:
        >>> # Initialise an instance of Croston, fit a time series and create
        >>> # a forecast.
        >>> from intermittent_forecast.forecasters import Croston
        >>> ts = [0, 3, 0, 4, 0, 0, 0, 2, 0]
        >>> cro = Croston().fit(ts=ts, alpha=0.5, beta=0.2)
        >>> cro.forecast(start=0, end=10)
        array([       nan,        nan, 1.125     , 1.125     , 1.38157895,
               1.38157895, 1.38157895, 1.38157895, 0.97287736, 0.97287736])

        >>> # Smoothing parameters can instead be optimised with a chosen
        >>> # error metric.
        >>> cro = Croston().fit(ts=ts, optimisation_metric="MSR")
        >>> cro.forecast(start=0, end=10)
        array([       nan,        nan, 1.125     , 1.125     , 1.18876368,
               1.18876368, 1.18876368, 1.18876368, 1.05619218, 1.05619218])

        >>> # Access a dict of the fitted values, get smoothing parameter beta.
        >>> result = cro.get_fit_result()
        >>> result["beta"]
        0.2145546005097181


    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._fitted_model_result: _FittedModelResult | None = None

    def fit(
        self,
        ts: TSInput,
        variant: str = "CRO",
        alpha: float | None = None,
        beta: float | None = None,
        optimisation_metric: str = "MSE",
    ) -> Croston:
        """Fit the model to the time-series.

        Args:
            ts (ArrayLike): Time series to fit the model to. Must be
                1-dimensional and contain at least two non-zero values.
            variant (str): The Croston variant to use. Options are "SBA",
                "SBJ", "TSB", or default "CRO". These correspond to
                Syntetos-Boylan Approximation, Shale-Boylan-Johnston,
                Teunter-Syntetos-Babai, and the original Croston method,
                respectively.
            alpha (float, optional): Demand smoothing factor in the range
                [0,1]. Values closer to 1 will favour recent demand. If not
                set, the value will be optimised. Defaults to None.
            beta (float, optional): Interval smoothing factor in the range
                [0,1]. Values closer to 1 will favour recent intervals.  If not
                set, the value will be optimised. Defaults to None.
            optimisation_metric (str, optional): Metric to use when optimising
                for alpha and beta. The selected metric is used when comparing
                the error between the time series and the fitted in-sample
                forecast. Defaults to 'MSE'.

        Returns:
            self (Croston): Fitted model instance.

        """
        # Validate time series.
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

        # Get the enum member from the variant string.
        variant_member = utils.get_enum_member_from_str(
            member_str=variant,
            enum_class=_CrostonVariant,
            member_name="variant",
        )

        # Optimise for any smoothing parameters not povided.
        if alpha is None or beta is None:
            optimsation_metric_func = error_metrics.ErrorMetricRegistry.get(
                optimisation_metric or "MSE",
            )

            fit_optimisation_config = _FitOptimisationConfig(
                ts=ts,
                variant=variant_member,
                optimisation_metric=optimsation_metric_func,
                alpha=alpha,
                beta=beta,
            )

            alpha, beta = self._find_optimal_parameters(
                fit_optimisation_config=fit_optimisation_config,
            )

        if variant_member == _CrostonVariant.TSB:
            # Compute forecast using TSB if required.
            forecast = self._compute_tsb_forecast(
                ts=ts,
                alpha=alpha,
                beta=beta,
            )

        else:
            # Compute forecast using Croston's method for other variants.
            forecast = self._compute_croston_forecast(
                ts=ts,
                alpha=alpha,
                beta=beta,
                bias_correction=self._get_bias_correction_value(
                    variant=variant_member,
                    beta=beta,
                ),
            )

        # Cache results
        self._fitted_model_result = _FittedModelResult(
            alpha=alpha,
            beta=beta,
            ts_base=ts,
            ts_fitted=forecast,
        )

        return self

    def forecast(
        self,
        start: int,
        end: int,
    ) -> TSArray:
        """Forecast the time series using the fitted parameters.

        The forecast is computed by appending the out of sample forecast to the
        fitted values.

        Args:
            start (int): Start index of the forecast (inclusive).
            end (int): End index of the forecast (inclusive).

        Returns:
            np.ndarray: Forecasted values.

        """
        start = utils.validate_non_negative_integer(start, name="start")
        end = utils.validate_positive_integer(end, name="end")

        # Get the fitted model result.
        fitted_values = self._get_fit_result_if_found()
        forecast = fitted_values.ts_fitted

        if len(forecast) < end:
            # Append with the out of sample forecast.
            forecast = np.concatenate(
                (forecast, np.full(end - len(forecast), forecast[-1])),
            )

        return forecast[start:end]

    def get_fit_result(self) -> dict[str, Any]:
        """Return a dictionary of results if model has been fit."""
        return self._get_fit_result_if_found()._asdict()

    def _get_fit_result_if_found(
        self,
    ) -> _FittedModelResult:
        """Private method for getting fitted results."""
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
    def _compute_croston_forecast(
        ts: TSArray,
        alpha: float,
        beta: float,
        bias_correction: float,
    ) -> TSArray:
        """Compute Croston's method."""
        non_zero_demand = Croston._get_nonzero_demand_array(ts)
        p_idx = Croston._get_nonzero_demand_indices(ts)
        p_diff = Croston._get_nonzero_demand_intervals(p_idx)

        # Intialise an array for the demand.
        z = Croston._initialise_array(
            array_length=len(non_zero_demand),
            initial_value=non_zero_demand[0],
        )

        # Intialise an array for the demand intervals.
        p = Croston._initialise_array(
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
        forecast = Croston._forward_fill(forecast)

        # Set values before first p_idx to NaN
        forecast[: p_idx[0]] = np.nan

        return np.insert(forecast, 0, np.nan)

    @staticmethod
    def _compute_tsb_forecast(
        ts: TSArray,
        alpha: float,
        beta: float,
    ) -> TSArray:
        """Perform forecasting using TSB method."""
        n = len(ts)
        p_idx = Croston._get_nonzero_demand_indices(ts)
        z = Croston._initialise_array(
            array_length=n,
            initial_value=ts[p_idx[0]],
        )

        p = Croston._initialise_array(
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

        # Offset the forecast by 1
        return np.insert(forecast, 0, np.nan)

    @staticmethod
    def _find_optimal_parameters(
        fit_optimisation_config: _FitOptimisationConfig,
    ) -> tuple[float, float]:
        """Optimise the smoothing parameters alpha and beta."""
        alpha = fit_optimisation_config.alpha
        beta = fit_optimisation_config.beta
        error_metric_func = fit_optimisation_config.optimisation_metric
        variant = fit_optimisation_config.variant
        ts = fit_optimisation_config.ts

        # Set the bounds for the smoothing parameters. If values have been
        # passed, then the bounds will be locked at that value. Else they are
        # set at (0,1).
        alpha_bounds = (alpha or 0, alpha or 1)
        beta_bounds = (beta or 0, beta or 1)

        # Do a quick grid search to find the best initial values.
        initial_guess = Croston._find_best_initial_values(
            fit_optimisation_config,
        )

        min_err = optimize.minimize(
            Croston._cost_function,
            initial_guess,
            args=(ts, error_metric_func, variant),
            bounds=[alpha_bounds, beta_bounds],
        )
        optimal_alpha, optimal_beta = min_err.x
        return optimal_alpha, optimal_beta

    @staticmethod
    def _find_best_initial_values(
        fit_optimisation_config: _FitOptimisationConfig,
    ) -> tuple[float, float]:
        """Do a grid search to find the initial values."""
        # Lock values if set, else create a grid.
        alpha = fit_optimisation_config.alpha
        alpha_grid = [alpha] if alpha else [0, 0.5, 1]
        beta = fit_optimisation_config.beta
        beta_grid = [beta] if beta else [0, 0.5, 1]
        parameter_grid_search = list(itertools.product(alpha_grid, beta_grid))

        # Find the best initial values
        best_error = np.inf
        best_params = (0.5, 0.5)
        for params in parameter_grid_search:
            error = Croston._cost_function(
                np.array(params),
                ts=fit_optimisation_config.ts,
                error_metric_func=fit_optimisation_config.optimisation_metric,
                variant=fit_optimisation_config.variant,
            )
            if error < best_error:
                best_error = error
                best_params = params

        return best_params

    @staticmethod
    def _cost_function(
        params: npt.NDArray[np.float64],
        /,
        ts: TSArray,
        error_metric_func: Callable[..., float],
        variant: _CrostonVariant,
    ) -> float:
        """Cost function used for optimisation of alpha and beta."""
        alpha, beta = params

        if variant == _CrostonVariant.TSB:
            # Compute using TSB if required.
            f = Croston._compute_tsb_forecast(
                ts=ts,
                alpha=alpha,
                beta=beta,
            )
        else:
            # For other methods, i.e. CRO, SBA, SBJ, compute using Croston and
            # apply bias correction.
            bias_correction = Croston._get_bias_correction_value(
                variant=variant,
                beta=beta,
            )
            f = Croston._compute_croston_forecast(
                ts=ts,
                alpha=alpha,
                beta=beta,
                bias_correction=bias_correction,
            )

        return error_metric_func(ts, f[:-1])

    @staticmethod
    def _get_bias_correction_value(
        variant: _CrostonVariant,
        beta: float,
    ) -> float:
        """Return the bias correction value, if applicable."""
        bias_correction = 1.0
        match variant:
            case _CrostonVariant.CRO:
                pass
            case _CrostonVariant.SBA:
                bias_correction = 1 - (beta / 2)
            case _CrostonVariant.SBJ:
                bias_correction = 1 - (beta / (2 - beta))

        return bias_correction

    @staticmethod
    def _get_nonzero_demand_array(
        ts: TSArray,
    ) -> TSArray:
        """Get non-zero demand values from the time series."""
        return np.asarray(ts[ts != 0], dtype=np.float64)

    @staticmethod
    def _get_nonzero_demand_indices(
        ts: TSArray,
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
    ) -> TSArray:
        """Initialise array and set value at the 0th index."""
        array = np.zeros(array_length)
        array[0] = initial_value
        return array

    @staticmethod
    def _forward_fill(arr: TSArray) -> TSArray:
        """Forward fills zeros in an array with the last non-zero value."""
        mask = arr != 0
        valid = np.where(mask, arr, 0)
        idx = np.where(mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return np.asarray(valid[idx], dtype=np.float64)
