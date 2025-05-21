"""Methods for forecasting time series using Triple Exponential Smoothing."""

from __future__ import annotations

from enum import Enum
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


class _SmoothingType(Enum):
    """Enum for smoothing modes."""

    ADD = "additive"
    MUL = "multiplicative"


class _FittedModelResult(NamedTuple):
    """TypedDict for the results after fitting the model."""

    alpha: float
    beta: float
    gamma: float
    ts_base: TSArray
    ts_fitted: TSArray
    trend_type: _SmoothingType
    seasonal_type: _SmoothingType
    period: int
    lvl_final: float
    trend_final: float
    seasonal_final: TSArray


class TripleExponentialSmoothing(_BaseForecaster):
    """A class for forecasting time series using Triple Exponential Smoothing.

    Triple Exponential Smoothing (`TES`), also referred to as Holt-Winters
    Exponential Smoothing, extends Double Exponential Smoothing by
    incorporating a seasonal component. It is designed for time series data
    that exhibits both trend and seasonality.The method simultaneously smooths
    the level, trend, and seasonal components of the series using exponential
    weighting. This class provides an easy-to-use interface to fit the TES
    model and to generate forecasts that capture both seasonal and trend
    behavior over time.

    The model uses three smoothing parameters: alpha (level), beta (trend), and
    gamma (seasonality). These can be specified manually or optimised
    automatically by minimising the difference between fitted and actual
    values. Seasonal patterns can be either additive or multiplicative, and the
    type should be selected based on the characteristics of the data.

    The `optimisation_metric` defaults to Mean Squared Error (`MSE`), but can
    also be set to Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), or
    Mean Squared Rate (`MSR`), among others.

    Example:
        >>> # Initialise an instance of TripleExponentialSmoothing, fit a time
        >>> # series and create a forecast.
        >>> from intermittent_forecast.forecasters import TripleExponentialSmoothing
        >>> ts = [5, 6, 8, 9,
        ...       6, 8, 7,10,
        ...       8, 8, 9,12]
        >>> tes = TripleExponentialSmoothing().fit(
        ...     ts=ts,
        ...     period=4,
        ...     trend_type="additive",
        ...     seasonal_type="multiplicative",
        ...     alpha=0.3,
        ...     beta=0.1,
        ...     gamma=0.1,
        ... )
        >>> tes.forecast(start=0, end=11) # In-sample forecast
        array([ 5.13392857,  6.26839286,  8.44762143,  9.55915626,  5.30905245,
                6.75937741,  9.74957716, 10.22368613,  5.83750991,  8.01345859,
               10.43690141, 11.79762156])

        >>> # The out of sample forecasts is constructed from the final level,
        >>> # trend and seasonal component values.
        >>> tes.forecast(start=12, end=19)
        array([ 7.10671315,  8.4233535 , 10.77655469, 12.88922365,  7.69870712,
                9.11071018, 11.63835466, 13.89977022])

        >>> # Smoothing parameters can instead be optimised with a chosen
        >>> # error metric.
        >>> tes =TripleExponentialSmoothing().fit(
        ...     ts=ts,
        ...     period=4,
        ...     trend_type="additive",
        ...     seasonal_type="multiplicative",
        ...     optimisation_metric="MAE"
        ... )
        >>> tes.forecast(start=12, end=19)
        array([ 8.30291671,  9.27943232, 10.31641935, 13.4928251 ,  9.43460888,
               10.50254632, 11.6328387 , 15.16134003])

        >>> # Access a dict of the fitted values.
        >>> result = tes.get_fit_result()
        >>> result["alpha"], result["beta"], result["gamma"]
        (0.08364575434612503, 1.0, 0.47060129090469816)

    """  # noqa: E501

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._fitted_model_result: _FittedModelResult | None = None

    def fit(
        self,
        ts: TSInput,
        period: int,
        trend_type: str = "additive",
        seasonal_type: str = "additive",
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        optimisation_metric: str | None = None,
    ) -> TripleExponentialSmoothing:
        """Fit the model to the time-series.

        Args:
            ts (ArrayLike): Time series to fit the model to. Must be
                1-dimensional and contain at least two non-zero values. If
                using multiplicative smoothing, the time series must be
                entirely positive.
            period (int): The period of the seasonal component.
            trend_type (str, optional): The type of trend smoothing to use.
                Options are "additive" or "multiplicative". Defaults to
                "additive". If using multiplicative smoothing, the time series
                must be entirely positive.
            seasonal_type (str, optional): The type of seasonal smoothing to
                use.Options are "additive" or "multiplicative". Defaults to
                "additive".If using multiplicative smoothing, the time series
                must be entirely positive.
            alpha (float, optional): Level smoothing factor in the range
                [0,1]. Values closer to 1 will favour recent demand. If not
                set, the value will be optimised.
            beta (float, optional): Trend smoothing factor in the range [0, 1].
                Values closer to 1 will favour recent demand. If not set, the
                value will be optimised.
            gamma (float, optional): Seasonal smoothing factor in the range
                [0,1]. Values closer to 1 will favour recent demand. If not
                set, the value will be optimised.
            optimisation_metric (str, optional): Metric to use when optimising
                for alpha and beta. Options are 'MAR', 'MAE', 'MSE', 'MSR',
                'PIS'. Defaults to 'MSE'. The selected metric is used when
                comparing the error between the time series and the fitted
                in-sample forecast.

        Returns:
            self (TripleExponentialSmoothing): Fitted model instance.

        """
        # Validate trend and seasonal types, and convert to enum members.
        trend_type_member = utils.get_enum_member_from_str(
            member_str=trend_type,
            enum_class=_SmoothingType,
            member_name="trend_type",
        )
        seasonal_type_member = utils.get_enum_member_from_str(
            member_str=seasonal_type,
            enum_class=_SmoothingType,
            member_name="seasonal_type",
        )

        period = utils.validate_non_negative_integer(
            value=period,
            name="period",
        )

        # Validate the time series.
        ts = utils.validate_time_series(ts)

        # If using multiplicative smoothing, ensure the time series contains
        # only positive values.
        if (
            trend_type_member is _SmoothingType.MUL
            or seasonal_type_member is _SmoothingType.MUL
        ) and not np.all(ts > 0):
            err_msg = (
                "The series must be all greater than 0 for multiplicative "
                "smoothing."
            )
            raise ValueError(err_msg)

        # Validate any provided smoothing parameters.
        for param, param_str in zip(
            [alpha, beta, gamma],
            ["alpha", "beta", "gamma"],
        ):
            if param is not None:
                utils.validate_float_within_inclusive_bounds(
                    name=param_str,
                    value=param,
                    min_value=0,
                    max_value=1,
                )

        # Optimise for any smoothing parameters not povided.
        if alpha is None or beta is None or gamma is None:
            error_metric_func = ErrorMetricRegistry.get(
                optimisation_metric or "MSE",
            )

            alpha, beta, gamma = (
                TripleExponentialSmoothing._find_optimal_parameters(
                    ts=ts,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    error_metric_func=error_metric_func,
                    period=period,
                    trend_type=trend_type_member,
                    seasonal_type=seasonal_type_member,
                )
            )

        lvl_final, trend_final, seasonal_final, ts_fitted = (
            TripleExponentialSmoothing._compute_exponential_smoothing(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                ts=ts,
                period=period,
                trend_type=trend_type_member,
                seasonal_type=seasonal_type_member,
            )
        )
        self._fitted_model_result = _FittedModelResult(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ts_base=ts,
            ts_fitted=ts_fitted,
            trend_type=trend_type_member,
            seasonal_type=seasonal_type_member,
            period=period,
            lvl_final=lvl_final,
            trend_final=trend_final,
            seasonal_final=seasonal_final,
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
            # Define trend functions, which differ based on additive or
            # multiplicative trend type, i.e. for additive smoothing the trend
            # final value is added to the level at each step, i.
            trend_funcs: dict[
                _SmoothingType,
                Callable[[float, float, int], float],
            ] = {
                _SmoothingType.ADD: lambda lvl, trend, i: lvl + i * trend,
                _SmoothingType.MUL: lambda lvl, trend, i: lvl * (trend**i),
            }

            # A seasonal combination functions will be used to combine the now
            # adjusted level with the seasonal component. The seasonal
            # component is either added or multiplied to the level, depending on
            # the seasonal type.
            seasonal_combine_funcs: dict[
                _SmoothingType,
                Callable[[float, float], float],
            ] = {
                _SmoothingType.ADD: lambda lvl_adjusted, seasonal: lvl_adjusted
                + seasonal,
                _SmoothingType.MUL: lambda lvl_adjusted, seasonal: lvl_adjusted
                * seasonal,
            }

            # Get the correct function based on smoothing type.
            apply_trend = trend_funcs[fitted_params.trend_type]
            apply_seasonal = seasonal_combine_funcs[fitted_params.seasonal_type]

            # Generate forecast
            forecast = [
                apply_seasonal(
                    apply_trend(
                        fitted_params.lvl_final,
                        fitted_params.trend_final,
                        i,
                    ),
                    fitted_params.seasonal_final[
                        (i - 1) % fitted_params.period
                    ],
                )
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
        beta: float,
        gamma: float,
        period: int,
        trend_type: _SmoothingType,
        seasonal_type: _SmoothingType,
    ) -> tuple[float, float, TSArray, TSArray]:
        """Map the smoothing types to the appropriate functions and call."""
        match trend_type, seasonal_type:
            case _SmoothingType.ADD, _SmoothingType.ADD:
                fn = TripleExponentialSmoothing._calculate_add_add_smoothing
            case _SmoothingType.ADD, _SmoothingType.MUL:
                fn = TripleExponentialSmoothing._calculate_add_mul_smoothing
            case _SmoothingType.MUL, _SmoothingType.ADD:
                fn = TripleExponentialSmoothing._calculate_mul_add_smoothing
            case _SmoothingType.MUL, _SmoothingType.MUL:
                fn = TripleExponentialSmoothing._calculate_mul_mul_smoothing

        return fn(
            ts=ts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            period=period,
        )

    @staticmethod
    def _calculate_add_add_smoothing(
        ts: TSArray,
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, TSArray, TSArray]:
        """Calculate with additive / additive smoothing."""
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=_SmoothingType.ADD,
            seasonal_type=_SmoothingType.ADD,
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
    def _calculate_add_mul_smoothing(
        ts: TSArray,
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, TSArray, TSArray]:
        """Calculate with additive / multiplicative smoothing."""
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=_SmoothingType.ADD,
            seasonal_type=_SmoothingType.MUL,
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
    def _calculate_mul_add_smoothing(
        ts: TSArray,
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, TSArray, TSArray]:
        """Calculate with multiplicative / additive smoothing."""
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=_SmoothingType.MUL,
            seasonal_type=_SmoothingType.ADD,
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
    def _calculate_mul_mul_smoothing(
        ts: TSArray,
        alpha: float,
        beta: float,
        gamma: float,
        period: int,
    ) -> tuple[float, float, TSArray, TSArray]:
        """Calculate with multiplicative / multiplicative smoothing."""
        lvl, b, s = TripleExponentialSmoothing._initialise_arrays(
            ts=ts,
            period=period,
            trend_type=_SmoothingType.MUL,
            seasonal_type=_SmoothingType.MUL,
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
    def _initialise_arrays(
        ts: TSArray,
        period: int,
        trend_type: _SmoothingType,
        seasonal_type: _SmoothingType,
    ) -> tuple[
        TSArray,
        TSArray,
        TSArray,
    ]:
        """Initialise arrays for exponential smoothing."""
        m = period
        n = len(ts)
        lvl = np.zeros(n + 1)
        b = np.zeros(n + 1)
        s = np.zeros(n + m)
        lvl[0] = ts[:m].mean()
        if trend_type == _SmoothingType.ADD:
            b[0] = (ts[m : 2 * m].sum() - ts[:m].sum()) / m**2
        elif trend_type == _SmoothingType.MUL:
            b[0] = (ts[m : 2 * m].sum() / ts[:m].sum()) ** (1 / m)
        if seasonal_type == _SmoothingType.ADD:
            s[:m] = ts[:m] - lvl[0]
        elif seasonal_type == _SmoothingType.MUL:
            s[:m] = ts[:m] / lvl[0]

        return lvl, b, s

    @staticmethod
    def _find_optimal_parameters(
        ts: TSArray,
        error_metric_func: Callable[..., float],
        period: int,
        trend_type: _SmoothingType,
        seasonal_type: _SmoothingType,
        alpha: float | None,
        beta: float | None,
        gamma: float | None,
    ) -> tuple[float, float, float]:
        """Find the optimal smoothing parameters that minimise the error."""
        # Set the bounds for the smoothing parameters. If values have been
        # passed, then the bounds will be locked at that value. Else they are
        # set at (0,1).
        alpha_bounds = (alpha or 0, alpha or 1)
        beta_bounds = (beta or 0, beta or 1)
        gamma_bounds = (gamma or 0, gamma or 1)

        # Set the initial guess as the midpoint of the bounds.
        initial_guess = np.array(
            [
                sum(alpha_bounds) / 2,
                sum(beta_bounds) / 2,
                sum(gamma_bounds) / 2,
            ],
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
                alpha_bounds,
                beta_bounds,
                gamma_bounds,
            ],
        )
        optimal_alpha, optimal_beta, optimal_gamma = min_err.x

        return optimal_alpha, optimal_beta, optimal_gamma

    @staticmethod
    def _cost_function(
        params: npt.NDArray[np.float64],
        /,
        ts: TSArray,
        error_metric_func: Callable[..., float],
        period: int,
        trend_type: _SmoothingType,
        seasonal_type: _SmoothingType,
    ) -> float:
        """Calculate the error between actual and fitted time series."""
        alpha, beta, gamma = params
        *_, ts_fitted = (
            TripleExponentialSmoothing._compute_exponential_smoothing(
                ts=ts,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                period=period,
                trend_type=trend_type,
                seasonal_type=seasonal_type,
            )
        )
        return error_metric_func(ts, ts_fitted)
