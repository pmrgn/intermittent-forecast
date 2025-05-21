"""Error metrics for time series and intermittent demand forecasting."""

from typing import Callable, ClassVar

import numpy as np

from intermittent_forecast.core import utils
from intermittent_forecast.core._types import TSArray

ErrorMetricFunc = Callable[
    [TSArray, TSArray],
    float,
]


class ErrorMetricRegistry:
    """Registry for error metrics.

    The registry is a dictionary that maps error metric names to error metric
    functions, allowing for easy lookup and retrieval.
    """

    _registry: ClassVar[dict[str, ErrorMetricFunc]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[ErrorMetricFunc], ErrorMetricFunc]:
        """Register a new error metric function.

        The function will be stored in the registry with the provided name.
        """

        def decorator(fn: ErrorMetricFunc) -> ErrorMetricFunc:
            cls._registry[name.upper()] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> ErrorMetricFunc:
        """Retrieve a registered error metric function by name."""
        if not isinstance(name, str):
            err_msg = "Error metric must be a string."
            raise TypeError(err_msg)

        try:
            cls._registry[name.upper()]
        except KeyError:
            err_msg = (
                f"Error metric '{name}' not found. "
                f"Available metrics: {list(cls._registry.keys())}"
            )
            raise ValueError(err_msg) from None
        return cls._registry[name.upper()]

    @classmethod
    def get_registry(cls) -> dict[str, ErrorMetricFunc]:
        """Return the registry of error metrics."""
        return cls._registry


class ErrorMetrics:
    """Class for error metrics."""

    @staticmethod
    @ErrorMetricRegistry.register("MAE")
    def mae(
        ts: TSArray,
        forecast: TSArray,
    ) -> float:
        """Return mean absolute error of two numpy arrays."""
        e = ts - forecast
        e = e[~np.isnan(e)]
        return ErrorMetrics.validate_non_negative_numeric(
            np.mean(np.abs(e)),
        )

    @staticmethod
    @ErrorMetricRegistry.register("MSE")
    def mse(
        ts: TSArray,
        forecast: TSArray,
    ) -> float:
        """Return mean squared error of two numpy arrays."""
        e = ts - forecast
        e = e[~np.isnan(e)]
        return ErrorMetrics.validate_non_negative_numeric(
            np.mean(e**2),
        )

    @staticmethod
    @ErrorMetricRegistry.register("MSR")
    def msr(
        ts: TSArray,
        forecast: TSArray,
    ) -> float:
        """Return mean squared rate of two numpy arrays."""
        d_rate = ErrorMetrics._compute_demand_rate_array(ts)
        return ErrorMetrics.mse(d_rate, forecast)

    @staticmethod
    @ErrorMetricRegistry.register("MAR")
    def mar(
        ts: TSArray,
        forecast: TSArray,
    ) -> float:
        """Return mean absolute rate of two numpy arrays."""
        d_rate = ErrorMetrics._compute_demand_rate_array(ts)
        return ErrorMetrics.mae(d_rate, forecast)

    @staticmethod
    @ErrorMetricRegistry.register("PIS")
    def pis(
        ts: TSArray,
        forecast: TSArray,
    ) -> float:
        """Return absolute periods in stock of two numpy arrays."""
        e = ts - forecast
        e = e[~np.isnan(e)]
        cfe = np.cumsum(e)
        pis = -np.cumsum(cfe)
        return ErrorMetrics.validate_non_negative_numeric(
            np.abs(pis[-1]),
        )

    @staticmethod
    def _compute_demand_rate_array(
        ts: TSArray,
    ) -> TSArray:
        """Return demand rate of a time series."""
        # Calculate the demand rate
        n = len(ts)
        demand_rate = np.cumsum(ts) / np.arange(1, n + 1)

        # As the initial value of the demand rate is going to be the first
        # value, this can introduce bias. To mitigate this, backfill the first
        # 10% of values.
        back_fill = int(np.ceil(0.1 * n))
        if back_fill > 1:
            demand_rate[:back_fill] = demand_rate[back_fill - 1]
        return utils.validate_array_is_numeric(demand_rate)

    @staticmethod
    def validate_non_negative_numeric(
        value: float,
    ) -> float:
        """Validate the value is non-negative and numeric."""
        if not isinstance(value, float):
            err_msg = f"Value expected to be numeric, got type: {type(value)}."
            raise TypeError(err_msg)

        if value < 0:
            err_msg = f"Value expected to be non-negative, got type: {value}."
            raise ValueError(err_msg)
        return value
