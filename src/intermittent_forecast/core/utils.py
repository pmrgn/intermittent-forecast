"""Utility functions for the package."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from intermittent_forecast.core._types import TSArray, TSInput


EnumMember = TypeVar("EnumMember", bound=Enum)


def get_enum_member_from_str(
    member_str: str,
    enum_class: type[EnumMember],
    member_name: str,
) -> EnumMember:
    """Get an enum member using the string."""
    try:
        return enum_class(member_str.lower())
    except ValueError:
        expected = [m.value for m in enum_class]
        err_msg = (
            f"Unknown {member_name} member: '{member_str}'. "
            f"Expected one of: {expected}",
        )
        raise ValueError(err_msg) from None


def validate_time_series(
    ts: TSInput,
) -> TSArray:
    """Ensure the time-series is valid and is 1-dimensional."""
    try:
        ts = np.asarray(ts, dtype=np.float64)
    except ValueError:
        err_msg = "Time-series must be an array of integers or floats."
        raise ValueError(err_msg) from None

    if ts.ndim != 1:
        err_msg = "Time-series must be 1-dimensional."
        raise ValueError(err_msg)

    min_length = 2
    if len(ts[ts != 0]) < min_length:
        err_msg = "Time-series needs at least two non-zero values"
        raise ValueError(err_msg)

    return ts


def validate_array_is_numeric(
    arr: TSArray,
) -> TSArray:
    """Validate the array is numeric."""
    if not np.issubdtype(arr.dtype, np.number):
        err_msg = "Array must contain numeric values."
        raise TypeError(err_msg)
    return arr


def validate_positive_integer(
    value: int,
    name: str,
) -> int:
    """Validate the value is a positive integer."""
    if not isinstance(value, int):
        err_msg = f"{name} must be an integer."
        raise TypeError(err_msg)

    if value < 1:
        err_msg = f"{name} must be greater than 0."
        raise ValueError(err_msg)

    return value


def validate_non_negative_integer(
    value: int,
    name: str,
) -> int:
    """Validate the value is a non-negative integer."""
    if not isinstance(value, int):
        err_msg = f"{name} must be an integer."
        raise TypeError(err_msg)

    if value < 0:
        err_msg = f"{name} must be 0 or greater."
        raise ValueError(err_msg)

    return value


def validate_float_within_inclusive_bounds(
    name: str,
    value: float,
    min_value: float = float("-inf"),
    max_value: float = float("inf"),
) -> float:
    """Validate a numeric parameter is within inclusive bounds."""
    if value is None:
        err_msg = (f"Parameter '{name}' must be provided and cannot be None.",)
        raise ValueError(err_msg)
    if not (min_value <= value <= max_value):
        err_msg = (
            f"Parameter '{name}'={value} is out of bounds. "
            f"Must be between {min_value} and {max_value}."
        )
        raise ValueError(err_msg)
    return value
