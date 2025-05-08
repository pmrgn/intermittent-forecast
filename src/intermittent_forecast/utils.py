# utils.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from enum import Enum


def get_enum_member_from_str(
    member_str: str,
    enum_class: type[Enum],
    member_name: str,
) -> Enum:
    """Convert a string to an enum value."""
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
    ts: list[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Validate the time-series is a 1-dimensional numpy array."""
    if isinstance(ts, list):
        ts = np.array(ts)

    elif not isinstance(ts, np.ndarray):
        err_msg = "Time-series must be a list or numpy array."
        raise TypeError(err_msg)

    if ts.ndim != 1:
        err_msg = "Time-series must be 1-dimensional."
        raise ValueError(err_msg)

    if not (
        np.issubdtype(ts.dtype, np.integer)
        or np.issubdtype(
            ts.dtype,
            np.floating,
        )
    ):
        err_msg = "Time-series must contain integers or floats."
        raise TypeError(err_msg)

    min_length = 2
    if len(ts[ts != 0]) < min_length:
        err_msg = "Time-series needs at least two non-zero values"
        raise ValueError(err_msg)

    return ts


def validate_array_is_numeric(
    arr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
