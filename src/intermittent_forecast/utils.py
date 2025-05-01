# utils.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from enum import Enum


def get_enum_from_str(
    mode_str: str,
    enum_class: type[Enum],
    mode_name: str,
) -> Enum:
    """Convert a string to an enum value."""
    try:
        return enum_class(mode_str.lower())
    except ValueError:
        expected = [m.value for m in enum_class]
        err_msg = (
            f"Unknown {mode_name} mode: '{mode_str}'. "
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
