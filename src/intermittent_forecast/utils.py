# utils.py

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
