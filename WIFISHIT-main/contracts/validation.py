"""
contracts/validation.py

Lightweight validation utilities for contract enforcement.

Provides shape checks, dtype checks, monotonic timestamp validation,
and other invariant checks. All validators raise ValidationError on failure.

Usage
-----
>>> from contracts.validation import validate_shape, validate_dtype
>>> validate_shape(arr, (64,), "amplitude")
>>> validate_dtype(arr, np.float64, "amplitude")
"""

import math
from typing import Tuple, Union, Sequence, Any, Optional

import numpy as np


class ValidationError(ValueError):
    """
    Raised when a contract validation fails.

    Subclass of ValueError for compatibility with existing error handling.
    """

    pass


def validate_shape(
    array: np.ndarray,
    expected_shape: Tuple[Optional[int], ...],
    name: str = "array",
) -> None:
    """
    Validate that array has the expected shape.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    expected_shape : Tuple[Optional[int], ...]
        Expected shape. Use None for dimensions that can be any size.
        Example: (None, 64) means any number of rows, exactly 64 columns.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If shape does not match.
    TypeError
        If array is not a numpy array.

    Examples
    --------
    >>> validate_shape(np.zeros((10, 64)), (None, 64), "amplitude")  # OK
    >>> validate_shape(np.zeros((64,)), (64,), "phase")  # OK
    >>> validate_shape(np.zeros((64,)), (32,), "phase")  # Raises
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(array).__name__}")

    if len(array.shape) != len(expected_shape):
        raise ValidationError(
            f"{name} must have {len(expected_shape)} dimensions, "
            f"got {len(array.shape)} with shape {array.shape}"
        )

    for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{name} dimension {i} must be {expected}, "
                f"got {actual} (shape: {array.shape})"
            )


def validate_ndim(
    array: np.ndarray,
    expected_ndim: int,
    name: str = "array",
) -> None:
    """
    Validate that array has the expected number of dimensions.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    expected_ndim : int
        Expected number of dimensions.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If ndim does not match.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(array).__name__}")

    if array.ndim != expected_ndim:
        raise ValidationError(
            f"{name} must be {expected_ndim}D, got {array.ndim}D with shape {array.shape}"
        )


def validate_dtype(
    array: np.ndarray,
    expected_dtype: Union[np.dtype, type, Sequence[Union[np.dtype, type]]],
    name: str = "array",
) -> None:
    """
    Validate that array has an acceptable dtype.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    expected_dtype : dtype or sequence of dtypes
        Acceptable dtype(s). Can be a single dtype or a sequence of dtypes.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If dtype does not match any expected dtype.

    Examples
    --------
    >>> validate_dtype(np.zeros(10), np.float64, "data")  # OK
    >>> validate_dtype(np.zeros(10), [np.float32, np.float64], "data")  # OK
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(array).__name__}")

    if isinstance(expected_dtype, (list, tuple)):
        acceptable = tuple(np.dtype(d) for d in expected_dtype)
    else:
        acceptable = (np.dtype(expected_dtype),)

    if array.dtype not in acceptable:
        acceptable_str = ", ".join(str(d) for d in acceptable)
        raise ValidationError(
            f"{name} dtype must be one of ({acceptable_str}), got {array.dtype}"
        )


def validate_finite(
    array: np.ndarray,
    name: str = "array",
) -> None:
    """
    Validate that all array elements are finite (not inf or nan).

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If any element is inf or nan.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(array).__name__}")

    if not np.all(np.isfinite(array)):
        n_inf = np.sum(np.isinf(array))
        n_nan = np.sum(np.isnan(array))
        raise ValidationError(
            f"{name} contains non-finite values: {n_inf} inf, {n_nan} nan"
        )


def validate_finite_scalar(
    value: float,
    name: str = "value",
) -> None:
    """
    Validate that a scalar value is finite.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If value is inf or nan.
    TypeError
        If value is not numeric.
    """
    try:
        float_val = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}") from e

    if not math.isfinite(float_val):
        raise ValidationError(f"{name} must be finite, got {value}")


def validate_positive(
    value: float,
    name: str = "value",
    allow_zero: bool = False,
) -> None:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Name for error messages.
    allow_zero : bool
        If True, zero is acceptable.

    Raises
    ------
    ValidationError
        If value is not positive (or non-negative if allow_zero).
    """
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")


def validate_range(
    value: float,
    min_val: Optional[float],
    max_val: Optional[float],
    name: str = "value",
    inclusive: bool = True,
) -> None:
    """
    Validate that a value is within a range.

    Parameters
    ----------
    value : float
        Value to validate.
    min_val : float or None
        Minimum acceptable value. None for no lower bound.
    max_val : float or None
        Maximum acceptable value. None for no upper bound.
    name : str
        Name for error messages.
    inclusive : bool
        If True, bounds are inclusive.

    Raises
    ------
    ValidationError
        If value is outside the range.
    """
    if inclusive:
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    else:
        if min_val is not None and value <= min_val:
            raise ValidationError(f"{name} must be > {min_val}, got {value}")
        if max_val is not None and value >= max_val:
            raise ValidationError(f"{name} must be < {max_val}, got {value}")


def validate_monotonic_timestamps(
    timestamps: Union[np.ndarray, Sequence[float]],
    strict: bool = True,
    name: str = "timestamps",
) -> None:
    """
    Validate that timestamps are monotonically increasing.

    Parameters
    ----------
    timestamps : array-like
        Sequence of timestamps.
    strict : bool
        If True, timestamps must be strictly increasing.
        If False, allows equal consecutive timestamps.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If timestamps are not monotonically increasing.
    """
    if len(timestamps) < 2:
        return  # Nothing to check

    ts = np.asarray(timestamps)

    if strict:
        diffs = np.diff(ts)
        if np.any(diffs <= 0):
            bad_idx = int(np.argmax(diffs <= 0))
            raise ValidationError(
                f"{name} must be strictly increasing. "
                f"Violation at index {bad_idx}: {ts[bad_idx]} -> {ts[bad_idx + 1]}"
            )
    else:
        diffs = np.diff(ts)
        if np.any(diffs < 0):
            bad_idx = int(np.argmax(diffs < 0))
            raise ValidationError(
                f"{name} must be monotonically increasing. "
                f"Violation at index {bad_idx}: {ts[bad_idx]} -> {ts[bad_idx + 1]}"
            )


def validate_non_empty(
    value: Any,
    name: str = "value",
) -> None:
    """
    Validate that a value is non-empty.

    Works with arrays, strings, lists, dicts, and other sized objects.

    Parameters
    ----------
    value : Any
        Value to validate.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If value is empty.
    """
    try:
        if len(value) == 0:
            raise ValidationError(f"{name} must not be empty")
    except TypeError:
        # Value doesn't support len(), probably OK
        pass


def validate_same_shape(
    array1: np.ndarray,
    array2: np.ndarray,
    name1: str = "array1",
    name2: str = "array2",
) -> None:
    """
    Validate that two arrays have the same shape.

    Parameters
    ----------
    array1, array2 : np.ndarray
        Arrays to compare.
    name1, name2 : str
        Names for error messages.

    Raises
    ------
    ValidationError
        If shapes differ.
    """
    if array1.shape != array2.shape:
        raise ValidationError(
            f"{name1} shape {array1.shape} does not match {name2} shape {array2.shape}"
        )


def validate_string_non_empty(
    value: str,
    name: str = "value",
) -> None:
    """
    Validate that a string is non-empty.

    Parameters
    ----------
    value : str
        String to validate.
    name : str
        Name for error messages.

    Raises
    ------
    ValidationError
        If string is empty.
    TypeError
        If value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if len(value) == 0:
        raise ValidationError(f"{name} must not be empty")
