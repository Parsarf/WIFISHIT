"""
contracts/measurement.py

Measurement data contracts for per-link scalar/vector outputs.

Defines containers for inference outputs that are not spatially resolved:
- QualityMetrics: Signal quality indicators
- Measurement: Per-link measurement with value and quality

Invariants
----------
- All scalar values must be finite
- Confidence values must be in [0, 1]
- Quality scores must be in [0, 1]
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import math

import numpy as np

from contracts.validation import (
    ValidationError,
    validate_finite,
    validate_finite_scalar,
    validate_range,
    validate_string_non_empty,
)


def _make_immutable_copy(array: np.ndarray) -> np.ndarray:
    """Create an immutable float64 copy of an array."""
    copy = np.array(array, dtype=np.float64, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True)
class QualityMetrics:
    """
    Signal quality metrics for a measurement.

    Encapsulates various quality indicators that help downstream
    consumers decide how to weight or filter measurements.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio in dB.
    confidence : float
        Overall confidence in the measurement, in [0, 1].
    completeness : float
        Fraction of expected data that was received, in [0, 1].
    stability : float
        Temporal stability of the measurement, in [0, 1].
        High stability means consistent over time.
    outlier_score : float
        Score indicating how much of an outlier this measurement is, in [0, 1].
        0 = typical, 1 = extreme outlier.

    Raises
    ------
    ValidationError
        If values are out of range.
    """

    snr: float = 0.0
    confidence: float = 1.0
    completeness: float = 1.0
    stability: float = 1.0
    outlier_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate all metrics."""
        validate_finite_scalar(self.snr, "snr")
        validate_finite_scalar(self.confidence, "confidence")
        validate_finite_scalar(self.completeness, "completeness")
        validate_finite_scalar(self.stability, "stability")
        validate_finite_scalar(self.outlier_score, "outlier_score")

        validate_range(self.confidence, 0.0, 1.0, "confidence")
        validate_range(self.completeness, 0.0, 1.0, "completeness")
        validate_range(self.stability, 0.0, 1.0, "stability")
        validate_range(self.outlier_score, 0.0, 1.0, "outlier_score")

    @property
    def is_reliable(self) -> bool:
        """
        Quick check if measurement is likely reliable.

        Returns True if confidence >= 0.5 and outlier_score < 0.5.
        """
        return self.confidence >= 0.5 and self.outlier_score < 0.5

    def combined_quality(self) -> float:
        """
        Compute a single combined quality score.

        Returns
        -------
        float
            Combined quality in [0, 1], higher is better.
        """
        return (
            self.confidence
            * self.completeness
            * self.stability
            * (1.0 - self.outlier_score)
        )


@dataclass(frozen=True)
class Measurement:
    """
    Per-link measurement with value and quality metrics.

    Represents a scalar or vector measurement from a single link,
    along with quality indicators and metadata.

    Parameters
    ----------
    link_id : str
        Identifier for the TX-RX link.
    timestamp : float
        Measurement time in seconds.
    value : float or np.ndarray
        The measurement value. Can be scalar or 1D vector.
    unit : str
        Unit of measurement (e.g., "dB", "m/s", "dimensionless").
    quality : QualityMetrics
        Quality indicators for this measurement.
    meta : dict
        Additional metadata.

    Attributes
    ----------
    is_scalar : bool
        True if value is a scalar.
    is_vector : bool
        True if value is a vector.
    dim : int
        Dimensionality (1 for scalar, length for vector).

    Raises
    ------
    TypeError
        If types are incorrect.
    ValidationError
        If values fail validation.

    Examples
    --------
    >>> m = Measurement(
    ...     link_id="link_0",
    ...     timestamp=0.0,
    ...     value=42.5,
    ...     unit="dB",
    ... )
    >>> m.is_scalar
    True

    >>> m = Measurement(
    ...     link_id="link_0",
    ...     timestamp=0.0,
    ...     value=np.array([1.0, 2.0, 3.0]),
    ...     unit="dimensionless",
    ... )
    >>> m.dim
    3
    """

    link_id: str
    timestamp: float
    value: Union[float, np.ndarray]
    unit: str = "dimensionless"
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze after initialization."""
        # Validate link_id
        validate_string_non_empty(self.link_id, "link_id")

        # Validate timestamp
        validate_finite_scalar(self.timestamp, "timestamp")

        # Validate unit
        if not isinstance(self.unit, str):
            raise TypeError(f"unit must be str, got {type(self.unit).__name__}")

        # Validate quality
        if not isinstance(self.quality, QualityMetrics):
            raise TypeError(
                f"quality must be QualityMetrics, got {type(self.quality).__name__}"
            )

        # Validate and freeze value
        if isinstance(self.value, np.ndarray):
            if self.value.ndim > 1:
                raise ValidationError(
                    f"value array must be 0D or 1D, got {self.value.ndim}D"
                )
            validate_finite(self.value, "value")
            object.__setattr__(self, "value", _make_immutable_copy(self.value))
        else:
            # Scalar
            try:
                float_val = float(self.value)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"value must be numeric or ndarray, got {type(self.value).__name__}"
                ) from e
            validate_finite_scalar(float_val, "value")
            object.__setattr__(self, "value", float_val)

        # Ensure meta is a dict copy
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def is_scalar(self) -> bool:
        """True if value is a scalar."""
        return not isinstance(self.value, np.ndarray) or self.value.ndim == 0

    @property
    def is_vector(self) -> bool:
        """True if value is a 1D vector."""
        return isinstance(self.value, np.ndarray) and self.value.ndim == 1

    @property
    def dim(self) -> int:
        """Dimensionality of the measurement."""
        if self.is_scalar:
            return 1
        return len(self.value)

    def as_float(self) -> float:
        """
        Get value as a float.

        For scalar values, returns the value directly.
        For vectors, raises an error.

        Returns
        -------
        float
            The scalar value.

        Raises
        ------
        ValueError
            If value is not scalar.
        """
        if not self.is_scalar:
            raise ValueError("Cannot convert vector measurement to float")
        if isinstance(self.value, np.ndarray):
            return float(self.value.item())
        return float(self.value)

    def as_array(self) -> np.ndarray:
        """
        Get value as a numpy array.

        For scalar values, returns a 0D array.
        For vectors, returns the vector.

        Returns
        -------
        np.ndarray
            The value as an array.
        """
        if isinstance(self.value, np.ndarray):
            return self.value
        return np.array(self.value)

    def __repr__(self) -> str:
        """Concise string representation."""
        if self.is_scalar:
            val_str = f"{self.value:.4g}"
        else:
            val_str = f"array({self.dim})"
        return (
            f"Measurement(link_id={self.link_id!r}, "
            f"timestamp={self.timestamp}, "
            f"value={val_str} {self.unit})"
        )
