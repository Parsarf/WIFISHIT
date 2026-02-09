"""
contracts/csi.py

CSI frame data contracts.

Defines the core CSI measurement containers used throughout the pipeline:
- CSIFrame: Raw measurement from a single wireless link
- ConditionedCSIFrame: Preprocessed frame ready for inference

Both are immutable (frozen dataclasses) with runtime validation.

Invariants
----------
- link_id must be a non-empty string identifying the TX-RX pair
- timestamp must be finite (no inf, no nan)
- amplitude and phase must be 1D arrays of the same length
- all array elements must be finite
- amplitude must have dtype float64 or float32
- phase is in radians
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import math

import numpy as np

from contracts.validation import (
    ValidationError,
    validate_shape,
    validate_dtype,
    validate_finite,
    validate_finite_scalar,
    validate_same_shape,
    validate_string_non_empty,
)


def _make_immutable_copy(array: np.ndarray) -> np.ndarray:
    """Create an immutable float64 copy of an array."""
    copy = np.array(array, dtype=np.float64, copy=True)
    copy.flags.writeable = False
    return copy


def _validate_csi_arrays(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> None:
    """Validate amplitude and phase arrays for CSI frames."""
    # Type check
    if not isinstance(amplitude, np.ndarray):
        raise TypeError(f"amplitude must be ndarray, got {type(amplitude).__name__}")
    if not isinstance(phase, np.ndarray):
        raise TypeError(f"phase must be ndarray, got {type(phase).__name__}")

    # 1D check
    if amplitude.ndim != 1:
        raise ValidationError(
            f"amplitude must be 1D, got {amplitude.ndim}D with shape {amplitude.shape}"
        )
    if phase.ndim != 1:
        raise ValidationError(
            f"phase must be 1D, got {phase.ndim}D with shape {phase.shape}"
        )

    # Shape match
    if amplitude.shape != phase.shape:
        raise ValidationError(
            f"amplitude shape {amplitude.shape} must match phase shape {phase.shape}"
        )

    # Finite check
    validate_finite(amplitude, "amplitude")
    validate_finite(phase, "phase")


@dataclass(frozen=True)
class CSIFrame:
    """
    Immutable container for a single CSI measurement.

    Represents a snapshot of channel state information from one wireless
    link (TX-RX pair) at a specific timestamp.

    Parameters
    ----------
    link_id : str
        Identifier for the TX-RX link (e.g., "ap1_sta2", "nexmon_wlan0").
    timestamp : float
        Measurement time in seconds. Must be finite.
    amplitude : np.ndarray
        Per-subcarrier amplitude values. 1D array.
    phase : np.ndarray
        Per-subcarrier phase values in radians. 1D array, same length as amplitude.
    meta : dict
        Optional metadata (e.g., RSSI, noise floor, MAC addresses).

    Attributes
    ----------
    num_subcarriers : int
        Number of subcarriers in this frame.
    shape : Tuple[int]
        Shape of the amplitude/phase arrays.

    Raises
    ------
    TypeError
        If types are incorrect.
    ValidationError
        If values fail validation (non-finite, shape mismatch, etc.).

    Examples
    --------
    >>> frame = CSIFrame(
    ...     link_id="link_0",
    ...     timestamp=0.0,
    ...     amplitude=np.ones(64),
    ...     phase=np.zeros(64),
    ...     meta={"rssi": -40}
    ... )
    >>> frame.num_subcarriers
    64
    """

    link_id: str
    timestamp: float
    amplitude: np.ndarray = field(repr=False)
    phase: np.ndarray = field(repr=False)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze arrays after initialization."""
        # Validate link_id
        validate_string_non_empty(self.link_id, "link_id")

        # Validate timestamp
        validate_finite_scalar(self.timestamp, "timestamp")

        # Validate arrays
        _validate_csi_arrays(self.amplitude, self.phase)

        # Make arrays immutable (bypass frozen restriction)
        object.__setattr__(self, "amplitude", _make_immutable_copy(self.amplitude))
        object.__setattr__(self, "phase", _make_immutable_copy(self.phase))

        # Ensure meta is a dict copy
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers in this frame."""
        return len(self.amplitude)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the amplitude and phase arrays."""
        return self.amplitude.shape

    def get_complex(self) -> np.ndarray:
        """
        Get complex-valued CSI representation.

        Returns
        -------
        np.ndarray
            Complex CSI values: amplitude * exp(1j * phase).
        """
        return self.amplitude * np.exp(1j * self.phase)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"CSIFrame(link_id={self.link_id!r}, "
            f"timestamp={self.timestamp}, "
            f"num_subcarriers={self.num_subcarriers})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another CSIFrame."""
        if not isinstance(other, CSIFrame):
            return NotImplemented

        return (
            self.link_id == other.link_id
            and self.timestamp == other.timestamp
            and np.array_equal(self.amplitude, other.amplitude)
            and np.array_equal(self.phase, other.phase)
        )

    def __hash__(self) -> int:
        """Compute hash for the frame."""
        return hash((
            self.link_id,
            self.timestamp,
            self.amplitude.tobytes(),
            self.phase.tobytes(),
        ))


@dataclass(frozen=True)
class ConditionedCSIFrame:
    """
    Preprocessed CSI frame ready for inference.

    Contains conditioned (normalized, filtered, unwrapped) CSI data
    along with derived features.

    Parameters
    ----------
    link_id : str
        Identifier for the TX-RX link.
    timestamp : float
        Measurement time in seconds.
    amplitude : np.ndarray
        Conditioned amplitude (e.g., normalized, baseline-subtracted).
    phase : np.ndarray
        Conditioned phase (e.g., unwrapped, offset-removed).
    amplitude_raw : np.ndarray, optional
        Original raw amplitude (for reference).
    phase_raw : np.ndarray, optional
        Original raw phase (for reference).
    conditioning_method : str
        Description of conditioning applied.
    meta : dict
        Additional metadata.

    Raises
    ------
    TypeError
        If types are incorrect.
    ValidationError
        If values fail validation.
    """

    link_id: str
    timestamp: float
    amplitude: np.ndarray = field(repr=False)
    phase: np.ndarray = field(repr=False)
    amplitude_raw: Optional[np.ndarray] = field(default=None, repr=False)
    phase_raw: Optional[np.ndarray] = field(default=None, repr=False)
    conditioning_method: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze arrays after initialization."""
        # Validate link_id
        validate_string_non_empty(self.link_id, "link_id")

        # Validate timestamp
        validate_finite_scalar(self.timestamp, "timestamp")

        # Validate arrays
        _validate_csi_arrays(self.amplitude, self.phase)

        # Make arrays immutable
        object.__setattr__(self, "amplitude", _make_immutable_copy(self.amplitude))
        object.__setattr__(self, "phase", _make_immutable_copy(self.phase))

        # Handle optional raw arrays
        if self.amplitude_raw is not None:
            if not isinstance(self.amplitude_raw, np.ndarray):
                raise TypeError("amplitude_raw must be ndarray")
            object.__setattr__(
                self, "amplitude_raw", _make_immutable_copy(self.amplitude_raw)
            )

        if self.phase_raw is not None:
            if not isinstance(self.phase_raw, np.ndarray):
                raise TypeError("phase_raw must be ndarray")
            object.__setattr__(
                self, "phase_raw", _make_immutable_copy(self.phase_raw)
            )

        # Validate conditioning_method
        if not isinstance(self.conditioning_method, str):
            raise TypeError("conditioning_method must be str")

        # Ensure meta is a dict copy
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers in this frame."""
        return len(self.amplitude)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the amplitude and phase arrays."""
        return self.amplitude.shape

    def get_complex(self) -> np.ndarray:
        """Get complex-valued CSI representation."""
        return self.amplitude * np.exp(1j * self.phase)

    @classmethod
    def from_raw(
        cls,
        frame: CSIFrame,
        amplitude: np.ndarray,
        phase: np.ndarray,
        conditioning_method: str = "default",
    ) -> "ConditionedCSIFrame":
        """
        Create a ConditionedCSIFrame from a raw CSIFrame.

        Parameters
        ----------
        frame : CSIFrame
            Original raw frame.
        amplitude : np.ndarray
            Conditioned amplitude.
        phase : np.ndarray
            Conditioned phase.
        conditioning_method : str
            Description of conditioning applied.

        Returns
        -------
        ConditionedCSIFrame
            New conditioned frame with raw data preserved.
        """
        return cls(
            link_id=frame.link_id,
            timestamp=frame.timestamp,
            amplitude=amplitude,
            phase=phase,
            amplitude_raw=frame.amplitude,
            phase_raw=frame.phase,
            conditioning_method=conditioning_method,
            meta=frame.meta,
        )

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"ConditionedCSIFrame(link_id={self.link_id!r}, "
            f"timestamp={self.timestamp}, "
            f"num_subcarriers={self.num_subcarriers}, "
            f"method={self.conditioning_method!r})"
        )
