"""
csi_frame.py

Lightweight, immutable container for a single Channel State Information (CSI) measurement.

A CSI frame represents a snapshot of the wireless channel at a single point in time.
It contains no logic for generation, processing, or interpretation.

All time units are seconds.
Amplitude and phase arrays correspond elementwise to subcarriers.
"""

import math
from typing import Tuple

import numpy as np


class CSIFrame:
    """
    Immutable container for a single CSI measurement.

    A CSI frame stores the amplitude and phase response of a wireless
    channel across multiple frequency subcarriers at a specific timestamp.

    Parameters
    ----------
    timestamp : float
        Measurement time in seconds. Must be finite.
    amplitude : np.ndarray
        Per-subcarrier amplitude values. Must be 1D.
    phase : np.ndarray
        Per-subcarrier phase values in radians. Must be 1D and
        same shape as amplitude.

    Attributes
    ----------
    timestamp : float
        Measurement time in seconds.
    amplitude : np.ndarray
        Per-subcarrier amplitude values (read-only view).
    phase : np.ndarray
        Per-subcarrier phase values in radians (read-only view).
    num_subcarriers : int
        Number of subcarriers in this frame.

    Raises
    ------
    TypeError
        If timestamp is not a float or arrays are not numpy arrays.
    ValueError
        If timestamp is not finite, arrays are not 1D, or arrays
        have mismatched shapes.

    Notes
    -----
    This class is immutable. The underlying arrays are stored as
    read-only views to prevent modification after creation.
    """

    __slots__ = ('_timestamp', '_amplitude', '_phase')

    def __init__(
        self,
        timestamp: float,
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> None:
        """Initialize an immutable CSI frame with validation."""
        # Validate timestamp
        self._validate_timestamp(timestamp)

        # Validate arrays
        self._validate_arrays(amplitude, phase)

        # Store timestamp
        self._timestamp: float = float(timestamp)

        # Store immutable copies of arrays
        self._amplitude: np.ndarray = self._make_immutable_copy(amplitude)
        self._phase: np.ndarray = self._make_immutable_copy(phase)

    @staticmethod
    def _validate_timestamp(timestamp: float) -> None:
        """
        Validate that timestamp is a finite float.

        Parameters
        ----------
        timestamp : float
            The timestamp to validate.

        Raises
        ------
        TypeError
            If timestamp is not a numeric type.
        ValueError
            If timestamp is not finite (inf or nan).
        """
        try:
            ts_float = float(timestamp)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"timestamp must be a numeric type convertible to float. "
                f"Got {type(timestamp).__name__}."
            ) from e

        if not math.isfinite(ts_float):
            raise ValueError(
                f"timestamp must be finite. Got {timestamp}."
            )

    @staticmethod
    def _validate_arrays(amplitude: np.ndarray, phase: np.ndarray) -> None:
        """
        Validate amplitude and phase arrays.

        Parameters
        ----------
        amplitude : np.ndarray
            The amplitude array to validate.
        phase : np.ndarray
            The phase array to validate.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays.
        ValueError
            If arrays are not 1D or have mismatched shapes.
        """
        # Check types
        if not isinstance(amplitude, np.ndarray):
            raise TypeError(
                f"amplitude must be a numpy ndarray. "
                f"Got {type(amplitude).__name__}."
            )
        if not isinstance(phase, np.ndarray):
            raise TypeError(
                f"phase must be a numpy ndarray. "
                f"Got {type(phase).__name__}."
            )

        # Check dimensionality
        if amplitude.ndim != 1:
            raise ValueError(
                f"amplitude must be 1-dimensional. "
                f"Got {amplitude.ndim} dimensions with shape {amplitude.shape}."
            )
        if phase.ndim != 1:
            raise ValueError(
                f"phase must be 1-dimensional. "
                f"Got {phase.ndim} dimensions with shape {phase.shape}."
            )

        # Check shape match
        if amplitude.shape != phase.shape:
            raise ValueError(
                f"amplitude and phase must have the same shape. "
                f"Got amplitude shape {amplitude.shape} and phase shape {phase.shape}."
            )

    @staticmethod
    def _make_immutable_copy(array: np.ndarray) -> np.ndarray:
        """
        Create an immutable copy of an array.

        Parameters
        ----------
        array : np.ndarray
            The array to copy.

        Returns
        -------
        np.ndarray
            A read-only copy of the array with float64 dtype.
        """
        copy = np.array(array, dtype=np.float64, copy=True)
        copy.flags.writeable = False
        return copy

    @property
    def timestamp(self) -> float:
        """Measurement time in seconds."""
        return self._timestamp

    @property
    def amplitude(self) -> np.ndarray:
        """Per-subcarrier amplitude values (read-only)."""
        return self._amplitude

    @property
    def phase(self) -> np.ndarray:
        """Per-subcarrier phase values in radians (read-only)."""
        return self._phase

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers in this frame."""
        return len(self._amplitude)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the amplitude and phase arrays."""
        return self._amplitude.shape

    def __repr__(self) -> str:
        """Return a string representation of the CSI frame."""
        return (
            f"CSIFrame(timestamp={self._timestamp}, "
            f"num_subcarriers={self.num_subcarriers})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another CSI frame.

        Two frames are equal if they have the same timestamp and
        identical amplitude and phase arrays.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if frames are equal, False otherwise.
        """
        if not isinstance(other, CSIFrame):
            return NotImplemented

        if self._timestamp != other._timestamp:
            return False

        if not np.array_equal(self._amplitude, other._amplitude):
            return False

        if not np.array_equal(self._phase, other._phase):
            return False

        return True

    def __hash__(self) -> int:
        """
        Compute hash for the CSI frame.

        Returns
        -------
        int
            Hash value based on timestamp and array contents.
        """
        return hash((
            self._timestamp,
            self._amplitude.tobytes(),
            self._phase.tobytes(),
        ))
