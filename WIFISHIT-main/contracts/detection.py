"""
contracts/detection.py

Detection and tracking data contracts.

Defines containers for spatial detections and tracks:
- Detection2D, Detection3D: Single-frame detections
- Track2D, Track3D: Multi-frame tracked entities
- TrackState: Enumeration of track lifecycle states

Invariants
----------
- All coordinates must be finite
- Confidence values must be in [0, 1]
- Track IDs must be positive integers
- Timestamps must be finite and monotonically increasing within tracks
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum, auto

import numpy as np

from contracts.validation import (
    ValidationError,
    validate_finite,
    validate_finite_scalar,
    validate_range,
    validate_positive,
    validate_monotonic_timestamps,
)


class TrackState(Enum):
    """
    Lifecycle state of a track.

    States
    ------
    TENTATIVE : Track is new, not yet confirmed.
    CONFIRMED : Track has been confirmed by multiple detections.
    COASTING : Track has no recent detections but is still active.
    DELETED : Track has been terminated.
    """

    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING = auto()
    DELETED = auto()


@dataclass(frozen=True)
class Detection2D:
    """
    Single detection in 2D space.

    Represents a detected entity at a single point in time in 2D space.
    Typically derived from a Field2D via clustering.

    Parameters
    ----------
    x : float
        X coordinate in world space (meters).
    y : float
        Y coordinate in world space (meters).
    timestamp : float
        Detection time in seconds.
    confidence : float
        Detection confidence in [0, 1].
    intensity : float
        Measured intensity/activity at this location.
    size : float
        Estimated size/radius in meters.
    bounding_box : Tuple[float, float, float, float], optional
        Axis-aligned bounding box (x_min, y_min, x_max, y_max) in meters.
    meta : dict
        Additional metadata (e.g., cluster info, source link IDs).

    Raises
    ------
    ValidationError
        If values fail validation.

    Examples
    --------
    >>> det = Detection2D(
    ...     x=1.5, y=2.0, timestamp=0.0,
    ...     confidence=0.9, intensity=0.8, size=0.5
    ... )
    >>> det.position
    (1.5, 2.0)
    """

    x: float
    y: float
    timestamp: float
    confidence: float = 1.0
    intensity: float = 0.0
    size: float = 0.0
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        validate_finite_scalar(self.x, "x")
        validate_finite_scalar(self.y, "y")
        validate_finite_scalar(self.timestamp, "timestamp")
        validate_finite_scalar(self.confidence, "confidence")
        validate_finite_scalar(self.intensity, "intensity")
        validate_finite_scalar(self.size, "size")

        validate_range(self.confidence, 0.0, 1.0, "confidence")
        validate_positive(self.size, "size", allow_zero=True)

        if self.bounding_box is not None:
            if len(self.bounding_box) != 4:
                raise ValidationError("bounding_box must have 4 elements")
            for i, v in enumerate(self.bounding_box):
                validate_finite_scalar(v, f"bounding_box[{i}]")

        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def position(self) -> Tuple[float, float]:
        """Position as (x, y) tuple."""
        return (self.x, self.y)

    def distance_to(self, other: "Detection2D") -> float:
        """Euclidean distance to another detection."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "size": self.size,
        }

    def __repr__(self) -> str:
        return (
            f"Detection2D(x={self.x:.2f}, y={self.y:.2f}, "
            f"t={self.timestamp:.2f}, conf={self.confidence:.2f})"
        )


@dataclass(frozen=True)
class Detection3D:
    """
    Single detection in 3D space.

    Represents a detected entity at a single point in time in 3D space.
    Typically derived from a Field3D via clustering.

    Parameters
    ----------
    x : float
        X coordinate in world space (meters).
    y : float
        Y coordinate in world space (meters).
    z : float
        Z coordinate in world space (meters).
    timestamp : float
        Detection time in seconds.
    confidence : float
        Detection confidence in [0, 1].
    intensity : float
        Measured intensity/activity at this location.
    size : float
        Estimated size/radius in meters.
    bounding_box : Tuple[float, ...], optional
        Axis-aligned bounding box (x_min, y_min, z_min, x_max, y_max, z_max).
    meta : dict
        Additional metadata.

    Examples
    --------
    >>> det = Detection3D(
    ...     x=1.5, y=2.0, z=1.0, timestamp=0.0,
    ...     confidence=0.9, intensity=0.8, size=0.5
    ... )
    >>> det.position
    (1.5, 2.0, 1.0)
    """

    x: float
    y: float
    z: float
    timestamp: float
    confidence: float = 1.0
    intensity: float = 0.0
    size: float = 0.0
    bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        validate_finite_scalar(self.x, "x")
        validate_finite_scalar(self.y, "y")
        validate_finite_scalar(self.z, "z")
        validate_finite_scalar(self.timestamp, "timestamp")
        validate_finite_scalar(self.confidence, "confidence")
        validate_finite_scalar(self.intensity, "intensity")
        validate_finite_scalar(self.size, "size")

        validate_range(self.confidence, 0.0, 1.0, "confidence")
        validate_positive(self.size, "size", allow_zero=True)

        if self.bounding_box is not None:
            if len(self.bounding_box) != 6:
                raise ValidationError("bounding_box must have 6 elements")
            for i, v in enumerate(self.bounding_box):
                validate_finite_scalar(v, f"bounding_box[{i}]")

        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def position(self) -> Tuple[float, float, float]:
        """Position as (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    def distance_to(self, other: "Detection3D") -> float:
        """Euclidean distance to another detection."""
        return np.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def to_2d(self) -> Detection2D:
        """Project to 2D (drop Z coordinate)."""
        bbox_2d = None
        if self.bounding_box is not None:
            bbox_2d = (
                self.bounding_box[0],
                self.bounding_box[1],
                self.bounding_box[3],
                self.bounding_box[4],
            )

        return Detection2D(
            x=self.x,
            y=self.y,
            timestamp=self.timestamp,
            confidence=self.confidence,
            intensity=self.intensity,
            size=self.size,
            bounding_box=bbox_2d,
            meta=self.meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "size": self.size,
        }

    def __repr__(self) -> str:
        return (
            f"Detection3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, "
            f"t={self.timestamp:.2f}, conf={self.confidence:.2f})"
        )


@dataclass(frozen=True)
class Track2D:
    """
    Tracked entity over time in 2D space.

    Represents a persistent entity that has been tracked across
    multiple frames. Maintains history and state.

    Parameters
    ----------
    track_id : int
        Unique track identifier. Must be positive.
    x : float
        Current X coordinate in world space (meters).
    y : float
        Current Y coordinate in world space (meters).
    timestamp : float
        Time of most recent update in seconds.
    state : TrackState
        Current lifecycle state.
    confidence : float
        Track confidence in [0, 1].
    velocity : Tuple[float, float], optional
        Estimated velocity (vx, vy) in m/s.
    age : int
        Number of frames since track creation.
    hits : int
        Number of frames with associated detections.
    time_since_update : float
        Time since last detection association.
    history : List[Tuple[float, float, float]], optional
        Position history as [(t, x, y), ...].
    meta : dict
        Additional metadata.

    Raises
    ------
    ValidationError
        If values fail validation.

    Examples
    --------
    >>> track = Track2D(
    ...     track_id=1, x=1.5, y=2.0, timestamp=0.5,
    ...     state=TrackState.CONFIRMED, confidence=0.9
    ... )
    >>> track.position
    (1.5, 2.0)
    """

    track_id: int
    x: float
    y: float
    timestamp: float
    state: TrackState = TrackState.TENTATIVE
    confidence: float = 1.0
    velocity: Optional[Tuple[float, float]] = None
    age: int = 1
    hits: int = 1
    time_since_update: float = 0.0
    history: Optional[List[Tuple[float, float, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        if not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValidationError(f"track_id must be positive int, got {self.track_id}")

        validate_finite_scalar(self.x, "x")
        validate_finite_scalar(self.y, "y")
        validate_finite_scalar(self.timestamp, "timestamp")
        validate_finite_scalar(self.confidence, "confidence")
        validate_finite_scalar(self.time_since_update, "time_since_update")

        validate_range(self.confidence, 0.0, 1.0, "confidence")
        validate_positive(self.time_since_update, "time_since_update", allow_zero=True)

        if not isinstance(self.state, TrackState):
            raise TypeError(f"state must be TrackState, got {type(self.state).__name__}")

        if not isinstance(self.age, int) or self.age < 1:
            raise ValidationError(f"age must be positive int, got {self.age}")

        if not isinstance(self.hits, int) or self.hits < 0:
            raise ValidationError(f"hits must be non-negative int, got {self.hits}")

        if self.velocity is not None:
            if len(self.velocity) != 2:
                raise ValidationError("velocity must have 2 elements")
            validate_finite_scalar(self.velocity[0], "velocity[0]")
            validate_finite_scalar(self.velocity[1], "velocity[1]")

        if self.history is not None:
            timestamps = [h[0] for h in self.history]
            validate_monotonic_timestamps(timestamps, strict=False, name="history timestamps")

        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def position(self) -> Tuple[float, float]:
        """Current position as (x, y) tuple."""
        return (self.x, self.y)

    @property
    def is_confirmed(self) -> bool:
        """True if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        """True if track is active (not deleted)."""
        return self.state != TrackState.DELETED

    def predict(self, dt: float) -> Tuple[float, float]:
        """
        Predict position after dt seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        Tuple[float, float]
            Predicted (x, y) position.
        """
        if self.velocity is None:
            return (self.x, self.y)
        return (
            self.x + self.velocity[0] * dt,
            self.y + self.velocity[1] * dt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.track_id,
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp,
            "state": self.state.name,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
        }

    def __repr__(self) -> str:
        return (
            f"Track2D(id={self.track_id}, x={self.x:.2f}, y={self.y:.2f}, "
            f"state={self.state.name}, conf={self.confidence:.2f})"
        )


@dataclass(frozen=True)
class Track3D:
    """
    Tracked entity over time in 3D space.

    Represents a persistent entity that has been tracked across
    multiple frames in 3D space.

    Parameters
    ----------
    track_id : int
        Unique track identifier. Must be positive.
    x : float
        Current X coordinate in world space (meters).
    y : float
        Current Y coordinate in world space (meters).
    z : float
        Current Z coordinate in world space (meters).
    timestamp : float
        Time of most recent update in seconds.
    state : TrackState
        Current lifecycle state.
    confidence : float
        Track confidence in [0, 1].
    velocity : Tuple[float, float, float], optional
        Estimated velocity (vx, vy, vz) in m/s.
    age : int
        Number of frames since track creation.
    hits : int
        Number of frames with associated detections.
    time_since_update : float
        Time since last detection association.
    history : List[Tuple[float, float, float, float]], optional
        Position history as [(t, x, y, z), ...].
    meta : dict
        Additional metadata.

    Examples
    --------
    >>> track = Track3D(
    ...     track_id=1, x=1.5, y=2.0, z=1.0, timestamp=0.5,
    ...     state=TrackState.CONFIRMED, confidence=0.9
    ... )
    >>> track.position
    (1.5, 2.0, 1.0)
    """

    track_id: int
    x: float
    y: float
    z: float
    timestamp: float
    state: TrackState = TrackState.TENTATIVE
    confidence: float = 1.0
    velocity: Optional[Tuple[float, float, float]] = None
    age: int = 1
    hits: int = 1
    time_since_update: float = 0.0
    history: Optional[List[Tuple[float, float, float, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        if not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValidationError(f"track_id must be positive int, got {self.track_id}")

        validate_finite_scalar(self.x, "x")
        validate_finite_scalar(self.y, "y")
        validate_finite_scalar(self.z, "z")
        validate_finite_scalar(self.timestamp, "timestamp")
        validate_finite_scalar(self.confidence, "confidence")
        validate_finite_scalar(self.time_since_update, "time_since_update")

        validate_range(self.confidence, 0.0, 1.0, "confidence")
        validate_positive(self.time_since_update, "time_since_update", allow_zero=True)

        if not isinstance(self.state, TrackState):
            raise TypeError(f"state must be TrackState, got {type(self.state).__name__}")

        if not isinstance(self.age, int) or self.age < 1:
            raise ValidationError(f"age must be positive int, got {self.age}")

        if not isinstance(self.hits, int) or self.hits < 0:
            raise ValidationError(f"hits must be non-negative int, got {self.hits}")

        if self.velocity is not None:
            if len(self.velocity) != 3:
                raise ValidationError("velocity must have 3 elements")
            for i in range(3):
                validate_finite_scalar(self.velocity[i], f"velocity[{i}]")

        if self.history is not None:
            timestamps = [h[0] for h in self.history]
            validate_monotonic_timestamps(timestamps, strict=False, name="history timestamps")

        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def position(self) -> Tuple[float, float, float]:
        """Current position as (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    @property
    def is_confirmed(self) -> bool:
        """True if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        """True if track is active (not deleted)."""
        return self.state != TrackState.DELETED

    def predict(self, dt: float) -> Tuple[float, float, float]:
        """
        Predict position after dt seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        Tuple[float, float, float]
            Predicted (x, y, z) position.
        """
        if self.velocity is None:
            return (self.x, self.y, self.z)
        return (
            self.x + self.velocity[0] * dt,
            self.y + self.velocity[1] * dt,
            self.z + self.velocity[2] * dt,
        )

    def to_2d(self) -> Track2D:
        """Project to 2D track (drop Z coordinate)."""
        vel_2d = None
        if self.velocity is not None:
            vel_2d = (self.velocity[0], self.velocity[1])

        hist_2d = None
        if self.history is not None:
            hist_2d = [(t, x, y) for t, x, y, z in self.history]

        return Track2D(
            track_id=self.track_id,
            x=self.x,
            y=self.y,
            timestamp=self.timestamp,
            state=self.state,
            confidence=self.confidence,
            velocity=vel_2d,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            history=hist_2d,
            meta=self.meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.track_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": self.timestamp,
            "state": self.state.name,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
        }

    def __repr__(self) -> str:
        return (
            f"Track3D(id={self.track_id}, "
            f"x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, "
            f"state={self.state.name}, conf={self.confidence:.2f})"
        )
