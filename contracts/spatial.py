"""
contracts/spatial.py

Spatial field data contracts.

Defines containers for 2D and 3D scalar fields with confidence and metadata:
- FieldMetadata: Common metadata for spatial fields
- Field2D: 2D scalar field (e.g., floor heatmap)
- Field3D: 3D scalar field (e.g., voxel grid)

Invariants
----------
- Data arrays must have the correct dimensionality
- Confidence arrays, if provided, must match data shape and be in [0, 1]
- Resolution must be positive
- Origin coordinates must be finite
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

import numpy as np

from contracts.validation import (
    ValidationError,
    validate_shape,
    validate_ndim,
    validate_finite,
    validate_finite_scalar,
    validate_positive,
    validate_range,
    validate_same_shape,
)


def _make_immutable_copy(array: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Create an immutable copy of an array."""
    copy = np.array(array, dtype=dtype, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True)
class FieldMetadata:
    """
    Common metadata for spatial fields.

    Parameters
    ----------
    timestamp : float
        Time at which field was computed, in seconds.
    source : str
        Description of how the field was generated.
    frame_count : int
        Number of CSI frames used to compute this field.
    link_ids : Tuple[str, ...]
        IDs of links that contributed to this field.
    extra : dict
        Additional metadata.
    """

    timestamp: float = 0.0
    source: str = "unknown"
    frame_count: int = 0
    link_ids: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        validate_finite_scalar(self.timestamp, "timestamp")

        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")

        if not isinstance(self.frame_count, int) or self.frame_count < 0:
            raise ValidationError(f"frame_count must be non-negative int, got {self.frame_count}")

        if not isinstance(self.link_ids, tuple):
            raise TypeError(f"link_ids must be tuple, got {type(self.link_ids).__name__}")

        if not isinstance(self.extra, dict):
            raise TypeError(f"extra must be dict, got {type(self.extra).__name__}")
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class Field2D:
    """
    2D scalar field with optional confidence and metadata.

    Represents a discretized 2D scalar field (e.g., floor heatmap,
    activity projection). Includes spatial reference information
    for converting between grid and world coordinates.

    Parameters
    ----------
    data : np.ndarray
        2D array of field values. Shape: (nx, ny).
    origin : Tuple[float, float]
        World-space origin (x_min, y_min) in meters.
    resolution : float
        Grid cell size in meters.
    confidence : np.ndarray, optional
        Per-cell confidence values in [0, 1]. Same shape as data.
    meta : FieldMetadata
        Field metadata.

    Attributes
    ----------
    shape : Tuple[int, int]
        Grid shape (nx, ny).
    dimensions : Tuple[float, float]
        Physical dimensions (width, height) in meters.

    Raises
    ------
    TypeError
        If types are incorrect.
    ValidationError
        If values fail validation.

    Examples
    --------
    >>> field = Field2D(
    ...     data=np.random.rand(40, 40),
    ...     origin=(-5.0, -5.0),
    ...     resolution=0.25,
    ... )
    >>> field.shape
    (40, 40)
    >>> field.dimensions
    (10.0, 10.0)
    """

    data: np.ndarray
    origin: Tuple[float, float]
    resolution: float
    confidence: Optional[np.ndarray] = None
    meta: FieldMetadata = field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        """Validate and freeze after initialization."""
        # Validate data
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be ndarray, got {type(self.data).__name__}")
        validate_ndim(self.data, 2, "data")
        validate_finite(self.data, "data")
        object.__setattr__(self, "data", _make_immutable_copy(self.data))

        # Validate origin
        if not isinstance(self.origin, tuple) or len(self.origin) != 2:
            raise TypeError("origin must be a tuple of 2 floats")
        validate_finite_scalar(self.origin[0], "origin[0]")
        validate_finite_scalar(self.origin[1], "origin[1]")

        # Validate resolution
        validate_finite_scalar(self.resolution, "resolution")
        validate_positive(self.resolution, "resolution")

        # Validate confidence
        if self.confidence is not None:
            if not isinstance(self.confidence, np.ndarray):
                raise TypeError(
                    f"confidence must be ndarray, got {type(self.confidence).__name__}"
                )
            validate_same_shape(self.confidence, self.data, "confidence", "data")
            validate_finite(self.confidence, "confidence")

            # Check range [0, 1]
            if np.any(self.confidence < 0) or np.any(self.confidence > 1):
                raise ValidationError("confidence values must be in [0, 1]")

            object.__setattr__(
                self, "confidence", _make_immutable_copy(self.confidence)
            )

        # Validate meta
        if not isinstance(self.meta, FieldMetadata):
            raise TypeError(
                f"meta must be FieldMetadata, got {type(self.meta).__name__}"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (nx, ny)."""
        return (self.data.shape[0], self.data.shape[1])

    @property
    def dimensions(self) -> Tuple[float, float]:
        """Physical dimensions (width, height) in meters."""
        return (
            self.data.shape[0] * self.resolution,
            self.data.shape[1] * self.resolution,
        )

    def cell_center(self, i: int, j: int) -> Tuple[float, float]:
        """
        Get the world-space center of a cell.

        Parameters
        ----------
        i, j : int
            Cell indices.

        Returns
        -------
        Tuple[float, float]
            World-space coordinates (x, y) in meters.
        """
        x = self.origin[0] + (i + 0.5) * self.resolution
        y = self.origin[1] + (j + 0.5) * self.resolution
        return (x, y)

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates to cell indices.

        Parameters
        ----------
        x, y : float
            World-space coordinates in meters.

        Returns
        -------
        Optional[Tuple[int, int]]
            Cell indices (i, j), or None if out of bounds.
        """
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)

        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            return (i, j)
        return None

    def value_at(self, x: float, y: float) -> Optional[float]:
        """
        Get field value at world coordinates.

        Parameters
        ----------
        x, y : float
            World-space coordinates in meters.

        Returns
        -------
        Optional[float]
            Field value, or None if out of bounds.
        """
        cell = self.world_to_cell(x, y)
        if cell is None:
            return None
        return float(self.data[cell[0], cell[1]])

    def max_value(self) -> float:
        """Maximum value in the field."""
        return float(np.max(self.data))

    def mean_value(self) -> float:
        """Mean value in the field."""
        return float(np.mean(self.data))

    def total_value(self) -> float:
        """Sum of all values in the field."""
        return float(np.sum(self.data))

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"Field2D(shape={self.shape}, "
            f"origin={self.origin}, "
            f"resolution={self.resolution})"
        )


@dataclass(frozen=True)
class Field3D:
    """
    3D scalar field with optional confidence and metadata.

    Represents a discretized 3D scalar field (e.g., voxel grid).
    Includes spatial reference information for converting between
    grid and world coordinates.

    Parameters
    ----------
    data : np.ndarray
        3D array of field values. Shape: (nx, ny, nz).
    origin : Tuple[float, float, float]
        World-space origin (x_min, y_min, z_min) in meters.
    resolution : float
        Voxel size in meters (same for all axes).
    confidence : np.ndarray, optional
        Per-voxel confidence values in [0, 1]. Same shape as data.
    meta : FieldMetadata
        Field metadata.

    Attributes
    ----------
    shape : Tuple[int, int, int]
        Grid shape (nx, ny, nz).
    dimensions : Tuple[float, float, float]
        Physical dimensions (width, height, depth) in meters.

    Raises
    ------
    TypeError
        If types are incorrect.
    ValidationError
        If values fail validation.

    Examples
    --------
    >>> field = Field3D(
    ...     data=np.random.rand(40, 40, 12),
    ...     origin=(-5.0, -5.0, 0.0),
    ...     resolution=0.25,
    ... )
    >>> field.shape
    (40, 40, 12)
    >>> field.dimensions
    (10.0, 10.0, 3.0)
    """

    data: np.ndarray
    origin: Tuple[float, float, float]
    resolution: float
    confidence: Optional[np.ndarray] = None
    meta: FieldMetadata = field(default_factory=FieldMetadata)

    def __post_init__(self) -> None:
        """Validate and freeze after initialization."""
        # Validate data
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be ndarray, got {type(self.data).__name__}")
        validate_ndim(self.data, 3, "data")
        validate_finite(self.data, "data")
        object.__setattr__(self, "data", _make_immutable_copy(self.data))

        # Validate origin
        if not isinstance(self.origin, tuple) or len(self.origin) != 3:
            raise TypeError("origin must be a tuple of 3 floats")
        validate_finite_scalar(self.origin[0], "origin[0]")
        validate_finite_scalar(self.origin[1], "origin[1]")
        validate_finite_scalar(self.origin[2], "origin[2]")

        # Validate resolution
        validate_finite_scalar(self.resolution, "resolution")
        validate_positive(self.resolution, "resolution")

        # Validate confidence
        if self.confidence is not None:
            if not isinstance(self.confidence, np.ndarray):
                raise TypeError(
                    f"confidence must be ndarray, got {type(self.confidence).__name__}"
                )
            validate_same_shape(self.confidence, self.data, "confidence", "data")
            validate_finite(self.confidence, "confidence")

            if np.any(self.confidence < 0) or np.any(self.confidence > 1):
                raise ValidationError("confidence values must be in [0, 1]")

            object.__setattr__(
                self, "confidence", _make_immutable_copy(self.confidence)
            )

        # Validate meta
        if not isinstance(self.meta, FieldMetadata):
            raise TypeError(
                f"meta must be FieldMetadata, got {type(self.meta).__name__}"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (nx, ny, nz)."""
        return (self.data.shape[0], self.data.shape[1], self.data.shape[2])

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Physical dimensions (width, height, depth) in meters."""
        return (
            self.data.shape[0] * self.resolution,
            self.data.shape[1] * self.resolution,
            self.data.shape[2] * self.resolution,
        )

    def voxel_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Get the world-space center of a voxel.

        Parameters
        ----------
        i, j, k : int
            Voxel indices.

        Returns
        -------
        Tuple[float, float, float]
            World-space coordinates (x, y, z) in meters.
        """
        x = self.origin[0] + (i + 0.5) * self.resolution
        y = self.origin[1] + (j + 0.5) * self.resolution
        z = self.origin[2] + (k + 0.5) * self.resolution
        return (x, y, z)

    def world_to_voxel(
        self, x: float, y: float, z: float
    ) -> Optional[Tuple[int, int, int]]:
        """
        Convert world coordinates to voxel indices.

        Parameters
        ----------
        x, y, z : float
            World-space coordinates in meters.

        Returns
        -------
        Optional[Tuple[int, int, int]]
            Voxel indices (i, j, k), or None if out of bounds.
        """
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)
        k = int((z - self.origin[2]) / self.resolution)

        if (
            0 <= i < self.shape[0]
            and 0 <= j < self.shape[1]
            and 0 <= k < self.shape[2]
        ):
            return (i, j, k)
        return None

    def value_at(self, x: float, y: float, z: float) -> Optional[float]:
        """
        Get field value at world coordinates.

        Parameters
        ----------
        x, y, z : float
            World-space coordinates in meters.

        Returns
        -------
        Optional[float]
            Field value, or None if out of bounds.
        """
        voxel = self.world_to_voxel(x, y, z)
        if voxel is None:
            return None
        return float(self.data[voxel[0], voxel[1], voxel[2]])

    def max_value(self) -> float:
        """Maximum value in the field."""
        return float(np.max(self.data))

    def mean_value(self) -> float:
        """Mean value in the field."""
        return float(np.mean(self.data))

    def total_value(self) -> float:
        """Sum of all values in the field."""
        return float(np.sum(self.data))

    def project_floor(self, method: str = "max") -> "Field2D":
        """
        Project to 2D by collapsing the Z axis.

        Parameters
        ----------
        method : str
            Projection method: "max", "mean", or "sum".

        Returns
        -------
        Field2D
            2D projection of this field.
        """
        if method == "max":
            data_2d = np.max(self.data, axis=2)
        elif method == "mean":
            data_2d = np.mean(self.data, axis=2)
        elif method == "sum":
            data_2d = np.sum(self.data, axis=2)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        confidence_2d = None
        if self.confidence is not None:
            if method == "max":
                # Use confidence of max voxel
                max_indices = np.argmax(self.data, axis=2)
                i_idx, j_idx = np.meshgrid(
                    np.arange(self.shape[0]),
                    np.arange(self.shape[1]),
                    indexing="ij",
                )
                confidence_2d = self.confidence[i_idx, j_idx, max_indices]
            else:
                confidence_2d = np.mean(self.confidence, axis=2)

        return Field2D(
            data=data_2d,
            origin=(self.origin[0], self.origin[1]),
            resolution=self.resolution,
            confidence=confidence_2d,
            meta=self.meta,
        )

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"Field3D(shape={self.shape}, "
            f"origin={self.origin}, "
            f"resolution={self.resolution})"
        )
