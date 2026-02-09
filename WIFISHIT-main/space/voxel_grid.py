"""
voxel_grid.py

3D discretized spatial representation of continuous disturbance fields.

Provides a bridge between continuous world-space disturbances and
discrete spatial analysis (projections, clustering).

All spatial quantities are in meters.
"""

from typing import Tuple, Optional

import numpy as np

from world.world import World


class VoxelGrid:
    """
    3D voxel grid for discretizing continuous disturbance fields.

    The grid covers a rectangular volume in world space and stores
    scalar disturbance values at each voxel.

    Parameters
    ----------
    origin : Tuple[float, float, float]
        World-space origin (x_min, y_min, z_min) in meters.
    dimensions : Tuple[float, float, float]
        Grid dimensions (x_size, y_size, z_size) in meters.
    resolution : float
        Voxel size in meters. Same for all axes.

    Attributes
    ----------
    origin : Tuple[float, float, float]
        World-space origin in meters.
    dimensions : Tuple[float, float, float]
        Grid dimensions in meters.
    resolution : float
        Voxel size in meters.
    shape : Tuple[int, int, int]
        Grid shape (nx, ny, nz) in voxels.
    data : np.ndarray
        3D array of voxel values. Shape: (nx, ny, nz).
    """

    def __init__(
        self,
        origin: Tuple[float, float, float],
        dimensions: Tuple[float, float, float],
        resolution: float,
    ) -> None:
        """Initialize the voxel grid."""
        if resolution <= 0:
            raise ValueError(f"resolution must be positive. Got {resolution}.")

        self._origin = origin
        self._dimensions = dimensions
        self._resolution = resolution

        # Compute grid shape
        nx = max(1, int(np.ceil(dimensions[0] / resolution)))
        ny = max(1, int(np.ceil(dimensions[1] / resolution)))
        nz = max(1, int(np.ceil(dimensions[2] / resolution)))
        self._shape = (nx, ny, nz)

        # Initialize data array
        self._data = np.zeros(self._shape, dtype=np.float64)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """World-space origin (x_min, y_min, z_min) in meters."""
        return self._origin

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Grid dimensions (x_size, y_size, z_size) in meters."""
        return self._dimensions

    @property
    def resolution(self) -> float:
        """Voxel size in meters."""
        return self._resolution

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (nx, ny, nz) in voxels."""
        return self._shape

    @property
    def data(self) -> np.ndarray:
        """3D array of voxel values. Shape: (nx, ny, nz)."""
        return self._data

    def clear(self) -> None:
        """Set all voxel values to zero."""
        self._data.fill(0.0)

    def decay(self, factor: float) -> None:
        """
        Apply exponential decay to all voxel values.

        Parameters
        ----------
        factor : float
            Decay multiplier (0 < factor <= 1).
        """
        self._data *= factor

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
        x = self._origin[0] + (i + 0.5) * self._resolution
        y = self._origin[1] + (j + 0.5) * self._resolution
        z = self._origin[2] + (k + 0.5) * self._resolution
        return (x, y, z)

    def world_to_voxel(
        self,
        x: float,
        y: float,
        z: float,
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
        i = int((x - self._origin[0]) / self._resolution)
        j = int((y - self._origin[1]) / self._resolution)
        k = int((z - self._origin[2]) / self._resolution)

        if 0 <= i < self._shape[0] and 0 <= j < self._shape[1] and 0 <= k < self._shape[2]:
            return (i, j, k)
        return None

    def sample_world(self, world: World) -> None:
        """
        Sample the continuous world disturbance field into the voxel grid.

        Queries world.disturbance_at() at each voxel center and stores
        the result. Previous values are overwritten.

        Parameters
        ----------
        world : World
            The world to sample from.
        """
        nx, ny, nz = self._shape

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.voxel_center(i, j, k)
                    disturbance = world.disturbance_at(x, y, z)
                    self._data[i, j, k] = disturbance

    def accumulate_world(self, world: World, weight: float = 1.0) -> None:
        """
        Accumulate world disturbance values into the voxel grid.

        Adds weighted disturbance values to existing voxel values
        instead of overwriting.

        Parameters
        ----------
        world : World
            The world to sample from.
        weight : float, optional
            Weight to apply to sampled values. Defaults to 1.0.
        """
        nx, ny, nz = self._shape

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.voxel_center(i, j, k)
                    disturbance = world.disturbance_at(x, y, z)
                    self._data[i, j, k] += disturbance * weight

    def get_value(self, i: int, j: int, k: int) -> float:
        """Get value at voxel indices."""
        return self._data[i, j, k]

    def set_value(self, i: int, j: int, k: int, value: float) -> None:
        """Set value at voxel indices."""
        self._data[i, j, k] = value

    def max_value(self) -> float:
        """Get maximum value in the grid."""
        return float(np.max(self._data))

    def total_value(self) -> float:
        """Get sum of all values in the grid."""
        return float(np.sum(self._data))
