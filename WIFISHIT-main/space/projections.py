"""
projections.py

Pure spatial projections of a 3D voxel disturbance grid.

Converts volumetric data into 2D representations for inspection and debugging.
Does not interpret or modify the underlying data.

Assumes voxel grid has shape (nx, ny, nz) where:
- X is the first axis (index 0)
- Y is the second axis (index 1)
- Z is the third axis (index 2, vertical)

Output arrays preserve spatial ordering.
"""

from typing import Protocol, Literal, runtime_checkable

import numpy as np


@runtime_checkable
class VoxelGridProtocol(Protocol):
    """
    Protocol defining the interface for voxel grid access.

    A voxel grid must provide read-only access to its underlying
    3D data array.
    """

    @property
    def data(self) -> np.ndarray:
        """
        The underlying 3D voxel data array.

        Returns
        -------
        np.ndarray
            3D array with shape (nx, ny, nz).
        """
        ...

    @property
    def shape(self) -> tuple:
        """
        Shape of the voxel grid (nx, ny, nz).

        Returns
        -------
        tuple
            Three-element tuple of grid dimensions.
        """
        ...


# Type alias for projection methods
ProjectionMethod = Literal["sum", "max", "mean"]


def _validate_grid(grid: VoxelGridProtocol) -> np.ndarray:
    """
    Validate voxel grid and return its data array.

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to validate.

    Returns
    -------
    np.ndarray
        The grid's 3D data array.

    Raises
    ------
    ValueError
        If the grid data is not 3-dimensional.
    """
    data = grid.data

    if data.ndim != 3:
        raise ValueError(
            f"Voxel grid data must be 3-dimensional. "
            f"Got {data.ndim} dimensions with shape {data.shape}."
        )

    return data


def _apply_reduction(
    data: np.ndarray,
    axis: int,
    method: ProjectionMethod,
) -> np.ndarray:
    """
    Apply a reduction operation along the specified axis.

    Parameters
    ----------
    data : np.ndarray
        The input array.
    axis : int
        The axis along which to reduce.
    method : ProjectionMethod
        The reduction method: "sum", "max", or "mean".

    Returns
    -------
    np.ndarray
        The reduced array.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == "sum":
        return np.sum(data, axis=axis)
    elif method == "max":
        return np.max(data, axis=axis)
    elif method == "mean":
        return np.mean(data, axis=axis)
    else:
        raise ValueError(
            f"Unknown projection method: {method}. "
            f"Must be one of: 'sum', 'max', 'mean'."
        )


def floor_projection(
    grid: VoxelGridProtocol,
    method: ProjectionMethod = "sum",
) -> np.ndarray:
    """
    Compute floor projection by collapsing the Z axis.

    Projects the 3D grid onto the XY plane (floor view, looking down).

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to project.
    method : ProjectionMethod, optional
        Reduction method: "sum", "max", or "mean". Defaults to "sum".

    Returns
    -------
    np.ndarray
        2D array with shape (nx, ny) representing the floor projection.
    """
    data = _validate_grid(grid)

    # Collapse Z axis (axis 2)
    projection = _apply_reduction(data, axis=2, method=method)

    return projection


def vertical_projection_xz(
    grid: VoxelGridProtocol,
    method: ProjectionMethod = "max",
) -> np.ndarray:
    """
    Compute vertical projection onto the XZ plane by collapsing Y axis.

    Projects the 3D grid onto the XZ plane (side view from Y direction).

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to project.
    method : ProjectionMethod, optional
        Reduction method: "sum", "max", or "mean". Defaults to "max".

    Returns
    -------
    np.ndarray
        2D array with shape (nx, nz) representing the XZ projection.
    """
    data = _validate_grid(grid)

    # Collapse Y axis (axis 1)
    projection = _apply_reduction(data, axis=1, method=method)

    return projection


def vertical_projection_yz(
    grid: VoxelGridProtocol,
    method: ProjectionMethod = "max",
) -> np.ndarray:
    """
    Compute vertical projection onto the YZ plane by collapsing X axis.

    Projects the 3D grid onto the YZ plane (side view from X direction).

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to project.
    method : ProjectionMethod, optional
        Reduction method: "sum", "max", or "mean". Defaults to "max".

    Returns
    -------
    np.ndarray
        2D array with shape (ny, nz) representing the YZ projection.
    """
    data = _validate_grid(grid)

    # Collapse X axis (axis 0)
    projection = _apply_reduction(data, axis=0, method=method)

    return projection


def horizontal_slice(
    grid: VoxelGridProtocol,
    z_index: int,
) -> np.ndarray:
    """
    Extract a horizontal slice at a given Z index.

    Returns a copy of the XY plane at the specified height.

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to slice.
    z_index : int
        The Z index of the slice. Must be in range [0, nz).

    Returns
    -------
    np.ndarray
        2D array with shape (nx, ny) representing the horizontal slice.

    Raises
    ------
    IndexError
        If z_index is out of bounds.
    """
    data = _validate_grid(grid)
    nz = data.shape[2]

    if z_index < 0 or z_index >= nz:
        raise IndexError(
            f"z_index {z_index} is out of bounds for grid with nz={nz}. "
            f"Must be in range [0, {nz})."
        )

    # Extract slice and return a copy
    slice_data = data[:, :, z_index].copy()

    return slice_data


def vertical_slice_x(
    grid: VoxelGridProtocol,
    x_index: int,
) -> np.ndarray:
    """
    Extract a vertical slice at a given X index.

    Returns a copy of the YZ plane at the specified X position.

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to slice.
    x_index : int
        The X index of the slice. Must be in range [0, nx).

    Returns
    -------
    np.ndarray
        2D array with shape (ny, nz) representing the vertical slice.

    Raises
    ------
    IndexError
        If x_index is out of bounds.
    """
    data = _validate_grid(grid)
    nx = data.shape[0]

    if x_index < 0 or x_index >= nx:
        raise IndexError(
            f"x_index {x_index} is out of bounds for grid with nx={nx}. "
            f"Must be in range [0, {nx})."
        )

    # Extract slice and return a copy
    slice_data = data[x_index, :, :].copy()

    return slice_data


def vertical_slice_y(
    grid: VoxelGridProtocol,
    y_index: int,
) -> np.ndarray:
    """
    Extract a vertical slice at a given Y index.

    Returns a copy of the XZ plane at the specified Y position.

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to slice.
    y_index : int
        The Y index of the slice. Must be in range [0, ny).

    Returns
    -------
    np.ndarray
        2D array with shape (nx, nz) representing the vertical slice.

    Raises
    ------
    IndexError
        If y_index is out of bounds.
    """
    data = _validate_grid(grid)
    ny = data.shape[1]

    if y_index < 0 or y_index >= ny:
        raise IndexError(
            f"y_index {y_index} is out of bounds for grid with ny={ny}. "
            f"Must be in range [0, {ny})."
        )

    # Extract slice and return a copy
    slice_data = data[:, y_index, :].copy()

    return slice_data


def all_projections(
    grid: VoxelGridProtocol,
    method: ProjectionMethod = "max",
) -> dict:
    """
    Compute all three axis-aligned projections.

    Convenience function returning floor (XY), XZ, and YZ projections.

    Parameters
    ----------
    grid : VoxelGridProtocol
        The voxel grid to project.
    method : ProjectionMethod, optional
        Reduction method for all projections. Defaults to "max".

    Returns
    -------
    dict
        Dictionary with keys "xy", "xz", "yz" mapping to 2D arrays.
    """
    return {
        "xy": floor_projection(grid, method=method),
        "xz": vertical_projection_xz(grid, method=method),
        "yz": vertical_projection_yz(grid, method=method),
    }
