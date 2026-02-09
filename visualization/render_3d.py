"""
render_3d.py

Basic 3D visualization of volumetric activity data stored in a voxel grid.

Renders the spatial distribution of disturbance values in three dimensions
to aid debugging, intuition, and demonstration.

Performs no inference or data modification.
All rendering is deterministic given the same input.
"""

from typing import Optional, Tuple, Union, Protocol, runtime_checkable

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.colors import Colormap, Normalize


@runtime_checkable
class VoxelGridProtocol(Protocol):
    """
    Protocol defining the interface for voxel grid access.

    A voxel grid must provide read-only access to its underlying
    3D data array and spatial parameters.
    """

    @property
    def data(self) -> np.ndarray:
        """The underlying 3D voxel data array. Shape: (nx, ny, nz)."""
        ...

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the voxel grid (nx, ny, nz)."""
        ...

    @property
    def resolution(self) -> float:
        """Voxel size in meters."""
        ...

    @property
    def origin(self) -> Tuple[float, float, float]:
        """World-space origin (x, y, z) in meters."""
        ...


def _voxel_indices_to_world(
    indices: np.ndarray,
    origin: Tuple[float, float, float],
    resolution: float,
) -> np.ndarray:
    """
    Convert voxel indices to world-space coordinates.

    Parameters
    ----------
    indices : np.ndarray
        Array of voxel indices. Shape: (n_points, 3).
    origin : Tuple[float, float, float]
        World-space origin in meters.
    resolution : float
        Voxel size in meters.

    Returns
    -------
    np.ndarray
        World-space coordinates in meters. Shape: (n_points, 3).
    """
    origin_array = np.array(origin)
    # Center of voxel: origin + (index + 0.5) * resolution
    return origin_array + (indices + 0.5) * resolution


def render_voxel_grid_scatter(
    grid: VoxelGridProtocol,
    min_value: Optional[float] = None,
    max_points: int = 10000,
    title: str = "3D Voxel Grid",
    cmap: Union[str, Colormap] = "viridis",
    point_size: float = 20,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (10, 8),
    view_elev: float = 30,
    view_azim: float = 45,
    show: bool = False,
) -> Figure:
    """
    Render a 3D voxel grid as a scatter plot.

    Each voxel with value above min_value is rendered as a point,
    colored by its intensity value.

    Parameters
    ----------
    grid : VoxelGridProtocol
        Voxel grid to render.
    min_value : Optional[float], optional
        Minimum value for a voxel to be rendered.
        If None, renders all non-zero voxels.
    max_points : int, optional
        Maximum number of points to render (for performance).
        If exceeded, samples uniformly. Defaults to 10000.
    title : str, optional
        Plot title.
    cmap : Union[str, Colormap], optional
        Colormap for intensity values.
    point_size : float, optional
        Size of scatter points.
    alpha : float, optional
        Point transparency (0 to 1).
    figsize : Tuple[float, float], optional
        Figure size in inches.
    view_elev : float, optional
        Elevation angle for 3D view in degrees.
    view_azim : float, optional
        Azimuth angle for 3D view in degrees.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    data = grid.data
    resolution = grid.resolution
    origin = grid.origin

    # Determine threshold
    if min_value is None:
        min_value = 0.0

    # Find voxels above threshold
    mask = data > min_value
    indices = np.argwhere(mask)
    values = data[mask]

    if len(indices) == 0:
        # No points to render
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{title} (no points above threshold)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if show:
            plt.show()

        return fig

    # Subsample if too many points
    if len(indices) > max_points:
        rng = np.random.default_rng(42)  # Fixed seed for determinism
        sample_idx = rng.choice(len(indices), size=max_points, replace=False)
        sample_idx = np.sort(sample_idx)  # Sort for determinism
        indices = indices[sample_idx]
        values = values[sample_idx]

    # Convert to world coordinates
    world_coords = _voxel_indices_to_world(indices, origin, resolution)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    scatter = ax.scatter(
        world_coords[:, 0],
        world_coords[:, 1],
        world_coords[:, 2],
        c=values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
    )

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Intensity")

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_voxel_grid_slices(
    grid: VoxelGridProtocol,
    n_slices: int = 5,
    axis: int = 2,
    title: str = "Voxel Grid Slices",
    cmap: Union[str, Colormap] = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = False,
) -> Figure:
    """
    Render multiple slices through a 3D voxel grid.

    Parameters
    ----------
    grid : VoxelGridProtocol
        Voxel grid to render.
    n_slices : int, optional
        Number of slices to display. Defaults to 5.
    axis : int, optional
        Axis to slice along (0=X, 1=Y, 2=Z). Defaults to 2 (Z).
    title : str, optional
        Overall figure title.
    cmap : Union[str, Colormap], optional
        Colormap for intensity values.
    figsize : Optional[Tuple[float, float]], optional
        Figure size. If None, auto-computed.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    data = grid.data
    resolution = grid.resolution
    origin = grid.origin
    shape = grid.shape

    axis_names = ["X", "Y", "Z"]
    slice_axis_name = axis_names[axis]

    # Compute slice indices
    axis_size = shape[axis]
    slice_indices = np.linspace(0, axis_size - 1, n_slices, dtype=int)

    # Compute slice positions in world coordinates
    slice_positions = origin[axis] + (slice_indices + 0.5) * resolution

    # Determine figure layout
    ncols = min(n_slices, 5)
    nrows = (n_slices + ncols - 1) // ncols

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows + 0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # Compute global color limits
    vmin = data.min()
    vmax = data.max()

    for i, (slice_idx, slice_pos) in enumerate(zip(slice_indices, slice_positions)):
        ax = axes_flat[i]

        # Extract slice
        if axis == 0:
            slice_data = data[slice_idx, :, :]
            xlabel, ylabel = "Y (m)", "Z (m)"
        elif axis == 1:
            slice_data = data[:, slice_idx, :]
            xlabel, ylabel = "X (m)", "Z (m)"
        else:  # axis == 2
            slice_data = data[:, :, slice_idx]
            xlabel, ylabel = "X (m)", "Y (m)"

        im = ax.imshow(
            slice_data.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
        )

        ax.set_title(f"{slice_axis_name}={slice_pos:.2f}m")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide unused axes
    for i in range(n_slices, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add shared colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label="Intensity")

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_voxel_grid_isosurface(
    grid: VoxelGridProtocol,
    level: float,
    title: str = "3D Isosurface",
    color: str = "blue",
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (10, 8),
    view_elev: float = 30,
    view_azim: float = 45,
    show: bool = False,
) -> Figure:
    """
    Render an isosurface of the voxel grid at a given level.

    Uses a simplified wireframe approximation by rendering faces of
    voxels at the boundary of the level set.

    Parameters
    ----------
    grid : VoxelGridProtocol
        Voxel grid to render.
    level : float
        Isosurface level (threshold value).
    title : str, optional
        Plot title.
    color : str, optional
        Surface color.
    alpha : float, optional
        Surface transparency.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    view_elev : float, optional
        Elevation angle for 3D view.
    view_azim : float, optional
        Azimuth angle for 3D view.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    data = grid.data
    resolution = grid.resolution
    origin = grid.origin

    # Find boundary voxels (above level with at least one neighbor below)
    above = data >= level
    boundary_indices = _find_boundary_voxels(above)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if len(boundary_indices) == 0:
        ax.set_title(f"{title} (no surface at level {level})")
    else:
        # Convert to world coordinates
        world_coords = _voxel_indices_to_world(boundary_indices, origin, resolution)

        # Render as scatter (simplified isosurface representation)
        ax.scatter(
            world_coords[:, 0],
            world_coords[:, 1],
            world_coords[:, 2],
            c=color,
            s=50,
            alpha=alpha,
        )

        ax.set_title(f"{title} (level={level})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=view_elev, azim=view_azim)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def _find_boundary_voxels(mask: np.ndarray) -> np.ndarray:
    """
    Find voxels that are True and adjacent to at least one False voxel.

    Parameters
    ----------
    mask : np.ndarray
        3D boolean array.

    Returns
    -------
    np.ndarray
        Indices of boundary voxels. Shape: (n_boundary, 3).
    """
    nx, ny, nz = mask.shape
    boundary = []

    # Pad mask to handle edges
    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i, j, k]:
                    continue

                # Check 6-connected neighbors (in padded coordinates)
                pi, pj, pk = i + 1, j + 1, k + 1

                neighbors = [
                    padded[pi - 1, pj, pk],
                    padded[pi + 1, pj, pk],
                    padded[pi, pj - 1, pk],
                    padded[pi, pj + 1, pk],
                    padded[pi, pj, pk - 1],
                    padded[pi, pj, pk + 1],
                ]

                # If any neighbor is False, this is a boundary voxel
                if not all(neighbors):
                    boundary.append([i, j, k])

    return np.array(boundary) if boundary else np.empty((0, 3), dtype=int)


def render_voxel_grid_orthographic(
    grid: VoxelGridProtocol,
    min_value: Optional[float] = None,
    title: str = "Orthographic Views",
    cmap: Union[str, Colormap] = "viridis",
    figsize: Tuple[float, float] = (15, 5),
    show: bool = False,
) -> Figure:
    """
    Render three orthographic projections (max intensity) of the voxel grid.

    Parameters
    ----------
    grid : VoxelGridProtocol
        Voxel grid to render.
    min_value : Optional[float], optional
        Minimum value to include in projections.
    title : str, optional
        Overall figure title.
    cmap : Union[str, Colormap], optional
        Colormap for intensity values.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    data = grid.data.copy()

    # Apply minimum value mask
    if min_value is not None:
        data = np.where(data > min_value, data, 0)

    # Compute max projections
    xy_proj = np.max(data, axis=2)  # Floor view
    xz_proj = np.max(data, axis=1)  # Side view (Y collapsed)
    yz_proj = np.max(data, axis=0)  # Side view (X collapsed)

    resolution = grid.resolution
    origin = grid.origin
    shape = grid.shape

    # Compute extent for proper axis scaling
    x_extent = [origin[0], origin[0] + shape[0] * resolution]
    y_extent = [origin[1], origin[1] + shape[1] * resolution]
    z_extent = [origin[2], origin[2] + shape[2] * resolution]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Global color limits
    vmin = 0
    vmax = max(xy_proj.max(), xz_proj.max(), yz_proj.max())

    # XY projection (floor)
    im0 = axes[0].imshow(
        xy_proj.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
    )
    axes[0].set_title("Floor (XY)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    # XZ projection
    im1 = axes[1].imshow(
        xz_proj.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        extent=[x_extent[0], x_extent[1], z_extent[0], z_extent[1]],
    )
    axes[1].set_title("Side (XZ)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")

    # YZ projection
    im2 = axes[2].imshow(
        yz_proj.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        extent=[y_extent[0], y_extent[1], z_extent[0], z_extent[1]],
    )
    axes[2].set_title("Side (YZ)")
    axes[2].set_xlabel("Y (m)")
    axes[2].set_ylabel("Z (m)")

    # Add shared colorbar
    fig.colorbar(im0, ax=axes, shrink=0.8, label="Max Intensity")

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_voxel_comparison(
    grid1: VoxelGridProtocol,
    grid2: VoxelGridProtocol,
    labels: Tuple[str, str] = ("Grid 1", "Grid 2"),
    min_value: Optional[float] = None,
    title: str = "Voxel Grid Comparison",
    cmap: Union[str, Colormap] = "viridis",
    figsize: Tuple[float, float] = (12, 5),
    view_elev: float = 30,
    view_azim: float = 45,
    show: bool = False,
) -> Figure:
    """
    Render two voxel grids side by side for comparison.

    Parameters
    ----------
    grid1 : VoxelGridProtocol
        First voxel grid.
    grid2 : VoxelGridProtocol
        Second voxel grid.
    labels : Tuple[str, str], optional
        Labels for each grid.
    min_value : Optional[float], optional
        Minimum value for rendering.
    title : str, optional
        Overall figure title.
    cmap : Union[str, Colormap], optional
        Colormap for intensity values.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    view_elev : float, optional
        Elevation angle for 3D view.
    view_azim : float, optional
        Azimuth angle for 3D view.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=figsize)

    grids = [grid1, grid2]

    # Compute shared color limits
    if min_value is None:
        min_value = 0.0

    vmin = min_value
    vmax = max(grid1.data.max(), grid2.data.max())

    for idx, (grid, label) in enumerate(zip(grids, labels)):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

        data = grid.data
        resolution = grid.resolution
        origin = grid.origin

        # Find visible voxels
        mask = data > min_value
        indices = np.argwhere(mask)
        values = data[mask]

        if len(indices) > 0:
            # Subsample if needed
            if len(indices) > 5000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(indices), size=5000, replace=False)
                sample_idx = np.sort(sample_idx)
                indices = indices[sample_idx]
                values = values[sample_idx]

            world_coords = _voxel_indices_to_world(indices, origin, resolution)

            scatter = ax.scatter(
                world_coords[:, 0],
                world_coords[:, 1],
                world_coords[:, 2],
                c=values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=15,
                alpha=0.6,
            )

        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=view_elev, azim=view_azim)

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def save_figure(
    fig: Figure,
    filepath: str,
    dpi: int = 150,
    transparent: bool = False,
) -> None:
    """
    Save a figure to file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    filepath : str
        Output file path.
    dpi : int, optional
        Resolution in dots per inch.
    transparent : bool, optional
        If True, save with transparent background.
    """
    fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches="tight")


def close_figure(fig: Figure) -> None:
    """
    Close a figure to free memory.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to close.
    """
    plt.close(fig)
