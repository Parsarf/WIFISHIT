"""
render_2d.py

Simple 2D visualization of spatial activity data.

Renders floor-projected activity maps and optional cluster summaries into
human-interpretable images. Performs no inference or data modification.

All rendering is deterministic given the same inputs.
"""

from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap


def render_heatmap(
    activity_map: np.ndarray,
    title: str = "Activity Heatmap",
    xlabel: str = "X (pixels)",
    ylabel: str = "Y (pixels)",
    colorbar_label: str = "Intensity",
    cmap: Union[str, Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    show: bool = False,
) -> Figure:
    """
    Render a 2D activity heatmap.

    Parameters
    ----------
    activity_map : np.ndarray
        2D array of activity intensity values.
    title : str, optional
        Plot title. Defaults to "Activity Heatmap".
    xlabel : str, optional
        X-axis label. Defaults to "X (pixels)".
    ylabel : str, optional
        Y-axis label. Defaults to "Y (pixels)".
    colorbar_label : str, optional
        Colorbar label. Defaults to "Intensity".
    cmap : Union[str, Colormap], optional
        Colormap name or instance. Defaults to "viridis".
    vmin : Optional[float], optional
        Minimum value for color scaling. If None, uses data min.
    vmax : Optional[float], optional
        Maximum value for color scaling. If None, uses data max.
    figsize : Tuple[float, float], optional
        Figure size in inches. Defaults to (8, 6).
    show : bool, optional
        If True, call plt.show(). Defaults to False.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If activity_map is not 2D.
    """
    if activity_map.ndim != 2:
        raise ValueError(
            f"activity_map must be 2D. Got {activity_map.ndim} dimensions."
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Render heatmap with origin at lower-left for intuitive spatial coordinates
    im = ax.imshow(
        activity_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_heatmap_with_clusters(
    activity_map: np.ndarray,
    cluster_centroids: Optional[np.ndarray] = None,
    cluster_sizes: Optional[List[int]] = None,
    cluster_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    title: str = "Activity Heatmap with Clusters",
    xlabel: str = "X (pixels)",
    ylabel: str = "Y (pixels)",
    colorbar_label: str = "Intensity",
    cmap: Union[str, Colormap] = "viridis",
    centroid_color: str = "red",
    centroid_marker: str = "x",
    centroid_size: float = 100,
    bbox_color: str = "red",
    bbox_linewidth: float = 2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    show: bool = False,
) -> Figure:
    """
    Render a 2D activity heatmap with cluster overlay.

    Parameters
    ----------
    activity_map : np.ndarray
        2D array of activity intensity values.
    cluster_centroids : Optional[np.ndarray], optional
        Array of cluster centroids. Shape: (n_clusters, 2).
        Each row is (row, col) in pixel coordinates.
    cluster_sizes : Optional[List[int]], optional
        Size of each cluster (for annotation).
    cluster_bboxes : Optional[List[Tuple[int, int, int, int]]], optional
        Bounding boxes as (min_row, min_col, max_row, max_col).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    colorbar_label : str, optional
        Colorbar label.
    cmap : Union[str, Colormap], optional
        Colormap name or instance.
    centroid_color : str, optional
        Color for centroid markers. Defaults to "red".
    centroid_marker : str, optional
        Marker style for centroids. Defaults to "x".
    centroid_size : float, optional
        Marker size for centroids. Defaults to 100.
    bbox_color : str, optional
        Color for bounding box edges. Defaults to "red".
    bbox_linewidth : float, optional
        Line width for bounding boxes. Defaults to 2.
    vmin : Optional[float], optional
        Minimum value for color scaling.
    vmax : Optional[float], optional
        Maximum value for color scaling.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if activity_map.ndim != 2:
        raise ValueError(
            f"activity_map must be 2D. Got {activity_map.ndim} dimensions."
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Render heatmap
    im = ax.imshow(
        activity_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
    )

    # Overlay cluster centroids
    if cluster_centroids is not None and len(cluster_centroids) > 0:
        # Note: centroids are (row, col), imshow with origin="lower" uses (y, x)
        # row corresponds to y, col corresponds to x
        rows = cluster_centroids[:, 0]
        cols = cluster_centroids[:, 1]

        ax.scatter(
            cols, rows,
            c=centroid_color,
            marker=centroid_marker,
            s=centroid_size,
            label="Centroids",
            zorder=10,
        )

        # Add size annotations if provided
        if cluster_sizes is not None:
            for i, (row, col) in enumerate(cluster_centroids):
                if i < len(cluster_sizes):
                    ax.annotate(
                        f"n={cluster_sizes[i]}",
                        (col, row),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        color=centroid_color,
                    )

    # Overlay bounding boxes
    if cluster_bboxes is not None:
        for bbox in cluster_bboxes:
            min_row, min_col, max_row, max_col = bbox
            width = max_col - min_col
            height = max_row - min_row

            rect = mpatches.Rectangle(
                (min_col, min_row),
                width,
                height,
                linewidth=bbox_linewidth,
                edgecolor=bbox_color,
                facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    # Add legend if centroids were plotted
    if cluster_centroids is not None and len(cluster_centroids) > 0:
        ax.legend(loc="upper right")

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_multiple_heatmaps(
    activity_maps: List[np.ndarray],
    titles: Optional[List[str]] = None,
    main_title: str = "Activity Maps",
    cmap: Union[str, Colormap] = "viridis",
    shared_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ncols: int = 3,
    show: bool = False,
) -> Figure:
    """
    Render multiple 2D heatmaps in a grid.

    Parameters
    ----------
    activity_maps : List[np.ndarray]
        List of 2D activity arrays.
    titles : Optional[List[str]], optional
        Title for each subplot. If None, uses index numbers.
    main_title : str, optional
        Overall figure title.
    cmap : Union[str, Colormap], optional
        Colormap name or instance.
    shared_colorbar : bool, optional
        If True, use shared color scale across all maps. Defaults to True.
    vmin : Optional[float], optional
        Minimum value for color scaling. If None and shared_colorbar,
        uses global min.
    vmax : Optional[float], optional
        Maximum value for color scaling. If None and shared_colorbar,
        uses global max.
    figsize : Optional[Tuple[float, float]], optional
        Figure size. If None, auto-computed from grid size.
    ncols : int, optional
        Number of columns in grid. Defaults to 3.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    n_maps = len(activity_maps)

    if n_maps == 0:
        raise ValueError("activity_maps cannot be empty.")

    # Compute grid dimensions
    nrows = (n_maps + ncols - 1) // ncols

    # Auto-compute figure size
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows + 0.5)

    # Compute shared color limits if needed
    if shared_colorbar:
        if vmin is None:
            vmin = min(m.min() for m in activity_maps)
        if vmax is None:
            vmax = max(m.max() for m in activity_maps)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    images = []
    for i, activity_map in enumerate(activity_maps):
        ax = axes_flat[i]

        im = ax.imshow(
            activity_map,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
        )
        images.append(im)

        # Set title
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Map {i}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Hide unused subplots
    for i in range(n_maps, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add shared colorbar
    if shared_colorbar and len(images) > 0:
        fig.colorbar(images[0], ax=axes, shrink=0.8, label="Intensity")

    fig.suptitle(main_title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_projections(
    xy_projection: np.ndarray,
    xz_projection: Optional[np.ndarray] = None,
    yz_projection: Optional[np.ndarray] = None,
    title: str = "Spatial Projections",
    cmap: Union[str, Colormap] = "viridis",
    figsize: Tuple[float, float] = (12, 4),
    show: bool = False,
) -> Figure:
    """
    Render spatial projections (floor and side views).

    Parameters
    ----------
    xy_projection : np.ndarray
        Floor projection (XY plane). 2D array.
    xz_projection : Optional[np.ndarray], optional
        Side projection (XZ plane). 2D array.
    yz_projection : Optional[np.ndarray], optional
        Side projection (YZ plane). 2D array.
    title : str, optional
        Overall figure title.
    cmap : Union[str, Colormap], optional
        Colormap name or instance.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Determine number of subplots
    n_plots = 1
    if xz_projection is not None:
        n_plots += 1
    if yz_projection is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # XY (floor) projection
    ax = axes[plot_idx]
    im = ax.imshow(xy_projection, cmap=cmap, origin="lower", aspect="auto")
    ax.set_title("Floor (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plot_idx += 1

    # XZ projection
    if xz_projection is not None:
        ax = axes[plot_idx]
        im = ax.imshow(xz_projection, cmap=cmap, origin="lower", aspect="auto")
        ax.set_title("Side (XZ)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plot_idx += 1

    # YZ projection
    if yz_projection is not None:
        ax = axes[plot_idx]
        im = ax.imshow(yz_projection, cmap=cmap, origin="lower", aspect="auto")
        ax.set_title("Side (YZ)")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plot_idx += 1

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_time_series(
    values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Time Series",
    xlabel: str = "Time (s)",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (10, 4),
    show: bool = False,
) -> Figure:
    """
    Render a time series plot.

    Useful for visualizing inference outputs over time.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to plot.
    timestamps : Optional[np.ndarray], optional
        1D array of timestamps. If None, uses integer indices.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1D. Got {values.ndim} dimensions.")

    if timestamps is None:
        timestamps = np.arange(len(values))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, values, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def render_multiple_time_series(
    values_dict: dict,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Time Series",
    xlabel: str = "Time (s)",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (10, 4),
    show: bool = False,
) -> Figure:
    """
    Render multiple time series on the same plot.

    Parameters
    ----------
    values_dict : dict
        Dictionary mapping series names to 1D arrays.
    timestamps : Optional[np.ndarray], optional
        1D array of timestamps. If None, uses integer indices.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : Tuple[float, float], optional
        Figure size in inches.
    show : bool, optional
        If True, call plt.show().

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in values_dict.items():
        if timestamps is None:
            ts = np.arange(len(values))
        else:
            ts = timestamps[:len(values)]

        ax.plot(ts, values, label=name, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        Output file path (e.g., "output.png").
    dpi : int, optional
        Resolution in dots per inch. Defaults to 150.
    transparent : bool, optional
        If True, save with transparent background. Defaults to False.
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
