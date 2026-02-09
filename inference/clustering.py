"""
clustering.py

Lightweight clustering utilities to group patterns into distinct clusters
without assigning semantic labels.

Used to estimate the number of distinct activity regions in spatial fields
or to cluster latent vectors for emergent archetypes.

All operations are deterministic. No semantic labels are assigned.
"""

from typing import List, Tuple, Optional, NamedTuple
from collections import deque

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================


class SpatialCluster(NamedTuple):
    """
    Compact representation of a spatial cluster.

    Attributes
    ----------
    centroid : Tuple[float, float]
        Cluster centroid (row, col) in pixel coordinates.
    size : int
        Number of pixels in the cluster.
    total_intensity : float
        Sum of intensity values in the cluster.
    bounding_box : Tuple[int, int, int, int]
        Bounding box (min_row, min_col, max_row, max_col).
    pixels : List[Tuple[int, int]]
        List of (row, col) pixel coordinates in the cluster.
    """
    centroid: Tuple[float, float]
    size: int
    total_intensity: float
    bounding_box: Tuple[int, int, int, int]
    pixels: List[Tuple[int, int]]


class LatentCluster(NamedTuple):
    """
    Compact representation of a latent vector cluster.

    Attributes
    ----------
    center : np.ndarray
        Cluster center in latent space.
    size : int
        Number of vectors in the cluster.
    indices : List[int]
        Indices of vectors assigned to this cluster.
    """
    center: np.ndarray
    size: int
    indices: List[int]


# =============================================================================
# Spatial Clustering (Connected Components)
# =============================================================================


def cluster_spatial_activity(
    activity_map: np.ndarray,
    threshold: float = 0.0,
    min_cluster_size: int = 1,
    connectivity: int = 4,
) -> List[SpatialCluster]:
    """
    Cluster spatial activity using connected component analysis.

    Finds contiguous regions of above-threshold activity in a 2D heatmap.

    Parameters
    ----------
    activity_map : np.ndarray
        2D array of activity intensity values.
    threshold : float, optional
        Minimum intensity to consider as active. Defaults to 0.0.
    min_cluster_size : int, optional
        Minimum number of pixels for a valid cluster. Defaults to 1.
    connectivity : int, optional
        Pixel connectivity: 4 (cardinal directions) or 8 (including diagonals).
        Defaults to 4.

    Returns
    -------
    List[SpatialCluster]
        List of detected clusters, sorted by size (largest first).

    Raises
    ------
    ValueError
        If activity_map is not 2D or connectivity is invalid.
    """
    if activity_map.ndim != 2:
        raise ValueError(
            f"activity_map must be 2D. Got {activity_map.ndim} dimensions."
        )

    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8. Got {connectivity}."
        )

    rows, cols = activity_map.shape

    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # connectivity == 8
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    # Create binary mask of active pixels
    active_mask = activity_map > threshold

    # Track visited pixels
    visited = np.zeros_like(active_mask, dtype=bool)

    clusters = []

    # Flood-fill to find connected components
    for start_row in range(rows):
        for start_col in range(cols):
            if not active_mask[start_row, start_col]:
                continue
            if visited[start_row, start_col]:
                continue

            # BFS to find all connected pixels
            cluster_pixels = []
            queue = deque([(start_row, start_col)])
            visited[start_row, start_col] = True

            while queue:
                r, c = queue.popleft()
                cluster_pixels.append((r, c))

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < rows and 0 <= nc < cols:
                        if active_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            # Filter by minimum size
            if len(cluster_pixels) < min_cluster_size:
                continue

            # Compute cluster properties
            cluster = _build_spatial_cluster(cluster_pixels, activity_map)
            clusters.append(cluster)

    # Sort by size (largest first)
    clusters.sort(key=lambda c: c.size, reverse=True)

    return clusters


def _build_spatial_cluster(
    pixels: List[Tuple[int, int]],
    activity_map: np.ndarray,
) -> SpatialCluster:
    """Build a SpatialCluster from a list of pixels."""
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]

    # Bounding box
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    # Intensity-weighted centroid
    total_intensity = 0.0
    weighted_row = 0.0
    weighted_col = 0.0

    for r, c in pixels:
        intensity = activity_map[r, c]
        total_intensity += intensity
        weighted_row += r * intensity
        weighted_col += c * intensity

    if total_intensity > 0:
        centroid = (weighted_row / total_intensity, weighted_col / total_intensity)
    else:
        # Fallback to geometric center
        centroid = (float(np.mean(rows)), float(np.mean(cols)))

    return SpatialCluster(
        centroid=centroid,
        size=len(pixels),
        total_intensity=total_intensity,
        bounding_box=(min_row, min_col, max_row, max_col),
        pixels=pixels,
    )


def count_activity_regions(
    activity_map: np.ndarray,
    threshold: float = 0.0,
    min_cluster_size: int = 1,
) -> int:
    """
    Count the number of distinct activity regions.

    Convenience function that returns only the count.

    Parameters
    ----------
    activity_map : np.ndarray
        2D array of activity intensity values.
    threshold : float, optional
        Minimum intensity to consider as active.
    min_cluster_size : int, optional
        Minimum pixels for a valid region.

    Returns
    -------
    int
        Number of distinct activity regions.
    """
    clusters = cluster_spatial_activity(activity_map, threshold, min_cluster_size)
    return len(clusters)


def get_cluster_centroids(
    activity_map: np.ndarray,
    threshold: float = 0.0,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """
    Get centroids of all activity clusters.

    Parameters
    ----------
    activity_map : np.ndarray
        2D array of activity intensity values.
    threshold : float, optional
        Minimum intensity to consider as active.
    min_cluster_size : int, optional
        Minimum pixels for a valid cluster.

    Returns
    -------
    np.ndarray
        Array of centroids. Shape: (n_clusters, 2).
        Each row is (row, col) in pixel coordinates.
        Empty array with shape (0, 2) if no clusters.
    """
    clusters = cluster_spatial_activity(activity_map, threshold, min_cluster_size)

    if len(clusters) == 0:
        return np.empty((0, 2))

    centroids = np.array([c.centroid for c in clusters])
    return centroids


# =============================================================================
# Latent Vector Clustering (K-Means)
# =============================================================================


def cluster_latent_vectors(
    vectors: np.ndarray,
    n_clusters: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[LatentCluster]]:
    """
    Cluster latent vectors using k-means algorithm.

    Parameters
    ----------
    vectors : np.ndarray
        Array of latent vectors. Shape: (n_samples, latent_dim).
    n_clusters : int
        Number of clusters to form.
    max_iterations : int, optional
        Maximum iterations. Defaults to 100.
    tolerance : float, optional
        Convergence tolerance for center movement. Defaults to 1e-6.
    random_seed : Optional[int], optional
        Seed for reproducible initialization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[LatentCluster]]
        assignments : Cluster assignment for each vector. Shape: (n_samples,).
        centers : Cluster centers. Shape: (n_clusters, latent_dim).
        clusters : List of LatentCluster objects with detailed info.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if vectors.ndim != 2:
        raise ValueError(
            f"vectors must be 2D (n_samples, latent_dim). "
            f"Got {vectors.ndim} dimensions."
        )

    n_samples, latent_dim = vectors.shape

    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive. Got {n_clusters}.")

    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})."
        )

    # Initialize centers using k-means++ style
    rng = np.random.default_rng(random_seed)
    centers = _kmeans_plusplus_init(vectors, n_clusters, rng)

    assignments = np.zeros(n_samples, dtype=np.int64)

    for iteration in range(max_iterations):
        # Assignment step: assign each vector to nearest center
        new_assignments = _assign_to_nearest(vectors, centers)

        # Update step: compute new centers
        new_centers = _compute_centers(vectors, new_assignments, n_clusters, latent_dim)

        # Check convergence
        center_shift = np.linalg.norm(new_centers - centers)

        centers = new_centers
        assignments = new_assignments

        if center_shift < tolerance:
            break

    # Build cluster objects
    clusters = _build_latent_clusters(vectors, assignments, centers, n_clusters)

    return assignments, centers, clusters


def _kmeans_plusplus_init(
    vectors: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize centers using k-means++ algorithm."""
    n_samples, latent_dim = vectors.shape
    centers = np.zeros((n_clusters, latent_dim))

    # First center: random sample
    first_idx = rng.integers(n_samples)
    centers[0] = vectors[first_idx]

    # Remaining centers: probability proportional to squared distance
    for k in range(1, n_clusters):
        # Compute squared distances to nearest existing center
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist_sq = np.inf
            for j in range(k):
                dist_sq = np.sum((vectors[i] - centers[j]) ** 2)
                min_dist_sq = min(min_dist_sq, dist_sq)
            distances[i] = min_dist_sq

        # Normalize to probability distribution
        prob = distances / distances.sum()

        # Sample next center
        next_idx = rng.choice(n_samples, p=prob)
        centers[k] = vectors[next_idx]

    return centers


def _assign_to_nearest(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each vector to the nearest center."""
    n_samples = vectors.shape[0]
    n_clusters = centers.shape[0]
    assignments = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        min_dist = np.inf
        min_idx = 0
        for k in range(n_clusters):
            dist = np.sum((vectors[i] - centers[k]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_idx = k
        assignments[i] = min_idx

    return assignments


def _compute_centers(
    vectors: np.ndarray,
    assignments: np.ndarray,
    n_clusters: int,
    latent_dim: int,
) -> np.ndarray:
    """Compute cluster centers as mean of assigned vectors."""
    centers = np.zeros((n_clusters, latent_dim))

    for k in range(n_clusters):
        mask = assignments == k
        if np.any(mask):
            centers[k] = vectors[mask].mean(axis=0)
        # Empty cluster: center remains at zero (will be reassigned next iteration)

    return centers


def _build_latent_clusters(
    vectors: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    n_clusters: int,
) -> List[LatentCluster]:
    """Build LatentCluster objects from clustering results."""
    clusters = []

    for k in range(n_clusters):
        indices = np.where(assignments == k)[0].tolist()
        clusters.append(LatentCluster(
            center=centers[k].copy(),
            size=len(indices),
            indices=indices,
        ))

    return clusters


def compute_cluster_inertia(
    vectors: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
) -> float:
    """
    Compute total within-cluster sum of squared distances (inertia).

    Parameters
    ----------
    vectors : np.ndarray
        Array of vectors. Shape: (n_samples, dim).
    assignments : np.ndarray
        Cluster assignments. Shape: (n_samples,).
    centers : np.ndarray
        Cluster centers. Shape: (n_clusters, dim).

    Returns
    -------
    float
        Total inertia (sum of squared distances to assigned centers).
    """
    inertia = 0.0
    for i, k in enumerate(assignments):
        inertia += np.sum((vectors[i] - centers[k]) ** 2)
    return float(inertia)


def find_optimal_k(
    vectors: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    max_iterations: int = 100,
    random_seed: Optional[int] = None,
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow heuristic.

    Returns the k with the largest relative decrease in inertia.

    Parameters
    ----------
    vectors : np.ndarray
        Array of latent vectors. Shape: (n_samples, latent_dim).
    k_range : Tuple[int, int], optional
        Range of k values to try (inclusive). Defaults to (2, 10).
    max_iterations : int, optional
        Max iterations per clustering. Defaults to 100.
    random_seed : Optional[int], optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[int, List[float]]
        optimal_k : Recommended number of clusters.
        inertias : List of inertia values for each k tried.
    """
    k_min, k_max = k_range
    n_samples = vectors.shape[0]

    # Clamp k_max to valid range
    k_max = min(k_max, n_samples)
    k_min = max(k_min, 1)

    if k_min > k_max:
        return k_min, []

    inertias = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        assignments, centers, _ = cluster_latent_vectors(
            vectors, k, max_iterations, random_seed=random_seed
        )
        inertia = compute_cluster_inertia(vectors, assignments, centers)
        inertias.append(inertia)

    # Find elbow: largest relative decrease
    if len(inertias) < 2:
        return k_values[0], inertias

    relative_decreases = []
    for i in range(1, len(inertias)):
        if inertias[i - 1] > 0:
            decrease = (inertias[i - 1] - inertias[i]) / inertias[i - 1]
        else:
            decrease = 0.0
        relative_decreases.append(decrease)

    # Return k corresponding to largest decrease
    best_idx = int(np.argmax(relative_decreases))
    optimal_k = k_values[best_idx + 1]  # +1 because decrease[i] corresponds to k[i+1]

    return optimal_k, inertias
