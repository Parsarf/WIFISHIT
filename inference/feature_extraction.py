"""
feature_extraction.py

Extracts structured numeric features from preprocessed CSI data.

Reduces raw numeric data into stable, informative representations for downstream
learning or inference, without making semantic decisions.

All operations are deterministic and do not perform semantic interpretation.
"""

from typing import Tuple, Optional

import numpy as np


# =============================================================================
# Temporal Features
# =============================================================================


def first_order_difference(data: np.ndarray) -> np.ndarray:
    """
    Compute first-order temporal differences.

    Calculates data[t] - data[t-1] for consecutive time steps.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        First-order differences. Shape: (n_frames - 1, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    return np.diff(data, axis=0)


def second_order_difference(data: np.ndarray) -> np.ndarray:
    """
    Compute second-order temporal differences.

    Calculates the difference of differences (acceleration-like).

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Second-order differences. Shape: (n_frames - 2, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    first_diff = np.diff(data, axis=0)
    return np.diff(first_diff, axis=0)


def short_time_energy(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute short-time energy (sum of squared values).

    Parameters
    ----------
    data : np.ndarray
        Input data. Typically (n_frames, n_subcarriers) or a window.
    axis : int, optional
        Axis along which to compute energy. Defaults to 0 (temporal).

    Returns
    -------
    np.ndarray
        Energy values. Shape depends on input and axis.
        For 2D input with axis=0: (n_subcarriers,).
        For 2D input with axis=1: (n_frames,).
    """
    return np.sum(data ** 2, axis=axis)


def windowed_energy(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute short-time energy over sliding windows.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).
    window_size : int
        Number of frames per window.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        Per-window energy. Shape: (n_windows, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    n_frames, n_subcarriers = data.shape

    if window_size <= 0 or window_size > n_frames:
        raise ValueError(
            f"window_size must be in range [1, {n_frames}]. Got {window_size}."
        )

    n_windows = (n_frames - window_size) // stride + 1
    energy = np.zeros((n_windows, n_subcarriers))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        energy[i] = np.sum(window ** 2, axis=0)

    return energy


def windowed_variance(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute variance over sliding windows.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).
    window_size : int
        Number of frames per window.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        Per-window variance. Shape: (n_windows, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    n_frames, n_subcarriers = data.shape

    if window_size <= 0 or window_size > n_frames:
        raise ValueError(
            f"window_size must be in range [1, {n_frames}]. Got {window_size}."
        )

    n_windows = (n_frames - window_size) // stride + 1
    variance = np.zeros((n_windows, n_subcarriers))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        variance[i] = np.var(window, axis=0)

    return variance


def windowed_mean(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute mean over sliding windows.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).
    window_size : int
        Number of frames per window.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        Per-window mean. Shape: (n_windows, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    n_frames, n_subcarriers = data.shape

    if window_size <= 0 or window_size > n_frames:
        raise ValueError(
            f"window_size must be in range [1, {n_frames}]. Got {window_size}."
        )

    n_windows = (n_frames - window_size) // stride + 1
    means = np.zeros((n_windows, n_subcarriers))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        means[i] = np.mean(window, axis=0)

    return means


def windowed_std(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute standard deviation over sliding windows.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).
    window_size : int
        Number of frames per window.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        Per-window standard deviation. Shape: (n_windows, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    n_frames, n_subcarriers = data.shape

    if window_size <= 0 or window_size > n_frames:
        raise ValueError(
            f"window_size must be in range [1, {n_frames}]. Got {window_size}."
        )

    n_windows = (n_frames - window_size) // stride + 1
    stds = np.zeros((n_windows, n_subcarriers))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        stds[i] = np.std(window, axis=0)

    return stds


# =============================================================================
# Spectral / Subcarrier Features
# =============================================================================


def subcarrier_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between subcarriers.

    Measures pairwise Pearson correlation across time.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Correlation matrix. Shape: (n_subcarriers, n_subcarriers).
        Values in range [-1, 1]. Diagonal is 1.
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    # np.corrcoef expects variables in rows, observations in columns
    # Transpose so subcarriers are rows, frames are columns
    return np.corrcoef(data.T)


def subcarrier_covariance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix between subcarriers.

    Measures pairwise covariance across time.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Covariance matrix. Shape: (n_subcarriers, n_subcarriers).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    # np.cov expects variables in rows, observations in columns
    return np.cov(data.T)


def mean_subcarrier_correlation(data: np.ndarray) -> float:
    """
    Compute mean correlation across all subcarrier pairs.

    Excludes the diagonal (self-correlation).

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    float
        Mean off-diagonal correlation value.
    """
    corr_matrix = subcarrier_correlation_matrix(data)
    n = corr_matrix.shape[0]

    # Extract upper triangle excluding diagonal
    upper_indices = np.triu_indices(n, k=1)
    off_diagonal = corr_matrix[upper_indices]

    return float(np.mean(off_diagonal))


def aggregate_statistics(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aggregated statistics across time for each subcarrier.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        mean : Shape (n_subcarriers,)
        std : Shape (n_subcarriers,)
        min : Shape (n_subcarriers,)
        max : Shape (n_subcarriers,)
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    return (
        np.mean(data, axis=0),
        np.std(data, axis=0),
        np.min(data, axis=0),
        np.max(data, axis=0),
    )


def subcarrier_range(data: np.ndarray) -> np.ndarray:
    """
    Compute the range (max - min) for each subcarrier across time.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Range values. Shape: (n_subcarriers,).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    return np.max(data, axis=0) - np.min(data, axis=0)


def subcarrier_median(data: np.ndarray) -> np.ndarray:
    """
    Compute median for each subcarrier across time.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Median values. Shape: (n_subcarriers,).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    return np.median(data, axis=0)


def inter_subcarrier_variance(data: np.ndarray) -> np.ndarray:
    """
    Compute variance across subcarriers for each time frame.

    Measures how much subcarriers differ from each other at each instant.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Per-frame variance across subcarriers. Shape: (n_frames,).
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    return np.var(data, axis=1)


# =============================================================================
# Combined Feature Extraction
# =============================================================================


def extract_temporal_features(
    data: np.ndarray,
    window_size: Optional[int] = None,
    stride: int = 1,
) -> dict:
    """
    Extract a set of temporal features from CSI data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).
    window_size : Optional[int], optional
        Window size for windowed features. If None, uses n_frames // 4.
    stride : int, optional
        Stride for windowed features. Defaults to 1.

    Returns
    -------
    dict
        Dictionary containing:
        - "first_diff": First-order differences (n_frames-1, n_subcarriers)
        - "second_diff": Second-order differences (n_frames-2, n_subcarriers)
        - "total_energy": Total energy per subcarrier (n_subcarriers,)
        - "windowed_variance": Per-window variance (n_windows, n_subcarriers)
        - "windowed_energy": Per-window energy (n_windows, n_subcarriers)
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    n_frames = data.shape[0]

    if window_size is None:
        window_size = max(1, n_frames // 4)

    features = {
        "first_diff": first_order_difference(data),
        "second_diff": second_order_difference(data),
        "total_energy": short_time_energy(data, axis=0),
        "windowed_variance": windowed_variance(data, window_size, stride),
        "windowed_energy": windowed_energy(data, window_size, stride),
    }

    return features


def extract_spectral_features(data: np.ndarray) -> dict:
    """
    Extract a set of spectral/subcarrier features from CSI data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, n_subcarriers).

    Returns
    -------
    dict
        Dictionary containing:
        - "correlation_matrix": Subcarrier correlation (n_subcarriers, n_subcarriers)
        - "mean_correlation": Scalar mean off-diagonal correlation
        - "mean": Per-subcarrier mean (n_subcarriers,)
        - "std": Per-subcarrier std (n_subcarriers,)
        - "range": Per-subcarrier range (n_subcarriers,)
        - "inter_subcarrier_variance": Per-frame variance (n_frames,)
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_frames, n_subcarriers). "
            f"Got {data.ndim} dimensions."
        )

    mean_vals, std_vals, _, _ = aggregate_statistics(data)

    features = {
        "correlation_matrix": subcarrier_correlation_matrix(data),
        "mean_correlation": mean_subcarrier_correlation(data),
        "mean": mean_vals,
        "std": std_vals,
        "range": subcarrier_range(data),
        "inter_subcarrier_variance": inter_subcarrier_variance(data),
    }

    return features
