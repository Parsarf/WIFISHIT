"""
preprocessing.py

Low-level preprocessing of CSI-derived numeric data to remove nuisance variation
and produce stable, comparable representations.

Prepares data for feature extraction and learning without performing inference
or interpretation.

All operations are deterministic and do not invent information.
"""

from typing import List, Optional, Tuple

import numpy as np

from csi.csi_frame import CSIFrame


# =============================================================================
# Phase Preprocessing
# =============================================================================


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase values to remove 2π discontinuities.

    Applies numpy's unwrap along the last axis, correcting for phase jumps
    greater than π.

    Parameters
    ----------
    phase : np.ndarray
        Phase values in radians. Can be 1D (single frame) or 2D (time x subcarriers).

    Returns
    -------
    np.ndarray
        Unwrapped phase values. Same shape as input.
    """
    return np.unwrap(phase, axis=-1)


def unwrap_phase_temporal(phase_sequence: np.ndarray) -> np.ndarray:
    """
    Unwrap phase values across time (first axis).

    Corrects for phase jumps between consecutive frames for each subcarrier.

    Parameters
    ----------
    phase_sequence : np.ndarray
        Phase values with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Temporally unwrapped phase. Shape: (n_frames, n_subcarriers).
    """
    if phase_sequence.ndim != 2:
        raise ValueError(
            f"phase_sequence must be 2D (n_frames, n_subcarriers). "
            f"Got {phase_sequence.ndim} dimensions."
        )

    return np.unwrap(phase_sequence, axis=0)


def difference_phase_temporal(phase_sequence: np.ndarray) -> np.ndarray:
    """
    Compute phase differences between consecutive frames.

    Returns the temporal derivative approximation (phase[t] - phase[t-1]).

    Parameters
    ----------
    phase_sequence : np.ndarray
        Phase values with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Phase differences. Shape: (n_frames - 1, n_subcarriers).
    """
    if phase_sequence.ndim != 2:
        raise ValueError(
            f"phase_sequence must be 2D (n_frames, n_subcarriers). "
            f"Got {phase_sequence.ndim} dimensions."
        )

    return np.diff(phase_sequence, axis=0)


def difference_phase_subcarrier(phase: np.ndarray) -> np.ndarray:
    """
    Compute phase differences between adjacent subcarriers.

    Returns the frequency derivative approximation (phase[k] - phase[k-1]).

    Parameters
    ----------
    phase : np.ndarray
        Phase values. Can be 1D (n_subcarriers,) or 2D (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Phase differences across subcarriers.
        Shape: (n_subcarriers - 1,) or (n_frames, n_subcarriers - 1).
    """
    return np.diff(phase, axis=-1)


def remove_phase_offset(phase_sequence: np.ndarray) -> np.ndarray:
    """
    Remove the mean phase offset from each frame.

    Centers phase values around zero for each frame independently.

    Parameters
    ----------
    phase_sequence : np.ndarray
        Phase values with shape (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Phase with per-frame mean removed. Same shape as input.
    """
    if phase_sequence.ndim != 2:
        raise ValueError(
            f"phase_sequence must be 2D (n_frames, n_subcarriers). "
            f"Got {phase_sequence.ndim} dimensions."
        )

    mean_per_frame = np.mean(phase_sequence, axis=1, keepdims=True)
    return phase_sequence - mean_per_frame


# =============================================================================
# Amplitude Preprocessing
# =============================================================================


def normalize_amplitude(amplitude: np.ndarray) -> np.ndarray:
    """
    Normalize amplitude to unit sum per frame.

    Divides each frame's amplitude by its sum, producing a distribution
    over subcarriers.

    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude values. Can be 1D (n_subcarriers,) or 2D (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Normalized amplitude. Same shape as input.
        Returns zeros if input sum is zero.
    """
    axis = -1
    total = np.sum(amplitude, axis=axis, keepdims=True)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(total != 0, amplitude / total, 0.0)

    return normalized


def normalize_amplitude_minmax(amplitude: np.ndarray) -> np.ndarray:
    """
    Normalize amplitude to [0, 1] range per frame.

    Applies min-max scaling independently to each frame.

    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude values. Can be 1D (n_subcarriers,) or 2D (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Amplitude scaled to [0, 1]. Same shape as input.
        Returns zeros if min equals max.
    """
    axis = -1
    min_val = np.min(amplitude, axis=axis, keepdims=True)
    max_val = np.max(amplitude, axis=axis, keepdims=True)
    range_val = max_val - min_val

    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(range_val != 0, (amplitude - min_val) / range_val, 0.0)

    return normalized


def normalize_amplitude_zscore(amplitude: np.ndarray) -> np.ndarray:
    """
    Normalize amplitude using z-score standardization per frame.

    Centers to zero mean and unit standard deviation.

    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude values. Can be 1D (n_subcarriers,) or 2D (n_frames, n_subcarriers).

    Returns
    -------
    np.ndarray
        Z-score normalized amplitude. Same shape as input.
        Returns zeros if standard deviation is zero.
    """
    axis = -1
    mean_val = np.mean(amplitude, axis=axis, keepdims=True)
    std_val = np.std(amplitude, axis=axis, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(std_val != 0, (amplitude - mean_val) / std_val, 0.0)

    return normalized


def subtract_baseline(
    amplitude_sequence: np.ndarray,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Subtract a baseline from amplitude measurements.

    If no baseline is provided, uses the mean across all frames.

    Parameters
    ----------
    amplitude_sequence : np.ndarray
        Amplitude values with shape (n_frames, n_subcarriers).
    baseline : Optional[np.ndarray], optional
        Baseline to subtract. Shape: (n_subcarriers,).
        If None, computes mean across frames.

    Returns
    -------
    np.ndarray
        Amplitude with baseline removed. Shape: (n_frames, n_subcarriers).
    """
    if amplitude_sequence.ndim != 2:
        raise ValueError(
            f"amplitude_sequence must be 2D (n_frames, n_subcarriers). "
            f"Got {amplitude_sequence.ndim} dimensions."
        )

    if baseline is None:
        baseline = np.mean(amplitude_sequence, axis=0)

    if baseline.shape[0] != amplitude_sequence.shape[1]:
        raise ValueError(
            f"Baseline length {baseline.shape[0]} does not match "
            f"number of subcarriers {amplitude_sequence.shape[1]}."
        )

    return amplitude_sequence - baseline


def compute_amplitude_ratio(
    amplitude_sequence: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute amplitude ratio relative to a reference.

    If no reference is provided, uses the first frame.

    Parameters
    ----------
    amplitude_sequence : np.ndarray
        Amplitude values with shape (n_frames, n_subcarriers).
    reference : Optional[np.ndarray], optional
        Reference amplitude. Shape: (n_subcarriers,).
        If None, uses first frame.

    Returns
    -------
    np.ndarray
        Amplitude ratios. Shape: (n_frames, n_subcarriers).
        Returns ones where reference is zero.
    """
    if amplitude_sequence.ndim != 2:
        raise ValueError(
            f"amplitude_sequence must be 2D (n_frames, n_subcarriers). "
            f"Got {amplitude_sequence.ndim} dimensions."
        )

    if reference is None:
        reference = amplitude_sequence[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(reference != 0, amplitude_sequence / reference, 1.0)

    return ratio


# =============================================================================
# Windowing Utilities
# =============================================================================


def frames_to_arrays(
    frames: List[CSIFrame],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of CSI frames to stacked arrays.

    Parameters
    ----------
    frames : List[CSIFrame]
        List of CSI frames.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        timestamps : Shape (n_frames,)
        amplitudes : Shape (n_frames, n_subcarriers)
        phases : Shape (n_frames, n_subcarriers)

    Raises
    ------
    ValueError
        If frames list is empty or frames have inconsistent shapes.
    """
    if len(frames) == 0:
        raise ValueError("frames list cannot be empty.")

    n_subcarriers = frames[0].num_subcarriers

    timestamps = np.zeros(len(frames))
    amplitudes = np.zeros((len(frames), n_subcarriers))
    phases = np.zeros((len(frames), n_subcarriers))

    for i, frame in enumerate(frames):
        if frame.num_subcarriers != n_subcarriers:
            raise ValueError(
                f"Frame {i} has {frame.num_subcarriers} subcarriers, "
                f"expected {n_subcarriers}."
            )

        timestamps[i] = frame.timestamp
        amplitudes[i] = frame.amplitude
        phases[i] = frame.phase

    return timestamps, amplitudes, phases


def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Extract sliding windows from sequential data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_frames, ...).
    window_size : int
        Number of frames per window. Must be positive.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    np.ndarray
        Windowed data with shape (n_windows, window_size, ...).

    Raises
    ------
    ValueError
        If window_size or stride is invalid.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive. Got {window_size}.")

    if stride <= 0:
        raise ValueError(f"stride must be positive. Got {stride}.")

    n_frames = data.shape[0]

    if window_size > n_frames:
        raise ValueError(
            f"window_size {window_size} exceeds number of frames {n_frames}."
        )

    # Compute number of windows
    n_windows = (n_frames - window_size) // stride + 1

    # Build window indices
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows.append(data[start:end])

    return np.array(windows)


def group_frames_by_window(
    frames: List[CSIFrame],
    window_size: int,
    stride: int = 1,
) -> List[List[CSIFrame]]:
    """
    Group consecutive CSI frames into overlapping windows.

    Parameters
    ----------
    frames : List[CSIFrame]
        List of CSI frames in temporal order.
    window_size : int
        Number of frames per window.
    stride : int, optional
        Step size between windows. Defaults to 1.

    Returns
    -------
    List[List[CSIFrame]]
        List of frame windows.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive. Got {window_size}.")

    if stride <= 0:
        raise ValueError(f"stride must be positive. Got {stride}.")

    n_frames = len(frames)

    if window_size > n_frames:
        raise ValueError(
            f"window_size {window_size} exceeds number of frames {n_frames}."
        )

    windows = []
    start = 0

    while start + window_size <= n_frames:
        window = frames[start:start + window_size]
        windows.append(window)
        start += stride

    return windows
