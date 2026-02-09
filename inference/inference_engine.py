"""
inference_engine.py

Lightweight, interpretable inference on latent representations.

Converts latent vectors into continuous signals such as activity intensity,
motion level, and novelty, without performing classification or semantic labeling.

All outputs are continuous values. Behavior is deterministic.
No semantic labels are assigned.
"""

from typing import Optional, Tuple
from collections import deque

import numpy as np


class InferenceEngine:
    """
    Lightweight inference engine for latent representations.

    Accepts latent vectors sequentially, maintains a rolling history,
    and computes continuous inference signals.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent vectors.
    history_size : int, optional
        Maximum number of latent vectors to retain in history. Defaults to 50.
    baseline_decay : float, optional
        Exponential decay factor for running baseline (0 < decay <= 1).
        Higher values give more weight to recent observations. Defaults to 0.02.

    Attributes
    ----------
    latent_dim : int
        Dimension of latent vectors.
    history_size : int
        Maximum history length.
    observation_count : int
        Total number of observations processed.
    """

    def __init__(
        self,
        latent_dim: int,
        history_size: int = 50,
        baseline_decay: float = 0.02,
    ) -> None:
        """Initialize inference engine."""
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive. Got {latent_dim}.")
        if history_size <= 0:
            raise ValueError(f"history_size must be positive. Got {history_size}.")
        if not (0.0 < baseline_decay <= 1.0):
            raise ValueError(
                f"baseline_decay must be in (0, 1]. Got {baseline_decay}."
            )

        self._latent_dim = latent_dim
        self._history_size = history_size
        self._baseline_decay = baseline_decay

        # Rolling history of latent vectors
        self._history: deque = deque(maxlen=history_size)

        # Running baseline (exponential moving average)
        self._baseline: Optional[np.ndarray] = None

        # Running variance estimate for normalization
        self._baseline_sq: Optional[np.ndarray] = None

        # Observation counter
        self._observation_count = 0

    @property
    def latent_dim(self) -> int:
        """Dimension of latent vectors."""
        return self._latent_dim

    @property
    def history_size(self) -> int:
        """Maximum history length."""
        return self._history_size

    @property
    def observation_count(self) -> int:
        """Total number of observations processed."""
        return self._observation_count

    @property
    def current_history_length(self) -> int:
        """Current number of vectors in history."""
        return len(self._history)

    @property
    def baseline(self) -> Optional[np.ndarray]:
        """
        Current running baseline.

        Returns
        -------
        Optional[np.ndarray]
            Running baseline vector, or None if no observations yet.
        """
        if self._baseline is None:
            return None
        return self._baseline.copy()

    def _validate_latent(self, latent: np.ndarray) -> np.ndarray:
        """Validate and reshape latent vector."""
        latent = np.asarray(latent, dtype=np.float64)

        if latent.ndim == 0:
            raise ValueError("latent must be a 1D array.")

        if latent.ndim != 1:
            raise ValueError(
                f"latent must be 1D. Got {latent.ndim} dimensions."
            )

        if latent.shape[0] != self._latent_dim:
            raise ValueError(
                f"latent dimension {latent.shape[0]} does not match "
                f"expected {self._latent_dim}."
            )

        return latent

    def observe(self, latent: np.ndarray) -> None:
        """
        Process a new latent vector observation.

        Updates the rolling history and running baseline.

        Parameters
        ----------
        latent : np.ndarray
            Latent vector. Shape: (latent_dim,).
        """
        latent = self._validate_latent(latent)

        # Add to history
        self._history.append(latent.copy())

        # Update running baseline with exponential moving average
        if self._baseline is None:
            self._baseline = latent.copy()
            self._baseline_sq = latent ** 2
        else:
            alpha = self._baseline_decay
            self._baseline = (1 - alpha) * self._baseline + alpha * latent
            self._baseline_sq = (1 - alpha) * self._baseline_sq + alpha * (latent ** 2)

        self._observation_count += 1

    def compute_intensity(self, latent: np.ndarray) -> float:
        """
        Compute activity/motion intensity as the L2 norm.

        Higher norm indicates stronger activation in latent space.

        Parameters
        ----------
        latent : np.ndarray
            Latent vector. Shape: (latent_dim,).

        Returns
        -------
        float
            L2 norm of the latent vector (non-negative).
        """
        latent = self._validate_latent(latent)
        return float(np.linalg.norm(latent))

    def compute_intensity_variance(self) -> float:
        """
        Compute intensity as variance over recent history.

        Measures how much the latent representation has varied recently.

        Returns
        -------
        float
            Mean variance across latent dimensions over history.
            Returns 0.0 if history is empty.
        """
        if len(self._history) == 0:
            return 0.0

        history_array = np.array(self._history)
        variance_per_dim = np.var(history_array, axis=0)
        return float(np.mean(variance_per_dim))

    def compute_temporal_change(self, latent: np.ndarray) -> float:
        """
        Compute temporal change from the most recent observation.

        Measures instantaneous change as Euclidean distance from previous.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        float
            Euclidean distance from previous observation.
            Returns 0.0 if no previous observation exists.
        """
        latent = self._validate_latent(latent)

        if len(self._history) == 0:
            return 0.0

        previous = self._history[-1]
        return float(np.linalg.norm(latent - previous))

    def compute_temporal_change_average(self, latent: np.ndarray) -> float:
        """
        Compute temporal change from the recent history average.

        Measures deviation from the mean of recent observations.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        float
            Euclidean distance from history mean.
            Returns 0.0 if history is empty.
        """
        latent = self._validate_latent(latent)

        if len(self._history) == 0:
            return 0.0

        history_array = np.array(self._history)
        history_mean = np.mean(history_array, axis=0)
        return float(np.linalg.norm(latent - history_mean))

    def compute_novelty(self, latent: np.ndarray) -> float:
        """
        Compute novelty as distance from running baseline.

        Measures how different the current observation is from the
        long-term average behavior.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        float
            Euclidean distance from running baseline.
            Returns 0.0 if no baseline established yet.
        """
        latent = self._validate_latent(latent)

        if self._baseline is None:
            return 0.0

        return float(np.linalg.norm(latent - self._baseline))

    def compute_novelty_normalized(self, latent: np.ndarray) -> float:
        """
        Compute normalized novelty using running variance.

        Novelty is scaled by the typical variation observed,
        similar to a z-score distance.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        float
            Normalized novelty score.
            Returns 0.0 if insufficient observations.
        """
        latent = self._validate_latent(latent)

        if self._baseline is None or self._baseline_sq is None:
            return 0.0

        # Compute running standard deviation
        variance = self._baseline_sq - self._baseline ** 2
        variance = np.maximum(variance, 0.0)  # Numerical stability
        std = np.sqrt(variance)

        # Avoid division by zero
        std = np.where(std > 1e-8, std, 1.0)

        # Normalized distance
        normalized_diff = (latent - self._baseline) / std
        return float(np.linalg.norm(normalized_diff) / np.sqrt(self._latent_dim))

    def compute_motion_energy(self) -> float:
        """
        Compute motion energy from history differences.

        Measures the total squared change across recent history.

        Returns
        -------
        float
            Sum of squared consecutive differences.
            Returns 0.0 if fewer than 2 observations in history.
        """
        if len(self._history) < 2:
            return 0.0

        history_array = np.array(self._history)
        diffs = np.diff(history_array, axis=0)
        return float(np.sum(diffs ** 2))

    def compute_all(self, latent: np.ndarray) -> dict:
        """
        Compute all inference signals for a latent vector.

        Does NOT automatically observe the vector. Call observe()
        separately to add it to history.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        dict
            Dictionary containing:
            - "intensity": L2 norm of latent
            - "intensity_variance": Variance over recent history
            - "temporal_change": Distance from previous observation
            - "temporal_change_avg": Distance from history mean
            - "novelty": Distance from running baseline
            - "novelty_normalized": Normalized novelty score
            - "motion_energy": Total squared change in history
        """
        latent = self._validate_latent(latent)

        return {
            "intensity": self.compute_intensity(latent),
            "intensity_variance": self.compute_intensity_variance(),
            "temporal_change": self.compute_temporal_change(latent),
            "temporal_change_avg": self.compute_temporal_change_average(latent),
            "novelty": self.compute_novelty(latent),
            "novelty_normalized": self.compute_novelty_normalized(latent),
            "motion_energy": self.compute_motion_energy(),
        }

    def observe_and_compute(self, latent: np.ndarray) -> dict:
        """
        Observe a latent vector and compute all inference signals.

        Combines observe() and compute_all() in a single call.
        Signals are computed BEFORE the observation is added to history.

        Parameters
        ----------
        latent : np.ndarray
            Current latent vector. Shape: (latent_dim,).

        Returns
        -------
        dict
            Dictionary of inference signals (see compute_all).
        """
        latent = self._validate_latent(latent)

        # Compute signals before observation
        signals = self.compute_all(latent)

        # Then observe
        self.observe(latent)

        return signals

    def get_history_array(self) -> np.ndarray:
        """
        Get history as a numpy array.

        Returns
        -------
        np.ndarray
            History array. Shape: (current_history_length, latent_dim).
            Empty array with shape (0, latent_dim) if no history.
        """
        if len(self._history) == 0:
            return np.empty((0, self._latent_dim))
        return np.array(self._history)

    def clear_history(self) -> None:
        """Clear the rolling history but retain baseline."""
        self._history.clear()

    def reset(self) -> None:
        """Reset all state including history and baseline."""
        self._history.clear()
        self._baseline = None
        self._baseline_sq = None
        self._observation_count = 0

    def set_baseline(self, baseline: np.ndarray) -> None:
        """
        Manually set the running baseline.

        Parameters
        ----------
        baseline : np.ndarray
            Baseline vector. Shape: (latent_dim,).
        """
        baseline = self._validate_latent(baseline)
        self._baseline = baseline.copy()
        self._baseline_sq = baseline ** 2
