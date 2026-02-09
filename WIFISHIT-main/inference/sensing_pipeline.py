"""
sensing_pipeline.py

Typed sensing-stage components used between CSI capture and spatial inference.

This module provides a lightweight but explicit implementation of:
- Signal conditioning
- Baseline/drift estimation
- Measurement extraction over short windows

The classes are deterministic and suitable for unit testing.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from collections import deque

import numpy as np

from csi.csi_frame import CSIFrame as LegacyCSIFrame
from contracts import ConditionedCSIFrame, Measurement, QualityMetrics
from contracts.compat import csi_frame_to_contract
from inference.preprocessing import (
    unwrap_phase,
    remove_phase_offset,
    normalize_amplitude_zscore,
)


@dataclass(frozen=True)
class LinkHealthMetrics:
    """Per-link quality and health indicators derived from conditioned CSI."""

    link_id: str
    snr_db: float
    amplitude_std: float
    phase_std: float
    outlier_ratio: float
    health_score: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "snr_db": float(self.snr_db),
            "amplitude_std": float(self.amplitude_std),
            "phase_std": float(self.phase_std),
            "outlier_ratio": float(self.outlier_ratio),
            "health_score": float(self.health_score),
            "status": str(self.status),
        }


class SignalConditioner:
    """Condition raw CSI frames with phase sanitization, normalization, and clipping."""

    def __init__(self, link_id: str = "synthetic_link", clip_z: float = 3.5) -> None:
        self._link_id = link_id
        self._clip_z = clip_z
        self._prev_phase_unwrapped: Optional[np.ndarray] = None

    @staticmethod
    def _clip_outliers(values: np.ndarray, clip_z: float) -> Tuple[np.ndarray, float]:
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-8:
            return values.copy(), 0.0

        z = (values - mean) / std
        outliers = np.abs(z) > clip_z
        clipped = values.copy()
        if np.any(outliers):
            clipped[outliers] = mean
        outlier_ratio = float(np.mean(outliers))
        return clipped, outlier_ratio

    def condition(self, frame: LegacyCSIFrame) -> Tuple[ConditionedCSIFrame, LinkHealthMetrics]:
        raw = csi_frame_to_contract(frame, link_id=self._link_id)

        # Amplitude normalization to stabilize per-frame magnitude spread.
        amp_norm = normalize_amplitude_zscore(raw.amplitude.reshape(1, -1))[0]

        # Phase sanitization:
        # 1) temporally align wrapped phase with previous unwrapped frame
        # 2) unwrap across subcarriers
        # 3) remove frame-wise offset
        phase_temporal = self._temporal_unwrap(raw.phase)
        phase_unwrapped = unwrap_phase(phase_temporal)
        phase_offset = float(np.mean(phase_unwrapped))
        phase_centered = remove_phase_offset(phase_unwrapped.reshape(1, -1))[0]

        amp_clean, amp_outlier_ratio = self._clip_outliers(amp_norm, self._clip_z)
        phase_clean, phase_outlier_ratio = self._clip_outliers(phase_centered, self._clip_z)
        outlier_ratio = 0.5 * (amp_outlier_ratio + phase_outlier_ratio)

        snr_db = self._estimate_snr_db(raw.amplitude)
        amplitude_std = float(np.std(amp_clean))
        phase_std = float(np.std(np.diff(phase_clean, prepend=phase_clean[0])))
        health_score = self._compute_health_score(
            snr_db=snr_db,
            amplitude_std=amplitude_std,
            phase_std=phase_std,
            outlier_ratio=outlier_ratio,
        )
        status = self._status_from_score(health_score)

        conditioned = ConditionedCSIFrame(
            link_id=raw.link_id,
            timestamp=raw.timestamp,
            amplitude=amp_clean,
            phase=phase_clean,
            amplitude_raw=raw.amplitude,
            phase_raw=raw.phase,
            conditioning_method="unwrap_offset_zscore_outlier_clip",
            meta={
                "phase_offset": phase_offset,
                "amplitude_mean_raw": float(np.mean(raw.amplitude)),
            },
        )

        health = LinkHealthMetrics(
            link_id=self._link_id,
            snr_db=snr_db,
            amplitude_std=amplitude_std,
            phase_std=phase_std,
            outlier_ratio=outlier_ratio,
            health_score=health_score,
            status=status,
        )

        return conditioned, health

    def _temporal_unwrap(self, phase_wrapped: np.ndarray) -> np.ndarray:
        phase_wrapped = np.asarray(phase_wrapped, dtype=np.float64)

        if self._prev_phase_unwrapped is None:
            current = unwrap_phase(phase_wrapped)
            self._prev_phase_unwrapped = current.copy()
            return current

        # Shift each subcarrier by integer multiples of 2π to keep continuity.
        two_pi = 2.0 * np.pi
        k = np.round((self._prev_phase_unwrapped - phase_wrapped) / two_pi)
        current = phase_wrapped + k * two_pi
        self._prev_phase_unwrapped = current.copy()
        return current

    @staticmethod
    def _estimate_snr_db(amplitude: np.ndarray) -> float:
        mean_amp = float(np.mean(amplitude))
        noise = amplitude - mean_amp
        signal_power = mean_amp * mean_amp
        noise_power = float(np.mean(noise * noise)) + 1e-8
        snr_linear = (signal_power + 1e-8) / noise_power
        return float(10.0 * np.log10(max(snr_linear, 1e-8)))

    @staticmethod
    def _compute_health_score(
        snr_db: float,
        amplitude_std: float,
        phase_std: float,
        outlier_ratio: float,
    ) -> float:
        snr_term = np.clip((snr_db + 10.0) / 30.0, 0.0, 1.0)
        amp_penalty = np.clip(amplitude_std / 4.0, 0.0, 1.0)
        phase_penalty = np.clip(phase_std / 4.0, 0.0, 1.0)

        score = 0.55 * snr_term + 0.20 * (1.0 - amp_penalty) + 0.20 * (1.0 - phase_penalty) + 0.05 * (1.0 - outlier_ratio)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _status_from_score(score: float) -> str:
        if score >= 0.75:
            return "NOMINAL"
        if score >= 0.45:
            return "DEGRADED"
        return "DISCONNECTED"


class BaselineDriftModel:
    """Tracks conditioned CSI baseline using EWMA and reports drift magnitudes."""

    def __init__(self, alpha: float = 0.03) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._baseline_amplitude: Optional[np.ndarray] = None
        self._baseline_phase: Optional[np.ndarray] = None
        self._last_amp_delta: Optional[np.ndarray] = None
        self._last_phase_delta: Optional[np.ndarray] = None
        self._baseline_phase_offset: Optional[float] = None
        self._baseline_amp_mean_raw: Optional[float] = None

    def update(self, frame: ConditionedCSIFrame) -> Dict[str, float]:
        amplitude = np.asarray(frame.amplitude)
        phase = np.asarray(frame.phase)
        phase_offset = float(frame.meta.get("phase_offset", 0.0))
        amp_mean_raw = float(frame.meta.get("amplitude_mean_raw", 0.0))

        if self._baseline_amplitude is None:
            self._baseline_amplitude = amplitude.copy()
            self._baseline_phase = phase.copy()
            self._last_amp_delta = np.zeros_like(amplitude)
            self._last_phase_delta = np.zeros_like(phase)
            self._baseline_phase_offset = phase_offset
            self._baseline_amp_mean_raw = amp_mean_raw
            return {
                "drift_norm": 0.0,
                "amplitude_delta": 0.0,
                "phase_delta": 0.0,
                "common_phase_delta": 0.0,
                "common_amp_delta": 0.0,
                "drift_common": 0.0,
            }

        amp_delta = amplitude - self._baseline_amplitude
        phase_delta = phase - self._baseline_phase

        amp_norm = float(np.mean(np.abs(amp_delta)))
        phase_norm = float(np.mean(np.abs(phase_delta)))
        common_phase_delta = abs(phase_offset - float(self._baseline_phase_offset))
        common_amp_delta = abs(amp_mean_raw - float(self._baseline_amp_mean_raw))
        drift_common = float(0.7 * common_phase_delta + 0.3 * common_amp_delta)

        drift_norm_local = float(0.6 * amp_norm + 0.4 * phase_norm)
        drift_norm = float(0.45 * drift_norm_local + 0.55 * drift_common)

        self._baseline_amplitude = (1.0 - self._alpha) * self._baseline_amplitude + self._alpha * amplitude
        self._baseline_phase = (1.0 - self._alpha) * self._baseline_phase + self._alpha * phase
        self._baseline_phase_offset = (
            (1.0 - self._alpha) * float(self._baseline_phase_offset) + self._alpha * phase_offset
        )
        self._baseline_amp_mean_raw = (
            (1.0 - self._alpha) * float(self._baseline_amp_mean_raw) + self._alpha * amp_mean_raw
        )
        self._last_amp_delta = amp_delta
        self._last_phase_delta = phase_delta

        return {
            "drift_norm": drift_norm,
            "amplitude_delta": amp_norm,
            "phase_delta": phase_norm,
            "common_phase_delta": float(common_phase_delta),
            "common_amp_delta": float(common_amp_delta),
            "drift_common": drift_common,
        }


class MeasurementExtractor:
    """Extract scalar activity measurements from short windows of conditioned CSI."""

    def __init__(
        self,
        link_id: str = "synthetic_link",
        window_size: int = 10,
        baseline_alpha: float = 0.05,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not (0.0 < baseline_alpha <= 1.0):
            raise ValueError("baseline_alpha must be in (0, 1]")
        self._link_id = link_id
        self._window_size = window_size
        self._baseline_alpha = baseline_alpha
        self._frames: deque[ConditionedCSIFrame] = deque(maxlen=window_size)
        self._raw_baseline: Optional[float] = None
        self._prev_raw_value: Optional[float] = None
        self._delta_noise_floor: Optional[float] = None

    def observe(
        self,
        frame: ConditionedCSIFrame,
        health: LinkHealthMetrics,
        drift: Dict[str, float],
    ) -> Measurement:
        self._frames.append(frame)

        amplitudes = np.array([f.amplitude for f in self._frames])
        phases = np.array([f.phase for f in self._frames])

        if len(self._frames) > 1:
            amp_temporal_delta = float(np.mean(np.abs(np.diff(amplitudes, axis=0))))
        else:
            amp_temporal_delta = 0.0

        if len(self._frames) > 1:
            phase_velocity = float(np.mean(np.abs(np.diff(phases, axis=0))))
        else:
            phase_velocity = 0.0

        drift_norm = float(drift.get("drift_norm", 0.0))
        drift_common = float(drift.get("drift_common", drift_norm))
        raw_value = float(
            0.35 * phase_velocity
            + 0.30 * amp_temporal_delta
            + 0.10 * drift_norm
            + 0.25 * drift_common
        )

        if self._raw_baseline is None:
            self._raw_baseline = raw_value
            delta = 0.0
            self._delta_noise_floor = 0.0
            value = 0.0
        else:
            baseline = self._raw_baseline
            prev = raw_value if self._prev_raw_value is None else self._prev_raw_value
            delta = abs(raw_value - prev)
            noise = 0.0 if self._delta_noise_floor is None else self._delta_noise_floor
            raw_residual = max(0.0, raw_value - baseline)
            delta_residual = max(0.0, delta - noise)
            value = 0.65 * raw_residual + 0.35 * delta_residual

            self._raw_baseline = (
                (1.0 - self._baseline_alpha) * baseline + self._baseline_alpha * raw_value
            )
            self._delta_noise_floor = (1.0 - self._baseline_alpha) * noise + self._baseline_alpha * delta

        self._prev_raw_value = raw_value

        stability = float(np.clip(1.0 - phase_velocity, 0.0, 1.0))
        confidence = float(np.clip(health.health_score * (1.0 - 0.5 * health.outlier_ratio), 0.0, 1.0))

        quality = QualityMetrics(
            snr=health.snr_db,
            confidence=confidence,
            completeness=1.0,
            stability=stability,
            outlier_score=health.outlier_ratio,
        )

        return Measurement(
            link_id=self._link_id,
            timestamp=float(frame.timestamp),
            value=value,
            unit="activity_index",
            quality=quality,
            meta={
                "raw_value": raw_value,
                "raw_baseline": float(self._raw_baseline),
                "delta_noise_floor": float(self._delta_noise_floor),
                "raw_delta": float(delta),
                "amp_temporal_delta": amp_temporal_delta,
                "phase_velocity": phase_velocity,
                "baseline_drift": drift_norm,
                "baseline_drift_common": drift_common,
                "window_size": len(self._frames),
            },
        )
