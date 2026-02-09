"""
geometry_fusion.py

Geometry-constrained 2D evidence fusion for multi-link RF sensing.

PH-2 goals implemented here:
- Immutable per-link geometry
- Cached physics-informed sensitivity kernels
- Evidence injection from conditioned/baseline-separated measurements
- Conservative multi-link fusion
- Temporal persistence with decay + hysteresis
- Confidence map aligned with fused activity field
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from contracts import Field2D, FieldMetadata


def _validate_xyz(value: Tuple[float, float, float], name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a tuple of length 3")
    for idx, coord in enumerate(value):
        if not np.isfinite(coord):
            raise ValueError(f"{name}[{idx}] must be finite")


@dataclass(frozen=True)
class LinkGeometry:
    """Immutable RF link geometry used for spatial inference."""

    link_id: str
    tx_xyz: Tuple[float, float, float]
    rx_xyz: Tuple[float, float, float]
    frequency_hz: float

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        _validate_xyz(self.tx_xyz, "tx_xyz")
        _validate_xyz(self.rx_xyz, "rx_xyz")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive and finite")

    @property
    def wavelength_m(self) -> float:
        return float(299792458.0 / self.frequency_hz)


@dataclass(frozen=True)
class LinkEvidence:
    """Per-link scalar evidence used by field fusion."""

    link_id: str
    activity: float
    health_score: float
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        for field_name in ("activity", "health_score", "confidence"):
            value = getattr(self, field_name)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.activity < 0.0:
            raise ValueError("activity must be >= 0")
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError("health_score must be in [0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class FusionOutput:
    """Fused 2D activity + confidence maps and per-link diagnostics."""

    activity: Field2D
    confidence: Field2D
    link_weights: Dict[str, float]
    active_links: Tuple[str, ...]


class LinkKernelCache:
    """Caches deterministic geometry kernels over a fixed 2D grid."""

    def __init__(
        self,
        origin_xy: Tuple[float, float],
        dimensions_xy: Tuple[float, float],
        resolution: float,
    ) -> None:
        if resolution <= 0.0 or not np.isfinite(resolution):
            raise ValueError("resolution must be positive and finite")
        if dimensions_xy[0] <= 0.0 or dimensions_xy[1] <= 0.0:
            raise ValueError("dimensions must be positive")

        self._origin_xy = origin_xy
        self._resolution = resolution
        self._shape = (
            int(np.ceil(dimensions_xy[0] / resolution)),
            int(np.ceil(dimensions_xy[1] / resolution)),
        )
        self._cache: Dict[str, np.ndarray] = {}

        nx, ny = self._shape
        x = origin_xy[0] + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
        y = origin_xy[1] + (np.arange(ny, dtype=np.float64) + 0.5) * resolution
        self._grid_x, self._grid_y = np.meshgrid(x, y, indexing="ij")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def get(self, geometry: LinkGeometry) -> np.ndarray:
        cached = self._cache.get(geometry.link_id)
        if cached is not None:
            return cached

        kernel = self._build_kernel(geometry)
        self._cache[geometry.link_id] = kernel
        return kernel

    def _build_kernel(self, geometry: LinkGeometry) -> np.ndarray:
        tx = np.array([geometry.tx_xyz[0], geometry.tx_xyz[1]], dtype=np.float64)
        rx = np.array([geometry.rx_xyz[0], geometry.rx_xyz[1]], dtype=np.float64)

        segment = rx - tx
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-9:
            raise ValueError(f"Invalid geometry for {geometry.link_id}: tx and rx overlap")

        unit = segment / seg_len
        px = self._grid_x - tx[0]
        py = self._grid_y - tx[1]
        proj = px * unit[0] + py * unit[1]
        proj_clamped = np.clip(proj, 0.0, seg_len)

        closest_x = tx[0] + proj_clamped * unit[0]
        closest_y = tx[1] + proj_clamped * unit[1]
        d_perp = np.hypot(self._grid_x - closest_x, self._grid_y - closest_y)

        outside = np.where(proj < 0.0, -proj, np.where(proj > seg_len, proj - seg_len, 0.0))

        d_tx = np.hypot(self._grid_x - tx[0], self._grid_y - tx[1])
        d_rx = np.hypot(self._grid_x - rx[0], self._grid_y - rx[1])
        path_excess = np.maximum(0.0, d_tx + d_rx - seg_len)

        sigma_perp = max(self._resolution * 0.75, seg_len * 0.05)
        sigma_outside = max(self._resolution, seg_len * 0.08)
        sigma_excess = max(self._resolution * 0.75, seg_len * 0.05)

        kernel = np.exp(-0.5 * (d_perp / sigma_perp) ** 2)
        kernel *= np.exp(-0.5 * (outside / sigma_outside) ** 2)
        kernel *= np.exp(-0.5 * (path_excess / sigma_excess) ** 2)

        max_val = float(np.max(kernel))
        if max_val > 0.0:
            kernel = kernel / max_val
        return kernel


class GeometryFieldFuser:
    """Fuses per-link evidence into a geometry-constrained 2D field."""

    def __init__(
        self,
        geometries: List[LinkGeometry],
        origin_xy: Tuple[float, float],
        dimensions_xy: Tuple[float, float],
        resolution: float,
        decay: float = 0.90,
        enter_threshold: float = 0.12,
        exit_threshold: float = 0.08,
        persistence_frames: int = 3,
        min_activity: float = 0.01,
        min_health: float = 0.35,
    ) -> None:
        if not geometries:
            raise ValueError("geometries must not be empty")
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")
        if not (0.0 <= exit_threshold <= enter_threshold <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= exit <= enter <= 1")
        if persistence_frames < 1:
            raise ValueError("persistence_frames must be >= 1")

        self._geometries = {g.link_id: g for g in geometries}
        self._origin_xy = origin_xy
        self._resolution = resolution
        self._decay = decay
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold
        self._persistence_frames = persistence_frames
        self._min_activity = min_activity
        self._min_health = min_health

        self._kernels = LinkKernelCache(origin_xy, dimensions_xy, resolution)
        self._shape = self._kernels.shape

        self._coverage_reference = np.zeros(self._shape, dtype=np.float64)
        for geometry in geometries:
            self._coverage_reference += self._kernels.get(geometry)
        self._coverage_reference = np.maximum(self._coverage_reference, 1e-9)

        self._field_prev = np.zeros(self._shape, dtype=np.float64)
        self._active_mask = np.zeros(self._shape, dtype=bool)
        self._persistence_count = np.zeros(self._shape, dtype=np.int32)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def fuse(self, evidence_map: Dict[str, LinkEvidence], timestamp: float) -> FusionOutput:
        evidence_sum = np.zeros(self._shape, dtype=np.float64)
        support_sum = np.zeros(self._shape, dtype=np.float64)
        health_num = np.zeros(self._shape, dtype=np.float64)
        health_den = np.zeros(self._shape, dtype=np.float64)

        link_weights: Dict[str, float] = {}
        kernel_maps: List[np.ndarray] = []
        active_links: List[str] = []
        strengths: List[float] = []

        for link_id, geometry in self._geometries.items():
            ev = evidence_map.get(link_id)
            if ev is None:
                continue

            if ev.health_score < self._min_health:
                link_weights[link_id] = 0.0
                continue

            activity = max(0.0, ev.activity - self._min_activity)
            if activity <= 0.0:
                link_weights[link_id] = 0.0
                continue

            kernel = self._kernels.get(geometry)
            strength = activity * ev.health_score * ev.confidence

            contribution = kernel * strength
            evidence_sum += contribution
            support_sum += kernel * ev.health_score
            health_num += kernel * ev.health_score * ev.confidence
            health_den += kernel

            kernel_maps.append(kernel)
            strengths.append(strength)
            link_weights[link_id] = float(strength)
            active_links.append(link_id)

        if active_links:
            stack = np.stack(kernel_maps, axis=0)
            avg_shape = np.mean(stack, axis=0)

            if len(kernel_maps) >= 2:
                intersection_shape = np.prod(np.clip(stack, 1e-9, 1.0), axis=0) ** (1.0 / len(kernel_maps))
                shape_map = 0.35 * avg_shape + 0.65 * intersection_shape
            else:
                shape_map = avg_shape

            mean_strength = float(np.mean(strengths))
            instantaneous = shape_map * mean_strength
        else:
            instantaneous = np.zeros(self._shape, dtype=np.float64)

        # Conservative normalization with hard upper cap at 1.
        norm = max(1.0, float(np.max(instantaneous)))
        instantaneous = np.clip(instantaneous / norm, 0.0, 1.0)

        # Temporal persistence with decay, minimum window, and hysteresis.
        field_decayed = self._decay * self._field_prev + (1.0 - self._decay) * instantaneous
        above_enter = field_decayed >= self._enter_threshold
        self._persistence_count = np.where(above_enter, self._persistence_count + 1, 0)

        enter_mask = self._persistence_count >= self._persistence_frames
        stay_mask = self._active_mask & (field_decayed >= self._exit_threshold)
        self._active_mask = enter_mask | stay_mask

        fused_activity = np.where(self._active_mask, field_decayed, 0.0)
        self._field_prev = field_decayed

        # Confidence components: geometry coverage, health, and conflict.
        coverage = np.clip(support_sum / self._coverage_reference, 0.0, 1.0)
        health_map = health_num / np.maximum(health_den, 1e-9)

        if len(kernel_maps) >= 2:
            stack = np.stack(kernel_maps, axis=0)
            mean = np.mean(stack, axis=0)
            std = np.std(stack, axis=0)
            conflict = np.clip(std / np.maximum(mean, 1e-9), 0.0, 1.0)
        else:
            conflict = np.zeros(self._shape, dtype=np.float64)

        confidence = coverage * health_map * (1.0 - 0.5 * conflict)
        confidence = np.clip(confidence, 0.0, 1.0)
        confidence = np.where(self._active_mask, confidence, 0.0)

        meta = FieldMetadata(
            timestamp=float(timestamp),
            source="geometry_fusion_ph2",
            frame_count=1,
            link_ids=tuple(active_links),
            extra={"active_link_count": len(active_links)},
        )

        activity_field = Field2D(
            data=fused_activity,
            origin=self._origin_xy,
            resolution=self._resolution,
            confidence=confidence,
            meta=meta,
        )
        confidence_field = Field2D(
            data=confidence,
            origin=self._origin_xy,
            resolution=self._resolution,
            meta=meta,
        )

        return FusionOutput(
            activity=activity_field,
            confidence=confidence_field,
            link_weights=link_weights,
            active_links=tuple(active_links),
        )
