"""
contracts

Core data contracts for the Wi-Fi CSI sensing research stack.

This package defines strict, documented dataclasses that form the data contracts
between capture, conditioning, inference, tracking, and UI layers.

All contracts are immutable (frozen dataclasses) with runtime validation.
Modules can evolve independently as long as they honor these contracts.

Contracts
---------
CSI Layer:
    CSIFrame : Raw CSI measurement from a single link
    ConditionedCSIFrame : Preprocessed CSI ready for inference

Measurement Layer:
    Measurement : Per-link scalar/vector measurement with quality metrics

Spatial Layer:
    Field2D : 2D scalar field with confidence and metadata
    Field3D : 3D scalar field (voxel grid) with confidence and metadata

Detection & Tracking Layer:
    Detection2D : Single detection in 2D space
    Detection3D : Single detection in 3D space
    Track2D : Tracked entity over time in 2D
    Track3D : Tracked entity over time in 3D
"""

from contracts.csi import CSIFrame, ConditionedCSIFrame
from contracts.measurement import Measurement, QualityMetrics
from contracts.spatial import Field2D, Field3D, FieldMetadata
from contracts.detection import (
    Detection2D,
    Detection3D,
    Track2D,
    Track3D,
    TrackState,
)
from contracts.validation import (
    validate_shape,
    validate_dtype,
    validate_finite,
    validate_monotonic_timestamps,
    validate_range,
    validate_positive,
    ValidationError,
)
from contracts.compat import (
    csi_frame_to_contract,
    contract_to_csi_frame,
    synthetic_frame_to_contract,
)

__all__ = [
    # CSI
    "CSIFrame",
    "ConditionedCSIFrame",
    # Measurement
    "Measurement",
    "QualityMetrics",
    # Spatial
    "Field2D",
    "Field3D",
    "FieldMetadata",
    # Detection & Tracking
    "Detection2D",
    "Detection3D",
    "Track2D",
    "Track3D",
    "TrackState",
    # Validation
    "validate_shape",
    "validate_dtype",
    "validate_finite",
    "validate_monotonic_timestamps",
    "validate_range",
    "validate_positive",
    "ValidationError",
    # Compatibility
    "csi_frame_to_contract",
    "contract_to_csi_frame",
    "synthetic_frame_to_contract",
]
