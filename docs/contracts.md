# Data Contracts

This document describes the core data contracts used throughout the Wi-Fi CSI sensing research stack. These contracts enable capture, conditioning, inference, tracking, and UI layers to evolve independently while maintaining interoperability.

All contracts are implemented as frozen dataclasses with runtime validation.

## Design Principles

1. **Immutability**: All contract instances are frozen (immutable) after creation
2. **Validation**: All inputs are validated at construction time
3. **Type Safety**: Full type annotations for static analysis
4. **Serialization**: All contracts can be converted to dictionaries for JSON serialization
5. **Zero-copy where safe**: Arrays are made read-only but not always copied

## Contract Hierarchy

```
CSI Layer
├── CSIFrame                 Raw measurement from a single link
└── ConditionedCSIFrame      Preprocessed CSI ready for inference

Measurement Layer
├── QualityMetrics           Signal quality indicators
└── Measurement              Per-link scalar/vector with quality

Spatial Layer
├── FieldMetadata            Common metadata for spatial fields
├── Field2D                  2D scalar field (floor heatmap)
└── Field3D                  3D scalar field (voxel grid)

Detection & Tracking Layer
├── Detection2D              Single detection in 2D
├── Detection3D              Single detection in 3D
├── Track2D                  Tracked entity in 2D
├── Track3D                  Tracked entity in 3D
└── TrackState               Track lifecycle enum
```

---

## CSI Layer

### CSIFrame

Raw CSI measurement from a single wireless link (TX-RX pair).

```python
from contracts import CSIFrame

frame = CSIFrame(
    link_id="ap1_sta2",           # Unique link identifier
    timestamp=0.0,                 # Measurement time (seconds)
    amplitude=np.ones(64),         # Per-subcarrier amplitude
    phase=np.zeros(64),            # Per-subcarrier phase (radians)
    meta={"rssi": -40}             # Optional metadata
)
```

**Invariants:**
- `link_id`: Non-empty string
- `timestamp`: Finite float
- `amplitude`: 1D float64 ndarray, all values finite
- `phase`: 1D float64 ndarray, same shape as amplitude, all values finite
- Arrays are made read-only after construction

**Properties:**
- `num_subcarriers`: Number of subcarriers
- `shape`: Shape of amplitude/phase arrays
- `get_complex()`: Returns complex CSI (amplitude * exp(1j * phase))

### ConditionedCSIFrame

Preprocessed CSI frame ready for inference.

```python
from contracts import ConditionedCSIFrame

conditioned = ConditionedCSIFrame.from_raw(
    frame=raw_frame,
    amplitude=normalized_amp,
    phase=unwrapped_phase,
    conditioning_method="zscore_unwrap"
)
```

**Additional fields:**
- `amplitude_raw`: Original raw amplitude (optional)
- `phase_raw`: Original raw phase (optional)
- `conditioning_method`: Description of preprocessing applied

---

## Measurement Layer

### QualityMetrics

Signal quality indicators for a measurement.

```python
from contracts import QualityMetrics

quality = QualityMetrics(
    snr=15.0,              # Signal-to-noise ratio (dB)
    confidence=0.9,        # Overall confidence [0, 1]
    completeness=1.0,      # Data completeness [0, 1]
    stability=0.8,         # Temporal stability [0, 1]
    outlier_score=0.1      # Outlier score [0, 1] (0=typical)
)

if quality.is_reliable:
    # Use measurement
    ...
```

**Invariants:**
- `confidence`, `completeness`, `stability`: Must be in [0, 1]
- `outlier_score`: Must be in [0, 1]
- All values must be finite

**Methods:**
- `is_reliable`: True if confidence >= 0.5 and outlier_score < 0.5
- `combined_quality()`: Single quality score in [0, 1]

### Measurement

Per-link scalar or vector measurement with quality metrics.

```python
from contracts import Measurement, QualityMetrics

# Scalar measurement
m_scalar = Measurement(
    link_id="link_0",
    timestamp=0.0,
    value=42.5,
    unit="dB",
    quality=QualityMetrics(confidence=0.9)
)

# Vector measurement
m_vector = Measurement(
    link_id="link_0",
    timestamp=0.0,
    value=np.array([1.0, 2.0, 3.0]),
    unit="dimensionless"
)
```

**Invariants:**
- `link_id`: Non-empty string
- `timestamp`: Finite float
- `value`: Finite scalar or 1D ndarray
- All quality metrics valid

**Properties:**
- `is_scalar`: True if value is scalar
- `is_vector`: True if value is 1D array
- `dim`: Dimensionality (1 for scalar, length for vector)
- `as_float()`: Get scalar value
- `as_array()`: Get value as ndarray

---

## Spatial Layer

### FieldMetadata

Common metadata for spatial fields.

```python
from contracts import FieldMetadata

meta = FieldMetadata(
    timestamp=0.5,
    source="world_sampling",
    frame_count=10,
    link_ids=("link_0", "link_1"),
    extra={"method": "max_projection"}
)
```

### Field2D

2D scalar field with optional confidence.

```python
from contracts import Field2D, FieldMetadata

field = Field2D(
    data=np.random.rand(40, 40),        # 2D array
    origin=(-5.0, -5.0),                 # World origin (meters)
    resolution=0.25,                      # Cell size (meters)
    confidence=np.ones((40, 40)) * 0.9,  # Optional confidence
    meta=FieldMetadata(source="floor_projection")
)

# Query at world coordinates
value = field.value_at(x=1.0, y=2.0)

# Get cell center
center = field.cell_center(i=20, j=20)
```

**Invariants:**
- `data`: 2D ndarray, all values finite
- `origin`: Tuple of 2 finite floats
- `resolution`: Positive finite float
- `confidence`: If provided, same shape as data, values in [0, 1]

**Properties:**
- `shape`: Grid shape (nx, ny)
- `dimensions`: Physical size (width, height) in meters

**Methods:**
- `cell_center(i, j)`: World coordinates of cell center
- `world_to_cell(x, y)`: Convert world to cell indices
- `value_at(x, y)`: Get value at world coordinates
- `max_value()`, `mean_value()`, `total_value()`: Statistics

### Field3D

3D scalar field (voxel grid) with optional confidence.

```python
from contracts import Field3D

field3d = Field3D(
    data=np.random.rand(40, 40, 12),
    origin=(-5.0, -5.0, 0.0),
    resolution=0.25
)

# Project to 2D
field2d = field3d.project_floor(method="max")

# Query at world coordinates
value = field3d.value_at(x=1.0, y=2.0, z=1.0)
```

**Additional methods:**
- `voxel_center(i, j, k)`: World coordinates of voxel center
- `world_to_voxel(x, y, z)`: Convert world to voxel indices
- `project_floor(method)`: Project to Field2D ("max", "mean", "sum")

---

## Detection & Tracking Layer

### TrackState

Enumeration of track lifecycle states.

```python
from contracts import TrackState

state = TrackState.TENTATIVE   # New, not yet confirmed
state = TrackState.CONFIRMED   # Confirmed by multiple detections
state = TrackState.COASTING    # No recent detections, still active
state = TrackState.DELETED     # Terminated
```

### Detection2D / Detection3D

Single-frame detections in 2D/3D space.

```python
from contracts import Detection2D, Detection3D

det2d = Detection2D(
    x=1.5, y=2.0,
    timestamp=0.0,
    confidence=0.9,
    intensity=0.8,
    size=0.5,
    bounding_box=(1.0, 1.5, 2.0, 2.5)  # Optional
)

det3d = Detection3D(
    x=1.5, y=2.0, z=1.0,
    timestamp=0.0,
    confidence=0.9,
    intensity=0.8,
    size=0.5
)

# Project 3D to 2D
det2d = det3d.to_2d()

# Compute distance
dist = det2d.distance_to(other_det)
```

**Invariants:**
- All coordinates: Finite floats
- `confidence`: In [0, 1]
- `size`: Non-negative

**Properties:**
- `position`: Tuple of coordinates

### Track2D / Track3D

Multi-frame tracked entities with persistence.

```python
from contracts import Track2D, Track3D, TrackState

track = Track2D(
    track_id=1,                          # Unique positive integer
    x=1.5, y=2.0,
    timestamp=0.5,
    state=TrackState.CONFIRMED,
    confidence=0.9,
    velocity=(0.5, 0.1),                 # Optional (m/s)
    age=10,                              # Frames since creation
    hits=8,                              # Frames with detections
    time_since_update=0.1,
    history=[(0.0, 1.0, 1.5), ...]       # Optional history
)

# Predict future position
future_pos = track.predict(dt=0.1)

# Check state
if track.is_confirmed:
    ...

# Convert to dict for JSON
data = track.to_dict()
```

**Invariants:**
- `track_id`: Positive integer
- `confidence`: In [0, 1]
- `age`: Positive integer >= 1
- `hits`: Non-negative integer
- `time_since_update`: Non-negative
- `history`: If provided, timestamps must be monotonic

**Properties:**
- `position`: Current position tuple
- `is_confirmed`: True if state is CONFIRMED
- `is_active`: True if state is not DELETED

---

## Validation Utilities

The `contracts.validation` module provides utilities for enforcing invariants:

```python
from contracts import (
    ValidationError,
    validate_shape,
    validate_dtype,
    validate_finite,
    validate_monotonic_timestamps,
    validate_range,
    validate_positive,
)

# Shape validation
validate_shape(arr, (64,), "amplitude")
validate_shape(arr, (None, 64), "batch_amplitude")  # None = any size

# Dtype validation
validate_dtype(arr, np.float64, "data")
validate_dtype(arr, [np.float32, np.float64], "data")

# Finiteness
validate_finite(arr, "data")
validate_finite_scalar(value, "timestamp")

# Monotonicity
validate_monotonic_timestamps(timestamps, strict=True)

# Range
validate_range(value, 0.0, 1.0, "confidence")
validate_positive(value, "resolution")
```

All validators raise `ValidationError` (subclass of `ValueError`) on failure.

---

## Usage Examples

### Pipeline Integration

```python
from contracts import CSIFrame, ConditionedCSIFrame, Field3D, Detection3D, Track3D

# 1. Capture CSI
frame = CSIFrame(
    link_id="nexmon_wlan0",
    timestamp=time.time(),
    amplitude=captured_amplitude,
    phase=captured_phase
)

# 2. Condition
conditioned = ConditionedCSIFrame.from_raw(
    frame, normalized_amp, unwrapped_phase,
    conditioning_method="standard"
)

# 3. Build spatial field
field = Field3D(
    data=voxel_data,
    origin=room_origin,
    resolution=0.25
)

# 4. Detect
detections = [
    Detection3D(x=c.x, y=c.y, z=c.z, timestamp=t, confidence=c.conf)
    for c in clusters
]

# 5. Track
tracks = tracker.update(detections)
```

### WebSocket Streaming

```python
# Serialize for dashboard
state_message = {
    "timestamp": track.timestamp,
    "entities": [t.to_dict() for t in tracks],
    "field": {
        "shape": field.shape,
        "max": field.max_value(),
    }
}
await websocket.send_json(state_message)
```

---

## Migration Guide

Existing code using the old `csi.csi_frame.CSIFrame` should migrate to `contracts.CSIFrame`:

```python
# Before
from csi.csi_frame import CSIFrame

frame = CSIFrame(
    timestamp=0.0,
    amplitude=amp,
    phase=phase
)

# After
from contracts import CSIFrame

frame = CSIFrame(
    link_id="default_link",  # New required field
    timestamp=0.0,
    amplitude=amp,
    phase=phase,
    meta={}
)
```

The new contract adds:
- `link_id`: Required identifier for multi-link setups
- `meta`: Optional metadata dictionary

All other fields and behaviors are compatible.
