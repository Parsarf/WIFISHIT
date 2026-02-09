"""
tests/test_contracts.py

Unit tests for the contracts package.

Tests validation, immutability, and serialization of all contract types.
"""

import math
import pytest
import numpy as np

from contracts import (
    # CSI
    CSIFrame,
    ConditionedCSIFrame,
    # Measurement
    Measurement,
    QualityMetrics,
    # Spatial
    Field2D,
    Field3D,
    FieldMetadata,
    # Detection & Tracking
    Detection2D,
    Detection3D,
    Track2D,
    Track3D,
    TrackState,
    # Validation
    ValidationError,
    validate_shape,
    validate_dtype,
    validate_finite,
    validate_monotonic_timestamps,
    validate_range,
    validate_positive,
)


# =============================================================================
# CSIFrame Tests
# =============================================================================


class TestCSIFrame:
    """Tests for CSIFrame contract."""

    def test_basic_creation(self):
        """Test basic CSIFrame creation."""
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.ones(64),
            phase=np.zeros(64),
        )
        assert frame.link_id == "link_0"
        assert frame.timestamp == 0.0
        assert frame.num_subcarriers == 64
        assert frame.shape == (64,)

    def test_with_meta(self):
        """Test CSIFrame with metadata."""
        frame = CSIFrame(
            link_id="link_0",
            timestamp=1.5,
            amplitude=np.ones(32),
            phase=np.zeros(32),
            meta={"rssi": -40, "noise": -90},
        )
        assert frame.meta["rssi"] == -40
        assert frame.meta["noise"] == -90

    def test_immutability(self):
        """Test that CSIFrame arrays are immutable."""
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.ones(64),
            phase=np.zeros(64),
        )
        with pytest.raises(ValueError):
            frame.amplitude[0] = 999.0

    def test_get_complex(self):
        """Test complex CSI computation."""
        amplitude = np.array([1.0, 2.0, 3.0])
        phase = np.array([0.0, np.pi / 2, np.pi])
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=amplitude,
            phase=phase,
        )
        complex_csi = frame.get_complex()
        assert complex_csi.dtype == np.complex128
        np.testing.assert_almost_equal(complex_csi[0], 1.0 + 0j)
        np.testing.assert_almost_equal(complex_csi[1], 2.0j)
        np.testing.assert_almost_equal(complex_csi[2], -3.0 + 0j)

    def test_empty_link_id_raises(self):
        """Test that empty link_id raises."""
        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="",
                timestamp=0.0,
                amplitude=np.ones(64),
                phase=np.zeros(64),
            )

    def test_non_finite_timestamp_raises(self):
        """Test that non-finite timestamp raises."""
        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="link_0",
                timestamp=float("inf"),
                amplitude=np.ones(64),
                phase=np.zeros(64),
            )

        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="link_0",
                timestamp=float("nan"),
                amplitude=np.ones(64),
                phase=np.zeros(64),
            )

    def test_shape_mismatch_raises(self):
        """Test that amplitude/phase shape mismatch raises."""
        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="link_0",
                timestamp=0.0,
                amplitude=np.ones(64),
                phase=np.zeros(32),
            )

    def test_non_1d_array_raises(self):
        """Test that non-1D arrays raise."""
        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="link_0",
                timestamp=0.0,
                amplitude=np.ones((64, 2)),
                phase=np.zeros((64, 2)),
            )

    def test_non_finite_array_raises(self):
        """Test that arrays with inf/nan raise."""
        amp = np.ones(64)
        amp[0] = float("inf")
        with pytest.raises(ValidationError):
            CSIFrame(
                link_id="link_0",
                timestamp=0.0,
                amplitude=amp,
                phase=np.zeros(64),
            )

    def test_equality(self):
        """Test CSIFrame equality."""
        frame1 = CSIFrame("link_0", 0.0, np.ones(64), np.zeros(64))
        frame2 = CSIFrame("link_0", 0.0, np.ones(64), np.zeros(64))
        frame3 = CSIFrame("link_0", 1.0, np.ones(64), np.zeros(64))

        assert frame1 == frame2
        assert frame1 != frame3

    def test_hash(self):
        """Test CSIFrame is hashable."""
        frame = CSIFrame("link_0", 0.0, np.ones(64), np.zeros(64))
        h = hash(frame)
        assert isinstance(h, int)


class TestConditionedCSIFrame:
    """Tests for ConditionedCSIFrame contract."""

    def test_basic_creation(self):
        """Test basic ConditionedCSIFrame creation."""
        frame = ConditionedCSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.ones(64),
            phase=np.zeros(64),
            conditioning_method="zscore",
        )
        assert frame.conditioning_method == "zscore"
        assert frame.num_subcarriers == 64

    def test_from_raw(self):
        """Test creating from raw CSIFrame."""
        raw = CSIFrame("link_0", 0.0, np.ones(64), np.zeros(64))
        conditioned = ConditionedCSIFrame.from_raw(
            raw,
            amplitude=np.ones(64) * 0.5,
            phase=np.zeros(64),
            conditioning_method="normalized",
        )
        assert conditioned.link_id == "link_0"
        assert conditioned.timestamp == 0.0
        assert conditioned.conditioning_method == "normalized"
        np.testing.assert_array_equal(conditioned.amplitude_raw, raw.amplitude)


# =============================================================================
# Measurement Tests
# =============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics contract."""

    def test_default_values(self):
        """Test default quality metrics."""
        q = QualityMetrics()
        assert q.snr == 0.0
        assert q.confidence == 1.0
        assert q.completeness == 1.0
        assert q.stability == 1.0
        assert q.outlier_score == 0.0

    def test_is_reliable(self):
        """Test is_reliable property."""
        reliable = QualityMetrics(confidence=0.8, outlier_score=0.2)
        assert reliable.is_reliable

        unreliable = QualityMetrics(confidence=0.3, outlier_score=0.2)
        assert not unreliable.is_reliable

    def test_combined_quality(self):
        """Test combined quality computation."""
        q = QualityMetrics(
            confidence=0.8,
            completeness=0.9,
            stability=0.7,
            outlier_score=0.1,
        )
        combined = q.combined_quality()
        expected = 0.8 * 0.9 * 0.7 * 0.9
        assert abs(combined - expected) < 1e-10

    def test_out_of_range_raises(self):
        """Test that out-of-range values raise."""
        with pytest.raises(ValidationError):
            QualityMetrics(confidence=1.5)

        with pytest.raises(ValidationError):
            QualityMetrics(outlier_score=-0.1)


class TestMeasurement:
    """Tests for Measurement contract."""

    def test_scalar_measurement(self):
        """Test scalar measurement."""
        m = Measurement(
            link_id="link_0",
            timestamp=0.0,
            value=42.5,
            unit="dB",
        )
        assert m.is_scalar
        assert not m.is_vector
        assert m.dim == 1
        assert m.as_float() == 42.5

    def test_vector_measurement(self):
        """Test vector measurement."""
        m = Measurement(
            link_id="link_0",
            timestamp=0.0,
            value=np.array([1.0, 2.0, 3.0]),
            unit="dimensionless",
        )
        assert not m.is_scalar
        assert m.is_vector
        assert m.dim == 3
        np.testing.assert_array_equal(m.as_array(), [1.0, 2.0, 3.0])

    def test_with_quality(self):
        """Test measurement with quality metrics."""
        q = QualityMetrics(confidence=0.9, snr=15.0)
        m = Measurement(
            link_id="link_0",
            timestamp=0.0,
            value=42.5,
            quality=q,
        )
        assert m.quality.confidence == 0.9
        assert m.quality.snr == 15.0


# =============================================================================
# Spatial Tests
# =============================================================================


class TestField2D:
    """Tests for Field2D contract."""

    def test_basic_creation(self):
        """Test basic Field2D creation."""
        data = np.random.rand(40, 40)
        field = Field2D(
            data=data,
            origin=(-5.0, -5.0),
            resolution=0.25,
        )
        assert field.shape == (40, 40)
        assert field.dimensions == (10.0, 10.0)

    def test_with_confidence(self):
        """Test Field2D with confidence."""
        data = np.random.rand(10, 10)
        conf = np.ones((10, 10)) * 0.9
        field = Field2D(
            data=data,
            origin=(0.0, 0.0),
            resolution=1.0,
            confidence=conf,
        )
        assert field.confidence is not None
        assert field.confidence.shape == field.shape

    def test_cell_center(self):
        """Test cell center computation."""
        field = Field2D(
            data=np.zeros((10, 10)),
            origin=(0.0, 0.0),
            resolution=1.0,
        )
        center = field.cell_center(0, 0)
        assert center == (0.5, 0.5)

        center = field.cell_center(5, 5)
        assert center == (5.5, 5.5)

    def test_world_to_cell(self):
        """Test world to cell conversion."""
        field = Field2D(
            data=np.zeros((10, 10)),
            origin=(-5.0, -5.0),
            resolution=1.0,
        )
        cell = field.world_to_cell(0.0, 0.0)
        assert cell == (5, 5)

        # Out of bounds
        cell = field.world_to_cell(100.0, 100.0)
        assert cell is None

    def test_value_at(self):
        """Test value_at query."""
        data = np.arange(100).reshape(10, 10).astype(float)
        field = Field2D(
            data=data,
            origin=(0.0, 0.0),
            resolution=1.0,
        )
        value = field.value_at(0.5, 0.5)
        assert value == 0.0

        value = field.value_at(5.5, 5.5)
        assert value == 55.0

    def test_immutability(self):
        """Test Field2D data is immutable."""
        field = Field2D(
            data=np.zeros((10, 10)),
            origin=(0.0, 0.0),
            resolution=1.0,
        )
        with pytest.raises(ValueError):
            field.data[0, 0] = 999.0

    def test_negative_resolution_raises(self):
        """Test that negative resolution raises."""
        with pytest.raises(ValidationError):
            Field2D(
                data=np.zeros((10, 10)),
                origin=(0.0, 0.0),
                resolution=-1.0,
            )

    def test_confidence_shape_mismatch_raises(self):
        """Test that confidence shape mismatch raises."""
        with pytest.raises(ValidationError):
            Field2D(
                data=np.zeros((10, 10)),
                origin=(0.0, 0.0),
                resolution=1.0,
                confidence=np.ones((5, 5)),
            )

    def test_confidence_out_of_range_raises(self):
        """Test that confidence values out of [0,1] raise."""
        with pytest.raises(ValidationError):
            Field2D(
                data=np.zeros((10, 10)),
                origin=(0.0, 0.0),
                resolution=1.0,
                confidence=np.ones((10, 10)) * 1.5,
            )


class TestField3D:
    """Tests for Field3D contract."""

    def test_basic_creation(self):
        """Test basic Field3D creation."""
        data = np.random.rand(40, 40, 12)
        field = Field3D(
            data=data,
            origin=(-5.0, -5.0, 0.0),
            resolution=0.25,
        )
        assert field.shape == (40, 40, 12)
        assert field.dimensions == (10.0, 10.0, 3.0)

    def test_project_floor(self):
        """Test floor projection."""
        data = np.random.rand(10, 10, 5)
        field3d = Field3D(
            data=data,
            origin=(0.0, 0.0, 0.0),
            resolution=1.0,
        )
        field2d = field3d.project_floor(method="max")
        assert field2d.shape == (10, 10)
        assert field2d.origin == (0.0, 0.0)
        assert field2d.resolution == 1.0

    def test_voxel_center(self):
        """Test voxel center computation."""
        field = Field3D(
            data=np.zeros((10, 10, 5)),
            origin=(0.0, 0.0, 0.0),
            resolution=1.0,
        )
        center = field.voxel_center(0, 0, 0)
        assert center == (0.5, 0.5, 0.5)


# =============================================================================
# Detection & Tracking Tests
# =============================================================================


class TestDetection2D:
    """Tests for Detection2D contract."""

    def test_basic_creation(self):
        """Test basic Detection2D creation."""
        det = Detection2D(
            x=1.5,
            y=2.0,
            timestamp=0.0,
            confidence=0.9,
            intensity=0.8,
            size=0.5,
        )
        assert det.position == (1.5, 2.0)
        assert det.confidence == 0.9

    def test_distance_to(self):
        """Test distance computation."""
        det1 = Detection2D(x=0.0, y=0.0, timestamp=0.0)
        det2 = Detection2D(x=3.0, y=4.0, timestamp=0.0)
        assert det1.distance_to(det2) == 5.0

    def test_to_dict(self):
        """Test serialization to dict."""
        det = Detection2D(x=1.0, y=2.0, timestamp=0.5, confidence=0.9)
        d = det.to_dict()
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["timestamp"] == 0.5
        assert d["confidence"] == 0.9

    def test_non_finite_coordinate_raises(self):
        """Test that non-finite coordinates raise."""
        with pytest.raises(ValidationError):
            Detection2D(x=float("inf"), y=0.0, timestamp=0.0)

    def test_confidence_out_of_range_raises(self):
        """Test that confidence out of [0,1] raises."""
        with pytest.raises(ValidationError):
            Detection2D(x=0.0, y=0.0, timestamp=0.0, confidence=1.5)


class TestDetection3D:
    """Tests for Detection3D contract."""

    def test_basic_creation(self):
        """Test basic Detection3D creation."""
        det = Detection3D(
            x=1.5,
            y=2.0,
            z=1.0,
            timestamp=0.0,
            confidence=0.9,
        )
        assert det.position == (1.5, 2.0, 1.0)

    def test_to_2d(self):
        """Test projection to 2D."""
        det3d = Detection3D(x=1.0, y=2.0, z=3.0, timestamp=0.5, confidence=0.9)
        det2d = det3d.to_2d()
        assert det2d.x == 1.0
        assert det2d.y == 2.0
        assert det2d.timestamp == 0.5
        assert det2d.confidence == 0.9


class TestTrack2D:
    """Tests for Track2D contract."""

    def test_basic_creation(self):
        """Test basic Track2D creation."""
        track = Track2D(
            track_id=1,
            x=1.5,
            y=2.0,
            timestamp=0.5,
            state=TrackState.CONFIRMED,
            confidence=0.9,
        )
        assert track.track_id == 1
        assert track.position == (1.5, 2.0)
        assert track.is_confirmed
        assert track.is_active

    def test_predict(self):
        """Test position prediction."""
        track = Track2D(
            track_id=1,
            x=0.0,
            y=0.0,
            timestamp=0.0,
            velocity=(1.0, 2.0),
        )
        predicted = track.predict(dt=1.0)
        assert predicted == (1.0, 2.0)

    def test_to_dict(self):
        """Test serialization to dict."""
        track = Track2D(
            track_id=1,
            x=1.0,
            y=2.0,
            timestamp=0.5,
            state=TrackState.CONFIRMED,
        )
        d = track.to_dict()
        assert d["id"] == 1
        assert d["x"] == 1.0
        assert d["state"] == "CONFIRMED"

    def test_non_positive_track_id_raises(self):
        """Test that non-positive track_id raises."""
        with pytest.raises(ValidationError):
            Track2D(track_id=0, x=0.0, y=0.0, timestamp=0.0)

        with pytest.raises(ValidationError):
            Track2D(track_id=-1, x=0.0, y=0.0, timestamp=0.0)


class TestTrack3D:
    """Tests for Track3D contract."""

    def test_basic_creation(self):
        """Test basic Track3D creation."""
        track = Track3D(
            track_id=1,
            x=1.5,
            y=2.0,
            z=1.0,
            timestamp=0.5,
            state=TrackState.CONFIRMED,
        )
        assert track.position == (1.5, 2.0, 1.0)

    def test_to_2d(self):
        """Test projection to Track2D."""
        track3d = Track3D(
            track_id=1,
            x=1.0,
            y=2.0,
            z=3.0,
            timestamp=0.5,
            state=TrackState.CONFIRMED,
            velocity=(0.1, 0.2, 0.3),
        )
        track2d = track3d.to_2d()
        assert track2d.track_id == 1
        assert track2d.position == (1.0, 2.0)
        assert track2d.velocity == (0.1, 0.2)


# =============================================================================
# Validation Utility Tests
# =============================================================================


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_shape_success(self):
        """Test validate_shape with valid input."""
        arr = np.zeros((10, 64))
        validate_shape(arr, (10, 64), "test")
        validate_shape(arr, (None, 64), "test")  # None = any

    def test_validate_shape_failure(self):
        """Test validate_shape with invalid input."""
        arr = np.zeros((10, 64))
        with pytest.raises(ValidationError):
            validate_shape(arr, (10, 32), "test")

    def test_validate_dtype_success(self):
        """Test validate_dtype with valid input."""
        arr = np.zeros(10, dtype=np.float64)
        validate_dtype(arr, np.float64, "test")
        validate_dtype(arr, [np.float32, np.float64], "test")

    def test_validate_dtype_failure(self):
        """Test validate_dtype with invalid input."""
        arr = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValidationError):
            validate_dtype(arr, np.float64, "test")

    def test_validate_finite_success(self):
        """Test validate_finite with valid input."""
        arr = np.array([1.0, 2.0, 3.0])
        validate_finite(arr, "test")

    def test_validate_finite_failure(self):
        """Test validate_finite with invalid input."""
        arr = np.array([1.0, float("inf"), 3.0])
        with pytest.raises(ValidationError):
            validate_finite(arr, "test")

        arr = np.array([1.0, float("nan"), 3.0])
        with pytest.raises(ValidationError):
            validate_finite(arr, "test")

    def test_validate_monotonic_timestamps_success(self):
        """Test validate_monotonic_timestamps with valid input."""
        timestamps = [0.0, 0.1, 0.2, 0.3]
        validate_monotonic_timestamps(timestamps, strict=True)

        # Non-strict allows equal timestamps
        timestamps = [0.0, 0.1, 0.1, 0.2]
        validate_monotonic_timestamps(timestamps, strict=False)

    def test_validate_monotonic_timestamps_failure(self):
        """Test validate_monotonic_timestamps with invalid input."""
        timestamps = [0.0, 0.2, 0.1, 0.3]
        with pytest.raises(ValidationError):
            validate_monotonic_timestamps(timestamps, strict=True)

    def test_validate_range_success(self):
        """Test validate_range with valid input."""
        validate_range(0.5, 0.0, 1.0, "test")
        validate_range(0.0, 0.0, 1.0, "test")  # Inclusive

    def test_validate_range_failure(self):
        """Test validate_range with invalid input."""
        with pytest.raises(ValidationError):
            validate_range(1.5, 0.0, 1.0, "test")

        with pytest.raises(ValidationError):
            validate_range(-0.1, 0.0, 1.0, "test")

    def test_validate_positive_success(self):
        """Test validate_positive with valid input."""
        validate_positive(1.0, "test")
        validate_positive(0.0, "test", allow_zero=True)

    def test_validate_positive_failure(self):
        """Test validate_positive with invalid input."""
        with pytest.raises(ValidationError):
            validate_positive(0.0, "test")

        with pytest.raises(ValidationError):
            validate_positive(-1.0, "test", allow_zero=True)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and integration."""

    def test_single_subcarrier_frame(self):
        """Test CSIFrame with single subcarrier."""
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.array([1.0]),
            phase=np.array([0.0]),
        )
        assert frame.num_subcarriers == 1

    def test_large_array(self):
        """Test with large arrays."""
        n = 10000
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.ones(n),
            phase=np.zeros(n),
        )
        assert frame.num_subcarriers == n

    def test_meta_mutation_isolation(self):
        """Test that external meta mutation doesn't affect frame."""
        meta = {"key": "value"}
        frame = CSIFrame(
            link_id="link_0",
            timestamp=0.0,
            amplitude=np.ones(64),
            phase=np.zeros(64),
            meta=meta,
        )
        meta["key"] = "changed"
        assert frame.meta["key"] == "value"  # Frame is unaffected

    def test_detection_3d_to_2d_preserves_meta(self):
        """Test that 3D->2D conversion preserves metadata."""
        det3d = Detection3D(
            x=1.0,
            y=2.0,
            z=3.0,
            timestamp=0.5,
            meta={"source": "test"},
        )
        det2d = det3d.to_2d()
        assert det2d.meta["source"] == "test"

    def test_field3d_projections(self):
        """Test all Field3D projection methods."""
        data = np.random.rand(5, 5, 3)
        field = Field3D(
            data=data,
            origin=(0.0, 0.0, 0.0),
            resolution=1.0,
        )

        for method in ["max", "mean", "sum"]:
            proj = field.project_floor(method=method)
            assert proj.shape == (5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
