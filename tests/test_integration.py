"""
test_integration.py

Minimal sanity test for full-system integration.

Verifies the complete pipeline:
    World → VoxelGrid → Projection → Clustering → State Message
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from world.world import World
from world.objects import MovingDisturbance, StaticDisturbance
from space.voxel_grid import VoxelGrid
from space.projections import floor_projection
from inference.clustering import cluster_spatial_activity
from csi.synthetic_csi import SyntheticCSIGenerator
from csi.csi_frame import CSIFrame


def test_world_disturbance():
    """Test that World correctly aggregates disturbance objects."""
    print("Testing World disturbance aggregation...")

    world = World()

    # Add a static disturbance at origin
    static = StaticDisturbance(x=0.0, y=0.0, z=1.0, radius=1.0, intensity=1.0)
    world.add_object(static)

    # Query disturbance at center - should be high
    center_value = world.disturbance_at(0.0, 0.0, 1.0)
    assert center_value > 0.9, f"Expected high disturbance at center, got {center_value}"

    # Query far away - should be near zero
    far_value = world.disturbance_at(10.0, 10.0, 1.0)
    assert far_value < 0.01, f"Expected low disturbance far away, got {far_value}"

    print("  [PASS] World disturbance aggregation works correctly")


def test_moving_disturbance():
    """Test that MovingDisturbance updates position correctly."""
    print("Testing MovingDisturbance movement...")

    obj = MovingDisturbance(
        x=0.0, y=0.0, z=1.0,
        vx=1.0, vy=0.5,
        radius=1.0, intensity=1.0,
        bounds=(-5.0, 5.0, -5.0, 5.0)
    )

    initial_x = obj.x
    initial_y = obj.y

    # Step forward
    obj.step(dt=0.1)

    assert obj.x > initial_x, "MovingDisturbance should have moved in x"
    assert obj.y > initial_y, "MovingDisturbance should have moved in y"

    print("  [PASS] MovingDisturbance moves correctly")


def test_voxel_grid_sampling():
    """Test that VoxelGrid correctly samples from World."""
    print("Testing VoxelGrid sampling...")

    # Create world with disturbance
    world = World()
    world.add_object(StaticDisturbance(x=0.0, y=0.0, z=1.0, radius=1.0, intensity=1.0))

    # Create voxel grid
    grid = VoxelGrid(
        origin=(-2.0, -2.0, 0.0),
        dimensions=(4.0, 4.0, 2.0),
        resolution=0.5
    )

    # Sample world into grid
    grid.sample_world(world)

    # Check that grid has non-zero values
    max_val = grid.max_value()
    assert max_val > 0, f"Expected non-zero max value, got {max_val}"

    # Check that center region has higher values
    center_idx = grid.world_to_voxel(0.0, 0.0, 1.0)
    assert center_idx is not None, "Center should be within grid bounds"

    center_val = grid.get_value(*center_idx)
    assert center_val > 0.5, f"Expected high value at center, got {center_val}"

    print("  [PASS] VoxelGrid sampling works correctly")


def test_floor_projection():
    """Test that floor_projection creates valid 2D heatmap."""
    print("Testing floor projection...")

    # Create world with disturbance
    world = World()
    world.add_object(StaticDisturbance(x=0.0, y=0.0, z=1.0, radius=1.0, intensity=1.0))

    # Create voxel grid and sample
    grid = VoxelGrid(
        origin=(-2.0, -2.0, 0.0),
        dimensions=(4.0, 4.0, 2.0),
        resolution=0.5
    )
    grid.sample_world(world)

    # Create floor projection
    heatmap = floor_projection(grid)

    # Verify shape
    expected_shape = (grid.shape[0], grid.shape[1])
    assert heatmap.shape == expected_shape, f"Expected shape {expected_shape}, got {heatmap.shape}"

    # Verify non-empty
    assert np.max(heatmap) > 0, "Heatmap should have non-zero values"

    print("  [PASS] Floor projection works correctly")


def test_clustering():
    """Test that clustering detects entities in heatmap."""
    print("Testing spatial clustering...")

    # Create world with disturbance
    world = World()
    world.add_object(StaticDisturbance(x=0.0, y=0.0, z=1.0, radius=0.5, intensity=1.0))

    # Create voxel grid and sample
    grid = VoxelGrid(
        origin=(-2.0, -2.0, 0.0),
        dimensions=(4.0, 4.0, 2.0),
        resolution=0.25
    )
    grid.sample_world(world)

    # Create floor projection
    heatmap = floor_projection(grid)

    # Cluster (returns pixel-coordinate centroids)
    clusters = cluster_spatial_activity(
        heatmap,
        threshold=0.1,
        min_cluster_size=2
    )

    # Should detect at least one cluster
    assert len(clusters) >= 1, f"Expected at least 1 cluster, got {len(clusters)}"

    # Cluster should have valid properties
    cluster = clusters[0]
    assert cluster.size >= 2, f"Cluster should have min size 2, got {cluster.size}"
    assert cluster.total_intensity > 0, "Cluster should have positive intensity"

    print("  [PASS] Clustering works correctly")


def test_csi_frame():
    """Test CSIFrame creation and access."""
    print("Testing CSIFrame...")

    amplitude = np.random.rand(64)
    phase = np.random.rand(64) * 2 * np.pi - np.pi

    frame = CSIFrame(
        timestamp=1.0,
        amplitude=amplitude,
        phase=phase
    )

    assert frame.timestamp == 1.0
    assert frame.num_subcarriers == 64
    assert np.allclose(frame.amplitude, amplitude)
    assert np.allclose(frame.phase, phase)

    print("  [PASS] CSIFrame works correctly")


def test_synthetic_csi_generator():
    """Test SyntheticCSIGenerator produces valid frames."""
    print("Testing SyntheticCSIGenerator...")

    world = World()
    world.add_object(MovingDisturbance(x=1.0, y=1.0, z=1.0, vx=0.5, vy=0.3))

    gen = SyntheticCSIGenerator(
        world=world,
        tx_position=(-3.0, 0.0, 1.5),
        rx_position=(3.0, 0.0, 1.5),
        num_subcarriers=64,
        center_frequency_hz=5.8e9,
        amplitude_noise_std=0.01,
        phase_noise_std=0.01
    )

    # Generate a frame
    frame = gen.generate(timestamp=0.0)

    assert frame.num_subcarriers == 64
    assert frame.timestamp == 0.0
    assert np.all(frame.amplitudes >= 0)

    print("  [PASS] SyntheticCSIGenerator works correctly")


def test_full_pipeline():
    """
    Full integration test: World → VoxelGrid → Projection → Clustering → State.

    This is the core sanity test that verifies the complete data flow.
    """
    print("Testing full pipeline integration...")

    # 1. Create world with moving disturbance
    world = World()
    person = MovingDisturbance(
        x=1.0, y=1.0, z=1.0,
        vx=0.3, vy=0.2,
        radius=0.8,
        intensity=1.0,
        bounds=(-4.0, 4.0, -4.0, 4.0)
    )
    world.add_object(person)

    # 2. Create voxel grid
    grid = VoxelGrid(
        origin=(-5.0, -5.0, 0.0),
        dimensions=(10.0, 10.0, 3.0),
        resolution=0.25
    )

    # 3. Run several update steps
    heatmap_history = []
    cluster_history = []

    for step in range(10):
        # Advance world
        world.step(dt=0.05)

        # Sample into grid
        grid.sample_world(world)

        # Create floor projection
        heatmap = floor_projection(grid)
        heatmap_history.append(heatmap.copy())

        # Cluster (returns pixel-coordinate centroids)
        clusters = cluster_spatial_activity(
            heatmap,
            threshold=0.05,
            min_cluster_size=3
        )
        cluster_history.append(clusters)

    # 4. Verify non-empty heatmaps
    all_heatmaps_valid = all(np.max(h) > 0 for h in heatmap_history)
    assert all_heatmaps_valid, "All heatmaps should have non-zero values"

    # 5. Verify clusters detected
    total_clusters = sum(len(c) for c in cluster_history)
    assert total_clusters > 0, "Should detect clusters across steps"

    # 6. Build a sample state message (simulating what ui_server would send)
    last_heatmap = heatmap_history[-1]
    last_clusters = cluster_history[-1]

    state_message = {
        "type": "state_update",
        "timestamp": 10 * 0.05,
        "room": {
            "width_m": 10.0,
            "height_m": 10.0,
            "origin": {"x": -5.0, "y": -5.0}
        },
        "heatmap": {
            "data": last_heatmap.flatten().tolist()[:100],  # Truncate for test
            "width": last_heatmap.shape[0],
            "height": last_heatmap.shape[1],
            "min_value": float(np.min(last_heatmap)),
            "max_value": float(np.max(last_heatmap))
        },
        "entities": [
            {
                "id": i,
                "x": c.centroid[0],  # row in pixel coords
                "y": c.centroid[1],  # col in pixel coords
                "activity": c.total_intensity,
                "area": c.size
            }
            for i, c in enumerate(last_clusters)
        ],
        "metrics": {
            "fps": 20.0,
            "latency_ms": 50.0
        }
    }

    # Verify message structure
    assert "type" in state_message
    assert "heatmap" in state_message
    assert "entities" in state_message
    assert state_message["heatmap"]["max_value"] > 0

    print("  [PASS] Full pipeline integration works correctly")
    print(f"    - Ran {len(heatmap_history)} steps")
    print(f"    - Detected {total_clusters} total clusters across all steps")
    print(f"    - Final heatmap max value: {state_message['heatmap']['max_value']:.4f}")
    print(f"    - Entities in final frame: {len(state_message['entities'])}")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUITE")
    print("="*60 + "\n")

    tests = [
        test_world_disturbance,
        test_moving_disturbance,
        test_voxel_grid_sampling,
        test_floor_projection,
        test_clustering,
        test_csi_frame,
        test_synthetic_csi_generator,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    if failed == 0:
        print(f"ALL TESTS PASSED ({passed}/{passed})")
    else:
        print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
