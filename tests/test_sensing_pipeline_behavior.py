"""Behavior tests for conditioning, link health, baseline drift, and measurement deltas."""

import numpy as np

from csi.csi_frame import CSIFrame
from csi.synthetic_csi import SyntheticCSIGenerator
from inference.sensing_pipeline import (
    SignalConditioner,
    BaselineDriftModel,
    MeasurementExtractor,
)
from pipeline.update_loop import UpdateLoop
from world.objects import MovingDisturbance
from world.world import World


def _make_frame(timestamp: float, amplitude: np.ndarray, phase: np.ndarray) -> CSIFrame:
    return CSIFrame(timestamp=timestamp, amplitude=amplitude, phase=phase)


def test_phase_sanitization_reduces_wrap_discontinuity() -> None:
    conditioner = SignalConditioner(link_id="test_link")

    phase_1 = np.array([2.9, 3.0, -3.1, -3.0, -2.9, 3.05, -3.0, -2.8], dtype=np.float64)
    phase_2 = np.array([-3.0, -2.9, 3.05, 3.1, -3.12, -3.0, 3.1, -2.9], dtype=np.float64)

    amp = np.linspace(1.0, 2.0, phase_1.shape[0])

    c1, _ = conditioner.condition(_make_frame(0.0, amp, phase_1))
    c2, _ = conditioner.condition(_make_frame(0.05, amp, phase_2))

    # Wrapped phase should not produce huge discontinuity after sanitization.
    temporal_delta = np.mean(np.abs(c2.phase - c1.phase))
    assert temporal_delta < 2.0


def test_amplitude_normalization_and_outlier_rejection() -> None:
    conditioner = SignalConditioner(link_id="test_link", clip_z=3.0)

    amplitude = np.ones(64, dtype=np.float64)
    amplitude[10] = 1000.0
    phase = np.zeros(64, dtype=np.float64)
    phase[5] = 40.0

    conditioned, health = conditioner.condition(_make_frame(0.0, amplitude, phase))

    assert abs(float(np.mean(conditioned.amplitude))) < 0.3
    assert health.outlier_ratio > 0.0
    assert health.status in {"DEGRADED", "DISCONNECTED"}


def test_baseline_and_drift_separation() -> None:
    conditioner = SignalConditioner(link_id="test_link")
    baseline = BaselineDriftModel(alpha=0.1)

    amp_a = np.linspace(1.0, 2.0, 64)
    ph_a = np.linspace(-1.0, 1.0, 64)
    frame_a, _ = conditioner.condition(_make_frame(0.0, amp_a, ph_a))

    drift_init = baseline.update(frame_a)
    assert drift_init["drift_norm"] == 0.0

    frame_same, _ = conditioner.condition(_make_frame(0.05, amp_a, ph_a))
    drift_same = baseline.update(frame_same)
    assert drift_same["drift_norm"] < 0.05

    amp_b = amp_a[::-1]
    ph_b = np.sin(np.linspace(0.0, 3.0 * np.pi, 64))
    frame_b, _ = conditioner.condition(_make_frame(0.1, amp_b, ph_b))
    drift_changed = baseline.update(frame_b)

    assert drift_changed["drift_norm"] > drift_same["drift_norm"]


def test_static_scene_quiet_and_motion_scene_has_clean_deltas() -> None:
    def run_sequence(with_motion: bool) -> np.ndarray:
        world = World(initial_time=0.0)
        if with_motion:
            world.add_object(
                MovingDisturbance(
                    x=0.0,
                    y=-3.0,
                    z=1.5,
                    vx=0.0,
                    vy=1.4,
                    radius=1.1,
                    intensity=1.5,
                    bounds=(-4.0, 4.0, -3.5, 3.5),
                )
            )

        generator = SyntheticCSIGenerator(
            world=world,
            tx_position=(-4.5, 0.0, 1.5),
            rx_position=(4.5, 0.0, 1.5),
            num_subcarriers=64,
                amplitude_noise_std=0.003,
                phase_noise_std=0.015,
            random_seed=7,
        )

        loop = UpdateLoop(world=world, csi_generator=generator, dt=0.05)

        conditioner = SignalConditioner(link_id="test_link")
        baseline = BaselineDriftModel(alpha=0.03)
        extractor = MeasurementExtractor(link_id="test_link", window_size=10, baseline_alpha=0.05)

        values = []
        for _ in range(80):
            frame = loop.step()
            conditioned, health = conditioner.condition(frame)
            drift = baseline.update(conditioned)
            measurement = extractor.observe(conditioned, health, drift)
            values.append(measurement.as_float())

        return np.array(values, dtype=np.float64)

    static_values = run_sequence(with_motion=False)
    motion_values = run_sequence(with_motion=True)

    # Discard warm-up to evaluate steady behavior.
    static_steady = static_values[20:]
    motion_steady = motion_values[20:]

    assert float(np.mean(static_steady)) < 0.03
    assert float(np.mean(motion_steady)) > float(np.mean(static_steady)) + 0.003
    assert float(np.mean(motion_steady)) > 3.0 * float(np.mean(static_steady))
