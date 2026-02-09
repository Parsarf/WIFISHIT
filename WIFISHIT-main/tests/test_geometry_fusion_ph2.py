"""PH-2 geometry fusion tests."""

from typing import Dict

import numpy as np

from inference.geometry_fusion import (
    GeometryFieldFuser,
    LinkEvidence,
    LinkGeometry,
)


def _make_fuser(geometries):
    return GeometryFieldFuser(
        geometries=geometries,
        origin_xy=(-5.0, -5.0),
        dimensions_xy=(10.0, 10.0),
        resolution=0.25,
        decay=0.85,
        enter_threshold=0.10,
        exit_threshold=0.06,
        persistence_frames=2,
        min_activity=0.0,
        min_health=0.35,
    )


def _run_frames(fuser: GeometryFieldFuser, evidence: Dict[str, LinkEvidence], n: int = 6):
    out = None
    for i in range(n):
        out = fuser.fuse(evidence_map=evidence, timestamp=0.05 * i)
    return out


def _weighted_spread(field: np.ndarray):
    x_idx, y_idx = np.indices(field.shape)
    weights = np.maximum(field, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.0, 0.0
    mx = float(np.sum(x_idx * weights) / total)
    my = float(np.sum(y_idx * weights) / total)
    sx = float(np.sqrt(np.sum(((x_idx - mx) ** 2) * weights) / total))
    sy = float(np.sqrt(np.sum(((y_idx - my) ** 2) * weights) / total))
    return sx, sy


def _peak_world(field: np.ndarray):
    idx = np.unravel_index(np.argmax(field), field.shape)
    x = -5.0 + (idx[0] + 0.5) * 0.25
    y = -5.0 + (idx[1] + 0.5) * 0.25
    return x, y


def test_single_link_produces_elongated_fuzzy_region() -> None:
    link_a = LinkGeometry(
        link_id="a",
        tx_xyz=(-4.5, 0.0, 1.5),
        rx_xyz=(4.5, 0.0, 1.5),
        frequency_hz=5.8e9,
    )
    fuser = _make_fuser([link_a])

    evidence = {
        "a": LinkEvidence(link_id="a", activity=1.0, health_score=1.0, confidence=1.0),
    }
    out = _run_frames(fuser, evidence)
    field = out.activity.data

    sx, sy = _weighted_spread(field)
    assert sx > sy * 1.4

    # Fuzzy area: not a single sharp point.
    assert int(np.sum(field > 0.20)) > 20


def test_two_links_tighten_intersection_and_reduce_floor() -> None:
    link_a = LinkGeometry("a", (-4.5, 0.0, 1.5), (4.5, 0.0, 1.5), 5.8e9)
    link_b = LinkGeometry("b", (0.0, -4.5, 1.5), (0.0, 4.5, 1.5), 5.8e9)

    fuser_single = _make_fuser([link_a])
    out_single = _run_frames(
        fuser_single,
        {"a": LinkEvidence("a", activity=1.0, health_score=1.0, confidence=1.0)},
    )
    field_single = out_single.activity.data

    fuser_dual = _make_fuser([link_a, link_b])
    out_dual = _run_frames(
        fuser_dual,
        {
            "a": LinkEvidence("a", activity=1.0, health_score=1.0, confidence=1.0),
            "b": LinkEvidence("b", activity=1.0, health_score=1.0, confidence=1.0),
        },
    )
    field_dual = out_dual.activity.data

    area_single = int(np.sum(field_single > 0.20))
    area_dual = int(np.sum(field_dual > 0.20))
    assert area_dual < area_single

    floor_single = float(np.median(field_single))
    floor_dual = float(np.median(field_dual))
    assert floor_dual <= floor_single + 1e-8


def test_bad_link_gating_improves_localization_with_three_links() -> None:
    link_a = LinkGeometry("a", (-4.5, 0.0, 1.5), (4.5, 0.0, 1.5), 5.8e9)
    link_b = LinkGeometry("b", (0.0, -4.5, 1.5), (0.0, 4.5, 1.5), 5.8e9)
    link_c = LinkGeometry("c", (-4.5, 3.0, 1.5), (4.5, 3.0, 1.5), 5.8e9)

    fuser = _make_fuser([link_a, link_b, link_c])

    noisy = _run_frames(
        fuser,
        {
            "a": LinkEvidence("a", activity=1.0, health_score=1.0, confidence=1.0),
            "b": LinkEvidence("b", activity=1.0, health_score=1.0, confidence=1.0),
            "c": LinkEvidence("c", activity=1.4, health_score=1.0, confidence=1.0),
        },
    )
    filtered = _run_frames(
        fuser,
        {
            "a": LinkEvidence("a", activity=1.0, health_score=1.0, confidence=1.0),
            "b": LinkEvidence("b", activity=1.0, health_score=1.0, confidence=1.0),
            # health below threshold => excluded by fusion
            "c": LinkEvidence("c", activity=1.4, health_score=0.10, confidence=1.0),
        },
    )
    center_idx = (20, 20)
    noisy_conf = float(noisy.confidence.data[center_idx])
    filtered_conf = float(filtered.confidence.data[center_idx])
    assert filtered_conf > noisy_conf

    noisy_activity = float(noisy.activity.data[center_idx])
    filtered_activity = float(filtered.activity.data[center_idx])
    assert filtered_activity >= noisy_activity


def test_replay_determinism_same_input_same_field() -> None:
    link_a = LinkGeometry("a", (-4.5, 0.0, 1.5), (4.5, 0.0, 1.5), 5.8e9)
    link_b = LinkGeometry("b", (0.0, -4.5, 1.5), (0.0, 4.5, 1.5), 5.8e9)

    evidence_seq = [
        {
            "a": LinkEvidence("a", activity=0.6 + 0.1 * np.sin(i * 0.2), health_score=0.9, confidence=0.95),
            "b": LinkEvidence("b", activity=0.5 + 0.1 * np.cos(i * 0.2), health_score=0.85, confidence=0.9),
        }
        for i in range(12)
    ]

    fuser_1 = _make_fuser([link_a, link_b])
    out_1 = [fuser_1.fuse(e, timestamp=0.05 * i).activity.data.copy() for i, e in enumerate(evidence_seq)]

    fuser_2 = _make_fuser([link_a, link_b])
    out_2 = [fuser_2.fuse(e, timestamp=0.05 * i).activity.data.copy() for i, e in enumerate(evidence_seq)]

    for a, b in zip(out_1, out_2):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)
