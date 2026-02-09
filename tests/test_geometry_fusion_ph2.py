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


def _weighted_radius_std(field: np.ndarray) -> float:
    x_idx, y_idx = np.indices(field.shape)
    weights = np.maximum(field, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.0
    mx = float(np.sum(x_idx * weights) / total)
    my = float(np.sum(y_idx * weights) / total)
    return float(np.sqrt(np.sum((((x_idx - mx) ** 2 + (y_idx - my) ** 2) * weights)) / total))


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
    # Avoid false pinpoint certainty from one link.
    assert float(np.max(field)) < 0.9


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

    x_world = -5.0 + (np.arange(field_single.shape[0]) + 0.5) * 0.25
    y_world = -5.0 + (np.arange(field_single.shape[1]) + 0.5) * 0.25
    xg, yg = np.meshgrid(x_world, y_world, indexing="ij")
    far_field_mask = np.hypot(xg, yg) > 2.5

    floor_single = float(np.mean(field_single[far_field_mask]))
    floor_dual = float(np.mean(field_dual[far_field_mask]))
    assert floor_dual <= floor_single * 0.5


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


def test_three_links_stabilize_field_under_intermittent_link_dropout() -> None:
    link_a = LinkGeometry("a", (-4.5, 0.0, 1.5), (4.5, 0.0, 1.5), 5.8e9)
    link_b = LinkGeometry("b", (0.0, -4.5, 1.5), (0.0, 4.5, 1.5), 5.8e9)
    link_c = LinkGeometry("c", (-4.5, -4.5, 1.5), (4.5, 4.5, 1.5), 5.8e9)

    fuser_two = _make_fuser([link_a, link_b])
    fuser_three = _make_fuser([link_a, link_b, link_c])

    def run_series(fuser: GeometryFieldFuser, include_c: bool) -> Dict[str, float]:
        peak_series = []
        spread_series = []
        area_series = []

        # Prime persistence/hysteresis.
        for i in range(5):
            warm_evidence = {
                "a": LinkEvidence("a", activity=1.0, health_score=1.0, confidence=1.0),
                "b": LinkEvidence("b", activity=1.0, health_score=1.0, confidence=1.0),
            }
            if include_c:
                warm_evidence["c"] = LinkEvidence("c", activity=1.0, health_score=1.0, confidence=1.0)
            fuser.fuse(warm_evidence, timestamp=0.01 * i)

        for i in range(40):
            b_health = 1.0 if (i % 5) != 0 else 0.10
            evidence = {
                "a": LinkEvidence(
                    "a",
                    activity=1.0 + 0.20 * np.sin(0.4 * i),
                    health_score=1.0,
                    confidence=0.95,
                ),
                "b": LinkEvidence(
                    "b",
                    activity=1.0 + 0.25 * np.cos(0.17 * i),
                    health_score=b_health,
                    confidence=0.95,
                ),
            }
            if include_c:
                evidence["c"] = LinkEvidence(
                    "c",
                    activity=1.0 + 0.15 * np.sin(0.23 * i + 0.7),
                    health_score=1.0,
                    confidence=0.95,
                )

            out = fuser.fuse(evidence, timestamp=0.2 + 0.01 * i)
            stable_field = out.activity.data * out.confidence.data
            peak_series.append(float(np.max(stable_field)))
            spread_series.append(_weighted_radius_std(stable_field))
            area_series.append(float(np.sum(stable_field > 0.20)))

        return {
            "peak_std": float(np.std(peak_series)),
            "peak_mean": float(np.mean(peak_series)),
            "spread_std": float(np.std(spread_series)),
            "area_std": float(np.std(area_series)),
        }

    stats_two = run_series(fuser_two, include_c=False)
    stats_three = run_series(fuser_three, include_c=True)

    # Redundant geometry should reduce volatility when one link degrades intermittently.
    assert stats_three["spread_std"] < stats_two["spread_std"] * 0.7
    assert stats_three["area_std"] < stats_two["area_std"] * 0.7
    assert stats_three["peak_std"] < stats_two["peak_std"] * 0.9
    assert stats_three["peak_mean"] >= stats_two["peak_mean"] * 0.9


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
