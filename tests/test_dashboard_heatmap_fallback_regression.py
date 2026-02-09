"""Regression tests for dashboard fused-field rendering behavior."""

from pathlib import Path


DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


def _dashboard_source() -> str:
    return DASHBOARD_PATH.read_text(encoding="utf-8")


def test_dashboard_ingests_backend_field_payload() -> None:
    src = _dashboard_source()
    assert "if (data.field)" in src
    assert "state.field = data.field;" in src


def test_dashboard_renders_backend_field_without_entity_synthesis() -> None:
    src = _dashboard_source()
    assert "const backendHeatmap = buildHeatmapFromField();" in src
    assert "generateHeatmapFromEntities" not in src
    assert "SOURCE: NO FIELD" in src


def test_dashboard_modulates_heatmap_with_confidence() -> None:
    src = _dashboard_source()
    assert "confidence_data" in src
    assert "const conf = confidence[i][j];" in src


def test_dashboard_overlay_reports_render_source() -> None:
    src = _dashboard_source()
    assert "SOURCE:" in src
