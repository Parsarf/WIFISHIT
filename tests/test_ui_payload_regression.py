"""Regression tests for websocket payload compatibility and field streaming."""

from visualization.ui_server import ServerConfig, SystemStateManager
from world.objects import MovingDisturbance


def _manager_with_activity() -> SystemStateManager:
    cfg = ServerConfig()
    manager = SystemStateManager(cfg)
    manager.add_disturbance_object(
        MovingDisturbance(
            x=0.0,
            y=0.0,
            z=0.0,
            vx=0.3,
            vy=0.2,
            radius=1.0,
            intensity=1.0,
        )
    )
    return manager


def test_state_message_keeps_legacy_keys_and_adds_field_payload() -> None:
    manager = _manager_with_activity()
    state = manager.step()

    # Backward-compatible keys
    for key in ("timestamp", "entities", "global", "system"):
        assert key in state

    # New field payload
    assert "field" in state
    field = state["field"]
    assert field["type"] == "floor"
    assert field["width"] > 0
    assert field["height"] > 0
    assert len(field["data"]) == field["width"] * field["height"]
    assert len(field["confidence_data"]) == field["width"] * field["height"]

    # Intermediate diagnostics
    assert "intermediate" in state
    assert "links" in state["intermediate"]


def test_field_payload_is_nontrivial_when_world_has_disturbance() -> None:
    manager = _manager_with_activity()

    # Advance several frames to pass temporal persistence.
    state = None
    for _ in range(12):
        state = manager.step()

    assert state is not None
    field = state["field"]
    assert field["max_value"] > 0.0
    assert field["max_value"] >= field["min_value"]


def test_system_health_and_measurement_schema_are_present() -> None:
    manager = _manager_with_activity()
    state = manager.step()

    system = state["system"]
    assert "csi_health" in system

    links = state["intermediate"]["links"]
    assert isinstance(links, dict)
    assert len(links) >= 1

    first_link = next(iter(links.values()))
    assert "measurement" in first_link
    assert "health" in first_link
    assert first_link["measurement"] is not None
    assert "value" in first_link["measurement"]
    assert "confidence" in first_link["measurement"]
