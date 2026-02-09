"""
ui_server.py

Real-time WebSocket server streaming live system state to a frontend dashboard.

PH-2 pipeline:
    World -> Multi-link synthetic CSI -> Conditioning/Baseline/Measurement
          -> Geometry-constrained fusion -> Temporal persistence
          -> Clustering -> Tracking -> WebSocket

Backward-compatibility:
- Existing keys are preserved: timestamp, entities, global, system
- Additional keys are included: field, intermediate
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn")

from world.world import World
from world.objects import MovingDisturbance
from csi.synthetic_csi import SyntheticCSIGenerator
from csi.csi_frame import CSIFrame
from inference.clustering import cluster_spatial_activity, SpatialCluster
from inference.geometry_fusion import (
    LinkGeometry,
    LinkEvidence,
    GeometryFieldFuser,
    FusionOutput,
)
from inference.sensing_pipeline import (
    SignalConditioner,
    BaselineDriftModel,
    MeasurementExtractor,
    LinkHealthMetrics,
)
from contracts import Measurement


@dataclass
class ServerConfig:
    """Server configuration parameters."""

    host: str = "localhost"
    port: int = 8000
    ws_path: str = "/ws"

    target_fps: float = 8.0
    dt: float = 0.05

    room_x_min: float = -5.0
    room_x_max: float = 5.0
    room_y_min: float = -5.0
    room_y_max: float = 5.0
    room_z_min: float = 0.0
    room_z_max: float = 3.0

    field_resolution: float = 0.25

    num_subcarriers: int = 64
    link_frequency_hz: float = 5.8e9
    amplitude_noise_std: float = 0.01
    phase_noise_std: float = 0.05

    activity_threshold: float = 0.05
    min_cluster_size: int = 4

    entity_match_threshold: float = 1.5
    stability_distance_scale: float = 0.5

    # PH-1 sensing stage
    conditioning_clip_z: float = 3.5
    baseline_alpha: float = 0.03
    measurement_window: int = 10

    # PH-2 fusion stage
    fusion_decay: float = 0.90
    fusion_enter_threshold: float = 0.12
    fusion_exit_threshold: float = 0.08
    fusion_persistence_frames: int = 3
    fusion_min_activity: float = 0.01
    fusion_min_health: float = 0.35


@dataclass
class TrackedEntity:
    """Persistent entity tracked across frames."""

    id: int
    x: float
    y: float
    z: float
    activity: float
    stability: float
    last_change: float
    radius: float
    frames_tracked: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "activity": self.activity,
            "stability": self.stability,
            "last_change": self.last_change,
            "radius": self.radius,
        }


class EntityTracker:
    """Tracks entities across frames with nearest-centroid matching."""

    def __init__(self, match_threshold: float = 1.5, stability_distance_scale: float = 0.5):
        self._match_threshold = match_threshold
        self._stability_distance_scale = stability_distance_scale
        self._next_id = 1
        self._entities: Dict[int, TrackedEntity] = {}

    def update(
        self,
        clusters: List[SpatialCluster],
        x_scale: float,
        y_scale: float,
        x_offset: float,
        y_offset: float,
        current_time: float,
    ) -> List[TrackedEntity]:
        candidates = []
        for cluster in clusters:
            world_x = x_offset + cluster.centroid[0] * x_scale
            world_y = y_offset + cluster.centroid[1] * y_scale
            world_z = 0.0

            radius = float(np.sqrt(cluster.size) * x_scale * 0.5)
            radius = max(0.3, min(2.0, radius))

            activity = min(1.0, float(cluster.total_intensity / max(cluster.size, 1)))

            candidates.append(
                {
                    "x": world_x,
                    "y": world_y,
                    "z": world_z,
                    "radius": radius,
                    "activity": activity,
                }
            )

        matched: Dict[int, TrackedEntity] = {}
        unmatched_candidates = list(range(len(candidates)))

        for entity_id, entity in self._entities.items():
            best_idx = None
            best_distance = float("inf")

            for idx in unmatched_candidates:
                cand = candidates[idx]
                distance = float(np.hypot(entity.x - cand["x"], entity.y - cand["y"]))
                if distance < best_distance and distance < self._match_threshold:
                    best_distance = distance
                    best_idx = idx

            if best_idx is None:
                continue

            cand = candidates[best_idx]
            unmatched_candidates.remove(best_idx)

            stability = max(0.0, 1.0 - best_distance * self._stability_distance_scale)
            stability = 0.7 * stability + 0.3 * entity.stability

            activity_change = abs(cand["activity"] - entity.activity)
            significant_change = best_distance > 0.3 or activity_change > 0.2

            matched[entity_id] = TrackedEntity(
                id=entity_id,
                x=cand["x"],
                y=cand["y"],
                z=cand["z"],
                activity=cand["activity"],
                stability=stability,
                last_change=current_time if significant_change else entity.last_change,
                radius=cand["radius"],
                frames_tracked=entity.frames_tracked + 1,
            )

        for idx in unmatched_candidates:
            cand = candidates[idx]
            new_id = self._next_id
            self._next_id += 1
            matched[new_id] = TrackedEntity(
                id=new_id,
                x=cand["x"],
                y=cand["y"],
                z=cand["z"],
                activity=cand["activity"],
                stability=0.5,
                last_change=current_time,
                radius=cand["radius"],
                frames_tracked=1,
            )

        self._entities = matched
        return list(matched.values())


@dataclass
class LinkRuntime:
    """Runtime components per RF link."""

    geometry: LinkGeometry
    generator: SyntheticCSIGenerator
    conditioner: SignalConditioner
    baseline: BaselineDriftModel
    extractor: MeasurementExtractor


class SystemStateManager:
    """Manages PH-1 and PH-2 state and builds dashboard-compatible updates."""

    def __init__(self, config: ServerConfig):
        self.config = config

        self._frame_times: List[float] = []
        self._world = World(initial_time=0.0)

        self._link_geometries = self._build_link_geometries(config)
        self._links = self._build_link_runtimes(config, self._link_geometries)

        room_x_size = config.room_x_max - config.room_x_min
        room_y_size = config.room_y_max - config.room_y_min

        self._fuser = GeometryFieldFuser(
            geometries=self._link_geometries,
            origin_xy=(config.room_x_min, config.room_y_min),
            dimensions_xy=(room_x_size, room_y_size),
            resolution=config.field_resolution,
            decay=config.fusion_decay,
            enter_threshold=config.fusion_enter_threshold,
            exit_threshold=config.fusion_exit_threshold,
            persistence_frames=config.fusion_persistence_frames,
            min_activity=config.fusion_min_activity,
            min_health=config.fusion_min_health,
        )

        self._heatmap_x_scale = room_x_size / self._fuser.shape[0]
        self._heatmap_y_scale = room_y_size / self._fuser.shape[1]
        self._heatmap_x_offset = config.room_x_min
        self._heatmap_y_offset = config.room_y_min

        self._entity_tracker = EntityTracker(
            match_threshold=config.entity_match_threshold,
            stability_distance_scale=config.stability_distance_scale,
        )

        self._entities: List[TrackedEntity] = []
        self._latest_clusters: List[SpatialCluster] = []
        self._latest_fusion: Optional[FusionOutput] = None

        self._latest_link_health: Dict[str, LinkHealthMetrics] = {}
        self._latest_measurements: Dict[str, Measurement] = {}
        self._latest_drifts: Dict[str, Dict[str, float]] = {}

        self._total_activity = 0.0
        self._novelty = 0.0
        self._env_drift = 0.0
        self._previous_activity_field = np.zeros(self._fuser.shape, dtype=np.float64)

        self._actual_fps = 0.0
        self._latency_ms = 0.0
        self._sensor_count = len(self._links) * 2
        self._csi_health = "NOMINAL"

    @staticmethod
    def _build_link_geometries(config: ServerConfig) -> List[LinkGeometry]:
        z = 1.5
        return [
            LinkGeometry(
                link_id="link_a",
                tx_xyz=(config.room_x_min + 0.5, 0.0, z),
                rx_xyz=(config.room_x_max - 0.5, 0.0, z),
                frequency_hz=config.link_frequency_hz,
            ),
            LinkGeometry(
                link_id="link_b",
                tx_xyz=(0.0, config.room_y_min + 0.5, z),
                rx_xyz=(0.0, config.room_y_max - 0.5, z),
                frequency_hz=config.link_frequency_hz,
            ),
            LinkGeometry(
                link_id="link_c",
                tx_xyz=(config.room_x_min + 0.5, config.room_y_min + 0.5, z),
                rx_xyz=(config.room_x_max - 0.5, config.room_y_max - 0.5, z),
                frequency_hz=config.link_frequency_hz,
            ),
        ]

    def _build_link_runtimes(self, config: ServerConfig, geometries: List[LinkGeometry]) -> Dict[str, LinkRuntime]:
        runtimes: Dict[str, LinkRuntime] = {}
        for idx, geometry in enumerate(geometries):
            runtimes[geometry.link_id] = LinkRuntime(
                geometry=geometry,
                generator=SyntheticCSIGenerator(
                    world=self._world,
                    tx_position=geometry.tx_xyz,
                    rx_position=geometry.rx_xyz,
                    num_subcarriers=config.num_subcarriers,
                    center_frequency_hz=geometry.frequency_hz,
                    amplitude_noise_std=config.amplitude_noise_std,
                    phase_noise_std=config.phase_noise_std,
                    random_seed=42 + idx,
                ),
                conditioner=SignalConditioner(link_id=geometry.link_id, clip_z=config.conditioning_clip_z),
                baseline=BaselineDriftModel(alpha=config.baseline_alpha),
                extractor=MeasurementExtractor(
                    link_id=geometry.link_id,
                    window_size=config.measurement_window,
                    baseline_alpha=0.05,
                ),
            )
        return runtimes

    @property
    def world(self) -> World:
        return self._world

    def add_disturbance_object(self, obj) -> None:
        self._world.add_object(obj)

    def step(self) -> Dict[str, Any]:
        step_start = time.time()

        self._world.step(self.config.dt)
        timestamp = self._world.time

        evidence_map: Dict[str, LinkEvidence] = {}
        self._latest_link_health.clear()
        self._latest_measurements.clear()
        self._latest_drifts.clear()

        for link_id, runtime in self._links.items():
            synthetic = runtime.generator.generate(timestamp=timestamp)
            frame = CSIFrame(
                timestamp=synthetic.timestamp,
                amplitude=synthetic.amplitudes,
                phase=synthetic.phases,
            )

            conditioned, health = runtime.conditioner.condition(frame)
            drift = runtime.baseline.update(conditioned)
            measurement = runtime.extractor.observe(conditioned, health, drift)

            self._latest_link_health[link_id] = health
            self._latest_measurements[link_id] = measurement
            self._latest_drifts[link_id] = drift

            # Scale measurement deltas into a stable [0, 1) activity score.
            activity_score = float(np.tanh(20.0 * measurement.as_float()))

            evidence_map[link_id] = LinkEvidence(
                link_id=link_id,
                activity=activity_score,
                health_score=health.health_score,
                confidence=measurement.quality.confidence,
            )

        fusion = self._fuser.fuse(evidence_map=evidence_map, timestamp=timestamp)
        self._latest_fusion = fusion

        cluster_input = fusion.activity.data * fusion.confidence.data
        clusters = cluster_spatial_activity(
            cluster_input,
            threshold=self.config.activity_threshold,
            min_cluster_size=self.config.min_cluster_size,
        )
        self._latest_clusters = clusters

        self._entities = self._entity_tracker.update(
            clusters=clusters,
            x_scale=self._heatmap_x_scale,
            y_scale=self._heatmap_y_scale,
            x_offset=self._heatmap_x_offset,
            y_offset=self._heatmap_y_offset,
            current_time=timestamp,
        )

        self._compute_global_metrics(fusion)

        step_end = time.time()
        self._latency_ms = (step_end - step_start) * 1000.0

        self._frame_times.append(step_end)
        cutoff = step_end - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]
        self._actual_fps = float(len(self._frame_times))

        self._update_health_status()

        return self._build_state_message()

    def _compute_global_metrics(self, fusion: FusionOutput) -> None:
        field = fusion.activity.data
        delta = np.mean(np.abs(field - self._previous_activity_field))
        self._previous_activity_field = field.copy()

        mean_link_drift = 0.0
        if self._latest_drifts:
            mean_link_drift = float(
                np.mean([d.get("drift_common", d.get("drift_norm", 0.0)) for d in self._latest_drifts.values()])
            )

        self._total_activity = float(np.clip(np.max(field), 0.0, 1.0))
        self._novelty = float(np.clip(delta, 0.0, 1.0))
        self._env_drift = float(np.clip(np.tanh(mean_link_drift), 0.0, 1.0))

    def _update_health_status(self) -> None:
        if not self._latest_link_health:
            self._csi_health = "DISCONNECTED"
            return

        mean_score = float(np.mean([h.health_score for h in self._latest_link_health.values()]))
        if mean_score >= 0.75:
            self._csi_health = "NOMINAL"
        elif mean_score >= 0.45:
            self._csi_health = "DEGRADED"
        else:
            self._csi_health = "DISCONNECTED"

    def _serialize_field(self) -> Dict[str, Any]:
        if self._latest_fusion is None:
            return {
                "type": "floor",
                "width": 0,
                "height": 0,
                "resolution_m": self.config.field_resolution,
                "origin": {"x": self.config.room_x_min, "y": self.config.room_y_min},
                "projection": "geometry_fusion",
                "min_value": 0.0,
                "max_value": 0.0,
                "data": [],
                "confidence_data": [],
            }

        activity = self._latest_fusion.activity.data
        confidence = self._latest_fusion.confidence.data

        return {
            "type": "floor",
            "width": int(activity.shape[0]),
            "height": int(activity.shape[1]),
            "resolution_m": float(self.config.field_resolution),
            "origin": {"x": float(self.config.room_x_min), "y": float(self.config.room_y_min)},
            "projection": "geometry_fusion",
            "min_value": float(np.min(activity)),
            "max_value": float(np.max(activity)),
            "data": activity.reshape(-1).astype(float).tolist(),
            "confidence_data": confidence.reshape(-1).astype(float).tolist(),
            "active_links": list(self._latest_fusion.active_links),
        }

    def _serialize_links(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for link_id, runtime in self._links.items():
            geometry = runtime.geometry
            health = self._latest_link_health.get(link_id)
            measurement = self._latest_measurements.get(link_id)
            drift = self._latest_drifts.get(link_id, {})

            out[link_id] = {
                "geometry": {
                    "tx_xyz": list(geometry.tx_xyz),
                    "rx_xyz": list(geometry.rx_xyz),
                    "frequency_hz": float(geometry.frequency_hz),
                    "wavelength_m": float(geometry.wavelength_m),
                },
                "health": health.to_dict() if health else None,
                "measurement": (
                    {
                        "value": float(measurement.as_float()),
                        "confidence": float(measurement.quality.confidence),
                        "snr_db": float(measurement.quality.snr),
                        "meta": {
                            key: float(val) if isinstance(val, (int, float, np.floating)) else val
                            for key, val in measurement.meta.items()
                        },
                    }
                    if measurement
                    else None
                ),
                "drift": {
                    key: float(val)
                    for key, val in drift.items()
                    if isinstance(val, (int, float, np.floating))
                },
            }

        return out

    def _serialize_clusters(self) -> List[Dict[str, Any]]:
        clusters = []
        for cluster in self._latest_clusters:
            clusters.append(
                {
                    "centroid_row": float(cluster.centroid[0]),
                    "centroid_col": float(cluster.centroid[1]),
                    "size": int(cluster.size),
                    "total_intensity": float(cluster.total_intensity),
                }
            )
        return clusters

    def _build_state_message(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self._world.time),
            "entities": [e.to_dict() for e in self._entities],
            "global": {
                "total_activity": float(self._total_activity),
                "novelty": float(self._novelty),
                "env_drift": float(self._env_drift),
            },
            "system": {
                "fps": float(self._actual_fps),
                "latency_ms": float(self._latency_ms),
                "sensor_count": int(self._sensor_count),
                "csi_health": str(self._csi_health),
            },
            "field": self._serialize_field(),
            "intermediate": {
                "cluster_count": len(self._latest_clusters),
                "clusters": self._serialize_clusters(),
                "links": self._serialize_links(),
            },
        }


if HAS_FASTAPI:
    app = FastAPI(title="CSI Sensing UI Server")

    state_manager: Optional[SystemStateManager] = None
    active_websocket: Optional[WebSocket] = None
    running = True

    @app.on_event("startup")
    async def startup_event() -> None:
        global state_manager

        config = ServerConfig()
        state_manager = SystemStateManager(config)

        state_manager.add_disturbance_object(
            MovingDisturbance(x=0.0, y=0.0, z=0.0, vx=0.5, vy=0.3, radius=1.0, intensity=1.0)
        )
        state_manager.add_disturbance_object(
            MovingDisturbance(x=-2.0, y=2.0, z=0.0, vx=-0.3, vy=0.4, radius=0.8, intensity=0.8)
        )

        print("System initialized with PH-2 geometry-constrained fusion pipeline")
        asyncio.create_task(update_loop())

    async def update_loop() -> None:
        global state_manager, active_websocket, running

        config = ServerConfig()
        target_interval = 1.0 / config.target_fps

        while running:
            loop_start = time.time()

            if state_manager is not None:
                state = state_manager.step()

                if active_websocket is not None:
                    try:
                        await active_websocket.send_json(state)
                    except Exception:
                        active_websocket = None

            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        global active_websocket

        await websocket.accept()
        print(f"Dashboard connected from {websocket.client}")

        active_websocket = websocket

        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

        except WebSocketDisconnect:
            print("Dashboard disconnected")
        except Exception as exc:
            print(f"WebSocket error: {exc}")
        finally:
            if active_websocket == websocket:
                active_websocket = None

    @app.get("/")
    async def root() -> Dict[str, str]:
        return {"status": "running", "service": "CSI Sensing UI Server"}

    @app.get("/status")
    async def status() -> Dict[str, Any]:
        if state_manager is not None:
            return state_manager._build_state_message()
        return {"status": "not initialized"}


async def standalone_server(config: ServerConfig) -> None:
    """Standalone WebSocket server fallback if FastAPI is unavailable."""

    try:
        import websockets
    except ImportError:
        print("Error: Neither FastAPI nor websockets is installed.")
        print("Install with: pip install fastapi uvicorn")
        print("         or: pip install websockets")
        return

    state_manager = SystemStateManager(config)

    state_manager.add_disturbance_object(
        MovingDisturbance(x=0.0, y=0.0, z=0.0, vx=0.5, vy=0.3, radius=1.0, intensity=1.0)
    )
    state_manager.add_disturbance_object(
        MovingDisturbance(x=-2.0, y=2.0, z=0.0, vx=-0.3, vy=0.4, radius=0.8, intensity=0.8)
    )

    print("System initialized with PH-2 geometry-constrained fusion pipeline")

    connected_clients = set()

    async def handler(websocket, path):
        connected_clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                if message == "ping":
                    await websocket.send("pong")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            connected_clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")

    async def broadcast_loop() -> None:
        target_interval = 1.0 / config.target_fps

        while True:
            loop_start = time.time()

            state = state_manager.step()
            state_json = json.dumps(state)

            if connected_clients:
                await asyncio.gather(
                    *[client.send(state_json) for client in connected_clients],
                    return_exceptions=True,
                )

            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)

    await websockets.serve(handler, config.host, config.port)
    print(f"WebSocket server running at ws://{config.host}:{config.port}{config.ws_path}")

    await broadcast_loop()


def run_server(host: str = "localhost", port: int = 8000, target_fps: float = 8.0) -> None:
    """Run the UI server."""

    config = ServerConfig(host=host, port=port, target_fps=target_fps)

    if HAS_FASTAPI:
        print(f"Starting FastAPI server at http://{host}:{port}")
        print(f"WebSocket endpoint: ws://{host}:{port}/ws")
        print(f"Target update rate: {target_fps} Hz")
        uvicorn.run(app, host=host, port=port)
    else:
        print("FastAPI not available, using standalone websockets server")
        asyncio.run(standalone_server(config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSI Sensing UI Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--fps", type=float, default=8.0, help="Target update rate (Hz)")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, target_fps=args.fps)
