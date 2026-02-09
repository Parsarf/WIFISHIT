"""
ui_server.py

Real-time WebSocket server streaming live system state to a frontend dashboard.

Bridges the core sensing pipeline to the UI without modifying or interpreting data.

PIPELINE ARCHITECTURE:
    World (continuous disturbances)
      → VoxelGrid (3D discretization via sampling)
      → Projection (2D floor map)
      → Clustering (entity detection)
      → WebSocket (dashboard streaming)

All spatial data originates from the world's disturbance field.
No spatial structure is fabricated from CSI data.

Runs at ws://localhost:8000/ws
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

# FastAPI and WebSocket imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn")

# Core system imports
from world.world import World
from world.objects import MovingDisturbance
from csi.synthetic_csi import SyntheticCSIGenerator
from csi.csi_frame import CSIFrame
from pipeline.update_loop import UpdateLoop
from space.voxel_grid import VoxelGrid
from space.projections import floor_projection
from inference.preprocessing import frames_to_arrays
from inference.representation import LinearEncoder
from inference.inference_engine import InferenceEngine
from inference.clustering import cluster_spatial_activity, SpatialCluster


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    """Server configuration parameters."""
    host: str = "localhost"
    port: int = 8000
    ws_path: str = "/ws"

    # Update rate
    target_fps: float = 8.0

    # Simulation parameters
    dt: float = 0.05  # 50ms timestep

    # Room dimensions (meters)
    room_x_min: float = -5.0
    room_x_max: float = 5.0
    room_y_min: float = -5.0
    room_y_max: float = 5.0
    room_z_min: float = 0.0
    room_z_max: float = 3.0

    # Voxel grid parameters
    voxel_resolution: float = 0.25  # meters per voxel

    # CSI parameters
    num_subcarriers: int = 64

    # Inference parameters
    latent_dim: int = 16

    # Clustering parameters
    activity_threshold: float = 0.05
    min_cluster_size: int = 4

    # Entity tracking parameters
    entity_match_threshold: float = 1.5  # meters - max distance for ID persistence
    stability_distance_scale: float = 0.5  # stability decay per meter of movement

    # Heatmap decay
    heatmap_decay: float = 0.9


# =============================================================================
# TRACKED ENTITY
# =============================================================================

@dataclass
class TrackedEntity:
    """
    Persistent entity tracked across frames.

    Maintains identity, position history, and temporal metadata.
    """
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
        """Convert to dashboard-compatible dictionary."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "activity": self.activity,
            "stability": self.stability,
            "last_change": self.last_change,
            "radius": self.radius
        }


# =============================================================================
# ENTITY TRACKER
# =============================================================================

class EntityTracker:
    """
    Tracks entities across frames using nearest-centroid matching.

    Maintains persistent entity IDs and computes stability based on
    centroid displacement and temporal consistency.
    """

    def __init__(
        self,
        match_threshold: float = 1.5,
        stability_distance_scale: float = 0.5
    ):
        """
        Initialize entity tracker.

        Parameters
        ----------
        match_threshold : float
            Maximum distance (meters) to match entities across frames.
        stability_distance_scale : float
            Stability decay rate per meter of movement.
        """
        self._match_threshold = match_threshold
        self._stability_distance_scale = stability_distance_scale
        self._next_id = 1
        self._entities: Dict[int, TrackedEntity] = {}
        self._previous_time = 0.0

    def update(
        self,
        clusters: List[SpatialCluster],
        x_scale: float,
        y_scale: float,
        x_offset: float,
        y_offset: float,
        current_time: float
    ) -> List[TrackedEntity]:
        """
        Update tracked entities from new cluster detections.

        Uses nearest-centroid matching to maintain entity identity.

        Parameters
        ----------
        clusters : List[SpatialCluster]
            Detected clusters from the current frame.
        x_scale, y_scale : float
            Scale factors to convert heatmap coords to world coords.
        x_offset, y_offset : float
            Offsets to convert heatmap coords to world coords.
        current_time : float
            Current simulation time in seconds.

        Returns
        -------
        List[TrackedEntity]
            Updated list of tracked entities.
        """
        # Convert clusters to candidate entities with world coordinates
        candidates = []
        for cluster in clusters:
            # Convert centroid from heatmap coordinates to world coordinates
            world_x = x_offset + cluster.centroid[0] * x_scale
            world_y = y_offset + cluster.centroid[1] * y_scale
            world_z = 0.0  # Floor level

            # Compute radius from cluster size
            radius = np.sqrt(cluster.size) * x_scale * 0.5
            radius = max(0.3, min(2.0, radius))

            # Activity from cluster intensity (normalized)
            activity = min(1.0, cluster.total_intensity / max(cluster.size, 1))

            candidates.append({
                "x": world_x,
                "y": world_y,
                "z": world_z,
                "radius": radius,
                "activity": activity,
                "cluster": cluster
            })

        # Match candidates to existing entities using nearest-centroid
        matched_entities: Dict[int, TrackedEntity] = {}
        unmatched_candidates = list(range(len(candidates)))
        used_entity_ids = set()

        # For each existing entity, find the nearest candidate
        for entity_id, entity in self._entities.items():
            best_match_idx = None
            best_distance = float('inf')

            for idx in unmatched_candidates:
                cand = candidates[idx]
                distance = np.sqrt(
                    (entity.x - cand["x"])**2 +
                    (entity.y - cand["y"])**2
                )

                if distance < best_distance and distance < self._match_threshold:
                    best_distance = distance
                    best_match_idx = idx

            if best_match_idx is not None:
                # Match found - update entity with new position
                cand = candidates[best_match_idx]
                unmatched_candidates.remove(best_match_idx)
                used_entity_ids.add(entity_id)

                # Compute stability based on displacement
                displacement = best_distance
                stability = max(0.0, 1.0 - displacement * self._stability_distance_scale)

                # Blend with previous stability for smoothness
                stability = 0.7 * stability + 0.3 * entity.stability

                # Determine if significant change occurred
                activity_change = abs(cand["activity"] - entity.activity)
                significant_change = displacement > 0.3 or activity_change > 0.2

                matched_entities[entity_id] = TrackedEntity(
                    id=entity_id,
                    x=cand["x"],
                    y=cand["y"],
                    z=cand["z"],
                    activity=cand["activity"],
                    stability=stability,
                    last_change=current_time if significant_change else entity.last_change,
                    radius=cand["radius"],
                    frames_tracked=entity.frames_tracked + 1
                )

        # Create new entities for unmatched candidates
        for idx in unmatched_candidates:
            cand = candidates[idx]
            new_id = self._next_id
            self._next_id += 1

            matched_entities[new_id] = TrackedEntity(
                id=new_id,
                x=cand["x"],
                y=cand["y"],
                z=cand["z"],
                activity=cand["activity"],
                stability=0.5,  # New entities start with moderate stability
                last_change=current_time,
                radius=cand["radius"],
                frames_tracked=1
            )

        # Update internal state
        self._entities = matched_entities
        self._previous_time = current_time

        return list(matched_entities.values())

    def clear(self) -> None:
        """Clear all tracked entities."""
        self._entities.clear()


# =============================================================================
# SYSTEM STATE MANAGER
# =============================================================================

class SystemStateManager:
    """
    Manages the complete sensing system state.

    Implements the world → voxel → projection → clustering pipeline.
    All spatial data originates from the world's disturbance field.
    """

    def __init__(self, config: ServerConfig):
        """Initialize the system with all components."""
        self.config = config

        # =====================================================================
        # TIMING STATE
        # =====================================================================
        self._start_time = time.time()
        self._last_update_time = self._start_time
        self._frame_times: List[float] = []
        self._update_count = 0

        # =====================================================================
        # WORLD INITIALIZATION
        # The world is the authoritative source of all spatial disturbances
        # =====================================================================
        self._world = World(initial_time=0.0)

        # =====================================================================
        # CSI GENERATOR
        # Generates synthetic CSI from world disturbances
        # =====================================================================
        tx_pos = (config.room_x_min + 0.5, 0.0, 1.5)
        rx_pos = (config.room_x_max - 0.5, 0.0, 1.5)

        self._csi_generator = SyntheticCSIGenerator(
            world=self._world,
            tx_position=tx_pos,
            rx_position=rx_pos,
            num_subcarriers=config.num_subcarriers,
            random_seed=42
        )

        # =====================================================================
        # UPDATE LOOP
        # Coordinates world advancement and CSI generation
        # =====================================================================
        self._update_loop = UpdateLoop(
            world=self._world,
            csi_generator=self._csi_generator,
            dt=config.dt
        )

        # =====================================================================
        # VOXEL GRID
        # Discretizes continuous world disturbances into 3D grid
        # =====================================================================
        room_x_size = config.room_x_max - config.room_x_min
        room_y_size = config.room_y_max - config.room_y_min
        room_z_size = config.room_z_max - config.room_z_min

        self._voxel_grid = VoxelGrid(
            origin=(config.room_x_min, config.room_y_min, config.room_z_min),
            dimensions=(room_x_size, room_y_size, room_z_size),
            resolution=config.voxel_resolution
        )

        # Precompute coordinate conversion factors for heatmap → world
        self._heatmap_x_scale = room_x_size / self._voxel_grid.shape[0]
        self._heatmap_y_scale = room_y_size / self._voxel_grid.shape[1]
        self._heatmap_x_offset = config.room_x_min
        self._heatmap_y_offset = config.room_y_min

        # =====================================================================
        # INFERENCE COMPONENTS
        # Process CSI for global metrics (not spatial structure)
        # =====================================================================
        feature_dim = config.num_subcarriers * 2
        self._encoder = LinearEncoder(
            input_dim=feature_dim,
            latent_dim=config.latent_dim,
            random_seed=42
        )

        self._inference_engine = InferenceEngine(
            latent_dim=config.latent_dim,
            history_size=50
        )

        # CSI frame buffer for temporal analysis
        self._frame_buffer: List[CSIFrame] = []
        self._buffer_size = 10

        # =====================================================================
        # ENTITY TRACKER
        # Tracks entities with persistent IDs across frames
        # =====================================================================
        self._entity_tracker = EntityTracker(
            match_threshold=config.entity_match_threshold,
            stability_distance_scale=config.stability_distance_scale
        )

        # Current entities (for external access)
        self._entities: List[TrackedEntity] = []

        # =====================================================================
        # GLOBAL METRICS
        # Derived from CSI inference (not spatial structure)
        # =====================================================================
        self._total_activity = 0.0
        self._novelty = 0.0
        self._env_drift = 0.0

        # =====================================================================
        # SYSTEM METRICS
        # =====================================================================
        self._actual_fps = 0.0
        self._latency_ms = 0.0
        self._sensor_count = 2  # TX + RX
        self._csi_health = "NOMINAL"

    @property
    def world(self) -> World:
        """Access to the world instance."""
        return self._world

    def add_disturbance_object(self, obj) -> None:
        """Add a disturbance object to the world."""
        self._world.add_object(obj)

    def step(self) -> Dict[str, Any]:
        """
        Advance the system by one timestep and return current state.

        Pipeline stages:
        1. Advance world and generate CSI
        2. Sample world disturbances into voxel grid
        3. Project voxel grid to 2D floor heatmap
        4. Cluster heatmap to detect entities
        5. Track entities across frames
        6. Compute global metrics from CSI
        7. Build and return state message

        Returns
        -------
        Dict[str, Any]
            Complete system state in dashboard-compatible schema.
        """
        step_start = time.time()

        # =====================================================================
        # STAGE 1: ADVANCE WORLD AND GENERATE CSI
        # =====================================================================
        csi_frame = self._update_loop.step()
        self._update_count += 1

        # Buffer CSI frames for temporal analysis
        self._frame_buffer.append(csi_frame)
        if len(self._frame_buffer) > self._buffer_size:
            self._frame_buffer.pop(0)

        # =====================================================================
        # STAGE 2: SAMPLE WORLD INTO VOXEL GRID
        # This is the ONLY source of spatial information
        # =====================================================================
        self._sample_world_to_voxels()

        # =====================================================================
        # STAGE 3: PROJECT VOXEL GRID TO 2D FLOOR HEATMAP
        # Uses space.projections.floor_projection()
        # =====================================================================
        heatmap = self._project_to_floor()

        # =====================================================================
        # STAGE 4: CLUSTER HEATMAP TO DETECT ENTITIES
        # Clusters are detected in the projected heatmap
        # =====================================================================
        clusters = self._cluster_heatmap(heatmap)

        # =====================================================================
        # STAGE 5: TRACK ENTITIES ACROSS FRAMES
        # Uses nearest-centroid matching for ID persistence
        # =====================================================================
        self._entities = self._entity_tracker.update(
            clusters=clusters,
            x_scale=self._heatmap_x_scale,
            y_scale=self._heatmap_y_scale,
            x_offset=self._heatmap_x_offset,
            y_offset=self._heatmap_y_offset,
            current_time=self._world.time
        )

        # =====================================================================
        # STAGE 6: COMPUTE GLOBAL METRICS FROM CSI
        # These metrics come from CSI inference, not spatial structure
        # =====================================================================
        self._compute_global_metrics(csi_frame)

        # =====================================================================
        # STAGE 7: COMPUTE TIMING METRICS
        # =====================================================================
        step_end = time.time()
        step_duration = step_end - step_start
        self._latency_ms = step_duration * 1000

        # Track FPS over last second
        self._frame_times.append(step_end)
        cutoff = step_end - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]
        self._actual_fps = len(self._frame_times)

        self._last_update_time = step_end

        # Build and return state message
        return self._build_state_message()

    def _sample_world_to_voxels(self) -> None:
        """
        Sample the continuous world disturbance field into the voxel grid.

        For each voxel center, queries world.disturbance_at(x, y, z).
        Applies decay to previous values before accumulating new samples.
        """
        # Apply decay to existing values
        self._voxel_grid.decay(self.config.heatmap_decay)

        # Sample world disturbances at each voxel center
        # Accumulate into existing (decayed) values
        self._voxel_grid.accumulate_world(self._world, weight=1.0)

    def _project_to_floor(self) -> np.ndarray:
        """
        Project the voxel grid to a 2D floor heatmap.

        Uses space.projections.floor_projection() with max projection
        to collapse the Z axis.

        Returns
        -------
        np.ndarray
            2D heatmap array (nx, ny).
        """
        # Use max projection to get floor view (collapse Z axis)
        heatmap = floor_projection(self._voxel_grid, method="max")
        return heatmap

    def _cluster_heatmap(self, heatmap: np.ndarray) -> List[SpatialCluster]:
        """
        Detect clusters in the floor heatmap.

        Parameters
        ----------
        heatmap : np.ndarray
            2D activity heatmap.

        Returns
        -------
        List[SpatialCluster]
            Detected spatial clusters.
        """
        clusters = cluster_spatial_activity(
            heatmap,
            threshold=self.config.activity_threshold,
            min_cluster_size=self.config.min_cluster_size
        )
        return clusters

    def _compute_global_metrics(self, csi_frame: CSIFrame) -> None:
        """
        Compute global metrics from CSI data.

        These metrics reflect overall system activity, not spatial structure.
        The CSI inference pipeline provides:
        - total_activity: Overall intensity of the latent representation
        - novelty: How different current state is from baseline
        - env_drift: Temporal change in the environment

        Parameters
        ----------
        csi_frame : CSIFrame
            Current CSI measurement.
        """
        # Extract features from CSI
        amplitude = csi_frame.amplitude
        phase = csi_frame.phase

        # Normalize and combine features
        amp_normalized = amplitude / (np.max(amplitude) + 1e-8)
        phase_diff = np.diff(phase, prepend=phase[0])
        feature_vector = np.concatenate([amp_normalized, phase_diff])

        # Encode to latent space
        latent = self._encoder.encode(feature_vector)

        # Get inference signals
        signals = self._inference_engine.observe_and_compute(latent)

        # Extract global metrics
        self._total_activity = signals["intensity"]
        self._novelty = signals["novelty_normalized"]
        self._env_drift = signals["temporal_change_avg"] * 0.1

    def _build_state_message(self) -> Dict[str, Any]:
        """
        Build the complete state message in dashboard schema.

        Returns
        -------
        Dict[str, Any]
            Message conforming to dashboard WebSocket schema.
        """
        return {
            "timestamp": float(self._world.time),
            "entities": [e.to_dict() for e in self._entities],
            "global": {
                "total_activity": float(self._total_activity),
                "novelty": float(self._novelty),
                "env_drift": float(self._env_drift)
            },
            "system": {
                "fps": float(self._actual_fps),
                "latency_ms": float(self._latency_ms),
                "sensor_count": int(self._sensor_count),
                "csi_health": str(self._csi_health)
            }
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

if HAS_FASTAPI:
    app = FastAPI(title="CSI Sensing UI Server")

    # Global state manager (initialized on startup)
    state_manager: Optional[SystemStateManager] = None

    # Active WebSocket connection
    active_websocket: Optional[WebSocket] = None

    # Control flags
    running = True


    @app.on_event("startup")
    async def startup_event():
        """Initialize system on server startup."""
        global state_manager

        config = ServerConfig()
        state_manager = SystemStateManager(config)

        # Add demo disturbance objects
        # These create real disturbances in the world that will be:
        # 1. Sampled into the voxel grid
        # 2. Projected to the floor heatmap
        # 3. Clustered into entities
        state_manager.add_disturbance_object(MovingDisturbance(
            x=0.0, y=0.0, z=0.0,
            vx=0.5, vy=0.3,
            radius=1.0, intensity=1.0
        ))
        state_manager.add_disturbance_object(MovingDisturbance(
            x=-2.0, y=2.0, z=0.0,
            vx=-0.3, vy=0.4,
            radius=0.8, intensity=0.8
        ))

        print("System initialized with world → voxel → projection pipeline")
        print(f"Voxel grid shape: {state_manager._voxel_grid.shape}")

        # Start update loop in background
        asyncio.create_task(update_loop())


    async def update_loop():
        """Main update loop running at target FPS."""
        global state_manager, active_websocket, running

        config = ServerConfig()
        target_interval = 1.0 / config.target_fps

        while running:
            loop_start = time.time()

            # Step the system
            if state_manager is not None:
                state = state_manager.step()

                # Send to connected client
                if active_websocket is not None:
                    try:
                        await active_websocket.send_json(state)
                    except Exception:
                        # Client disconnected
                        active_websocket = None

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)


    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for dashboard connection."""
        global active_websocket

        await websocket.accept()
        print(f"Dashboard connected from {websocket.client}")

        # Set as active connection (replaces previous)
        active_websocket = websocket

        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

        except WebSocketDisconnect:
            print("Dashboard disconnected")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            if active_websocket == websocket:
                active_websocket = None


    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "running", "service": "CSI Sensing UI Server"}


    @app.get("/status")
    async def status():
        """Get current system status."""
        if state_manager is not None:
            return state_manager._build_state_message()
        return {"status": "not initialized"}


# =============================================================================
# STANDALONE SERVER (WITHOUT FASTAPI)
# =============================================================================

async def standalone_server(config: ServerConfig):
    """
    Standalone WebSocket server using websockets library.
    Fallback if FastAPI is not available.
    """
    try:
        import websockets
    except ImportError:
        print("Error: Neither FastAPI nor websockets is installed.")
        print("Install with: pip install fastapi uvicorn")
        print("         or: pip install websockets")
        return

    # Initialize system
    state_manager = SystemStateManager(config)

    # Add demo disturbances
    state_manager.add_disturbance_object(MovingDisturbance(
        x=0.0, y=0.0, z=0.0, vx=0.5, vy=0.3, radius=1.0, intensity=1.0
    ))
    state_manager.add_disturbance_object(MovingDisturbance(
        x=-2.0, y=2.0, z=0.0, vx=-0.3, vy=0.4, radius=0.8, intensity=0.8
    ))

    print("System initialized with world → voxel → projection pipeline")
    print(f"Voxel grid shape: {state_manager._voxel_grid.shape}")

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

    async def broadcast_loop():
        target_interval = 1.0 / config.target_fps

        while True:
            loop_start = time.time()

            state = state_manager.step()
            state_json = json.dumps(state)

            if connected_clients:
                await asyncio.gather(
                    *[client.send(state_json) for client in connected_clients],
                    return_exceptions=True
                )

            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)

    server = await websockets.serve(handler, config.host, config.port)
    print(f"WebSocket server running at ws://{config.host}:{config.port}{config.ws_path}")

    await broadcast_loop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_server(
    host: str = "localhost",
    port: int = 8000,
    target_fps: float = 8.0
):
    """
    Run the UI server.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to bind to.
    target_fps : float
        Target update rate in Hz.
    """
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
