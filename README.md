# Wi-Fi CSI Spatial Sensing Stack

A real-time spatial sensing system that uses Wi-Fi **Channel State Information (CSI)** to detect and track moving entities in a space — without cameras or wearables. The stack simulates RF link sensing, runs a geometry-constrained fusion pipeline, and streams live entity tracks, activity heatmaps, and system metrics to a browser-based dashboard over WebSocket.

## Overview

The pipeline runs end to end:

World simulation -> Synthetic CSI generation -> Signal conditioning -> Geometry fusion -> Clustering -> Entity tracking -> WebSocket broadcast

A FastAPI backend serves both the WebSocket stream and a single-page dashboard that renders detected entities and a floor activity heatmap in real time.

## Architecture

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Capture / simulation | `world/`, `csi/` | World model with moving disturbances; synthetic CSI frame generation |
| Spatial representation | `space/` | Voxel grid and 2D/3D spatial projections |
| Inference | `inference/` | Preprocessing, feature extraction, geometry fusion, clustering, tracking |
| Data contracts | `contracts/` | Typed, validated, immutable data structures (see `docs/contracts.md`) |
| Pipeline | `pipeline/` | Update loop tying the stages together |
| Visualization | `visualization/`, `dashboard/` | FastAPI + WebSocket server and HTML dashboard |
| Tests | `tests/` | Pytest suite |

## Tech Stack

- **Python 3.11 / 3.12**
- **FastAPI** + **Uvicorn** — WebSocket server and dashboard host
- **NumPy** — signal processing and spatial math
- **WebSockets** — live streaming to the browser
- **Pytest** — test suite (run in CI on 3.11 and 3.12)

## Quick Start

```bash
./scripts/bootstrap.sh      # set up environment
./scripts/run_server.sh     # start the backend server
./scripts/run_tests.sh      # run the test suite
```

Or use the Makefile:

```bash
make install
make test
make run-server
```

The server runs on `0.0.0.0:5000`. Open the dashboard in your browser to view live entity tracks and the activity heatmap; the frontend connects to the WebSocket using the current page host.

> If your system Python blocks global package installs, use the virtual environment created by `./scripts/bootstrap.sh`.

## Data Contracts

Core data structures (CSI frames, spatial fields, detections, tracks) are implemented as frozen, runtime-validated dataclasses for type safety and clean layer separation. See [`docs/contracts.md`](docs/contracts.md) for the full reference.

## Continuous Integration

GitHub Actions runs the Pytest suite on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).
