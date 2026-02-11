# CSI Spatial Sensing Dashboard

## Overview
A real-time WebSocket-based spatial sensing dashboard that visualizes CSI (Channel State Information) data. The system simulates RF link sensing with geometry-constrained fusion pipeline and displays entities, activity heatmaps, and system metrics in a browser-based dashboard.

## Architecture
- **Backend**: Python FastAPI server with WebSocket streaming (`visualization/ui_server.py`)
- **Frontend**: Single-page HTML dashboard (`dashboard/index.html`) served by FastAPI
- **Pipeline**: World simulation -> Synthetic CSI generation -> Signal conditioning -> Geometry fusion -> Clustering -> Entity tracking -> WebSocket broadcast

## Key Directories
- `visualization/` - UI server (FastAPI + WebSocket)
- `dashboard/` - Frontend HTML dashboard
- `inference/` - Signal processing pipeline (clustering, feature extraction, geometry fusion)
- `csi/` - CSI frame and synthetic CSI generation
- `world/` - World simulation with moving disturbances
- `space/` - Voxel grid and spatial projections
- `contracts/` - Data contracts and validation
- `pipeline/` - Update loop
- `tests/` - Pytest test suite

## Running
- Server runs on `0.0.0.0:5000` via `python -m visualization.ui_server --host 0.0.0.0 --port 5000 --fps 8`
- Dashboard connects to backend WebSocket dynamically using the current page host
- Deployment target: autoscale

## Dependencies
- Python 3.11
- fastapi, uvicorn, websockets, numpy, pytest (see requirements.txt)
