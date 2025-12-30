# Routes Module

FastAPI routes for the RingRift AI service REST API.

## Overview

This module provides HTTP endpoints for:

- Game replay browsing and playback
- Cluster status and sync coordination
- Training feedback and Elo velocity
- Database statistics

## Modules

| File          | Description                               | Prefix          |
| ------------- | ----------------------------------------- | --------------- |
| `replay.py`   | Game replay browsing and playback         | `/api/replay`   |
| `cluster.py`  | Cluster status, sync, and node management | `/api/cluster`  |
| `training.py` | Training feedback and Elo tracking        | `/api/training` |

## Endpoints

### Replay API (`replay.py`)

| Endpoint                              | Method | Description                |
| ------------------------------------- | ------ | -------------------------- |
| `/api/replay/games`                   | GET    | List games with filters    |
| `/api/replay/games/{game_id}`         | GET    | Get game details + players |
| `/api/replay/games/{game_id}/moves`   | GET    | Get moves for a game       |
| `/api/replay/games/{game_id}/state`   | GET    | Get state at move N        |
| `/api/replay/games/{game_id}/choices` | GET    | Get player choices         |
| `/api/replay/stats`                   | GET    | Get database statistics    |

### Query Parameters

**List Games** (`/api/replay/games`):

- `board_type`: Filter by board type (square8, hex8, etc.)
- `num_players`: Filter by player count (2-4)
- `winner`: Filter by winning player
- `termination_reason`: Filter by how game ended
- `source`: Filter by game source
- `min_moves`, `max_moves`: Filter by move count
- `limit`: Max results (default 20, max 100)
- `offset`: Pagination offset

## Configuration

Environment variables:

- `GAME_REPLAY_DB_PATH`: Path to SQLite database (default: `data/games/selfplay.db`)

## Usage

```python
from fastapi import FastAPI
from app.routes.replay import router

app = FastAPI()
app.include_router(router)
```

## Response Models

All responses use Pydantic models for validation:

- `GameMetadata`: Game info including players
- `MoveRecord`: Individual move with metadata
- `StatsResponse`: Database statistics

### Cluster API (`cluster.py`)

| Endpoint                         | Method | Description                 |
| -------------------------------- | ------ | --------------------------- |
| `/api/cluster/status`            | GET    | Cluster-wide status summary |
| `/api/cluster/sync/status`       | GET    | Sync subsystem status       |
| `/api/cluster/sync/trigger`      | POST   | Trigger manual sync         |
| `/api/cluster/manifest`          | GET    | Data manifest summary       |
| `/api/cluster/nodes`             | GET    | List all cluster nodes      |
| `/api/cluster/nodes/{node_id}`   | GET    | Get specific node details   |
| `/api/cluster/health`            | GET    | Simple health check         |
| `/api/cluster/health/aggregated` | GET    | Aggregated cluster health   |
| `/api/cluster/config`            | GET    | Cluster configuration       |

### Training API (`training.py`)

| Endpoint                              | Method | Description                  |
| ------------------------------------- | ------ | ---------------------------- |
| `/api/training/status`                | GET    | Training subsystem status    |
| `/api/training/feedback`              | GET    | Training feedback status     |
| `/api/training/{config_key}/velocity` | GET    | Elo velocity for config      |
| `/api/training/{config_key}/momentum` | GET    | Training momentum for config |

## Database Integration

Uses `GameReplayDB` from `app.db.game_replay` for data access:

- Singleton pattern for connection reuse
- Read-only endpoints (DB open may still trigger schema migrations)
- Auto-migrates schema v1+ databases to current schema v16 on open

## Response Models

All responses use Pydantic models:

- **Replay**: `GameMetadata`, `MoveRecord`, `StatsResponse`
- **Cluster**: `ClusterStatusResponse`, `NodeInventoryResponse`, `SyncStatusResponse`
- **Training**: `EloVelocityResponse`, `MomentumResponse`, `TrainingStatusResponse`
