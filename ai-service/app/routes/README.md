# Routes Module

FastAPI routes for the RingRift AI service REST API.

## Overview

This module provides HTTP endpoints for:

- Game replay browsing and playback
- Database statistics

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

## Database Integration

Uses `GameReplayDB` from `app.db.game_replay` for data access:

- Singleton pattern for connection reuse
- Read-only operations
- Supports schema v5+ databases
