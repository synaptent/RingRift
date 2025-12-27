# AI Service API Reference

## Overview

The RingRift AI service exposes three main API layers:

1. **Flask REST API** (port 5000) - Game replay and training status endpoints
2. **P2P Orchestrator** (port 8770) - Cluster coordination and status
3. **Health Server** (port 8790) - Daemon health and Prometheus metrics

## Flask REST API Endpoints (Port 5000)

### Training Status Routes

#### GET /api/training/{config_key}/velocity

Get Elo velocity (improvement rate) for a configuration.

**Parameters:**

- `config_key` (path): Configuration key (e.g., "hex8_2p", "square8_4p")
- `lookback_hours` (query, default: 24.0): Hours to look back for trend calculation

**Response:**

```json
{
  "config_key": "hex8_2p",
  "elo_per_hour": 2.5,
  "elo_per_game": 0.15,
  "trend": "improving|stable|declining",
  "lookback_hours": 24.0,
  "games_in_period": 150,
  "current_elo": 1750.0
}
```

#### GET /api/training/{config_key}/momentum

Get full momentum status for a configuration including training intensity and improvement tracking.

**Response:**

```json
{
  "config_key": "hex8_2p",
  "current_elo": 1750.0,
  "momentum_state": "accelerating|maintaining|decelerating|stalled",
  "intensity": "low|medium|high|peak",
  "games_since_training": 0,
  "consecutive_improvements": 3,
  "consecutive_plateaus": 0,
  "elo_trend": 5.5,
  "improvement_rate_per_hour": 2.5
}
```

#### GET /api/training/status

Get aggregate training status across all configurations.

#### GET /api/training/feedback

Get FeedbackAccelerator status with selfplay recommendations.

### Cluster Routes

#### GET /api/cluster/status

Get cluster health summary including leader election and peer connectivity.

**Response:**

```json
{
  "node_id": "mac-studio",
  "is_leader": true,
  "leader_id": "mac-studio",
  "alive_peers": 28,
  "total_peers": 30,
  "uptime_seconds": 3600.0,
  "job_count": 127,
  "cluster_healthy": true
}
```

#### GET /api/cluster/sync/status

Get sync daemon status with transfer statistics.

#### GET /api/cluster/manifest

Get cluster manifest summary with data counts and replication status.

#### GET /api/cluster/nodes

List all nodes with their inventory.

#### GET /api/cluster/nodes/{node_id}

Get detailed inventory for a specific node.

#### POST /api/cluster/sync/trigger

Manually trigger a sync operation.

#### GET /api/cluster/health

Quick cluster health check (returns 200 if healthy, 503 otherwise).

#### GET /api/cluster/config

Get cluster configuration summary with host list and sync settings.

#### GET /api/cluster/health/aggregated

Get comprehensive aggregated health from all cluster subsystems.

### Replay API Endpoints

#### GET /api/replay/games

List games with optional filters.

**Query Parameters:**

- `board_type` (string, max 50): Filter by board type
- `num_players` (integer, 2-4): Filter by player count
- `winner` (integer, 1-4): Filter by winning player
- `limit` (integer, 1-100, default 20): Max results to return
- `offset` (integer, default 0): Pagination offset

#### GET /api/replay/games/{game_id}

Get detailed metadata for a specific game including player info.

#### GET /api/replay/games/{game_id}/state

Get reconstructed game state at a specific move.

**Query Parameters:**

- `move_number` (integer, default 0): Move number (0 = initial state)

#### GET /api/replay/games/{game_id}/moves

Get moves for a game in a range.

#### GET /api/replay/games/{game_id}/choices

Get player choices made at a specific move.

#### GET /api/replay/stats

Get database statistics.

#### POST /api/replay/games

Store a game from the sandbox.

## P2P Orchestrator Endpoints (Port 8770)

### GET /status

Get P2P cluster status with peer information and leader election status.

**Response:**

```json
{
  "node_id": "mac-studio",
  "leader_id": "mac-studio",
  "is_leader": true,
  "alive_peers": 28,
  "total_peers": 30,
  "uptime": 3600.0,
  "job_count": 127,
  "peers": [...],
  "swim_raft": {
    "membership_mode": "http",
    "consensus_mode": "bully"
  }
}
```

### GET /health

Health check endpoint (returns 200 if healthy, 503 otherwise).

### POST /gossip

Gossip protocol endpoint for peer discovery and state propagation (internal use).

### GET /jobs

List active jobs on the cluster.

### GET /elect

Leader election endpoint (internal use for Raft consensus).

## Health Server Endpoints (Port 8790)

### GET /health

Liveness probe - returns 200 if daemon is alive and responsive.

### GET /ready

Readiness probe - returns 200 if system is ready to handle requests, 503 otherwise.

### GET /metrics

Prometheus-style metrics for monitoring.

**Response (text/plain):**

```
# HELP daemon_count Number of daemons
# TYPE daemon_count gauge
daemon_count{state="running"} 45
daemon_count{state="stopped"} 18

# HELP daemon_health_score Overall health score (0-1)
# TYPE daemon_health_score gauge
daemon_health_score 0.95
```

### GET /status

Detailed daemon status (alternative to /health).

## Event Types (DataEventType)

The coordination infrastructure uses 50+ event types for pipeline coordination:

### Training Events

- `TRAINING_STARTED` - Training job started
- `TRAINING_COMPLETED` - Training job completed
- `TRAINING_FAILED` - Training job failed

### Evaluation Events

- `EVALUATION_STARTED` - Evaluation started
- `EVALUATION_COMPLETED` - Evaluation completed

### Model Events

- `MODEL_PROMOTED` - Model promoted to production
- `MODEL_DISTRIBUTION_COMPLETE` - Model distributed to cluster

### Selfplay Events

- `SELFPLAY_COMPLETE` - Selfplay batch finished
- `NEW_GAMES_AVAILABLE` - New games available for training

### Data Sync Events

- `DATA_SYNC_COMPLETED` - Data sync completed
- `DATA_STALE` - Training data is stale

### Cluster Events

- `HOST_ONLINE` / `HOST_OFFLINE` - Host state changes
- `P2P_NODE_DEAD` - Single node confirmed dead
- `LEADER_ELECTED` - New leader elected

### Daemon Events

- `DAEMON_STARTED` / `DAEMON_STOPPED` - Daemon lifecycle
- `ALL_CRITICAL_DAEMONS_READY` - All critical daemons ready

### Event Subscription Example

```python
from app.coordination.event_router import DataEventType, get_event_bus

bus = get_event_bus()
bus.subscribe(DataEventType.TRAINING_COMPLETED.value, on_training_completed)
```

## Error Responses

All endpoints return appropriate HTTP status codes:

- **200 OK** - Request successful
- **400 Bad Request** - Invalid parameters
- **404 Not Found** - Resource not found
- **500 Internal Server Error** - Server error
- **503 Service Unavailable** - Service degraded

**Error Response Format:**

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Configuration

### Environment Variables

**Flask API Server:**

- `RINGRIFT_API_HOST` - API server host (default: 127.0.0.1)
- `RINGRIFT_API_PORT` - API server port (default: 5000)

**P2P Orchestrator:**

- `RINGRIFT_P2P_PORT` - P2P port (default: 8770)

**Health Server:**

- `RINGRIFT_HEALTH_PORT` - Health server port (default: 8790)

### Board Type Configuration

All endpoints support these board types:

- `square8` - 8x8 square board
- `square19` - 19x19 large square board
- `hex8` - Small hexagonal board (radius 4, 61 cells)
- `hexagonal` - Large hexagonal board (radius 12, 469 cells)

All board types support 2, 3, and 4 player counts.
