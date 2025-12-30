# Daemon Manager Operations Runbook

**Last Updated**: December 30, 2025
**Version**: Wave 7

## Overview

The `DaemonManager` coordinates 89 daemon types (78 active, 11 deprecated) in the RingRift training pipeline. It handles lifecycle management, health monitoring, and auto-restart with exponential backoff. For the full list, see `../DAEMON_REGISTRY.md`.

## Daemon Types

Daemons are organized by category:

### Core Infrastructure

| Daemon            | Purpose               | Health Check |
| ----------------- | --------------------- | ------------ |
| `EVENT_ROUTER`    | Central event bus     | Yes          |
| `DAEMON_WATCHDOG` | Health monitoring     | Yes          |
| `HEALTH_SERVER`   | HTTP health endpoints | Yes          |

### Synchronization

| Daemon               | Purpose                     | Health Check |
| -------------------- | --------------------------- | ------------ |
| `AUTO_SYNC`          | Automated P2P data sync     | Yes          |
| `MODEL_DISTRIBUTION` | Model distribution to nodes | Yes          |
| `ELO_SYNC`           | Elo rating synchronization  | Yes          |
| `NPZ_DISTRIBUTION`   | Training data distribution  | Yes          |
| `STALE_FALLBACK`     | Stale model fallback        | Yes          |

### Training Pipeline

| Daemon                  | Purpose                      | Health Check |
| ----------------------- | ---------------------------- | ------------ |
| `DATA_PIPELINE`         | Pipeline stage orchestration | Yes          |
| `SELFPLAY_COORDINATOR`  | Priority selfplay scheduling | Yes          |
| `TRAINING_NODE_WATCHER` | Training activity detection  | Yes          |
| `TRAINING_TRIGGER`      | Training initiation          | Yes          |

### Training Activity Config

The `TRAINING_NODE_WATCHER` daemon is backed by `TrainingActivityDaemon` and uses
the `RINGRIFT_TRAINING_ACTIVITY_*` env prefix (via `DaemonConfig.from_env`):

- `RINGRIFT_TRAINING_ACTIVITY_ENABLED` (default: `1`)
- `RINGRIFT_TRAINING_ACTIVITY_INTERVAL` (seconds, default: `30`)
- `RINGRIFT_TRAINING_ACTIVITY_HANDLE_SIGNALS` (default: `1`)
- `RINGRIFT_TRAINING_ACTIVITY_TRIGGER_SYNC` (default: `1`)

### Evaluation

| Daemon           | Purpose                      | Health Check |
| ---------------- | ---------------------------- | ------------ |
| `EVALUATION`     | Model evaluation             | Yes          |
| `AUTO_PROMOTION` | Automatic model promotion    | Yes          |
| `FEEDBACK_LOOP`  | Training feedback controller | Yes          |

### Resource Management

| Daemon             | Purpose            | Health Check |
| ------------------ | ------------------ | ------------ |
| `IDLE_RESOURCE`    | Idle GPU detection | Yes          |
| `NODE_RECOVERY`    | Node recovery      | Yes          |
| `CLUSTER_WATCHDOG` | Cluster health     | Yes          |
| `MEMORY_MONITOR`   | VRAM/RAM pressure  | Yes          |

## Daemon Lifecycle States

```
STARTING -> RUNNING -> STOPPING -> STOPPED
              |
              v
           FAILED (after max restarts)
```

| State      | Description                   |
| ---------- | ----------------------------- |
| `STARTING` | Daemon initializing           |
| `RUNNING`  | Normal operation              |
| `STOPPING` | Graceful shutdown in progress |
| `STOPPED`  | Cleanly stopped               |
| `FAILED`   | Exceeded max restart attempts |

## Health Check Protocol

All daemons implement `health_check()` returning `HealthCheckResult`:

```python
from app.coordination.protocols import HealthCheckResult

result = HealthCheckResult(
    healthy=True,
    status="running",
    message="Last sync 30s ago",
    details={
        "files_synced": 42,
        "errors_count": 0,
    }
)
```

### Health Status Values

| Status      | Meaning                 |
| ----------- | ----------------------- |
| `healthy`   | Normal operation        |
| `degraded`  | Working but with issues |
| `unhealthy` | Not functioning         |
| `unknown`   | Cannot determine status |

## Auto-Restart Behavior

Failed daemons are automatically restarted with exponential backoff:

| Attempt | Delay            |
| ------- | ---------------- |
| 1       | 1 second         |
| 2       | 2 seconds        |
| 3       | 4 seconds        |
| 4       | 8 seconds        |
| 5       | 16 seconds (max) |

After 5 failed attempts, daemon enters `FAILED` state and is not restarted.

## Profile-Based Startup

Different node roles start different daemon subsets:

### Coordinator Profile

```python
COORDINATOR_DAEMONS = [
    "EVENT_ROUTER",
    "DATA_PIPELINE",
    "FEEDBACK_LOOP",
    "AUTO_SYNC",
    "MODEL_DISTRIBUTION",
    "EVALUATION",
    "COORDINATOR_DISK_MANAGER",
]
```

### Training Node Profile

```python
TRAINING_NODE_DAEMONS = [
    "EVENT_ROUTER",
    "TRAINING_NODE_WATCHER",
    "DISK_SPACE_MANAGER",
]
```

### Selfplay Node Profile

```python
SELFPLAY_NODE_DAEMONS = [
    "EVENT_ROUTER",
    "IDLE_RESOURCE",
    "DISK_SPACE_MANAGER",
]
```

## Operational Commands

### Check Daemon Status

```bash
python scripts/launch_daemons.py --status
```

Output:

```
Daemon Status:
  EVENT_ROUTER: RUNNING (healthy)
  DATA_PIPELINE: RUNNING (healthy)
  AUTO_SYNC: RUNNING (degraded - 2 sync errors)
  EVALUATION: STOPPED
```

### Start All Daemons

```bash
python scripts/launch_daemons.py --all
```

### Start Specific Daemons

```bash
python scripts/launch_daemons.py --daemons AUTO_SYNC,DATA_PIPELINE,FEEDBACK_LOOP
```

### Stop All Daemons

```bash
python scripts/launch_daemons.py --stop
```

### Health Check via HTTP

```bash
# Liveness probe (are daemons running?)
curl http://localhost:8790/health

# Readiness probe (are daemons ready to serve?)
curl http://localhost:8790/ready

# Detailed metrics
curl http://localhost:8790/metrics
```

## Programmatic Access

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Start a daemon
await dm.start(DaemonType.AUTO_SYNC)

# Check health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(f"Status: {health['status']}, Details: {health['details']}")

# Get all daemon health
all_health = dm.get_all_daemon_health()

# Liveness probe
if dm.liveness_probe():
    print("All critical daemons healthy")
```

## Graceful Shutdown

Shutdown order is reverse of startup to respect dependencies:

1. Resource daemons (IDLE_RESOURCE, NODE_RECOVERY)
2. Evaluation daemons (EVALUATION, AUTO_PROMOTION)
3. Pipeline daemons (DATA_PIPELINE, TRAINING_TRIGGER)
4. Sync daemons (AUTO_SYNC, MODEL_DISTRIBUTION)
5. Core daemons (EVENT_ROUTER, DAEMON_WATCHDOG)

Each daemon gets 30 seconds to shut down gracefully.

## Common Issues

### 1. Daemon Stuck in STARTING

**Symptoms**: Daemon never reaches RUNNING state

**Diagnosis**:

```python
health = dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(health)
```

**Resolution**:

1. Check for import errors in daemon module
2. Verify dependencies are available
3. Check logs: `grep "AUTO_SYNC" logs/daemons.log`

### 2. Daemon Keeps Crashing

**Symptoms**: High restart count, eventually FAILED

**Diagnosis**:

```python
info = dm.get_daemon_info(DaemonType.AUTO_SYNC)
print(f"Restarts: {info.restart_count}, Last error: {info.last_error}")
```

**Resolution**:

1. Check exception in last_error
2. Verify configuration is valid
3. Check resource availability (disk, network)

### 3. Health Checks Failing

**Symptoms**: Daemon reports unhealthy

**Diagnosis**:

```python
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(f"Message: {health['message']}")
print(f"Details: {health['details']}")
```

**Resolution**:

1. Check `details` field for specific issues
2. Verify external dependencies (database, network)
3. Check disk space on data directories

## Daemon Registry

All 89 daemon types are defined in `app/coordination/daemon_registry.py`:

```python
from app.coordination.daemon_registry import (
    DAEMON_REGISTRY,
    get_daemons_by_category,
    validate_registry,
)

# Get all sync daemons
sync_daemons = get_daemons_by_category("sync")

# Validate registry at startup
errors = validate_registry()
if errors:
    raise RuntimeError(f"Registry errors: {errors}")
```

## Environment Variables

| Variable                           | Default | Description                     |
| ---------------------------------- | ------- | ------------------------------- |
| `RINGRIFT_DAEMON_HEALTH_PORT`      | 8790    | Health endpoint port            |
| `RINGRIFT_DAEMON_RESTART_MAX`      | 5       | Max restart attempts            |
| `RINGRIFT_DAEMON_HEALTH_INTERVAL`  | 30      | Health check interval (sec)     |
| `RINGRIFT_DAEMON_SHUTDOWN_TIMEOUT` | 30      | Graceful shutdown timeout (sec) |

## See Also

- [P2P_ORCHESTRATOR_OPERATIONS.md](P2P_ORCHESTRATOR_OPERATIONS.md) - P2P cluster
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event routing
