# Daemon System Architecture

## Overview

RingRift uses a daemon-based architecture for background automation. The `DaemonManager` orchestrates 30+ daemon types that handle selfplay, training, sync, health monitoring, and cluster coordination.

## Quick Start

```bash
# Full automation mode (recommended)
python scripts/master_loop.py

# Watch status without running
python scripts/master_loop.py --watch

# Launch specific daemons
python scripts/launch_daemons.py --all
python scripts/launch_daemons.py --sync
python scripts/launch_daemons.py --status
```

## Architecture

```
MasterLoopController
├── DaemonManager: Lifecycle for all daemons
│   ├── DaemonType enum (30+ types)
│   ├── DaemonState (INITIALIZING, RUNNING, PAUSED, SHUTTING_DOWN)
│   └── DaemonAdapter wrappers
├── ClusterMonitor: Real-time cluster health
├── SelfplayScheduler: Priority-based selfplay allocation
├── FeedbackLoopController: Training feedback signals
├── DataPipelineOrchestrator: Pipeline stage tracking
└── QueuePopulator: Work queue maintenance
```

## Daemon Categories

### Sync Daemons

| Daemon               | File                           | Purpose                                  |
| -------------------- | ------------------------------ | ---------------------------------------- |
| `AUTO_SYNC`          | `auto_sync_daemon.py`          | Push-from-generator + gossip replication |
| `EPHEMERAL_SYNC`     | `ephemeral_sync.py`            | Aggressive 5s sync for Vast.ai           |
| `CLUSTER_DATA_SYNC`  | `cluster_data_sync.py`         | P2P mesh data sync                       |
| `MODEL_DISTRIBUTION` | `model_distribution_daemon.py` | Model sync after promotion               |

### Training Daemons

| Daemon             | File                         | Purpose                    |
| ------------------ | ---------------------------- | -------------------------- |
| `TRAINING_TRIGGER` | `training_trigger_daemon.py` | Data threshold → training  |
| `AUTO_EXPORT`      | `auto_export_daemon.py`      | DB → NPZ export automation |
| `EVALUATION`       | `evaluation_daemon.py`       | Model gauntlet evaluation  |
| `AUTO_PROMOTION`   | `auto_promotion_daemon.py`   | Gauntlet → promotion       |

### Health & Resource Daemons

| Daemon                  | File                              | Purpose                         |
| ----------------------- | --------------------------------- | ------------------------------- |
| `UNIFIED_NODE_HEALTH`   | `unified_node_health_daemon.py`   | Centralized health checks       |
| `QUALITY_MONITOR`       | `quality_monitor_daemon.py`       | Data quality tracking           |
| `IDLE_RESOURCE`         | `idle_resource_daemon.py`         | Spawn selfplay on idle GPUs     |
| `UNIFIED_IDLE_SHUTDOWN` | `unified_idle_shutdown_daemon.py` | Cloud instance idle termination |

### Queue & Scheduling

| Daemon               | File                         | Purpose                   |
| -------------------- | ---------------------------- | ------------------------- |
| `QUEUE_POPULATOR`    | `unified_queue_populator.py` | Maintain work queue       |
| `SELFPLAY_SCHEDULER` | Built into master_loop       | Priority-based allocation |
| `JOB_REAPER`         | `job_reaper.py`              | Kill stuck processes      |

## Daemon Lifecycle

```
INITIALIZING → RUNNING → PAUSED → SHUTTING_DOWN → STOPPED
                  ↑         ↓
                  └─────────┘ (resume)
```

### State Transitions

1. **INITIALIZING**: Daemon started, loading state
2. **RUNNING**: Active and processing
3. **PAUSED**: Temporarily stopped (can resume)
4. **SHUTTING_DOWN**: Graceful shutdown in progress
5. **STOPPED**: Fully stopped

## DaemonManager API

```python
from app.coordination.daemon_manager import DaemonManager, DaemonType

# Get singleton manager
manager = DaemonManager.get_instance()

# Start specific daemon
await manager.start_daemon(DaemonType.AUTO_SYNC)

# Stop daemon
await manager.stop_daemon(DaemonType.AUTO_SYNC)

# Get status
status = manager.get_status(DaemonType.AUTO_SYNC)

# Start all daemons in a profile
await manager.start_profile("full")  # coordinator, training_node, ephemeral, selfplay, full, minimal
```

## Daemon Adapters

Existing daemons are wrapped via `DaemonAdapter`:

```python
from app.coordination.daemon_adapters import create_daemon_adapter

# Wrap an existing daemon class
adapter = create_daemon_adapter(
    daemon_type=DaemonType.AUTO_SYNC,
    daemon_class=AutoSyncDaemon,
    config={"interval": 60}
)
```

## Dependencies

Some daemons must start before others:

```
AUTO_SYNC ─────────────┐
CLUSTER_DATA_SYNC ─────┤
                       ├──→ TRAINING_TRIGGER ──→ EVALUATION ──→ AUTO_PROMOTION
AUTO_EXPORT ───────────┤
                       │
QUALITY_MONITOR ───────┘
```

## Event Integration

Daemons emit and subscribe to events via `EventRouter`:

```python
from app.coordination.event_router import get_router
from app.distributed.data_events import DataEventType

# Subscribe to events
router = get_router()
router.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games)

# Emit events
await router.emit(DataEventType.TRAINING_STARTED, {"config_key": "hex8_2p"})
```

## Configuration

### Environment Variables

| Variable                      | Default | Description                            |
| ----------------------------- | ------- | -------------------------------------- |
| `RINGRIFT_DAEMON_INTERVAL`    | 60      | Default daemon loop interval (seconds) |
| `RINGRIFT_SYNC_INTERVAL`      | 60      | Sync daemon interval                   |
| `RINGRIFT_EPHEMERAL_INTERVAL` | 5       | Ephemeral sync interval (Vast.ai)      |

### YAML Configuration

```yaml
# config/unified_loop.yaml
daemons:
  auto_sync:
    enabled: true
    interval: 60
  ephemeral_sync:
    enabled: true
    interval: 5
  training_trigger:
    enabled: true
    threshold_games: 1000
```

## Monitoring

### Health Checks

```bash
# Check daemon status
curl http://localhost:8770/daemon/status

# Check specific daemon
curl http://localhost:8770/daemon/auto_sync/health
```

### Logs

```bash
# Master loop logs
tail -f logs/master_loop.log

# Per-daemon logs
tail -f logs/auto_sync.log
tail -f logs/training_trigger.log
```

## Troubleshooting

### Daemon Not Starting

1. Check dependencies are running
2. Verify database permissions: `chown ubuntu:ubuntu data/*.db`
3. Check logs: `tail -f logs/master_loop.log`

### Daemon Stuck

1. Check for long-running operations: `pgrep -f daemon_name`
2. Use JOB_REAPER to kill stuck processes
3. Restart via: `python scripts/launch_daemons.py --restart daemon_name`

### Memory Issues

1. Check daemon memory: `ps aux | grep daemon_name`
2. Reduce batch sizes in daemon config
3. Enable incremental processing modes

## Files Reference

- `app/coordination/daemon_manager.py` - Core daemon lifecycle
- `app/coordination/daemon_adapters.py` - Wrapper adapters
- `app/coordination/daemon_types.py` - DaemonType enum
- `scripts/master_loop.py` - Unified automation entry point
- `scripts/launch_daemons.py` - Manual daemon launcher
