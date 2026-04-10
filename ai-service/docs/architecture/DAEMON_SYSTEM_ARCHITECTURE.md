# Daemon System Architecture

## Overview

RingRift uses a daemon-based architecture for background automation. The `DaemonManager` registry currently covers all 127 `DaemonType` enum values. Of those, 26 are marked deprecated in the registry and should not be started by the audited `full` master-loop profile.

The current strategy is to keep `scripts/minimal_alphazero_loop.py` as the reproducible proof harness, while auditing and reusing the larger daemon/P2P infrastructure for supervision, health, job lifecycle, sync, and evaluation scheduling. The larger daemon stack should not be treated as the research source of truth until its data, training, and evaluation contracts match the minimal loop.

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
│   ├── DaemonType enum (127 types)
│   ├── DAEMON_REGISTRY (127 registered specs)
│   ├── DaemonState (INITIALIZING, RUNNING, PAUSED, SHUTTING_DOWN)
│   └── DaemonAdapter wrappers
├── ClusterMonitor: Real-time cluster health
├── SelfplayScheduler: Priority-based selfplay allocation
├── FeedbackLoopController: Training feedback signals
├── DataPipelineOrchestrator: Pipeline stage tracking
└── QueuePopulator: Work queue maintenance
```

## Master-Loop Profiles

`scripts/master_loop.py` now validates profile names through `validate_daemon_profile()` before constructing the controller. The supported profiles are:

| Profile    | Baseline daemons | Purpose                                                                 |
| ---------- | ---------------- | ----------------------------------------------------------------------- |
| `minimal`  | 11               | Event, health, sync, feedback, data pipeline, and auto-export basics    |
| `lean`     | 23               | Reuse-focused profile for pipeline, health, sync, training, and eval    |
| `standard` | 46               | Larger automation stack with more consolidation, recovery, and feedback |
| `full`     | 101              | All non-deprecated registered daemon types                              |

These counts are before optional S3/PARITY additions and before coordinator or standby coordinator filtering. The `lean` profile is the preferred reuse target because it starts the essential pipeline without intentionally starting the highest-risk legacy process spawners. Current tests assert that supported profiles validate, resolve only active registry daemons, and avoid duplicates.

## Daemon Categories

### Sync Daemons

| Daemon               | File                             | Purpose                                  |
| -------------------- | -------------------------------- | ---------------------------------------- |
| `AUTO_SYNC`          | `auto_sync_daemon.py`            | Push-from-generator + gossip replication |
| `MODEL_DISTRIBUTION` | `unified_distribution_daemon.py` | Model/NPZ sync after promotion           |

**Deprecated Sync Daemons (legacy daemon types, Q2 2026 removal):**

| Daemon              | File                  | Replacement                                     |
| ------------------- | --------------------- | ----------------------------------------------- |
| `EPHEMERAL_SYNC`    | `auto_sync_daemon.py` | `AutoSyncDaemon(strategy="ephemeral")` (legacy) |
| `CLUSTER_DATA_SYNC` | `auto_sync_daemon.py` | `AutoSyncDaemon(strategy="broadcast")` (legacy) |

Note: `ephemeral_sync.py` and `cluster_data_sync.py` were removed during consolidation; their
behavior now lives in `auto_sync_daemon.py`. `model_distribution_daemon.py` and
`npz_distribution_daemon.py` have been consolidated into `unified_distribution_daemon.py`.

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
                       │
                       ├──→ TRAINING_TRIGGER ──→ EVALUATION ──→ AUTO_PROMOTION
AUTO_EXPORT ───────────┤
                       │
QUALITY_MONITOR ───────┘
```

Note: `CLUSTER_DATA_SYNC` was removed from the dependency graph (deprecated Dec 2025).
`AUTO_SYNC` with appropriate strategy replaces both `EPHEMERAL_SYNC` and `CLUSTER_DATA_SYNC`.

## Event Integration

Active coordination code should emit through `app.coordination.event_emission_helpers.safe_emit_event()` or `safe_emit_event_async()`. `app.coordination.safe_event_emitter.SafeEventEmitterMixin` remains the compatibility mixin for classes that need `_safe_emit_event()` and delegates to the consolidated helper. Direct calls to router-level `emit_event()` should not be added to active coordination daemons.

```python
from app.coordination.event_emission_helpers import safe_emit_event
from app.distributed.data_events import DataEventType

safe_emit_event(
    DataEventType.TRAINING_STARTED,
    {"config_key": "hex8_2p"},
    source="training_trigger",
)
```

The event router remains the delivery layer and still exposes compatibility helpers for lower-level event infrastructure. The audited import-path tests currently guard active coordination code against drifting back to direct `emit_event` imports, and the safe-event emitter tests verify the canonical compatibility path.

## Configuration

### Environment Variables

| Variable                            | Default | Description                            |
| ----------------------------------- | ------- | -------------------------------------- |
| `RINGRIFT_DAEMON_HEALTH_INTERVAL`   | 60      | Daemon health check interval (seconds) |
| `RINGRIFT_DATA_SYNC_INTERVAL`       | 120     | Games sync interval (seconds)          |
| `RINGRIFT_FAST_SYNC_INTERVAL`       | 30      | Fast sync interval (seconds)           |
| `RINGRIFT_MIN_SYNC_INTERVAL`        | 2.0     | Minimum auto-sync interval             |
| `RINGRIFT_AUTO_SYNC_MAX_CONCURRENT` | 6       | Max concurrent auto-sync transfers     |
| `RINGRIFT_SYNC_TIMEOUT`             | 300     | Sync timeout (seconds)                 |

### YAML Configuration

```yaml
# config/unified_loop.yaml
daemons:
  auto_sync:
    enabled: true
    interval: 60
  ephemeral_sync: # AutoSyncDaemon(strategy="ephemeral")
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
- `app/coordination/daemon_registry.py` - Runner specs, deprecation metadata, health contracts
- `app/coordination/event_emission_helpers.py` - Canonical safe event emission helper
- `app/coordination/safe_event_emitter.py` - Compatibility mixin delegating to the helper
- `scripts/master_loop.py` - Unified automation entry point
- `scripts/launch_daemons.py` - Manual daemon launcher

## Verification Contracts

The current daemon-system cleanup is covered by focused unit tests plus the full unit/contracts gate:

- `tests/unit/coordination/test_daemon_registry.py` and `tests/unit/coordination/test_health_check_compliance.py` validate registry/profile consistency and daemon health contracts.
- `tests/unit/coordination/test_event_subscription_completeness.py` validates dead-event diagnostics, import-path audit coverage, and the lean profile subscription matrix.
- `tests/unit/coordination/test_infrastructure_quality_contracts.py` validates master-loop profile names, active registry resolution, and unused-import cleanliness for the active infrastructure tests.
- `tests/unit/coordination/test_handler_base.py` covers the tightened `safe_subscribe()` and task helper behavior.

Phase verification command:

```bash
PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```
