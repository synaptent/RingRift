# Coordination Architecture

> **NOTE (December 2025)**: This document describes the pre-consolidation architecture. The event
> systems described here (EventBus, DataEventBus, StageEventBus, CrossProcessEventQueue) have been
> unified into `app/coordination/event_router.py`. For current usage, import from event_router.py.
> See `ai-service/docs/CONSOLIDATION_STATUS_2025_12_19.md` for consolidation details.

## Overview

The RingRift AI service uses a multi-layered event-driven coordination system for
distributed training. This enables loose coupling between components while maintaining
strong consistency guarantees.

## Event Systems

### 1. EventBus (`app/core/event_bus.py`)

The core in-memory event bus providing:

- Topic-based pub/sub
- Async and sync handlers
- Event filtering and routing
- Event history and replay
- Type-safe events

```python
from app.core.event_bus import get_event_bus

bus = get_event_bus()

@bus.subscribe("training.completed")
async def on_training_completed(event):
    print(f"Training completed: {event.payload}")

await bus.publish(event)
```

### 2. DataEventBus (`app/distributed/data_events.py`)

Pipeline data events for training workflow:

- `GAME_BATCH_READY` - Selfplay games ready for training
- `TRAINING_COMPLETED` - Training iteration finished
- `MODEL_PROMOTED` - New model available
- `EVALUATION_RESULT` - Elo evaluation complete

### 3. StageEventBus (`app/distributed/stage_events.py`)

Pipeline stage completion signals for orchestration.

### 4. CrossProcessEventQueue (`app/distributed/cross_process_events.py`)

SQLite-backed queue for cross-process coordination:

- Persistent across restarts
- Multi-process safe
- Used for daemon-to-daemon communication

## Unified Event Router

The `EventRouter` (`app/coordination/event_router.py`) consolidates all event systems:

```python
from app.coordination.event_router import get_router, publish

# Publish to all systems automatically
await publish(
    DataEventType.TRAINING_COMPLETED,
    payload={"config": "square8_2p"},
    source="training_daemon"
)
```

## Component Communication

### Training Pipeline Flow

```
Selfplay Generator
       │
       ├── GAME_BATCH_READY ──────────────────┐
       │                                       │
       ▼                                       ▼
   Training Loop ◄──── EventBus ───► Evaluation Loop
       │                                       │
       ├── TRAINING_COMPLETED                  │
       │                                       │
       ▼                                       │
   Model Promoter ◄─── EVALUATION_RESULT ──────┘
       │
       ├── MODEL_PROMOTED
       ▼
   Model Registry
```

### Key Event Types

| Event                | Producer   | Consumers             |
| -------------------- | ---------- | --------------------- |
| `GAME_BATCH_READY`   | Selfplay   | Training              |
| `TRAINING_COMPLETED` | Training   | Evaluation, Dashboard |
| `EVALUATION_RESULT`  | Evaluation | Promoter, Dashboard   |
| `MODEL_PROMOTED`     | Promoter   | Selfplay, All         |
| `TRAINING_PROGRESS`  | Training   | Dashboard             |

## Distributed Coordination

### P2P Sync (`app/distributed/p2p_sync.py`)

Peer-to-peer synchronization for:

- Model weights sharing
- Game data distribution
- Health monitoring

### Unified Data Sync (`app/distributed/unified_data_sync.py`)

Coordinates data flow between nodes:

- Detects new training data
- Propagates to training nodes
- Manages data locality

## Configuration

### Environment Variables

- `RINGRIFT_COORDINATOR_URL` - Central coordinator endpoint
- `RINGRIFT_P2P_URL` - Peer-to-peer sync endpoint
- `RINGRIFT_CLUSTER_AUTH_TOKEN` - Cluster authentication

### Signal Computer

The `UnifiedSignalComputer` (`app/training/unified_signals.py`) aggregates signals:

- ELO trend analysis
- Training urgency calculation
- Regression detection

## Best Practices

1. **Use EventRouter for new code** - Don't use individual buses directly
2. **Include source in events** - Helps with debugging and auditing
3. **Handle missing dependencies** - Use `HAS_*` guards for optional imports
4. **Prefer async handlers** - Better throughput for I/O-bound operations
5. **Don't block in handlers** - Spawn tasks for long-running work

## Debugging

Enable event debugging:

```bash
export RINGRIFT_DEBUG=true
```

View event flow:

```python
router = get_router()
for event in router.get_recent_events(limit=100):
    print(f"{event.timestamp}: {event.type} from {event.source}")
```

---

## December 2025: Coordination Infrastructure Consolidation

### New Pipeline Actions (`app/coordination/pipeline_actions.py`)

Triggers actual work for each pipeline stage with circuit breaker protection:

```python
from app.coordination.pipeline_actions import (
    trigger_data_sync,
    trigger_npz_export,
    trigger_training,
    trigger_evaluation,
    trigger_promotion,
)

# Trigger data sync from cluster
result = await trigger_data_sync(
    board_type="hex8",
    num_players=2,
    iteration=1,
)

if result.success:
    print(f"Sync completed in {result.duration_seconds:.1f}s")
else:
    print(f"Sync failed: {result.error}")
```

### Circuit Breaker (`app/coordination/data_pipeline_orchestrator.py`)

Prevents cascading failures by opening after repeated errors:

```python
from app.coordination.data_pipeline_orchestrator import CircuitBreaker

cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=300)

if cb.can_execute():
    try:
        result = await trigger_training(...)
        cb.record_success("training")
    except Exception as e:
        cb.record_failure("training", str(e))
        # After 3 failures, circuit opens for 5 minutes
```

States:

- **CLOSED**: Normal operation
- **OPEN**: Rejecting requests (auto-recovers after timeout)
- **HALF_OPEN**: Testing recovery with limited requests

### Bandwidth Coordination (`app/coordination/sync_bandwidth.py`)

Prevents network contention during parallel syncs:

```python
from app.coordination.sync_bandwidth import (
    BandwidthCoordinatedRsync,
    TransferPriority,
    get_bandwidth_manager,
)

rsync = BandwidthCoordinatedRsync()
result = await rsync.sync(
    source="/data/games/",
    dest="ubuntu@gpu-node-1:/data/games/",
    host="gpu-node-1",
    priority=TransferPriority.HIGH,
)

# Check manager status
manager = get_bandwidth_manager()
print(manager.get_status())
```

Features:

- Per-host bandwidth allocation
- Priority-based scheduling (LOW, NORMAL, HIGH, CRITICAL)
- Concurrent transfer limits
- Automatic allocation expiry

### Daemon Adapters (`app/coordination/daemon_adapters.py`)

Unified lifecycle management for all daemons:

```python
from app.coordination.daemon_adapters import get_daemon_adapter
from app.coordination.daemon_manager import DaemonType

adapter = get_daemon_adapter(DaemonType.DISTILLATION)
await adapter.run()  # Handles role acquisition, health checks, restart
```

Available adapters:

- `DistillationDaemonAdapter` - Acquires `DISTILLATION_LEADER` role
- `PromotionDaemonAdapter` - Acquires `PROMOTION_LEADER` role
- `ExternalDriveSyncAdapter` - Acquires `EXTERNAL_SYNC_LEADER` role
- `VastCpuPipelineAdapter` - Acquires `VAST_PIPELINE_LEADER` role

### Master Daemon Launcher (`scripts/launch_daemons.py`)

Single command to launch all daemons:

```bash
# Launch all daemons
python scripts/launch_daemons.py --all

# Launch specific types
python scripts/launch_daemons.py --sync
python scripts/launch_daemons.py --training-only

# Check status
python scripts/launch_daemons.py --status
```

### Backwards Compatibility

The `event_router.py` provides aliases for `unified_event_coordinator.py`:

```python
# Old imports still work
from app.coordination.event_router import (
    UnifiedEventCoordinator,  # Alias for UnifiedEventRouter
    get_event_coordinator,    # Alias for get_router
    CoordinatorStats,         # Backwards-compat dataclass
    emit_training_started,    # Helper functions
    emit_training_completed,
)
```

### Archived Modules

The following modules have been superseded:

| Old Module                     | Replacement                  |
| ------------------------------ | ---------------------------- |
| `unified_event_coordinator.py` | `event_router.py`            |
| `cluster_monitor_daemon.py`    | `unified_cluster_monitor.py` |
| `robust_cluster_monitor.py`    | `unified_cluster_monitor.py` |

See `archive/deprecated_coordination/README.md` for migration guides.
