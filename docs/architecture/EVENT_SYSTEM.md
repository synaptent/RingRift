# Event System Architecture

> Unified event routing for RingRift AI training coordination (December 2025)

## Overview

RingRift uses a unified event system to coordinate training pipeline components, daemons, and cluster operations. The system consolidates three legacy event buses into a single routing layer.

## Architecture

```
                          EventRouter (Unified Entry Point)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
   EventBus (Async)          StageEventBus           CrossProcessEventQueue
   ├── In-memory             ├── Pipeline stages     ├── SQLite-backed
   ├── Pub/sub model         ├── Stage transitions   ├── Cross-process IPC
   └── data_events.py        └── stage_events.py     └── cross_process_events.py
```

### Components

<<<<<<< Updated upstream
| Component | Location | Purpose | Persistence |
|-----------|----------|---------|-------------|
| **EventRouter** | `event_router.py` | Unified API, routes to all buses | In-memory |
| **EventBus** | `data_events.py` | Training/data events | In-memory |
| **StageEventBus** | `stage_events.py` | Pipeline stage completion | In-memory |
| **CrossProcessEventQueue** | `cross_process_events.py` | Multi-process coordination | SQLite |
=======
| Component                  | Location                  | Purpose                          | Persistence |
| -------------------------- | ------------------------- | -------------------------------- | ----------- |
| **EventRouter**            | `event_router.py`         | Unified API, routes to all buses | In-memory   |
| **EventBus**               | `data_events.py`          | Training/data events             | In-memory   |
| **StageEventBus**          | `stage_events.py`         | Pipeline stage completion        | In-memory   |
| **CrossProcessEventQueue** | `cross_process_events.py` | Multi-process coordination       | SQLite      |
>>>>>>> Stashed changes

## Event Types

Events are defined in `DataEventType` enum (252 total, ~100 actively used):

### Training Events
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
- `TRAINING_STARTED`, `TRAINING_COMPLETED`, `TRAINING_FAILED`
- `TRAINING_LOSS_TREND`, `TRAINING_LOSS_ANOMALY`
- `TRAINING_EARLY_STOPPED`, `TRAINING_STALLED`

### Model Events
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
- `MODEL_PROMOTED`, `MODEL_REGISTERED`, `MODEL_DISTRIBUTED`
- `PROMOTION_CANDIDATE`, `PROMOTION_FAILED`
- `EVALUATION_COMPLETED`, `GAUNTLET_COMPLETED`

### Data Events
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
- `DATA_SYNC_COMPLETED`, `DATA_SYNC_FAILED`
- `GAMES_GENERATED`, `NPZ_EXPORTED`
- `QUALITY_SCORE_UPDATED`, `QUALITY_DEGRADED`

### Cluster Events
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
- `HOST_ONLINE`, `HOST_OFFLINE`
- `LEADER_ELECTED`, `NODE_OVERLOADED`
- `CLUSTER_CAPACITY_CHANGED`

### Daemon Events
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
- `DAEMON_STARTED`, `DAEMON_STOPPED`, `DAEMON_STATUS_CHANGED`

## Usage

### Publishing Events

```python
from app.coordination.event_router import get_router, publish

# Method 1: Using publish() function (recommended)
await publish(
    DataEventType.TRAINING_COMPLETED,
    payload={"config_key": "hex8_2p", "success": True, "epoch": 50},
    source="training_daemon"
)

# Method 2: Using router directly
router = get_router()
await router.emit(
    DataEventType.MODEL_PROMOTED,
    payload={"model_path": "models/hex8_2p.pth", "elo": 1850},
    source="promotion_controller"
)
```

### Subscribing to Events

```python
from app.coordination.event_router import get_router
from app.distributed.data_events import DataEventType

router = get_router()

# Subscribe to single event type
async def on_training_completed(event: dict) -> None:
    config = event.get("payload", {}).get("config_key")
    logger.info(f"Training completed for {config}")

router.subscribe(DataEventType.TRAINING_COMPLETED, on_training_completed)

# Subscribe to multiple event types
router.subscribe_many([
    DataEventType.MODEL_PROMOTED,
    DataEventType.EVALUATION_COMPLETED,
], on_model_event)
```

### Using Emit Functions (Convenience)

```python
from app.distributed.data_events import (
    emit_training_completed,
    emit_model_promoted,
    emit_quality_degraded,
)

# These handle payload construction
emit_training_completed(config_key="hex8_2p", success=True)
emit_model_promoted(model_path="models/hex8_2p.pth", elo=1850)
emit_quality_degraded(config_key="hex8_2p", quality_drop_percent=25)
```

## Event Flow

### Training Pipeline Flow

```
SELFPLAY_COMPLETED
       │
       ▼
GAMES_GENERATED ──────► NPZ_EXPORTED
       │                     │
       ▼                     ▼
DATA_SYNC_COMPLETED    TRAINING_STARTED
                             │
                             ▼
                      TRAINING_COMPLETED
                             │
                       ┌─────┴─────┐
                       ▼           ▼
              EVALUATION_STARTED  TRAINING_FAILED
                       │
                       ▼
              EVALUATION_COMPLETED
                       │
               ┌───────┴───────┐
               ▼               ▼
       MODEL_PROMOTED    PROMOTION_FAILED
               │
               ▼
       MODEL_DISTRIBUTED
```

### Feedback Loops (13 loops documented)

<<<<<<< Updated upstream
| Loop | Trigger | Handler | Action |
|------|---------|---------|--------|
| 1 | `MODEL_PROMOTED` | SelfplayScheduler | Adjust curriculum weights |
| 2 | `TRAINING_LOSS_TREND` | FeedbackLoopController | Adjust exploration boost |
| 3 | `QUALITY_DEGRADED` | QualityMonitorDaemon | Throttle data generation |
| 4 | `ELO_VELOCITY_CHANGED` | CurriculumFeedback | Rebalance config priorities |
| 5 | `EVALUATION_COMPLETED` | TrainingCoordinator | Adjust hyperparameters |
| 6 | `REGRESSION_DETECTED` | RollbackManager | Auto-rollback to checkpoint |
| ... | ... | ... | ... |
=======
| Loop | Trigger                | Handler                | Action                      |
| ---- | ---------------------- | ---------------------- | --------------------------- |
| 1    | `MODEL_PROMOTED`       | SelfplayScheduler      | Adjust curriculum weights   |
| 2    | `TRAINING_LOSS_TREND`  | FeedbackLoopController | Adjust exploration boost    |
| 3    | `QUALITY_DEGRADED`     | QualityMonitorDaemon   | Throttle data generation    |
| 4    | `ELO_VELOCITY_CHANGED` | CurriculumFeedback     | Rebalance config priorities |
| 5    | `EVALUATION_COMPLETED` | TrainingCoordinator    | Adjust hyperparameters      |
| 6    | `REGRESSION_DETECTED`  | RollbackManager        | Auto-rollback to checkpoint |
| ...  | ...                    | ...                    | ...                         |
>>>>>>> Stashed changes

## Cross-Process Events

For multi-process coordination (e.g., between P2P orchestrator and training):

```python
from app.coordination.cross_process_events import (
    CrossProcessEventQueue,
    push_event,
    poll_events,
)

# Push from one process
push_event(
    event_type="TRAINING_COMPLETED",
    payload={"config_key": "hex8_2p"},
    source_process="training_worker_1"
)

# Poll from another process
queue = CrossProcessEventQueue()
events = queue.poll(since_id=last_seen_id)
for event in events:
    process_event(event)
```

## Event Deduplication

The router uses SHA256-based content deduplication to prevent duplicate event processing:

```python
# Same payload + event type = deduplicated (within 60s window)
await publish(DataEventType.MODEL_PROMOTED, {"model": "foo"})
await publish(DataEventType.MODEL_PROMOTED, {"model": "foo"})  # Deduplicated
await publish(DataEventType.MODEL_PROMOTED, {"model": "bar"})  # Different payload - emitted
```

## Integration with Daemons

### DaemonManager Event Wiring

Daemons subscribe to events during initialization:

```python
class MyDaemon:
    def __init__(self):
        router = get_router()
        router.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_completed)
        router.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)

    async def _on_training_completed(self, event: dict) -> None:
        # Handle event
        pass
```

### Common Daemon Subscriptions

<<<<<<< Updated upstream
| Daemon | Events Subscribed |
|--------|-------------------|
| `FeedbackLoopController` | `TRAINING_LOSS_TREND`, `TRAINING_LOSS_ANOMALY`, `QUALITY_DEGRADED` |
| `SelfplayScheduler` | `MODEL_PROMOTED`, `ELO_VELOCITY_CHANGED`, `CURRICULUM_ADVANCED` |
| `AutoSyncDaemon` | `GAMES_GENERATED`, `NPZ_EXPORTED`, `DATA_SYNC_FAILED` |
| `ModelDistributionDaemon` | `MODEL_PROMOTED`, `MODEL_REGISTERED` |
| `OrphanDetectionDaemon` | `DATABASE_CREATED`, `SYNC_COMPLETED` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RINGRIFT_EVENT_LOG_LEVEL` | `INFO` | Event logging verbosity |
| `RINGRIFT_EVENT_DEDUP_WINDOW_SECS` | `60` | Deduplication time window |
| `COORDINATOR_CROSS_PROCESS_POLL_INTERVAL` | `5` | Cross-process poll frequency |
=======
| Daemon                    | Events Subscribed                                                  |
| ------------------------- | ------------------------------------------------------------------ |
| `FeedbackLoopController`  | `TRAINING_LOSS_TREND`, `TRAINING_LOSS_ANOMALY`, `QUALITY_DEGRADED` |
| `SelfplayScheduler`       | `MODEL_PROMOTED`, `ELO_VELOCITY_CHANGED`, `CURRICULUM_ADVANCED`    |
| `AutoSyncDaemon`          | `GAMES_GENERATED`, `NPZ_EXPORTED`, `DATA_SYNC_FAILED`              |
| `ModelDistributionDaemon` | `MODEL_PROMOTED`, `MODEL_REGISTERED`                               |
| `OrphanDetectionDaemon`   | `DATABASE_CREATED`, `SYNC_COMPLETED`                               |

## Environment Variables

| Variable                                  | Default | Description                  |
| ----------------------------------------- | ------- | ---------------------------- |
| `RINGRIFT_EVENT_LOG_LEVEL`                | `INFO`  | Event logging verbosity      |
| `RINGRIFT_EVENT_DEDUP_WINDOW_SECS`        | `60`    | Deduplication time window    |
| `COORDINATOR_CROSS_PROCESS_POLL_INTERVAL` | `5`     | Cross-process poll frequency |
>>>>>>> Stashed changes

## Best Practices

1. **Use publish() for new code** - Unified routing ensures all systems receive events
2. **Include source in payload** - Helps with debugging and event tracing
3. **Handle exceptions in handlers** - Router catches but logs errors; handler should be resilient
4. **Prefer async handlers** - The router supports both sync and async, but async is preferred
5. **Don't emit in handlers** - Avoid event cascades; queue secondary events instead

## Migration from Legacy Systems

The legacy `DataEventBus`, `StageEventBus`, and direct `emit_*` calls still work but route through `EventRouter` internally. New code should use:

```python
# OLD (still works, deprecated)
from app.distributed.data_events import EventBus
bus = EventBus()
bus.emit(DataEventType.TRAINING_COMPLETED, payload)

# NEW (recommended)
from app.coordination.event_router import publish
await publish(DataEventType.TRAINING_COMPLETED, payload, source="my_module")
```

## Files

<<<<<<< Updated upstream
| File | Purpose |
|------|---------|
| `app/coordination/event_router.py` | Unified router (main entry point) |
| `app/distributed/data_events.py` | DataEventType enum, EventBus, emit functions |
| `app/coordination/stage_events.py` | Pipeline stage events |
| `app/coordination/cross_process_events.py` | SQLite-backed cross-process queue |
| `app/coordination/event_normalization.py` | Event type normalization utilities |
| `app/coordination/event_emitters.py` | Convenience emit functions |
=======
| File                                       | Purpose                                      |
| ------------------------------------------ | -------------------------------------------- |
| `app/coordination/event_router.py`         | Unified router (main entry point)            |
| `app/distributed/data_events.py`           | DataEventType enum, EventBus, emit functions |
| `app/coordination/stage_events.py`         | Pipeline stage events                        |
| `app/coordination/cross_process_events.py` | SQLite-backed cross-process queue            |
| `app/coordination/event_normalization.py`  | Event type normalization utilities           |
| `app/coordination/event_emitters.py`       | Convenience emit functions                   |
>>>>>>> Stashed changes

## See Also

- [COORDINATION_SYSTEM.md](../../ai-service/docs/architecture/COORDINATION_SYSTEM.md) - Full coordination architecture
- [DAEMON_REGISTRY.md](../../ai-service/docs/DAEMON_REGISTRY.md) - Daemon event subscriptions
- [FEEDBACK_LOOPS.md](./FEEDBACK_LOOPS.md) - Training feedback loop details
