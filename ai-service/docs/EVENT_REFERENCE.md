# Event System Reference

**Date:** December 2025
**Status:** Production

This document summarizes the most common events in the RingRift AI training coordination system. For the canonical list, see `EVENT_CATALOG.md`. `EVENT_REFERENCE_AUTO.md` is a generated scan of event strings across code/tests and includes non-canonical or test-only events.

---

## Overview

The event system consists of three integrated buses:

| Bus                      | Type            | Scope          | Use Case                  |
| ------------------------ | --------------- | -------------- | ------------------------- |
| `EventBus`               | In-memory async | Single process | Fast event delivery       |
| `StageEventBus`          | In-memory       | Single process | Pipeline stage completion |
| `CrossProcessEventQueue` | SQLite-backed   | Multi-process  | Daemon coordination       |

All buses are accessible through the unified `event_router.py`:

```python
from app.coordination.event_router import (
    get_router, publish, subscribe, DataEventType
)

# Publish to all buses automatically
await publish(DataEventType.TRAINING_COMPLETED, {"config": "hex8_2p"})

# Subscribe to events
router = get_router()
router.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)
```

---

## Event Categories

### Data Collection Events

| Event                 | Value            | Description               | Payload                         |
| --------------------- | ---------------- | ------------------------- | ------------------------------- |
| `NEW_GAMES_AVAILABLE` | `new_games`      | Fresh game data ready     | `{host, new_games, board_type}` |
| `DATA_SYNC_STARTED`   | `sync_started`   | Sync operation begins     | `{targets, data_type}`          |
| `DATA_SYNC_COMPLETED` | `sync_completed` | Sync succeeded            | `{duration, file_count}`        |
| `DATA_SYNC_FAILED`    | `sync_failed`    | Sync failed               | `{error, target_node}`          |
| `GAME_SYNCED`         | `game_synced`    | Individual game(s) synced | `{game_ids, target}`            |

### Training Events

| Event                        | Value                   | Description              | Payload                             |
| ---------------------------- | ----------------------- | ------------------------ | ----------------------------------- |
| `TRAINING_THRESHOLD_REACHED` | `training_threshold`    | Enough games to train    | `{config_key, game_count}`          |
| `TRAINING_STARTED`           | `training_started`      | Training job begins      | `{config_key, node_id, job_id}`     |
| `TRAINING_PROGRESS`          | `training_progress`     | Epoch progress update    | `{epoch, loss, accuracy}`           |
| `TRAINING_COMPLETED`         | `training_completed`    | Training finished        | `{config_key, model_path, metrics}` |
| `TRAINING_FAILED`            | `training_failed`       | Training job failed      | `{error, stack_trace}`              |
| `TRAINING_LOSS_ANOMALY`      | `training_loss_anomaly` | Loss spike/drop detected | `{epoch, loss, threshold}`          |
| `TRAINING_LOSS_TREND`        | `training_loss_trend`   | Loss trend changed       | `{direction, magnitude}`            |

### Evaluation Events

| Event                  | Value                  | Description        | Payload                      |
| ---------------------- | ---------------------- | ------------------ | ---------------------------- |
| `EVALUATION_STARTED`   | `evaluation_started`   | Gauntlet begins    | `{model_path, opponents}`    |
| `EVALUATION_PROGRESS`  | `evaluation_progress`  | Match progress     | `{completed, total, wins}`   |
| `EVALUATION_COMPLETED` | `evaluation_completed` | Gauntlet finished  | `{win_rate, elo_estimate}`   |
| `EVALUATION_FAILED`    | `evaluation_failed`    | Evaluation failed  | `{error}`                    |
| `ELO_UPDATED`          | `elo_updated`          | Elo rating changed | `{old_elo, new_elo, config}` |

### Promotion Events

| Event                 | Value                 | Description              | Payload                  |
| --------------------- | --------------------- | ------------------------ | ------------------------ |
| `PROMOTION_CANDIDATE` | `promotion_candidate` | Model qualifies          | `{model_path, win_rate}` |
| `PROMOTION_STARTED`   | `promotion_started`   | Promotion process begins | `{model_path}`           |
| `MODEL_PROMOTED`      | `model_promoted`      | Model is new canonical   | `{old_model, new_model}` |
| `PROMOTION_FAILED`    | `promotion_failed`    | Promotion failed         | `{reason}`               |
| `PROMOTION_COMPLETED` | `promotion_completed` | Promotion finalized      | `{config_key, success}`  |
| `PROMOTION_REJECTED`  | `promotion_rejected`  | Model didn't qualify     | `{threshold, actual}`    |
| `MODEL_UPDATED`       | `model_updated`       | Model metadata changed   | `{model_path, changes}`  |

### Curriculum Events

| Event                    | Value                    | Description              | Payload                      |
| ------------------------ | ------------------------ | ------------------------ | ---------------------------- |
| `CURRICULUM_REBALANCED`  | `curriculum_rebalanced`  | Weights redistributed    | `{old_weights, new_weights}` |
| `CURRICULUM_ADVANCED`    | `curriculum_advanced`    | Moved to harder tier     | `{old_tier, new_tier}`       |
| `WEIGHT_UPDATED`         | `weight_updated`         | Single weight changed    | `{config, old, new}`         |
| `ELO_SIGNIFICANT_CHANGE` | `elo_significant_change` | Elo changed >100         | `{delta, config}`            |
| `ELO_VELOCITY_CHANGED`   | `elo_velocity_changed`   | Improvement rate changed | `{velocity, config}`         |

### Selfplay Events

| Event                          | Value                          | Description                  | Payload                            |
| ------------------------------ | ------------------------------ | ---------------------------- | ---------------------------------- |
| `SELFPLAY_COMPLETE`            | `selfplay_complete`            | Batch finished               | `{game_count, duration}`           |
| `SELFPLAY_TARGET_UPDATED`      | `selfplay_target_updated`      | Target changed               | `{old_target, new_target}`         |
| `SELFPLAY_RATE_CHANGED`        | `selfplay_rate_changed`        | Rate multiplier >20%         | `{rate, reason}`                   |
| `SELFPLAY_ALLOCATION_UPDATED`  | `selfplay_allocation_updated`  | Allocation changed           | `{config, allocation}`             |
| `ARCHITECTURE_WEIGHTS_UPDATED` | `architecture_weights_updated` | Architecture weights updated | `{config_key, weights, timestamp}` |

### Quality Events

| Event                       | Value                       | Description        | Payload                  |
| --------------------------- | --------------------------- | ------------------ | ------------------------ |
| `QUALITY_CHECK_REQUESTED`   | `quality_check_requested`   | On-demand check    | `{config, requester}`    |
| `QUALITY_SCORE_UPDATED`     | `quality_score_updated`     | Score recalculated | `{old_score, new_score}` |
| `QUALITY_DEGRADED`          | `quality_degraded`          | Below threshold    | `{score, threshold}`     |
| `QUALITY_FEEDBACK_ADJUSTED` | `quality_feedback_adjusted` | Feedback updated   | `{config, adjustment}`   |
| `DATA_QUALITY_ALERT`        | `data_quality_alert`        | Quality issue      | `{severity, message}`    |
| `QUALITY_PENALTY_APPLIED`   | `quality_penalty_applied`   | Rate reduced       | `{config, penalty}`      |

### Regression Events

| Event                 | Value                 | Description          | Payload               |
| --------------------- | --------------------- | -------------------- | --------------------- |
| `REGRESSION_DETECTED` | `regression_detected` | Any regression       | `{severity, metrics}` |
| `REGRESSION_MINOR`    | `regression_minor`    | Minor regression     | `{delta, threshold}`  |
| `REGRESSION_MODERATE` | `regression_moderate` | Moderate regression  | `{delta, threshold}`  |
| `REGRESSION_SEVERE`   | `regression_severe`   | Severe regression    | `{delta, threshold}`  |
| `REGRESSION_CRITICAL` | `regression_critical` | Rollback recommended | `{delta, action}`     |
| `REGRESSION_CLEARED`  | `regression_cleared`  | Recovered            | `{recovery_time}`     |

### Cluster Health Events

| Event                   | Value                   | Description         | Payload                |
| ----------------------- | ----------------------- | ------------------- | ---------------------- |
| `HOST_ONLINE`           | `host_online`           | Node came online    | `{host_id, ip}`        |
| `HOST_OFFLINE`          | `host_offline`          | Node went offline   | `{host_id, last_seen}` |
| `NODE_UNHEALTHY`        | `node_unhealthy`        | Health check failed | `{host_id, reason}`    |
| `NODE_RECOVERED`        | `node_recovered`        | Node recovered      | `{host_id, downtime}`  |
| `NODE_OVERLOADED`       | `node_overloaded`       | Resource overload   | `{host_id, metrics}`   |
| `P2P_CLUSTER_HEALTHY`   | `p2p_cluster_healthy`   | Cluster healthy     | `{alive_nodes}`        |
| `P2P_CLUSTER_UNHEALTHY` | `p2p_cluster_unhealthy` | Cluster degraded    | `{dead_nodes}`         |

### Resource Events

| Event                          | Value                          | Description        | Payload                                      |
| ------------------------------ | ------------------------------ | ------------------ | -------------------------------------------- |
| `CLUSTER_CAPACITY_CHANGED`     | `cluster_capacity_changed`     | Total capacity     | `{old, new, reason}`                         |
| `BACKPRESSURE_ACTIVATED`       | `backpressure_activated`       | Throttling started | `{queue_depth}`                              |
| `BACKPRESSURE_RELEASED`        | `backpressure_released`        | Throttling stopped | `{queue_depth}`                              |
| `IDLE_RESOURCE_DETECTED`       | `idle_resource_detected`       | Idle GPU/CPU       | `{host_id, idle_time}`                       |
| `DISK_SPACE_LOW`               | `disk_space_low`               | Disk usage high    | `{host_id, usage_pct}`                       |
| `RESOURCE_CONSTRAINT`          | `resource_constraint`          | Resource pressure  | `{source, resource_type, ram_utilization}`   |
| `MEMORY_PRESSURE`              | `memory_pressure`              | VRAM/RAM pressure  | `{source, gpu_utilization, ram_utilization}` |
| `RESOURCE_CONSTRAINT_DETECTED` | `resource_constraint_detected` | Resource limit hit | `{node_id, resource_type, threshold}`        |

### Leader Election Events

| Event             | Value             | Description       | Payload             |
| ----------------- | ----------------- | ----------------- | ------------------- |
| `LEADER_ELECTED`  | `leader_elected`  | New leader        | `{leader_id, term}` |
| `LEADER_LOST`     | `leader_lost`     | Leader failed     | `{old_leader}`      |
| `LEADER_STEPDOWN` | `leader_stepdown` | Graceful stepdown | `{leader_id}`       |

### Work Queue Events

| Event            | Value            | Description    | Payload              |
| ---------------- | ---------------- | -------------- | -------------------- |
| `WORK_QUEUED`    | `work_queued`    | Work added     | `{work_id, type}`    |
| `WORK_CLAIMED`   | `work_claimed`   | Work claimed   | `{work_id, node_id}` |
| `WORK_COMPLETED` | `work_completed` | Work finished  | `{work_id, result}`  |
| `WORK_FAILED`    | `work_failed`    | Work failed    | `{work_id, error}`   |
| `WORK_TIMEOUT`   | `work_timeout`   | Work timed out | `{work_id, elapsed}` |

### Daemon Lifecycle Events

| Event                   | Value                   | Description       | Payload              |
| ----------------------- | ----------------------- | ----------------- | -------------------- |
| `DAEMON_STARTED`        | `daemon_started`        | Daemon started    | `{daemon_type}`      |
| `DAEMON_STOPPED`        | `daemon_stopped`        | Daemon stopped    | `{daemon_type}`      |
| `DAEMON_STATUS_CHANGED` | `daemon_status_changed` | Status changed    | `{daemon, old, new}` |
| `HANDLER_TIMEOUT`       | `handler_timeout`       | Handler timed out | `{handler, event}`   |
| `HANDLER_FAILED`        | `handler_failed`        | Handler exception | `{handler, error}`   |

### Orphan Detection Events

| Event                     | Value                     | Description        | Payload                 |
| ------------------------- | ------------------------- | ------------------ | ----------------------- |
| `ORPHAN_GAMES_DETECTED`   | `orphan_games_detected`   | Unregistered DBs   | `{orphan_count, paths}` |
| `ORPHAN_GAMES_REGISTERED` | `orphan_games_registered` | Orphans registered | `{registered_count}`    |
| `DATABASE_CREATED`        | `database_created`        | New DB file        | `{db_path, node_id}`    |

---

## Event Flow Diagrams

### Training Pipeline Flow

```
SELFPLAY_COMPLETE
       ↓
NEW_GAMES_AVAILABLE
       ↓
DATA_SYNC_COMPLETED → TRAINING_THRESHOLD_REACHED
                              ↓
                      TRAINING_STARTED
                              ↓
                      TRAINING_COMPLETED
                              ↓
                      EVALUATION_STARTED
                              ↓
                      EVALUATION_COMPLETED
                              ↓
                      PROMOTION_CANDIDATE → MODEL_PROMOTED
                              ↓
                      CURRICULUM_REBALANCED
```

### Health Recovery Flow

```
NODE_UNHEALTHY
       ↓
RECOVERY_INITIATED
       ↓
  ┌────┴────┐
  ↓         ↓
RECOVERY_COMPLETED  RECOVERY_FAILED
       ↓                    ↓
NODE_RECOVERED     HOST_OFFLINE
```

### Backpressure Flow

```
CLUSTER_CAPACITY_CHANGED
       ↓
[queue depth > threshold]
       ↓
BACKPRESSURE_ACTIVATED
       ↓
SELFPLAY_TARGET_UPDATED (reduced)
       ↓
[queue depth < threshold]
       ↓
BACKPRESSURE_RELEASED
       ↓
SELFPLAY_TARGET_UPDATED (restored)
```

---

## Subscribing to Events

### In Coordination Modules

```python
from app.coordination.event_router import get_router, DataEventType

class MyDaemon:
    def __init__(self):
        self._subscribed = False

    async def start(self):
        router = get_router()
        router.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training)
        router.subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion)
        self._subscribed = True

    async def _on_training(self, event):
        config = event.payload.get("config_key")
        # Handle training completion
```

### In Daemon Manager

```python
from app.coordination.daemon_manager import DaemonManager

dm = DaemonManager()
dm.subscribe_to_events()  # Auto-subscribes to all relevant events
```

---

## Emitting Events

### Using Typed Emitters (Recommended)

```python
from app.distributed.data_events import (
    emit_training_completed,
    emit_model_promoted,
    emit_quality_degraded,
)

# Type-safe emission
emit_training_completed(
    config_key="hex8_2p",
    model_path="/models/hex8_2p_v42.pth",
    metrics={"loss": 0.123, "accuracy": 0.95}
)
```

### Using Generic Publisher

```python
from app.coordination.event_router import publish, DataEventType

await publish(
    DataEventType.CUSTOM_EVENT,
    payload={"key": "value"},
    source="my_daemon"
)
```

---

## Handler Index

Key event handlers by module:

| Module                           | Events Handled                              | Purpose               |
| -------------------------------- | ------------------------------------------- | --------------------- |
| `data_pipeline_orchestrator.py`  | SYNC*COMPLETED, TRAINING*\_, EVALUATION\_\_ | Pipeline coordination |
| `feedback_loop_controller.py`    | TRAINING_COMPLETED, EVALUATION_COMPLETED    | Training feedback     |
| `selfplay_scheduler.py`          | ELO*\*, CURRICULUM*\_, QUALITY\_\_          | Selfplay allocation   |
| `daemon_manager.py`              | DAEMON*\*, HANDLER*\*, REGRESSION_CRITICAL  | Daemon lifecycle      |
| `auto_sync_daemon.py`            | TRAINING_STARTED, NODE_RECOVERED            | Sync triggers         |
| `unified_distribution_daemon.py` | MODEL_PROMOTED, MODEL_UPDATED               | Model distribution    |
| `idle_resource_daemon.py`        | BACKPRESSURE*\*, IDLE_RESOURCE*\*           | GPU utilization       |

---

## Best Practices

### 1. Use Typed Event Types

```python
# Good - type-safe
router.subscribe(DataEventType.TRAINING_COMPLETED, handler)

# Avoid - string literals
router.subscribe("training_completed", handler)
```

### 2. Include Correlation IDs

```python
await publish(
    DataEventType.TRAINING_STARTED,
    payload={
        "correlation_id": str(uuid.uuid4()),
        "config_key": "hex8_2p",
    }
)
```

### 3. Handle Events Asynchronously

```python
async def _on_event(self, event):
    # Don't block the event loop
    asyncio.create_task(self._process_event(event))
```

### 4. Log Event Reception

```python
async def _on_training(self, event):
    logger.info(f"[{self.__class__.__name__}] Received TRAINING_COMPLETED: {event.payload}")
```

---

## Troubleshooting

### Event Not Received

1. Check subscription timing (subscribe before emitter starts)
2. Verify event type spelling
3. Check handler isn't raising exceptions

### Duplicate Events

The router includes content-based deduplication (SHA256). If you see duplicates:

1. Check if event is published from multiple sources
2. Verify cross-process queue is being polled

### Handler Timeout

Default timeout is 30 seconds. Increase for slow handlers:

```python
router.subscribe(DataEventType.LARGE_BATCH, handler, timeout=120)
```

---

## Files Reference

| File                      | Purpose                      |
| ------------------------- | ---------------------------- |
| `event_router.py`         | Unified router               |
| `data_events.py`          | DataEventType enum, EventBus |
| `stage_events.py`         | StageEventBus                |
| `cross_process_events.py` | SQLite queue                 |
| `event_emitters.py`       | Typed emitter functions      |
| `event_normalization.py`  | Event type normalization     |

---

_Last Updated: December 27, 2025_
