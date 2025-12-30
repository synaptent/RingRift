# Coordination Event System Runbook

**Last Updated**: December 29, 2025
**Version**: Wave 7

## Overview

The RingRift coordination event system provides an event-driven architecture for pipeline orchestration. It uses a unified event bus with content-based deduplication and dead letter queue (DLQ) handling.

## Event Types

202 event types are defined in `app/distributed/data_events.py`. Key categories:

### Training Pipeline Events

| Event                  | Emitter             | Subscribers                                | Purpose              |
| ---------------------- | ------------------- | ------------------------------------------ | -------------------- |
| `TRAINING_STARTED`     | TrainingCoordinator | SyncRouter, IdleShutdown                   | Pause idle detection |
| `TRAINING_COMPLETED`   | TrainingCoordinator | FeedbackLoop, DataPipeline                 | Trigger evaluation   |
| `TRAINING_FAILED`      | TrainingCoordinator | AlertManager                               | Handle failures      |
| `EVALUATION_COMPLETED` | GameGauntlet        | CurriculumIntegration, AutoPromotionDaemon | Update curriculum    |
| `MODEL_PROMOTED`       | AutoPromotionDaemon | UnifiedDistributionDaemon                  | Distribute model     |

### Data Flow Events

| Event                   | Emitter                  | Subscribers              | Purpose         |
| ----------------------- | ------------------------ | ------------------------ | --------------- |
| `DATA_SYNC_COMPLETED`   | AutoSyncDaemon           | DataPipelineOrchestrator | Trigger export  |
| `NEW_GAMES_AVAILABLE`   | DataPipelineOrchestrator | SelfplayScheduler        | Signal new data |
| `ORPHAN_GAMES_DETECTED` | OrphanDetectionDaemon    | DataPipelineOrchestrator | Recovery sync   |

### Health Events

| Event                 | Emitter                  | Subscribers        | Purpose           |
| --------------------- | ------------------------ | ------------------ | ----------------- |
| `NODE_UNHEALTHY`      | HealthCheckOrchestrator  | NodeRecoveryDaemon | Trigger recovery  |
| `NODE_RECOVERED`      | NodeRecoveryDaemon       | SelfplayScheduler  | Resume scheduling |
| `REGRESSION_DETECTED` | ModelPerformanceWatchdog | DataPipeline       | Handle regression |

## Event Bus Architecture

```
Emitters
    |
    v
UnifiedEventRouter (deduplication via SHA256)
    |
    +-> In-memory subscribers (same process)
    |
    +-> Stage event bus (pipeline stages)
    |
    +-> Cross-process queue (SQLite-backed)
    |
    v
Subscribers (handlers)
```

### Key Components

| Component            | Location          | Purpose                 |
| -------------------- | ----------------- | ----------------------- |
| `UnifiedEventRouter` | `event_router.py` | Central event dispatch  |
| `DataEventType`      | `data_events.py`  | Event type enum         |
| `DataEvent`          | `data_events.py`  | Event data class        |
| `EventBus`           | `event_router.py` | Subscription management |

## Subscription Wiring

Event wiring is delegated through `app/coordination/event_subscription_registry.py` and executed during `coordination_bootstrap` initialization.

### Check Subscription Status

```bash
curl -s http://localhost:8770/status | jq '.event_subscription_status'
```

### Critical Subscriptions

These must succeed for the pipeline to function:

```python
# manager_events group includes:
- TRAINING_STARTED
- TRAINING_COMPLETED
- DATA_SYNC_STARTED
- DATA_SYNC_COMPLETED
- TASK_SPAWNED
- TASK_COMPLETED
- TASK_FAILED
```

### Subscription with Retry

The P2P orchestrator uses retry with exponential backoff:

```python
async def _subscribe_with_retry(
    self,
    event_name: str,
    handler: Any,
    max_attempts: int = 3,
    is_critical: bool = False,
) -> bool:
    for attempt in range(max_attempts):
        try:
            subscribe(event_name, handler)
            return True
        except Exception as e:
            delay = 2 ** attempt  # 1s, 2s, 4s
            await asyncio.sleep(delay)
    return False
```

## Dead Letter Queue (DLQ)

Failed events are stored in DLQ for later processing:

### DLQ Storage

```sql
-- app/coordination/dead_letter_queue.py
CREATE TABLE dead_letter_queue (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_data JSON NOT NULL,
    error TEXT NOT NULL,
    created_at REAL NOT NULL,
    retry_count INTEGER DEFAULT 0
);
```

### DLQ Operations

```python
from app.coordination.dead_letter_queue import get_dead_letter_queue

dlq = get_dead_letter_queue()

# Get failed events
failed = dlq.get_pending_events(limit=100)

# Retry failed event
dlq.retry_event(event_id)

# Clear old events
dlq.cleanup(older_than_hours=24)
```

### DLQ Dashboard (CLI)

For operational visibility and retries, use the dashboard script:

```bash
# Summary stats + health
python scripts/dlq_dashboard.py

# List pending events
python scripts/dlq_dashboard.py --pending --limit 20

# Retry pending events
python scripts/dlq_dashboard.py --retry --max-events 10

# Watch mode (refresh every 30s)
python scripts/dlq_dashboard.py --watch --interval 30
```

## Event Deduplication

Events are deduplicated using SHA256 content hash:

```python
# Hash includes: event_type + json(payload)
hash_key = hashlib.sha256(
    f"{event.event_type}:{json.dumps(event.payload, sort_keys=True)}".encode()
).hexdigest()

# Dedup window: 60 seconds
if hash_key in recent_events and (now - recent_events[hash_key]) < 60:
    return  # Skip duplicate
```

## Subscribing to Events

```python
from app.coordination.event_router import subscribe, DataEventType

def my_handler(event):
    payload = event.payload if hasattr(event, "payload") else event
    config_key = payload.get("config_key", "unknown")
    # Handle event...

# Subscribe by string name
subscribe("TRAINING_COMPLETED", my_handler)

# Or by enum
subscribe(DataEventType.TRAINING_COMPLETED.value, my_handler)
```

## Emitting Events

### Using Typed Emitters (Preferred)

```python
from app.coordination.event_emitters import (
    emit_training_complete,
    emit_data_sync_completed,
    emit_model_promoted,
)

emit_training_complete(
    config_key="hex8_2p",
    model_path="models/canonical_hex8_2p.pth",
    epochs_completed=50,
)
```

### Using EventBus Directly

```python
from app.distributed.data_events import DataEventType, DataEvent
from app.coordination.event_router import get_event_bus

bus = get_event_bus()
event = DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"config_key": "hex8_2p", "model_path": "..."},
    source="TrainingCoordinator",
)
bus.publish(event)
```

## Cross-Process Events

For events that need to cross process boundaries (e.g., P2P to daemons):

```python
from app.coordination.cross_process_events import (
    get_cross_process_queue,
    publish_cross_process,
)

# Publish
publish_cross_process("TRAINING_COMPLETED", {"config_key": "hex8_2p"})

# Consume in another process
queue = get_cross_process_queue()
events = queue.get_pending_events()
```

## Troubleshooting

### 1. Events Not Being Delivered

**Symptoms**: Handler never called, no logs

**Diagnosis**:

```python
from app.coordination.event_router import get_event_bus

bus = get_event_bus()
print(f"Subscribers: {bus.list_subscribers()}")
```

**Resolution**:

1. Verify subscription: Check event name matches exactly
2. Check handler exceptions: Wrap in try/except with logging
3. Verify EVENT_ROUTER daemon is running

### 2. Duplicate Events

**Symptoms**: Handler called multiple times for same event

**Diagnosis**:
Check dedup window:

```python
from app.coordination.event_router import get_router
router = get_router()
print(f"Recent events: {len(router._recent_events)}")
```

**Resolution**:

1. Ensure events have consistent payload serialization
2. Increase dedup window if needed

### 3. Handler Timeout

**Symptoms**: HANDLER_TIMEOUT events, slow processing

**Diagnosis**:

```bash
curl -s http://localhost:8770/status | jq '.handler_timeouts'
```

**Resolution**:

1. Make handlers async if doing I/O
2. Offload heavy work to background tasks
3. Increase handler timeout if necessary

### 4. DLQ Growing

**Symptoms**: Many events in dead letter queue

**Diagnosis**:

```python
dlq = get_dead_letter_queue()
print(f"Pending events: {dlq.get_pending_count()}")
print(f"Errors: {dlq.get_error_summary()}")
```

**Resolution**:

1. Fix underlying handler errors
2. Retry failed events: `dlq.retry_all()`
3. Clean old events: `dlq.cleanup(older_than_hours=24)`

## Event Wiring Verification

Run this script to find orphan events (emitted but not subscribed):

```python
from app.coordination.event_router import get_event_bus
from app.distributed.data_events import DataEventType

bus = get_event_bus()
subscribers = bus.list_subscribers()

for event_type in DataEventType:
    if event_type.value not in subscribers:
        print(f"WARNING: {event_type.value} has no subscribers")
```

## Environment Variables

| Variable                           | Default | Description                              |
| ---------------------------------- | ------- | ---------------------------------------- |
| `RINGRIFT_EVENT_HANDLER_TIMEOUT`   | 600     | Event handler timeout (seconds)          |
| `RINGRIFT_EVENT_VALIDATION_STRICT` | false   | Reject unknown events                    |
| `RINGRIFT_LOG_LEVEL`               | INFO    | Log verbosity (use DEBUG for event logs) |

DLQ retention is managed via `DeadLetterQueue.cleanup(older_than_hours=...)`.

## See Also

- [P2P_ORCHESTRATOR_OPERATIONS.md](P2P_ORCHESTRATOR_OPERATIONS.md) - P2P cluster
- [DAEMON_MANAGER_OPERATIONS.md](DAEMON_MANAGER_OPERATIONS.md) - Daemon lifecycle
- `ai-service/CLAUDE.md` - Full event type reference
