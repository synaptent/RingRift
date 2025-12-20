# Coordination Architecture

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
