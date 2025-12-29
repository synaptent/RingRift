# Event Wiring Guide

This document describes the event system used for coordination between daemons, training pipelines, and cluster operations in the RingRift AI Service.

## Overview

The event system consists of three layers, unified through `event_router.py`:

1. **EventBus** (`data_events.py`) - In-memory async event bus
2. **StageEventBus** (`stage_events.py`) - Pipeline stage completion events
3. **CrossProcessEventQueue** (`cross_process_events.py`) - SQLite-backed cross-process queue

## Quick Start

```python
from app.coordination.event_router import (
    get_router,
    publish,
    subscribe,
    DataEventType,
)

# Subscribe to events
router = get_router()
router.subscribe(DataEventType.TRAINING_COMPLETED, my_handler)

# Publish events
await publish(
    DataEventType.TRAINING_COMPLETED,
    payload={"config": "hex8_2p", "model_path": "models/hex8_2p.pth"},
    source="training_daemon"
)

# Handler signature
async def my_handler(event: dict) -> None:
    config = event.get("config")
    # Process event...
```

## Core Event Types

These tables are representative. For the full event list, see `EVENT_CATALOG.md`.

### Training Pipeline Events

| Event Type               | Emitter               | Subscribers                                         | Purpose                               |
| ------------------------ | --------------------- | --------------------------------------------------- | ------------------------------------- |
| `TRAINING_STARTED`       | `TrainingCoordinator` | `SyncRouter`, `IdleShutdown`                        | Pause idle detection, prioritize sync |
| `TRAINING_COMPLETED`     | `TrainingCoordinator` | `FeedbackLoop`, `DataPipeline`, `ModelDistribution` | Trigger evaluation, sync              |
| `TRAINING_FAILED`        | `TrainingCoordinator` | `DaemonManager`, `RecoveryOrchestrator`             | Handle failure, retry                 |
| `TRAINING_EARLY_STOPPED` | `train.py`            | `FeedbackLoop`                                      | Track early stopping                  |

### Evaluation Events

| Event Type             | Emitter            | Subscribers                                              | Purpose                                |
| ---------------------- | ------------------ | -------------------------------------------------------- | -------------------------------------- |
| `EVALUATION_COMPLETED` | `GameGauntlet`     | `FeedbackLoop`, `CurriculumIntegration`, `AutoPromotion` | Update curriculum, promotion decisions |
| `EVALUATION_PROGRESS`  | `GameGauntlet`     | `MetricsAnalysisOrchestrator`                            | Real-time tracking                     |
| `EVALUATION_STARTED`   | `EvaluationDaemon` | `Monitoring`                                             | Track evaluation lifecycle             |

### Model Events

| Event Type            | Emitter                    | Subscribers                                 | Purpose                        |
| --------------------- | -------------------------- | ------------------------------------------- | ------------------------------ |
| `MODEL_PROMOTED`      | `PromotionController`      | `UnifiedDistributionDaemon`, `FeedbackLoop` | Distribute to cluster          |
| `MODEL_UPDATED`       | `ModelRegistry`            | `UnifiedDistributionDaemon`                 | Sync metadata                  |
| `PROMOTION_FAILED`    | `AutoPromotionDaemon`      | `ModelLifecycleCoordinator`, `DataPipeline` | Track failures                 |
| `REGRESSION_DETECTED` | `ModelPerformanceWatchdog` | `ModelLifecycleCoordinator`, `DataPipeline` | Rollback bad models            |
| `REGRESSION_CRITICAL` | `RegressionDetector`       | `DaemonManager`, `AlertManager`             | Critical alert, pause training |

### Data Sync Events

| Event Type            | Emitter                          | Subscribers                            | Purpose                   |
| --------------------- | -------------------------------- | -------------------------------------- | ------------------------- |
| `DATA_SYNC_STARTED`   | `P2POrchestrator`                | `Monitoring`                           | Track sync lifecycle      |
| `DATA_SYNC_COMPLETED` | `P2POrchestrator`, `SyncPlanner` | `DataPipelineOrchestrator`             | Trigger NPZ export        |
| `DATA_SYNC_FAILED`    | `AutoSyncDaemon`                 | `RecoveryOrchestrator`                 | Handle sync failures      |
| `NEW_GAMES_AVAILABLE` | `DataPipelineOrchestrator`       | `SelfplayScheduler`, `ExportScheduler` | Trigger training pipeline |

### Cluster Events

| Event Type               | Emitter              | Subscribers                          | Purpose          |
| ------------------------ | -------------------- | ------------------------------------ | ---------------- |
| `HOST_ONLINE`            | `P2POrchestrator`    | `SyncRouter`, `SelfplayScheduler`    | Node available   |
| `HOST_OFFLINE`           | `P2POrchestrator`    | `UnifiedHealthManager`, `SyncRouter` | Node unavailable |
| `LEADER_ELECTED`         | `P2POrchestrator`    | `LeadershipCoordinator`              | Leader change    |
| `NODE_RECOVERED`         | `NodeRecoveryDaemon` | `SyncRouter`, `SelfplayScheduler`    | Node back online |
| `IDLE_RESOURCE_DETECTED` | `IdleDetectionLoop`  | `SelfplayScheduler`                  | Spawn selfplay   |

### Curriculum & Quality Events

| Event Type               | Emitter                 | Subscribers                                 | Purpose               |
| ------------------------ | ----------------------- | ------------------------------------------- | --------------------- |
| `CURRICULUM_REBALANCED`  | `CurriculumIntegration` | `SelfplayScheduler`, `SelfplayOrchestrator` | Update priorities     |
| `CURRICULUM_ADVANCED`    | `CurriculumController`  | `DataPipeline`                              | Track progression     |
| `ELO_VELOCITY_CHANGED`   | `QueuePopulator`        | `SelfplayScheduler`, `UnifiedFeedback`      | Adjust selfplay rate  |
| `ELO_SIGNIFICANT_CHANGE` | `EloSyncManager`        | `CurriculumIntegration`, `DataPipeline`     | Trigger rebalance     |
| `QUALITY_SCORE_UPDATED`  | `QualityMonitor`        | `DataPipeline`                              | Aggregate metrics     |
| `QUALITY_DEGRADED`       | `QualityMonitor`        | `AlertManager`                              | Alert on quality drop |

### Daemon Lifecycle Events

| Event Type                   | Emitter                | Subscribers                                    | Purpose              |
| ---------------------------- | ---------------------- | ---------------------------------------------- | -------------------- |
| `DAEMON_STATUS_CHANGED`      | `DaemonManager`        | `DaemonWatchdog`, `Monitoring`                 | Track daemon health  |
| `ALL_CRITICAL_DAEMONS_READY` | `DaemonManager`        | `MasterLoop`                                   | Startup complete     |
| `HYPERPARAMETER_UPDATED`     | `ImprovementOptimizer` | `EvaluationFeedbackHandler`, `TrainingTrigger` | Apply LR adjustments |
| `ADAPTIVE_PARAMS_CHANGED`    | `ImprovementOptimizer` | `EvaluationFeedbackHandler`                    | Real-time tuning     |

### Recovery Events

| Event Type              | Emitter                 | Subscribers                | Purpose                   |
| ----------------------- | ----------------------- | -------------------------- | ------------------------- |
| `REPAIR_COMPLETED`      | `RepairDaemon`          | `DataPipelineOrchestrator` | Retrigger sync            |
| `REPAIR_FAILED`         | `RepairDaemon`          | `DataPipelineOrchestrator` | Track for circuit breaker |
| `ORPHAN_GAMES_DETECTED` | `OrphanDetectionDaemon` | `DataPipelineOrchestrator` | Priority sync             |
| `TASK_ABANDONED`        | `JobManager`            | `SelfplayOrchestrator`     | Track cancelled jobs      |

## Event Flow Diagrams

### Training Completion Flow

```
TrainingCoordinator
       │
       ├──► TRAINING_COMPLETED
       │           │
       │           ├──► FeedbackLoopController._trigger_evaluation()
       │           │           │
       │           │           └──► GameGauntlet.run()
       │           │                       │
       │           │                       └──► EVALUATION_COMPLETED
       │           │                                   │
       │           │                                   ├──► CurriculumIntegration
       │           │                                   └──► AutoPromotionDaemon
       │           │
       │           ├──► DataPipelineOrchestrator._on_training_completed()
       │           │
       │           └──► UnifiedDistributionDaemon._on_training_completed()
       │
       └──► Model saved to disk
```

### Data Sync Flow

```
P2POrchestrator._sync_selfplay_to_training_nodes()
       │
       ├──► DATA_SYNC_STARTED
       │
       ├──► [Sync files to training nodes]
       │
       └──► DATA_SYNC_COMPLETED
                   │
                   └──► DataPipelineOrchestrator._on_sync_complete()
                               │
                               └──► NPZ export triggered
                                           │
                                           └──► NEW_GAMES_AVAILABLE
```

### Model Promotion Flow

```
AutoPromotionDaemon
       │
       ├──► [Evaluate model vs baselines]
       │
       └──► MODEL_PROMOTED (if passes thresholds)
                   │
                   ├──► UnifiedDistributionDaemon._on_model_promoted()
                   │           │
                   │           └──► Distribute to all cluster nodes
                   │
                   └──► FeedbackLoopController._on_promotion()
                               │
                               └──► Update training params
```

## Startup Order Dependencies

Critical daemons must start in order to ensure events aren't lost:

1. **EVENT_ROUTER** - Must start first (event bus)
2. **FEEDBACK_LOOP** - Subscribes to training/evaluation events
3. **DATA_PIPELINE** - Subscribes to sync events
4. **AUTO_SYNC** - Emits sync events (subscribers must be ready)

This order is enforced in `daemon_lifecycle.py:_reorder_for_critical_startup()`.

## Adding New Events

### Step 1: Define Event Type

Add to `app/distributed/data_events.py`:

```python
class DataEventType(str, Enum):
    # ... existing events ...
    MY_NEW_EVENT = "my_new_event"
```

### Step 2: Create Emit Function

Add to `app/distributed/data_events.py`:

```python
def emit_my_new_event(
    config_key: str,
    extra_data: dict | None = None,
) -> None:
    """Emit MY_NEW_EVENT when something happens."""
    emit_data_event(
        DataEventType.MY_NEW_EVENT,
        {
            "config_key": config_key,
            "extra_data": extra_data or {},
            "timestamp": time.time(),
        }
    )
```

### Step 3: Add Cross-Process Mapping (if needed)

If the event needs to propagate across processes, add to `event_mappings.py`:

```python
DATA_TO_CROSS_PROCESS_MAP = {
    # ... existing mappings ...
    DataEventType.MY_NEW_EVENT: "MY_NEW_EVENT",
}
```

### Step 4: Subscribe in Consumer

In the consuming daemon or coordinator:

```python
from app.coordination.event_router import get_router, DataEventType

class MyConsumer:
    def __init__(self):
        router = get_router()
        router.subscribe(DataEventType.MY_NEW_EVENT, self._on_my_event)

    async def _on_my_event(self, event: dict) -> None:
        config_key = event.get("config_key")
        # Handle event...
```

### Step 5: Document in This Guide

Add your event to the appropriate table above.

## Best Practices

1. **Always include `timestamp`** in event payloads for ordering
2. **Use typed emit functions** rather than raw `emit_data_event()`
3. **Subscribe in `__init__`** to ensure handlers are ready before events arrive
4. **Check for `None` payloads** - events may have minimal data
5. **Use `fire_and_forget()`** for non-critical event emissions
6. **Log event handling** at DEBUG level for tracing

## Debugging Events

### Enable Event Tracing

```bash
export RINGRIFT_EVENT_TRACE=true
export RINGRIFT_LOG_LEVEL=DEBUG
```

### Check Event Flow

```python
from app.coordination.event_router import get_router

router = get_router()
print(f"Subscribers: {router.get_subscription_counts()}")
print(f"Recent events: {router.get_recent_events()}")
```

### Verify Subscription Wiring

```python
from app.coordination.data_pipeline_orchestrator import get_data_pipeline_orchestrator

dpo = get_data_pipeline_orchestrator()
# Check event_subscriptions attribute or _subscribed flag
```

## Related Documentation

- `CLAUDE.md` - Daemon management and lifecycle
- `daemon_registry.py` - Daemon specifications and dependencies
- `event_subscription_registry.py` - Delegated event wiring (DELEGATION_REGISTRY)
- `coordination_bootstrap.py` - Coordinator initialization and wiring glue
- `DEPRECATION_GUIDE.md` - Deprecated event patterns

---

_Last updated: December 29, 2025_
