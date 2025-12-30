# Event System Quick Reference Card

**Last Updated**: December 30, 2025

One-page reference for common event patterns. See `EVENT_SYSTEM_REFERENCE.md` for full documentation.

---

## Subscribe & Publish

```python
# Subscribe (HandlerBase - recommended)
from app.coordination.handler_base import HandlerBase

class MyHandler(HandlerBase):
    def _get_subscriptions(self):
        return {"training_completed": self._on_training}

# Subscribe (direct)
from app.coordination.event_router import get_event_bus
bus = get_event_bus()
bus.subscribe("training_completed", my_handler)

# Publish (typed emitter - recommended)
from app.coordination.event_emitters import emit_training_complete
emit_training_complete(config_key="hex8_2p", model_path="...", epochs=50)

# Publish (direct)
from app.distributed.data_events import DataEvent, DataEventType
bus.publish(DataEvent(DataEventType.TRAINING_COMPLETED, {"config_key": "hex8_2p"}))
```

---

## Common Event Flows

### Training Pipeline

```
NEW_GAMES_AVAILABLE → CONSOLIDATION_COMPLETE → NPZ_EXPORT_COMPLETE
    → TRAINING_STARTED → TRAINING_PROGRESS → TRAINING_COMPLETED
    → EVALUATION_STARTED → EVALUATION_COMPLETED → MODEL_PROMOTED
```

### Selfplay Feedback Loop

```
SELFPLAY_COMPLETE → QUALITY_SCORE_UPDATED → EXPLORATION_BOOST
    → SELFPLAY_ALLOCATION_UPDATED → ELO_UPDATED
    → CURRICULUM_REBALANCED → SELFPLAY_TARGET_UPDATED
```

### Architecture Feedback

```
EVALUATION_COMPLETED → ARCHITECTURE_WEIGHTS_UPDATED → SELFPLAY_ALLOCATION_UPDATED
```

### Sync & Distribution

```
DATA_SYNC_STARTED → DATA_SYNC_COMPLETED → NEW_GAMES_AVAILABLE
MODEL_PROMOTED → MODEL_DISTRIBUTION_STARTED → MODEL_DISTRIBUTION_COMPLETE
```

### Health & Recovery

```
NODE_UNHEALTHY → RECOVERY_INITIATED → RECOVERY_COMPLETED
REGRESSION_DETECTED → REGRESSION_CRITICAL → TRAINING_ROLLBACK_NEEDED
```

---

## Event Categories at a Glance

| Category         | Key Events                                              | Primary Emitter              |
| ---------------- | ------------------------------------------------------- | ---------------------------- |
| **Data**         | `NEW_GAMES_AVAILABLE`, `DATA_SYNC_*`                    | AutoSyncDaemon, DataPipeline |
| **Training**     | `TRAINING_*`, `TRAINING_LOSS_*`                         | TrainingCoordinator          |
| **Evaluation**   | `EVALUATION_*`, `ELO_UPDATED`                           | EvaluationDaemon             |
| **Promotion**    | `MODEL_PROMOTED`, `PROMOTION_*`                         | AutoPromotionDaemon          |
| **Selfplay**     | `SELFPLAY_*`, `ARCHITECTURE_WEIGHTS_UPDATED`, `BATCH_*` | SelfplayScheduler            |
| **Quality**      | `QUALITY_*`, `TRAINING_BLOCKED_BY_*`                    | QualityMonitorDaemon         |
| **Curriculum**   | `CURRICULUM_*`, `EXPLORATION_*`                         | CurriculumIntegration        |
| **Regression**   | `REGRESSION_*`, `PLATEAU_*`                             | RegressionDetector           |
| **Distribution** | `MODEL_DISTRIBUTION_*`, `SYNC_*`                        | UnifiedDistributionDaemon    |
| **Cluster**      | `NODE_*`, `HOST_*`, `CLUSTER_*`                         | P2POrchestrator              |
| **Resources**    | `BACKPRESSURE_*`, `IDLE_*`, `CAPACITY_*`                | ResourceMonitors             |
| **Daemon**       | `DAEMON_*`, `COORDINATOR_*`                             | DaemonManager                |
| **Work Queue**   | `WORK_*`, `TASK_*`, `JOB_*`                             | JobManager                   |
| **Leader**       | `LEADER_*`, `SPLIT_BRAIN_*`                             | P2POrchestrator              |

---

## Critical Event Wiring

These events drive the main feedback loops:

| Event                          | Emitter                        | Must-Have Subscriber                       | Purpose                     |
| ------------------------------ | ------------------------------ | ------------------------------------------ | --------------------------- |
| `TRAINING_COMPLETED`           | TrainingCoordinator            | FeedbackLoopController                     | Trigger evaluation          |
| `EVALUATION_COMPLETED`         | GameGauntlet                   | CurriculumIntegration, AutoPromotionDaemon | Update weights              |
| `MODEL_PROMOTED`               | AutoPromotionDaemon            | UnifiedDistributionDaemon                  | Distribute model            |
| `DATA_SYNC_COMPLETED`          | AutoSyncDaemon                 | DataPipelineOrchestrator                   | Trigger export              |
| `ELO_UPDATED`                  | EloSyncManager                 | SelfplayScheduler                          | Rebalance allocation        |
| `QUALITY_SCORE_UPDATED`        | QualityMonitorDaemon           | FeedbackLoopController                     | Adjust exploration          |
| `REGRESSION_CRITICAL`          | RegressionDetector             | DaemonManager                              | Halt training               |
| `PLATEAU_DETECTED`             | FeedbackLoopController         | ImprovementOptimizer                       | Boost exploration           |
| `ORPHAN_GAMES_DETECTED`        | OrphanDetectionDaemon          | DataPipelineOrchestrator                   | Priority sync               |
| `ARCHITECTURE_WEIGHTS_UPDATED` | ArchitectureFeedbackController | SelfplayScheduler                          | Update architecture weights |

---

## Event Payload Patterns

```python
# Training events
{"config_key": str, "model_path": str, "epochs": int, "loss": float}

# Evaluation events
{"config_key": str, "elo": float, "win_rate": float, "games_played": int}

# Sync events
{"sync_type": str, "target_nodes": list, "files_synced": int, "duration": float}

# Quality events
{"config_key": str, "quality_score": float, "threshold": float}

# Node events
{"node_id": str, "status": str, "reason": str}

# Selfplay events
{"config_key": str, "games_completed": int, "samples_generated": int}
```

---

## Debugging Events

```python
# List all subscriptions
from app.coordination.event_router import get_router
router = get_router()
for event_type, handlers in router._subscriptions.items():
    print(f"{event_type}: {len(handlers)} handlers")

# Trace event flow
import logging
logging.getLogger("app.coordination.event_router").setLevel(logging.DEBUG)

# Check cross-process queue
from app.coordination.cross_process_events import CrossProcessEventQueue
queue = CrossProcessEventQueue()
pending = queue.get_pending_events(limit=10)
```

---

## Quick Imports

```python
# Core
from app.coordination.event_router import get_event_bus, get_router
from app.distributed.data_events import DataEventType, DataEvent

# Typed emitters (70+ available)
from app.coordination.event_emitters import (
    emit_training_complete,
    emit_evaluation_complete,
    emit_selfplay_complete,
    emit_sync_complete,
    emit_model_promoted,
)

# Handler base
from app.coordination.handler_base import HandlerBase, HealthCheckResult

# Cross-process
from app.coordination.cross_process_events import CrossProcessEventQueue
```

---

## See Also

- `EVENT_SYSTEM_REFERENCE.md` - Full 140+ event documentation
- `EVENT_CATALOG.md` - All emitter/subscriber mappings
- `EVENT_WIRING_GUIDE.md` - How to add new events
- `HANDLER_BASE_API.md` - Handler base class reference
