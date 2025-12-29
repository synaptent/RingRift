# Event Wiring Diagram

This document provides a visual representation of event flows in the RingRift AI coordination infrastructure, showing how components communicate through the event system.

## Overview

The event system uses three layers unified through `event_router.py`:

| Component                  | File                                       | Purpose                                   |
| -------------------------- | ------------------------------------------ | ----------------------------------------- |
| **UnifiedEventRouter**     | `app/coordination/event_router.py`         | Unified API routing to all event buses    |
| **DataEventType**          | `app/distributed/data_events.py`           | In-memory async event types (100+ events) |
| **StageEvent**             | `app/coordination/stage_events.py`         | Pipeline stage completion events          |
| **CrossProcessEventQueue** | `app/coordination/cross_process_events.py` | SQLite-backed cross-process persistence   |

Note: Diagrams emphasize the primary flow and omit optional stages (for example
NPZ combination and data freshness gates). For a complete event list, see
`EVENT_CATALOG.md`.

### Key Concepts

- **Emitter**: Component that publishes an event
- **Subscriber**: Component that handles an event
- **Router**: Central hub that routes events to all appropriate buses
- **Deduplication**: Content-hash based prevention of duplicate event delivery

## Event Flow Diagrams

### Training Pipeline Flow

```
                    +------------------+
                    | TrainingDaemon   |
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              |     TRAINING_STARTED        |
              +-----------------------------+
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
+---------------+    +---------------+    +---------------+
| SyncRouter    |    | IdleShutdown  |    | DataPipeline  |
| (pause other  |    | (pause idle   |    | (track stage) |
|  transfers)   |    |  detection)   |    |               |
+---------------+    +---------------+    +---------------+
                             |
                             v
                    +------------------+
                    |  [Training...]   |
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              |    TRAINING_COMPLETED       |
              +-----------------------------+
                             |
    +------------------------+------------------------+
    |                        |                        |
    v                        v                        v
+---------------+    +---------------+    +------------------+
| FeedbackLoop  |    | DataPipeline  |    | ModelDistribution|
| (trigger      |    | (track stage) |    | (prepare for     |
|  evaluation)  |    |               |    |  promotion)      |
+---------------+    +---------------+    +------------------+
        |
        v
+---------------+
| GameGauntlet  |
+-------+-------+
        |
        v
+-----------------------------+
|   EVALUATION_COMPLETED      |
+-----------------------------+
        |
        +------------------------+
        |                        |
        v                        v
+---------------+    +------------------+
| Curriculum    |    | AutoPromotion    |
| Integration   |    | Daemon           |
+---------------+    +--------+---------+
                              |
                              v (if passed)
              +-----------------------------+
              |      MODEL_PROMOTED         |
              +-----------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +------------------+           +------------------+
    | Distribution     |           | FeedbackLoop     |
    | Daemon           |           | (update params)  |
    +------------------+           +------------------+
```

### Data Sync Flow

```
+----------------------+
| Selfplay Completed   |
| (any node)           |
+----------+-----------+
           |
           v
+-----------------------------+
|     SELFPLAY_COMPLETE       |
+-----------------------------+
           |
           v
+----------------------+
| P2P Orchestrator     |
| _sync_selfplay_...() |
+----------+-----------+
           |
           v
+-----------------------------+
|    DATA_SYNC_STARTED        |
+-----------------------------+
           |
           v
+----------------------+
| [rsync files to      |
|  training nodes]     |
+----------+-----------+
           |
           v
+-----------------------------+
|   DATA_SYNC_COMPLETED       |
+-----------------------------+
           |
           v
+----------------------+
| DataPipeline         |
| _on_sync_complete()  |
+----------+-----------+
           |
           v
+----------------------+
| NPZ Export           |
+----------+-----------+
           |
           v
+-----------------------------+
|   NEW_GAMES_AVAILABLE       |
+-----------------------------+
           |
           +-------------------+
           |                   |
           v                   v
+---------------+    +-----------------+
| Selfplay      |    | Export          |
| Scheduler     |    | Scheduler       |
+---------------+    +-----------------+
```

### Orphan Games Recovery Flow

```
+--------------------+
| OrphanDetection    |
| Daemon             |
+---------+----------+
          |
          v (on scan cycle)
+-----------------------------+
|   ORPHAN_GAMES_DETECTED     |
+-----------------------------+
          |
          v
+--------------------+
| DataPipeline       |
| ._on_orphan_games_ |
| detected()         |
+---------+----------+
          |
          v
+--------------------+
| SyncFacade         |
| .trigger_priority_ |
| sync()             |
+---------+----------+
          |
          v
+-----------------------------+
|   DATA_SYNC_COMPLETED       |
+-----------------------------+
          |
          v
+--------------------+
| DataPipeline       |
| ._on_orphan_games_ |
| registered()       |
+---------+----------+
          |
          v
+-----------------------------+
|   NEW_GAMES_AVAILABLE       |
+-----------------------------+
```

### Cluster Health Flow

```
+--------------------+        +--------------------+
| P2P Orchestrator   |        | NodeRecoveryDaemon |
+---------+----------+        +---------+----------+
          |                             |
          v                             v
+------------------+          +------------------+
|   HOST_OFFLINE   |          |  NODE_RECOVERED  |
+------------------+          +------------------+
          |                             |
          +-------+-------+-------------+
                  |       |
                  v       v
    +----------------------+    +----------------+
    | UnifiedHealthManager |    | SyncRouter     |
    | ._on_host_offline()  |    | ._on_node_     |
    +----------------------+    | recovered()   |
                                +----------------+
                                        |
                                        v
                                +----------------+
                                | Selfplay       |
                                | Scheduler      |
                                +----------------+
```

## Event Reference Table

### Core Pipeline Events

| Event                  | Emitter(s)          | Subscriber(s)                                 | Purpose                               |
| ---------------------- | ------------------- | --------------------------------------------- | ------------------------------------- |
| `TRAINING_STARTED`     | TrainingCoordinator | SyncRouter, IdleShutdown, DataPipeline        | Pause idle detection, prioritize sync |
| `TRAINING_COMPLETED`   | TrainingCoordinator | FeedbackLoop, DataPipeline, ModelDistribution | Trigger evaluation pipeline           |
| `TRAINING_FAILED`      | TrainingCoordinator | DaemonManager, RecoveryOrchestrator           | Handle failure, initiate recovery     |
| `EVALUATION_COMPLETED` | GameGauntlet        | FeedbackLoop, CurriculumIntegration           | Update curriculum weights             |
| `MODEL_PROMOTED`       | PromotionController | UnifiedDistributionDaemon, FeedbackLoop       | Distribute model to cluster           |

### Data Synchronization Events

| Event                     | Emitter(s)                  | Subscriber(s)                      | Purpose                    |
| ------------------------- | --------------------------- | ---------------------------------- | -------------------------- |
| `DATA_SYNC_STARTED`       | P2POrchestrator             | Monitoring                         | Track sync lifecycle       |
| `DATA_SYNC_COMPLETED`     | AutoSyncDaemon, SyncPlanner | DataPipelineOrchestrator           | Trigger NPZ export         |
| `DATA_SYNC_FAILED`        | AutoSyncDaemon              | RecoveryOrchestrator               | Handle sync failures       |
| `NEW_GAMES_AVAILABLE`     | DataPipelineOrchestrator    | SelfplayScheduler, ExportScheduler | Signal training data ready |
| `ORPHAN_GAMES_DETECTED`   | OrphanDetectionDaemon       | DataPipelineOrchestrator           | Trigger priority recovery  |
| `ORPHAN_GAMES_REGISTERED` | DataPipelineOrchestrator    | DataPipelineOrchestrator           | Complete recovery cycle    |

### Model Lifecycle Events

| Event                 | Emitter(s)               | Subscriber(s)                           | Purpose                        |
| --------------------- | ------------------------ | --------------------------------------- | ------------------------------ |
| `PROMOTION_CANDIDATE` | AutoPromotionDaemon      | DataPipelineOrchestrator                | Track promotion candidates     |
| `PROMOTION_STARTED`   | AutoPromotionDaemon      | DataPipelineOrchestrator                | Track promotion lifecycle      |
| `PROMOTION_FAILED`    | AutoPromotionDaemon      | ModelLifecycleCoordinator, DataPipeline | Track and notify failures      |
| `MODEL_UPDATED`       | ModelRegistry            | UnifiedDistributionDaemon               | Sync model metadata            |
| `REGRESSION_DETECTED` | ModelPerformanceWatchdog | ModelLifecycleCoordinator, DataPipeline | Rollback bad models            |
| `REGRESSION_CRITICAL` | RegressionDetector       | DaemonManager, AlertManager             | Critical alert, pause training |

### Cluster Management Events

| Event                    | Emitter(s)         | Subscriber(s)                        | Purpose                         |
| ------------------------ | ------------------ | ------------------------------------ | ------------------------------- |
| `HOST_ONLINE`            | P2POrchestrator    | SyncRouter, SelfplayScheduler        | Node became available           |
| `HOST_OFFLINE`           | P2POrchestrator    | UnifiedHealthManager, SyncRouter     | Node became unavailable         |
| `NODE_RECOVERED`         | NodeRecoveryDaemon | SyncRouter, SelfplayScheduler        | Node back online after recovery |
| `LEADER_ELECTED`         | P2POrchestrator    | LeadershipCoordinator, DaemonManager | Cluster leader change           |
| `IDLE_RESOURCE_DETECTED` | IdleDetectionLoop  | SelfplayScheduler                    | Spawn selfplay on idle GPU      |

### Curriculum & Quality Events

| Event                    | Emitter(s)            | Subscriber(s)                           | Purpose                      |
| ------------------------ | --------------------- | --------------------------------------- | ---------------------------- |
| `CURRICULUM_REBALANCED`  | CurriculumIntegration | SelfplayScheduler, SelfplayOrchestrator | Update selfplay priorities   |
| `CURRICULUM_ADVANCED`    | CurriculumController  | DataPipeline                            | Track curriculum progression |
| `ELO_VELOCITY_CHANGED`   | QueuePopulator        | SelfplayScheduler, UnifiedFeedback      | Adjust selfplay rate         |
| `ELO_SIGNIFICANT_CHANGE` | EloSyncManager        | CurriculumIntegration, DataPipeline     | Trigger curriculum rebalance |
| `QUALITY_SCORE_UPDATED`  | QualityMonitor        | DataPipelineOrchestrator                | Aggregate quality metrics    |
| `QUALITY_DEGRADED`       | QualityMonitor        | AlertManager, FeedbackLoop              | Alert on quality drop        |

### Repair & Recovery Events

| Event                         | Emitter(s)                | Subscriber(s)             | Purpose                         |
| ----------------------------- | ------------------------- | ------------------------- | ------------------------------- |
| `REPAIR_COMPLETED`            | RepairDaemon              | DataPipelineOrchestrator  | Retrigger sync after repair     |
| `REPAIR_FAILED`               | RepairDaemon              | DataPipelineOrchestrator  | Track for circuit breaker       |
| `TASK_ABANDONED`              | JobManager                | SelfplayOrchestrator      | Track cancelled jobs            |
| `TRAINING_ROLLBACK_NEEDED`    | RegressionDetector        | ModelLifecycleCoordinator | Rollback to previous checkpoint |
| `TRAINING_ROLLBACK_COMPLETED` | ModelLifecycleCoordinator | FeedbackLoop              | Confirm rollback success        |

### Backpressure Events

| Event                          | Emitter(s)          | Subscriber(s)                        | Purpose                  |
| ------------------------------ | ------------------- | ------------------------------------ | ------------------------ |
| `BACKPRESSURE_ACTIVATED`       | BackpressureMonitor | SyncRouter, DataPipelineOrchestrator | Pause or slow operations |
| `BACKPRESSURE_RELEASED`        | BackpressureMonitor | SyncRouter, DataPipelineOrchestrator | Resume normal operations |
| `RESOURCE_CONSTRAINT_DETECTED` | ResourceMonitor     | DataPipelineOrchestrator             | Adapt to resource limits |

## Bootstrap Initialization Order

Coordinators are initialized in `coordination_bootstrap.py` in dependency order:

### Layer 1: Foundational (No Dependencies)

1. **task_coordinator** - TaskLifecycleCoordinator
2. **global_task_coordinator** - Global TaskCoordinator
3. **resource_coordinator** - ResourceMonitoringCoordinator
4. **cache_orchestrator** - CacheCoordinationOrchestrator

### Layer 2: Infrastructure Support

5. **health_manager** - UnifiedHealthManager
6. **error_coordinator** - Delegates to health_manager
7. **model_coordinator** - ModelLifecycleCoordinator

### Layer 3: Sync and Training

8. **sync_coordinator** - SyncCoordinator (+ SyncRouter wiring)
9. **training_coordinator** - TrainingCoordinator

### Layer 4: Data Integrity

10. **transfer_verifier** - TransferVerifier
11. **ephemeral_guard** - EphemeralDataGuard
12. **queue_populator** - UnifiedQueuePopulator

### Layer 5: Selfplay

13. **selfplay_orchestrator** - SelfplayOrchestrator
14. **selfplay_scheduler** - SelfplayScheduler

### Layer 6: Pipeline and Jobs

15. **pipeline_orchestrator** - DataPipelineOrchestrator (special handling)
16. **multi_provider** - MultiProviderOrchestrator
17. **job_scheduler** - JobScheduler

### Layer 7: Daemons

18. **auto_export_daemon** - AutoExportDaemon
19. **evaluation_daemon** - EvaluationDaemon
20. **model_distribution_daemon** - UnifiedDistributionDaemon
21. **idle_resource_daemon** - IdleResourceDaemon
22. **quality_monitor_daemon** - QualityMonitorDaemon
23. **orphan_detection_daemon** - OrphanDetectionDaemon
24. **curriculum_integration** - CurriculumIntegration

### Layer 8: Metrics and Optimization

25. **metrics_orchestrator** - MetricsAnalysisOrchestrator
26. **optimization_coordinator** - OptimizationCoordinator

### Layer 9: Leadership (Coordinates All Others)

27. **leadership_coordinator** - LeadershipCoordinator

## Daemon Startup Order

Critical daemons must start in order to prevent event loss:

```
1. EVENT_ROUTER        (event bus - must be first)
       |
2. FEEDBACK_LOOP       (subscribes to training/evaluation)
       |
3. DATA_PIPELINE       (subscribes to sync events)
       |
4. AUTO_SYNC           (emits sync events - subscribers must be ready)
```

This order is enforced in `daemon_lifecycle.py:_reorder_for_critical_startup()`.

## Troubleshooting Guide

### Event Not Received

1. **Check subscription order**: Subscriber must be initialized before emitter starts

   ```python
   from app.coordination.event_router import get_router
   router = get_router()
   print(router.get_subscription_counts())  # Check if subscribed
   ```

2. **Verify event type**: Ensure emitter uses correct DataEventType enum

   ```python
   from app.distributed.data_events import DataEventType
   # Use DataEventType.DATA_SYNC_COMPLETED.value for subscriptions
   ```

3. **Check daemon startup**: Ensure daemon started successfully
   ```bash
   python scripts/launch_daemons.py --status
   ```

### Duplicate Events

1. **Content-hash deduplication**: Router uses SHA256 hash of stable payload fields
2. **Check event_id**: Each event has unique ID for deduplication
3. **Increase max_seen_events**: Default is 10000, increase if needed

### Event Handler Timeout

1. **Default timeout**: 30 seconds (configurable via `RINGRIFT_EVENT_HANDLER_TIMEOUT`)
2. **Check handler complexity**: Simplify or make async
3. **Enable tracing**:
   ```bash
   export RINGRIFT_EVENT_TRACE=true
   export RINGRIFT_LOG_LEVEL=DEBUG
   ```

### Missing Event Emissions

1. **Check emitter implementation**: Ensure emit function called on all paths
2. **Verify error handling**: Events should emit on both success and failure
3. **Check P2P orchestrator events**: Verify HOST_OFFLINE, LEADER_ELECTED emissions

### Debugging Commands

```python
# Check router state
from app.coordination.event_router import get_router
router = get_router()
print(f"Subscriptions: {router.get_subscription_counts()}")
print(f"Recent events: {router.get_recent_events(limit=10)}")
print(f"Duplicates prevented: {router._duplicates_prevented}")

# Check DataPipelineOrchestrator subscriptions
from app.coordination.data_pipeline_orchestrator import get_data_pipeline_orchestrator
dpo = get_data_pipeline_orchestrator()
print(f"Subscribed: {dpo._subscribed}")

# Enable event tracing
import os
os.environ["RINGRIFT_EVENT_TRACE"] = "true"
```

## Related Documentation

- `EVENT_WIRING_GUIDE.md` - Detailed event wiring instructions
- `EVENT_SYSTEM_REFERENCE.md` - Full event type catalog
- `DAEMON_REGISTRY.md` - Daemon specifications and dependencies
- `coordination_bootstrap.py` - Coordinator initialization
- `docs/runbooks/EVENT_WIRING_VERIFICATION.md` - Verification procedures

---

_Last updated: December 2025_
