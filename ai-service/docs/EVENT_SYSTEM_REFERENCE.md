# Event System Reference

The RingRift coordination infrastructure uses an event-driven architecture with 100+ event types. This document provides a comprehensive reference for the event system.

**Created**: December 2025 (Wave 4 Phase 2)

## Overview

The event system enables loose coupling between components of the AI training pipeline:

- Data collection → Training → Evaluation → Promotion
- Cluster health monitoring → Recovery
- Selfplay scheduling → Curriculum advancement

### Key Components

| Component            | Location          | Purpose                     |
| -------------------- | ----------------- | --------------------------- |
| `EventBus`           | `data_events.py`  | Core pub/sub mechanism      |
| `UnifiedEventRouter` | `event_router.py` | Routes events to all buses  |
| `DataEventType`      | `data_events.py`  | Enum of all event types     |
| `HandlerBase`        | `handler_base.py` | Canonical base for handlers |

## Quick Start

### Subscribing to Events

```python
from app.coordination.event_router import DataEventType, get_event_bus

bus = get_event_bus()
bus.subscribe(DataEventType.TRAINING_COMPLETED, my_handler)

def my_handler(event):
    config_key = event.payload.get("config_key")
    # Process event...
```

### Using HandlerBase (Recommended)

```python
from app.coordination.handler_base import HandlerBase

class MyHandler(HandlerBase):
    def __init__(self):
        super().__init__("MyHandler")

    def _get_subscriptions(self):
        return {
            DataEventType.TRAINING_COMPLETED: self._on_training_completed,
            DataEventType.EVALUATION_COMPLETED: self._on_evaluation_completed,
        }

    async def _on_training_completed(self, event):
        # Handle event
        pass
```

`BaseEventHandler` remains as a backward-compatible alias in `handler_base.py`, but new
handlers should import `HandlerBase` directly.

### Publishing Events

```python
from app.coordination.event_router import DataEventType, get_event_bus, DataEvent

bus = get_event_bus()
await bus.publish(DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"config_key": "hex8_2p", "model_path": "/path/to/model.pth"},
    source="TrainingCoordinator",
))
```

---

## Event Types by Category

### Data Collection Events

| Event                    | Value                    | Emitters                       | Subscribers                                 | Purpose               |
| ------------------------ | ------------------------ | ------------------------------ | ------------------------------------------- | --------------------- |
| `NEW_GAMES_AVAILABLE`    | `new_games`              | SelfplayRunner, AutoSyncDaemon | DataPipelineOrchestrator, ExportScheduler   | Trigger data export   |
| `DATA_SYNC_STARTED`      | `sync_started`           | SyncPlanner, AutoSyncDaemon    | DataPipelineOrchestrator                    | Track sync progress   |
| `DATA_SYNC_COMPLETED`    | `sync_completed`         | SyncPlanner, P2POrchestrator   | DataPipelineOrchestrator, SelfplayScheduler | Trigger export stage  |
| `DATA_SYNC_FAILED`       | `sync_failed`            | SyncPlanner                    | RecoveryOrchestrator                        | Retry sync            |
| `GAME_SYNCED`            | `game_synced`            | AutoSyncDaemon                 | ClusterManifest                             | Update game registry  |
| `DATA_STALE`             | `data_stale`             | TrainingFreshness              | SyncRouter                                  | Trigger priority sync |
| `DATA_FRESH`             | `data_fresh`             | TrainingFreshness              | TrainCLI                                    | Proceed with training |
| `SYNC_TRIGGERED`         | `sync_triggered`         | TrainingFreshness              | SyncRouter                                  | Force immediate sync  |
| `CONSOLIDATION_STARTED`  | `consolidation_started`  | DatabaseConsolidator           | ProgressTracker                             | Track consolidation   |
| `CONSOLIDATION_COMPLETE` | `consolidation_complete` | DatabaseConsolidator           | DataPipelineOrchestrator                    | Trigger export        |

### Training Events

| Event                         | Value                         | Emitters             | Subscribers                                   | Purpose                |
| ----------------------------- | ----------------------------- | -------------------- | --------------------------------------------- | ---------------------- |
| `TRAINING_THRESHOLD_REACHED`  | `training_threshold`          | QueuePopulator       | TrainingTrigger                               | Schedule training      |
| `TRAINING_STARTED`            | `training_started`            | TrainingCoordinator  | SyncRouter, IdleShutdown                      | Pause idle detection   |
| `TRAINING_PROGRESS`           | `training_progress`           | TrainLoop            | ProgressMonitor                               | Track training         |
| `TRAINING_COMPLETED`          | `training_completed`          | TrainingCoordinator  | FeedbackLoop, DataPipeline, ModelDistribution | Trigger evaluation     |
| `TRAINING_FAILED`             | `training_failed`             | TrainingCoordinator  | RecoveryOrchestrator                          | Handle failure         |
| `TRAINING_EARLY_STOPPED`      | `training_early_stopped`      | TrainLoop            | FeedbackLoop                                  | Adjust hyperparameters |
| `TRAINING_ROLLBACK_NEEDED`    | `training_rollback_needed`    | RegressionDetector   | RecoveryOrchestrator                          | Restore checkpoint     |
| `TRAINING_ROLLBACK_COMPLETED` | `training_rollback_completed` | RecoveryOrchestrator | FeedbackLoop                                  | Resume training        |

### Evaluation Events

| Event                    | Value                    | Emitters       | Subscribers                         | Purpose              |
| ------------------------ | ------------------------ | -------------- | ----------------------------------- | -------------------- |
| `EVALUATION_STARTED`     | `evaluation_started`     | GameGauntlet   | ProgressMonitor                     | Track evaluation     |
| `EVALUATION_PROGRESS`    | `evaluation_progress`    | GameGauntlet   | MetricsAnalysis                     | Real-time tracking   |
| `EVALUATION_COMPLETED`   | `evaluation_completed`   | GameGauntlet   | FeedbackLoop, CurriculumIntegration | Adjust curriculum    |
| `EVALUATION_FAILED`      | `evaluation_failed`      | GameGauntlet   | RecoveryOrchestrator                | Retry evaluation     |
| `ELO_UPDATED`            | `elo_updated`            | EloService     | EloSyncManager                      | Sync Elo ratings     |
| `ELO_SIGNIFICANT_CHANGE` | `elo_significant_change` | EloSyncManager | CurriculumIntegration               | Rebalance curriculum |
| `ELO_VELOCITY_CHANGED`   | `elo_velocity_changed`   | QueuePopulator | SelfplayScheduler, UnifiedFeedback  | Adjust selfplay rate |

### Promotion Events

| Event                   | Value                   | Emitters             | Subscribers                             | Purpose                |
| ----------------------- | ----------------------- | -------------------- | --------------------------------------- | ---------------------- |
| `PROMOTION_CANDIDATE`   | `promotion_candidate`   | GameGauntlet         | PromotionController                     | Evaluate for promotion |
| `PROMOTION_STARTED`     | `promotion_started`     | PromotionController  | ProgressMonitor                         | Track promotion        |
| `MODEL_PROMOTED`        | `model_promoted`        | PromotionController  | ModelDistribution, FeedbackLoop         | Distribute new model   |
| `PROMOTION_FAILED`      | `promotion_failed`      | PromotionController  | ModelLifecycleCoordinator, DataPipeline | Track failures         |
| `PROMOTION_REJECTED`    | `promotion_rejected`    | PromotionController  | FeedbackLoop                            | Adjust training        |
| `PROMOTION_ROLLED_BACK` | `promotion_rolled_back` | RecoveryOrchestrator | ModelLifecycleCoordinator               | Restore previous model |
| `MODEL_UPDATED`         | `model_updated`         | ModelRegistry        | UnifiedDistributionDaemon               | Sync model metadata    |

### Curriculum Events

| Event                   | Value                   | Emitters                   | Subscribers                             | Purpose             |
| ----------------------- | ----------------------- | -------------------------- | --------------------------------------- | ------------------- |
| `CURRICULUM_REBALANCED` | `curriculum_rebalanced` | CurriculumIntegration      | SelfplayScheduler, SelfplayOrchestrator | Update allocations  |
| `CURRICULUM_ADVANCED`   | `curriculum_advanced`   | GauntletFeedbackController | SelfplayScheduler                       | Move to harder tier |
| `WEIGHT_UPDATED`        | `weight_updated`        | CurriculumIntegration      | SelfplayScheduler                       | Adjust priorities   |
| `OPPONENT_MASTERED`     | `opponent_mastered`     | GauntletFeedbackController | CurriculumIntegration                   | Advance curriculum  |

### Selfplay Events

| Event                         | Value                         | Emitters               | Subscribers              | Purpose                |
| ----------------------------- | ----------------------------- | ---------------------- | ------------------------ | ---------------------- |
| `SELFPLAY_COMPLETE`           | `selfplay_complete`           | SelfplayRunner         | DataPipelineOrchestrator | Process results        |
| `SELFPLAY_TARGET_UPDATED`     | `selfplay_target_updated`     | QueuePopulator         | SelfplayScheduler        | Adjust game count      |
| `SELFPLAY_RATE_CHANGED`       | `selfplay_rate_changed`       | FeedbackLoopController | SelfplayScheduler        | Adjust generation rate |
| `SELFPLAY_ALLOCATION_UPDATED` | `selfplay_allocation_updated` | CurriculumIntegration  | SelfplayScheduler        | Change config weights  |

### Quality Events

| Event                         | Value                         | Emitters             | Subscribers          | Purpose             |
| ----------------------------- | ----------------------------- | -------------------- | -------------------- | ------------------- |
| `DATA_QUALITY_ALERT`          | `data_quality_alert`          | DataQualityChecker   | QualityMonitor       | Log warning         |
| `QUALITY_CHECK_REQUESTED`     | `quality_check_requested`     | TrainingTrigger      | QualityMonitorDaemon | Run quality check   |
| `QUALITY_CHECK_FAILED`        | `quality_check_failed`        | QualityMonitorDaemon | TrainingTrigger      | Block training      |
| `QUALITY_SCORE_UPDATED`       | `quality_score_updated`       | DataQualityChecker   | UnifiedQualityScorer | Update weights      |
| `HIGH_QUALITY_DATA_AVAILABLE` | `high_quality_data_available` | QualityMonitorDaemon | TrainingTrigger      | Enable training     |
| `QUALITY_DEGRADED`            | `quality_degraded`            | QualityMonitorDaemon | SelfplayScheduler    | Boost quality focus |
| `TRAINING_BLOCKED_BY_QUALITY` | `training_blocked_by_quality` | TrainingTrigger      | DaemonManager        | Pause training      |

### Regression Events

| Event                 | Value                 | Emitters                 | Subscribers                         | Purpose            |
| --------------------- | --------------------- | ------------------------ | ----------------------------------- | ------------------ |
| `REGRESSION_DETECTED` | `regression_detected` | ModelPerformanceWatchdog | ModelLifecycleCoordinator           | Track regression   |
| `REGRESSION_MINOR`    | `regression_minor`    | ModelPerformanceWatchdog | FeedbackLoop                        | Minor adjustment   |
| `REGRESSION_MODERATE` | `regression_moderate` | ModelPerformanceWatchdog | FeedbackLoop                        | Increase scrutiny  |
| `REGRESSION_SEVERE`   | `regression_severe`   | ModelPerformanceWatchdog | RecoveryOrchestrator                | Consider rollback  |
| `REGRESSION_CRITICAL` | `regression_critical` | ModelPerformanceWatchdog | DaemonManager, RecoveryOrchestrator | Immediate rollback |
| `REGRESSION_CLEARED`  | `regression_cleared`  | ModelPerformanceWatchdog | ModelLifecycleCoordinator           | Resume normal ops  |

### Cluster Health Events

| Event                    | Value                    | Emitters                | Subscribers                   | Purpose            |
| ------------------------ | ------------------------ | ----------------------- | ----------------------------- | ------------------ |
| `HOST_ONLINE`            | `host_online`            | P2POrchestrator         | ClusterMonitor, SyncRouter    | Track availability |
| `HOST_OFFLINE`           | `host_offline`           | P2POrchestrator         | ClusterMonitor, SyncRouter    | Handle failure     |
| `NODE_UNHEALTHY`         | `node_unhealthy`         | HealthCheckOrchestrator | UnifiedHealthManager          | Trigger recovery   |
| `NODE_RECOVERED`         | `node_recovered`         | RecoveryOrchestrator    | SyncRouter, SelfplayScheduler | Resume operations  |
| `NODE_OVERLOADED`        | `node_overloaded`        | ResourceMonitor         | JobManager                    | Redistribute work  |
| `P2P_CLUSTER_HEALTHY`    | `p2p_cluster_healthy`    | P2POrchestrator         | DaemonManager                 | Normal operations  |
| `P2P_CLUSTER_UNHEALTHY`  | `p2p_cluster_unhealthy`  | P2POrchestrator         | RecoveryOrchestrator          | Emergency mode     |
| `CLUSTER_STALL_DETECTED` | `cluster_stall_detected` | ClusterWatchdog         | RecoveryOrchestrator          | Investigate stall  |

### Resource Events

| Event                      | Value                      | Emitters            | Subscribers                   | Purpose             |
| -------------------------- | -------------------------- | ------------------- | ----------------------------- | ------------------- |
| `CLUSTER_CAPACITY_CHANGED` | `cluster_capacity_changed` | ClusterMonitor      | ResourceMonitoringCoordinator | Adjust allocations  |
| `NODE_CAPACITY_UPDATED`    | `node_capacity_updated`    | NodeHealthMonitor   | UtilizationOptimizer          | Update capacity map |
| `BACKPRESSURE_ACTIVATED`   | `backpressure_activated`   | BackpressureMonitor | SelfplayScheduler             | Reduce generation   |
| `BACKPRESSURE_RELEASED`    | `backpressure_released`    | BackpressureMonitor | SelfplayScheduler             | Resume normal rate  |
| `IDLE_RESOURCE_DETECTED`   | `idle_resource_detected`   | IdleResourceDaemon  | SelfplayScheduler             | Spawn work          |
| `DISK_SPACE_LOW`           | `disk_space_low`           | DiskSpaceManager    | CleanupDaemon                 | Trigger cleanup     |

Backpressure events include a `level` string from `BackpressureLevel` in
`app/coordination/types.py`. Queue-based backpressure emits `none/soft/hard/stop`,
while resource-based backpressure emits `none/low/medium/high/critical`.

### Work Queue Events

| Event            | Value            | Emitters       | Subscribers          | Purpose            |
| ---------------- | ---------------- | -------------- | -------------------- | ------------------ |
| `WORK_QUEUED`    | `work_queued`    | QueuePopulator | WorkerNodes          | New work available |
| `WORK_CLAIMED`   | `work_claimed`   | WorkerNode     | QueueManager         | Track assignment   |
| `WORK_COMPLETED` | `work_completed` | WorkerNode     | QueueManager         | Update statistics  |
| `WORK_FAILED`    | `work_failed`    | WorkerNode     | RecoveryOrchestrator | Handle failure     |
| `WORK_TIMEOUT`   | `work_timeout`   | QueueManager   | RecoveryOrchestrator | Reassign work      |
| `JOB_PREEMPTED`  | `job_preempted`  | JobManager     | WorkerNode           | Stop current work  |

### Leader Election Events

| Event             | Value             | Emitters        | Subscribers           | Purpose           |
| ----------------- | ----------------- | --------------- | --------------------- | ----------------- |
| `LEADER_ELECTED`  | `leader_elected`  | P2POrchestrator | LeadershipCoordinator | Assume leadership |
| `LEADER_LOST`     | `leader_lost`     | P2POrchestrator | LeadershipCoordinator | Start election    |
| `LEADER_STEPDOWN` | `leader_stepdown` | P2POrchestrator | LeadershipCoordinator | Graceful handoff  |

### Daemon Lifecycle Events

| Event                   | Value                   | Emitters         | Subscribers             | Purpose         |
| ----------------------- | ----------------------- | ---------------- | ----------------------- | --------------- |
| `DAEMON_STARTED`        | `daemon_started`        | DaemonManager    | HealthCheckOrchestrator | Track lifecycle |
| `DAEMON_STOPPED`        | `daemon_stopped`        | DaemonManager    | HealthCheckOrchestrator | Track lifecycle |
| `DAEMON_STATUS_CHANGED` | `daemon_status_changed` | DaemonWatchdog   | DaemonManager           | Auto-restart    |
| `COORDINATOR_HEARTBEAT` | `coordinator_heartbeat` | All Coordinators | DaemonWatchdog          | Liveness check  |

---

## Event Flow Diagrams

### Training Pipeline Flow

```
SELFPLAY_COMPLETE
       │
       ▼
NEW_GAMES_AVAILABLE ──► DATA_SYNC_STARTED ──► DATA_SYNC_COMPLETED
                                                      │
                                                      ▼
                                              [Export to NPZ]
                                                      │
                                                      ▼
                        TRAINING_THRESHOLD_REACHED ◄──┘
                                    │
                                    ▼
                           TRAINING_STARTED
                                    │
                                    ▼
                          TRAINING_COMPLETED
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          EVALUATION_STARTED            MODEL_DISTRIBUTION_STARTED
                    │                               │
                    ▼                               ▼
          EVALUATION_COMPLETED          MODEL_DISTRIBUTION_COMPLETE
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  PROMOTION_CANDIDATE    CURRICULUM_REBALANCED
        │
        ▼
  MODEL_PROMOTED
```

### Feedback Loop Flow

```
EVALUATION_COMPLETED
        │
        ├──► GauntletFeedbackController
        │           │
        │           ├──► Strong: HYPERPARAMETER_UPDATED (reduce exploration)
        │           ├──► Weak: SELFPLAY_TARGET_UPDATED (more games)
        │           └──► Plateau: CURRICULUM_ADVANCED
        │
        └──► CurriculumIntegration
                    │
                    └──► CURRICULUM_REBALANCED ──► SelfplayScheduler
```

### Recovery Flow

```
NODE_UNHEALTHY
       │
       ▼
UnifiedHealthManager
       │
       ├──► Degraded: Log warning
       ├──► Unhealthy: RECOVERY_INITIATED
       └──► Evicted: HOST_OFFLINE
               │
               ▼
         RecoveryOrchestrator
               │
               ├──► Restart daemon
               ├──► Failover to peer
               └──► Escalate to operator
                       │
                       ▼
                 RECOVERY_COMPLETED or RECOVERY_FAILED
```

---

## Event Subscription Order

**Critical**: Subscribers must start before emitters to receive events.

The correct startup order (enforced by `master_loop.py`):

1. `EVENT_ROUTER` - Core event bus
2. `FEEDBACK_LOOP` - Receives TRAINING_COMPLETED
3. `DATA_PIPELINE` - Receives SYNC_COMPLETED
4. `AUTO_SYNC` - Emits SYNC_COMPLETED
5. Other daemons...

If events are "lost", check that:

- Subscriber started before emitter
- Event type string matches exactly (use enum, not `.value`)
- Handler doesn't throw exceptions

---

## Debugging Events

### Enable Event Tracing

```bash
export RINGRIFT_EVENT_TRACE=1
```

This logs all events to the coordination log:

```
[EventRouter] Published: TRAINING_COMPLETED (source=TrainingCoordinator)
[EventRouter] Delivered to: FeedbackLoopController, DataPipelineOrchestrator
```

### Check Event Subscriptions

```python
from app.coordination.event_router import get_event_bus

bus = get_event_bus()
print(bus.get_subscription_count())  # Returns dict of event_type -> count
```

### Common Issues

1. **Event not received**
   - Check subscription order (subscriber must start before emitter)
   - Verify event type matches exactly (use `DataEventType.X`, not `"x"`)

2. **Duplicate events**
   - Check deduplication SHA256 hash in router
   - Verify handler isn't registered multiple times

3. **Handler exceptions**
   - Events are delivered even if previous handler threw
   - Check logs for `Handler error` messages

4. **Wrong event type**
   - Some handlers expect `event.payload`, others expect raw dict
   - Check handler signature matches event structure

---

## Best Practices

### 1. Use HandlerBase for New Handlers

```python
class MyHandler(HandlerBase):
    def _get_subscriptions(self):
        return {DataEventType.X: self._on_x}
```

Benefits:

- Automatic subscription lifecycle
- Built-in error counting
- Standard health check

### 2. Use Typed Emitters When Available

```python
# Prefer this:
from app.coordination.event_emitters import emit_training_completed
await emit_training_completed(config_key="hex8_2p", model_path="/path")

# Over this:
await bus.publish(DataEvent(event_type=DataEventType.TRAINING_COMPLETED, ...))
```

### 3. Include Source in Events

```python
DataEvent(
    event_type=DataEventType.ERROR,
    payload={"message": "Something failed"},
    source="MyComponent",  # For debugging
)
```

### 4. Handle Errors in Handlers

```python
async def my_handler(event):
    try:
        # Process event
        pass
    except Exception as e:
        logger.error(f"Handler failed: {e}")
        # Don't re-raise - other handlers should still receive event
```

---

## See Also

- `app/distributed/data_events.py` - Event type definitions
- `app/coordination/event_router.py` - Unified router implementation
- `app/coordination/handler_base.py` - Canonical base class for handlers
- `app/coordination/event_emitters.py` - Typed emitter functions
- `docs/DAEMON_REGISTRY.md` - Daemon lifecycle documentation
