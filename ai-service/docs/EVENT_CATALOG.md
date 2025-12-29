# Event Catalog

This document catalogs all events in the RingRift AI-Service event system.

**Last Updated**: December 29, 2025

## Overview

Events are defined in `app/distributed/data_events.py` as `DataEventType` enum values (118 total).
The unified event router (`app/coordination/event_router.py`) handles publishing and subscribing.

## Critical Events (Pipeline Flow)

These events form the main training pipeline:

| Event                  | Emitters                         | Subscribers                                  | Effect                                    |
| ---------------------- | -------------------------------- | -------------------------------------------- | ----------------------------------------- |
| `SELFPLAY_COMPLETE`    | P2P orchestrator, SelfplayRunner | SelfplayScheduler, QueuePopulator            | Updates game counts, triggers sync        |
| `DATA_SYNC_COMPLETED`  | AutoSyncDaemon, P2P orchestrator | TrainingCoordinator, PipelineOrchestrator    | Enables training on fresh data            |
| `TRAINING_COMPLETED`   | TrainingCoordinator              | SelfplayScheduler, EvaluationDaemon          | Triggers model evaluation                 |
| `EVALUATION_COMPLETED` | GauntletRunner, EvaluationDaemon | PromotionController, CurriculumIntegration   | Enables model promotion                   |
| `MODEL_PROMOTED`       | PromotionController              | UnifiedDistributionDaemon, SelfplayScheduler | Distributes new model, updates curriculum |

---

## Data Collection Events

| Event                 | Emitter                     | Subscribers                            | Purpose                           |
| --------------------- | --------------------------- | -------------------------------------- | --------------------------------- |
| `NEW_GAMES_AVAILABLE` | DataPipelineOrchestrator    | SelfplayScheduler, ExportScheduler     | Signal new training data ready    |
| `DATA_SYNC_STARTED`   | AutoSyncDaemon, SyncPlanner | DataPipelineOrchestrator               | Track sync progress               |
| `DATA_SYNC_COMPLETED` | AutoSyncDaemon, SyncPlanner | DataPipelineOrchestrator               | Trigger NPZ export after sync     |
| `DATA_SYNC_FAILED`    | AutoSyncDaemon              | DataPipelineOrchestrator, AlertManager | Handle sync failures              |
| `GAME_SYNCED`         | AutoSyncDaemon              | DataPipelineOrchestrator               | Individual game sync notification |

## Data Freshness Events

| Event            | Emitter           | Subscribers                 | Purpose                          |
| ---------------- | ----------------- | --------------------------- | -------------------------------- |
| `DATA_STALE`     | TrainingFreshness | SyncRouter, TrainingTrigger | Training data is stale           |
| `DATA_FRESH`     | TrainingFreshness | TrainingTrigger             | Training data is fresh           |
| `SYNC_TRIGGERED` | SyncRouter        | AutoSyncDaemon              | Sync triggered due to stale data |
| `SYNC_REQUEST`   | SyncRouter        | AutoSyncDaemon              | Explicit sync request            |

## Data Consolidation Events

| Event                    | Emitter                 | Subscribers              | Purpose                      |
| ------------------------ | ----------------------- | ------------------------ | ---------------------------- |
| `CONSOLIDATION_STARTED`  | DataConsolidationDaemon | DaemonManager            | Track consolidation state    |
| `CONSOLIDATION_COMPLETE` | DataConsolidationDaemon | DataPipelineOrchestrator | Games merged to canonical DB |

---

## Cluster Health Events

| Event            | Emitter         | Subscribers                | Purpose                        |
| ---------------- | --------------- | -------------------------- | ------------------------------ |
| `SWIM_RECOVERED` | MembershipMixin | Metrics/Alerting/Observers | SWIM membership auto-recovered |

---

## Training Events

| Event                         | Emitter                  | Subscribers                                   | Purpose                               |
| ----------------------------- | ------------------------ | --------------------------------------------- | ------------------------------------- |
| `TRAINING_THRESHOLD_REACHED`  | DataPipelineOrchestrator | TrainingTrigger                               | Sufficient data for training          |
| `TRAINING_STARTED`            | TrainingCoordinator      | SyncRouter, IdleShutdown, DataPipeline        | Pause idle detection, prioritize sync |
| `TRAINING_PROGRESS`           | TrainingCoordinator      | MetricsAnalysisOrchestrator                   | Track training progress               |
| `TRAINING_COMPLETED`          | TrainingCoordinator      | FeedbackLoop, DataPipeline, ModelDistribution | Trigger evaluation pipeline           |
| `TRAINING_FAILED`             | TrainingCoordinator      | AlertManager, DataPipeline                    | Handle training failures              |
| `TRAINING_EARLY_STOPPED`      | TrainingCoordinator      | FeedbackLoop                                  | Early stopping triggered              |
| `TRAINING_ROLLBACK_NEEDED`    | RegressionDetector       | TrainingCoordinator                           | Rollback to previous checkpoint       |
| `TRAINING_ROLLBACK_COMPLETED` | TrainingCoordinator      | FeedbackLoop                                  | Rollback completed                    |

## Evaluation Events

| Event                  | Emitter          | Subscribers                         | Purpose                   |
| ---------------------- | ---------------- | ----------------------------------- | ------------------------- |
| `EVALUATION_STARTED`   | EvaluationDaemon | MetricsAnalysisOrchestrator         | Evaluation begins         |
| `EVALUATION_PROGRESS`  | GameGauntlet     | MetricsAnalysisOrchestrator         | Real-time eval progress   |
| `EVALUATION_COMPLETED` | GameGauntlet     | FeedbackLoop, CurriculumIntegration | Adjust curriculum weights |
| `EVALUATION_FAILED`    | EvaluationDaemon | AlertManager                        | Handle eval failures      |
| `ELO_UPDATED`          | EloSyncManager   | CurriculumIntegration               | Elo rating updated        |

## Promotion Events

| Event                   | Emitter                   | Subscribers                             | Purpose                              |
| ----------------------- | ------------------------- | --------------------------------------- | ------------------------------------ |
| `PROMOTION_CANDIDATE`   | EvaluationDaemon          | AutoPromotionDaemon                     | Model is promotion candidate         |
| `PROMOTION_STARTED`     | AutoPromotionDaemon       | DataPipeline                            | Promotion process started            |
| `MODEL_PROMOTED`        | PromotionController       | UnifiedDistributionDaemon, FeedbackLoop | Distribute model to cluster          |
| `PROMOTION_FAILED`      | AutoPromotionDaemon       | ModelLifecycleCoordinator, DataPipeline | Track failed promotions              |
| `PROMOTION_REJECTED`    | AutoPromotionDaemon       | FeedbackLoop                            | Promotion rejected (below threshold) |
| `PROMOTION_ROLLED_BACK` | ModelLifecycleCoordinator | FeedbackLoop                            | Promotion rolled back                |
| `MODEL_UPDATED`         | ModelRegistry             | UnifiedDistributionDaemon               | Sync model metadata to nodes         |

---

## Curriculum Events

| Event                    | Emitter               | Subscribers                             | Purpose                            |
| ------------------------ | --------------------- | --------------------------------------- | ---------------------------------- |
| `CURRICULUM_REBALANCED`  | CurriculumIntegration | SelfplayScheduler, SelfplayOrchestrator | Update selfplay allocation weights |
| `CURRICULUM_ADVANCED`    | CurriculumIntegration | FeedbackLoopController                  | Move to harder curriculum tier     |
| `WEIGHT_UPDATED`         | CurriculumIntegration | SelfplayScheduler                       | Curriculum weight change           |
| `ELO_SIGNIFICANT_CHANGE` | EloSyncManager        | CurriculumIntegration, DataPipeline     | Trigger curriculum rebalancing     |

## Selfplay Feedback Events

| Event                         | Emitter                | Subscribers              | Purpose                        |
| ----------------------------- | ---------------------- | ------------------------ | ------------------------------ |
| `SELFPLAY_COMPLETE`           | SelfplayRunner         | DataPipelineOrchestrator | Selfplay batch finished        |
| `SELFPLAY_TARGET_UPDATED`     | FeedbackLoopController | SelfplayScheduler        | Request more/fewer games       |
| `SELFPLAY_RATE_CHANGED`       | SelfplayScheduler      | IdleResourceDaemon       | Rate multiplier changed (>20%) |
| `SELFPLAY_ALLOCATION_UPDATED` | SelfplayScheduler      | IdleResourceDaemon       | Allocation changed             |

## Batch Scheduling Events

| Event              | Emitter           | Subscribers                 | Purpose                |
| ------------------ | ----------------- | --------------------------- | ---------------------- |
| `BATCH_SCHEDULED`  | SelfplayScheduler | MetricsAnalysisOrchestrator | Batch of jobs selected |
| `BATCH_DISPATCHED` | JobManager        | MetricsAnalysisOrchestrator | Batch sent to nodes    |

---

## Optimization Events

| Event                     | Emitter                | Subscribers                                | Purpose                        |
| ------------------------- | ---------------------- | ------------------------------------------ | ------------------------------ |
| `CMAES_TRIGGERED`         | ImprovementOptimizer   | TrainingCoordinator                        | CMA-ES optimization triggered  |
| `CMAES_COMPLETED`         | ImprovementOptimizer   | FeedbackLoop                               | CMA-ES optimization completed  |
| `PLATEAU_DETECTED`        | FeedbackLoopController | ImprovementOptimizer                       | Training plateau detected      |
| `HYPERPARAMETER_UPDATED`  | ImprovementOptimizer   | EvaluationFeedbackHandler                  | Hyperparams changed            |
| `ELO_VELOCITY_CHANGED`    | QueuePopulator         | SelfplayScheduler, UnifiedFeedback         | Adjust selfplay rate           |
| `ADAPTIVE_PARAMS_CHANGED` | ImprovementOptimizer   | EvaluationFeedbackHandler, TrainingTrigger | Apply real-time LR adjustments |

## PBT Events (Population-Based Training)

| Event                     | Emitter         | Subscribers                 | Purpose                 |
| ------------------------- | --------------- | --------------------------- | ----------------------- |
| `PBT_STARTED`             | PBTOrchestrator | MetricsAnalysisOrchestrator | PBT training started    |
| `PBT_GENERATION_COMPLETE` | PBTOrchestrator | MetricsAnalysisOrchestrator | PBT generation finished |
| `PBT_COMPLETED`           | PBTOrchestrator | FeedbackLoop                | PBT training completed  |

## NAS Events (Neural Architecture Search)

| Event                     | Emitter         | Subscribers                 | Purpose                 |
| ------------------------- | --------------- | --------------------------- | ----------------------- |
| `NAS_TRIGGERED`           | NASOrchestrator | TrainingCoordinator         | NAS search triggered    |
| `NAS_STARTED`             | NASOrchestrator | MetricsAnalysisOrchestrator | NAS search started      |
| `NAS_GENERATION_COMPLETE` | NASOrchestrator | MetricsAnalysisOrchestrator | NAS generation finished |
| `NAS_COMPLETED`           | NASOrchestrator | FeedbackLoop                | NAS search completed    |
| `NAS_BEST_ARCHITECTURE`   | NASOrchestrator | ModelRegistry               | Best architecture found |

## PER Events (Prioritized Experience Replay)

| Event                    | Emitter   | Subscribers         | Purpose                   |
| ------------------------ | --------- | ------------------- | ------------------------- |
| `PER_BUFFER_REBUILT`     | PERBuffer | TrainingCoordinator | Experience buffer rebuilt |
| `PER_PRIORITIES_UPDATED` | PERBuffer | TrainingCoordinator | Priorities recalculated   |

---

## Tier Gating Events

| Event                  | Emitter    | Subscribers           | Purpose                 |
| ---------------------- | ---------- | --------------------- | ----------------------- |
| `TIER_PROMOTION`       | TierGating | CurriculumIntegration | Tier promotion achieved |
| `CROSSBOARD_PROMOTION` | TierGating | CurriculumIntegration | Multi-config promotion  |

## Parity Validation Events

| Event                         | Emitter         | Subscribers              | Purpose                     |
| ----------------------------- | --------------- | ------------------------ | --------------------------- |
| `PARITY_VALIDATION_STARTED`   | ParityValidator | DataPipelineOrchestrator | Parity check started        |
| `PARITY_VALIDATION_COMPLETED` | ParityValidator | DataPipelineOrchestrator | Parity check completed      |
| `PARITY_FAILURE_RATE_CHANGED` | ParityValidator | QualityMonitorDaemon     | Parity failure rate changed |

---

## Data Quality Events

| Event                          | Emitter                  | Subscribers              | Purpose                      |
| ------------------------------ | ------------------------ | ------------------------ | ---------------------------- |
| `DATA_QUALITY_ALERT`           | QualityMonitorDaemon     | AlertManager             | Quality issue detected       |
| `QUALITY_CHECK_REQUESTED`      | DataPipelineOrchestrator | QualityMonitorDaemon     | Request on-demand check      |
| `QUALITY_CHECK_FAILED`         | QualityMonitorDaemon     | DataPipelineOrchestrator | Quality check failed         |
| `QUALITY_SCORE_UPDATED`        | QualityMonitorDaemon     | DataPipelineOrchestrator | Quality recalculated         |
| `QUALITY_DISTRIBUTION_CHANGED` | QualityMonitorDaemon     | CurriculumIntegration    | Significant quality shift    |
| `HIGH_QUALITY_DATA_AVAILABLE`  | QualityMonitorDaemon     | TrainingTrigger          | Ready for training           |
| `QUALITY_DEGRADED`             | QualityMonitorDaemon     | FeedbackLoop             | Quality below threshold      |
| `LOW_QUALITY_DATA_WARNING`     | QualityMonitorDaemon     | AlertManager             | Below warning threshold      |
| `TRAINING_BLOCKED_BY_QUALITY`  | QualityMonitorDaemon     | TrainingTrigger          | Quality too low to train     |
| `QUALITY_FEEDBACK_ADJUSTED`    | QualityMonitorDaemon     | SelfplayScheduler        | Quality feedback updated     |
| `QUALITY_PENALTY_APPLIED`      | QualityMonitorDaemon     | SelfplayScheduler        | Reduce selfplay rate         |
| `SCHEDULER_REGISTERED`         | TemperatureScheduler     | SelfplayScheduler        | Scheduler registered         |
| `EXPLORATION_BOOST`            | FeedbackLoopController   | SelfplayScheduler        | Boost exploration temp       |
| `EXPLORATION_ADJUSTED`         | FeedbackLoopController   | SelfplayScheduler        | Exploration strategy changed |
| `OPPONENT_MASTERED`            | CurriculumIntegration    | FeedbackLoop             | Advance curriculum           |

## Training Loss Monitoring Events

| Event                   | Emitter            | Subscribers  | Purpose                 |
| ----------------------- | ------------------ | ------------ | ----------------------- |
| `TRAINING_LOSS_ANOMALY` | RegressionDetector | AlertManager | Unusual loss spike/drop |
| `TRAINING_LOSS_TREND`   | RegressionDetector | FeedbackLoop | Loss trend changed      |

---

## Registry & Metrics Events

| Event               | Emitter                     | Subscribers               | Purpose           |
| ------------------- | --------------------------- | ------------------------- | ----------------- |
| `REGISTRY_UPDATED`  | ModelRegistry               | UnifiedDistributionDaemon | Registry changed  |
| `METRICS_UPDATED`   | MetricsAnalysisOrchestrator | Dashboard                 | Metrics updated   |
| `CACHE_INVALIDATED` | CacheCoordinator            | DataPipelineOrchestrator  | Cache invalidated |

## Regression Detection Events

| Event                 | Emitter                  | Subscribers                             | Purpose                         |
| --------------------- | ------------------------ | --------------------------------------- | ------------------------------- |
| `REGRESSION_DETECTED` | ModelPerformanceWatchdog | ModelLifecycleCoordinator, DataPipeline | Any regression detected         |
| `REGRESSION_MINOR`    | RegressionDetector       | FeedbackLoop                            | Minor regression                |
| `REGRESSION_MODERATE` | RegressionDetector       | FeedbackLoop                            | Moderate regression             |
| `REGRESSION_SEVERE`   | RegressionDetector       | ModelLifecycleCoordinator               | Severe regression               |
| `REGRESSION_CRITICAL` | RegressionDetector       | DaemonManager                           | Critical - rollback recommended |
| `REGRESSION_CLEARED`  | RegressionDetector       | FeedbackLoop                            | Model recovered                 |

---

## P2P/Model Sync Events

| Event                         | Emitter                   | Subscribers                 | Purpose                |
| ----------------------------- | ------------------------- | --------------------------- | ---------------------- |
| `MODEL_SYNC_REQUESTED`        | SelfplayScheduler         | UnifiedDistributionDaemon   | Model sync requested   |
| `MODEL_DISTRIBUTION_STARTED`  | UnifiedDistributionDaemon | MetricsAnalysisOrchestrator | Distribution initiated |
| `MODEL_DISTRIBUTION_COMPLETE` | UnifiedDistributionDaemon | SelfplayScheduler           | Distribution completed |
| `MODEL_DISTRIBUTION_FAILED`   | UnifiedDistributionDaemon | AlertManager                | Distribution failed    |
| `SYNC_STALLED`                | AutoSyncDaemon            | AlertManager                | Sync operation stalled |
| `SYNC_CHECKSUM_FAILED`        | AutoSyncDaemon            | AlertManager                | Checksum mismatch      |
| `SYNC_NODE_UNREACHABLE`       | AutoSyncDaemon            | NodeRecoveryDaemon          | Node unreachable       |

## Orphan Detection Events

| Event                     | Emitter                  | Subscribers                 | Purpose                  |
| ------------------------- | ------------------------ | --------------------------- | ------------------------ |
| `ORPHAN_GAMES_DETECTED`   | OrphanDetectionDaemon    | DataPipelineOrchestrator    | Unregistered games found |
| `ORPHAN_GAMES_REGISTERED` | DataPipelineOrchestrator | MetricsAnalysisOrchestrator | Orphans auto-registered  |

## Replication Repair Events

| Event               | Emitter                  | Subscribers              | Purpose                  |
| ------------------- | ------------------------ | ------------------------ | ------------------------ |
| `REPAIR_COMPLETED`  | UnifiedReplicationDaemon | DataPipelineOrchestrator | Repair job succeeded     |
| `REPAIR_FAILED`     | UnifiedReplicationDaemon | AlertManager             | Repair job failed        |
| `REPLICATION_ALERT` | UnifiedReplicationDaemon | AlertManager             | Replication health alert |

## Database Lifecycle Events

| Event              | Emitter      | Subscribers              | Purpose              |
| ------------------ | ------------ | ------------------------ | -------------------- |
| `DATABASE_CREATED` | GameReplayDB | DataPipelineOrchestrator | New database created |

---

## System Events

| Event                       | Emitter         | Subscribers                 | Purpose                |
| --------------------------- | --------------- | --------------------------- | ---------------------- |
| `DAEMON_STARTED`            | DaemonManager   | MetricsAnalysisOrchestrator | Daemon started         |
| `DAEMON_STOPPED`            | DaemonManager   | MetricsAnalysisOrchestrator | Daemon stopped         |
| `DAEMON_STATUS_CHANGED`     | DaemonWatchdog  | AlertManager                | Daemon status change   |
| `DAEMON_PERMANENTLY_FAILED` | DaemonManager   | AlertManager                | Exceeded restart limit |
| `HOST_ONLINE`               | P2POrchestrator | UnifiedHealthManager        | Node came online       |
| `HOST_OFFLINE`              | P2POrchestrator | UnifiedHealthManager        | Node went offline      |
| `ERROR`                     | Various         | AlertManager                | General error event    |

## Health & Recovery Events

| Event                 | Emitter                 | Subscribers                 | Purpose                      |
| --------------------- | ----------------------- | --------------------------- | ---------------------------- |
| `HEALTH_CHECK_PASSED` | HealthCheckOrchestrator | MetricsAnalysisOrchestrator | Node healthy                 |
| `HEALTH_CHECK_FAILED` | HealthCheckOrchestrator | NodeRecoveryDaemon          | Node unhealthy               |
| `HEALTH_ALERT`        | HealthCheckOrchestrator | AlertManager                | General health warning       |
| `RESOURCE_CONSTRAINT` | ResourceTargetManager   | IdleResourceDaemon          | CPU/GPU/Memory/Disk pressure |
| `NODE_OVERLOADED`     | ResourceTargetManager   | SelfplayScheduler           | Node overloaded              |
| `RECOVERY_INITIATED`  | NodeRecoveryDaemon      | MetricsAnalysisOrchestrator | Auto-recovery started        |
| `RECOVERY_COMPLETED`  | NodeRecoveryDaemon      | MetricsAnalysisOrchestrator | Auto-recovery finished       |
| `RECOVERY_FAILED`     | NodeRecoveryDaemon      | AlertManager                | Auto-recovery failed         |

---

## Work Queue Events

| Event            | Emitter               | Subscribers                 | Purpose                 |
| ---------------- | --------------------- | --------------------------- | ----------------------- |
| `WORK_QUEUED`    | UnifiedQueuePopulator | MetricsAnalysisOrchestrator | Work added to queue     |
| `WORK_CLAIMED`   | WorkQueue             | MetricsAnalysisOrchestrator | Work claimed by node    |
| `WORK_STARTED`   | JobManager            | MetricsAnalysisOrchestrator | Work execution started  |
| `WORK_COMPLETED` | JobManager            | DataPipelineOrchestrator    | Work completed          |
| `WORK_FAILED`    | JobManager            | AlertManager                | Work failed permanently |
| `WORK_RETRY`     | JobManager            | MetricsAnalysisOrchestrator | Work failed, will retry |
| `WORK_TIMEOUT`   | JobReaperLoop         | AlertManager                | Work timed out          |
| `WORK_CANCELLED` | JobManager            | MetricsAnalysisOrchestrator | Work cancelled          |
| `JOB_PREEMPTED`  | JobManager            | MetricsAnalysisOrchestrator | Job preempted           |

## Cluster Status Events

| Event                    | Emitter                 | Subscribers                   | Purpose               |
| ------------------------ | ----------------------- | ----------------------------- | --------------------- |
| `CLUSTER_STATUS_CHANGED` | ClusterMonitor          | DaemonManager                 | Cluster status change |
| `CLUSTER_STALL_DETECTED` | ClusterWatchdog         | AlertManager                  | Node(s) stuck         |
| `NODE_UNHEALTHY`         | HealthCheckOrchestrator | NodeRecoveryDaemon            | Node marked unhealthy |
| `NODE_RECOVERED`         | NodeRecoveryDaemon      | SelfplayScheduler, SyncRouter | Node recovered        |
| `NODE_ACTIVATED`         | ClusterActivator        | SelfplayScheduler             | Node activated        |
| `NODE_TERMINATED`        | IdleShutdownDaemon      | SelfplayScheduler             | Node terminated       |

---

## Lock/Synchronization Events

| Event               | Emitter         | Subscribers  | Purpose                  |
| ------------------- | --------------- | ------------ | ------------------------ |
| `LOCK_TIMEOUT`      | DistributedLock | AlertManager | Lock acquisition timeout |
| `DEADLOCK_DETECTED` | DistributedLock | AlertManager | Deadlock detected        |

## Checkpoint Events

| Event               | Emitter             | Subscribers   | Purpose           |
| ------------------- | ------------------- | ------------- | ----------------- |
| `CHECKPOINT_SAVED`  | TrainingCoordinator | ModelRegistry | Checkpoint saved  |
| `CHECKPOINT_LOADED` | TrainingCoordinator | FeedbackLoop  | Checkpoint loaded |

## Backup Events

| Event                   | Emitter                | Subscribers                 | Purpose                  |
| ----------------------- | ---------------------- | --------------------------- | ------------------------ |
| `DATA_BACKUP_COMPLETED` | CoordinatorDiskManager | MetricsAnalysisOrchestrator | External backup finished |

## CPU Pipeline Events

| Event                        | Emitter                 | Subscribers              | Purpose               |
| ---------------------------- | ----------------------- | ------------------------ | --------------------- |
| `CPU_PIPELINE_JOB_COMPLETED` | CPUPipelineOrchestrator | DataPipelineOrchestrator | Vast CPU job finished |

---

## Task Lifecycle Events

| Event            | Emitter       | Subscribers                 | Purpose                      |
| ---------------- | ------------- | --------------------------- | ---------------------------- |
| `TASK_SPAWNED`   | JobManager    | MetricsAnalysisOrchestrator | Task spawned                 |
| `TASK_HEARTBEAT` | JobManager    | JobReaperLoop               | Task heartbeat               |
| `TASK_COMPLETED` | JobManager    | DataPipelineOrchestrator    | Task completed               |
| `TASK_FAILED`    | JobManager    | AlertManager                | Task failed                  |
| `TASK_ORPHANED`  | JobReaperLoop | NodeRecoveryDaemon          | Task orphaned                |
| `TASK_CANCELLED` | JobManager    | MetricsAnalysisOrchestrator | Task cancelled               |
| `TASK_ABANDONED` | JobManager    | SelfplayOrchestrator        | Task intentionally abandoned |

## Capacity/Resource Events

| Event                          | Emitter                 | Subscribers       | Purpose                  |
| ------------------------------ | ----------------------- | ----------------- | ------------------------ |
| `CLUSTER_CAPACITY_CHANGED`     | ClusterMonitor          | SelfplayScheduler | Cluster capacity changed |
| `NODE_CAPACITY_UPDATED`        | HealthCheckOrchestrator | SelfplayScheduler | Node capacity updated    |
| `BACKPRESSURE_ACTIVATED`       | BackpressureMonitor     | SyncRouter        | Backpressure activated   |
| `BACKPRESSURE_RELEASED`        | BackpressureMonitor     | SyncRouter        | Backpressure released    |
| `IDLE_RESOURCE_DETECTED`       | IdleResourceDaemon      | SelfplayScheduler | Idle GPU/CPU detected    |
| `RESOURCE_CONSTRAINT_DETECTED` | ResourceTargetManager   | SelfplayScheduler | Resource limit hit       |

---

## Leader Election Events

| Event                  | Emitter         | Subscribers           | Purpose                   |
| ---------------------- | --------------- | --------------------- | ------------------------- |
| `LEADER_ELECTED`       | P2POrchestrator | LeadershipCoordinator | Node became leader        |
| `LEADER_LOST`          | P2POrchestrator | LeadershipCoordinator | Lost leadership           |
| `LEADER_STEPDOWN`      | P2POrchestrator | LeadershipCoordinator | Leader stepping down      |
| `SPLIT_BRAIN_DETECTED` | P2POrchestrator | AlertManager          | Multiple leaders detected |
| `SPLIT_BRAIN_RESOLVED` | P2POrchestrator | LeadershipCoordinator | Split-brain resolved      |

## State Persistence Events

| Event             | Emitter         | Subscribers           | Purpose                     |
| ----------------- | --------------- | --------------------- | --------------------------- |
| `STATE_PERSISTED` | P2POrchestrator | StateManager          | P2P state saved to database |
| `EPOCH_ADVANCED`  | P2POrchestrator | LeadershipCoordinator | Cluster epoch incremented   |

---

## Error Recovery & Resilience Events

| Event                         | Emitter       | Subscribers                 | Purpose                 |
| ----------------------------- | ------------- | --------------------------- | ----------------------- |
| `MODEL_CORRUPTED`             | ModelRegistry | AlertManager                | Model file corruption   |
| `COORDINATOR_HEALTHY`         | DaemonManager | MetricsAnalysisOrchestrator | Coordinator healthy     |
| `COORDINATOR_UNHEALTHY`       | DaemonManager | AlertManager                | Coordinator unhealthy   |
| `COORDINATOR_HEALTH_DEGRADED` | DaemonManager | AlertManager                | Coordinator degraded    |
| `COORDINATOR_SHUTDOWN`        | DaemonManager | MetricsAnalysisOrchestrator | Graceful shutdown       |
| `COORDINATOR_INIT_FAILED`     | DaemonManager | AlertManager                | Init failed             |
| `COORDINATOR_HEARTBEAT`       | DaemonManager | ClusterWatchdog             | Liveness signal         |
| `HANDLER_TIMEOUT`             | EventRouter   | AlertManager                | Handler timed out       |
| `HANDLER_FAILED`              | EventRouter   | AlertManager                | Handler threw exception |

## Idle State Broadcast Events

| Event                  | Emitter            | Subscribers        | Purpose                   |
| ---------------------- | ------------------ | ------------------ | ------------------------- |
| `IDLE_STATE_BROADCAST` | IdleResourceDaemon | SelfplayScheduler  | Node idle state broadcast |
| `IDLE_STATE_REQUEST`   | SelfplayScheduler  | IdleResourceDaemon | Request idle state update |

## Disk Space Management Events

| Event                    | Emitter                | Subscribers                      | Purpose                    |
| ------------------------ | ---------------------- | -------------------------------- | -------------------------- |
| `DISK_SPACE_LOW`         | DiskSpaceManagerDaemon | MaintenanceDaemon, DaemonManager | Disk usage above threshold |
| `DISK_CLEANUP_TRIGGERED` | DiskSpaceManagerDaemon | MetricsAnalysisOrchestrator      | Proactive cleanup started  |

---

## Subscribing to Events

```python
from app.coordination.event_router import subscribe, get_router

# Method 1: Direct subscription
def on_training_complete(event):
    payload = event.payload if hasattr(event, 'payload') else event
    config_key = payload.get('config_key')
    print(f"Training completed for {config_key}")

subscribe("TRAINING_COMPLETED", on_training_complete)

# Method 2: Via router instance
router = get_router()
if router:
    router.subscribe("SELFPLAY_COMPLETE", handler)

# Method 3: Via DataEventType enum (preferred)
from app.distributed.data_events import DataEventType

router.subscribe(DataEventType.SELFPLAY_COMPLETE.value, handler)
```

## Emitting Events

```python
from app.coordination.event_router import publish_async, get_router

# Method 1: Convenience function (async)
await publish_async("SELFPLAY_COMPLETE", {
    "config_key": "hex8_2p",
    "games_completed": 100,
    "quality_score": 0.85,
})

# Method 2: Via router instance
router = get_router()
if router:
    await router.publish_async("MODEL_PROMOTED", {
        "config_key": "hex8_2p",
        "model_path": "models/canonical_hex8_2p.pth",
    })

# Method 3: Typed emitters (preferred)
from app.coordination.event_emitters import emit_training_complete

emit_training_complete(
    config_key="hex8_2p",
    model_path="models/canonical_hex8_2p.pth",
    epochs_completed=50,
)
```

## Event System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ SelfplayRunner  │────▶│  EventRouter    │────▶│ SelfplayScheduler│
│                 │     │  (event_router) │     │                  │
│ emits:          │     │                 │     │ subscribes:      │
│ SELFPLAY_COMPLETE     │ deduplication   │     │ SELFPLAY_COMPLETE│
└─────────────────┘     │ SHA256 hash     │     │ ELO_VELOCITY_CHG │
                        └─────────────────┘     │ QUALITY_DEGRADED │
                               │                └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │ DataEventBus    │
                        │ (data_events)   │
                        │                 │
                        │ cross-process   │
                        │ SQLite-backed   │
                        └─────────────────┘
```

## Adding a New Event Type

1. Add to `DataEventType` enum in `app/distributed/data_events.py`
2. Document in this catalog
3. Add typed emitter in `app/coordination/event_emitters.py` (optional)
4. Add subscriber(s) in relevant coordinators
5. Test with `bootstrap_coordination()` smoke test

## Event Count Summary

| Category                    | Count    |
| --------------------------- | -------- |
| Data Collection & Freshness | 9        |
| Data Consolidation          | 2        |
| Training                    | 8        |
| Evaluation                  | 5        |
| Promotion                   | 7        |
| Curriculum                  | 4        |
| Selfplay Feedback           | 4        |
| Batch Scheduling            | 2        |
| Optimization                | 6        |
| PBT/NAS/PER                 | 10       |
| Tier Gating & Parity        | 5        |
| Data Quality                | 15       |
| Training Loss               | 2        |
| Registry & Metrics          | 3        |
| Regression                  | 6        |
| P2P/Model Sync              | 7        |
| Orphan Detection & Repair   | 5        |
| Database Lifecycle          | 1        |
| System                      | 7        |
| Health & Recovery           | 8        |
| Work Queue                  | 9        |
| Cluster Status              | 6        |
| Lock/Checkpoint/Backup      | 5        |
| CPU Pipeline                | 1        |
| Task Lifecycle              | 7        |
| Capacity/Resource           | 6        |
| Leader Election             | 5        |
| State Persistence           | 2        |
| Error Recovery              | 9        |
| Idle State                  | 2        |
| Disk Space                  | 2        |
| **Total**                   | **~118** |

## See Also

- `app/distributed/data_events.py` - Event type definitions
- `app/coordination/event_router.py` - Unified event router
- `app/coordination/event_emitters.py` - Typed event emitters
- `app/coordination/coordination_bootstrap.py` - Event wiring at startup
- `docs/runbooks/EVENT_WIRING_VERIFICATION.md` - Verification procedures
