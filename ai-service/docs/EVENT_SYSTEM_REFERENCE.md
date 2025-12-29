# Event System Reference

The RingRift coordination infrastructure uses an event-driven architecture with 140+ event types organized into 25+ categories. This document provides a comprehensive reference for the event system.

**Created**: December 2025 (Wave 4 Phase 2)
**Updated**: December 29, 2025 (Event wiring + payload updates)

## Overview

The event system enables loose coupling between components of the AI training pipeline:

- Data collection → Training → Evaluation → Promotion
- Cluster health monitoring → Recovery
- Selfplay scheduling → Curriculum advancement

Emitters and subscribers listed below are primary owners; some events are used
by multiple components. For a complete list, cross-check `EVENT_CATALOG.md`.

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

## Cross-Process Event Bridging

The event system supports cross-process communication via SQLite-backed queues. Events published in one process (e.g., P2P orchestrator) can be received by handlers in another process (e.g., training script).

### How It Works

1. Events are published to the in-memory `EventBus`
2. The `UnifiedEventRouter` checks if the event type is in `DATA_TO_CROSS_PROCESS_MAP`
3. If mapped, the event is also written to a SQLite queue (`coordination.db`)
4. Other processes poll this queue and deliver events to local subscribers

### Bridged Event Types (December 28, 2025)

The following 129 event types are bridged for cross-process communication:

#### Cluster Coordination Events

| Event                        | Cross-Process Key            | Purpose                            |
| ---------------------------- | ---------------------------- | ---------------------------------- |
| `MODEL_DISTRIBUTION_STARTED` | `model_distribution_started` | Model sync initiated               |
| `MODEL_DISTRIBUTION_FAILED`  | `model_distribution_failed`  | Model sync failed (triggers retry) |
| `SPLIT_BRAIN_DETECTED`       | `split_brain_detected`       | Multiple leaders detected          |
| `SPLIT_BRAIN_RESOLVED`       | `split_brain_resolved`       | Split-brain resolved               |
| `CLUSTER_STALL_DETECTED`     | `cluster_stall_detected`     | Cluster stall detected             |
| `NODE_TERMINATED`            | `node_terminated`            | Node shutdown notification         |

#### Data Integrity Events

| Event                    | Cross-Process Key        | Purpose                    |
| ------------------------ | ------------------------ | -------------------------- |
| `CONSOLIDATION_STARTED`  | `consolidation_started`  | Database merge started     |
| `CONSOLIDATION_COMPLETE` | `consolidation_complete` | Database merge finished    |
| `REPAIR_COMPLETED`       | `repair_completed`       | Data repair succeeded      |
| `REPAIR_FAILED`          | `repair_failed`          | Data repair failed         |
| `REPLICATION_ALERT`      | `replication_alert`      | Replication health warning |

#### Sync Integrity Events

| Event                   | Cross-Process Key       | Purpose                      |
| ----------------------- | ----------------------- | ---------------------------- |
| `SYNC_REQUEST`          | `sync_request`          | Explicit sync requested      |
| `SYNC_CHECKSUM_FAILED`  | `sync_checksum_failed`  | Checksum mismatch detected   |
| `SYNC_NODE_UNREACHABLE` | `sync_node_unreachable` | Node unreachable during sync |

#### Recovery Events

| Event                | Cross-Process Key    | Purpose                 |
| -------------------- | -------------------- | ----------------------- |
| `RECOVERY_INITIATED` | `recovery_initiated` | Auto-recovery started   |
| `RECOVERY_COMPLETED` | `recovery_completed` | Auto-recovery succeeded |
| `RECOVERY_FAILED`    | `recovery_failed`    | Auto-recovery failed    |

### Configuring Cross-Process Bridging

Events are mapped in `app/coordination/event_mappings.py`:

```python
from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

# Check if an event is bridged
is_bridged = "my_event" in DATA_TO_CROSS_PROCESS_MAP

# Get the cross-process key
cross_key = DATA_TO_CROSS_PROCESS_MAP.get("my_event")
```

### Adding New Bridged Events

To bridge a new event type:

1. Add the mapping to `DATA_TO_CROSS_PROCESS_MAP` in `event_mappings.py`:

   ```python
   DATA_TO_CROSS_PROCESS_MAP = {
       # ... existing mappings ...
       "my_new_event": "MY_NEW_EVENT",
   }
   ```

2. Ensure the event type exists in `DataEventType` enum in `data_events.py`

3. Test with:

   ```python
   from app.coordination.cross_process_events import get_cross_process_queue

   queue = get_cross_process_queue()
   # Publish in process A
   queue.publish("my_new_event", {"data": "value"})
   # Poll in process B
   events = queue.poll()
   ```

---

## Event Types by Category

### Data Collection Events

| Event                     | Value                     | Emitters                       | Subscribers                                 | Purpose                  |
| ------------------------- | ------------------------- | ------------------------------ | ------------------------------------------- | ------------------------ |
| `NEW_GAMES_AVAILABLE`     | `new_games`               | SelfplayRunner, AutoSyncDaemon | DataPipelineOrchestrator, ExportScheduler   | Trigger data export      |
| `DATA_SYNC_STARTED`       | `sync_started`            | SyncPlanner, AutoSyncDaemon    | DataPipelineOrchestrator                    | Track sync progress      |
| `DATA_SYNC_COMPLETED`     | `sync_completed`          | SyncPlanner, P2POrchestrator   | DataPipelineOrchestrator, SelfplayScheduler | Trigger export stage     |
| `DATA_SYNC_FAILED`        | `sync_failed`             | SyncPlanner                    | RecoveryOrchestrator                        | Retry sync               |
| `GAME_SYNCED`             | `game_synced`             | AutoSyncDaemon                 | ClusterManifest                             | Update game registry     |
| `DATA_STALE`              | `data_stale`              | TrainingFreshness              | SyncRouter                                  | Trigger priority sync    |
| `DATA_FRESH`              | `data_fresh`              | TrainingFreshness              | TrainCLI                                    | Proceed with training    |
| `SYNC_TRIGGERED`          | `sync_triggered`          | TrainingFreshness              | SyncRouter                                  | Force immediate sync     |
| `SYNC_REQUEST`            | `sync_request`            | SyncRouter                     | AutoSyncDaemon                              | Explicit sync request    |
| `CONSOLIDATION_STARTED`   | `consolidation_started`   | DatabaseConsolidator           | ProgressTracker                             | Track consolidation      |
| `CONSOLIDATION_COMPLETE`  | `consolidation_complete`  | DatabaseConsolidator           | DataPipelineOrchestrator                    | Trigger export           |
| `ORPHAN_GAMES_DETECTED`   | `orphan_games_detected`   | OrphanDetectionDaemon          | DataPipelineOrchestrator                    | Trigger priority sync    |
| `ORPHAN_GAMES_REGISTERED` | `orphan_games_registered` | DataPipelineOrchestrator       | MetricsAnalysisOrchestrator                 | Orphans auto-registered  |
| `REPAIR_COMPLETED`        | `repair_completed`        | RecoveryOrchestrator           | DataPipelineOrchestrator                    | Retrigger sync           |
| `REPAIR_FAILED`           | `repair_failed`           | RecoveryOrchestrator           | DataPipelineOrchestrator                    | Track repair failures    |
| `REPLICATION_ALERT`       | `replication_alert`       | UnifiedReplicationDaemon       | AlertManager                                | Replication health alert |
| `DATABASE_CREATED`        | `database_created`        | GameReplayDB                   | DataPipelineOrchestrator                    | New database created     |
| `TASK_ABANDONED`          | `task_abandoned`          | P2POrchestrator                | SelfplayOrchestrator                        | Track cancelled jobs     |

### Batch Scheduling Events

| Event              | Value              | Emitters          | Subscribers                 | Purpose                |
| ------------------ | ------------------ | ----------------- | --------------------------- | ---------------------- |
| `BATCH_SCHEDULED`  | `batch_scheduled`  | SelfplayScheduler | JobManager                  | Batch of jobs selected |
| `BATCH_DISPATCHED` | `batch_dispatched` | JobManager        | MetricsAnalysisOrchestrator | Batch sent to nodes    |

### Training Events

| Event                         | Value                         | Emitters             | Subscribers                                   | Purpose                |
| ----------------------------- | ----------------------------- | -------------------- | --------------------------------------------- | ---------------------- |
| `TRAINING_THRESHOLD_REACHED`  | `training_threshold`          | QueuePopulator       | TrainingTriggerDaemon                         | Schedule training      |
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
| `PROMOTION_REJECTED`    | `promotion_rejected`    | PromotionController  | FeedbackLoop, CurriculumFeedback        | Adjust training        |
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

| Event                         | Value                         | Emitters               | Subscribers                 | Purpose                 |
| ----------------------------- | ----------------------------- | ---------------------- | --------------------------- | ----------------------- |
| `SELFPLAY_COMPLETE`           | `selfplay_complete`           | SelfplayRunner         | DataPipelineOrchestrator    | Process results         |
| `SELFPLAY_TARGET_UPDATED`     | `selfplay_target_updated`     | QueuePopulator         | SelfplayScheduler           | Adjust game count       |
| `SELFPLAY_RATE_CHANGED`       | `selfplay_rate_changed`       | FeedbackLoopController | SelfplayScheduler           | Adjust generation rate  |
| `SELFPLAY_ALLOCATION_UPDATED` | `selfplay_allocation_updated` | CurriculumIntegration  | SelfplayScheduler           | Change config weights   |
| `P2P_SELFPLAY_SCALED`         | `p2p_selfplay_scaled`         | SelfplayScheduler      | MetricsAnalysisOrchestrator | Selfplay scaled up/down |

### Optimization Events

| Event                     | Value                     | Emitters               | Subscribers                                      | Purpose                        |
| ------------------------- | ------------------------- | ---------------------- | ------------------------------------------------ | ------------------------------ |
| `CMAES_TRIGGERED`         | `cmaes_triggered`         | ImprovementOptimizer   | TrainingCoordinator                              | CMA-ES optimization triggered  |
| `CMAES_COMPLETED`         | `cmaes_completed`         | ImprovementOptimizer   | FeedbackLoop                                     | CMA-ES optimization completed  |
| `NAS_TRIGGERED`           | `nas_triggered`           | NASOrchestrator        | TrainingCoordinator                              | NAS search triggered           |
| `PLATEAU_DETECTED`        | `plateau_detected`        | FeedbackLoopController | ImprovementOptimizer                             | Training plateau detected      |
| `HYPERPARAMETER_UPDATED`  | `hyperparameter_updated`  | ImprovementOptimizer   | EvaluationFeedbackHandler                        | Hyperparams changed            |
| `ADAPTIVE_PARAMS_CHANGED` | `adaptive_params_changed` | ImprovementOptimizer   | EvaluationFeedbackHandler, TrainingTriggerDaemon | Apply real-time LR adjustments |

### PBT Events

| Event                     | Value                     | Emitters        | Subscribers                 | Purpose                 |
| ------------------------- | ------------------------- | --------------- | --------------------------- | ----------------------- |
| `PBT_STARTED`             | `pbt_started`             | PBTOrchestrator | MetricsAnalysisOrchestrator | PBT training started    |
| `PBT_GENERATION_COMPLETE` | `pbt_generation_complete` | PBTOrchestrator | MetricsAnalysisOrchestrator | PBT generation finished |
| `PBT_COMPLETED`           | `pbt_completed`           | PBTOrchestrator | FeedbackLoop                | PBT training completed  |

### NAS Events

| Event                     | Value                     | Emitters        | Subscribers                 | Purpose                 |
| ------------------------- | ------------------------- | --------------- | --------------------------- | ----------------------- |
| `NAS_STARTED`             | `nas_started`             | NASOrchestrator | MetricsAnalysisOrchestrator | NAS search started      |
| `NAS_GENERATION_COMPLETE` | `nas_generation_complete` | NASOrchestrator | MetricsAnalysisOrchestrator | NAS generation finished |
| `NAS_COMPLETED`           | `nas_completed`           | NASOrchestrator | FeedbackLoop                | NAS search completed    |
| `NAS_BEST_ARCHITECTURE`   | `nas_best_architecture`   | NASOrchestrator | ModelRegistry               | Best architecture found |

### PER (Prioritized Experience Replay) Events

| Event                    | Value                    | Emitters  | Subscribers         | Purpose                   |
| ------------------------ | ------------------------ | --------- | ------------------- | ------------------------- |
| `PER_BUFFER_REBUILT`     | `per_buffer_rebuilt`     | PERBuffer | TrainingCoordinator | Experience buffer rebuilt |
| `PER_PRIORITIES_UPDATED` | `per_priorities_updated` | PERBuffer | TrainingCoordinator | Priorities recalculated   |

### Tier Gating Events

| Event                  | Value                  | Emitters   | Subscribers           | Purpose                 |
| ---------------------- | ---------------------- | ---------- | --------------------- | ----------------------- |
| `TIER_PROMOTION`       | `tier_promotion`       | TierGating | CurriculumIntegration | Tier promotion achieved |
| `CROSSBOARD_PROMOTION` | `crossboard_promotion` | TierGating | CurriculumIntegration | Multi-config promotion  |

### Parity Validation Events

| Event                         | Value                         | Emitters        | Subscribers              | Purpose                     |
| ----------------------------- | ----------------------------- | --------------- | ------------------------ | --------------------------- |
| `PARITY_VALIDATION_STARTED`   | `parity_validation_started`   | ParityValidator | DataPipelineOrchestrator | Parity check started        |
| `PARITY_VALIDATION_COMPLETED` | `parity_validation_completed` | ParityValidator | DataPipelineOrchestrator | Parity check completed      |
| `PARITY_FAILURE_RATE_CHANGED` | `parity_failure_rate_changed` | ParityValidator | QualityMonitorDaemon     | Parity failure rate changed |

### Quality Events

| Event                          | Value                          | Emitters               | Subscribers           | Purpose                               |
| ------------------------------ | ------------------------------ | ---------------------- | --------------------- | ------------------------------------- |
| `DATA_QUALITY_ALERT`           | `data_quality_alert`           | DataQualityChecker     | QualityMonitor        | Log warning                           |
| `QUALITY_CHECK_REQUESTED`      | `quality_check_requested`      | TrainingTriggerDaemon  | QualityMonitorDaemon  | Run quality check                     |
| `QUALITY_CHECK_FAILED`         | `quality_check_failed`         | QualityMonitorDaemon   | TrainingTriggerDaemon | Block training                        |
| `QUALITY_SCORE_UPDATED`        | `quality_score_updated`        | DataQualityChecker     | UnifiedQualityScorer  | Update weights                        |
| `QUALITY_DISTRIBUTION_CHANGED` | `quality_distribution_changed` | QualityMonitorDaemon   | CurriculumIntegration | Significant quality shift             |
| `HIGH_QUALITY_DATA_AVAILABLE`  | `high_quality_data_available`  | QualityMonitorDaemon   | TrainingTriggerDaemon | Enable training                       |
| `QUALITY_DEGRADED`             | `quality_degraded`             | QualityMonitorDaemon   | SelfplayScheduler     | Boost quality focus                   |
| `LOW_QUALITY_DATA_WARNING`     | `low_quality_data_warning`     | QualityMonitorDaemon   | AlertManager          | Below threshold                       |
| `TRAINING_BLOCKED_BY_QUALITY`  | `training_blocked_by_quality`  | TrainingTriggerDaemon  | DaemonManager         | Pause training                        |
| `QUALITY_FEEDBACK_ADJUSTED`    | `quality_feedback_adjusted`    | QualityMonitorDaemon   | SelfplayScheduler     | Quality feedback updated              |
| `QUALITY_PENALTY_APPLIED`      | `quality_penalty_applied`      | QualityMonitorDaemon   | SelfplayScheduler     | Reduce selfplay rate                  |
| `SCHEDULER_REGISTERED`         | `scheduler_registered`         | TemperatureScheduler   | SelfplayScheduler     | Scheduler registered                  |
| `EXPLORATION_BOOST`            | `exploration_boost`            | FeedbackLoopController | SelfplayScheduler     | Boost exploration temp                |
| `EXPLORATION_ADJUSTED`         | `exploration_adjusted`         | FeedbackLoopController | SelfplayScheduler     | Quality-driven exploration adjustment |

`EXPLORATION_ADJUSTED` payload includes:

- `config_key`
- `quality_score` (0-1)
- `trend` (improving, declining, stable)
- `position_difficulty` (normal, medium-hard, hard)
- `mcts_budget_multiplier`
- `exploration_temp_boost`

### Training Loss Monitoring Events

| Event                   | Value                   | Emitters           | Subscribers  | Purpose                 |
| ----------------------- | ----------------------- | ------------------ | ------------ | ----------------------- |
| `TRAINING_LOSS_ANOMALY` | `training_loss_anomaly` | RegressionDetector | AlertManager | Unusual loss spike/drop |
| `TRAINING_LOSS_TREND`   | `training_loss_trend`   | RegressionDetector | FeedbackLoop | Loss trend changed      |

### Registry & Metrics Events

| Event               | Value               | Emitters                    | Subscribers               | Purpose           |
| ------------------- | ------------------- | --------------------------- | ------------------------- | ----------------- |
| `REGISTRY_UPDATED`  | `registry_updated`  | ModelRegistry               | UnifiedDistributionDaemon | Registry changed  |
| `METRICS_UPDATED`   | `metrics_updated`   | MetricsAnalysisOrchestrator | Dashboard                 | Metrics updated   |
| `CACHE_INVALIDATED` | `cache_invalidated` | CacheCoordinator            | DataPipelineOrchestrator  | Cache invalidated |

### Regression Events

| Event                 | Value                 | Emitters                 | Subscribers                         | Purpose            |
| --------------------- | --------------------- | ------------------------ | ----------------------------------- | ------------------ |
| `REGRESSION_DETECTED` | `regression_detected` | ModelPerformanceWatchdog | ModelLifecycleCoordinator           | Track regression   |
| `REGRESSION_MINOR`    | `regression_minor`    | ModelPerformanceWatchdog | FeedbackLoop                        | Minor adjustment   |
| `REGRESSION_MODERATE` | `regression_moderate` | ModelPerformanceWatchdog | FeedbackLoop                        | Increase scrutiny  |
| `REGRESSION_SEVERE`   | `regression_severe`   | ModelPerformanceWatchdog | RecoveryOrchestrator                | Consider rollback  |
| `REGRESSION_CRITICAL` | `regression_critical` | ModelPerformanceWatchdog | DaemonManager, RecoveryOrchestrator | Immediate rollback |
| `REGRESSION_CLEARED`  | `regression_cleared`  | ModelPerformanceWatchdog | ModelLifecycleCoordinator           | Resume normal ops  |

### P2P/Model Sync Events

| Event                         | Value                         | Emitters                  | Subscribers                 | Purpose                |
| ----------------------------- | ----------------------------- | ------------------------- | --------------------------- | ---------------------- |
| `P2P_MODEL_SYNCED`            | `p2p_model_synced`            | UnifiedDistributionDaemon | MetricsAnalysisOrchestrator | Model synced to peer   |
| `MODEL_SYNC_REQUESTED`        | `model_sync_requested`        | SelfplayScheduler         | UnifiedDistributionDaemon   | Model sync requested   |
| `MODEL_DISTRIBUTION_STARTED`  | `model_distribution_started`  | UnifiedDistributionDaemon | MetricsAnalysisOrchestrator | Distribution initiated |
| `MODEL_DISTRIBUTION_COMPLETE` | `model_distribution_complete` | UnifiedDistributionDaemon | SelfplayScheduler           | Distribution completed |
| `MODEL_DISTRIBUTION_FAILED`   | `model_distribution_failed`   | UnifiedDistributionDaemon | AlertManager                | Distribution failed    |
| `SYNC_STALLED`                | `sync_stalled`                | AutoSyncDaemon            | AlertManager                | Sync operation stalled |
| `SYNC_CHECKSUM_FAILED`        | `sync_checksum_failed`        | AutoSyncDaemon            | AlertManager                | Checksum mismatch      |
| `SYNC_NODE_UNREACHABLE`       | `sync_node_unreachable`       | AutoSyncDaemon            | NodeRecoveryDaemon          | Node unreachable       |
| `P2P_NODE_DEAD`               | `p2p_node_dead`               | P2POrchestrator           | NodeRecoveryDaemon          | Single node dead       |
| `P2P_NODES_DEAD`              | `p2p_nodes_dead`              | P2POrchestrator           | NodeRecoveryDaemon          | Batch of nodes dead    |

### Cluster Health Events

| Event                    | Value                    | Emitters                | Subscribers                   | Purpose               |
| ------------------------ | ------------------------ | ----------------------- | ----------------------------- | --------------------- |
| `HOST_ONLINE`            | `host_online`            | P2POrchestrator         | ClusterMonitor, SyncRouter    | Track availability    |
| `HOST_OFFLINE`           | `host_offline`           | P2POrchestrator         | ClusterMonitor, SyncRouter    | Handle failure        |
| `NODE_UNHEALTHY`         | `node_unhealthy`         | HealthCheckOrchestrator | UnifiedHealthManager          | Trigger recovery      |
| `NODE_RECOVERED`         | `node_recovered`         | RecoveryOrchestrator    | SyncRouter, SelfplayScheduler | Resume operations     |
| `NODE_OVERLOADED`        | `node_overloaded`        | ResourceMonitor         | JobManager                    | Redistribute work     |
| `NODE_ACTIVATED`         | `node_activated`         | ClusterActivator        | SelfplayScheduler             | Node activated        |
| `NODE_TERMINATED`        | `node_terminated`        | IdleShutdownDaemon      | SelfplayScheduler             | Node terminated       |
| `P2P_CLUSTER_HEALTHY`    | `p2p_cluster_healthy`    | P2POrchestrator         | DaemonManager                 | Normal operations     |
| `P2P_CLUSTER_UNHEALTHY`  | `p2p_cluster_unhealthy`  | P2POrchestrator         | RecoveryOrchestrator          | Emergency mode        |
| `CLUSTER_STATUS_CHANGED` | `cluster_status_changed` | ClusterMonitor          | DaemonManager                 | Cluster status change |
| `CLUSTER_STALL_DETECTED` | `cluster_stall_detected` | ClusterWatchdog         | RecoveryOrchestrator          | Investigate stall     |

### Resource Events

| Event                          | Value                          | Emitters              | Subscribers                      | Purpose                 |
| ------------------------------ | ------------------------------ | --------------------- | -------------------------------- | ----------------------- |
| `CLUSTER_CAPACITY_CHANGED`     | `cluster_capacity_changed`     | ClusterMonitor        | ResourceMonitoringCoordinator    | Adjust allocations      |
| `NODE_CAPACITY_UPDATED`        | `node_capacity_updated`        | NodeHealthMonitor     | UtilizationOptimizer             | Update capacity map     |
| `BACKPRESSURE_ACTIVATED`       | `backpressure_activated`       | BackpressureMonitor   | SelfplayScheduler, DaemonManager | Reduce generation       |
| `BACKPRESSURE_RELEASED`        | `backpressure_released`        | BackpressureMonitor   | SelfplayScheduler, DaemonManager | Resume normal rate      |
| `IDLE_RESOURCE_DETECTED`       | `idle_resource_detected`       | IdleResourceDaemon    | SelfplayScheduler                | Spawn work              |
| `RESOURCE_CONSTRAINT`          | `resource_constraint`          | ResourceTargetManager | IdleResourceDaemon               | CPU/GPU/Memory pressure |
| `RESOURCE_CONSTRAINT_DETECTED` | `resource_constraint_detected` | ResourceTargetManager | SelfplayScheduler                | Resource limit hit      |
| `DISK_SPACE_LOW`               | `disk_space_low`               | DiskSpaceManager      | CleanupDaemon                    | Trigger cleanup         |
| `DISK_CLEANUP_TRIGGERED`       | `disk_cleanup_triggered`       | DiskSpaceManager      | MetricsAnalysisOrchestrator      | Cleanup started         |

Backpressure events include a `level` string from `BackpressureLevel` in
`app/coordination/types.py`. Queue-based backpressure emits `none/soft/hard/stop`,
while resource-based backpressure emits `none/low/medium/high/critical`.

### Work Queue Events

| Event            | Value            | Emitters       | Subscribers                 | Purpose              |
| ---------------- | ---------------- | -------------- | --------------------------- | -------------------- |
| `WORK_QUEUED`    | `work_queued`    | QueuePopulator | WorkerNodes                 | New work available   |
| `WORK_CLAIMED`   | `work_claimed`   | WorkerNode     | QueueManager                | Track assignment     |
| `WORK_STARTED`   | `work_started`   | JobManager     | MetricsAnalysisOrchestrator | Work execution began |
| `WORK_COMPLETED` | `work_completed` | WorkerNode     | QueueManager                | Update statistics    |
| `WORK_FAILED`    | `work_failed`    | WorkerNode     | RecoveryOrchestrator        | Handle failure       |
| `WORK_RETRY`     | `work_retry`     | JobManager     | MetricsAnalysisOrchestrator | Failed, will retry   |
| `WORK_TIMEOUT`   | `work_timeout`   | QueueManager   | RecoveryOrchestrator        | Reassign work        |
| `WORK_CANCELLED` | `work_cancelled` | JobManager     | MetricsAnalysisOrchestrator | Work cancelled       |
| `JOB_PREEMPTED`  | `job_preempted`  | JobManager     | WorkerNode                  | Stop current work    |

### Leader Election Events

| Event                  | Value                  | Emitters        | Subscribers           | Purpose                   |
| ---------------------- | ---------------------- | --------------- | --------------------- | ------------------------- |
| `LEADER_ELECTED`       | `leader_elected`       | P2POrchestrator | LeadershipCoordinator | Assume leadership         |
| `LEADER_LOST`          | `leader_lost`          | P2POrchestrator | LeadershipCoordinator | Start election            |
| `LEADER_STEPDOWN`      | `leader_stepdown`      | P2POrchestrator | LeadershipCoordinator | Graceful handoff          |
| `SPLIT_BRAIN_DETECTED` | `split_brain_detected` | P2POrchestrator | AlertManager          | Multiple leaders detected |
| `SPLIT_BRAIN_RESOLVED` | `split_brain_resolved` | P2POrchestrator | LeadershipCoordinator | Split-brain resolved      |

### Daemon Lifecycle Events

| Event                       | Value                       | Emitters         | Subscribers                         | Purpose                |
| --------------------------- | --------------------------- | ---------------- | ----------------------------------- | ---------------------- |
| `DAEMON_STARTED`            | `daemon_started`            | DaemonManager    | DaemonManager, UnifiedHealthManager | Track lifecycle        |
| `DAEMON_STOPPED`            | `daemon_stopped`            | DaemonManager    | DaemonManager, UnifiedHealthManager | Track lifecycle        |
| `DAEMON_STATUS_CHANGED`     | `daemon_status_changed`     | DaemonWatchdog   | DaemonManager                       | Auto-restart           |
| `DAEMON_PERMANENTLY_FAILED` | `daemon_permanently_failed` | DaemonManager    | AlertManager                        | Exceeded restart limit |
| `COORDINATOR_HEARTBEAT`     | `coordinator_heartbeat`     | All Coordinators | DaemonWatchdog                      | Liveness check         |

### Health & Recovery Events

| Event                 | Value                 | Emitters                | Subscribers                                                             | Purpose                |
| --------------------- | --------------------- | ----------------------- | ----------------------------------------------------------------------- | ---------------------- |
| `HEALTH_CHECK_PASSED` | `health_check_passed` | ClusterWatchdogDaemon   | HealthCheckOrchestrator, FeedbackLoopController, NodeHealthOrchestrator | Node healthy           |
| `HEALTH_CHECK_FAILED` | `health_check_failed` | ClusterWatchdogDaemon   | HealthCheckOrchestrator, FeedbackLoopController, NodeHealthOrchestrator | Node unhealthy         |
| `HEALTH_ALERT`        | `health_alert`        | HealthCheckOrchestrator | AlertManager                                                            | General health warning |
| `RECOVERY_INITIATED`  | `recovery_initiated`  | NodeRecoveryDaemon      | MetricsAnalysisOrchestrator                                             | Auto-recovery started  |
| `RECOVERY_COMPLETED`  | `recovery_completed`  | NodeRecoveryDaemon      | MetricsAnalysisOrchestrator                                             | Auto-recovery finished |
| `RECOVERY_FAILED`     | `recovery_failed`     | NodeRecoveryDaemon      | AlertManager                                                            | Auto-recovery failed   |

### Task Lifecycle Events

| Event            | Value            | Emitters      | Subscribers                 | Purpose        |
| ---------------- | ---------------- | ------------- | --------------------------- | -------------- |
| `TASK_SPAWNED`   | `task_spawned`   | JobManager    | MetricsAnalysisOrchestrator | Task spawned   |
| `TASK_HEARTBEAT` | `task_heartbeat` | JobManager    | JobReaperLoop               | Task heartbeat |
| `TASK_COMPLETED` | `task_completed` | JobManager    | DataPipelineOrchestrator    | Task completed |
| `TASK_FAILED`    | `task_failed`    | JobManager    | AlertManager                | Task failed    |
| `TASK_ORPHANED`  | `task_orphaned`  | JobReaperLoop | NodeRecoveryDaemon          | Task orphaned  |
| `TASK_CANCELLED` | `task_cancelled` | JobManager    | MetricsAnalysisOrchestrator | Task cancelled |

### Checkpoint Events

| Event               | Value               | Emitters            | Subscribers   | Purpose           |
| ------------------- | ------------------- | ------------------- | ------------- | ----------------- |
| `CHECKPOINT_SAVED`  | `checkpoint_saved`  | TrainingCoordinator | ModelRegistry | Checkpoint saved  |
| `CHECKPOINT_LOADED` | `checkpoint_loaded` | TrainingCoordinator | FeedbackLoop  | Checkpoint loaded |

### Lock/Synchronization Events

| Event               | Value               | Emitters        | Subscribers  | Purpose                  |
| ------------------- | ------------------- | --------------- | ------------ | ------------------------ |
| `LOCK_TIMEOUT`      | `lock_timeout`      | DistributedLock | AlertManager | Lock acquisition timeout |
| `DEADLOCK_DETECTED` | `deadlock_detected` | DistributedLock | AlertManager | Deadlock detected        |

### Backup Events

| Event                   | Value                   | Emitters               | Subscribers                 | Purpose                  |
| ----------------------- | ----------------------- | ---------------------- | --------------------------- | ------------------------ |
| `DATA_BACKUP_COMPLETED` | `data_backup_completed` | CoordinatorDiskManager | MetricsAnalysisOrchestrator | External backup finished |

### CPU Pipeline Events

| Event                        | Value                        | Emitters                | Subscribers              | Purpose               |
| ---------------------------- | ---------------------------- | ----------------------- | ------------------------ | --------------------- |
| `CPU_PIPELINE_JOB_COMPLETED` | `cpu_pipeline_job_completed` | CPUPipelineOrchestrator | DataPipelineOrchestrator | Vast CPU job finished |

### Idle State Broadcast Events

| Event                  | Value                  | Emitters           | Subscribers        | Purpose                   |
| ---------------------- | ---------------------- | ------------------ | ------------------ | ------------------------- |
| `IDLE_STATE_BROADCAST` | `idle_state_broadcast` | IdleResourceDaemon | SelfplayScheduler  | Node idle state broadcast |
| `IDLE_STATE_REQUEST`   | `idle_state_request`   | SelfplayScheduler  | IdleResourceDaemon | Request idle state update |

### Error Recovery & Resilience Events

| Event                         | Value                         | Emitters      | Subscribers                 | Purpose                 |
| ----------------------------- | ----------------------------- | ------------- | --------------------------- | ----------------------- |
| `MODEL_CORRUPTED`             | `model_corrupted`             | ModelRegistry | AlertManager                | Model file corruption   |
| `COORDINATOR_HEALTHY`         | `coordinator_healthy`         | DaemonManager | MetricsAnalysisOrchestrator | Coordinator healthy     |
| `COORDINATOR_UNHEALTHY`       | `coordinator_unhealthy`       | DaemonManager | AlertManager                | Coordinator unhealthy   |
| `COORDINATOR_HEALTH_DEGRADED` | `coordinator_health_degraded` | DaemonManager | AlertManager                | Coordinator degraded    |
| `COORDINATOR_SHUTDOWN`        | `coordinator_shutdown`        | DaemonManager | MetricsAnalysisOrchestrator | Graceful shutdown       |
| `COORDINATOR_INIT_FAILED`     | `coordinator_init_failed`     | DaemonManager | AlertManager                | Init failed             |
| `HANDLER_TIMEOUT`             | `handler_timeout`             | EventRouter   | AlertManager                | Handler timed out       |
| `HANDLER_FAILED`              | `handler_failed`              | EventRouter   | AlertManager                | Handler threw exception |
| `ERROR`                       | `error`                       | Various       | AlertManager                | General error event     |

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
