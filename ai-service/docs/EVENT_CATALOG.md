# EVENT_CATALOG.md - RingRift AI Training Event System

**Last Updated:** December 26, 2025
**Total Event Types:** 159 (see [Quick Reference](#quick-reference) for complete list)
**Source Files:**

- `app/distributed/data_events.py` - DataEventType definitions
- `app/coordination/stage_events.py` - StageEvent definitions
- `app/coordination/event_router.py` - Unified event routing
- `app/coordination/event_emitters.py` - Typed event emitters

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Event Categories](#event-categories)
4. [Quick Reference](#quick-reference)
5. [Event Ordering Guarantees](#event-ordering-guarantees)
6. [Usage Examples](#usage-examples)
7. [Event Flow Diagrams](#event-flow-diagrams)
8. [Best Practices](#best-practices)
9. [Migration Guide](#migration-guide)
10. [Debugging](#debugging)

---

## Overview

RingRift's AI training infrastructure uses an event-driven architecture to coordinate distributed training, selfplay, evaluation, and model management across a cluster of GPU nodes. The event system enables loosely-coupled components to react to pipeline stages, data availability, model promotions, and system health changes.

### Architecture

The event system consists of three unified layers:

1. **DataEventBus** (`app/distributed/data_events.py`) - In-memory async event bus for data pipeline events
2. **StageEventBus** (`app/coordination/stage_events.py`) - Pipeline stage completion events
3. **CrossProcessEventQueue** (`app/coordination/cross_process_events.py`) - SQLite-backed persistent cross-process events

These are unified through:

4. **UnifiedEventRouter** (`app/coordination/event_router.py`) - Single entry point that routes events to all appropriate buses with deduplication

### Key Features

- **Automatic routing**: Events published to router automatically go to all appropriate buses
- **Cross-process persistence**: Events survive process restarts via SQLite
- **Content-based deduplication**: Prevents event amplification loops using SHA256 content hashing
- **Event history**: Bounded history for debugging and metrics
- **Error isolation**: Failed callbacks don't affect other subscribers
- **Type safety**: Enum-based event types with typed payloads

---

## Event Categories

This catalog documents **all actual event types** found in the codebase, organized by category.

### Training Events

Events related to neural network training lifecycle.

#### TRAINING_STARTED

**Description**: Fired when a training job begins execution.

**Payload Schema**:

```python
{
    "config_key": str,        # Board config (e.g., "square8_2p")
    "node_name": str,         # Node running training
    "job_id": str,            # Unique job identifier
    "board_type": str,        # Board type (hex8, square8, etc.)
    "num_players": int,       # Number of players
    "model_version": str,     # Model architecture version
    "timestamp": str          # ISO 8601 timestamp
}
```

**Typical Producers**: `app/training/train.py`, `TrainingCoordinator`

**Typical Consumers**: `MetricsAnalysisOrchestrator`, `ClusterMonitor`, `TrainingFreshness`

**Emit Function**: `emit_training_started()` in `event_emitters.py`

---

#### TRAINING_COMPLETED

**Description**: Fired when training finishes successfully.

**Payload Schema**:

```python
{
    "config_key": str,
    "model_id": str,          # Trained model identifier
    "model_path": str,        # Path to saved model
    "val_loss": float,        # Final validation loss
    "train_loss": float,      # Final training loss
    "epochs": int,            # Epochs completed
    "timestamp": str,
    "success": bool           # Always True for this event
}
```

**Typical Producers**: `app/training/train.py`, `TrainingCoordinator`

**Typical Consumers**: `AutoEvaluationDaemon`, `ModelDistributionDaemon`, `CurriculumFeedback`, `QualityMonitorDaemon`

**Emit Function**: `emit_training_complete()` in `event_emitters.py`

---

#### TRAINING_FAILED

**Description**: Fired when training fails or crashes.

**Payload Schema**:

```python
{
    "config_key": str,
    "error": str,             # Error message
    "error_details": str,     # Full traceback
    "timestamp": str,
    "success": bool           # Always False
}
```

**Typical Producers**: `app/training/train.py`, `TrainingCoordinator`

**Typical Consumers**: `RollbackManager`, `MetricsAnalysisOrchestrator`, `ClusterWatchdog`

**Emit Function**: `emit_training_failed()` in `event_router.py`

---

#### TRAINING_THRESHOLD_REACHED

**Description**: Fired when training is triggered (decision made, before actual training starts).

**Payload Schema**:

```python
{
    "config": str,
    "job_id": str,
    "trigger_reason": str,    # "threshold", "manual", "scheduled"
    "games": int,             # Available games
    "threshold": int,         # Threshold that triggered
    "priority": str           # "low", "normal", "high"
}
```

**Typical Producers**: `TrainingTriggerDaemon`, `AutoExportDaemon`

**Typical Consumers**: `TrainingCoordinator`, `FeedbackLoopController`

**Emit Function**: `emit_training_triggered()` in `event_emitters.py`

---

#### TRAINING_PROGRESS

**Description**: Periodic updates during training.

**Payload Schema**:

```python
{
    "config_key": str,
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "learning_rate": float,
    "timestamp": str
}
```

**Typical Producers**: `app/training/train.py`

**Typical Consumers**: `MetricsAnalysisOrchestrator`, `AnomalyDetection`

---

#### TRAINING_ROLLBACK_NEEDED

**Description**: Training produced bad model, rollback to checkpoint needed.

**Payload Schema**:

```python
{
    "model_id": str,
    "reason": str,            # Why rollback needed
    "checkpoint_path": str,   # Target checkpoint
    "severity": str           # "minor", "moderate", "severe", "critical"
}
```

**Typical Producers**: `RegressionDetector`, `AnomalyDetection`

**Typical Consumers**: `RollbackManager`, `TrainingCoordinator`

**Emit Function**: `emit_training_rollback_needed()` in `event_emitters.py`

---

#### TRAINING_ROLLBACK_COMPLETED

**Description**: Model successfully rolled back to previous checkpoint.

**Payload Schema**:

```python
{
    "model_id": str,
    "checkpoint_path": str,
    "rollback_from": str,     # Previous model that was replaced
    "reason": str
}
```

**Typical Producers**: `RollbackManager`

**Typical Consumers**: `ModelDistributionDaemon`, `CacheInvalidation`

**Emit Function**: `emit_training_rollback_completed()` in `event_emitters.py`

---

#### TRAINING_EARLY_STOPPED

**Description**: Early stopping triggered due to stagnation or regression.

**Payload Schema**:

```python
{
    "config_key": str,
    "reason": str,            # "stagnation", "regression", "plateau"
    "epochs_completed": int,
    "best_epoch": int,
    "val_loss": float
}
```

**Typical Producers**: `app/training/train.py`

**Typical Consumers**: `AnomalyDetection`, `FeedbackLoopController`

---

#### TRAINING_LOSS_ANOMALY

**Description**: Training loss spike detected.

**Payload Schema**:

```python
{
    "config_key": str,
    "epoch": int,
    "current_loss": float,
    "expected_loss": float,
    "anomaly_score": float
}
```

**Typical Producers**: `AnomalyDetection`

**Typical Consumers**: `TrainingCoordinator`, `RollbackManager`

---

#### TRAINING_LOSS_TREND

**Description**: Training loss trend analysis (improving/stalled/degrading).

**Payload Schema**:

```python
{
    "config_key": str,
    "trend": str,             # "improving", "stalled", "degrading"
    "window_epochs": int,
    "loss_delta": float
}
```

**Typical Producers**: `AnomalyDetection`

**Typical Consumers**: `FeedbackLoopController`, `AdaptiveController`

---

### Selfplay Events

Events for game generation and self-play.

#### SELFPLAY_COMPLETE

**Description**: Standard selfplay batch finished generating games.

**Payload Schema**:

```python
{
    "task_id": str,
    "board_type": str,
    "num_players": int,
    "games_generated": int,
    "success": bool,
    "node_id": str,           # Node that ran selfplay
    "duration_seconds": float,
    "selfplay_type": str,     # "standard", "gpu_accelerated", "canonical"
    "iteration": int,
    "error": str | None
}
```

**Typical Producers**: `SelfplayRunner`, `HeuristicSelfplayRunner`, `BackgroundSelfplay`

**Typical Consumers**: `AutoSyncDaemon`, `AutoExportDaemon`, `QualityMonitorDaemon`

**Emit Function**: `emit_selfplay_complete()` in `event_emitters.py`

---

#### GPU_SELFPLAY_COMPLETE

**Description**: GPU-accelerated selfplay batch finished (6.5x speedup).

**Payload Schema**: Same as `SELFPLAY_COMPLETE` with `selfplay_type: "gpu_accelerated"`

**Typical Producers**: `GumbelMCTSSelfplayRunner`, `GPUParallelGames`

**Typical Consumers**: Same as `SELFPLAY_COMPLETE`

**Emit Function**: `emit_selfplay_complete()` with `selfplay_type="gpu_accelerated"`

---

#### CANONICAL_SELFPLAY_COMPLETE

**Description**: Canonical selfplay for parity validation finished.

**Payload Schema**: Same as `SELFPLAY_COMPLETE` with `selfplay_type: "canonical"`

**Typical Producers**: Canonical selfplay scripts

**Typical Consumers**: `ParityValidation`, `AutoSyncDaemon`

**Emit Function**: `emit_selfplay_complete()` with `selfplay_type="canonical"`

---

#### SELFPLAY_TARGET_UPDATED

**Description**: Request to change selfplay rate (more/fewer games needed).

**Payload Schema**:

```python
{
    "config_key": str,
    "old_target": int,
    "new_target": int,
    "reason": str             # "quality_low", "training_stalled", "manual"
}
```

**Typical Producers**: `QueuePopulator`, `SelfplayScheduler`

**Typical Consumers**: `IdleResourceDaemon`, `SelfplayOrchestrator`

---

#### SELFPLAY_RATE_CHANGED

**Description**: Selfplay rate multiplier changed by >20%.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_rate": float,
    "new_rate": float,
    "change_pct": float
}
```

**Typical Producers**: `SelfplayScheduler`, `FeedbackSignals`

**Typical Consumers**: `IdleResourceDaemon`, `MetricsAnalysisOrchestrator`

---

### Evaluation Events

Events for model evaluation and tournaments.

#### EVALUATION_STARTED

**Description**: Evaluation/gauntlet started.

**Payload Schema**:

```python
{
    "model_id": str,
    "board_type": str,
    "num_players": int,
    "evaluation_type": str,   # "gauntlet", "tournament", "shadow"
    "games_planned": int
}
```

**Typical Producers**: `AutoEvaluationDaemon`, `game_gauntlet.py`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### EVALUATION_COMPLETED

**Description**: Evaluation finished with results.

**Payload Schema**:

```python
{
    "model_id": str,
    "board_type": str,
    "num_players": int,
    "elo": float,
    "win_rate": float,
    "games_played": int,
    "elo_delta": float,       # Change from previous
    "opponents": list[str],   # Opponent types evaluated against
    "timestamp": str
}
```

**Typical Producers**: `game_gauntlet.py`, `AutoEvaluationDaemon`, `TournamentDaemon`

**Typical Consumers**: `AutoPromotionDaemon`, `CurriculumFeedback`, `EloService`, `GauntletFeedbackController`

**Emit Function**: `emit_evaluation_complete()` in `event_emitters.py`

---

#### EVALUATION_FAILED

**Description**: Evaluation crashed or failed.

**Payload Schema**:

```python
{
    "model_id": str,
    "error": str,
    "timestamp": str
}
```

**Typical Producers**: `game_gauntlet.py`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### ELO_UPDATED

**Description**: Model Elo rating updated.

**Payload Schema**:

```python
{
    "model_id": str,
    "config_key": str,
    "old_elo": float,
    "new_elo": float,
    "elo_delta": float,
    "games_played": int
}
```

**Typical Producers**: `EloService`, `UnifiedEloDB`

**Typical Consumers**: `CurriculumFeedback`, `SelfplayScheduler`, `EloVelocityTracker`

---

#### ELO_VELOCITY_CHANGED

**Description**: Significant change in Elo improvement rate detected.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_velocity": float,    # Elo/hour
    "new_velocity": float,
    "change_pct": float
}
```

**Typical Producers**: `EloService`, `SelfplayScheduler`

**Typical Consumers**: `AdaptiveController`, `SelfplayScheduler`, `CurriculumFeedback`

---

#### ELO_SIGNIFICANT_CHANGE

**Description**: Large Elo change that may trigger curriculum rebalance.

**Payload Schema**:

```python
{
    "config_key": str,
    "elo_delta": float,
    "new_elo": float,
    "threshold": float        # Threshold for "significant"
}
```

**Typical Producers**: `EloService`

**Typical Consumers**: `CurriculumFeedback`, `SelfplayScheduler`

---

### Sync Events

Events for data synchronization across cluster.

#### DATA_SYNC_STARTED

**Description**: Data sync operation started.

**Payload Schema**:

```python
{
    "sync_type": str,         # "data", "model", "elo", "registry"
    "source": str,            # Source node
    "targets": list[str],     # Target nodes
    "items_to_sync": int
}
```

**Typical Producers**: `AutoSyncDaemon`, `UnifiedDataSync`, `ClusterDataSync`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### DATA_SYNC_COMPLETED

**Description**: Data sync finished successfully.

**Payload Schema**:

```python
{
    "sync_type": str,
    "items_synced": int,
    "success": bool,
    "duration_seconds": float,
    "source": str,
    "iteration": int,
    "components": list[str],  # Component names synced
    "errors": list[str],      # Any non-fatal errors
    "timestamp": str
}
```

**Typical Producers**: `AutoSyncDaemon`, `UnifiedDataSync`, `ClusterDataSync`

**Typical Consumers**: `AutoExportDaemon`, `TrainingFreshness`, `ParityValidation`

**Emit Function**: `emit_sync_completed()` in `event_router.py`

---

#### DATA_SYNC_FAILED

**Description**: Data sync failed.

**Payload Schema**:

```python
{
    "sync_type": str,
    "error": str,
    "source": str,
    "timestamp": str
}
```

**Typical Producers**: `AutoSyncDaemon`, `UnifiedDataSync`

**Typical Consumers**: `MetricsAnalysisOrchestrator`, `ClusterWatchdog`

---

#### SYNC_COMPLETE

**Description**: Generic sync completion (stage event variant).

**Payload Schema**:

```python
{
    "sync_type": str,
    "items_synced": int,
    "success": bool,
    "duration_seconds": float,
    "iteration": int
}
```

**Typical Producers**: `AutoSyncDaemon`, `SyncRouter`

**Typical Consumers**: `DataPipelineOrchestrator`, `AutoExportDaemon`

**Emit Function**: `emit_sync_complete()` in `event_emitters.py`

---

#### SYNC_STALLED

**Description**: Sync operation stalled/timed out.

**Payload Schema**:

```python
{
    "sync_type": str,
    "source": str,
    "duration_seconds": float,
    "timeout_threshold": float
}
```

**Typical Producers**: `AutoSyncDaemon`, `UnifiedDataSync`

**Typical Consumers**: `ClusterWatchdog`, `NodeRecoveryDaemon`

---

#### NEW_GAMES_AVAILABLE

**Description**: New games synced or generated, available for training.

**Payload Schema**:

```python
{
    "host": str,              # Host with new games
    "new_games": int,
    "total_games": int,
    "timestamp": str
}
```

**Typical Producers**: `AutoSyncDaemon`, `SelfplayRunner`

**Typical Consumers**: `AutoExportDaemon`, `TrainingTriggerDaemon`, `QualityMonitorDaemon`

**Emit Function**: `emit_new_games()` in `event_emitters.py`

---

#### GAME_SYNCED

**Description**: Individual game(s) synced to target nodes.

**Payload Schema**:

```python
{
    "game_ids": list[str],
    "source": str,
    "targets": list[str],
    "count": int
}
```

**Typical Producers**: `AutoSyncDaemon`

**Typical Consumers**: `ReplicationMonitor`

---

### Quality Events

Events for data quality monitoring and feedback.

#### QUALITY_SCORE_UPDATED

**Description**: Game quality score calculated or recalculated.

**Payload Schema** (aggregate):

```python
{
    "board_type": str,
    "num_players": int,
    "avg_quality": float,     # 0.0-1.0
    "total_games": int,
    "high_quality_games": int,
    "timestamp": str
}
```

**Payload Schema** (per-game):

```python
{
    "game_id": str,
    "quality_score": float,
    "quality_category": str,  # "excellent", "good", "adequate", "poor", "unusable"
    "training_weight": float,
    "game_length": int,
    "is_decisive": bool,
    "is_per_game": bool,      # True for per-game events
    "source": str
}
```

**Typical Producers**: `UnifiedQualityScorer`, `QualityMonitorDaemon`

**Typical Consumers**: `FeedbackLoopController`, `TrainingEnhancements`, `CurriculumFeedback`

**Emit Functions**: `emit_quality_updated()`, `emit_game_quality_score()` in `event_emitters.py`

---

#### QUALITY_DEGRADED

**Description**: Data quality dropped below threshold.

**Payload Schema**:

```python
{
    "config_key": str,
    "current_quality": float,
    "threshold": float,
    "quality_drop_pct": float
}
```

**Typical Producers**: `QualityMonitorDaemon`

**Typical Consumers**: `SelfplayScheduler`, `FeedbackLoopController`

---

#### QUALITY_DISTRIBUTION_CHANGED

**Description**: Significant shift in quality distribution detected.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_avg": float,
    "new_avg": float,
    "shift_magnitude": float
}
```

**Typical Producers**: `QualityMonitorDaemon`

**Typical Consumers**: `FeedbackLoopController`, `CurriculumFeedback`

---

#### HIGH_QUALITY_DATA_AVAILABLE

**Description**: Sufficient high-quality data ready for training.

**Payload Schema**:

```python
{
    "config_key": str,
    "high_quality_count": int,
    "total_count": int,
    "avg_quality": float
}
```

**Typical Producers**: `QualityMonitorDaemon`

**Typical Consumers**: `TrainingTriggerDaemon`

---

#### LOW_QUALITY_DATA_WARNING

**Description**: Quality below threshold, may block training.

**Payload Schema**:

```python
{
    "config_key": str,
    "avg_quality": float,
    "threshold": float
}
```

**Typical Producers**: `QualityMonitorDaemon`

**Typical Consumers**: `SelfplayScheduler`, `FeedbackLoopController`

---

#### TRAINING_BLOCKED_BY_QUALITY

**Description**: Training blocked due to insufficient quality.

**Payload Schema**:

```python
{
    "config_key": str,
    "quality": float,
    "required_quality": float
}
```

**Typical Producers**: `TrainingCoordinator`, `TrainingTriggerDaemon`

**Typical Consumers**: `SelfplayScheduler`, `FeedbackLoopController`

---

#### QUALITY_FEEDBACK_ADJUSTED

**Description**: Quality feedback parameters updated for config.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_weight": float,
    "new_weight": float,
    "reason": str
}
```

**Typical Producers**: `FeedbackLoopController`, `CurriculumFeedback`

**Typical Consumers**: `SelfplayScheduler`

---

#### QUALITY_PENALTY_APPLIED

**Description**: Quality penalty applied, reducing selfplay rate.

**Payload Schema**:

```python
{
    "config_key": str,
    "penalty_factor": float,  # Multiplier < 1.0
    "reason": str
}
```

**Typical Producers**: `FeedbackLoopController`

**Typical Consumers**: `SelfplayScheduler`

---

### Health/Monitoring Events

Events for cluster health and node monitoring.

#### HOST_ONLINE

**Description**: Node came online or was discovered.

**Payload Schema**:

```python
{
    "node_id": str,
    "host_id": str,           # Alias for compatibility
    "host_type": str,         # "gh200", "cpu", "runpod", etc.
    "capabilities": dict,     # GPU info, resources, etc.
    "timestamp": str
}
```

**Typical Producers**: `ClusterMonitor`, `P2POrchestrator`

**Typical Consumers**: `IdleResourceDaemon`, `UtilizationOptimizer`, `WorkDistributor`

**Emit Function**: `emit_host_online()` in `event_emitters.py`

---

#### HOST_OFFLINE

**Description**: Node went offline or became unreachable.

**Payload Schema**:

```python
{
    "node_id": str,
    "host_id": str,
    "reason": str,
    "timestamp": str
}
```

**Typical Producers**: `ClusterMonitor`, `P2POrchestrator`

**Typical Consumers**: `NodeRecoveryDaemon`, `WorkQueue`, `ClusterWatchdog`

**Emit Function**: `emit_host_offline()` in `event_emitters.py`

---

#### NODE_UNHEALTHY

**Description**: Node health check failed.

**Payload Schema**:

```python
{
    "node_id": str,
    "health_score": float,    # 0.0-1.0
    "issues": list[str],
    "timestamp": str
}
```

**Typical Producers**: `UnifiedNodeHealthDaemon`, `HealthChecks`

**Typical Consumers**: `NodeRecoveryDaemon`, `WorkDistributor`

---

#### NODE_RECOVERED

**Description**: Previously offline/unhealthy node recovered.

**Payload Schema**:

```python
{
    "node_id": str,
    "host_id": str,
    "recovery_type": str,     # "automatic", "manual"
    "offline_duration_seconds": float,
    "timestamp": str
}
```

**Typical Producers**: `NodeRecoveryDaemon`

**Typical Consumers**: `IdleResourceDaemon`, `WorkDistributor`, `ClusterMonitor`

**Emit Function**: `emit_node_recovered()` in `event_emitters.py`

---

#### HEALTH_CHECK_PASSED

**Description**: Node health check succeeded.

**Payload Schema**:

```python
{
    "node_id": str,
    "health_score": float,
    "timestamp": str
}
```

**Typical Producers**: `UnifiedNodeHealthDaemon`, `HealthChecks`

**Typical Consumers**: `ClusterMonitor`

---

#### HEALTH_CHECK_FAILED

**Description**: Node health check failed.

**Payload Schema**:

```python
{
    "node_id": str,
    "reason": str,
    "error": str,
    "timestamp": str
}
```

**Typical Producers**: `UnifiedNodeHealthDaemon`, `HealthChecks`

**Typical Consumers**: `NodeRecoveryDaemon`, `ClusterWatchdog`

---

#### P2P_CLUSTER_HEALTHY

**Description**: P2P cluster is healthy (quorum achieved).

**Payload Schema**:

```python
{
    "alive_peers": int,
    "total_peers": int,
    "leader_id": str,
    "quorum": bool
}
```

**Typical Producers**: `P2POrchestrator`

**Typical Consumers**: `ClusterMonitor`

---

#### P2P_CLUSTER_UNHEALTHY

**Description**: P2P cluster unhealthy (lost quorum or leader).

**Payload Schema**:

```python
{
    "alive_peers": int,
    "total_peers": int,
    "reason": str
}
```

**Typical Producers**: `P2POrchestrator`

**Typical Consumers**: `ClusterWatchdog`, `NodeRecoveryDaemon`

---

#### NODE_OVERLOADED

**Description**: Node CPU/GPU/memory utilization critical.

**Payload Schema**:

```python
{
    "node_id": str,
    "cpu_util": float,
    "gpu_util": float,
    "memory_util": float,
    "threshold": float
}
```

**Typical Producers**: `ResourceMonitoringCoordinator`, `SystemHealthMonitor`

**Typical Consumers**: `WorkDistributor`, `BackpressureController`

---

#### IDLE_RESOURCE_DETECTED

**Description**: Idle GPU/CPU detected, can spawn work.

**Payload Schema**:

```python
{
    "node_id": str,
    "resource_type": str,     # "gpu", "cpu"
    "idle_duration": float,
    "utilization": float
}
```

**Typical Producers**: `IdleResourceDaemon`

**Typical Consumers**: `SelfplayOrchestrator`, `WorkDistributor`

---

#### COORDINATOR_HEARTBEAT

**Description**: Liveness signal from coordinator.

**Payload Schema**:

```python
{
    "coordinator_name": str,
    "health_score": float,
    "active_handlers": int,
    "events_processed": int,
    "timestamp": str
}
```

**Typical Producers**: All coordinators

**Typical Consumers**: `ClusterWatchdog`, `DaemonWatchdog`

**Emit Function**: `emit_coordinator_heartbeat()` in `event_emitters.py`

---

#### COORDINATOR_HEALTH_DEGRADED

**Description**: Coordinator health degraded but still functional.

**Payload Schema**:

```python
{
    "coordinator_name": str,
    "reason": str,
    "health_score": float,
    "issues": list[str],
    "timestamp": str
}
```

**Typical Producers**: Coordinators

**Typical Consumers**: `ClusterWatchdog`, `DaemonWatchdog`

**Emit Function**: `emit_coordinator_health_degraded()` in `event_emitters.py`

---

#### COORDINATOR_SHUTDOWN

**Description**: Coordinator shutting down gracefully.

**Payload Schema**:

```python
{
    "coordinator_name": str,
    "reason": str,            # "graceful", "error", "forced"
    "remaining_tasks": int,
    "state_snapshot": dict,
    "timestamp": str
}
```

**Typical Producers**: All coordinators

**Typical Consumers**: `DaemonManager`, `ClusterWatchdog`

**Emit Function**: `emit_coordinator_shutdown()` in `event_emitters.py`

---

### Optimization Events

Events for hyperparameter optimization and architecture search.

#### CMAES_TRIGGERED

**Description**: CMA-ES hyperparameter optimization started.

**Payload Schema**:

```python
{
    "run_id": str,
    "reason": str,
    "parameters_searched": int,
    "search_space": dict,
    "generations": int,
    "population_size": int,
    "timestamp": str
}
```

**Typical Producers**: `OptimizationOrchestrator`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

**Emit Function**: `emit_optimization_triggered(optimization_type="cmaes")` in `event_emitters.py`

---

#### NAS_TRIGGERED

**Description**: Neural Architecture Search started.

**Payload Schema**: Same as `CMAES_TRIGGERED`

**Typical Producers**: `OptimizationOrchestrator`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

**Emit Function**: `emit_optimization_triggered(optimization_type="nas")` in `event_emitters.py`

---

#### PLATEAU_DETECTED

**Description**: Training metric plateaued (no improvement).

**Payload Schema**:

```python
{
    "metric_name": str,
    "plateau_type": str,      # "loss", "elo", "metric"
    "current_value": float,
    "best_value": float,
    "epochs_since_improvement": int,
    "timestamp": str
}
```

**Typical Producers**: `AnomalyDetection`

**Typical Consumers**: `AdaptiveController`, `OptimizationOrchestrator`, `FeedbackLoopController`

**Emit Function**: `emit_plateau_detected()` in `event_emitters.py`

---

#### HYPERPARAMETER_UPDATED

**Description**: Hyperparameter changed (manually or automatically).

**Payload Schema**:

```python
{
    "config": str,
    "param_name": str,
    "old_value": any,
    "new_value": any,
    "optimizer": str,         # "cmaes", "nas", "manual", "adaptive"
    "timestamp": str
}
```

**Typical Producers**: `AdaptiveController`, `OptimizationOrchestrator`

**Typical Consumers**: `TrainingCoordinator`, `MetricsAnalysisOrchestrator`

**Emit Function**: `emit_hyperparameter_updated()` in `event_emitters.py`

---

#### ADAPTIVE_PARAMS_CHANGED

**Description**: Training parameters adjusted based on Elo velocity.

**Payload Schema**:

```python
{
    "config_key": str,
    "params_changed": dict,   # param_name -> new_value
    "elo_velocity": float,
    "reason": str
}
```

**Typical Producers**: `AdaptiveController`

**Typical Consumers**: `TrainingCoordinator`

---

### Error/Recovery Events

Events for error detection and recovery.

#### REGRESSION_DETECTED

**Description**: Model performance regression detected.

**Payload Schema**:

```python
{
    "metric_name": str,
    "current_value": float,
    "previous_value": float,
    "severity": str,          # "minor", "moderate", "severe", "critical"
    "regression_amount": float,
    "timestamp": str
}
```

**Typical Producers**: `RegressionDetector`

**Typical Consumers**: `RollbackManager`, `AutoPromotionDaemon`, `TrainingCoordinator`

**Emit Function**: `emit_regression_detected()` in `event_emitters.py`

---

#### MODEL_CORRUPTED

**Description**: Model file corruption detected.

**Payload Schema**:

```python
{
    "model_id": str,
    "model_path": str,
    "corruption_type": str,   # "checksum", "format", "missing"
    "timestamp": str
}
```

**Typical Producers**: Model loading code

**Typical Consumers**: `RollbackManager`, `ModelDistributionDaemon`

**Emit Function**: `emit_model_corrupted()` in `event_emitters.py`

---

#### HANDLER_FAILED

**Description**: Event handler threw exception.

**Payload Schema**:

```python
{
    "handler_name": str,
    "event_type": str,
    "error": str,
    "coordinator": str,
    "timestamp": str
}
```

**Typical Producers**: Event buses

**Typical Consumers**: `ClusterWatchdog`, `HandlerResilience`

**Emit Function**: `emit_handler_failed()` in `event_emitters.py`

---

#### HANDLER_TIMEOUT

**Description**: Event handler timed out.

**Payload Schema**:

```python
{
    "handler_name": str,
    "event_type": str,
    "timeout_seconds": float,
    "coordinator": str,
    "timestamp": str
}
```

**Typical Producers**: Event buses

**Typical Consumers**: `ClusterWatchdog`, `HandlerResilience`

**Emit Function**: `emit_handler_timeout()` in `event_emitters.py`

---

#### TASK_ORPHANED

**Description**: Task lost its parent worker (worker failure).

**Payload Schema**:

```python
{
    "task_id": str,
    "task_type": str,
    "node_id": str,
    "last_heartbeat": float,
    "reason": str,
    "timestamp": str
}
```

**Typical Producers**: `TaskLifecycleCoordinator`

**Typical Consumers**: `WorkQueue`, `NodeRecoveryDaemon`

**Emit Function**: `emit_task_orphaned()` in `event_emitters.py`

---

#### TASK_ABANDONED

**Description**: Task intentionally abandoned (not orphaned).

**Payload Schema**:

```python
{
    "task_id": str,
    "task_type": str,
    "node_id": str,
    "reason": str,
    "timestamp": str
}
```

**Typical Producers**: `TaskLifecycleCoordinator`

**Typical Consumers**: `WorkQueue`, `MetricsAnalysisOrchestrator`

**Emit Function**: `emit_task_abandoned()` in `event_emitters.py`

---

#### RECOVERY_INITIATED

**Description**: Auto-recovery started.

**Payload Schema**:

```python
{
    "recovery_type": str,     # "node", "task", "model", "data"
    "target": str,            # What's being recovered
    "reason": str
}
```

**Typical Producers**: `NodeRecoveryDaemon`, `RollbackManager`

**Typical Consumers**: `ClusterWatchdog`

---

#### RECOVERY_COMPLETED

**Description**: Auto-recovery finished successfully.

**Payload Schema**:

```python
{
    "recovery_type": str,
    "target": str,
    "duration_seconds": float,
    "actions_taken": list[str]
}
```

**Typical Producers**: `NodeRecoveryDaemon`, `RollbackManager`

**Typical Consumers**: `ClusterWatchdog`, `MetricsAnalysisOrchestrator`

---

#### RECOVERY_FAILED

**Description**: Auto-recovery failed.

**Payload Schema**:

```python
{
    "recovery_type": str,
    "target": str,
    "error": str,
    "timestamp": str
}
```

**Typical Producers**: `NodeRecoveryDaemon`, `RollbackManager`

**Typical Consumers**: `ClusterWatchdog`

---

### Promotion Events

Events for model promotion and tier gating.

#### MODEL_PROMOTED

**Description**: Model promoted to production/champion tier.

**Payload Schema**:

```python
{
    "model_id": str,
    "tier": str,              # "production", "champion"
    "elo": float,
    "promotion_type": str,    # "production", "champion"
    "elo_improvement": float,
    "model_path": str,
    "timestamp": str
}
```

**Typical Producers**: `AutoPromotionDaemon`, `PromotionController`

**Typical Consumers**: `ModelDistributionDaemon`, `CacheInvalidation`, `CurriculumFeedback`, `NPZDistributionDaemon`

**Emit Function**: `emit_model_promoted()` in `event_router.py`

---

#### PROMOTION_COMPLETE

**Description**: Promotion stage finished (StageEvent variant).

**Payload Schema**:

```python
{
    "model_id": str,
    "board_type": str,
    "num_players": int,
    "promotion_type": str,
    "elo_improvement": float,
    "model_path": str,
    "success": bool,
    "iteration": int,
    "timestamp": str
}
```

**Typical Producers**: `AutoPromotionDaemon`, `PromotionController`

**Typical Consumers**: Same as `MODEL_PROMOTED`

**Emit Function**: `emit_promotion_complete()` in `event_emitters.py`

---

#### PROMOTION_CANDIDATE

**Description**: Model identified as promotion candidate.

**Payload Schema**:

```python
{
    "model_id": str,
    "elo": float,
    "win_rate": float,
    "reason": str
}
```

**Typical Producers**: `AutoPromotionDaemon`, `PromotionController`

**Typical Consumers**: `AutoEvaluationDaemon`

---

#### PROMOTION_REJECTED

**Description**: Model failed promotion criteria.

**Payload Schema**:

```python
{
    "model_id": str,
    "reason": str,
    "elo": float,
    "required_elo": float
}
```

**Typical Producers**: `AutoPromotionDaemon`, `PromotionController`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### PROMOTION_ROLLED_BACK

**Description**: Promotion rolled back due to issues.

**Payload Schema**:

```python
{
    "model_id": str,
    "reason": str,
    "rollback_to": str,
    "timestamp": str
}
```

**Typical Producers**: `PromotionController`, `RollbackManager`

**Typical Consumers**: `ModelDistributionDaemon`, `CacheInvalidation`

---

### Curriculum Events

Events for curriculum learning and weight rebalancing.

#### CURRICULUM_REBALANCED

**Description**: Curriculum weights rebalanced across configs.

**Payload Schema**:

```python
{
    "config": str,
    "old_weights": dict,      # config -> weight
    "new_weights": dict,
    "reason": str,
    "trigger": str,           # "automatic", "manual", "elo_change"
    "timestamp": str
}
```

**Typical Producers**: `CurriculumFeedback`

**Typical Consumers**: `SelfplayScheduler`, `QueuePopulator`

**Emit Function**: `emit_curriculum_rebalanced()` in `event_emitters.py`

---

#### WEIGHT_UPDATED

**Description**: Individual config weight updated.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_weight": float,
    "new_weight": float,
    "reason": str
}
```

**Typical Producers**: `CurriculumFeedback`, `SelfplayScheduler`

**Typical Consumers**: `QueuePopulator`, `IdleResourceDaemon`

---

#### EXPLORATION_ADJUSTED

**Description**: Exploration temperature strategy changed.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_strategy": str,
    "new_strategy": str,
    "reason": str
}
```

**Typical Producers**: `FeedbackSignals`, `TemperatureScheduling`

**Typical Consumers**: `SelfplayRunner`

---

### Backpressure Events

Events for backpressure management.

#### BACKPRESSURE_ACTIVATED

**Description**: Backpressure activated due to resource constraints.

**Payload Schema**:

```python
{
    "node_id": str,
    "level": str,             # "low", "medium", "high", "critical"
    "reason": str,
    "resource_type": str,     # "gpu", "memory", "disk"
    "utilization": float,
    "timestamp": str
}
```

**Typical Producers**: `ResourceMonitoringCoordinator`

**Typical Consumers**: `WorkDistributor`, `SelfplayOrchestrator`, `IdleResourceDaemon`

**Emit Function**: `emit_backpressure_activated()` in `event_emitters.py`

---

#### BACKPRESSURE_RELEASED

**Description**: Backpressure released, resources available.

**Payload Schema**:

```python
{
    "node_id": str,
    "previous_level": str,
    "duration_seconds": float,
    "timestamp": str
}
```

**Typical Producers**: `ResourceMonitoringCoordinator`

**Typical Consumers**: `WorkDistributor`, `IdleResourceDaemon`

**Emit Function**: `emit_backpressure_released()` in `event_emitters.py`

---

### Cache Events

Events for cache invalidation.

#### CACHE_INVALIDATED

**Description**: Cache entries invalidated (model or node).

**Payload Schema**:

```python
{
    "invalidation_type": str, # "model", "node"
    "target_id": str,         # Model ID or node ID
    "count": int,             # Entries invalidated
    "affected_nodes": list[str],
    "affected_models": list[str],
    "timestamp": str
}
```

**Typical Producers**: `CacheInvalidation`

**Typical Consumers**: `ModelLoadingCache`, `UnifiedModelStore`

**Emit Function**: `emit_cache_invalidated()` in `event_emitters.py`

---

### Data Freshness Events

Events for training data freshness monitoring.

#### DATA_STALE

**Description**: Training data is stale (too old).

**Payload Schema**:

```python
{
    "config_key": str,
    "data_age_hours": float,
    "threshold_hours": float,
    "last_sync": str          # ISO timestamp
}
```

**Typical Producers**: `TrainingFreshness`

**Typical Consumers**: `AutoSyncDaemon`, `TrainingCoordinator`

---

#### DATA_FRESH

**Description**: Training data is fresh (recently synced).

**Payload Schema**:

```python
{
    "config_key": str,
    "data_age_hours": float,
    "last_sync": str
}
```

**Typical Producers**: `TrainingFreshness`

**Typical Consumers**: `TrainingCoordinator`

---

#### SYNC_TRIGGERED

**Description**: Sync triggered due to stale data.

**Payload Schema**:

```python
{
    "config_key": str,
    "reason": str,
    "data_age_hours": float
}
```

**Typical Producers**: `TrainingFreshness`

**Typical Consumers**: `AutoSyncDaemon`

---

### Export Events

Events for NPZ export operations.

#### NPZ_EXPORT_STARTED

**Description**: NPZ export started.

**Payload Schema**:

```python
{
    "board_type": str,
    "num_players": int,
    "output_path": str,
    "games_to_export": int
}
```

**Typical Producers**: `AutoExportDaemon`, `export_replay_dataset.py`

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### NPZ_EXPORT_COMPLETE

**Description**: NPZ export finished, ready for training.

**Payload Schema**:

```python
{
    "board_type": str,
    "num_players": int,
    "samples_exported": int,
    "games_exported": int,
    "output_path": str,
    "success": bool,
    "duration_seconds": float,
    "config_key": str,
    "timestamp": str
}
```

**Typical Producers**: `AutoExportDaemon`, `export_replay_dataset.py`

**Typical Consumers**: `TrainingTriggerDaemon`, `TrainingCoordinator`, `NPZDistributionDaemon`

**Emit Function**: `emit_npz_export_complete()` in `event_emitters.py`

---

### Parity Events

Events for TypeScript/Python parity validation.

#### PARITY_VALIDATION_STARTED

**Description**: Parity validation started.

**Payload Schema**:

```python
{
    "board_type": str,
    "num_players": int,
    "games_to_validate": int
}
```

**Typical Producers**: Parity validation scripts

**Typical Consumers**: `MetricsAnalysisOrchestrator`

---

#### PARITY_VALIDATION_COMPLETE

**Description**: Parity validation finished.

**Payload Schema**:

```python
{
    "board_type": str,
    "num_players": int,
    "success": bool,
    "games_validated": int,
    "parity_rate": float,
    "timestamp": str
}
```

**Typical Producers**: Parity validation scripts

**Typical Consumers**: `MetricsAnalysisOrchestrator`, `QualityMonitorDaemon`

---

#### PARITY_FAILURE_RATE_CHANGED

**Description**: Parity failure rate changed significantly.

**Payload Schema**:

```python
{
    "config_key": str,
    "old_rate": float,
    "new_rate": float,
    "threshold": float
}
```

**Typical Producers**: Parity validation scripts

**Typical Consumers**: `QualityMonitorDaemon`, `ClusterWatchdog`

---

## Quick Reference

Complete list of all 159 event types organized by category:

### Training Pipeline Events

| Event                         | Description                    | Key Fields                         |
| ----------------------------- | ------------------------------ | ---------------------------------- |
| `TRAINING_STARTED`            | Training job begins            | config_key, node_name, job_id      |
| `TRAINING_PROGRESS`           | Periodic training updates      | epoch, train_loss, val_loss        |
| `TRAINING_COMPLETED`          | Training finished successfully | model_id, model_path, val_loss     |
| `TRAINING_FAILED`             | Training crashed               | error, error_details               |
| `TRAINING_THRESHOLD_REACHED`  | Training triggered             | games, threshold, priority         |
| `TRAINING_EARLY_STOPPED`      | Early stopping triggered       | reason, epochs_completed           |
| `TRAINING_LOSS_ANOMALY`       | Loss spike detected            | current_loss, expected_loss        |
| `TRAINING_LOSS_TREND`         | Loss trend analysis            | trend: improving/stalled/degrading |
| `TRAINING_ROLLBACK_NEEDED`    | Rollback required              | checkpoint_path, severity          |
| `TRAINING_ROLLBACK_COMPLETED` | Rollback successful            | model_id, rollback_from            |

### Selfplay Events

| Event                     | Description             | Key Fields                              |
| ------------------------- | ----------------------- | --------------------------------------- |
| `SELFPLAY_COMPLETE`       | Selfplay batch finished | games_generated, node_id, selfplay_type |
| `SELFPLAY_TARGET_UPDATED` | Game target changed     | old_target, new_target, reason          |
| `SELFPLAY_RATE_CHANGED`   | Rate multiplier changed | old_rate, new_rate, change_pct          |

### Evaluation Events

| Event                    | Description                  | Key Fields                   |
| ------------------------ | ---------------------------- | ---------------------------- |
| `EVALUATION_STARTED`     | Evaluation begins            | model_id, evaluation_type    |
| `EVALUATION_PROGRESS`    | Evaluation progress          | games_completed, current_elo |
| `EVALUATION_COMPLETED`   | Evaluation finished          | elo, win_rate, elo_delta     |
| `EVALUATION_FAILED`      | Evaluation crashed           | error                        |
| `ELO_UPDATED`            | Elo rating changed           | old_elo, new_elo, elo_delta  |
| `ELO_VELOCITY_CHANGED`   | Elo improvement rate changed | old_velocity, new_velocity   |
| `ELO_SIGNIFICANT_CHANGE` | Large Elo change             | elo_delta, threshold         |

### Sync Events

| Event                 | Description                  | Key Fields                          |
| --------------------- | ---------------------------- | ----------------------------------- |
| `DATA_SYNC_STARTED`   | Sync operation started       | sync_type, source, targets          |
| `DATA_SYNC_COMPLETED` | Sync finished                | items_synced, duration_seconds      |
| `DATA_SYNC_FAILED`    | Sync failed                  | error, source                       |
| `SYNC_STALLED`        | Sync timed out               | duration_seconds, timeout_threshold |
| `SYNC_REQUEST`        | Explicit sync request        | source, target, priority            |
| `SYNC_TRIGGERED`      | Sync triggered by stale data | config_key, data_age_hours          |
| `NEW_GAMES_AVAILABLE` | New games ready              | host, new_games, total_games        |
| `GAME_SYNCED`         | Games synced to targets      | game_ids, source, targets           |

### Model Events

| Event                         | Description                    | Key Fields                |
| ----------------------------- | ------------------------------ | ------------------------- |
| `MODEL_PROMOTED`              | Model promoted                 | tier, elo, promotion_type |
| `MODEL_UPDATED`               | Model metadata updated         | model_id, changes         |
| `MODEL_CORRUPTED`             | Model corruption detected      | corruption_type           |
| `MODEL_SYNC_REQUESTED`        | Model sync requested           | model_id, target_nodes    |
| `MODEL_DISTRIBUTION_COMPLETE` | Model distributed              | model_id, nodes           |
| `PROMOTION_CANDIDATE`         | Promotion candidate identified | elo, win_rate             |
| `PROMOTION_STARTED`           | Promotion process started      | model_id                  |
| `PROMOTION_REJECTED`          | Promotion failed criteria      | reason, required_elo      |
| `PROMOTION_ROLLED_BACK`       | Promotion rolled back          | reason, rollback_to       |

### Quality Events

| Event                          | Description                  | Key Fields                 |
| ------------------------------ | ---------------------------- | -------------------------- |
| `QUALITY_SCORE_UPDATED`        | Quality score calculated     | avg_quality, total_games   |
| `QUALITY_DEGRADED`             | Quality below threshold      | current_quality, threshold |
| `QUALITY_DISTRIBUTION_CHANGED` | Quality distribution shifted | old_avg, new_avg           |
| `HIGH_QUALITY_DATA_AVAILABLE`  | High-quality data ready      | high_quality_count         |
| `LOW_QUALITY_DATA_WARNING`     | Quality below threshold      | avg_quality, threshold     |
| `TRAINING_BLOCKED_BY_QUALITY`  | Quality too low              | required_quality           |
| `QUALITY_FEEDBACK_ADJUSTED`    | Quality feedback updated     | old_weight, new_weight     |
| `QUALITY_PENALTY_APPLIED`      | Penalty applied              | penalty_factor             |
| `QUALITY_CHECK_REQUESTED`      | On-demand check requested    | reason                     |
| `QUALITY_CHECK_FAILED`         | Quality check failed         | error                      |

### Cluster Health Events

| Event                 | Description               | Key Fields                       |
| --------------------- | ------------------------- | -------------------------------- |
| `HOST_ONLINE`         | Node came online          | node_id, host_type, capabilities |
| `HOST_OFFLINE`        | Node went offline         | node_id, reason                  |
| `NODE_UNHEALTHY`      | Health check failed       | health_score, issues             |
| `NODE_RECOVERED`      | Node recovered            | recovery_type, offline_duration  |
| `NODE_ACTIVATED`      | Node activated by cluster | node_id                          |
| `NODE_TERMINATED`     | Node terminated           | node_id, reason                  |
| `NODE_OVERLOADED`     | Node overloaded           | cpu_util, gpu_util, memory_util  |
| `HEALTH_CHECK_PASSED` | Health check succeeded    | health_score                     |
| `HEALTH_CHECK_FAILED` | Health check failed       | reason, error                    |
| `HEALTH_ALERT`        | General health warning    | alert_type, severity             |

### Work Queue Events

| Event            | Description                | Key Fields          |
| ---------------- | -------------------------- | ------------------- |
| `WORK_QUEUED`    | Work added to queue        | work_type, priority |
| `WORK_CLAIMED`   | Work claimed by node       | node_id, work_id    |
| `WORK_STARTED`   | Work execution started     | work_id             |
| `WORK_COMPLETED` | Work finished successfully | work_id, duration   |
| `WORK_FAILED`    | Work failed permanently    | work_id, error      |
| `WORK_RETRY`     | Work will retry            | work_id, attempt    |
| `WORK_TIMEOUT`   | Work timed out             | work_id             |
| `WORK_CANCELLED` | Work cancelled             | work_id, reason     |
| `JOB_PREEMPTED`  | Job preempted              | job_id, reason      |

### Daemon Lifecycle Events

| Event                         | Description                 | Key Fields                     |
| ----------------------------- | --------------------------- | ------------------------------ |
| `DAEMON_STARTED`              | Daemon started              | daemon_type, config            |
| `DAEMON_STOPPED`              | Daemon stopped              | daemon_type, reason            |
| `DAEMON_STATUS_CHANGED`       | Daemon status changed       | old_status, new_status         |
| `COORDINATOR_HEARTBEAT`       | Coordinator liveness        | health_score, events_processed |
| `COORDINATOR_HEALTH_DEGRADED` | Coordinator degraded        | reason, issues                 |
| `COORDINATOR_SHUTDOWN`        | Coordinator shutting down   | remaining_tasks                |
| `COORDINATOR_INIT_FAILED`     | Coordinator failed to start | error                          |

### Recovery Events

| Event                 | Description            | Key Fields                      |
| --------------------- | ---------------------- | ------------------------------- |
| `RECOVERY_INITIATED`  | Recovery started       | recovery_type, target           |
| `RECOVERY_COMPLETED`  | Recovery succeeded     | duration_seconds, actions_taken |
| `RECOVERY_FAILED`     | Recovery failed        | error                           |
| `REGRESSION_DETECTED` | Performance regression | severity, regression_amount     |
| `REGRESSION_CLEARED`  | Regression resolved    | model_id                        |

### Resource Events

| Event                      | Description              | Key Fields                   |
| -------------------------- | ------------------------ | ---------------------------- |
| `CLUSTER_CAPACITY_CHANGED` | Cluster capacity changed | old_capacity, new_capacity   |
| `NODE_CAPACITY_UPDATED`    | Node capacity updated    | node_id, resources           |
| `BACKPRESSURE_ACTIVATED`   | Backpressure activated   | level, resource_type         |
| `BACKPRESSURE_RELEASED`    | Backpressure released    | previous_level, duration     |
| `IDLE_RESOURCE_DETECTED`   | Idle resource found      | resource_type, idle_duration |
| `RESOURCE_CONSTRAINT`      | Resource pressure        | constraint_type, utilization |

### Optimization Events

| Event                     | Description                        | Key Fields                       |
| ------------------------- | ---------------------------------- | -------------------------------- |
| `CMAES_TRIGGERED`         | CMA-ES optimization started        | parameters_searched, generations |
| `CMAES_COMPLETED`         | CMA-ES finished                    | best_params, improvement         |
| `NAS_TRIGGERED`           | Neural Architecture Search started | search_space                     |
| `NAS_COMPLETED`           | NAS finished                       | best_architecture                |
| `PLATEAU_DETECTED`        | Training plateau detected          | epochs_since_improvement         |
| `HYPERPARAMETER_UPDATED`  | Hyperparameter changed             | param_name, old_value, new_value |
| `ADAPTIVE_PARAMS_CHANGED` | Adaptive parameters changed        | params_changed, elo_velocity     |

### Curriculum Events

| Event                   | Description                 | Key Fields                 |
| ----------------------- | --------------------------- | -------------------------- |
| `CURRICULUM_REBALANCED` | Weights rebalanced          | old_weights, new_weights   |
| `CURRICULUM_ADVANCED`   | Advanced to harder tier     | old_tier, new_tier         |
| `WEIGHT_UPDATED`        | Config weight updated       | old_weight, new_weight     |
| `EXPLORATION_ADJUSTED`  | Exploration changed         | old_strategy, new_strategy |
| `EXPLORATION_BOOST`     | Exploration boost requested | boost_factor               |
| `OPPONENT_MASTERED`     | Opponent mastered           | opponent_type              |

### Leader Election Events

| Event             | Description         | Key Fields              |
| ----------------- | ------------------- | ----------------------- |
| `LEADER_ELECTED`  | New leader elected  | leader_id, term         |
| `LEADER_LOST`     | Leader lost         | previous_leader, reason |
| `LEADER_STEPDOWN` | Leader stepped down | reason                  |

### Task Lifecycle Events

| Event            | Description                  | Key Fields              |
| ---------------- | ---------------------------- | ----------------------- |
| `TASK_SPAWNED`   | Task spawned                 | task_id, task_type      |
| `TASK_HEARTBEAT` | Task heartbeat               | task_id, progress       |
| `TASK_COMPLETED` | Task completed               | task_id, result         |
| `TASK_FAILED`    | Task failed                  | task_id, error          |
| `TASK_ORPHANED`  | Task lost parent worker      | task_id, last_heartbeat |
| `TASK_CANCELLED` | Task cancelled               | task_id, reason         |
| `TASK_ABANDONED` | Task intentionally abandoned | task_id, reason         |

### Data Integrity Events

| Event                     | Description              | Key Fields       |
| ------------------------- | ------------------------ | ---------------- |
| `ORPHAN_GAMES_DETECTED`   | Unregistered games found | count, databases |
| `ORPHAN_GAMES_REGISTERED` | Orphans auto-registered  | count            |
| `REPAIR_STARTED`          | Repair job started       | target           |
| `REPAIR_COMPLETED`        | Repair succeeded         | target, duration |
| `REPAIR_FAILED`           | Repair failed            | target, error    |
| `DATABASE_CREATED`        | New database created     | path, config     |

### Lock Events

| Event               | Description       | Key Fields        |
| ------------------- | ----------------- | ----------------- |
| `LOCK_ACQUIRED`     | Lock acquired     | lock_name, holder |
| `LOCK_RELEASED`     | Lock released     | lock_name         |
| `LOCK_TIMEOUT`      | Lock timed out    | lock_name         |
| `DEADLOCK_DETECTED` | Deadlock detected | locks_involved    |

### Checkpoint Events

| Event               | Description       | Key Fields             |
| ------------------- | ----------------- | ---------------------- |
| `CHECKPOINT_SAVED`  | Checkpoint saved  | checkpoint_path, epoch |
| `CHECKPOINT_LOADED` | Checkpoint loaded | checkpoint_path        |

### Parity Events

| Event                         | Description           | Key Fields           |
| ----------------------------- | --------------------- | -------------------- |
| `PARITY_VALIDATION_STARTED`   | Parity check started  | games_to_validate    |
| `PARITY_VALIDATION_COMPLETED` | Parity check finished | success, parity_rate |
| `PARITY_FAILURE_RATE_CHANGED` | Failure rate changed  | old_rate, new_rate   |

### Cache Events

| Event               | Description       | Key Fields               |
| ------------------- | ----------------- | ------------------------ |
| `CACHE_INVALIDATED` | Cache invalidated | invalidation_type, count |

---

## Event Ordering Guarantees

### General Principles

1. **No Global Ordering**: Events from different sources are NOT ordered relative to each other. Don't assume `SELFPLAY_COMPLETE` from node A happens before `SELFPLAY_COMPLETE` from node B.

2. **Local Ordering**: Events emitted by the same producer are delivered in emission order to each subscriber.

3. **At-Most-Once Delivery**: Events are deduplicated by content hash. If the same event payload is emitted twice within the deduplication window (1 hour), the second is dropped.

4. **Eventual Consistency**: Cross-process events via SQLite may be delayed by the poll interval (default: 5 seconds).

### Pipeline Stage Ordering

The training pipeline enforces stage ordering via event dependencies:

```
SELFPLAY_COMPLETE
    ↓ (triggers sync)
NEW_GAMES_AVAILABLE
    ↓ (triggers export)
NPZ_EXPORT_COMPLETE
    ↓ (triggers training if threshold reached)
TRAINING_THRESHOLD_REACHED
    ↓
TRAINING_STARTED
    ↓
TRAINING_COMPLETED
    ↓ (triggers evaluation)
EVALUATION_COMPLETED
    ↓ (triggers promotion if criteria met)
MODEL_PROMOTED
    ↓ (triggers distribution)
MODEL_DISTRIBUTION_COMPLETE
```

### Ordering Guarantees by Category

| Category             | Guarantee                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| Training lifecycle   | `TRAINING_STARTED` → `TRAINING_PROGRESS` (0..N) → `TRAINING_COMPLETED`/`TRAINING_FAILED`         |
| Evaluation lifecycle | `EVALUATION_STARTED` → `EVALUATION_PROGRESS` (0..N) → `EVALUATION_COMPLETED`/`EVALUATION_FAILED` |
| Sync lifecycle       | `DATA_SYNC_STARTED` → `DATA_SYNC_COMPLETED`/`DATA_SYNC_FAILED`                                   |
| Recovery lifecycle   | `RECOVERY_INITIATED` → `RECOVERY_COMPLETED`/`RECOVERY_FAILED`                                    |
| Work lifecycle       | `WORK_QUEUED` → `WORK_CLAIMED` → `WORK_STARTED` → `WORK_COMPLETED`/`WORK_FAILED`                 |

### Cross-Process Event Ordering

Events that cross process boundaries via `CrossProcessEventQueue`:

1. Events are written to SQLite with monotonic `event_id`
2. Consumers poll in `event_id` order
3. Events are marked processed after handler completes
4. Failed handlers don't block other events (fire-and-forget mode)

```python
# Poll order is guaranteed
event_1 (event_id=100) processed before event_2 (event_id=101)

# But handler execution may overlap for async handlers
handler_1(event_1)  # May not complete before handler_2 starts
handler_2(event_2)  # Starts after event_2 polled
```

### Deduplication Behavior

Content-based deduplication uses SHA256 hash of `(event_type, payload)`:

```python
# These are considered duplicates (same hash):
await publish(TRAINING_COMPLETED, {"model_id": "abc123", "success": True})
await publish(TRAINING_COMPLETED, {"model_id": "abc123", "success": True})  # Dropped

# These are NOT duplicates (different payload):
await publish(TRAINING_COMPLETED, {"model_id": "abc123", "success": True})
await publish(TRAINING_COMPLETED, {"model_id": "xyz789", "success": True})  # Delivered
```

### Best Practices for Ordering

1. **Don't rely on timing**: Use event chaining instead of timing assumptions
2. **Include sequence numbers**: For multi-part operations, include `sequence` in payload
3. **Use idempotent handlers**: Handlers should tolerate duplicate delivery
4. **Chain via event subscription**: Trigger next stage by subscribing to previous stage's completion event

```python
# Good: Event chaining
async def on_training_complete(event):
    model_id = event.payload["model_id"]
    await start_evaluation(model_id)  # Trigger next stage

# Bad: Timing assumption
async def training_loop():
    await train_model()
    await asyncio.sleep(5)  # Hope training event was processed
    await start_evaluation()  # May race with event handlers
```

---

## Usage Examples

### Emitting Events

#### Using Event Emitters (Recommended)

```python
from app.coordination.event_emitters import (
    emit_training_complete,
    emit_selfplay_complete,
    emit_evaluation_complete,
    emit_promotion_complete,
)

# Emit training completion
await emit_training_complete(
    job_id="square8_2p_123",
    board_type="square8",
    num_players=2,
    success=True,
    final_loss=0.05,
    final_elo=1650.0,
    model_path="models/square8_2p_new.pth",
    epochs_completed=20,
)

# Emit selfplay completion
await emit_selfplay_complete(
    task_id="selfplay_001",
    board_type="hex8",
    num_players=2,
    games_generated=500,
    success=True,
    node_id="runpod-h100",
    duration_seconds=120.5,
    selfplay_type="gpu_accelerated",
)

# Emit evaluation completion
await emit_evaluation_complete(
    model_id="hex8_2p_v123",
    board_type="hex8",
    num_players=2,
    success=True,
    win_rate=0.75,
    elo_delta=50.0,
    games_played=100,
)
```

#### Using Event Router Directly

```python
from app.coordination.event_router import publish, DataEventType

# Publish with typed event
await publish(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={
        "config_key": "square8_2p",
        "model_id": "sq8_2p_v456",
        "val_loss": 0.05,
        "epochs": 20,
    },
    source="training_daemon",
)
```

### Subscribing to Events

#### Basic Subscription

```python
from app.coordination.event_router import subscribe, RouterEvent, DataEventType

async def handle_training_complete(event: RouterEvent):
    config = event.payload.get("config_key")
    model_id = event.payload.get("model_id")
    print(f"Training complete for {config}: {model_id}")

    # Trigger evaluation
    await start_evaluation(model_id)

# Subscribe to specific event
subscribe(DataEventType.TRAINING_COMPLETED, handle_training_complete)
```

#### Pipeline Coordinator Pattern

```python
from app.coordination.event_router import get_router, DataEventType, StageEvent

class MyCoordinator:
    def __init__(self):
        self.router = get_router()
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers."""
        self.router.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_done)
        self.router.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_done)
        self.router.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_eval_done)

    async def _on_selfplay_done(self, event):
        if event.payload.get("success"):
            games = event.payload.get("games_generated", 0)
            print(f"Selfplay generated {games} games, triggering sync")
            await self.trigger_sync()

    async def _on_training_done(self, event):
        model_id = event.payload.get("model_id")
        print(f"Training done: {model_id}, triggering evaluation")
        await self.trigger_evaluation(model_id)

    async def _on_eval_done(self, event):
        elo = event.payload.get("elo")
        if elo > 1700:
            print(f"High Elo {elo}, triggering promotion")
            await self.trigger_promotion(event.payload.get("model_id"))
```

---

## Event Flow Diagrams

### Full Training Pipeline

```
┌─────────────────┐
│ SELFPLAY_       │
│ COMPLETE        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NEW_GAMES_      │
│ AVAILABLE       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DATA_SYNC_      │
│ COMPLETED       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NPZ_EXPORT_     │
│ COMPLETE        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TRAINING_       │
│ STARTED         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TRAINING_       │
│ COMPLETED       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EVALUATION_     │
│ COMPLETED       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MODEL_          │
│ PROMOTED        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MODEL_          │
│ DISTRIBUTION_   │
│ COMPLETE        │
└─────────────────┘
```

### Quality Feedback Loop

```
┌─────────────────┐
│ SELFPLAY_       │
│ COMPLETE        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ QUALITY_SCORE_  │
│ UPDATED         │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────────────┐
│Quality │ │ Quality        │
│ Good   │ │ Degraded       │
└───┬────┘ └────────┬───────┘
    │               │
    │               ▼
    │      ┌─────────────────┐
    │      │ QUALITY_        │
    │      │ PENALTY_        │
    │      │ APPLIED         │
    │      └────────┬────────┘
    │               │
    │               ▼
    │      ┌─────────────────┐
    │      │ SELFPLAY_RATE_  │
    │      │ CHANGED         │
    │      └─────────────────┘
    │
    ▼
┌─────────────────┐
│ HIGH_QUALITY_   │
│ DATA_AVAILABLE  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TRAINING_       │
│ THRESHOLD_      │
│ REACHED         │
└─────────────────┘
```

### Error Recovery Flow

```
┌─────────────────┐
│ TRAINING_       │
│ COMPLETED       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EVALUATION_     │
│ COMPLETED       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌─────────────────┐
│No      │ │ REGRESSION_     │
│Regress │ │ DETECTED        │
└───┬────┘ └────────┬────────┘
    │               │
    │               ▼
    │      ┌─────────────────┐
    │      │ TRAINING_       │
    │      │ ROLLBACK_       │
    │      │ NEEDED          │
    │      └────────┬────────┘
    │               │
    │               ▼
    │      ┌─────────────────┐
    │      │ RECOVERY_       │
    │      │ INITIATED       │
    │      └────────┬────────┘
    │               │
    │          ┌────┴────┐
    │          │         │
    │          ▼         ▼
    │      ┌───────┐ ┌────────┐
    │      │Success│ │ Failed │
    │      └───┬───┘ └───┬────┘
    │          │         │
    │          ▼         ▼
    │    ┌─────────┐ ┌──────────┐
    │    │RECOVERY_│ │RECOVERY_ │
    │    │COMPLETE │ │FAILED    │
    │    └────┬────┘ └──────────┘
    │         │
    │         ▼
    │    ┌─────────────────┐
    │    │ TRAINING_       │
    │    │ ROLLBACK_       │
    │    │ COMPLETED       │
    │    └────────┬────────┘
    │             │
    │             ▼
    │    ┌─────────────────┐
    │    │ CACHE_          │
    │    │ INVALIDATED     │
    │    └─────────────────┘
    │
    ▼
┌─────────────────┐
│ MODEL_          │
│ PROMOTED        │
└─────────────────┘
```

---

## Best Practices

### 1. Use Event Emitters

Always prefer using typed event emitters over direct event publishing:

```python
# Good
await emit_training_complete(job_id="...", board_type="hex8", num_players=2, ...)

# Avoid (unless you need custom events)
await publish(event_type="TRAINING_COMPLETED", payload={...})
```

### 2. Handle Failures Gracefully

Event handlers should be resilient:

```python
async def handle_training_complete(event: RouterEvent):
    try:
        model_id = event.payload.get("model_id")
        if not model_id:
            logger.warning("Training complete event missing model_id")
            return

        await start_evaluation(model_id)
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        # Don't raise - let other handlers run
```

### 3. Subscribe Early

Register event handlers during coordinator initialization:

```python
class MyCoordinator:
    def __init__(self):
        self.router = get_router()
        self._register_handlers()  # Subscribe early

    def _register_handlers(self):
        self.router.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training)
```

### 4. Clean Up Subscriptions

Unsubscribe when coordinator shuts down:

```python
async def shutdown(self):
    self.router.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training)
```

### 5. Use Meaningful Sources

Always specify the source when emitting events:

```python
await publish(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={...},
    source="training_coordinator",  # Clear origin
)
```

---

## Migration Guide

### From Direct EventBus to Router

**Old code**:

```python
from app.distributed.data_events import get_event_bus, DataEvent, DataEventType

bus = get_event_bus()
await bus.publish(DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"model_id": "..."}
))
```

**New code**:

```python
from app.coordination.event_router import publish

await publish(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"model_id": "..."},
    source="my_component"
)
```

### From StageEventBus to Router

**Old code**:

```python
from app.coordination.stage_events import get_event_bus, StageCompletionResult, StageEvent

bus = get_event_bus()
await bus.emit(StageCompletionResult(
    event=StageEvent.TRAINING_COMPLETE,
    success=True,
    ...
))
```

**New code**:

```python
from app.coordination.event_emitters import emit_training_complete

await emit_training_complete(
    job_id="...",
    board_type="hex8",
    num_players=2,
    success=True,
    ...
)
```

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("app.coordination.event_router").setLevel(logging.DEBUG)
```

### Inspect Event Flow

```python
from app.coordination.event_router import get_router

router = get_router()

# See what events have been routed
stats = router.get_stats()
print(f"Events by type: {stats['events_routed_by_type']}")

# Check for event loops
if stats['duplicates_prevented'] > 1000:
    print("WARNING: High duplicate rate, possible event loop")
```

### Validate Event Flow

```python
from app.coordination.event_router import validate_event_flow

result = validate_event_flow()
if not result["healthy"]:
    print("Event system issues:")
    for issue in result["issues"]:
        print(f"  - {issue}")
```

---

## See Also

- `event_router.py` - Unified event routing implementation
- `event_emitters.py` - Typed event emitter functions
- `data_events.py` - DataEventType definitions
- `stage_events.py` - StageEvent definitions
- `ARCHITECTURE.md` - System architecture overview
- `COORDINATOR_ARCHITECTURE.md` - Coordinator pattern documentation
