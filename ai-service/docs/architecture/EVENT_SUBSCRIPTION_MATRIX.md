# Event Subscription Matrix

**Last Updated**: December 30, 2025

**Integration Status**: VERIFIED COMPLETE (Dec 30, 2025)

All critical events have been verified to have proper emitters and subscribers. Events that
were previously thought to be "orphaned" (TASK_SPAWNED, TASK_HEARTBEAT, QUALITY_DEGRADED,
PROGRESS_STALL_DETECTED) have been verified to have 2-5 active subscribers each.

This document provides a comprehensive matrix of all 211 event types in the RingRift coordination layer,
their emitters, and subscribers. This is the single source of truth for event integration.

## Quick Reference

| Category   | Event Count | Description                 |
| ---------- | ----------- | --------------------------- |
| Training   | 15          | Training lifecycle events   |
| Selfplay   | 12          | Selfplay coordination       |
| Evaluation | 8           | Model evaluation pipeline   |
| Promotion  | 9           | Model promotion/rollback    |
| Sync       | 11          | Data synchronization        |
| Health     | 26          | Node and cluster health     |
| Daemon     | 13          | Daemon lifecycle            |
| Quality    | 10          | Data quality signals        |
| Curriculum | 8           | Curriculum adjustments      |
| Work Queue | 12          | Distributed work management |
| Other      | 12+         | Miscellaneous events        |

---

## 1. Training Events

### TRAINING_STARTED

- **Emitter**: `training_coordinator.py`, `train.py`
- **Subscribers**:
  - `unified_idle_shutdown_daemon.py` - Resets idle timer
  - `unified_feedback.py` - Tracks training state

### TRAINING_COMPLETED

- **Emitter**: `training_coordinator.py`, `train.py`
- **Subscribers**:
  - `unified_queue_populator.py` - Updates work queue priorities
  - `unified_feedback.py` - Triggers feedback loop
  - `feedback_loop_controller.py` - Triggers evaluation
  - `data_pipeline_orchestrator.py` - Advances pipeline stage

### TRAINING_FAILED

- **Emitter**: `training_coordinator.py`, `training_trigger_daemon.py`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks failure patterns
  - `daemon_manager.py` - May trigger recovery

### TRAINING_PROGRESS

- **Emitter**: `training_coordinator.py`
- **Subscribers**:
  - `unified_distribution_daemon.py` - Prefetches model for distribution

### TRAINING_BLOCKED_BY_QUALITY

- **Emitter**: `quality_gating.py`, `training_coordinator.py`
- **Subscribers**:
  - `unified_queue_populator.py` - Pauses training queue
  - `training_coordinator.py` - Waits for quality improvement

### TRAINING_LOCK_ACQUIRED

- **Emitter**: `training_coordinator.py:1798-1811`
- **Subscribers**: (Internal tracking only)

### TRAINING_SLOT_UNAVAILABLE

- **Emitter**: `training_coordinator.py:1813-1844`
- **Subscribers**: (Internal tracking only)

### TRAINING_LOSS_ANOMALY

- **Emitter**: `anomaly_detector.py`
- **Subscribers**:
  - `unified_feedback.py` - Adjusts learning rate

---

## 2. Selfplay Events

### SELFPLAY_COMPLETE

- **Emitter**: `selfplay_runner.py`, P2P orchestrator
- **Subscribers**:
  - `data_consolidation_daemon.py` - Consolidates game data
  - `unified_feedback.py` - Updates feedback signals
  - `unified_idle_shutdown_daemon.py` - Resets idle timer
  - `unified_queue_populator.py` - Updates queue stats
  - `coordination_bootstrap.py` - Triggers sync

### NEW_GAMES_AVAILABLE

- **Emitter**: `auto_export_daemon.py`, P2P orchestrator
- **Subscribers**:
  - `data_consolidation_daemon.py` - Triggers consolidation
  - `unified_replication_daemon.py` - Triggers replication
  - `data_pipeline_orchestrator.py` - Checks export threshold
  - `training_coordinator.py` - May trigger training
  - `unified_queue_populator.py` - Updates priorities

### SELFPLAY_TARGET_UPDATED

- **Emitter**: `selfplay_scheduler.py`
- **Subscribers**:
  - `daemon_event_handlers.py` - Adjusts daemon behavior
  - `unified_queue_populator.py` - Updates targets

### SELFPLAY_ALLOCATION_UPDATED

- **Emitter**: `selfplay_scheduler.py`
- **Subscribers**:
  - `curriculum_integration.py` - Adjusts curriculum weights

### SELFPLAY_RATE_CHANGED

- **Emitter**: `selfplay_scheduler.py`
- **Subscribers**:
  - `curriculum_integration.py` - Adjusts pacing

---

## 3. Evaluation Events

### EVALUATION_STARTED

- **Emitter**: `evaluation_daemon.py`
- **Subscribers**:
  - `coordination_bootstrap.py` - Tracks evaluation state

### EVALUATION_COMPLETED

- **Emitter**: `evaluation_daemon.py`, `game_gauntlet.py`
- **Subscribers**:
  - `auto_promotion_daemon.py` - Checks promotion threshold
  - `curriculum_integration.py` - Updates curriculum
  - `unified_feedback.py` - Adjusts training parameters
  - `feedback_loop_controller.py` - Closes feedback loop
  - `daemon_manager.py` - Routes to feedback accelerator

### EVALUATION_FAILED

- **Emitter**: `evaluation_daemon.py`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks failures

### EVALUATION_BACKPRESSURE

- **Emitter**: `evaluation_daemon.py` (Dec 29, 2025)
- **Subscribers**:
  - `training_coordinator.py` - Pauses training
  - `data_pipeline_orchestrator.py` - Pauses export

### EVALUATION_BACKPRESSURE_RELEASED

- **Emitter**: `evaluation_daemon.py` (Dec 29, 2025)
- **Subscribers**:
  - `training_coordinator.py` - Resumes training
  - `data_pipeline_orchestrator.py` - Resumes export

---

## 4. Promotion Events

### MODEL_PROMOTED

- **Emitter**: `promotion_controller.py`, `auto_promotion_daemon.py`
- **Subscribers**:
  - `unified_distribution_daemon.py` - Distributes model
  - `curriculum_integration.py` - Updates curriculum
  - `unified_feedback.py` - Closes promotion loop
  - `cache_coordination_orchestrator.py` - Updates caches

### PROMOTION_ROLLED_BACK

- **Emitter**: `promotion_controller.py`
- **Subscribers**:
  - `cache_coordination_orchestrator.py` - Reverts caches
  - `training_coordinator.py` - May re-train

### MODEL_DISTRIBUTION_STARTED

- **Emitter**: `unified_distribution_daemon.py`
- **Subscribers**: (Internal tracking)

### MODEL_DISTRIBUTION_COMPLETE

- **Emitter**: `unified_distribution_daemon.py`
- **Subscribers**:
  - Wait functions for sync completion

### MODEL_DISTRIBUTION_FAILED

- **Emitter**: `unified_distribution_daemon.py`
- **Subscribers**:
  - `coordination_bootstrap.py` - Handles failure
  - `unified_distribution_daemon.py` - Retry logic

### MODEL_EVALUATION_BLOCKED

- **Emitter**: Various
- **Subscribers**:
  - `unified_distribution_daemon.py` - Waits for evaluation

### PROMOTION_CANDIDATE

- **Emitter**: `auto_promotion_daemon.py`, `gauntlet_runner.py`
- **Purpose**: Model passed gauntlet and is candidate for promotion
- **Payload**: `model_path`, `config_key`, `gauntlet_results`, `elo_gain`
- **Subscribers**:
  - `promotion_controller.py` - Evaluates promotion decision
  - `curriculum_integration.py` - Prepares for potential promotion
- **Added**: December 2025

### PROMOTION_STARTED

- **Emitter**: `promotion_controller.py`
- **Purpose**: Model promotion process has started
- **Payload**: `model_path`, `config_key`, `promotion_type`
- **Subscribers**:
  - `unified_distribution_daemon.py` - Prepares for distribution
  - `coordination_bootstrap.py` - Tracks promotion state
- **Added**: December 2025

### PROMOTION_REJECTED

- **Emitter**: `promotion_controller.py`, `auto_promotion_daemon.py`
- **Purpose**: Model failed promotion criteria (regression, poor gauntlet, etc.)
- **Payload**: `model_path`, `config_key`, `rejection_reason`, `gauntlet_results`
- **Subscribers**:
  - `curriculum_integration.py` - May adjust training parameters
  - `unified_feedback.py` - Updates feedback signals
- **Added**: December 2025

---

## 5. Sync Events

### DATA_SYNC_COMPLETED

- **Emitter**: `auto_sync_daemon.py`, P2P orchestrator
- **Subscribers**:
  - `data_cleanup_daemon.py` - Triggers cleanup
  - `coordination_bootstrap.py` - Updates sync state
  - `unified_replication_daemon.py` - Verifies replication
  - `transfer_verification.py` - Verifies integrity
  - `data_pipeline_orchestrator.py` - May advance pipeline

### DATA_SYNC_FAILED

- **Emitter**: `auto_sync_daemon.py`
- **Subscribers**:
  - `coordination_bootstrap.py` - Handles failure

### SYNC_NODE_UNREACHABLE

- **Emitter**: `auto_sync_daemon.py`
- **Subscribers**: (Logged for debugging)

### SYNC_TRIGGERED

- **Emitter**: `training_freshness.py`
- **Subscribers**: (Logging only)

### P2P_MODEL_SYNCED

- **Emitter**: P2P orchestrator
- **Subscribers**:
  - `coordination_bootstrap.py` - Model sync complete

### DATA_FRESH / DATA_STALE

- **Emitter**: `training_freshness.py`
- **Subscribers**:
  - `unified_feedback.py` - Adjusts training decisions

### GAME_SYNCED

- **Emitter**: `auto_sync_daemon.py`, P2P orchestrator
- **Purpose**: Individual game or batch of games synced to node
- **Payload**: `game_ids`, `source_node`, `target_node`, `sync_type` (ephemeral/broadcast)
- **Subscribers**:
  - `data_consolidation_daemon.py` - Updates local game counts
  - `training_freshness.py` - Refreshes data staleness tracking
- **Added**: December 2025 (ephemeral sync events)

---

## 6. Health Events

### HOST_OFFLINE

- **Emitter**: P2P orchestrator, `node_monitor.py`
- **Subscribers**:
  - `cluster_watchdog_daemon.py` - Updates cluster view
  - `unified_health_manager.py` - Updates health score
  - `unified_replication_daemon.py` - Adjusts replication targets
  - `daemon_event_handlers.py` - Triggers failover

### HOST_ONLINE

- **Emitter**: P2P orchestrator, `node_monitor.py`
- **Subscribers**:
  - `cluster_watchdog_daemon.py` - Updates cluster view
  - `unified_health_manager.py` - Restores health score
  - `daemon_event_handlers.py` - Restores services

### NODE_UNHEALTHY

- **Emitter**: `unified_health_manager.py`, `auto_scaler.py`
- **Subscribers**:
  - `availability/node_monitor.py` - Triggers recovery

### NODE_RECOVERED

- **Emitter**: `availability/node_monitor.py`
- **Subscribers**:
  - `unified_health_manager.py` - Updates health

### P2P_CLUSTER_HEALTHY

- **Emitter**: P2P orchestrator
- **Subscribers**:
  - `auto_scaler.py` - Enables scaling
  - `cluster_watchdog_daemon.py` - Updates view
  - `training_coordinator.py` - Enables training

### P2P_CLUSTER_UNHEALTHY

- **Emitter**: P2P orchestrator
- **Subscribers**:
  - `auto_scaler.py` - Pauses scaling
  - `cluster_watchdog_daemon.py` - Alerts
  - `training_coordinator.py` - May pause training

### LEADER_ELECTED

- **Emitter**: P2P orchestrator
- **Subscribers**:
  - `daemon_event_handlers.py` - Adjusts daemon roles

### PROGRESS_STALL_DETECTED

- **Emitter**: `progress_watchdog_daemon.py`
- **Purpose**: Detects when a config's Elo progress stalls (6+ hours without positive velocity)
- **Payload**: `config_key`, `action`, `stall_duration_hours`, `recent_elo_velocity`
- **Subscribers**:
  - `selfplay_scheduler.py` - Boosts selfplay priority for stalled config
  - `daemon_event_handlers.py` - May trigger recovery actions
- **Added**: December 2025 (48-hour autonomous operation)

### PROGRESS_RECOVERED

- **Emitter**: `progress_watchdog_daemon.py`
- **Purpose**: Indicates a previously stalled config has resumed positive Elo progress
- **Payload**: `config_key`, `recovery_duration_hours`, `current_elo`, `velocity`
- **Subscribers**:
  - `selfplay_scheduler.py` - Normalizes priority
- **Added**: December 2025 (48-hour autonomous operation)

### ORPHAN_GAMES_DETECTED

- **Emitter**: `orphan_detection_daemon.py`
- **Purpose**: Detects games that are orphaned (not in any database, on terminating nodes)
- **Payload**: `node_id`, `orphan_count`, `game_ids`, `total_moves`
- **Subscribers**:
  - `data_pipeline_orchestrator.py` - Triggers recovery sync
  - `unified_replication_daemon.py` - Initiates emergency replication
- **Added**: December 2025

### MEMORY_PRESSURE

- **Emitter**: `memory_monitor_daemon.py`
- **Purpose**: GPU VRAM or system memory is critically low
- **Payload**: `node_id`, `vram_used_pct`, `system_used_pct`, `threshold`
- **Subscribers**:
  - `selfplay_scheduler.py` - Pauses job spawning on affected nodes
  - `daemon_event_handlers.py` - May trigger process cleanup
- **Added**: December 2025 (48-hour autonomous operation)

### HEALTH_ALERT

- **Emitter**: `daemon_event_handlers.py`
- **Subscribers**:
  - `auto_scaler.py` - Scales resources

### RECOVERY_INITIATED / RECOVERY_COMPLETED / RECOVERY_FAILED

- **Emitter**: `availability/recovery_engine.py`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks recovery state

### NODE_SUSPECT

- **Emitter**: `node_availability/daemon.py`, `unified_health_manager.py`
- **Purpose**: Node health is degraded but not yet offline (suspect state in SWIM protocol)
- **Payload**: `node_id`, `reason`, `suspect_since`, `health_score`
- **Subscribers**:
  - `selfplay_scheduler.py` - Reduces job allocation to suspect nodes
  - `sync_router.py` - Prefers healthy nodes for sync targets
- **Added**: December 2025

### NODE_RETIRED

- **Emitter**: `node_availability/daemon.py`, P2P orchestrator
- **Purpose**: Node has been permanently removed from cluster (offline too long)
- **Payload**: `node_id`, `retired_at`, `reason`, `data_migrated`
- **Subscribers**:
  - `unified_health_manager.py` - Removes from active node list
  - `sync_router.py` - Updates routing tables
- **Added**: December 2025

### NODE_INCOMPATIBLE_WITH_WORKLOAD

- **Emitter**: `job_dispatcher.py`, `node_selector.py`
- **Purpose**: Node cannot run assigned workload (missing deps, wrong GPU, etc.)
- **Payload**: `node_id`, `workload_type`, `reason`, `alternatives`
- **Subscribers**:
  - `selfplay_scheduler.py` - Excludes node for this workload type
- **Added**: December 2025

### HEALTH_CHECK_PASSED / HEALTH_CHECK_FAILED

- **Emitter**: `health_check_orchestrator.py`
- **Purpose**: Per-node health check result
- **Payload**: `node_id`, `check_type`, `latency_ms`, `details`
- **Subscribers**:
  - `unified_health_manager.py` - Updates node health scores
  - `daemon_event_handlers.py` - May trigger recovery on failures
- **Added**: December 2025

### SYNC_STALLED

- **Emitter**: `auto_sync_daemon.py`, `sync_router.py`
- **Purpose**: Data sync has not completed within expected time
- **Payload**: `sync_id`, `stall_duration_seconds`, `pending_bytes`, `target_node`
- **Subscribers**:
  - `daemon_event_handlers.py` - May trigger alternative sync path
  - `unified_health_manager.py` - Tracks sync health
- **Added**: December 2025

### NODE_OVERLOADED

- **Emitter**: `resource_monitor.py`, `selfplay_scheduler.py`
- **Purpose**: Node has too many active jobs or resource utilization is high
- **Payload**: `node_id`, `active_jobs`, `cpu_pct`, `gpu_pct`, `memory_pct`
- **Subscribers**:
  - `selfplay_scheduler.py` - Reduces job allocation
  - `job_dispatcher.py` - Routes jobs elsewhere
- **Added**: December 2025 (48-hour autonomous operation)

### IDLE_RESOURCE_DETECTED

- **Emitter**: `idle_resource_daemon.py`
- **Purpose**: GPU or CPU resources are idle and available for work
- **Payload**: `node_id`, `idle_gpus`, `idle_cpus`, `idle_duration_seconds`
- **Subscribers**:
  - `selfplay_scheduler.py` - Spawns jobs on idle resources
  - `job_dispatcher.py` - Prioritizes idle nodes
- **Added**: December 2025

---

## 7. Daemon Events

### DAEMON_STARTED / DAEMON_STOPPED

- **Emitter**: `daemon_manager.py`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks daemon state

### DAEMON_STATUS_CHANGED

- **Emitter**: `daemon_watchdog.py`, `daemon_manager.py`
- **Subscribers**:
  - `daemon_event_handlers.py` - Routes to handlers
  - `unified_health_manager.py` - Updates health

### DAEMON_PERMANENTLY_FAILED

- **Emitter**: `daemon_manager.py`
- **Subscribers**:
  - `unified_health_manager.py` - Triggers alert

### DAEMON_CRASH_LOOP_DETECTED

- **Emitter**: `daemon_manager.py`, `daemon_watchdog.py`
- **Purpose**: Daemon has restarted too many times in short period (crash loop)
- **Payload**: `daemon_type`, `restart_count`, `last_error`, `window_seconds`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks daemon health
  - `daemon_event_handlers.py` - May escalate or pause daemon
- **Added**: December 2025

### ALL_CRITICAL_DAEMONS_READY

- **Emitter**: `daemon_manager.py`
- **Subscribers**: (Startup coordination)

### COORDINATOR_HEARTBEAT / COORDINATOR_HEALTHY / COORDINATOR_UNHEALTHY

- **Emitter**: `coordinator_health_monitor_daemon.py`
- **Subscribers**:
  - `unified_health_manager.py` - Tracks coordinator health

### COORDINATOR_INIT_FAILED / COORDINATOR_SHUTDOWN

- **Emitter**: Various coordinators
- **Subscribers**:
  - `coordinator_health_monitor_daemon.py` - Routes events

---

## 8. Quality Events

### QUALITY_SCORE_UPDATED

- **Emitter**: `unified_quality.py`
- **Subscribers**:
  - `curriculum_integration.py` - Adjusts curriculum

### QUALITY_DEGRADED

- **Emitter**: `quality_monitor_daemon.py`
- **Subscribers**:
  - `unified_feedback.py` - Adjusts training

### QUALITY_PENALTY_APPLIED

- **Emitter**: `quality_gating.py`
- **Subscribers**:
  - `curriculum_integration.py` - Reduces config weight
  - `unified_feedback.py` - Adjusts parameters

### QUALITY_FEEDBACK_ADJUSTED

- **Emitter**: `quality_feedback_handler.py`
- **Subscribers**:
  - `curriculum_integration.py` - Updates weights

### DATA_QUALITY_ALERT

- **Emitter**: `quality_monitor_daemon.py`
- **Subscribers**:
  - `data_cleanup_daemon.py` - Triggers cleanup

### LOW_QUALITY_DATA_WARNING

- **Emitter**: `quality_gating.py`
- **Subscribers**:
  - `training_coordinator.py` - May skip training

---

## 9. Curriculum Events

### CURRICULUM_REBALANCED

- **Emitter**: `curriculum_feedback_handler.py`
- **Subscribers**:
  - `selfplay_scheduler.py` - Updates allocation

### CURRICULUM_ADVANCED

- **Emitter**: `curriculum_progression.py`
- **Subscribers**:
  - `daemon_manager.py` - Routes to curriculum handlers

### CURRICULUM_ADVANCEMENT_NEEDED

- **Emitter**: `training_trigger_daemon.py`
- **Subscribers**:
  - `curriculum_integration.py` - Advances curriculum

### TIER_PROMOTION / CROSSBOARD_PROMOTION

- **Emitter**: `curriculum_progression.py`
- **Subscribers**:
  - `curriculum_integration.py` - Updates weights

### OPPONENT_MASTERED

- **Emitter**: `opponent_tracker_integration.py`
- **Subscribers**:
  - `curriculum_integration.py` - Advances opponent tier

---

## 10. Regression Events

### REGRESSION_DETECTED

- **Emitter**: `regression_detector.py`
- **Subscribers**:
  - `training_coordinator.py` - May pause training
  - `unified_feedback.py` - Adjusts parameters
  - `unified_health_manager.py` - Updates health

### REGRESSION_CRITICAL

- **Emitter**: `regression_detector.py`
- **Subscribers**:
  - `training_coordinator.py` - Pauses training
  - `curriculum_integration.py` - Emergency adjustment
  - `daemon_event_handlers.py` - Triggers recovery
  - `daemon_manager.py` - Routes to handlers
  - `unified_health_manager.py` - Critical alert

### REGRESSION_CLEARED

- **Emitter**: `regression_detector.py`
- **Subscribers**:
  - `training_coordinator.py` - Resumes training

### TRAINING_ROLLBACK_NEEDED

- **Emitter**: `regression_detector.py`
- **Subscribers**:
  - `training_coordinator.py` - Triggers rollback

---

## 11. Work Queue Events

### WORK_QUEUED / WORK_CLAIMED / WORK_STARTED

- **Emitter**: `work_queue.py`
- **Subscribers**:
  - `unified_idle_shutdown_daemon.py` - Resets idle timer

### WORK_COMPLETED / WORK_FAILED / WORK_TIMEOUT

- **Emitter**: `work_queue.py`
- **Subscribers**:
  - `unified_queue_populator.py` - Updates queue stats

### WORK_CANCELLED / WORK_RETRY

- **Emitter**: `work_queue.py`
- **Subscribers**: (Internal tracking)

### TASK_SPAWNED / TASK_FAILED / TASK_ABANDONED

- **Emitter**: `task_coordinator.py`, P2P orchestrator
- **Subscribers**:
  - `unified_idle_shutdown_daemon.py` - TASK_SPAWNED
  - `unified_health_manager.py` - TASK_FAILED
  - `unified_queue_populator.py` - TASK_ABANDONED

### BATCH_SCHEDULED

- **Emitter**: `job_dispatcher.py`, `selfplay_scheduler.py`
- **Purpose**: Batch of selfplay/training jobs scheduled for execution
- **Payload**: `batch_id`, `job_count`, `config_keys`, `target_nodes`, `priority`
- **Subscribers**:
  - `unified_queue_populator.py` - Updates queue stats
  - `resource_monitor.py` - Tracks expected resource usage
- **Added**: December 2025

### BATCH_DISPATCHED

- **Emitter**: `job_dispatcher.py`, P2P orchestrator
- **Purpose**: Batch of jobs has been dispatched to target nodes
- **Payload**: `batch_id`, `dispatched_count`, `nodes`, `dispatch_time_ms`
- **Subscribers**:
  - `selfplay_scheduler.py` - Updates job tracking
  - `unified_idle_shutdown_daemon.py` - Resets idle timers
- **Added**: December 2025

---

## 12. Other Events

### ELO_UPDATED / ELO_SIGNIFICANT_CHANGE / ELO_VELOCITY_CHANGED

- **Emitter**: `elo_sync_manager.py`, P2P orchestrator
- **Subscribers**:
  - `unified_queue_populator.py` - Updates priorities
  - `curriculum_integration.py` - Adjusts weights
  - `unified_feedback.py` - Adjusts training

### PLATEAU_DETECTED

- **Emitter**: `plateau_detector.py`
- **Subscribers**:
  - `unified_feedback.py` - Triggers exploration boost

### EXPLORATION_BOOST

- **Emitter**: `plateau_handler.py`
- **Subscribers**:
  - `daemon_event_handlers.py` - Increases temperature

### BACKPRESSURE_ACTIVATED / BACKPRESSURE_RELEASED

- **Emitter**: `queue_monitor.py`
- **Subscribers**:
  - `daemon_event_handlers.py` - Adjusts production
  - `unified_queue_populator.py` - Pauses/resumes

### CONSOLIDATION_STARTED / CONSOLIDATION_COMPLETE

- **Emitter**: `data_consolidation_daemon.py`
- **Subscribers**:
  - `data_pipeline_orchestrator.py` - Advances stage

### DISK_SPACE_LOW

- **Emitter**: `disk_space_manager_daemon.py`
- **Subscribers**:
  - `daemon_event_handlers.py` - Triggers cleanup

### RESOURCE_CONSTRAINT

- **Emitter**: `daemon_manager.py`
- **Subscribers**:
  - `auto_scaler.py` - Scales resources

### NODE_TERMINATED

- **Emitter**: `unified_idle_shutdown_daemon.py`
- **Subscribers**: (Logging only)

### REPLICATION_ALERT

- **Emitter**: `unified_replication_daemon.py`
- **Subscribers**: (Alerting only)

### HYPERPARAMETER_UPDATED / ADAPTIVE_PARAMS_CHANGED

- **Emitter**: `hyperparameter_tuner.py`, `adaptive_training.py`
- **Subscribers**:
  - `daemon_manager.py` - Routes to handlers

### DEADLOCK_DETECTED

- **Emitter**: `deadlock_detector.py`, `resource_monitor.py`
- **Purpose**: Circular wait or resource deadlock detected in pipeline
- **Payload**: `deadlock_type`, `involved_components`, `detection_time`, `recovery_action`
- **Subscribers**:
  - `daemon_manager.py` - Initiates deadlock recovery
  - `unified_health_manager.py` - Tracks system health
- **Added**: December 2025

### CHECKPOINT_SAVED

- **Emitter**: `training_coordinator.py`, `train.py`
- **Purpose**: Training checkpoint saved to disk
- **Payload**: `checkpoint_path`, `config_key`, `epoch`, `metrics`
- **Subscribers**:
  - `unified_distribution_daemon.py` - May distribute intermediate checkpoints
  - `data_pipeline_orchestrator.py` - Tracks training progress
- **Added**: December 2025

### CHECKPOINT_LOADED

- **Emitter**: `training_coordinator.py`, `train.py`
- **Purpose**: Training checkpoint loaded (resume or transfer learning)
- **Payload**: `checkpoint_path`, `config_key`, `source_epoch`, `transfer_type`
- **Subscribers**:
  - `curriculum_integration.py` - Updates curriculum based on checkpoint
  - `unified_feedback.py` - Adjusts feedback signals
- **Added**: December 2025

---

## Event Validation

To validate that all events have subscribers at startup:

```python
from app.coordination.event_router import get_event_bus
from app.coordination.data_events import DataEventType

def validate_event_coverage():
    """Ensure no orphaned events exist."""
    bus = get_event_bus()
    orphaned = []
    for event_type in DataEventType:
        if not bus.get_subscribers(event_type.value):
            orphaned.append(event_type.value)
    if orphaned:
        logger.warning(f"Orphaned events: {orphaned}")
    return len(orphaned) == 0
```

---

## Adding New Events

When adding a new event:

1. Define in `data_events.py` under `DataEventType` enum
2. Add emitter in appropriate module
3. Add at least one subscriber
4. Update this matrix
5. Add integration test

---

## See Also

- `docs/architecture/EVENT_FLOW_INTEGRATION.md` - Event flow diagrams
- `docs/EVENT_SYSTEM_REFERENCE.md` - Detailed event documentation
- `app/coordination/data_events.py` - Event type definitions
- `app/coordination/event_router.py` - Event bus implementation
