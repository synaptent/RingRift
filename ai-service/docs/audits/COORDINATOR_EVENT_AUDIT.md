# Coordinator Event Subscription Audit

**Audit Date:** December 28, 2025
**Auditor:** Claude Code
**Scope:** 27 coordinators in `COORDINATOR_REGISTRY` (coordination_bootstrap.py)

## Executive Summary

| Metric                                     | Count                |
| ------------------------------------------ | -------------------- |
| Total coordinators in registry             | 27                   |
| Coordinators with event subscriptions      | 24                   |
| Coordinators using SKIP/DELEGATE patterns  | 2                    |
| Coordinators with no subscriptions (issue) | 1                    |
| Total DataEventType values                 | 175                  |
| Events mapped to cross-process             | 143                  |
| Unmapped events                            | 32                   |
| Orphan events (emitted, no subscribers)    | 31 (was 32, fixed 1) |
| Dead subscriptions (no emitter found)      | 46                   |
| Well-connected events                      | 92 (was 91, fixed 1) |

**Overall Status:** MOSTLY HEALTHY - Minor issues identified and documented below.

**Fixes Applied (Dec 28, 2025):**

- Added `DAEMON_PERMANENTLY_FAILED` subscriber to `unified_health_manager.py`
- Added 18 high-impact event mappings to `DATA_TO_CROSS_PROCESS_MAP` in `event_mappings.py`

---

## 1. COORDINATOR_REGISTRY Analysis

The registry is defined in `app/coordination/coordination_bootstrap.py` and contains 27 coordinators organized in 8 initialization layers.

### Layer 1 - Foundational (No Dependencies)

| Coordinator             | File                               | Pattern | Subscriptions | Status |
| ----------------------- | ---------------------------------- | ------- | ------------- | ------ |
| task_coordinator        | task_lifecycle_coordinator.py      | WIRE    | 9 events      | OK     |
| global_task_coordinator | task_coordinator.py                | WIRE    | 5 events      | OK     |
| resource_coordinator    | resource_monitoring_coordinator.py | WIRE    | 6 events      | OK     |
| cache_orchestrator      | cache_coordination_orchestrator.py | WIRE    | 2 events      | OK     |

### Layer 2 - Infrastructure Support

| Coordinator       | File                           | Pattern  | Subscriptions     | Status |
| ----------------- | ------------------------------ | -------- | ----------------- | ------ |
| health_manager    | unified_health_manager.py      | WIRE     | 16 events         | OK     |
| error_coordinator | N/A                            | DELEGATE | -> health_manager | OK     |
| recovery_manager  | N/A                            | SKIP     | Deprecated        | OK     |
| model_coordinator | model_lifecycle_coordinator.py | WIRE     | 9 events          | OK     |

### Layer 3 - Sync and Training

| Coordinator          | File                        | Pattern | Subscriptions | Status |
| -------------------- | --------------------------- | ------- | ------------- | ------ |
| sync_coordinator     | cluster/sync.py (re-export) | WIRE    | 3 events      | OK     |
| training_coordinator | training_coordinator.py     | WIRE    | 17 events     | OK     |

### Layer 4 - Data Integrity

| Coordinator       | File                       | Pattern | Subscriptions | Status |
| ----------------- | -------------------------- | ------- | ------------- | ------ |
| transfer_verifier | transfer_verification.py   | WIRE    | 1 event       | OK     |
| ephemeral_guard   | ephemeral_data_guard.py    | WIRE    | 3 events      | OK     |
| queue_populator   | unified_queue_populator.py | WIRE    | 7+ events     | OK     |

### Layer 5 - Selfplay

| Coordinator           | File                     | Pattern | Subscriptions | Status |
| --------------------- | ------------------------ | ------- | ------------- | ------ |
| selfplay_orchestrator | selfplay_orchestrator.py | WIRE    | 13 events     | OK     |
| selfplay_scheduler    | selfplay_scheduler.py    | GET     | 23 events     | OK     |

### Layer 6 - Multi-Provider and Jobs

| Coordinator    | File                           | Pattern | Subscriptions | Status |
| -------------- | ------------------------------ | ------- | ------------- | ------ |
| multi_provider | multi_provider_orchestrator.py | WIRE    | 3 events      | OK     |
| job_scheduler  | job_scheduler.py               | WIRE    | 2 events      | OK     |

### Layer 7 - Daemons

| Coordinator               | File                           | Pattern | Subscriptions | Status |
| ------------------------- | ------------------------------ | ------- | ------------- | ------ |
| auto_export_daemon        | auto_export_daemon.py          | GET     | Via start()   | OK     |
| evaluation_daemon         | evaluation_daemon.py           | GET     | 1 event       | OK     |
| model_distribution_daemon | unified_distribution_daemon.py | IMPORT  | Via start()   | OK     |
| idle_resource_daemon      | idle_resource_daemon.py        | IMPORT  | Via start()   | OK     |
| quality_monitor_daemon    | quality_monitor_daemon.py      | IMPORT  | Via start()   | OK     |
| orphan_detection_daemon   | orphan_detection_daemon.py     | IMPORT  | Via start()   | OK     |
| curriculum_integration    | curriculum_integration.py      | WIRE    | 8 events      | OK     |

### Layer 8 - Top-Level Coordination

| Coordinator              | File                             | Pattern | Subscriptions | Status |
| ------------------------ | -------------------------------- | ------- | ------------- | ------ |
| metrics_orchestrator     | metrics_analysis_orchestrator.py | WIRE    | 7 events      | OK     |
| optimization_coordinator | optimization_coordinator.py      | WIRE    | 7 events      | OK     |
| leadership_coordinator   | leadership_coordinator.py        | WIRE    | 5 events      | OK     |

---

## 2. Event Subscription Details by Coordinator

### task_coordinator (TaskLifecycleCoordinator)

**Subscribed Events:**

- TASK_SPAWNED
- TASK_COMPLETED
- TASK_FAILED
- TASK_HEARTBEAT
- TASK_ORPHANED
- TASK_CANCELLED
- HOST_ONLINE
- HOST_OFFLINE
- NODE_RECOVERED

**Status:** All events are valid DataEventType values.

### global_task_coordinator (TaskCoordinator)

**Subscribed Events:**

- TASK_SPAWNED
- TASK_HEARTBEAT
- TASK_COMPLETED
- TASK_FAILED
- TASK_CANCELLED

**Status:** All events are valid DataEventType values.

### resource_coordinator (ResourceMonitoringCoordinator)

**Subscribed Events:**

- CLUSTER_CAPACITY_CHANGED
- NODE_CAPACITY_UPDATED
- BACKPRESSURE_ACTIVATED
- BACKPRESSURE_RELEASED
- RESOURCE_CONSTRAINT
- JOB_PREEMPTED

**Status:** All events are valid DataEventType values.

### health_manager (UnifiedHealthManager)

**Subscribed Events:**

- ERROR
- RECOVERY_INITIATED
- RECOVERY_COMPLETED
- RECOVERY_FAILED
- TRAINING_FAILED
- TASK_FAILED
- REGRESSION_DETECTED
- REGRESSION_CRITICAL
- HOST_OFFLINE
- HOST_ONLINE
- NODE_RECOVERED
- PARITY_FAILURE_RATE_CHANGED
- COORDINATOR_HEALTH_DEGRADED
- COORDINATOR_SHUTDOWN
- COORDINATOR_HEARTBEAT

**Status:** All events are valid DataEventType values.

### training_coordinator (TrainingCoordinator)

**Subscribed Events:**

- P2P_CLUSTER_HEALTHY
- P2P_CLUSTER_UNHEALTHY
- CLUSTER_CAPACITY_CHANGED
- NODE_RECOVERED
- NODE_UNHEALTHY
- REGRESSION_DETECTED
- REGRESSION_CRITICAL
- REGRESSION_CLEARED
- TRAINING_ROLLBACK_NEEDED
- PROMOTION_ROLLED_BACK
- LOW_QUALITY_DATA_WARNING
- TRAINING_BLOCKED_BY_QUALITY
- DATA_SYNC_COMPLETED
- NEW_GAMES_AVAILABLE
- MODEL_PROMOTED
- TRAINING_STARTED
- TRAINING_PROGRESS
- TRAINING_COMPLETED
- TRAINING_FAILED

**Status:** All events are valid DataEventType values.

### selfplay_scheduler (SelfplayScheduler)

**Subscribed Events:**

- SELFPLAY_COMPLETE
- TRAINING_COMPLETED
- MODEL_PROMOTED
- SELFPLAY_TARGET_UPDATED
- QUALITY_DEGRADED
- CURRICULUM_REBALANCED
- SELFPLAY_RATE_CHANGED
- TRAINING_BLOCKED_BY_QUALITY
- OPPONENT_MASTERED
- TRAINING_EARLY_STOPPED
- ELO_VELOCITY_CHANGED
- EXPLORATION_BOOST
- CURRICULUM_ADVANCED
- ADAPTIVE_PARAMS_CHANGED
- LOW_QUALITY_DATA_WARNING
- NODE_UNHEALTHY
- NODE_RECOVERED
- P2P_NODE_DEAD
- P2P_CLUSTER_UNHEALTHY
- P2P_CLUSTER_HEALTHY
- HOST_OFFLINE
- REGRESSION_DETECTED

**Status:** All events are valid DataEventType values.

### curriculum_integration (CurriculumIntegration)

**Subscribed Events:**

- EVALUATION_COMPLETED
- SELFPLAY_RATE_CHANGED
- ELO_SIGNIFICANT_CHANGE
- SELFPLAY_ALLOCATION_UPDATED
- MODEL_PROMOTED
- TIER_PROMOTION
- REGRESSION_CRITICAL
- QUALITY_PENALTY_APPLIED
- QUALITY_FEEDBACK_ADJUSTED
- QUALITY_SCORE_UPDATED

**Status:** All events are valid DataEventType values.

---

## 3. Events Not Mapped to Cross-Process Queue

The following DataEventType values are NOT in `DATA_TO_CROSS_PROCESS_MAP` and therefore cannot be bridged to cross-process communication:

| Event                        | Impact                          |
| ---------------------------- | ------------------------------- |
| adaptive_params_changed      | Medium - Training feedback loop |
| batch_dispatched             | Low - Internal pipeline         |
| batch_scheduled              | Low - Internal pipeline         |
| checkpoint_loaded            | Low - Internal state            |
| checkpoint_saved             | Low - Internal state            |
| coordinator_health_degraded  | Medium - Monitoring gap         |
| coordinator_init_failed      | Medium - Bootstrap errors       |
| cpu_pipeline_job_completed   | Low - Vast CPU jobs             |
| crossboard_promotion         | Low - Multi-config              |
| daemon_permanently_failed    | High - Critical alerts          |
| data_backup_completed        | Low - Backup notification       |
| deadlock_detected            | High - Critical error           |
| disk_cleanup_triggered       | Low - Maintenance               |
| disk_space_low               | High - Resource alert           |
| epoch_advanced               | Low - P2P internal              |
| error                        | Medium - General errors         |
| exploration_adjusted         | Low - Selfplay tuning           |
| exploration_boost            | Low - Selfplay tuning           |
| handler_timeout              | Medium - Debug/monitoring       |
| health_alert                 | High - Health monitoring        |
| health_check_failed          | High - Health monitoring        |
| health_check_passed          | Low - Routine                   |
| idle_state_broadcast         | Low - GPU utilization           |
| idle_state_request           | Low - GPU utilization           |
| leader_stepdown              | Medium - P2P leadership         |
| lock_timeout                 | Medium - Concurrency            |
| model_corrupted              | High - Data integrity           |
| nas_best_architecture        | Low - NAS optimization          |
| nas_generation_complete      | Low - NAS optimization          |
| nas_started                  | Low - NAS optimization          |
| node_capacity_updated        | Medium - Scaling                |
| node_overloaded              | High - Resource alert           |
| opponent_mastered            | Low - Curriculum                |
| pbt_completed                | Low - PBT optimization          |
| pbt_started                  | Low - PBT optimization          |
| per_buffer_rebuilt           | Low - PER buffer                |
| per_priorities_updated       | Low - PER buffer                |
| promotion_rolled_back        | High - Model lifecycle          |
| quality_feedback_adjusted    | Medium - Data quality           |
| quality_penalty_applied      | Medium - Data quality           |
| resource_constraint_detected | High - Resource alert           |
| s3_backup_completed          | Low - Backup notification       |
| scheduler_registered         | Low - Internal state            |
| selfplay_allocation_updated  | Medium - Scheduling             |
| state_persisted              | Low - P2P internal              |
| training_blocked_by_quality  | High - Pipeline control         |
| training_early_stopped       | Medium - Training feedback      |
| training_loss_anomaly        | High - Training health          |
| training_loss_trend          | Medium - Training health        |
| training_rollback_completed  | High - Recovery                 |
| training_rollback_needed     | High - Recovery                 |
| weight_updated               | Low - Curriculum                |

**Recommendation:** Add high-impact events to `DATA_TO_CROSS_PROCESS_MAP`:

- daemon_permanently_failed
- deadlock_detected
- disk_space_low
- health_alert
- health_check_failed
- model_corrupted
- node_overloaded
- promotion_rolled_back
- resource_constraint_detected
- training_blocked_by_quality
- training_loss_anomaly
- training_rollback_completed
- training_rollback_needed

---

## 4. Orphan Events (Emitted but No Subscribers)

The following 32 events are emitted but have no subscribers:

| Event                      | Emitter                       | Severity                                              |
| -------------------------- | ----------------------------- | ----------------------------------------------------- |
| COORDINATOR_HEALTHY        | coordination_bootstrap.py     | Low (informational)                                   |
| COORDINATOR_INIT_FAILED    | **init**.py                   | Medium                                                |
| COORDINATOR_UNHEALTHY      | coordination_bootstrap.py     | Medium                                                |
| CURRICULUM_UPDATED         | event_emitters.py             | Low                                                   |
| DAEMON_PERMANENTLY_FAILED  | data_events.py                | FIXED - subscriber added to unified_health_manager.py |
| EPOCH_ADVANCED             | state_manager.py              | Low (P2P internal)                                    |
| EVALUATION_COMPLETE        | pipeline_actions.py           | Low (use EVALUATION_COMPLETED)                        |
| FUNC                       | event_emission_mixin.py       | Low (utility)                                         |
| HANDLER_FAILED             | handler_resilience.py         | Medium                                                |
| HANDLER_TIMEOUT            | handler_resilience.py         | Medium                                                |
| MODEL_DISTRIBUTION_STARTED | data_events.py                | Low                                                   |
| MY_EVENT                   | event_router.py               | Low (test/example)                                    |
| NEW_GAMES                  | unified_data_sync.py          | Low (use NEW_GAMES_AVAILABLE)                         |
| PROMOTION_COMPLETE         | pipeline_actions.py           | Low (use MODEL_PROMOTED)                              |
| QUALITY_CHECK_REQUESTED    | data_events.py                | Medium                                                |
| REPLICATION_ALERT          | unified_replication_daemon.py | Medium                                                |
| SELFPLAY_COMPLETION        | selfplay_orchestrator.py      | Low (use SELFPLAY_COMPLETE)                           |
| SPLIT_BRAIN_RESOLVED       | leader_election.py            | Low (P2P internal)                                    |
| STATE_PERSISTED            | state_manager.py              | Low (P2P internal)                                    |
| STARTUP_STATE_VALIDATED    | state_manager.py              | Low (P2P internal)                                    |
| SYNC_COMPLETE              | pipeline_actions.py           | Low (use DATA_SYNC_COMPLETED)                         |
| SYNC_FAILURE_CRITICAL      | sync_coordinator.py           | High - needs subscriber                               |
| SYNC_RETRY_REQUESTED       | sync_router.py                | Low                                                   |
| TASK_COMPLETE              | task_decorators.py            | Low (use TASK_COMPLETED)                              |
| TRAINING_COMPLETE          | pipeline_actions.py           | Low (use TRAINING_COMPLETED)                          |
| TRAINING_TRIGGERED         | pipeline_actions.py           | Low                                                   |
| WEIGHT_UPDATED             | data_events.py                | Low                                                   |

**High Priority Fixes Needed:**

1. ~~`DAEMON_PERMANENTLY_FAILED`~~ - FIXED: Subscriber added to `unified_health_manager.py` (Dec 28, 2025)
2. `SYNC_FAILURE_CRITICAL` - Add subscriber in SyncRouter or AlertManager

**Naming Convention Issues:**
Some events use inconsistent naming (e.g., `TRAINING_COMPLETE` vs `TRAINING_COMPLETED`). These should be normalized.

---

## 5. Dead Subscriptions (No Emitter Found)

The following 46 events have subscribers but no detected emitters:

| Event                        | Subscriber                       | Severity |
| ---------------------------- | -------------------------------- | -------- |
| CACHE_INVALIDATED            | metrics_analysis_orchestrator.py | Medium   |
| CLUSTER_STATUS_CHANGED       | multi_provider_orchestrator.py   | Medium   |
| CMAES_COMPLETED              | optimization_coordinator.py      | Medium   |
| CONSOLIDATION_COMPLETE       | data_pipeline_orchestrator.py    | Medium   |
| CONSOLIDATION_STARTED        | data_pipeline_orchestrator.py    | Low      |
| CPU_PIPELINE_JOB_COMPLETED   | feedback_loop_controller.py      | Low      |
| DATABASE_CREATED             | data_pipeline_orchestrator.py    | Medium   |
| HIGH_QUALITY_DATA_AVAILABLE  | sync_coordinator.py              | Medium   |
| IDLE_STATE_BROADCAST         | idle_resource_daemon.py          | Low      |
| IDLE_STATE_REQUEST           | idle_resource_daemon.py          | Low      |
| LEADER_STEPDOWN              | leadership_coordinator.py        | Medium   |
| METRICS_UPDATED              | metrics_analysis_orchestrator.py | Medium   |
| MODEL_CORRUPTED              | model_lifecycle_coordinator.py   | High     |
| MODEL_SYNC_REQUESTED         | sync_router.py                   | Medium   |
| NAS_COMPLETED                | optimization_coordinator.py      | Medium   |
| OPPONENT_MASTERED            | selfplay_scheduler.py            | Low      |
| PROMOTION_ROLLED_BACK        | multiple                         | High     |
| QUALITY_CHECK_FAILED         | feedback_loop_controller.py      | Medium   |
| QUALITY_DISTRIBUTION_CHANGED | data_pipeline_orchestrator.py    | Medium   |
| QUALITY_FEEDBACK_ADJUSTED    | curriculum_integration.py        | Medium   |
| RECOVERY_COMPLETED           | unified_health_manager.py        | Medium   |
| RECOVERY_FAILED              | unified_health_manager.py        | Medium   |
| RECOVERY_INITIATED           | unified_health_manager.py        | Medium   |
| REGRESSION_CLEARED           | training_coordinator.py          | Medium   |
| REGRESSION_CRITICAL          | daemon_manager.py                | High     |
| REGRESSION_DETECTED          | data_pipeline_orchestrator.py    | High     |
| SCHEDULER_REGISTERED         | feedback_loop_controller.py      | Low      |
| SYNC_CHECKSUM_FAILED         | data_pipeline_orchestrator.py    | Medium   |
| TRAINING_PROGRESS            | optimization_coordinator.py      | Medium   |
| TRAINING_ROLLBACK_COMPLETED  | feedback_loop_controller.py      | Medium   |
| TRAINING_ROLLBACK_NEEDED     | training_coordinator.py          | High     |
| WORK_CLAIMED                 | work_queue_monitor_daemon.py     | Low      |
| WORK_COMPLETED               | work_queue_monitor_daemon.py     | Low      |
| WORK_FAILED                  | unified_queue_populator.py       | Medium   |
| WORK_QUEUED                  | data_pipeline_orchestrator.py    | Low      |
| WORK_STARTED                 | work_queue_monitor_daemon.py     | Low      |

**Note:** Some of these may have emitters in non-Python code (TypeScript) or in dynamically constructed event names. Manual verification recommended for High priority items.

---

## 6. Recommendations

### Immediate Actions (High Priority)

1. **Add subscribers for critical orphan events:**
   - ~~`DAEMON_PERMANENTLY_FAILED`~~ -> FIXED: Subscriber added to `unified_health_manager.py` (Dec 28, 2025)
   - `SYNC_FAILURE_CRITICAL` -> Subscribe in `sync_router.py` or `AlertManager`

2. **Add emitters for critical dead subscriptions:**
   - `REGRESSION_CRITICAL` -> Verify emitter exists in `RegressionDetector`
   - `TRAINING_ROLLBACK_NEEDED` -> Verify emitter in training pipeline
   - `MODEL_CORRUPTED` -> Add emitter in model validation code

3. **Normalize event naming:**
   - Use `*_COMPLETED` consistently (not `*_COMPLETE`)
   - Deprecate `NEW_GAMES` in favor of `NEW_GAMES_AVAILABLE`

### Medium Priority

4. **Add missing cross-process mappings** for high-impact events listed in Section 3.

5. **Review dead subscriptions** to determine if:
   - Emitter is missing (needs to be added)
   - Subscription is dead code (can be removed)
   - Emitter is in TypeScript/external (document dependency)

### Low Priority

6. **Clean up test/example events** like `MY_EVENT`, `FUNC`, `DATA_EVENT`.

7. **Document P2P internal events** that intentionally have no external subscribers.

---

## 7. Deprecated Coordinators

The following coordinators are deprecated and should not be used:

| Coordinator       | Replacement    | Notes                             |
| ----------------- | -------------- | --------------------------------- |
| error_coordinator | health_manager | Delegates to UnifiedHealthManager |
| recovery_manager  | health_manager | SKIP pattern - fully deprecated   |

---

## 8. File References

- **Registry:** `app/coordination/coordination_bootstrap.py:260-464`
- **Event Types:** `app/distributed/data_events/event_types.py` (`DataEventType`)
- **Event Mappings:** `app/coordination/event_mappings.py`
- **Event Emitters:** `app/coordination/event_emitters.py`

---

## Appendix: Coordinator Initialization Patterns

| Pattern  | Description                    | Event Wiring              |
| -------- | ------------------------------ | ------------------------- |
| WIRE     | Call wire\_\*() function       | Immediate subscription    |
| GET      | Call get\_\*() singleton       | Deferred to start()       |
| IMPORT   | Import class only              | Deferred to DaemonManager |
| SKIP     | Deprecated coordinator         | No wiring                 |
| DELEGATE | Forward to another coordinator | Use delegate's wiring     |

---

_Generated by Claude Code - December 28, 2025_
