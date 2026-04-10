# Coordination Module Audit

Updated: April 10, 2026

This audit tracks the largest files in `ai-service/app/coordination/` before further decomposition. The goal is to make the legacy coordination layer reusable alongside the minimal training loop by documenting file responsibilities, public surfaces, and coupling before extracting execution helpers.

## Oversized Files

These files exceeded 3,000 LOC at the start of Part 3 Phase 3.

| File                            |   LOC | Purpose                                                        | Public API Surface                                                                                                                                                                                                                                                                                                                                                    | Methods | Coordination Dependencies |
| ------------------------------- | ----: | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: | ------------------------: |
| `training_trigger_daemon.py`    | 3,924 | Automatic training decision logic.                             | `TrainingTriggerDaemon`, `get_training_trigger_daemon`, `reset_training_trigger_daemon`, `start_training_trigger_daemon`                                                                                                                                                                                                                                              |      70 |                        19 |
| `daemon_manager.py`             | 3,809 | Lifecycle coordination for background services.                | `DaemonStatusCircuitBreaker`, `get_daemon_status_breaker`, `DaemonManager`, `start_profile`, `get_daemon_manager`, `reset_daemon_manager`, `setup_signal_handlers`                                                                                                                                                                                                    |      90 |                        17 |
| `evaluation_daemon.py`          | 3,685 | Auto-evaluate models after training completes.                 | `EvaluationStats`, `EvaluationConfig`, `EvaluationDaemon`, `get_evaluation_daemon`, `start_evaluation_daemon`                                                                                                                                                                                                                                                         |      54 |                        10 |
| `unified_queue_populator.py`    | 3,497 | Automatic work queue population.                               | `QueuePopulatorConfig`, `ConfigTarget`, `UnifiedQueuePopulator`, `UnifiedQueuePopulatorDaemon`, `get_queue_populator`, `get_queue_populator_daemon`, `reset_queue_populator`, `start_queue_populator_daemon`, `wire_queue_populator_events`, `load_populator_config_from_yaml`                                                                                        |      78 |                        13 |
| `data_pipeline_orchestrator.py` | 3,485 | Unified pipeline stage coordination.                           | `PipelineCircuitBreaker`, `PipelineStage`, `OperationMode`, `StageTransition`, `IterationRecord`, `PipelineStats`, `DataPipelineOrchestrator`, `get_pipeline_orchestrator`, `wire_pipeline_events`, `get_pipeline_status`, `get_current_pipeline_stage`, `get_pipeline_health`, `is_pipeline_healthy`                                                                 |      96 |                        19 |
| `curriculum_integration.py`     | 3,360 | Feedback-loop bridge for curriculum weighting and exploration. | `MomentumToCurriculumBridge`, `PFSPWeaknessWatcher`, `PromotionFailedToCurriculumWatcher`, `PromotionCompletedToCurriculumWatcher`, `RegressionCriticalToCurriculumWatcher`, `QualityPenaltyToCurriculumWatcher`, `ArchitectureToCurriculumBridge`, `QualityToTemperatureWatcher`, `wire_all_feedback_loops`, `unwire_all_feedback_loops`, status/query/reset helpers |      86 |                        10 |
| `work_queue.py`                 | 3,222 | Centralized cluster work distribution queue.                   | `WorkQueueBackendType`, `reset_raft_work_queue_cache`, `get_raft_work_queue`, `SlackWorkQueueNotifier`, `WorkType`, `WorkItem`, `ClaimRejectionStats`, `WorkQueue`, `get_work_queue`, `reset_work_queue`                                                                                                                                                              |      90 |                         8 |
| `training_coordinator.py`       | 3,151 | Cluster-wide training slot and progress management.            | `TrainingJob`, `TrainingCoordinator`, `get_training_coordinator`, `request_training_slot`, `release_training_slot`, `update_training_progress`, `can_train`, `get_training_status`, `training_slot`, `wire_training_events`, `TrainingCoordinatorDaemon`, `get_training_coordinator_daemon`, `reset_training_coordinator_daemon`                                      |      77 |                         9 |
| `unified_health_manager.py`     | 3,127 | Consolidated error recovery and health management.             | Health/recovery enums and dataclasses, `UnifiedHealthManager`, `get_health_manager`, `wire_health_events`, `reset_health_manager`, health/error/recovery facade helpers                                                                                                                                                                                               |      95 |                         6 |
| `idle_resource_daemon.py`       | 3,045 | Idle resource detection and opportunistic work spawning.       | `IdleResourceConfig`, `NodeStatus`, `SpawnAttempt`, `NodeSpawnHistory`, `ConfigSpawnHistory`, `SpawnStats`, `NodeIdleState`, `ClusterIdleState`, `IdleResourceDaemon`                                                                                                                                                                                                 |      82 |                        13 |

## Extraction Priorities

1. `training_trigger_daemon.py`: Extract training execution and subprocess/action helpers while keeping daemon lifecycle, scheduling, and event handling in place.
2. `daemon_manager.py`: Extract start/stop/restart/health-check lifecycle operations while keeping registry and profile resolution in the manager.
3. `evaluation_daemon.py`: Extract gauntlet execution and result processing while preserving daemon scheduling and events.
4. `unified_queue_populator.py`: Extract selfplay/training/evaluation population strategies under `app/coordination/queue_strategies/`.
5. `data_pipeline_orchestrator.py`: Extract export/upload/validate stage executors.
6. `curriculum_integration.py`: Extract curriculum strategies and weight calculation helpers.
7. `work_queue.py`: Extract serialization, priority calculation, and persistence helpers.
8. `training_coordinator.py`: Extract leader election and work distribution protocol helpers.

## Notes

- Method counts include all class methods in the file.
- Coordination dependency counts include imports whose module path starts with `app.coordination`.
- This audit intentionally documents before refactoring; completion status and final file sizes should be appended after each extraction.

## Part 3 Extraction Status

The first extraction batch completed two of the largest coordination files while preserving the daemon-facing public APIs:

| Original File                | Before LOC | After LOC | Extracted Module               | Extracted LOC | Status                                        |
| ---------------------------- | ---------: | --------: | ------------------------------ | ------------: | --------------------------------------------- |
| `training_trigger_daemon.py` |      3,924 |     1,836 | `training_executor_actions.py` |         2,226 | Focused training-trigger tests passed         |
| `daemon_manager.py`          |      3,809 |     1,483 | `daemon_manager_lifecycle.py`  |         2,462 | Focused daemon-manager lifecycle tests passed |

`daemon_lifecycle.py` remains the existing composition-based lifecycle manager at 1,064 LOC. The new `daemon_manager_lifecycle.py` module was intentionally kept separate so no new coordination file exceeds the upcoming 2,500 LOC size contract.
