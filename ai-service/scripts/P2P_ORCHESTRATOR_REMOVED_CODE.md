# P2P Orchestrator Removed Code Registry

This document tracks code removed from `scripts/p2p_orchestrator.py` during the December 2025 refactoring effort. Methods were removed because they were either dead code or delegated to manager classes.

## Summary

| Category              | Methods Removed | Total LOC Saved |
| --------------------- | --------------- | --------------- |
| Selfplay Scheduling   | 7               | ~471            |
| Training Coordination | 5               | ~160            |
| Background Loops      | 5               | ~400            |
| Sync Operations       | 2               | ~61             |
| Job Management        | 3               | ~163            |
| Deprecated Stubs      | 2               | ~323            |
| **Total**             | **24**          | **~1,578**      |

---

## Selfplay Scheduling (Delegated to SelfplayScheduler)

### `_target_selfplay_jobs_for_node()` - 160 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24498
- **Replacement:** `self.selfplay_scheduler.target_selfplay_jobs_for_node()`
- **Reason:** Delegated to SelfplayScheduler manager

### `_get_hybrid_job_targets()` - 38 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24501
- **Replacement:** `self.selfplay_scheduler.get_hybrid_job_targets()`
- **Reason:** Delegated to SelfplayScheduler manager

### `_should_spawn_cpu_only_jobs()` - 33 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24504
- **Replacement:** `self.selfplay_scheduler.should_spawn_cpu_only_jobs()`
- **Reason:** Delegated to SelfplayScheduler manager

### `_pick_weighted_selfplay_config()` - 95 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24117
- **Replacement:** `self.selfplay_scheduler.pick_weighted_selfplay_config()`
- **Reason:** Delegated to SelfplayScheduler manager

### `_get_elo_based_priority_boost()` - 45 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24114
- **Replacement:** `self.selfplay_scheduler.get_elo_based_priority_boost()`
- **Reason:** Delegated to SelfplayScheduler manager

### `_get_diversity_metrics()` and `_track_selfplay_diversity()`

- **Removal Date:** Dec 27, 2025
- **Location:** Line 4467
- **Replacement:** `self.selfplay_scheduler.get_diversity_metrics()`
- **Reason:** Delegated to SelfplayScheduler manager

---

## Training Coordination (Delegated to TrainingCoordinator)

### `_dispatch_training_job()` - 9 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10429
- **Replacement:** `self.training_coordinator.dispatch_training_job()`
- **Reason:** Delegated to TrainingCoordinator manager

### `_handle_training_job_completion()` - 9 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10879
- **Replacement:** `self.training_coordinator.handle_training_job_completion()`
- **Reason:** Delegated to TrainingCoordinator manager

### `_schedule_model_comparison_tournament()` - 9 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10882
- **Replacement:** `self.training_coordinator.schedule_model_comparison_tournament()`
- **Reason:** Delegated to TrainingCoordinator manager

### `_run_post_training_gauntlet()` - 9 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10928
- **Replacement:** `self.training_coordinator.run_post_training_gauntlet()`
- **Reason:** Delegated to TrainingCoordinator manager

### `_find_running_training_job()` and `_find_resumable_training_job()` - 29 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10425
- **Replacement:** `self.training_coordinator.find_running_training_job()`
- **Reason:** Delegated to TrainingCoordinator manager

### `_check_training_readiness()` - 95 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 10421
- **Replacement:** `self.training_coordinator.check_training_readiness()`
- **Reason:** Delegated to TrainingCoordinator manager

---

## Background Loops (Migrated to LoopManager)

### `_elo_sync_loop()` - 29 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Lines 15564, 26725
- **Replacement:** `scripts/p2p/loops/elo_sync_loop.py::EloSyncLoop`
- **Reason:** Migrated to LoopManager architecture

### `_idle_detection_loop()` - 128 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Lines 15807, 26733
- **Replacement:** `scripts/p2p/loops/job_loops.py::IdleDetectionLoop`
- **Reason:** Migrated to LoopManager architecture

### `_auto_scaling_loop()` - 101 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Lines 16055, 26738
- **Replacement:** `scripts/p2p/loops/coordination_loops.py::AutoScalingLoop`
- **Reason:** Migrated to LoopManager architecture

### `_job_reaper_loop()` - 60 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Lines 16230, 26746
- **Replacement:** `scripts/p2p/loops/job_loops.py::JobReaperLoop`
- **Reason:** Migrated to LoopManager architecture

### `_queue_populator_loop()` - 82 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Lines 16351, 26751
- **Replacement:** `scripts/p2p/loops/queue_populator_loop.py::QueuePopulatorLoop`
- **Reason:** Migrated to LoopManager architecture

---

## Sync Operations (Delegated to SyncPlanner)

### `_collect_local_data_manifest_legacy()` - ~50 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 4662
- **Replacement:** `self.sync_planner.collect_local_data_manifest()`
- **Reason:** Delegated to SyncPlanner manager

### `_generate_sync_plan()` - 61 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 4905
- **Replacement:** `self.sync_planner.generate_sync_plan()`
- **Reason:** Dead code - SyncPlanner handles this

---

## Job Management (Delegated to JobManager)

### `_run_gpu_selfplay_job()` - 123 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 8005
- **Replacement:** `self.job_manager.run_gpu_selfplay_job()`
- **Reason:** Delegated to JobManager

### `_run_distributed_tournament()` - 9 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 9610
- **Replacement:** `self.job_manager.run_distributed_tournament()`
- **Reason:** Wrapper removed, call site updated to use job_manager directly

### `_cleanup_old_completed_jobs()` - 31 LOC

- **Removal Date:** Dec 27, 2025
- **Location:** Line 24111
- **Replacement:** `self.job_manager.cleanup_old_completed_jobs()`
- **Reason:** Delegated to JobManager

---

## Methods Removed at Line 10410

The following thin wrapper methods were removed and call sites updated to use job_manager directly:

- `_run_distributed_tournament()` -> `self.job_manager.run_distributed_tournament()`
- `_run_distributed_selfplay()` -> `self.job_manager.run_distributed_selfplay()`
- `_export_training_data()` -> `self.job_manager.export_training_data()`
- `_run_training()` -> `self.job_manager.run_training()`

---

## Deprecated Loop Stubs (Dec 28, 2025)

### `_data_management_loop_DEPRECATED()` - ~180 LOC

- **Removal Date:** Dec 28, 2025
- **Location:** Line 7356
- **Replacement:** `scripts/p2p/loops/data_loops.py::DataManagementLoop`
- **Reason:** Deprecated stub retained after loop extraction, marked for Q1 2026 removal

### `_model_sync_loop_DEPRECATED()` - ~143 LOC

- **Removal Date:** Dec 28, 2025
- **Location:** Line 7544
- **Replacement:** `scripts/p2p/loops/data_loops.py::ModelSyncLoop`
- **Reason:** Deprecated stub retained after loop extraction, marked for Q1 2026 removal

---

## Notes

1. All removed methods have corresponding NOTE comments in the code at their original locations
2. The manager classes are located in `scripts/p2p/managers/`
3. The loop classes are located in `scripts/p2p/loops/`
4. Feature flag `RINGRIFT_EXTRACTED_LOOPS=true` controls loop manager usage (default: enabled)

## See Also

- `scripts/p2p/managers/README.md` - Manager architecture documentation
- `scripts/p2p/loops/README.md` - Loop manager documentation
- `CLAUDE.md` - P2P Manager Delegation section
