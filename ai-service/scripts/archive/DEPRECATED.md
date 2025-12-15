# Deprecated Scripts Archive

This directory contains scripts that have been superseded by `unified_ai_loop.py` and related infrastructure. These scripts are kept for reference but should not be used in production.

## Archived Scripts

### cluster_orchestrator.py

**Archived:** 2025-12-15
**Reason:** Functionality consolidated into `unified_ai_loop.py`
**Replacement:** Use `unified_ai_loop.py --help` for cluster orchestration

The cluster orchestrator managed compute resources with resource-aware scheduling. This functionality is now integrated into:

- `unified_ai_loop.py` - Main orchestration loop
- `p2p_orchestrator.py` - P2P cluster coordination
- `app/coordination/job_scheduler.py` - Priority-based job scheduling with:
  - `PriorityJobScheduler` class with `JobPriority` enum (CRITICAL, HIGH, NORMAL, LOW)
  - `ScheduledJob` dataclass for job configuration
  - `select_curriculum_config()` for Elo-driven curriculum learning
  - `get_cpu_rich_hosts()` / `get_gpu_rich_hosts()` for distributed tournaments
- `app/coordination/resource_optimizer.py` - PID-controlled utilization targeting

### pipeline_orchestrator.py

**Archived:** 2025-12-15
**Reason:** Functionality consolidated into `unified_ai_loop.py`
**Replacement:** Use `unified_ai_loop.py --help` for pipeline management

The pipeline orchestrator managed the complete AI training pipeline. Its phases are now handled by:

- Data ingestion → `UnifiedLoop._sync_data_from_hosts()`
- Training → `UnifiedLoop._run_training()`
- Evaluation → `UnifiedLoop._run_shadow_evaluation()`
- Promotion → `UnifiedLoop._check_promotion()`
- Elo calibration → `UnifiedLoop._run_calibration_analysis()`

Key classes have been consolidated into `app/coordination/`:

- `P2PBackend` → `app/coordination/p2p_backend.py`
  - REST API client for P2P orchestrator
  - `discover_p2p_leader_url()` for resilient leader discovery
  - `get_p2p_backend()` convenience function
- `StageEventBus` → `app/coordination/stage_events.py`
  - Event-driven pipeline with `StageEvent` enum
  - `StageCompletionResult` for callback data
  - `register_standard_callbacks()` for typical pipeline flow

### improvement_cycle_manager.py

**Archived:** 2025-12-15
**Reason:** Functionality consolidated into `unified_ai_loop.py` and `model_promotion_manager.py`
**Replacement:** Use `unified_ai_loop.py` for the main improvement loop

The improvement cycle manager bridged P2P orchestrator with AI training. Its features are now handled by:

- Diverse selfplay scheduling → `unified_ai_loop.py` with feedback-driven scheduling
- Training job triggers → `UnifiedLoop.TrainingScheduler` with event-driven triggers
- Tournament scheduling → `shadow_tournament_service.py` for continuous evaluation
- Model promotion → `model_promotion_manager.py` with `--min-games` and `--significance`
- Rollback detection → `model_promotion_manager.py` with `check_for_elo_regression()`
- CMA-ES weight integration → Use `scripts/cmaes_weight_optimizer.py` directly
- A/B testing → Use `model_promotion_manager.py` with significance testing
- Curriculum learning → `UnifiedLoop.AdaptiveCurriculum` with Elo-weighted training

### Other Archived Scripts (from previous consolidation)

- `master_self_improvement.py` - Superseded by unified_ai_loop.py
- `unified_improvement_controller.py` - Superseded by unified_ai_loop.py
- `integrated_self_improvement.py` - Superseded by unified_ai_loop.py
- `export_replay_dataset.py` - Functionality in data pipeline
- `validate_canonical_training_sources.py` - Functionality in validation module

## Migration Guide

To migrate from deprecated scripts to the unified system:

1. **For cluster management:**

   ```bash
   # Old
   python scripts/cluster_orchestrator.py

   # New
   python scripts/unified_ai_loop.py --coordinator
   ```

2. **For pipeline execution:**

   ```bash
   # Old
   python scripts/pipeline_orchestrator.py --phase selfplay

   # New
   python scripts/unified_ai_loop.py  # Handles all phases automatically
   ```

3. **For configuration:**
   - All settings are now in `config/unified_loop.yaml`
   - Use `app/config/unified_config.py` for programmatic access

4. **For P2P orchestrator communication:**

   ```python
   # Old (from pipeline_orchestrator.py)
   from scripts.archive.pipeline_orchestrator import P2PBackend, discover_p2p_leader_url

   # New
   from app.coordination import P2PBackend, discover_p2p_leader_url, get_p2p_backend

   # Or use the convenience function
   backend = await get_p2p_backend(seed_urls=["http://node1:8770"])
   ```

5. **For priority job scheduling:**

   ```python
   # Old (from cluster_orchestrator.py)
   from scripts.archive.cluster_orchestrator import PriorityJobScheduler, JobPriority

   # New
   from app.coordination import (
       PriorityJobScheduler, JobPriority, ScheduledJob,
       get_job_scheduler, select_curriculum_config,
   )

   scheduler = get_job_scheduler()
   scheduler.schedule(ScheduledJob(
       job_type="selfplay",
       priority=JobPriority.NORMAL,
       config={"board": "square8", "players": 2}
   ))
   ```

6. **For event-driven pipeline:**

   ```python
   # Old (from pipeline_orchestrator.py)
   from scripts.archive.pipeline_orchestrator import StageEventBus, StageEvent

   # New
   from app.coordination import (
       StageEventBus, StageEvent, StageCompletionResult,
       get_stage_event_bus, register_standard_callbacks,
   )

   bus = get_stage_event_bus()
   register_standard_callbacks(bus)
   ```

## Questions?

If you need functionality from these scripts that isn't available in the unified system, please open an issue or check if the feature should be added to `unified_ai_loop.py`.
