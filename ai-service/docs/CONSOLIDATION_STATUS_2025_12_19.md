# RingRift AI-Service Consolidation Status

**Date:** December 19, 2025
**Status:** Phase 3 Complete (All priority consolidation done)

---

## Completed Consolidation Work

### 1. Monitoring and Cluster Script Cleanup

- Removed legacy cluster monitor/control/sync scripts in favor of:
  - `scripts/p2p_orchestrator.py` (primary cluster orchestration)
  - `python -m scripts.monitor status|health|alert` (unified monitoring)
  - `scripts/unified_cluster_monitor.py` (deep health checks)
- Removal list includes:
  - `scripts/cluster_*` (monitor/control/manager/worker/sync shell + python)
  - `scripts/monitor_10h_enhanced.sh`

### 2. Data Validator Consolidation

- Added deprecation notice to `app/training/data_validation.py`
- Added backward-compatible re-exports in `app/training/unified_data_validator.py`
- Users can now import from either module during migration

### 3. Event Router Integration

- Added router imports to `app/coordination/event_emitters.py`
- Added `_emit_via_router()` and `_emit_via_router_sync()` helpers
- Updated `_emit_stage_event()` to optionally use unified router
- Set `USE_UNIFIED_ROUTER = True` for automatic routing

### 4. Daemon Manager Integration

- Added `DaemonManager` import to `app/main.py`
- Integrated daemon startup/shutdown in FastAPI lifespan
- Enabled via `RINGRIFT_START_DAEMONS=1` environment variable

### 5. Configuration Migration

- Added threshold constant re-exports to `app/config/__init__.py`:
  - `TRAINING_TRIGGER_GAMES`, `TRAINING_MIN_INTERVAL_SECONDS`
  - `ELO_DROP_ROLLBACK`, `WIN_RATE_DROP_ROLLBACK`
  - `ELO_IMPROVEMENT_PROMOTE`, `MIN_GAMES_PROMOTE`
  - `INITIAL_ELO_RATING`, `ELO_K_FACTOR`, `MIN_GAMES_FOR_ELO`
- Added queue/scaling/duration defaults to `app/config/coordination_defaults.py`

### 6. Training Pipeline Rationalization

- Added cross-reference documentation to:
  - `app/training/unified_orchestrator.py` (execution-level)
  - `app/training/orchestrated_training.py` (service-level)

### 7. Event Wiring Helpers

- Added event wiring hooks for coordinators:
  - `queue_populator`, `sync_scheduler`, `task_coordinator`
  - `training_coordinator`, `transfer_verification`, `ephemeral_data_guard`
  - `multi_provider_orchestrator`

### 8. Coordination Exports

- Standardized `__all__` exports across coordination modules for canonical imports.

---

## Open Gaps / Risks

- Unified module tests referenced in prior notes are not present in repo.
- `tests/test_model_registry.py` and `tests/test_promotion_controller.py` were removed;
  replacement coverage is still needed.

---

## Phase 2 Consolidation Work (December 19, 2025)

### 9. Model Registry Consolidation ✓

- Added deprecation notice to `app/training/model_registry.py`
- Added `get_model_registry()` singleton function
- Added backward-compatible re-exports to `unified_model_store.py`:
  - `ModelRegistry`, `RegistryDatabase`, `ModelStage`, `ModelType`
  - `ModelMetrics`, `TrainingConfig`, `ModelVersion`, `ValidationStatus`
  - `AutoPromoter`, `get_model_registry`

### 10. Sync Coordinator Clarification ✓

- `app/coordination/sync_coordinator.py` already uses `SyncScheduler` as canonical name
- `SyncCoordinator` is a backward-compatible alias
- Added deprecation notice to `reset_sync_coordinator()`

### 11. Centralized Timeout Constants ✓

- Added cluster transport timeouts to `app/config/thresholds.py`:
  - `CLUSTER_CONNECT_TIMEOUT`, `CLUSTER_OPERATION_TIMEOUT`, `P2P_HTTP_TIMEOUT`
- Updated `distributed_lock.py` to import from `thresholds.py`
- Updated `cluster_transport.py` to import from `thresholds.py`
- `orchestrator_registry.py` already imports from `thresholds.py`

### 12. Data Validator Integration ✓

- Added `validate_game_parity()` method to `UnifiedDataValidator`
- Added `GAME_PARITY` validation type
- Added re-exports from `territory_dataset_validation.py`:
  - `validate_territory_example`, `validate_territory_dataset_file`, `iter_territory_dataset_errors`
- Added re-exports from `db/parity_validator.py`:
  - `validate_parity`, `validate_after_recording`, `ParityValidationError`
  - `ParityDivergence`, `ParityMode`, `is_parity_validation_enabled`

### 13. Deprecation Notices ✓

- Added deprecation notice to `app/training/checkpointing.py` pointing to `checkpoint_unified.py`

---

## Phase 3 Consolidation Work (December 19, 2025)

### 14. Standardized Package Imports ✓

- Added to `app/training/__init__.py`:
  - `TrainingEnvConfig`, `make_env` from `env.py`
  - `seed_all` from `seed_utils.py`
  - `infer_victory_reason` from `tournament.py`

### 15. Orchestrator Registry Cleanup ✓

- Added `get_all_coordinators()` helper function to `coordinator_base.py`
- Added `get_coordinator_statuses()` helper function
- Updated `__all__` exports

### 16. Clean Deprecation Warnings ✓

- Updated `app/distributed/__init__.py` `__getattr__` to emit `DeprecationWarning`
- Warnings now point to replacement modules:
  - `ClusterCoordinator` -> `TaskCoordinator`
  - `TaskRole` -> `OrchestratorRole`
  - `TaskInfo` -> `OrchestratorInfo`

### 17. AI Module Organization ✓

- Expanded `app/ai/__init__.py` with:
  - `BaseAI` base class import
  - `AIType` enum
  - Lazy-loaded AI implementation classes:
    - `HeuristicAI`, `MCTSAI`, `DescentAI`, `GumbelMCTSAI`
    - `MaxNAI`, `MinimaxAI`, `RandomAI`, `PolicyOnlyAI`
    - `GMOAI`, `EBMOAI`
- Updated module docstring with architecture overview

---

## Recommended Next Steps (Priority Order)

### Priority 1: Integration Verification

#### 1.1 Verify Unified Module Usage

Check that new unified modules are used instead of old:

- `unified_orchestrator.py` vs `train_loop.py`
- `checkpoint_unified.py` vs `checkpointing.py` ✓ (deprecation notice added)
- `distributed_unified.py` vs `distributed.py`

#### 1.2 Event System Unification

Verify all event emission uses unified router:

- Check `app/distributed/event_helpers.py`
- Update old emitters to use `coordination.event_router`

---

## File Changes Summary

### Added Files

| File                                  | Purpose                       |
| ------------------------------------- | ----------------------------- |
| `app/core/thread_spawner.py`          | Thread supervision helper     |
| `app/ai/ebmo_ai.py`                   | EBMO inference AI             |
| `app/ai/ebmo_network.py`              | EBMO network architecture     |
| `app/training/ebmo_dataset.py`        | EBMO dataset utilities        |
| `app/training/ebmo_trainer.py`        | EBMO training loop            |
| `app/training/model_state_machine.py` | Model lifecycle state machine |
| `app/training/train_gmo.py`           | GMO training script           |
| `tests/test_gmo_ai.py`                | GMO tests                     |
| `docs/GMO_ALGORITHM.md`               | GMO algorithm reference       |

### Removed Files

| File                                 | Notes                                                        |
| ------------------------------------ | ------------------------------------------------------------ |
| `scripts/cluster_*`                  | Legacy cluster scripts removed (see `scripts/DEPRECATED.md`) |
| `scripts/monitor_10h_enhanced.sh`    | Legacy monitoring script removed                             |
| `tests/test_model_registry.py`       | Removed pending consolidation                                |
| `tests/test_promotion_controller.py` | Removed pending consolidation                                |

---

## Import Migration Guide

### Old -> New Import Patterns

```python
# Monitoring (DEPRECATED)
# OLD: from scripts.cluster_monitor import ClusterMonitor
# NEW:
from scripts.unified_cluster_monitor import UnifiedClusterMonitor

# Unified monitor CLI
# python -m scripts.monitor status|health|alert

# Data Validation (DEPRECATED)
# OLD: from app.training.data_validation import DataValidator
# NEW:
from app.training.unified_data_validator import (
    UnifiedDataValidator,
    get_validator,
    # Legacy re-exports also available:
    DataValidator,
    validate_npz_file,
)

# Model Store
from app.training import (
    UnifiedModelStore,
    get_model_store,
    ModelInfo,
)

# Model Lifecycle
from app.coordination import (
    ModelLifecycleCoordinator,
    get_model_coordinator,
    wire_model_events,
)

# Configuration Thresholds
from app.config import (
    TRAINING_TRIGGER_GAMES,
    ELO_DROP_ROLLBACK,
    INITIAL_ELO_RATING,
)

# Event Routing (Unified)
from app.coordination import (
    get_event_router,
    router_publish_event,
    publish_event_sync,
    subscribe_event,
)

# Daemon Management
from app.coordination.daemon_manager import (
    DaemonManager,
    get_daemon_manager,
    DaemonType,
)
```
