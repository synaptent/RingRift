# RingRift AI-Service Consolidation Status

**Date:** December 19, 2025
**Status:** Phase 7 Complete (Safe Loading + Load Test Enhancements)

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

- ~~Unified module tests referenced in prior notes are not present in repo.~~ ✓ **RESOLVED**: Added tests in Phase 4
- `tests/test_model_registry.py` and `tests/test_promotion_controller.py` were removed;
  replacement coverage partially addressed by `test_unified_model_store.py`
- 63 files still import directly from `data_events.py` - gradual migration recommended

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
    - `GMOAI`, `EBMOAI`, `IGGMO`
- Updated module docstring with architecture overview

---

## Phase 4 Consolidation Work (December 19, 2025)

### 18. Unit Tests for Unified Modules ✓

Added comprehensive test files for unified modules:

- **`tests/unit/training/test_unified_model_store.py`** (43 tests)
  - Import verification for unified API
  - Import verification for legacy re-exports
  - `ModelStoreStage` enum tests
  - `ModelInfo` dataclass creation and serialization
  - Singleton pattern verification
  - Interface method presence checks

- **`tests/unit/training/test_unified_data_validator.py`**
  - Import verification for unified API
  - Legacy re-exports (DataValidator, validate_npz_file)
  - Territory validation re-exports
  - Parity validation re-exports
  - `ValidationType` and `ValidationSeverity` enums
  - `UnifiedValidationResult` creation, issue tracking, serialization
  - Singleton pattern verification

- **`tests/unit/coordination/test_coordinator_base.py`**
  - Import verification for enums, protocols, base classes, mixins
  - `CoordinatorStatus` enum values
  - `CoordinatorStats` dataclass defaults and serialization
  - `get_all_coordinators()` and `get_coordinator_statuses()` helpers
  - `CoordinatorRegistry` singleton pattern
  - Health summary generation

### 19. Event System Audit ✓

**Findings:**

- **73 files** reference EventBus/publish_event/emit_event patterns
- **63 files** import directly from `data_events.py` (bypassing unified router)
- **Only 4 files** explicitly import from `event_router.py`

**Actions Taken:**

1. Added deprecation notice to `app/distributed/event_helpers.py` pointing to unified router
2. Added router integration to `event_helpers.py`:
   - `has_event_router()` - check if router is available
   - `set_use_router_by_default(bool)` - global switch
   - `USE_ROUTER_BY_DEFAULT` - configuration flag
   - Updated `emit_event_safe()` with `use_router` parameter
3. Router integration falls back to direct EventBus if router fails
4. Added `__all__` exports for all public functions

**Migration Path:**

```python
# Option 1: Use unified router directly (recommended for new code)
from app.coordination import (
    get_event_router,
    router_publish_event,  # async
    publish_event_sync,    # sync
)

# Option 2: Use event_helpers with router (gradual migration)
from app.distributed.event_helpers import (
    emit_event_safe,
    set_use_router_by_default,
)
set_use_router_by_default(True)  # Route all events through router
await emit_event_safe("MODEL_PROMOTED", payload, source)

# Option 3: Per-call router opt-in
await emit_event_safe("MODEL_PROMOTED", payload, source, use_router=True)
```

### 20. Import Migration Updates ✓

Updated imports in key files to use unified modules:

- `app/training/train.py` - migrated to use `unified_data_validator`
- `app/training/data_pipeline_controller.py` - migrated imports
- `app/training/hot_data_buffer.py` - migrated imports

### 21. Deprecation Warnings ✓

Added runtime `DeprecationWarning` to legacy modules:

- `app/training/checkpointing.py` - warns on import
- `app/training/data_validation.py` - warns on import
- `app/distributed/__init__.py` - warns on legacy symbol access

---

## Phase 5 Consolidation Work (December 19, 2025)

### 22. Checkpointing Import Migration ✓

Migrated all checkpointing imports to use unified module:

- Added re-exports to `checkpoint_unified.py`:
  - `GracefulShutdownHandler`
  - `save_checkpoint`, `load_checkpoint`
  - `AsyncCheckpointer`
- Updated imports in:
  - `train_setup.py` → imports from `checkpoint_unified`
  - `train.py` → imports from `checkpoint_unified`
  - `app/training/__init__.py` → re-exports from `checkpoint_unified`

### 23. Global Event Router Activation ✓

Enabled unified event router at application startup:

- Added `set_use_router_by_default(True)` in `app/main.py` lifespan
- All `emit_*_safe` calls now route through unified router
- Router provides cross-system event delivery (EventBus + StageEventBus + CrossProcessEventQueue)

### 24. High-Traffic Module Router Migration ✓

Migrated key event emitters to use router directly:

- `app/training/hot_data_buffer.py` - Added `router_publish` with fallback
- `app/monitoring/unified_cluster_monitor.py` - Added `_emit_event()` helper using router
- `app/coordination/event_emitters.py` - Already had `USE_UNIFIED_ROUTER = True`

### 25. Comprehensive Test Coverage ✓

Added extensive test coverage:

- **5,554 total tests** (up from ~1,400)
- New test files:
  - `test_lib_metrics.py` (600 lines)
  - `test_lib_transfer.py` (509 lines)
  - `test_gpu_game_types.py` (190 lines)
  - `test_training_pipeline.py` (322 lines)
  - `test_multi_provider_orchestrator.py` (530 lines)
  - `test_sync_orchestrator.py` (441 lines)
- Added `pytest-cov` for coverage reporting

### 26. New AI Module: IG-GMO ✓

Added Information-Gain GMO (IG-GMO) research module:

- `app/ai/ig_gmo.py` (676 lines)
- Uses mutual information for exploration instead of variance
- GNN-based state encoding
- Soft legality constraints during optimization
- Registered in `AIType` and `AIFactory` as experimental (`ig_gmo`)

---

## Recommended Next Steps (Priority Order)

### Priority 1: Integration Verification ✓ (Completed)

#### 1.1 Verify Unified Module Usage ✓

- `checkpoint_unified.py` vs `checkpointing.py` ✓ (deprecation notice added)
- `unified_data_validator.py` vs `data_validation.py` ✓ (deprecation + re-exports)
- `unified_model_store.py` vs `model_registry.py` ✓ (deprecation + re-exports)

#### 1.2 Event System Unification ✓

- Added router integration to `event_helpers.py` ✓
- Added `use_router` parameter to `emit_event_safe()` ✓
- Added global `USE_ROUTER_BY_DEFAULT` switch ✓

### Priority 2: Remaining Consolidation

#### 2.1 Migrate Remaining Direct EventBus Imports

63 files still import directly from `data_events.py`. Consider:

- Enabling `USE_ROUTER_BY_DEFAULT = True` in production
- Gradually migrating high-traffic emitters to use router

#### 2.2 Orchestrator Unification

- `unified_orchestrator.py` vs `train_loop.py`
- `distributed_unified.py` vs `distributed.py`

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

## Phase 6 Assessment & Fixes (December 19, 2025)

### 27. Codebase Health Assessment ✓

**Test Suite Status:**

- **6,520 tests collected** (up from 5,554)
- **2,930 unit tests passing**
- **145 integration tests passing**
- 1 flaky test (`test_bootstrap_selective` - passes in isolation)

**Code Metrics:**

- 447 Python source files
- 301,262 lines of code
- 0 mypy errors (with `--ignore-missing-imports`)
- 198 flake8 E128 stylistic warnings (indentation in metrics modules)

### 28. Bug Fixes ✓

| File                                | Issue                                | Fix                           |
| ----------------------------------- | ------------------------------------ | ----------------------------- |
| `scripts/holdout_validation.py:657` | SyntaxError - line outside try block | Fixed indentation             |
| `app/training/train.py:3174`        | Missing policy accuracy tracking     | Added accuracy computation    |
| `app/training/env.py:791`           | Unclear repetition detection code    | Improved clarity and comments |

### 29. Documentation Added ✓

New documentation files created:

- **`docs/GPU_VECTORIZATION.md`** - GPU module architecture and known limitations
  - Module structure overview
  - Performance characteristics (CUDA vs CPU vs MPS)
  - Known limitations (recovery gate, chain captures)
  - Shadow validation and parity testing

- **`docs/COORDINATION_ARCHITECTURE.md`** - Event system and coordination
  - EventBus, DataEventBus, StageEventBus overview
  - CrossProcessEventQueue for daemon communication
  - Unified EventRouter consolidation
  - Component communication flow diagrams
  - Configuration and best practices

### 30. Training Improvements ✓

Added policy accuracy tracking to training pipeline:

- `app/training/train.py`:
  - Added `val_policy_correct` and `val_policy_total` accumulators
  - Compute accuracy as `(pred_move == target_move).float().mean()`
  - Added to epoch logs: `Policy Acc: {accuracy:.1%}`
  - Added to EventBus payload: `policy_accuracy`
  - Updated `record_training_step()` to use computed accuracy

### 31. Known Issues (Non-Blocking)

| Issue                               | Status                                       | Priority |
| ----------------------------------- | -------------------------------------------- | -------- |
| Contract vector tests (10 failures) | Phase/player tracking mismatch with fixtures | Medium   |
| E128 flake8 warnings (198)          | Stylistic indentation in metric definitions  | Low      |
| `test_bootstrap_selective`          | Flaky - test isolation issue                 | Low      |

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

---

## Phase 7 Work (December 19, 2025)

### 32. Safe Checkpoint Loading Migration ✓

Migrated all AI modules to use `safe_load_checkpoint` utility:

- **`app/utils/torch_utils.py`**: Added `safe_load_checkpoint()` function
  - Tries `weights_only=True` first (PyTorch 2.6+ security default)
  - Falls back to full loading for legacy checkpoints with metadata
  - Optional warning on unsafe fallback via `warn_on_unsafe` parameter

- **Files updated:**
  - `app/ai/cage_ai.py`
  - `app/ai/ebmo_network.py`
  - `app/ai/gpu_parallel_games.py`
  - `app/ai/ig_gmo.py`
  - `app/ai/mcts_ai.py`
  - `app/ai/minimax_ai.py`
  - `app/ai/nnue_policy.py`
  - `app/training/checkpoint_utils.py`
  - `app/training/checkpointing.py`

### 33. Type Annotation Fixes ✓

Fixed `callable | None` type annotation errors across codebase:

| File                                | Issue                                           | Fix                                 |
| ----------------------------------- | ----------------------------------------------- | ----------------------------------- |
| `app/training/parallel_selfplay.py` | `callable` (builtin) used instead of `Callable` | Added `from typing import Callable` |
| `app/distributed/queue.py`          | Same issue                                      | Same fix                            |
| `app/utils/resource_guard.py`       | 4 occurrences                                   | Same fix                            |

### 34. Load Test Enhancements ✓

Major improvements to k6 load testing infrastructure:

- **JWT TTL Derivation**: Extract token TTL from JWT `exp` claim when `expiresIn` missing
  - `tests/load/auth/helpers.js`: Added `deriveJwtTtlSeconds()` and `decodeBase64Url()`
  - `tests/load/scripts/preflight-check.js`: Added JWT parsing with validation

- **Rate Limit Bypass**: Token-based bypass for load test reliability
  - `.env.staging.example`: Added `RATE_LIMIT_BYPASS_TOKEN` configuration
  - `tests/load/auth/helpers.js`: Added `getBypassHeaders()` helper
  - Updated all HTTP requests to include bypass headers

- **WebSocket Reconnection Testing**: Simulate connection drops during gameplay
  - New metrics: `ws_reconnect_attempts_total`, `ws_reconnect_success_rate`, `ws_reconnect_latency_ms`
  - New config: `WS_RECONNECT_PROBABILITY`, `WS_RECONNECT_MAX_PER_GAME`, `WS_RECONNECT_DELAY_MS`
  - Reconnect scheduling and tracking in `websocket-gameplay.js`

### 35. Tournament Improvements ✓

- **Model Discovery Tournament**: `scripts/run_p2p_elo_tournament.py`
  - Added `--models` flag for automatic model discovery
  - Uses `discover_models()` from `app/models/discovery`
  - Configurable ELO database path per tournament type

- **EBMO Online Learning**: New experimental module
  - `app/ai/ebmo_online.py`: TD-Energy updates during gameplay
  - Rolling buffer for stability
  - Outcome-weighted contrastive loss
  - Test script: `scripts/test_ebmo_online.py`

### 36. Unused Import Cleanup ✓

Automated cleanup of unused imports across distributed modules:

- `app/distributed/unified_data_sync.py`
- `app/distributed/ingestion_wal.py`
- `app/distributed/sync_coordinator.py`
- `app/distributed/data_sync_robust.py`
- `app/distributed/p2p_sync_client.py`
- `app/distributed/ssh_transport.py`
- `app/distributed/sync_orchestrator.py`
- `app/tournament/orchestrator.py`

**Note**: One import (`get_adaptive_timeout`) was incorrectly removed and restored in fix commit.

### 37. Test Suite Status ✓

| Category                        | Count | Status  |
| ------------------------------- | ----- | ------- |
| Unit Tests                      | 2,998 | Passing |
| Integration Tests               | 152   | Passing |
| Critical Flake8 (E9/F63/F7/F82) | 0     | Clean   |

---

## Future Enhancements (Planned)

### Training Pipeline Feedback Loops

A detailed plan exists at `~/.claude/plans/reactive-cooking-pony.md` for:

1. **Evaluation Feedback to Selfplay Priority** (~60 lines)
   - Regressing configs get priority boost for more selfplay data
   - Uses `UnifiedSignalComputer.elo_trend` signals

2. **Diversity Signal to Engine Selection** (~30 lines)
   - When training shows overfitting, use more exploratory selfplay
   - Switch to MCTS when `diversity_needed > 0.7`

3. **Training Quality Exposure** (~30 lines)
   - Add `get_training_quality()` to `unified_orchestrator.py`
   - Detect loss plateau and overfitting signals

**Status**: Plan complete, implementation pending
