# RingRift AI-Service Consolidation Status

**Date:** December 19, 2025
**Status:** Phase 1 Complete

---

## Completed Consolidation Work

### 1. Tests for Unified Modules

- Created `tests/test_event_router.py` (~350 lines)
- Created `tests/test_daemon_manager.py` (~350 lines)
- Created `tests/test_unified_cluster_monitor.py` (~350 lines)

### 2. Monitoring Script Cleanup

Deprecated 4 cluster monitoring scripts with notices pointing to `unified_cluster_monitor.py`:

- `scripts/cluster_monitor.py`
- `scripts/cluster_health_monitor.py`
- `scripts/monitor_cluster_health.py`
- `scripts/cluster_health_check.py`

### 3. Data Validator Consolidation

- Added deprecation notice to `app/training/data_validation.py`
- Added backward-compatible re-exports in `unified_data_validator.py`
- Users can now import from either module during migration

### 4. Event Router Integration

- Added router imports to `app/coordination/event_emitters.py`
- Added `_emit_via_router()` and `_emit_via_router_sync()` helpers
- Updated `_emit_stage_event()` to optionally use unified router
- Set `USE_UNIFIED_ROUTER = True` for automatic routing

### 5. Model Store Unification

- Added `ModelLifecycleCoordinator` exports to `app/coordination/__init__.py`
- Added `UnifiedModelStore` exports to `app/training/__init__.py`

### 6. Daemon Manager Integration

- Added `DaemonManager` import to `app/main.py`
- Integrated daemon startup/shutdown in FastAPI lifespan
- Enabled via `RINGRIFT_START_DAEMONS=1` environment variable

### 7. Configuration Migration

- Added threshold constant re-exports to `app/config/__init__.py`:
  - `TRAINING_TRIGGER_GAMES`, `TRAINING_MIN_INTERVAL_SECONDS`
  - `ELO_DROP_ROLLBACK`, `WIN_RATE_DROP_ROLLBACK`
  - `ELO_IMPROVEMENT_PROMOTE`, `MIN_GAMES_PROMOTE`
  - `INITIAL_ELO_RATING`, `ELO_K_FACTOR`, `MIN_GAMES_FOR_ELO`

### 8. Training Pipeline Rationalization

- Added cross-reference documentation to:
  - `app/training/unified_orchestrator.py` (execution-level)
  - `app/training/orchestrated_training.py` (service-level)

---

## Recommended Next Steps (Priority Order)

### Priority 1: Critical Consolidation

#### 1.1 Sync Coordinator Naming Clarification

**Files:**

- `app/coordination/sync_coordinator.py` (SCHEDULING)
- `app/distributed/sync_coordinator.py` (EXECUTION)

**Action:** Finalize `SyncScheduler` as canonical name for scheduling layer.

#### 1.2 Model Registry Consolidation

**Files to consolidate:**

- `app/training/model_registry.py` → deprecate
- `app/training/unified_model_store.py` → canonical
- `app/training/training_registry.py` → rename to `TrainingJobRegistry`

**Action:** Make `unified_model_store.UnifiedModelStore` the canonical model lifecycle tracker.

#### 1.3 Complete Data Validation Unification

**Action:** Integrate remaining validators into `unified_data_validator.py`:

- `territory_dataset_validation.py`
- `db/parity_validator.py`

### Priority 2: High-Value Consolidation

#### 2.1 Centralize Hardcoded Constants

Create new config files:

- `app/config/timeout_defaults.py`
- `app/config/threshold_defaults.py`

Replace 30+ hardcoded values across:

- `orchestrator_registry.py` (HEARTBEAT_TIMEOUT_SECONDS)
- `distributed_lock.py` (DEFAULT_LOCK_TIMEOUT)
- `cluster_transport.py` (DEFAULT_CONNECT_TIMEOUT)
- `sync_mutex.py` (LOCK_TIMEOUT_SECONDS)

#### 2.2 Standardize Package Imports

Add missing exports to `app/training/__init__.py`:

- `TrainingEnvConfig`, `make_env` from `env.py`
- `seed_all` from `seed_utils.py`
- `infer_victory_reason` from `tournament.py`

#### 2.3 Orchestrator Registry Cleanup

- Verify all orchestrators extend `CoordinatorBase`
- Standardize `.get_status()` method signature
- Add `get_all_orchestrators()` helper function

### Priority 3: Medium-Value Cleanup

#### 3.1 Clean Deprecation Warnings

Update lazy loading to explicit deprecation warnings:

- `app/distributed/__init__.py` (cluster_coordinator symbols)
- `app/distributed/ingestion_wal.py`

#### 3.2 AI Module Organization

Expand `app/ai/__init__.py` to centralize AI class exports.

### Priority 4: Integration Verification

#### 4.1 Verify Unified Module Usage

Check that new unified modules are used instead of old:

- `unified_orchestrator.py` vs `train_loop.py`
- `checkpoint_unified.py` vs `checkpointing.py`
- `distributed_unified.py` vs `distributed.py`

#### 4.2 Event System Unification

Verify all event emission uses unified router:

- Check `app/distributed/event_helpers.py`
- Update old emitters to use `coordination.event_router`

---

## File Changes Summary

### Created Files

| File                                    | Purpose                      |
| --------------------------------------- | ---------------------------- |
| `tests/test_event_router.py`            | Tests for UnifiedEventRouter |
| `tests/test_daemon_manager.py`          | Tests for DaemonManager      |
| `tests/test_unified_cluster_monitor.py` | Tests for cluster monitoring |

### Modified Files

| File                                     | Changes                                 |
| ---------------------------------------- | --------------------------------------- |
| `scripts/cluster_monitor.py`             | Added deprecation notice                |
| `scripts/cluster_health_monitor.py`      | Added deprecation notice                |
| `scripts/monitor_cluster_health.py`      | Added deprecation notice                |
| `scripts/cluster_health_check.py`        | Added deprecation notice                |
| `app/training/data_validation.py`        | Added deprecation notice                |
| `app/training/unified_data_validator.py` | Added backward-compatible re-exports    |
| `app/coordination/event_emitters.py`     | Added router integration                |
| `app/coordination/__init__.py`           | Added ModelLifecycleCoordinator exports |
| `app/training/__init__.py`               | Added UnifiedModelStore exports         |
| `app/main.py`                            | Added DaemonManager integration         |
| `app/config/__init__.py`                 | Added threshold re-exports              |
| `app/training/unified_orchestrator.py`   | Added cross-reference docs              |
| `app/training/orchestrated_training.py`  | Added cross-reference docs              |

---

## Import Migration Guide

### Old → New Import Patterns

```python
# Monitoring (DEPRECATED)
# OLD: from scripts.cluster_monitor import ClusterMonitor
# NEW:
from scripts.unified_cluster_monitor import UnifiedClusterMonitor

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
    publish_event,
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

## Notes

- All deprecations use `DeprecationWarning` - old code will continue to work
- Unified modules provide backward-compatible re-exports during migration
- Set `RINGRIFT_START_DAEMONS=1` to enable automatic daemon startup in FastAPI
- Event router integration enabled by default (`USE_UNIFIED_ROUTER = True`)
