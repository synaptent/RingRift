# RingRift AI-Service Consolidation Status

**Date:** December 27, 2025
**Status:** Phase 11 Complete (P0 Critical Event System Fixes + P2P Feedback Loop Wiring)

> Status: Historical snapshot (Dec 2025). Kept for reference; consult `README.md` for current docs.

---

## December 27, 2025 - Critical Infrastructure Fixes

### P0.1: SyncPlanner Event Emission Fix

**File:** `scripts/p2p/managers/sync_planner.py`

**Issue:** The `_emit_sync_event()` method was importing a non-existent `emit_sync` function, causing DATA_SYNC_COMPLETED events to never be emitted.

**Fix:** Changed import from `emit_sync` to `publish_sync`:

```python
# Before (broken):
from app.coordination.event_router import emit_sync

# After (working):
from app.coordination.event_router import publish_sync
```

**Impact:** Pipeline triggering now works correctly after sync operations.

---

### P0.2: SelfplayScheduler SELFPLAY_TARGET_UPDATED Events

**File:** `scripts/p2p/managers/selfplay_scheduler.py`

**Issue:** P2P SelfplayScheduler didn't emit target update events, breaking feedback loops with the coordination layer.

**Fix:** Added `_emit_selfplay_target_updated()` method with:

- Tracking variables for previous targets and priorities
- Threshold-based emission (priority change >= 2, or target change >= 3 or >= 50%)
- Integration in `pick_weighted_config()` and `get_target_jobs_for_node()`

**Changes:**

```python
# Added tracking state:
self._previous_targets: dict[str, int] = {}
self._previous_priorities: dict[str, int] = {}

# Added event emission method:
def _emit_selfplay_target_updated(
    self,
    config_key: str,
    priority: str,
    reason: str,
    *,
    target_jobs: int | None = None,
    effective_priority: int | None = None,
    exploration_boost: float | None = None,
) -> None
```

**Impact:** Feedback loops between P2P scheduler and coordination layer now functional.

---

### P0.3: DaemonManager Event Subscriptions

**File:** `app/coordination/daemon_manager.py`

**Issue:** DaemonManager was missing subscriptions for 3 critical feedback loop events.

**Fix:** Added subscriptions in `_subscribe_to_critical_events()`:

```python
# P0.3 (December 2025): Subscribe to feedback loop events
if hasattr(DataEventType, 'SELFPLAY_TARGET_UPDATED'):
    router.subscribe(DataEventType.SELFPLAY_TARGET_UPDATED.value, self._on_selfplay_target_updated)
if hasattr(DataEventType, 'EXPLORATION_BOOST'):
    router.subscribe(DataEventType.EXPLORATION_BOOST.value, self._on_exploration_boost)
if hasattr(DataEventType, 'DAEMON_STATUS_CHANGED'):
    router.subscribe(DataEventType.DAEMON_STATUS_CHANGED.value, self._on_daemon_status_changed)
```

**Added Handlers:**

1. `_on_selfplay_target_updated()` - Adjusts daemon worker counts based on priority
2. `_on_exploration_boost()` - Adjusts daemon temperature parameters
3. `_on_daemon_status_changed()` - Self-healing restart for failed daemons

---

### P0.4: Async/Sync Boundary Fix in coordination_bootstrap.py

**File:** `app/coordination/coordination_bootstrap.py`

**Issue:** Line 1596 called `coordinator.shutdown()` without await, but many coordinators have `async def shutdown()`.

**Fix:** Added proper async detection:

```python
# P0.4 (December 2025): Handle async shutdown methods properly
if coordinator and hasattr(coordinator, "shutdown"):
    import asyncio
    import inspect
    shutdown_method = getattr(coordinator, "shutdown")
    if inspect.iscoroutinefunction(shutdown_method):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(shutdown_method())
        except RuntimeError:
            asyncio.run(shutdown_method())
    else:
        shutdown_method()
```

**Impact:** Eliminates "coroutine was never awaited" warnings.

---

### P0.5: Deprecated Event Bus Migration

**Files Updated:**

1. `app/coordination/selfplay_orchestrator.py` (lines 232-234)
2. `app/coordination/data_pipeline_orchestrator.py` (lines 656-658)
3. `app/coordination/auto_export_daemon.py` (lines 295-297)
4. `app/coordination/training_trigger_daemon.py` (lines 258-260)

**Change:** Migrated from deprecated `get_stage_event_bus()` to unified `get_router()`:

```python
# Before:
from app.coordination.event_router import get_stage_event_bus
bus = get_stage_event_bus()

# After:
from app.coordination.event_router import get_router
router = get_router()
```

**Note:** `app/coordination/auto_evaluation_daemon.py` was already archived - no update needed.

---

## Test Coverage Status

| Module                   | Tests    | Status  |
| ------------------------ | -------- | ------- |
| `event_router.py`        | 61 tests | Passing |
| `sync_router.py`         | 61 tests | Passing |
| Total coordination tests | 305+     | Passing |

---

## P2P Manager Delegation Status (100% Complete)

All 7 P2P managers fully delegated from `p2p_orchestrator.py`:

| Manager             | Methods | LOC Removed | Status   |
| ------------------- | ------- | ----------- | -------- |
| StateManager        | 7/7     | ~200        | Complete |
| TrainingCoordinator | 5/5     | ~450        | Complete |
| JobManager          | 7/7     | ~400        | Complete |
| SyncPlanner         | 4/4     | ~60         | Complete |
| SelfplayScheduler   | 7/7     | ~430        | Complete |
| NodeSelector        | 6/6     | ~50         | Complete |
| LoopManager         | 5/5     | ~400        | Complete |

**Total:** ~1,990 LOC removed from p2p_orchestrator.py

---

## Remaining Consolidation Work (P1)

### P1.1: Train.py Decomposition (5,471 LOC)

**Current State:**

- `train.py`: 5,471 LOC with `train_model()` at 4,840 LOC

**Planned Decomposition:**
| Target Module | Est. LOC | Purpose |
|---------------|----------|---------|
| `train_loop_core.py` | ~1,500 | Core training loop, epoch iteration |
| `train_optimization.py` | ~600 | Optimizer, LR scheduling |
| `train_data.py` | ~700 | Data loading, streaming |
| `train_checkpoints.py` | ~500 | Save/load, resume |
| `train_distributed.py` | ~400 | DDP setup |
| `train_enhancements.py` | ~600 | Curriculum, mining |
| `train.py` | ~500 | Facade only |

**Target:** Reduce train.py from 5,471 to <800 LOC.

### P1.2: Deprecated Module Removal (Q2 2026)

| Module                     | Replacement                            | Active Callers                        |
| -------------------------- | -------------------------------------- | ------------------------------------- |
| `cluster_data_sync.py`     | `AutoSyncDaemon(strategy="broadcast")` | daemon_manager.py                     |
| `ephemeral_sync.py`        | `AutoSyncDaemon(strategy="ephemeral")` | daemon_manager.py, selfplay_runner.py |
| `system_health_monitor.py` | `unified_health_manager.py`            | daemon_manager.py                     |
| `node_health_monitor.py`   | `health_check_orchestrator.py`         | None (safe to archive)                |

---

## Summary of December 2025 Changes

| Phase    | Date   | Changes                                                    | LOC Impact    |
| -------- | ------ | ---------------------------------------------------------- | ------------- |
| Phase 10 | Dec 26 | Idle daemon consolidation, distribution daemon unification | -1,418        |
| Phase 11 | Dec 27 | P0 critical event fixes, P2P feedback loop wiring          | ~+200 (fixes) |

**Key Achievements (Dec 27):**

- All P0 critical issues resolved
- Event system now properly wired between P2P and coordination layers
- 61 tests for event_router.py, 61 tests for sync_router.py (passing)
- P2P manager delegation 100% complete

---

## Next Steps

1. **P1.1**: Begin train.py decomposition planning
2. **Archive**: Move node_health_monitor.py (no active callers)
3. **Q2 2026**: Remove deprecated sync modules after migration period

---

## Session 6: Type Consolidation & Test Coverage (Dec 27, 2025 - Afternoon)

### Type Definition Consolidation

Resolved critical duplicate type definitions:

| Type                | Before                                                                        | After                                             | Change                  |
| ------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------- |
| `CoordinatorStatus` | Defined in both `contracts.py` and `handler_base.py` with different values    | `handler_base.py` now imports from `contracts.py` | Single source of truth  |
| `HealthCheckResult` | Defined in both modules with different defaults                               | `handler_base.py` now imports from `contracts.py` | Single source of truth  |
| `HandlerStats`      | Defined in both `base_handler.py` and `handler_base.py` with different fields | Unified in `handler_base.py` with all fields      | Backward-compat aliases |

### Base Class Infrastructure

**HandlerBase** (`app/coordination/handler_base.py` - 550 LOC, 45 tests):

- Unified base class for 53+ handlers/daemons
- Provides: singleton pattern, event subscription, health checks, error tracking, lifecycle management
- Backward-compatible aliases: `BaseEventHandler`, `BaseSingletonHandler`, `MultiEventHandler`
- Tests: `tests/unit/coordination/test_handler_base.py`

**P2PMixinBase** (`scripts/p2p/p2p_mixin_base.py` - 502 LOC, 30 tests):

- Unified base class for 6 P2P mixin files
- Provides: DB helpers, state initialization, peer counting, event emission
- Tests: `tests/unit/p2p/test_p2p_mixin_base.py`

### Test Coverage Additions

| Test File                    | Tests           | Coverage                         |
| ---------------------------- | --------------- | -------------------------------- |
| `test_contracts.py`          | 48              | All contracts.py exports         |
| `test_handler_base.py`       | 45              | HandlerBase full coverage        |
| `test_providers.py`          | 28 (+2 skipped) | Cloud provider abstraction       |
| `test_training_reexports.py` | 13              | Training coordination re-exports |
| `test_p2p_mixin_base.py`     | 30              | P2P mixin base class             |

**Total new tests this session:** ~164 tests added/verified

### HandlerBase Adoption Candidates (Future Work)

Large orchestrators that could benefit from HandlerBase adoption:

| File                            | LOC   | Current Pattern  | Migration Risk |
| ------------------------------- | ----- | ---------------- | -------------- |
| `auto_sync_daemon.py`           | 3,668 | Manual lifecycle | HIGH           |
| `data_pipeline_orchestrator.py` | 3,505 | Manual lifecycle | HIGH           |
| `feedback_loop_controller.py`   | 2,716 | Manual singleton | HIGH           |
| `idle_resource_daemon.py`       | 2,299 | Manual lifecycle | HIGH           |
| `curriculum_integration.py`     | 1,452 | Manual singleton | MEDIUM         |

**Recommendation:** Defer large orchestrator refactoring. New handlers should use HandlerBase.

### Deprecated Module Status

Archived to `archive/deprecated_coordination/` (6,005 LOC):

- `queue_populator_daemon.py` → `unified_queue_populator.py`
- `replication_monitor.py` → `unified_replication_daemon.py`
- `replication_repair_daemon.py` → `unified_replication_daemon.py`
- `model_distribution_daemon.py` → `unified_distribution_daemon.py`
- `npz_distribution_daemon.py` → `unified_distribution_daemon.py`

### Summary

| Phase     | Date      | Changes                           | LOC Impact   |
| --------- | --------- | --------------------------------- | ------------ |
| Phase 10  | Dec 26    | Idle daemon consolidation         | -1,418       |
| Phase 11  | Dec 27 AM | P0 critical event fixes           | +200 (fixes) |
| Session 6 | Dec 27 PM | Type consolidation, test coverage | +164 tests   |

**Key Achievements (Session 6):**

- Type definitions unified (CoordinatorStatus, HealthCheckResult, HandlerStats)
- 116 consolidation tests passing
- HandlerBase and P2PMixinBase verified production-ready
- Provider tests fixed (Lambda provider properly skipped)

---

_Last Updated: December 27, 2025 (Session 6)_
