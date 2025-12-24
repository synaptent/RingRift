# Deprecated Coordination Modules

This directory contains coordination modules that have been superseded by consolidated implementations.

## unified_event_coordinator.py

**Archived**: December 2025

**Reason**: Functionality consolidated into `app/coordination/event_router.py`

**Migration**:

- All imports from `unified_event_coordinator` can be replaced with imports from `event_router`
- The following aliases are provided in `event_router.py` for backwards compatibility:
  - `UnifiedEventCoordinator` -> alias for `UnifiedEventRouter`
  - `get_event_coordinator()` -> alias for `get_router()`
  - `start_coordinator()` / `stop_coordinator()`
  - `get_coordinator_stats()` -> returns `CoordinatorStats`
  - All `emit_*` helper functions

**Original Purpose**:
The unified_event_coordinator bridged three event systems:

1. DataEventBus (data_events.py) - In-memory async events
2. StageEventBus (stage_events.py) - Pipeline stage events
3. CrossProcessEventQueue (cross_process_events.py) - SQLite-backed IPC

This functionality is now provided by `UnifiedEventRouter` in `event_router.py`, which:

- Provides the same bridging between event systems
- Has a cleaner API with unified `publish()` and `subscribe()` methods
- Includes event history and metrics
- Supports cross-process polling
