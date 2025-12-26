# Deprecated Coordination Modules

This directory contains coordination modules that have been superseded by consolidated implementations or are no longer needed.

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

## lambda_idle_daemon.py

**Archived**: December 2025

**Reason**: Lambda Labs GPU account terminated December 2025

**Migration**: Not needed - functionality specific to Lambda infrastructure

**Alternatives**:

If you need similar idle node shutdown functionality for other providers:
- `app/coordination/idle_resource_daemon.py` - Generic idle resource management
- `app/coordination/utilization_optimizer.py` - Multi-provider workload optimization
- `app/coordination/cluster_watchdog_daemon.py` - Self-healing cluster monitoring

**Original Purpose**:

The Lambda Idle Daemon automatically terminated idle Lambda Labs GPU nodes to reduce costs.

Key features:
- Monitored Lambda nodes for idle detection (30+ minutes at <5% GPU utilization)
- Checked for pending work before termination
- Graceful shutdown with pending job drain
- Cost tracking and savings reporting
- Event emission for observability

The daemon integrated with:
- P2P orchestrator for cluster node discovery
- Work queue for pending job detection
- Lambda Labs API for instance termination
- SSH fallback for manual shutdown

**Deprecation Status**:

- `DaemonType.LAMBDA_IDLE` in `daemon_manager.py`: Marked deprecated, removal planned Q2 2026
- Environment variables in `app/config/env.py`: Marked deprecated
- No code migration needed - daemon no longer runs in cluster
