# Deprecated Coordination Modules

This directory contains coordination modules that have been superseded by consolidated implementations or are no longer needed.

## lambda_idle_daemon.py & vast_idle_daemon.py

**Archived**: December 26, 2025

**Reason**: Consolidated into `app/coordination/unified_idle_shutdown_daemon.py` (318 LOC saved)

**Superseded By**:

- `app/coordination/unified_idle_shutdown_daemon.py` - Provider-agnostic idle shutdown

**Original Purpose**:
Provider-specific idle detection and shutdown for cloud GPU instances:

- `lambda_idle_daemon.py`: Lambda Labs GPU idle detection (30 min threshold)
- `vast_idle_daemon.py`: Vast.ai GPU idle detection (15 min threshold)

**Migration**:

```python
# OLD - Provider-specific daemons
from app.coordination.lambda_idle_daemon import LambdaIdleDaemon
from app.coordination.vast_idle_daemon import VastIdleDaemon

lambda_daemon = LambdaIdleDaemon()
vast_daemon = VastIdleDaemon()

# NEW - Unified provider-agnostic daemon
from app.coordination.unified_idle_shutdown_daemon import (
    create_lambda_idle_daemon,
    create_vast_idle_daemon,
    create_runpod_idle_daemon,
)

lambda_daemon = create_lambda_idle_daemon()
vast_daemon = create_vast_idle_daemon()
runpod_daemon = create_runpod_idle_daemon()  # NEW! RunPod support added
```

**Key improvements in unified daemon**:

- Provider-agnostic design using CloudProvider interface
- Configurable thresholds per provider (Lambda: 30min, Vast: 15min, RunPod: 20min)
- Pending work check before termination
- Minimum node retention to maintain cluster capacity
- Graceful shutdown with drain period
- Cost tracking and reporting

**Environment variables (per-provider)**:

- `{PROVIDER}_IDLE_ENABLED` - Enable/disable daemon (default: true)
- `{PROVIDER}_IDLE_THRESHOLD` - Idle threshold in seconds
- `{PROVIDER}_IDLE_UTIL_THRESHOLD` - GPU utilization % below which is idle (default: 10)
- `{PROVIDER}_MIN_NODES` - Minimum nodes to retain
- `{PROVIDER}_DRAIN_PERIOD` - Drain period before termination
- `{PROVIDER}_IDLE_DRY_RUN` - Log actions without executing

---

## sync_coordination_core.py

**Archived**: December 26, 2025

**Reason**: Zero external imports detected. Functionality superseded by active sync modules.

**Superseded By**:

The sync coordination functionality is now handled by these active modules:

- `app/coordination/sync_coordinator.py` - Actual sync scheduling and coordination (17+ imports)
- `app/coordination/auto_sync_daemon.py` - Automated P2P data sync
- `app/coordination/sync_router.py` - Intelligent sync routing decisions
- `app/coordination/sync_bandwidth.py` - Bandwidth-coordinated transfers
- `app/distributed/sync_coordinator.py` - Distributed sync coordination

**Original Purpose**:

Central coordinator for sync operations with the following responsibilities:

- Listen for SYNC_REQUEST events and execute sync operations
- Track sync state across the cluster
- Manage sync priorities and queuing
- Emit sync completion/failure events
- Integrate with SyncRouter and bandwidth management

**Migration**:

No migration needed - this module had zero external imports and was not being used.

If you need sync coordination functionality, use:

```python
# For sync scheduling and coordination
from app.coordination.sync_coordinator import get_sync_coordinator
coordinator = get_sync_coordinator()

# For automated P2P sync
from app.coordination.auto_sync_daemon import AutoSyncDaemon
daemon = AutoSyncDaemon()

# For sync routing decisions
from app.coordination.sync_router import get_sync_router
router = get_sync_router()
```

**Verification**:

Grep analysis confirmed zero imports:

```bash
grep -r "from app.coordination.sync_coordination_core import" --include="*.py" .
# Result: No matches found (only self-references in the file itself)
```

---

## sync_coordinator.py

**Archived**: December 26, 2025

**Reason**: Functionality split into more focused modules as part of coordination layer consolidation.

**Superseded By**:

The sync coordination functionality is now handled by these specialized modules:

- `app/coordination/auto_sync_daemon.py` - Automated P2P data sync (push-from-generator + gossip)
- `app/coordination/sync_facade.py` - Unified API for manual sync operations
- `app/coordination/sync_router.py` - Intelligent routing and source selection
- `app/coordination/sync_bandwidth.py` - Bandwidth-coordinated transfers
- `app/distributed/cluster_manifest.py` - Central registry for data locations

**Original Purpose**:

Smart Sync Coordinator (aka SyncScheduler) for cluster-wide data synchronization:

- Unified view of data state across all hosts
- Priority-based sync scheduling (ephemeral hosts prioritized)
- Bandwidth-aware transfer management
- Automatic recovery from sync failures
- Host state tracking (games, sync times, failures)
- Cluster health scoring

Key features that needed to be preserved:

- Priority scoring for hosts (ephemeral hosts get 3x priority)
- Games-behind and time-since-sync weighting
- Backpressure integration with queue monitoring
- Sync history tracking in SQLite

**Migration**:

The old `SyncCoordinator`/`SyncScheduler` tried to do too much (1,400+ lines):

- Sync scheduling AND execution
- Host state tracking AND sync operations
- Event bridging AND data routing

This led to tight coupling and overlapping functionality with:

- `app.distributed.sync_coordinator.SyncCoordinator` (execution layer)
- `SyncRouter`, `SyncBandwidth` (routing and bandwidth)
- `ClusterManifest` (data location tracking)

**New Architecture** (separation of concerns):

1. **AutoSyncDaemon** - Automated background sync scheduling
2. **SyncFacade** - Simple API for manual sync requests
3. **SyncRouter** - Intelligent routing and source selection
4. **SyncBandwidth** - Bandwidth management during transfers
5. **ClusterManifest** - Centralized data location tracking

### Migration Examples

**Automated Background Sync:**

```python
# OLD
from app.coordination.sync_coordinator import get_sync_scheduler, execute_priority_sync
scheduler = get_sync_scheduler()
await execute_priority_sync(max_syncs=3)

# NEW
from app.coordination.auto_sync_daemon import AutoSyncDaemon
daemon = AutoSyncDaemon()
await daemon.start()  # Handles all scheduling automatically
```

**Manual Sync Requests:**

```python
# OLD
from app.coordination.sync_coordinator import get_sync_recommendations
recommendations = get_sync_recommendations(max_recommendations=5)

# NEW
from app.coordination.sync_facade import sync
await sync('games', priority='high')
```

**Cluster Data Tracking:**

```python
# OLD
from app.coordination.sync_coordinator import get_cluster_data_status
status = get_cluster_data_status()
print(f"Stale hosts: {status.stale_hosts}")

# NEW
from app.distributed.cluster_manifest import get_cluster_manifest
manifest = get_cluster_manifest()
stale = manifest.find_stale_data(max_age_hours=1)
```

**Active Imports** (as of December 26, 2025):

- `app/metrics/coordinator.py` - Uses SyncScheduler for metrics
- `app/coordination/coordination_bootstrap.py` - Uses wire_sync_events
- `app/coordination/cluster/sync.py` - Imports SyncScheduler
- `app/coordination/__init__.py` - Re-exports with deprecation warnings suppressed
- `app/distributed/sync_orchestrator.py` - Uses get_sync_scheduler
- `tests/unit/coordination/test_sync_coordinator.py` - Unit tests

**Migration Plan**:

1. Update `app/metrics/coordinator.py` to use ClusterManifest
2. Update `coordination_bootstrap.py` to wire AutoSyncDaemon
3. Update `cluster/sync.py` to use SyncFacade
4. Update `sync_orchestrator.py` to use AutoSyncDaemon
5. Update tests to test new components
6. Delete original file after Q2 2026 verification

**Verification Commands**:

```bash
# Check for active imports
grep -r "from app.coordination.sync_coordinator import" --include="*.py" .

# Check for direct references
grep -r "SyncScheduler\|get_sync_coordinator" --include="*.py" . | grep -v deprecated
```

---

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

---

## Batch Archive: December 27, 2025 (3,339 LOC)

The following 8 modules were archived from `app/coordination/deprecated/`:

### \_deprecated_auto_evaluation_daemon.py

**Original Purpose**: Background daemon for automatic model evaluation.

**Superseded By**: Integrated into `daemon_manager.py` with EVALUATION_DAEMON type and modern event-driven triggers.

### \_deprecated_cross_process_events.py

**Original Purpose**: SQLite-backed inter-process event queue.

**Superseded By**: `app/coordination/event_router.py` with unified event routing.

### \_deprecated_event_emitters.py

**Original Purpose**: Helper functions for emitting training pipeline events.

**Superseded By**: `app/coordination/event_router.py` emit\_\* functions.

### \_deprecated_health_check_orchestrator.py

**Original Purpose**: Orchestrated health checks across cluster nodes.

**Superseded By**: `app/coordination/unified_health_manager.py` with SystemHealthScore tracking.

### \_deprecated_host_health_policy.py

**Original Purpose**: Policy-based health evaluation for cluster hosts.

**Superseded By**: `app/coordination/unified_health_manager.py` with configurable thresholds.

### \_deprecated_queue_populator_daemon.py

**Original Purpose**: Background daemon for maintaining work queues.

**Superseded By**: `app/coordination/queue_populator.py` active implementation with Elo-based targets.

### \_deprecated_sync_coordinator.py

**Original Purpose**: Scheduling layer for sync operations (1,344 LOC).

**Superseded By**:

- `app/coordination/auto_sync_daemon.py` - Automated sync scheduling
- `app/coordination/sync_router.py` - Intelligent routing
- `app/distributed/sync_coordinator.py` - Active execution layer (2,204 LOC)

**Note**: Two files named "sync_coordinator.py" existed with different purposes:

- `app/coordination/sync_coordinator.py` (DEPRECATED) - Scheduling layer
- `app/distributed/sync_coordinator.py` (ACTIVE) - Execution layer

### \_deprecated_system_health_monitor.py

**Original Purpose**: System-wide health monitoring and scoring.

**Superseded By**: Health scoring consolidated into `app/coordination/unified_health_manager.py`:

- `get_system_health_score()` - Calculate system health
- `get_system_health_level()` - Classify health level
- `should_pause_pipeline()` - Check if pipeline should pause

---

## Batch Archive: December 28, 2025 (1,253 LOC)

The following 5 modules were archived as part of consolidation cleanup:

### \_deprecated_core_base.py (141 LOC)

**Archived**: December 28, 2025

**Reason**: Re-export facade module that was never adopted. Part of "157→15 module consolidation (Phase 5)" that was planned but never completed.

**Original Purpose**: Consolidated re-exports from `coordinator_base.py` and `coordinator_dependencies.py`.

**Migration**: Not needed - no external callers. Use the original modules directly:

```python
from app.coordination.coordinator_base import CoordinatorBase, CoordinatorRegistry
from app.coordination.coordinator_dependencies import get_initialization_order
```

### \_deprecated_core_daemons.py (187 LOC)

**Archived**: December 28, 2025

**Reason**: Re-export facade module that was never adopted.

**Original Purpose**: Consolidated re-exports for daemon-related classes.

**Migration**: Use original modules directly.

### \_deprecated_core_sync.py (199 LOC)

**Archived**: December 28, 2025

**Reason**: Re-export facade module that was never adopted.

**Original Purpose**: Consolidated re-exports for sync-related classes.

**Migration**: Use original modules directly.

### \_deprecated_alert_types.py (380 LOC)

**Archived**: December 28, 2025

**Reason**: Only imported by test files, not used in production code.

**Original Purpose**: Alert type enumerations and dataclasses including `AlertLevel`, `AlertCategory`, `ClusterAlert`.

**Migration**: If alert types are needed, the active implementation is in `app/coordination/types.py`.

### \_deprecated_event_subscription_mixin.py (346 LOC)

**Archived**: December 28, 2025

**Reason**: Never imported by any file.

**Original Purpose**: Mixin for adding event subscription capabilities to classes.

**Migration**: Not needed - functionality available via `app/coordination/event_router.py` subscribe/publish pattern.

---

## Verification

All archived modules have **zero external imports** verified by:

```bash
# December 27, 2025 batch
grep -r "from app.coordination.deprecated._deprecated" --include="*.py" .
# Result: No matches found

# December 28, 2025 batch
grep -rn "core_base\|core_daemons\|core_sync\|alert_types\|event_subscription_mixin" app/ scripts/ --include="*.py" | grep -v __pycache__
# Result: Only self-references and __init__.py re-exports (now removed)
```

## Batch Archive: December 28, 2025 Session 9 - Re-export Shims (~160 LOC)

The following 3 modules were identified as unused re-export shims and archived:

### \_deprecated_sync_safety.py (~54 LOC)

**Archived**: December 28, 2025

**Reason**: Pure re-export module with no external callers.

**Original Purpose**: Convenience re-export of sync reliability features.

**Migration**: Import directly from specialized modules:

```python
# OLD
from app.coordination.sync_safety import SyncWAL, check_sqlite_integrity

# NEW
from app.coordination.sync_durability import SyncWAL
from app.coordination.sync_integrity import check_sqlite_integrity
```

### \_deprecated_base_handler.py (~56 LOC)

**Archived**: December 28, 2025

**Reason**: Pure re-export module superseded by handler_base.py.

**Original Purpose**: Base classes for event handlers.

**Migration**:

```python
# OLD
from app.coordination.base_handler import BaseEventHandler

# NEW
from app.coordination.handler_base import HandlerBase
```

### \_deprecated_event_subscription_mixin.py (~346 LOC)

**Archived**: December 27, 2025 (updated December 28, 2025)

**Reason**: Superseded by handler_base.py event subscription pattern.

**Migration**: Use HandlerBase.\_get_event_subscriptions() instead.

---

## Archive Summary

| Date              | Modules                                                                   | LOC        |
| ----------------- | ------------------------------------------------------------------------- | ---------- |
| December 26, 2025 | lambda_idle_daemon, vast_idle_daemon                                      | ~600       |
| December 27, 2025 | 8 deprecated modules                                                      | 3,339      |
| December 28, 2025 | core_base, core_daemons, core_sync, alert_types, event_subscription_mixin | 1,253      |
| December 28, 2025 | sync_safety, base_handler (re-export shims)                               | ~160       |
| **Total**         | **17 modules**                                                            | **~5,352** |
