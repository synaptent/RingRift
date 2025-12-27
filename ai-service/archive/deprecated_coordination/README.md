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
