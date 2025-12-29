# Sync Implementation Consolidation Plan

**Date**: December 26, 2025
**Status**: Analysis Complete - Implementation Ready
**Goal**: Consolidate 8 competing sync implementations into unified facade

> Status: Historical snapshot (Dec 2025). Kept for reference; current sync docs live under `ai-service/docs/`.

## Executive Summary

The codebase has evolved 8 different sync implementations over time, creating confusion and maintenance burden. This document analyzes all implementations, identifies which are actively used, and provides a consolidation plan with minimal disruption.

**Key Finding**: Only 4 implementations are actively used. The remaining 4 can be deprecated with notices.

## Current Implementations

### 1. SyncCoordinator (`app/distributed/sync_coordinator.py`)

- **Purpose**: Low-level sync EXECUTION layer
- **Features**: aria2, SSH, P2P, NFS optimization
- **Status**: ✅ **ACTIVE** - Core transport layer
- **Usage**: Used by daemon_manager.py (`_create_sync_coordinator`)
- **Keep**: Yes - This is the canonical transport execution layer

### 2. SyncScheduler (`app/coordination/sync_coordinator.py`)

- **Purpose**: Sync SCHEDULING layer (decides WHEN/WHAT to sync)
- **Features**: Data freshness tracking, priority-based scheduling
- **Status**: ⚠️ **DEPRECATED** (marked in code)
- **Usage**: Very limited - mostly compatibility exports
- **Action**: Add strong deprecation notice, migrate to AutoSyncDaemon

### 3. UnifiedDataSync (`app/distributed/unified_data_sync.py`)

- **Purpose**: Legacy unified sync service
- **Features**: Continuous polling, multi-transport, WAL, deduplication
- **Status**: ⚠️ **DEPRECATED** (internal module) - superseded by newer daemons
- **Usage**: Used by `scripts/unified_data_sync.py` and test coverage; keep CLI support until AutoSyncDaemon/SyncFacade fully replaces it
- **Action**: Maintain deprecation notice; preserve the external CLI while migrating call sites

### 4. SyncOrchestrator (`app/distributed/sync_orchestrator.py`)

- **Purpose**: Orchestrates multiple sync components (data, model, elo, registry)
- **Features**: Unified initialization, coordinated scheduling
- **Status**: 🤔 **UNCERTAIN** - Wrapper, may not be needed
- **Usage**: Referenced but unclear if actively used
- **Action**: Evaluate if still needed, possibly deprecate in favor of facade

### 5. AutoSyncDaemon (`app/coordination/auto_sync_daemon.py`)

- **Purpose**: Automated P2P data sync
- **Features**: Push-from-generator + P2P gossip replication
- **Status**: ✅ **ACTIVE** - Primary automated sync mechanism
- **Usage**: Used by daemon_manager.py (`_create_auto_sync`), registered as DaemonType.AUTO_SYNC
- **Keep**: Yes - This is the main automated sync daemon

### 6. ClusterDataSyncDaemon (`app/coordination/cluster_data_sync.py`)

- **Purpose**: Push-based cluster-wide data sync
- **Features**: Leader-driven push to all nodes, event-driven + periodic
- **Status**: ✅ **ACTIVE** - Registered daemon
- **Usage**: Used by daemon_manager.py (`_create_cluster_data_sync`), registered as DaemonType.CLUSTER_DATA_SYNC
- **Keep**: Yes - Provides push-based sync complementary to P2P gossip

### 7. SyncRouter (`app/coordination/sync_router.py`)

- **Purpose**: Intelligent data routing based on node capabilities
- **Features**: Disk capacity checks, NFS detection, ephemeral node handling
- **Status**: ✅ **ACTIVE** - Used by multiple components
- **Usage**: Used by replication_repair_daemon, imports throughout codebase
- **Keep**: Yes - Provides intelligent routing logic

### 8. EphemeralSyncDaemon (`app/coordination/ephemeral_sync.py`)

- **Purpose**: Aggressive sync for ephemeral hosts (Vast.ai, spot instances)
- **Features**: 5-second poll, immediate push, termination handling
- **Status**: ✅ **ACTIVE** - Registered daemon
- **Usage**: Used by daemon_manager.py (`_create_ephemeral_sync`), registered as DaemonType.EPHEMERAL_SYNC
- **Keep**: Yes - Critical for ephemeral node data safety

## Consolidation Strategy

### Phase 1: Create SyncFacade (COMPLETE ✅)

**File**: `app/coordination/sync_facade.py`

**Features**:

- Single entry point: `sync(data_type, targets, **kwargs)`
- Routes to appropriate backend based on request
- Logs which implementation is used
- Provides metrics and stats
- Backward compatible

**Usage**:

```python
from app.coordination.sync_facade import sync

# Simple usage
await sync("games")

# With targets
await sync("models", targets=["node-1", "node-2"])

# With routing hints
await sync("games", board_type="hex8", priority="high", prefer_ephemeral=True)
```

### Phase 2: Add Deprecation Notices

Mark deprecated implementations with clear warnings and migration paths:

#### 2.1 SyncScheduler

**File**: `app/coordination/sync_coordinator.py`

**Current Status**: Already has deprecation warning at import time

**Action**: Strengthen deprecation notice with migration example:

```python
warnings.warn(
    "SyncScheduler is deprecated and will be removed in Q2 2026. "
    "Use AutoSyncDaemon for automated sync or SyncFacade for manual sync:\n"
    "  from app.coordination.sync_facade import sync\n"
    "  await sync('games', priority='high')\n"
    "Alternatively, use app.coordination.auto_sync_daemon.AutoSyncDaemon "
    "for automated P2P sync.",
    DeprecationWarning,
    stacklevel=2,
)
```

#### 2.2 UnifiedDataSync

**File**: `app/distributed/unified_data_sync.py`

**Action**: Add deprecation warning at module level:

```python
warnings.warn(
    "UnifiedDataSyncService is deprecated. "
    "Use AutoSyncDaemon for automated sync or SyncFacade for direct sync:\n"
    "  from app.coordination import AutoSyncDaemon\n"
    "  daemon = AutoSyncDaemon()\n"
    "  await daemon.start()\n"
    "This module will be archived in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)
```

**CLI note**: `scripts/unified_data_sync.py` remains the supported external sync service entry point while AutoSyncDaemon/SyncFacade parity is completed.

#### 2.3 SyncOrchestrator (Conditional)

**File**: `app/distributed/sync_orchestrator.py`

**Action**: Evaluate usage first. If not actively used, add deprecation:

```python
warnings.warn(
    "SyncOrchestrator may be deprecated. "
    "Consider using SyncFacade for unified sync operations:\n"
    "  from app.coordination.sync_facade import sync\n"
    "  await sync('all')  # Syncs data, models, elo, registry\n"
    "Check with maintainers before relying on this module.",
    PendingDeprecationWarning,
    stacklevel=2,
)
```

### Phase 3: Update Documentation

#### 3.1 Update CLAUDE.md

Add SyncFacade to the "Key Utilities" section:

````markdown
### SyncFacade (`app/coordination/sync_facade.py`)

Unified entry point for all cluster sync operations:

```python
from app.coordination.sync_facade import sync

# Sync games to all eligible nodes
await sync("games")

# Sync models to specific targets
await sync("models", targets=["node-1", "node-2"])

# High-priority sync with routing
await sync("games", board_type="hex8", priority="high")
```
````

**Active Implementations** (Do not use directly - use SyncFacade):

- `AutoSyncDaemon` - Automated P2P sync
- `ClusterDataSyncDaemon` - Push-based cluster sync
- `SyncRouter` - Intelligent routing
- `EphemeralSyncDaemon` - Aggressive sync for ephemeral hosts
- `SyncCoordinator` (distributed) - Low-level transport

**Deprecated Implementations**:

- ❌ `SyncScheduler` - Use `AutoSyncDaemon`
- ❌ `UnifiedDataSync` - Use `AutoSyncDaemon`
- ⚠️ `SyncOrchestrator` - Use `SyncFacade`

````

#### 3.2 Create Migration Guide
**File**: `docs/sync_migration_guide.md`

Document how to migrate from each deprecated implementation to SyncFacade.

### Phase 4: Update Imports

Update high-value locations to use SyncFacade:

#### 4.1 app/coordination/__init__.py
SyncFacade exports are already wired; keep this in sync with `app/coordination/sync_facade.py`:
```python
# Sync Facade (December 2025 - unified sync entry point)
from app.coordination.sync_facade import (
    SyncBackend,
    SyncFacade,
    SyncRequest,
    SyncResponse,
    get_sync_facade,
    reset_sync_facade,
    sync,
)
````

#### 4.2 Training Scripts

Update scripts that manually trigger sync to use facade:

- `scripts/run_training_loop.py`
- `scripts/sync_models.py`
- `scripts/unified_data_sync.py` (keep supported CLI; rebase onto AutoSyncDaemon/SyncFacade before deprecating the internal module)

### Phase 5: Testing

#### 5.1 Unit Tests

**File**: `tests/unit/coordination/test_sync_facade.py`

Test all routing logic:

- Backend selection
- Request normalization
- Error handling
- Metrics tracking

#### 5.2 Integration Tests

**File**: `tests/integration/test_sync_consolidation.py`

Verify backward compatibility:

- Old imports still work (with warnings)
- Facade routes to correct backends
- All active daemons function correctly

### Phase 6: Monitoring & Rollout

1. **Add metrics** to track which backends are used via facade
2. **Deploy to staging** cluster first
3. **Monitor logs** for deprecation warnings
4. **Identify stragglers** still using deprecated implementations
5. **Migrate stragglers** to SyncFacade
6. **Full rollout** to production cluster

## Implementation Status

### Completed ✅

- [x] Analysis of all 8 sync implementations
- [x] Identification of active vs deprecated implementations
- [x] Creation of SyncFacade with routing logic
- [x] Documentation of consolidation plan

### In Progress 🔄

- [ ] Add deprecation notices to modules
- [ ] Update documentation (CLAUDE.md, migration guide)
- [ ] Update app/coordination/**init**.py exports

### Pending 📋

- [ ] Write unit tests for SyncFacade
- [ ] Write integration tests
- [ ] Update training scripts to use SyncFacade
- [ ] Deploy to staging and monitor
- [ ] Full production rollout
- [ ] Archive deprecated modules (Q2 2026)

## Backend Selection Logic

The facade uses this logic to select backends:

```
if prefer_ephemeral:
    → EphemeralSyncDaemon
elif priority in ["high", "critical"]:
    → ClusterDataSyncDaemon (push-based)
elif specific targets:
    → SyncRouter (intelligent routing) → SyncCoordinator (transport)
else:
    → AutoSyncDaemon (P2P gossip)
```

## Migration Examples

### Example 1: Replace Manual Sync Call

**Before**:

```python
from app.distributed.sync_coordinator import SyncCoordinator

coordinator = SyncCoordinator.get_instance()
await coordinator.sync_games(board_type="hex8")
```

**After**:

```python
from app.coordination.sync_facade import sync

await sync("games", board_type="hex8")
```

### Example 2: Replace SyncScheduler

**Before**:

```python
from app.coordination.sync_coordinator import get_sync_scheduler

scheduler = get_sync_scheduler()
await scheduler.schedule_priority_sync()
```

**After**:

```python
from app.coordination.sync_facade import sync

# High-priority sync to all nodes
await sync("games", priority="high")
```

### Example 3: Replace UnifiedDataSync

**Before**:

```python
from app.distributed.unified_data_sync import UnifiedDataSyncService

service = UnifiedDataSyncService.from_config(config_path)
await service.run()
```

**After**:

```python
from app.coordination import AutoSyncDaemon

# For automated continuous sync
daemon = AutoSyncDaemon()
await daemon.start()

# For one-time sync
from app.coordination.sync_facade import sync
await sync("all")
```

## Metrics & Observability

The SyncFacade tracks:

1. **Total syncs** by backend
2. **Bytes transferred** per backend
3. **Error rates** per backend
4. **Average latency** per backend

Access metrics:

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()
stats = facade.get_stats()

print(f"Total syncs: {stats['total_syncs']}")
print(f"By backend: {stats['by_backend']}")
print(f"Total bytes: {stats['total_bytes']}")
```

## Risks & Mitigations

### Risk 1: Breaking existing code that imports deprecated modules directly

**Mitigation**: Keep deprecated modules with warnings for 6 months (Q2 2026 removal)

### Risk 2: Performance regression from indirection

**Mitigation**: Facade is a thin router with minimal overhead (<1ms)

### Risk 3: Missed usages of deprecated implementations

**Mitigation**: grep codebase for imports, emit loud warnings, monitor logs

### Risk 4: Daemon coordination issues

**Mitigation**: Facade delegates to existing daemons - no new coordination logic

## Timeline

- **Week 1** (Dec 26-Jan 2): Complete deprecation notices and docs
- **Week 2** (Jan 3-9): Write tests, update scripts
- **Week 3** (Jan 10-16): Deploy to staging, monitor
- **Week 4** (Jan 17-23): Production rollout
- **Q2 2026**: Archive deprecated modules

## Success Criteria

1. All new code uses SyncFacade
2. No direct imports of deprecated modules in new commits
3. Zero sync-related bugs during rollout
4. Deprecation warnings visible in logs
5. All 4 active daemons continue functioning
6. Documentation updated and clear

## Questions & Decisions

### Q1: Should SyncOrchestrator be deprecated?

**Decision**: Mark as PendingDeprecationWarning, evaluate usage first

### Q2: Should we expose sync_now() methods on daemons?

**Decision**: Yes - add to AutoSyncDaemon, ClusterDataSyncDaemon for facade to call

### Q3: Should facade cache backend instances?

**Decision**: Yes - use singleton pattern for each backend type

### Q4: Should we add retry logic to facade?

**Decision**: No - backends handle retries. Facade just routes.

## References

- **SyncFacade**: `ai-service/app/coordination/sync_facade.py`
- **Active Daemons**:
  - `ai-service/app/coordination/auto_sync_daemon.py`
  - `ai-service/app/coordination/cluster_data_sync.py`
  - `ai-service/app/coordination/ephemeral_sync.py`
  - `ai-service/app/coordination/sync_router.py`
- **Transport Layer**: `ai-service/app/distributed/sync_coordinator.py`
- **Deprecated**:
  - `ai-service/app/coordination/sync_coordinator.py` (SyncScheduler)
  - `ai-service/app/distributed/unified_data_sync.py`
  - `ai-service/app/distributed/sync_orchestrator.py` (tentative)
