# Sync Modules Usage Analysis

**Date**: December 26, 2025
**Analyst**: Claude
**Purpose**: Identify unused/deprecated sync modules for archival

> Status: Historical snapshot (Dec 2025). Kept for reference; current sync design is documented in `ai-service/docs/sync_architecture.md`.

## Executive Summary

Analyzed sync-related modules in `ai-service/app/coordination/` to identify candidates for archival.

**Result**: 1 unused module identified for archival.

## Modules Analyzed

### 1. cluster_data_sync.py

**Location**: `ai-service/app/coordination/cluster_data_sync.py`

**Status**: ✅ **ACTIVELY USED - DO NOT ARCHIVE**

**Import Count**: 5 active imports

**Imported By**:

1. `app/coordination/daemon_manager.py` (3 imports):
   - Line 1541: `from app.coordination.cluster_data_sync import get_training_node_watcher`
   - Line 2008: `from app.coordination.cluster_data_sync import ClusterDataSyncDaemon`
   - Line 2627: `from app.coordination.cluster_data_sync import sync_game`

2. `app/coordination/daemon_adapters.py` (2 imports):
   - Line 338: `from app.coordination.cluster_data_sync import ClusterDataSyncDaemon`
   - Line 359: `from app.coordination.cluster_data_sync import SYNC_INTERVAL_SECONDS`

**Purpose**:

- Cluster Data Sync Daemon - ensures game data availability across cluster nodes
- Push-based sync to eligible nodes with disk space filtering
- Integrates with DaemonManager lifecycle
- Registered as `DaemonType.CLUSTER_DATA_SYNC` and `DaemonType.TRAINING_NODE_WATCHER`

**Conclusion**: This is an active, production module. Must NOT be archived.

---

### 2. sync_coordination_core.py

**Location**: `ai-service/app/coordination/sync_coordination_core.py`

**Status**: ⚠️ **UNUSED - CANDIDATE FOR ARCHIVAL**

**Import Count**: 0 external imports

**Self-References Only**:

- All references to `SyncCoordinationCore` and `get_sync_coordination_core` are within the file itself
- No other modules in the codebase import from this file

**Purpose** (as documented):

- Central coordinator for sync operations
- Manages sync lifecycle: request → queue → execute → track state
- Integrates with SyncRouter and bandwidth management
- Event-driven sync request handling

**Why It's Unused**:

- The actual sync coordination is handled by:
  - `app/coordination/sync_coordinator.py` (17 imports - ACTIVE)
  - `app/coordination/sync_router.py` (used by sync_coordination_core itself)
  - `app/coordination/sync_bandwidth.py` (ACTIVE)
  - `app/coordination/auto_sync_daemon.py` (ACTIVE)
  - `app/distributed/sync_coordinator.py` (ACTIVE)

**Superseded By**:

- `sync_coordinator.py` - provides actual sync scheduling and coordination
- `auto_sync_daemon.py` - provides automated P2P sync
- `sync_router.py` - provides routing logic
- Event system already integrated into active modules

**Recommendation**: Archive to `archive/deprecated_coordination/sync_coordination_core.py`

---

## Additional Sync Modules Checked

### 3. app/coordination/cluster/sync.py

**Status**: ✅ **ACTIVE** (Re-export module)

This is a convenience re-export module that consolidates:

- `sync_coordinator.py` exports (SyncScheduler, SyncCoordinator, etc.)
- `sync_bandwidth.py` exports (BandwidthCoordinatedRsync, etc.)
- `sync_mutex.py` exports (SyncMutex, etc.)

**Import Count**: Referenced by `app/coordination/cluster/__init__.py`

**Conclusion**: Active module providing unified import interface.

---

### 4. app/distributed/unified_data_sync.py

**Status**: ⚠️ **DEPRECATED (still in use)** (3 imports)

**Imported By**:

- `scripts/unified_data_sync.py` (supported CLI wrapper)
- `tests/test_unified_loop.py`
- `tests/unit/distributed/test_checksum_validation.py`

**Conclusion**: Deprecated module that remains active via the CLI and tests. Keep until AutoSyncDaemon/SyncFacade parity replaces the CLI path.

---

### 5. app/distributed/sync_orchestrator.py

**Status**: ✅ **ACTIVE** (2 imports)

**Imported By**:

- Self-reference
- `tests/unit/distributed/test_sync_orchestrator.py`

**Conclusion**: Active with dedicated test suite.

---

### 6. app/distributed/sync_coordinator.py

**Status**: ✅ **ACTIVE** (17 imports)

Heavily used across:

- `app/coordination/daemon_manager.py`
- `app/coordination/sync_coordinator.py`
- `scripts/sync_models.py`
- `scripts/p2p_orchestrator.py`
- Multiple test files

**Conclusion**: Core active module.

---

## Recommendation Summary

### Archive (1 module):

1. ✅ `app/coordination/sync_coordination_core.py`
   - Zero external imports
   - Functionality covered by other active modules
   - Move to: `archive/deprecated_coordination/sync_coordination_core.py`

### Keep Active (All others):

1. ✅ `app/coordination/cluster_data_sync.py` - 5 imports, daemon integration
2. ✅ `app/coordination/sync_coordinator.py` - Core sync coordination
3. ✅ `app/coordination/sync_bandwidth.py` - Bandwidth management
4. ✅ `app/coordination/sync_router.py` - Routing logic
5. ✅ `app/coordination/auto_sync_daemon.py` - P2P automation
6. ✅ `app/distributed/sync_coordinator.py` - 17 imports
7. ✅ `app/distributed/unified_data_sync.py` - 3 imports
8. ✅ `app/distributed/sync_orchestrator.py` - 2 imports

## Archival Actions Completed

### sync_coordination_core.py - ARCHIVED ✅

**Date**: December 26, 2025

**Actions Completed**:

1. ✅ Moved file to archive:

   ```bash
   mv app/coordination/sync_coordination_core.py archive/deprecated_coordination/
   ```

   - Source: `ai-service/app/coordination/sync_coordination_core.py`
   - Destination: `ai-service/archive/deprecated_coordination/sync_coordination_core.py`
   - File size: 27KB

2. ✅ Updated `archive/deprecated_coordination/README.md` with comprehensive documentation:
   - Reason for archival (zero external imports)
   - Superseded by list (5 active modules)
   - Original purpose documentation
   - Migration guidance
   - Verification commands

3. ✅ Verified no broken imports:

   ```bash
   # All active sync modules import successfully
   ✓ sync_coordinator imports successfully
   ✓ sync_router imports successfully
   ✓ auto_sync_daemon imports successfully
   ✓ daemon_manager imports successfully
   ✓ CLUSTER_DATA_SYNC in DaemonType: True
   ```

4. ✅ Verified file moved correctly:
   ```bash
   SUCCESS: File moved to archive
   # File no longer exists in app/coordination/
   # File exists in archive/deprecated_coordination/
   ```

**Impact**: NONE - Zero imports means no code was using this module

## Verification Commands Used

```bash
# Search for imports
grep -r "from app.coordination.cluster_data_sync import" --include="*.py" .
grep -r "from app.coordination.sync_coordination_core import" --include="*.py" .

# Count references
grep -r "cluster_data_sync" --include="*.py" . | wc -l
grep -r "sync_coordination_core" --include="*.py" . | wc -l

# Check daemon registrations
grep -r "CLUSTER_DATA_SYNC\|TRAINING_NODE_WATCHER" app/coordination/daemon_types.py
```

## Notes

- The `sync_constants.py` module provides shared constants (SyncState, SyncPriority, SyncResult) and is referenced by both `cluster_data_sync.py` and `sync_coordination_core.py`
- Migration from `sync_coordination_core.py` to active modules was likely completed in December 2025 as part of the coordination infrastructure consolidation
- No deprecation warnings are currently emitted for `sync_coordination_core.py` (unlike other deprecated modules)
