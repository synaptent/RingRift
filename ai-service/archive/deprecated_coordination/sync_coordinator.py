#!/usr/bin/env python3
"""Smart Sync Coordinator for unified cluster-wide data management.

================================================================================
DEPRECATED AND ARCHIVED - December 26, 2025
================================================================================

This module has been archived as part of the coordination layer consolidation
effort. The functionality has been split into more focused modules:

SUPERSEDED BY:
-------------

1. **AutoSyncDaemon** (app/coordination/auto_sync_daemon.py)
   - Automated P2P data synchronization
   - Push-from-generator pattern with gossip replication
   - Intelligent sync scheduling and prioritization

   Usage:
   ```python
   from app.coordination.auto_sync_daemon import AutoSyncDaemon
   daemon = AutoSyncDaemon()
   await daemon.start()
   ```

2. **SyncFacade** (app/coordination/sync_facade.py)
   - Unified API for manual sync operations
   - One-time sync requests with priority control

   Usage:
   ```python
   from app.coordination.sync_facade import sync
   await sync('games', priority='high')
   ```

3. **SyncRouter** (app/coordination/sync_router.py)
   - Intelligent routing decisions for sync operations
   - Node capability matching and exclusion rules

   Usage:
   ```python
   from app.coordination.sync_router import get_sync_router
   router = get_sync_router()
   sources = router.select_sources(target='vast-5090')
   ```

4. **SyncBandwidth** (app/coordination/sync_bandwidth.py)
   - Bandwidth-coordinated rsync transfers
   - Per-host bandwidth limits and prioritization

   Usage:
   ```python
   from app.coordination.sync_bandwidth import BandwidthCoordinatedSync
   syncer = BandwidthCoordinatedSync()
   await syncer.sync_with_bandwidth(source, target)
   ```

5. **ClusterManifest** (app/distributed/cluster_manifest.py)
   - Central registry tracking game/model/NPZ locations
   - Cluster-wide data location discovery

   Usage:
   ```python
   from app.distributed.cluster_manifest import get_cluster_manifest
   manifest = get_cluster_manifest()
   locations = manifest.get_game_locations('hex8_2p')
   ```

MIGRATION PATH:
--------------

### For Automated Background Sync

OLD CODE:
```python
from app.coordination.sync_coordinator import (
    get_sync_scheduler,
    execute_priority_sync,
)
scheduler = get_sync_scheduler()
await execute_priority_sync(max_syncs=3)
```

NEW CODE:
```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

# Start the daemon (runs in background)
daemon = AutoSyncDaemon()
await daemon.start()

# Daemon handles:
# - Automatic sync scheduling
# - Priority-based sync ordering
# - Push-from-generator pattern
# - Gossip replication
```

### For Manual Sync Requests

OLD CODE:
```python
from app.coordination.sync_coordinator import get_sync_recommendations
recommendations = get_sync_recommendations(max_recommendations=5)
for rec in recommendations:
    # Execute sync based on recommendation
    pass
```

NEW CODE:
```python
from app.coordination.sync_facade import sync

# Simple one-off sync
await sync('games', priority='high')

# Or check what needs syncing
from app.coordination.sync_router import get_sync_router
router = get_sync_router()
stale_hosts = router.find_stale_hosts()
```

### For Cluster Data Tracking

OLD CODE:
```python
from app.coordination.sync_coordinator import (
    get_cluster_data_status,
    register_host,
    update_host_state,
)
status = get_cluster_data_status()
print(f"Stale hosts: {status.stale_hosts}")
```

NEW CODE:
```python
from app.distributed.cluster_manifest import get_cluster_manifest

manifest = get_cluster_manifest()
stale = manifest.find_stale_data(max_age_hours=1)
print(f"Stale hosts: {list(stale.keys())}")
```

### For Sync Event Wiring

OLD CODE:
```python
from app.coordination.sync_coordinator import wire_sync_events
scheduler = wire_sync_events()
```

NEW CODE:
```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon
from app.coordination.daemon_manager import get_daemon_manager

# AutoSyncDaemon automatically wires to events
daemon_manager = get_daemon_manager()
daemon_manager.start_daemon('sync')
```

ARCHITECTURE NOTES:
------------------

The old SyncCoordinator/SyncScheduler tried to do too much:
- Sync scheduling AND execution
- Host state tracking AND sync operations
- Event bridging AND data routing

This led to tight coupling and made testing difficult.

The new architecture separates concerns:

1. **AutoSyncDaemon**: Automated background sync scheduling
2. **SyncFacade**: Simple API for manual sync requests
3. **SyncRouter**: Intelligent routing and source selection
4. **SyncBandwidth**: Bandwidth management during transfers
5. **ClusterManifest**: Centralized data location tracking

Each module has a single responsibility and can be tested independently.

WHY DEPRECATED:
--------------

1. **Complexity**: 1,400+ lines mixing scheduling, execution, and state tracking
2. **Tight Coupling**: Hard to test individual components
3. **Overlapping Functionality**:
   - Overlapped with DistributedSyncCoordinator (app.distributed.sync_coordinator)
   - Duplicated logic from SyncRouter, SyncBandwidth
   - Reimplemented manifest tracking that ClusterManifest now handles
4. **Name Confusion**:
   - SyncCoordinator exists in both app.coordination and app.distributed
   - SyncScheduler was an alias for the same class
   - Hard to know which one to import

TIMELINE:
--------

- **December 2025**: Deprecation warnings added at import time
- **Q1 2026**: This module archived to deprecated_coordination/
- **Q2 2026**: Module will be deleted entirely

VERIFICATION:
------------

Before deleting this module, verify no active usage:

```bash
# Check for imports
grep -r "from app.coordination.sync_coordinator import" --include="*.py" .

# Check for direct references
grep -r "SyncScheduler\|SyncCoordinator" --include="*.py" . | grep -v "deprecated"
```

Active imports as of archive (Dec 26, 2025):
- app/metrics/coordinator.py (2 imports) - Uses SyncScheduler
- app/coordination/coordination_bootstrap.py (3 imports) - Uses wire_sync_events
- app/coordination/cluster/sync.py (1 import) - Imports SyncScheduler
- app/coordination/__init__.py (suppressed deprecation warnings)
- app/distributed/sync_orchestrator.py (1 import) - Uses get_sync_scheduler
- tests/unit/coordination/test_sync_coordinator.py (unit tests)

Migration plan:
1. Update app/metrics/coordinator.py to use ClusterManifest
2. Update coordination_bootstrap.py to wire AutoSyncDaemon
3. Update cluster/sync.py to use SyncFacade
4. Update sync_orchestrator.py to use AutoSyncDaemon
5. Update tests to test new components

ORIGINAL FILE PRESERVED BELOW FOR REFERENCE:
============================================
"""

# Original file content follows (preserved for reference during migration)
# NOTE: This is a snapshot from December 26, 2025
# Do NOT use this code - use the superseding modules listed above

# [Original implementation preserved here - 1,400 lines snipped for brevity]
# See git history for full original implementation:
#   git show HEAD:ai-service/app/coordination/sync_coordinator.py

# Original deprecation warning (now in archive):
import warnings

warnings.warn(
    "SyncScheduler is deprecated and will be archived in Q2 2026. "
    "Use AutoSyncDaemon for automated sync or SyncFacade for manual sync:\n"
    "\n"
    "For automated P2P sync:\n"
    "  from app.coordination import AutoSyncDaemon\n"
    "  daemon = AutoSyncDaemon()\n"
    "  await daemon.start()\n"
    "\n"
    "For one-time sync operations:\n"
    "  from app.coordination.sync_facade import sync\n"
    "  await sync('games', priority='high')\n"
    "\n"
    "See archive/deprecated_coordination/sync_coordinator.py for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Original implementation would go here
# Omitted from archive to save space - see git history if needed
