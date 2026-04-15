# Deprecation Roadmap

**Last Updated:** December 27, 2025
**Target Removal:** Q2 2026 (April-June 2026)

This document tracks all deprecated modules, functions, and exports scheduled for removal. Each entry includes the deprecated item, its replacement, current status, and removal timeline.

---

## Summary

| Category                   | Deprecated Items | Target Removal             |
| -------------------------- | ---------------- | -------------------------- |
| Coordination Modules       | 8                | Q2 2026                    |
| Daemon Types               | 4                | Q2 2026                    |
| Sync Modules               | 4                | Q2 2026                    |
| Training Modules           | 3                | Q2 2026                    |
| Exports from `__init__.py` | 10               | Already removed (Dec 2025) |

---

## Q2 2026 Removals

### Coordination Modules

| Module                                       | Replacement                                                   | Status     | Notes                                                                |
| -------------------------------------------- | ------------------------------------------------------------- | ---------- | -------------------------------------------------------------------- |
| `app/coordination/sync_coordinator.py`       | `auto_sync_daemon.py` + `app/distributed/sync_coordinator.py` | DEPRECATED | Scheduling → AutoSyncDaemon, execution → distributed SyncCoordinator |
| `app/coordination/cluster_data_sync.py`      | `AutoSyncDaemon(strategy="broadcast")`                        | ARCHIVED   | Module removed; legacy daemon type only                              |
| `app/coordination/ephemeral_sync.py`         | `AutoSyncDaemon(strategy="ephemeral")`                        | ARCHIVED   | Module removed; legacy daemon type only                              |
| `app/coordination/system_health_monitor.py`  | `unified_health_manager.py`                                   | ARCHIVED   | Module removed; health facade aliases remain                         |
| `app/coordination/node_health_monitor.py`    | `health_check_orchestrator.py`                                | ARCHIVED   | Module removed; health facade aliases remain                         |
| `app/coordination/queue_populator.py`        | `unified_queue_populator.py`                                  | DEPRECATED | Re-export wrapper with deprecation warning                           |
| `app/coordination/resources/*`               | Direct imports from source modules                            | ARCHIVED   | Moved to archive Dec 2025                                            |
| `app/coordination/auto_evaluation_daemon.py` | `evaluation_daemon.py`                                        | ARCHIVED   | Moved to archive Dec 2025                                            |

### Daemon Types

| Deprecated                    | Replacement                       | Status     |
| ----------------------------- | --------------------------------- | ---------- |
| `DaemonType.SYNC_COORDINATOR` | `DaemonType.AUTO_SYNC`            | DEPRECATED |
| `DaemonType.HEALTH_CHECK`     | `DaemonType.NODE_HEALTH_MONITOR`  | DEPRECATED |
| `vast_idle_daemon.py`         | `unified_idle_shutdown_daemon.py` | ARCHIVED   |
| `lambda_idle_daemon.py`       | `unified_idle_shutdown_daemon.py` | ARCHIVED   |

### Sync Modules

| Module                                          | Replacement                                                                                | Status     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------- |
| `app/distributed/unified_data_sync.py`          | `SyncFacade` or `AutoSyncDaemon`                                                           | DEPRECATED |
| `app/coordination/replication_monitor.py`       | `unified_replication_daemon.py`                                                            | ARCHIVED   |
| `app/coordination/replication_repair_daemon.py` | `unified_replication_daemon.py`                                                            | ARCHIVED   |
| `app/coordination/model_distribution_daemon.py` | `app/coordination/unified_distribution_daemon.py` (`create_unified_distribution_daemon()`) | ARCHIVED   |
| `app/coordination/npz_distribution_daemon.py`   | `app/coordination/unified_distribution_daemon.py` (`create_unified_distribution_daemon()`) | ARCHIVED   |

### Training Modules

| Module                                                 | Replacement                                                        | Status     |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ---------- |
| `archive/deprecated_training/orchestrated_training.py` | `unified_orchestrator.py` + `app.training` compatibility re-export | ARCHIVED   |
| `integrated_enhancements.py`                           | `unified_orchestrator.py`                                          | DEPRECATED |
| `unified_ai_loop.py`                                   | `master_loop.py`                                                   | DEPRECATED |

### Deprecated Exports (Already Removed Dec 2025)

The following exports were removed from `app/coordination/__init__.py`:

| Export                        | Original Purpose                            |
| ----------------------------- | ------------------------------------------- |
| `CoordinationSyncCoordinator` | Alias for SyncScheduler                     |
| `LambdaIdleConfig`            | Alias for IdleShutdownConfig                |
| `LambdaIdleDaemon`            | Alias for UnifiedIdleShutdownDaemon         |
| `LambdaNodeStatus`            | Alias for NodeIdleStatus                    |
| `VastIdleConfig`              | Alias for IdleShutdownConfig                |
| `VastIdleDaemon`              | Alias for UnifiedIdleShutdownDaemon         |
| `VastNodeStatus`              | Alias for NodeIdleStatus                    |
| `ModelCacheEntry`             | Alias for CacheEntry                        |
| `get_sync_coordinator`        | Deprecated in favor of get_sync_scheduler   |
| `reset_sync_coordinator`      | Deprecated in favor of reset_sync_scheduler |

---

## Migration Guides

### Migrating from `cluster_data_sync.py`

**Before:**

```python
from app.coordination.cluster_data_sync import ClusterDataSync

sync = ClusterDataSync()
await sync.sync_to_all_nodes()
```

**After:**

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon(strategy="broadcast")
await daemon.start()
# Or use programmatically:
await daemon.sync_now()
```

### Migrating from `ephemeral_sync.py`

**Before:**

```python
from app.coordination.ephemeral_sync import EphemeralSync

sync = EphemeralSync()
await sync.sync_ephemeral_nodes()
```

**After:**

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon(strategy="ephemeral")
await daemon.start()
```

### Migrating from `system_health_monitor.py`

**Before:**

```python
from app.coordination.system_health_monitor import get_system_health_score
score = get_system_health_score()
```

**After:**

```python
from app.coordination.unified_health_manager import get_system_health_score
# Or use the health facade:
from app.coordination.health_facade import get_system_health_score

score = get_system_health_score()
```

### Migrating from `queue_populator.py`

**Before:**

```python
from app.coordination.queue_populator import QueuePopulator, PopulatorConfig
```

**After:**

```python
from app.coordination.unified_queue_populator import UnifiedQueuePopulator, QueuePopulatorConfig
```

---

## Deprecation Warning Implementation

All deprecated modules emit `DeprecationWarning` on import:

```python
import warnings
warnings.warn(
    "app.coordination.cluster_data_sync is deprecated. "
    "Use AutoSyncDaemon(strategy='broadcast') instead. "
    "Scheduled for removal in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)
```

To see deprecation warnings during development:

```bash
python -W default::DeprecationWarning your_script.py
```

---

## Archived Modules

The following modules have been moved to `archive/deprecated_coordination/`:

| Original Location                               | Archive Location                                            | Date Archived |
| ----------------------------------------------- | ----------------------------------------------------------- | ------------- |
| `queue_populator_daemon.py`                     | `_deprecated_queue_populator_daemon.py`                     | Dec 27, 2025  |
| `replication_monitor.py`                        | `_deprecated_replication_monitor.py`                        | Dec 27, 2025  |
| `replication_repair_daemon.py`                  | `_deprecated_replication_repair_daemon.py`                  | Dec 27, 2025  |
| `app/coordination/model_distribution_daemon.py` | `app/coordination/_deprecated_model_distribution_daemon.py` | Dec 27, 2025  |
| `app/coordination/npz_distribution_daemon.py`   | `app/coordination/_deprecated_npz_distribution_daemon.py`   | Dec 27, 2025  |
| `vast_idle_daemon.py`                           | `_deprecated_vast_idle_daemon.py`                           | Dec 26, 2025  |
| `lambda_idle_daemon.py`                         | `_deprecated_lambda_idle_daemon.py`                         | Dec 26, 2025  |
| `auto_evaluation_daemon.py`                     | `_deprecated_auto_evaluation_daemon.py`                     | Dec 27, 2025  |
| `resources/` (directory)                        | `_deprecated_resources/`                                    | Dec 27, 2025  |

---

## Pre-Removal Checklist (Q2 2026)

Before removing deprecated modules:

1. **Grep for imports** - Ensure no active code imports deprecated modules:

   ```bash
   grep -r "from app.coordination.cluster_data_sync" --include="*.py" | grep -v archive
   ```

2. **Run tests** - All tests must pass without deprecated modules:

   ```bash
   pytest tests/ -v
   ```

3. **Update CLAUDE.md** - Remove deprecated module documentation

4. **Update `__init__.py`** - Remove any remaining re-exports

5. **Delete archived files** - Remove from `archive/deprecated_coordination/`

---

## Contact

For questions about deprecations or migrations, see:

- `CLAUDE.md` - Main AI service documentation
- `archive/deprecated_coordination/README.md` - Archive documentation
- `CLUSTER_INTEGRATION_GUIDE.md` - Integration patterns
