# Deprecated Coordination Modules

**December 2025**: This directory contains deprecated modules that have been consolidated.

## Archived Modules (December 26, 2025)

The following modules have been moved here from the parent directory:

| Module                                     | Replacement           | Notes                      |
| ------------------------------------------ | --------------------- | -------------------------- |
| `_deprecated_cross_process_events.py`      | `event_router.py`     | Cross-process event queue  |
| `_deprecated_event_emitters.py`            | `event_router.emit()` | Centralized event emission |
| `_deprecated_health_check_orchestrator.py` | `cluster.health`      | Health check orchestration |
| `_deprecated_host_health_policy.py`        | `cluster.health`      | Host health policies       |
| `_deprecated_system_health_monitor.py`     | `cluster.health`      | System health monitoring   |

## Archived Modules (December 27, 2025)

| Module                                  | Replacement                                   | Notes                      |
| --------------------------------------- | --------------------------------------------- | -------------------------- |
| `_deprecated_auto_evaluation_daemon.py` | `evaluation_daemon` + `auto_promotion_daemon` | Split into focused daemons |
| `_deprecated_sync_coordinator.py`       | `AutoSyncDaemon`                              | Unified sync scheduling    |
| `_deprecated_queue_populator_daemon.py` | `unified_queue_populator`                     | Consolidated queue manager |

Import from `app.coordination.deprecated.*` will emit deprecation warnings.

## Deprecated Re-export Modules (December 28, 2025)

The following modules are pure re-exports that now emit deprecation warnings.
They remain functional for backward compatibility but will be removed in Q2 2026.

| Module                                   | Canonical Imports                                          | Notes             |
| ---------------------------------------- | ---------------------------------------------------------- | ----------------- |
| `app.coordination.queue_populator`       | `unified_queue_populator`                                  | ~58 LOC re-export |
| `app.coordination.training.scheduler`    | `job_scheduler`, `duration_scheduler`, `unified_scheduler` | ~74 LOC re-export |
| `app.coordination.training.orchestrator` | `training_coordinator`, `selfplay_orchestrator`            | ~73 LOC re-export |
| `app.coordination.cluster.sync`          | `sync_coordinator`, `sync_bandwidth`, `sync_mutex`         | ~74 LOC re-export |
| `app.core.singleton_mixin`               | `app.coordination.singleton_mixin`                         | ~72 LOC re-export |

### Deprecation Warning Format

All deprecated modules emit a `DeprecationWarning` on import with:

- What to use instead
- When it will be removed (Q2 2026)

Example warning:

```
DeprecationWarning: app.coordination.queue_populator is deprecated.
Use app.coordination.unified_queue_populator instead.
This module will be removed in Q2 2026.
```

### Removed Modules (No Longer Exist)

The following modules have been fully removed and are now documented only in migration guides:

| Removed Module                           | Replacement                                | Removal Date  |
| ---------------------------------------- | ------------------------------------------ | ------------- |
| `app.coordination.cluster_data_sync`     | `auto_sync_daemon.py`                      | December 2025 |
| `app.coordination.ephemeral_sync`        | `auto_sync_daemon.py` (strategy=ephemeral) | December 2025 |
| `app.coordination.node_health_monitor`   | `health_check_orchestrator.py`             | December 2025 |
| `app.coordination.system_health_monitor` | `unified_health_manager.py`                | December 2025 |

See `archive/deprecated_coordination/README.md` for historical migration guides for fully removed modules.

### Migration Examples

```python
# Old (deprecated) - emits DeprecationWarning
from app.coordination.training.scheduler import PriorityJobScheduler

# New (canonical)
from app.coordination.job_scheduler import PriorityJobScheduler
```

```python
# Old (deprecated) - emits DeprecationWarning
from app.coordination.training.orchestrator import TrainingCoordinator

# New (canonical)
from app.coordination.training_coordinator import TrainingCoordinator
```

```python
# Old (deprecated) - emits DeprecationWarning
from app.coordination.cluster.sync import SyncScheduler

# New (canonical)
from app.coordination.sync_coordinator import SyncScheduler
```

## Consolidation Summary

The `app/coordination/` module was consolidated from 75 modules → 15 focused components organized into 4 packages:

### New Package Structure

```
app/coordination/
├── core/                    # 4 components
│   ├── events.py            # Unified event system
│   ├── tasks.py             # Task coordination
│   └── pipeline.py          # Training pipeline orchestration
├── cluster/                 # 4 components
│   ├── health.py            # Node and host health monitoring
│   ├── sync.py              # Data synchronization
│   ├── transport.py         # Cluster transport layer
│   └── p2p.py               # Peer-to-peer backend
├── training/                # 2 components
│   ├── orchestrator.py      # Training and selfplay orchestration
│   └── scheduler.py         # Job and duration scheduling
└── resources/               # 3 components
    ├── manager.py           # Resource optimization
    ├── bandwidth.py         # Bandwidth management
    └── thresholds.py        # Dynamic thresholds
```

## Migration Guide

### Health Modules

| Old Location                         | New Location     |
| ------------------------------------ | ---------------- |
| `unified_health_manager.py`          | `cluster.health` |
| `host_health_policy.py`              | `cluster.health` |
| `node_health_monitor.py`             | `cluster.health` |
| `resource_monitoring_coordinator.py` | `cluster.health` |

```python
# Old import
from app.coordination.unified_health_manager import UnifiedHealthManager

# New import
from app.coordination.cluster.health import UnifiedHealthManager
# or
from app.coordination.cluster import UnifiedHealthManager
```

### Sync Modules

| Old Location               | New Location   |
| -------------------------- | -------------- |
| `sync_coordinator.py`      | `cluster.sync` |
| `sync_bandwidth.py`        | `cluster.sync` |
| `sync_base.py`             | `cluster.sync` |
| `sync_mutex.py`            | `cluster.sync` |
| `transfer_verification.py` | `cluster.sync` |

```python
# Old import
from app.coordination.sync_coordinator import SyncScheduler

# New import
from app.coordination.cluster.sync import SyncScheduler
```

### Event Modules

| Old Location              | New Location  |
| ------------------------- | ------------- |
| `event_router.py`         | `core.events` |
| `event_emitters.py`       | `core.events` |
| `event_mappings.py`       | `core.events` |
| `cross_process_events.py` | `core.events` |
| `stage_events.py`         | `core.events` |

```python
# Old import
from app.coordination.event_router import EventRouter, emit

# New import
from app.coordination.core.events import EventRouter, emit
```

### Task Modules

| Old Location                    | New Location |
| ------------------------------- | ------------ |
| `task_coordinator.py`           | `core.tasks` |
| `task_lifecycle_coordinator.py` | `core.tasks` |
| `task_decorators.py`            | `core.tasks` |

### Scheduler Modules

| Old Location            | New Location         |
| ----------------------- | -------------------- |
| `job_scheduler.py`      | `training.scheduler` |
| `duration_scheduler.py` | `training.scheduler` |
| `work_distributor.py`   | `training.scheduler` |
| `unified_scheduler.py`  | `training.scheduler` |

### Resource Modules

| Old Location                   | New Location           |
| ------------------------------ | ---------------------- |
| `resource_optimizer.py`        | `resources.manager`    |
| `adaptive_resource_manager.py` | `resources.manager`    |
| `resource_targets.py`          | `resources.manager`    |
| `bandwidth_manager.py`         | `resources.bandwidth`  |
| `dynamic_thresholds.py`        | `resources.thresholds` |

## Backwards Compatibility

The original module files remain in place and continue to work. The new package structure provides:

1. Cleaner imports via consolidated packages
2. Better discoverability through grouped functionality
3. Reduced confusion from overlapping modules

Eventually, deprecation warnings will be added to the original modules directing users to the new locations.

## Modules Not Yet Consolidated

The following modules remain at the top level for now:

- `facade.py` - High-level coordination facade
- `coordinator_base.py` - Base class for coordinators
- `coordination_bootstrap.py` - Bootstrap utilities
- `helpers.py` - General helper functions
- `safeguards.py` - Safety checks and guards

These may be consolidated in a future cleanup pass.
