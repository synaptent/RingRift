# Coordination Module Deprecation Guide

**Last Updated:** December 2025
**Removal Target:** Q2 2026

This guide documents all deprecated modules in `app/coordination/` and their replacements.

## Quick Reference

| Deprecated Module                             | Replacement                                         | Status           |
| --------------------------------------------- | --------------------------------------------------- | ---------------- |
| `auto_evaluation_daemon.py`                   | `evaluation_daemon.py` + `auto_promotion_daemon.py` | Emits warning    |
| `replication_monitor.py`                      | `unified_replication_daemon.py`                     | Emits warning    |
| `replication_repair_daemon.py`                | `unified_replication_daemon.py`                     | Emits warning    |
| `cross_process_events.py`                     | `event_router.py`                                   | Archived         |
| `sync_coordinator.py` (class SyncCoordinator) | `SyncScheduler` (same file)                         | Alias exists     |
| `bandwidth_manager.py`                        | `resources/bandwidth.py`                            | Emits warning    |
| `system_health_monitor.py` (scoring)          | `unified_health_manager.py`                         | Removed Dec 2025 |
| `tracing.py`                                  | `core_utils.py`                                     | Emits warning    |
| `distributed_lock.py`                         | `core_utils.py`                                     | Emits warning    |
| `event_mappings.py`                           | `core_events.py`                                    | Emits warning    |
| `event_normalization.py`                      | `core_events.py`                                    | Emits warning    |

## Phase 5: Module Consolidation (December 2025)

Three new consolidated modules reduce the 157→15 module target:

### core_utils.py (Consolidated)

**Consolidates:** `tracing.py`, `distributed_lock.py`, `optional_imports.py`, `yaml_utils.py`

**Old:**

```python
from app.coordination.tracing import TraceContext, new_trace
from app.coordination.distributed_lock import DistributedLock
from app.utils.optional_imports import TORCH_AVAILABLE
from app.utils.yaml_utils import load_yaml
```

**New:**

```python
from app.coordination.core_utils import (
    # Tracing
    TraceContext, new_trace, span, get_trace_id, set_trace_id,
    # Locking
    DistributedLock, training_lock, acquire_training_lock,
    # Optional imports
    TORCH_AVAILABLE, CUDA_AVAILABLE, get_module,
    # YAML
    load_yaml, safe_load_yaml, dump_yaml,
)
```

### core_base.py (Consolidated)

**Consolidates:** `coordinator_base.py`, `coordinator_dependencies.py`

**Old:**

```python
from app.coordination.coordinator_base import CoordinatorBase, CoordinatorStatus
from app.coordination.coordinator_dependencies import get_initialization_order
```

**New:**

```python
from app.coordination.core_base import (
    # Base classes
    CoordinatorBase, CoordinatorStats,
    # Protocols and enums
    CoordinatorProtocol, CoordinatorStatus, HealthCheckResult,
    # Mixins
    SQLitePersistenceMixin, SingletonMixin, CallbackMixin,
    # Registry
    CoordinatorRegistry, get_coordinator_registry,
    # Dependencies
    CoordinatorDependencyGraph, get_initialization_order,
)
```

### core_events.py (Consolidated)

**Consolidates:** `event_router.py`, `event_mappings.py`, `event_emitters.py`, `event_normalization.py`

**Old:**

```python
from app.coordination.event_router import UnifiedEventRouter, get_router
from app.coordination.event_mappings import STAGE_TO_DATA_EVENT_MAP
from app.coordination.event_emitters import emit_training_complete
from app.coordination.event_normalization import normalize_event_type
```

**New:**

```python
from app.coordination.core_events import (
    # Router core
    UnifiedEventRouter, get_router, publish, subscribe,
    # Event types
    DataEventType, DataEvent, EventBus, StageEvent,
    # Mappings
    STAGE_TO_DATA_EVENT_MAP, DATA_TO_CROSS_PROCESS_MAP,
    # Typed emitters (70+)
    emit_training_complete, emit_selfplay_complete, emit_sync_complete,
    # Normalization
    normalize_event_type, CANONICAL_EVENT_NAMES,
)
```

## Detailed Migration

### 1. Auto-Evaluation Daemon

**Old:**

```python
from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon
daemon = AutoEvaluationDaemon()
await daemon.start()
```

**New:**

```python
from app.coordination.evaluation_daemon import EvaluationDaemon
from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

# Use separate daemons for better control
eval_daemon = EvaluationDaemon()
promo_daemon = AutoPromotionDaemon()
await eval_daemon.start()
await promo_daemon.start()
```

### 2. Replication Daemons

**Old:**

```python
from app.coordination.replication_monitor import ReplicationMonitor
from app.coordination.replication_repair_daemon import ReplicationRepairDaemon

monitor = ReplicationMonitor()
repair = ReplicationRepairDaemon()
```

**New:**

```python
from app.coordination.unified_replication_daemon import (
    UnifiedReplicationDaemon,
    create_replication_monitor,  # Backward-compat factory
    create_replication_repair_daemon,  # Backward-compat factory
)

# Single daemon handles both monitoring and repair
daemon = UnifiedReplicationDaemon()
await daemon.start()

# Or use factories for drop-in replacement
monitor = create_replication_monitor()
repair = create_replication_repair_daemon()
```

### 3. Cross-Process Events

**Old:**

```python
from app.coordination.cross_process_events import (
    CrossProcessEventQueue,
    publish,
    poll_events,
)
```

**New:**

```python
from app.coordination.event_router import (
    CrossProcessEventQueue,
    cp_publish as publish,
    cp_poll_events as poll_events,
    get_cross_process_queue,
)

# Or use the unified event system
from app.coordination.event_router import EventRouter, emit
router = EventRouter.get_instance()
await emit("EVENT_TYPE", {"data": "value"})
```

### 4. Sync Coordinator Alias

**Old:**

```python
from app.coordination.sync_coordinator import SyncCoordinator
coordinator = SyncCoordinator()
```

**New:**

```python
from app.coordination.sync_coordinator import SyncScheduler
scheduler = SyncScheduler()  # Same class, new canonical name

# Or use the helper functions
from app.coordination import get_sync_scheduler
scheduler = get_sync_scheduler()
```

### 5. Health Scoring

**Old:**

```python
from app.coordination.system_health_monitor import (
    calculate_system_health_score,
    get_system_health_level,
)
```

**New:**

```python
from app.coordination.unified_health_manager import (
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    SystemHealthLevel,
    SystemHealthScore,
)
```

### 6. Bandwidth Manager

**Old:**

```python
from app.coordination.bandwidth_manager import BandwidthManager
```

**New:**

```python
from app.coordination.resources.bandwidth import BandwidthManager
# Or via package
from app.coordination.resources import BandwidthManager
```

## Suppressing Deprecation Warnings

During migration, you can suppress warnings:

```python
import warnings

# Suppress all coordination deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"app\.coordination\..*"
)

# Or suppress specific module
warnings.filterwarnings(
    "ignore",
    message=r".*auto_evaluation_daemon.*"
)
```

## Timeline

| Phase     | Date         | Action                      |
| --------- | ------------ | --------------------------- |
| Warning   | Dec 2025     | Deprecation warnings active |
| Migration | Jan-Mar 2026 | Update all internal imports |
| Removal   | Q2 2026      | Deprecated modules archived |

## Archived Modules

Already moved to `app/coordination/deprecated/`:

- `_deprecated_cross_process_events.py` → `event_router.py`
- `_deprecated_event_emitters.py` → `event_router.emit()`
- `_deprecated_health_check_orchestrator.py` → `cluster.health`
- `_deprecated_host_health_policy.py` → `cluster.health`
- `_deprecated_system_health_monitor.py` → `cluster.health`

## Package Structure

The coordination module is being reorganized into focused packages:

```
app/coordination/
├── core/                    # Event system, tasks, pipeline
├── cluster/                 # Health, sync, transport, P2P
├── training/                # Training orchestration, scheduling
├── resources/               # Bandwidth, thresholds, optimization
└── deprecated/              # Archived modules with shims
```

See `app/coordination/deprecated/README.md` for the full package migration guide.

## Checking Your Code

Find deprecated imports in your code:

```bash
# Find auto_evaluation_daemon usage
grep -r "from app.coordination.auto_evaluation_daemon" .

# Find replication_monitor usage
grep -r "from app.coordination.replication_monitor" .

# Find all deprecated imports
grep -rE "from app\.coordination\.(auto_evaluation_daemon|replication_monitor|replication_repair_daemon|cross_process_events)" .
```

## Questions?

- See `app/coordination/deprecated/README.md` for package structure
- See `app/coordination/COORDINATOR_GUIDE.md` for usage patterns
- File issues at https://github.com/anthropics/ringrift/issues
