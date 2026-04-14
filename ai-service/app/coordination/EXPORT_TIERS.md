# Coordination Package Export Tiers

**Analysis Date**: April 14, 2026
**Current State**: 594 exports in `__all__`, 251 LOC in `__init__.py`

The root package is now a thin lazy facade backed by `_exports_core.py`,
`_exports_sync.py`, `_exports_daemon.py`, `_exports_events.py`,
`_exports_orchestrators.py`, and `_exports_utils.py`. Focused tests in
`tests/unit/coordination/test_package_exports.py` and
`tests/unit/coordination/test_import_hygiene.py` now lock that public surface
and keep runtime code off the package root unless the compatibility layer is
intentional.

## Usage Analysis

Most callers already use direct submodule imports rather than importing from `__init__.py`:

| Import Pattern                                    | Count | Recommended        |
| ------------------------------------------------- | ----- | ------------------ |
| `from app.coordination.daemon_runners import ...` | 6     | ✅ Yes             |
| `from app.coordination.event_emitters import ...` | 5     | ✅ Yes             |
| `from app.coordination.enums import ...`          | 4     | ✅ Yes             |
| `from app.coordination.stage_events import ...`   | 3     | ✅ Yes             |
| `from app.coordination import TaskType, ...`      | 2     | Consider submodule |

## Tier 1: Essential Exports (~30 items)

These should remain easily accessible from `__init__.py`:

### Core Infrastructure

```python
# Task coordination
TaskCoordinator, TaskType, can_spawn, get_coordinator

# Daemon management
DaemonManager, DaemonType, DaemonState, get_daemon_manager

# Event system (prefer core_events for new code)
core_events, core_utils

# Sync
SyncFacade, get_sync_facade, sync

# Health
get_health_orchestrator, get_system_health
```

### Lifecycle Functions

```python
initialize_all_coordinators
shutdown_all_coordinators
get_all_coordinator_status
```

### Common Helpers

```python
can_spawn_safe
should_throttle_production
acquire_orchestrator_role
```

## Tier 2: Commonly Used (~100 items)

Used by many callers but should import from submodule:

| Category | Submodule          | Example Imports                                |
| -------- | ------------------ | ---------------------------------------------- |
| Events   | `event_emitters`   | `emit_training_complete`, `emit_sync_complete` |
| Stage    | `stage_events`     | `StageEvent`, `get_stage_event_bus`            |
| Health   | `health_facade`    | `get_node_health`, `get_healthy_nodes`         |
| Daemons  | `daemon_runners`   | Runner functions                               |
| Queue    | `queue_monitor`    | `QueueType`, `check_backpressure`              |
| Resource | `resource_targets` | `get_host_targets`, `should_scale_up`          |

## Tier 3: Specialized (~400+ items)

These should ALWAYS be imported from their specific submodule:

- `tracing` - Distributed tracing utilities
- `transaction_isolation` - ACID-like merge operations
- `transfer_verification` - Checksum utilities
- `sync_bloom_filter` - P2P set membership
- `sync_durability` - WAL/DLQ utilities
- `ephemeral_data_guard` - Ephemeral host data protection
- `coordinator_persistence` - State snapshots
- `dynamic_thresholds` - Adaptive thresholds
- All `*_orchestrator` classes (import from their module)

## Recommended Import Patterns

### ✅ Good: Direct submodule import

```python
from app.coordination.daemon_runners import create_auto_sync
from app.coordination.event_emitters import emit_training_complete
from app.coordination.health_facade import get_node_health
from app.coordination.enums import DaemonType
```

### ✅ Good: Consolidated modules

```python
from app.coordination import core_events  # All event-related
from app.coordination import core_utils   # Tracing, locks, etc.
```

### ⚠️ Acceptable: Tier 1 from **init**

```python
from app.coordination import (
    TaskCoordinator,
    get_coordinator,
    initialize_all_coordinators,
)
```

### ❌ Avoid: Mass imports from **init**

```python
# Don't do this - imports 568 symbols
from app.coordination import *

# Don't do this - too many symbols
from app.coordination import (
    TaskCoordinator, TaskType, TaskInfo, TaskLimits, CoordinatedTask,
    OrchestratorRole, OrchestratorState, OrchestratorInfo, OrchestratorLock,
    # ... 50 more symbols
)
```

## Current Guardrails

- `app.coordination.__dir__()` exposes the intended root surface explicitly.
- `tests/unit/coordination/test_package_exports.py` ratchets the package root.
- `tests/unit/coordination/test_import_hygiene.py` keeps new runtime callers off
  the root facade.
- The root module stays small because it resolves exports lazily through the
  `_exports_*.py` shims instead of importing the whole coordination tree eagerly.

## Next Cleanup Opportunities

- Continue draining runtime callers toward focused submodules where that reduces
  dependency fan-out.
- Shrink historical compatibility exports only after callers have been removed
  and ratcheted.
- Keep rejecting broad root imports such as `from app.coordination import *`.

## Notes

The package root is no longer a bloated eager-import module; it is now a thin
lazy facade that still preserves a large historical compatibility surface.
Modern Python practice still favors explicit submodule imports for:

1. **Clarity**: Readers know exactly where a symbol comes from
2. **Performance**: Only load what you need
3. **IDE support**: Better autocomplete and navigation
4. **Maintainability**: Changes to submodules don't affect unrelated code
