# Circular Dependencies in Coordination Modules

This document catalogs known circular dependency cycles in the ai-service coordination
infrastructure and the patterns used to resolve them.

## Quick Reference

| Cycle                                 | Pattern Used           | Location                       |
| ------------------------------------- | ---------------------- | ------------------------------ |
| daemon_manager ↔ event_router         | Lazy import + callback | `daemon_manager.py:114`        |
| selfplay_scheduler ↔ backpressure     | Protocol + injection   | `selfplay_scheduler.py:100`    |
| resource_optimizer ↔ resource_targets | Lazy accessor          | `resource_optimizer.py:61-103` |
| daemon_factory ↔ daemon_types         | Deferred loading       | `daemon_factory.py:61-71`      |
| contracts module                      | Zero-dependency        | `contracts.py:4`               |
| cross_process_events ↔ connections    | `_from_init` flag      | `cross_process_events.py:256`  |

---

## Pattern 1: Lazy Import Inside Functions

**When to use**: Module A needs Module B at runtime, but importing B at module scope
causes B to import A.

**Example** (`daemon_manager.py:114`):

```python
# At module scope: No import
def emit_daemon_event(event_type: str, daemon_type: str) -> None:
    """Emit daemon lifecycle event."""
    # Lazy import inside function - only loaded when needed
    from app.coordination.event_router import get_router
    router = get_router()
    router.publish(event_type, {"daemon": daemon_type})
```

**Files using this pattern**:

- `daemon_manager.py:114` - Event emission
- `coordinator_base.py:437` - Event router access
- `base_daemon.py:477` - Event router access
- `selfplay_scheduler.py:3337` - QueueType import

---

## Pattern 2: Protocol + Dependency Injection

**When to use**: Module A needs to use Module B's class, but B imports A for shared types.

**Example** (`selfplay_scheduler.py:100`):

```python
# In protocols.py or contracts.py - NO dependencies
class IBackpressureMonitor(Protocol):
    """Protocol for backpressure access without importing backpressure.py."""
    def get_spawn_rate_multiplier(self) -> float: ...
    def is_under_pressure(self) -> bool: ...

# In selfplay_scheduler.py
class SelfplayScheduler:
    def __init__(
        self,
        backpressure_monitor: IBackpressureMonitor | None = None,  # Injected
    ):
        self._backpressure = backpressure_monitor
```

**Files using this pattern**:

- `selfplay_scheduler.py:490-547` - Backpressure injection
- `interfaces/__init__.py:27` - Protocol definitions

---

## Pattern 3: Lazy Accessor Functions

**When to use**: Singleton access where the imported module also imports the accessor.

**Example** (`resource_optimizer.py:61-103`):

```python
# Module-level placeholder
_resource_targets: ResourceTargetManager | None = None

def get_resource_targets() -> ResourceTargetManager:
    """Lazy load resource targets to avoid circular import.

    Uses lazy import pattern (Dec 2025) to break circular dependency with
    resource_targets.py which imports ResourceOptimizer.
    """
    global _resource_targets
    if _resource_targets is None:
        # Import only when first accessed
        from app.coordination.resource_targets import get_resource_target_manager
        _resource_targets = get_resource_target_manager()
    return _resource_targets
```

**Files using this pattern**:

- `resource_optimizer.py:61-103` - ResourceTargetManager
- `daemon_factory.py:61-71` - DaemonType loading

---

## Pattern 4: Zero-Dependency Contracts Module

**When to use**: Shared types/interfaces needed by many modules that would otherwise
create N-way circular dependencies.

**Example** (`contracts.py:1-19`):

```python
"""Coordination contracts - ZERO dependency types.

NO dependencies on other coordination modules. This breaks circular import
chains by providing shared types that all modules can safely import.

Purpose: Break 8-cycle circular dependency chain in coordination modules
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

# All types defined here have NO imports from app.coordination.*
class DaemonState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
```

**Files using this pattern**:

- `contracts.py` - Core state enums, protocols
- `interfaces/__init__.py` - Protocol definitions

---

## Pattern 5: TYPE_CHECKING Guard

**When to use**: Type hints need a class from a cyclically-dependent module.

**Example**:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported during static analysis, not at runtime
    from app.coordination.daemon_manager import DaemonManager

class MyDaemon:
    def __init__(self, manager: "DaemonManager"):  # Quoted forward reference
        self._manager = manager
```

**Files using this pattern**:

- `job_scheduler.py:208` - Host config type
- Many coordination modules for type hints

---

## Pattern 6: `_from_init` Flag for Recursive Initialization

**When to use**: A class's `_ensure_db()` calls `_init_db()` which calls a method that
calls `_ensure_db()` again.

**Example** (`cross_process_events.py:256`):

```python
def _get_connection(self, _from_init: bool = False) -> sqlite3.Connection:
    """Get database connection.

    Args:
        _from_init: If True, skip _ensure_db() to break recursion.
    """
    if not _from_init:
        self._ensure_db()  # Normally ensure DB is ready
    return self._connection

def _init_db(self) -> None:
    """Initialize database."""
    # Pass _from_init=True to break circular dependency (December 2025 fix)
    conn = self._get_connection(_from_init=True)
    conn.execute("CREATE TABLE IF NOT EXISTS ...")
```

---

## Pattern 7: Callback Registration

**When to use**: Module A needs to call Module B, but B needs to call A during initialization.

**Example** (`daemon_types.py:754-795`):

```python
# In daemon_types.py - defines the callback registration
_mark_daemon_ready_callback: Callable[[str], None] | None = None

def register_mark_daemon_ready_callback(callback: Callable[[str], None]) -> None:
    """Register callback for mark_daemon_ready() to avoid circular import."""
    global _mark_daemon_ready_callback
    _mark_daemon_ready_callback = callback

# In daemon_manager.py - registers its method
from app.coordination.daemon_types import register_mark_daemon_ready_callback

class DaemonManager:
    def __init__(self):
        # Register callback for mark_daemon_ready() to break circular dependency
        register_mark_daemon_ready_callback(self._mark_daemon_ready)
```

---

## Debugging Circular Imports

### Symptoms

1. `ImportError: cannot import name 'X' from partially initialized module`
2. `AttributeError: module 'app.coordination.X' has no attribute 'Y'`
3. Methods/classes mysteriously `None` at runtime

### Diagnosis

```python
# Add to module to trace import order
import sys
print(f"Importing {__name__}, already loaded: {list(sys.modules.keys())[-10:]}")
```

### Prevention

1. **New modules**: Import from `contracts.py` or `protocols.py` first
2. **Avoid top-level imports** of heavyweight modules in `__init__.py`
3. **Use TYPE_CHECKING** for type hints when possible
4. **Test imports**: Run `python -c "from app.coordination.X import Y"` before committing

---

## Adding New Circular Dependency Workarounds

When adding a new workaround:

1. Document it in this file with the pattern number
2. Add a comment at the workaround location pointing here:
   ```python
   # Lazy import to break circular dep - see docs/coordination/CIRCULAR_DEPENDENCIES.md
   ```
3. Consider if the dependency should be broken differently (e.g., moving code)

---

## Known Cycles (December 2025)

| From                 | To                 | Via           | Resolution         |
| -------------------- | ------------------ | ------------- | ------------------ |
| daemon_manager       | event_router       | emit events   | Lazy import        |
| event_router         | daemon_manager     | health checks | TYPE_CHECKING      |
| selfplay_scheduler   | backpressure       | spawn rate    | Protocol injection |
| backpressure         | selfplay_scheduler | queue depth   | Protocol           |
| resource_optimizer   | resource_targets   | get targets   | Lazy accessor      |
| resource_targets     | resource_optimizer | optimization  | TYPE_CHECKING      |
| cross_process_events | \_get_connection   | \_ensure_db   | `_from_init` flag  |

---

_Last updated: December 29, 2025_
