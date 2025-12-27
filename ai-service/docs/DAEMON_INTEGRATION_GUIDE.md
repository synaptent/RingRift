# Daemon Integration Guide

This guide explains how to add new daemon types to the RingRift coordination layer.

## Overview

The daemon management system provides lifecycle management for background services across the cluster. Key components:

- **DaemonManager** (`app/coordination/daemon_manager.py`) - Singleton managing all daemon lifecycles
- **DaemonLifecycleManager** (`app/coordination/daemon_lifecycle.py`) - Core start/stop/restart logic
- **DaemonType** (`app/coordination/daemon_types.py`) - Enum of all daemon types
- **DaemonInfo** (`app/coordination/daemon_types.py`) - State tracking per daemon

## Adding a New Daemon Type

### Step 1: Define the DaemonType

Add your new daemon to the `DaemonType` enum in `app/coordination/daemon_types.py`:

```python
class DaemonType(str, Enum):
    # ... existing types ...

    # Your new daemon (add with descriptive comment)
    MY_NEW_DAEMON = "my_new_daemon"  # Purpose: Brief description
```

### Step 2: Create the Factory Method

Add a factory method in `DaemonManager` that creates and runs your daemon:

```python
async def _create_my_new_daemon(self) -> None:
    """Create and run my new daemon.

    Purpose: Describe what this daemon does and why it exists.

    Dependencies: List any daemons this depends on.

    Events emitted:
    - SOME_EVENT_TYPE when X happens
    - OTHER_EVENT when Y completes
    """
    try:
        from app.my_module import MyDaemon

        daemon = MyDaemon()
        await daemon.start()

        # Keep daemon alive
        while daemon.is_running():
            await asyncio.sleep(10)

    except ImportError as e:
        logger.error(f"MyDaemon not available: {e}")
        raise  # Propagate to mark daemon as FAILED
```

### Step 3: Register the Factory

Register your factory in `DaemonManager._register_default_factories()`:

```python
def _register_default_factories(self):
    # ... existing registrations ...

    # My new daemon (December 2025) - brief purpose description
    self.register_factory(
        DaemonType.MY_NEW_DAEMON,
        self._create_my_new_daemon,
        depends_on=[DaemonType.EVENT_ROUTER],  # List dependencies
        auto_restart=True,  # Restart on crash?
        max_restarts=5,  # How many restarts before giving up
    )
```

### Step 4: Add to Startup Profile (Optional)

If your daemon should start automatically, add it to the appropriate profile in `DAEMON_PROFILES`:

```python
DAEMON_PROFILES = {
    "full": [
        # ... existing daemons ...
        DaemonType.MY_NEW_DAEMON,
    ],
    "minimal": [
        # Only add if essential for minimal operation
    ],
}
```

## Daemon Lifecycle

### States

```
STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                    \-> FAILED (crash/error)
                    \-> IMPORT_FAILED (missing dependencies)
                    \-> RESTARTING (auto-restart in progress)
```

### Auto-Restart Behavior

When a daemon crashes with `auto_restart=True`:

1. State transitions to `RESTARTING`
2. Exponential backoff delay applied (min 1s, doubling each restart)
3. Factory called again
4. `restart_count` incremented
5. If `restart_count >= max_restarts`, state becomes `FAILED`
6. After 1 hour of stability, `restart_count` resets to 0

### Dependency Resolution

Daemons with `depends_on` are started after their dependencies:

```python
# EVENT_ROUTER will be started before MY_DAEMON
self.register_factory(
    DaemonType.MY_DAEMON,
    self._create_my_daemon,
    depends_on=[DaemonType.EVENT_ROUTER],
)
```

If a dependency fails, dependent daemons receive a cascade restart signal.

## Event Emission Requirements

Critical daemons should emit lifecycle events:

```python
from app.coordination.event_router import get_router, DataEventType

async def _create_my_daemon(self) -> None:
    router = get_router()

    # Emit start event
    await router.publish(DataEventType.DAEMON_STARTED, {
        "daemon_type": "my_new_daemon",
        "timestamp": time.time(),
    })

    try:
        # ... daemon logic ...
    except Exception as e:
        # Emit failure event
        await router.publish(DataEventType.DAEMON_FAILED, {
            "daemon_type": "my_new_daemon",
            "error": str(e),
        })
        raise
```

## Health Check Implementation

Daemons can implement health checks via the `health_check()` method:

```python
class MyDaemon:
    async def health_check(self) -> dict:
        """Return health status for monitoring."""
        return {
            "healthy": self.is_healthy(),
            "last_activity": self.last_activity_time,
            "queue_depth": len(self.pending_items),
            "error_count": self.error_count,
        }
```

The DaemonManager calls `health_check()` periodically (default: 30s).

## Testing Your Daemon

### Unit Tests

```python
# tests/unit/coordination/test_my_daemon.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_daemon_starts_successfully():
    from app.my_module import MyDaemon

    daemon = MyDaemon()
    await daemon.start()

    assert daemon.is_running()

    await daemon.stop()
    assert not daemon.is_running()
```

### Integration Tests

```python
# tests/integration/coordination/test_my_daemon_integration.py
@pytest.mark.asyncio
async def test_daemon_emits_events():
    from app.coordination.daemon_manager import DaemonManager, DaemonType

    manager = DaemonManager()
    events_received = []

    # Subscribe to events
    router = get_router()
    await router.subscribe(DataEventType.DAEMON_STARTED, events_received.append)

    # Start daemon
    await manager.start(DaemonType.MY_NEW_DAEMON)
    await asyncio.sleep(0.5)

    # Verify event emitted
    assert any(e.get("daemon_type") == "my_new_daemon" for e in events_received)
```

## Common Patterns

### Daemon with Background Loop

```python
async def _create_background_loop_daemon(self) -> None:
    """Daemon that processes items in a loop."""
    from app.my_module import ItemProcessor

    processor = ItemProcessor()

    while not self._shutdown_event.is_set():
        try:
            items = await processor.get_pending_items()
            for item in items:
                await processor.process(item)
            await asyncio.sleep(10)  # Polling interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Processing error: {e}")
            await asyncio.sleep(30)  # Backoff on error
```

### Daemon with Event Subscription

```python
async def _create_event_driven_daemon(self) -> None:
    """Daemon that reacts to events."""
    from app.coordination.event_router import get_router, DataEventType

    router = get_router()

    async def handle_event(event: dict):
        # Process the event
        await process_data(event)

    # Subscribe to relevant events
    await router.subscribe(DataEventType.SELFPLAY_COMPLETED, handle_event)
    await router.subscribe(DataEventType.TRAINING_COMPLETED, handle_event)

    # Keep daemon alive while subscriptions active
    while not self._shutdown_event.is_set():
        await asyncio.sleep(60)
```

## Debugging Daemons

### Check Daemon Status

```python
from app.coordination.daemon_manager import get_daemon_manager

manager = get_daemon_manager()
status = manager.get_status()

for daemon_type, info in status["daemons"].items():
    print(f"{daemon_type}: {info['state']} (restarts: {info['restart_count']})")
```

### View Daemon Logs

Daemons log to `logger = logging.getLogger(__name__)`. Filter logs by module:

```bash
grep "app.coordination.my_module" logs/daemon.log
```

### Force Restart Failed Daemon

```python
await manager.restart_failed_daemon(DaemonType.MY_DAEMON, force=True)
```

The `force=True` resets restart count and clears error state.

---

_Last updated: December 27, 2025_
