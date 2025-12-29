# New Daemon Checklist

This guide walks through creating a new daemon for the RingRift coordination infrastructure.

## Quick Reference

| Step | Required | Description                                                        |
| ---- | -------- | ------------------------------------------------------------------ |
| 1    | ✅       | Create daemon class inheriting from `HandlerBase` or `MonitorBase` |
| 2    | ✅       | Implement `health_check()` returning `HealthCheckResult`           |
| 3    | ✅       | Register in `DaemonType` enum                                      |
| 4    | ✅       | Add runner function to `daemon_runners.py`                         |
| 5    | ✅       | Add to `DAEMON_REGISTRY` in `daemon_registry.py`                   |
| 6    | ✅       | Add unit tests                                                     |
| 7    | ⚠️       | Wire event subscriptions (if event-driven)                         |
| 8    | ⚠️       | Add to daemon profile in `master_loop.py` (if auto-started)        |

---

## Step 1: Create Daemon Class

Use `HandlerBase` for event-driven daemons or `MonitorBase` for periodic monitoring:

```python
# app/coordination/my_daemon.py
"""My daemon that does X."""

import logging
from typing import Any, Callable

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


class MyDaemon(HandlerBase):
    """Daemon that monitors/handles X.

    Subscribes to: EVENT_A, EVENT_B
    Emits: EVENT_C
    """

    _instance: "MyDaemon | None" = None

    def __init__(self, cycle_interval: float = 60.0):
        super().__init__(name="my_daemon", cycle_interval=cycle_interval)
        self._errors = 0
        self._last_success = 0.0

    @classmethod
    def get_instance(cls) -> "MyDaemon":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        if cls._instance is not None:
            cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Define event subscriptions."""
        return {
            "EVENT_A": self._on_event_a,
            "EVENT_B": self._on_event_b,
        }

    async def _run_cycle(self) -> None:
        """Main periodic work loop."""
        try:
            # Do periodic work here
            self._last_success = time.time()
        except Exception as e:
            self._errors += 1
            logger.error(f"[MyDaemon] Cycle error: {e}")

    def _on_event_a(self, event: Any) -> None:
        """Handle EVENT_A."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            # Process event
        except Exception as e:
            self._errors += 1
            logger.warning(f"[MyDaemon] Error handling EVENT_A: {e}")

    def _on_event_b(self, event: Any) -> None:
        """Handle EVENT_B."""
        pass  # Implement

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager."""
        import time

        stale_threshold = self._cycle_interval * 3
        is_stale = (time.time() - self._last_success) > stale_threshold

        if self._errors > 10:
            return HealthCheckResult(
                healthy=False,
                message=f"Too many errors: {self._errors}",
                details={"errors": self._errors, "last_success": self._last_success}
            )

        if is_stale and self._last_success > 0:
            return HealthCheckResult(
                healthy=False,
                message="Daemon stale - no recent successful cycles",
                details={"last_success": self._last_success}
            )

        return HealthCheckResult(
            healthy=True,
            message="OK",
            details={"errors": self._errors, "last_success": self._last_success}
        )


# Singleton accessor (optional convenience)
def get_my_daemon() -> MyDaemon:
    """Get the singleton MyDaemon instance."""
    return MyDaemon.get_instance()
```

---

## Step 2: Implement health_check()

**Required interface:**

```python
from app.coordination.protocols import HealthCheckResult

def health_check(self) -> HealthCheckResult:
    """Return health status.

    Returns:
        HealthCheckResult with:
        - healthy: bool - True if daemon is functioning
        - message: str - Human-readable status
        - details: dict - Metrics for debugging
    """
```

**Key metrics to include:**

| Metric         | Purpose                                     |
| -------------- | ------------------------------------------- |
| `errors`       | Count of errors since start                 |
| `last_success` | Unix timestamp of last successful operation |
| `cycles`       | Total cycles completed                      |
| `queue_depth`  | Pending work items (if applicable)          |
| `uptime`       | Seconds since daemon started                |

---

## Step 3: Register in DaemonType Enum

Add to `app/coordination/daemon_types.py`:

```python
class DaemonType(str, Enum):
    # ... existing entries ...

    # Add yours (alphabetically within category)
    MY_DAEMON = "my_daemon"
```

---

## Step 4: Add Runner Function

Add to `app/coordination/daemon_runners.py`:

```python
async def create_my_daemon() -> None:
    """Create and run MyDaemon."""
    from app.coordination.my_daemon import get_my_daemon

    daemon = get_my_daemon()
    await daemon.start()

    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        await daemon.stop()
```

---

## Step 5: Add to DAEMON_REGISTRY

Add to `app/coordination/daemon_registry.py`:

```python
DAEMON_REGISTRY: dict[DaemonType, DaemonSpec] = {
    # ... existing entries ...

    DaemonType.MY_DAEMON: DaemonSpec(
        runner_name="create_my_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),  # Dependencies that must start first
        category="misc",  # sync, event, health, pipeline, resource, misc
        auto_restart=True,
        max_restarts=5,
        health_check_interval=30.0,
    ),
}
```

---

## Step 6: Add Unit Tests

Create `tests/unit/coordination/test_my_daemon.py`:

```python
"""Unit tests for MyDaemon."""

import pytest
from unittest.mock import MagicMock, patch

from app.coordination.my_daemon import MyDaemon, get_my_daemon


class TestMyDaemon:
    """Tests for MyDaemon class."""

    def setup_method(self):
        """Reset singleton before each test."""
        MyDaemon.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        MyDaemon.reset_instance()

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        d1 = get_my_daemon()
        d2 = get_my_daemon()
        assert d1 is d2

    def test_health_check_healthy(self):
        """Test health check returns healthy when OK."""
        daemon = get_my_daemon()
        result = daemon.health_check()
        assert result.healthy is True
        assert "OK" in result.message

    def test_health_check_unhealthy_on_errors(self):
        """Test health check fails after too many errors."""
        daemon = get_my_daemon()
        daemon._errors = 15
        result = daemon.health_check()
        assert result.healthy is False
        assert "errors" in result.message.lower()

    def test_event_subscriptions(self):
        """Test event subscriptions are defined."""
        daemon = get_my_daemon()
        subs = daemon._get_event_subscriptions()
        assert "EVENT_A" in subs
        assert "EVENT_B" in subs

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """Test start/stop lifecycle."""
        daemon = get_my_daemon()
        await daemon.start()
        assert daemon._running is True
        await daemon.stop()
        assert daemon._running is False
```

**Run tests:**

```bash
pytest tests/unit/coordination/test_my_daemon.py -v
```

---

## Step 7: Wire Event Subscriptions (If Event-Driven)

If your daemon needs to subscribe to events at startup:

1. Use `_get_event_subscriptions()` method (called by HandlerBase)
2. Or manually subscribe in `start()`:

```python
async def start(self) -> None:
    from app.coordination.event_router import get_event_bus, DataEventType

    bus = get_event_bus()
    bus.subscribe(DataEventType.MY_EVENT, self._on_my_event)

    await super().start()
```

---

## Step 8: Add to Daemon Profile (If Auto-Started)

If daemon should start with `master_loop.py`, add to a profile in `scripts/master_loop.py`:

```python
DAEMON_PROFILES = {
    "standard": [
        # ... existing daemons ...
        DaemonType.MY_DAEMON,
    ],
}
```

---

## Common Patterns

### Pattern 1: Fire-and-Forget Event Emission

```python
from app.coordination.event_emitters import emit_my_event

def _on_work_complete(self, result: dict) -> None:
    emit_my_event(
        config_key=result["config_key"],
        status="complete",
    )
```

### Pattern 2: Request-Response with Timeout

```python
import asyncio

async def _request_with_timeout(self, node: str) -> dict | None:
    try:
        return await asyncio.wait_for(
            self._send_request(node),
            timeout=TransportDefaults.HTTP_TIMEOUT
        )
    except asyncio.TimeoutError:
        self._errors += 1
        return None
```

### Pattern 3: Graceful Shutdown

```python
async def stop(self) -> None:
    """Stop daemon with cleanup."""
    self._stopping = True

    # Wait for in-progress work
    if self._current_task:
        try:
            await asyncio.wait_for(self._current_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._current_task.cancel()

    await super().stop()
```

---

## Troubleshooting

### Daemon Not Starting

1. Check DaemonType is registered
2. Check runner function exists in daemon_runners.py
3. Check DAEMON_REGISTRY entry
4. Check for import errors: `python -c "from app.coordination.my_daemon import MyDaemon"`

### Events Not Received

1. Verify event subscriptions in `_get_event_subscriptions()`
2. Check EVENT_ROUTER daemon is running
3. Verify event name matches DataEventType enum value

### Health Check Failing

1. Check `_last_success` is being updated
2. Verify error count isn't accumulating
3. Check cycle interval vs stale threshold

---

## See Also

- `app/coordination/handler_base.py` - Base class for event-driven daemons
- `app/coordination/monitor_base.py` - Base class for monitoring daemons
- `app/coordination/daemon_manager.py` - Daemon lifecycle management
- `docs/EVENT_SYSTEM_REFERENCE.md` - Complete event documentation
