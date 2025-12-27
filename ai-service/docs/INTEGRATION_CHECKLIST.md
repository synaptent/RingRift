# Integration Checklist for New Features

**Date:** December 27, 2025
**Purpose:** Ensure new features are properly integrated with the coordination infrastructure.

---

## Quick Reference

Before merging any new feature, verify:

- [ ] Event emissions wired
- [ ] Event subscriptions registered
- [ ] Daemon factory added (if new daemon)
- [ ] Health check implemented
- [ ] Tests added
- [ ] Documentation updated

---

## 1. Event Integration

### Adding a New Event Type

**Step 1:** Add to DataEventType enum

```python
# app/coordination/data_events.py
class DataEventType(Enum):
    # ... existing events ...
    MY_NEW_EVENT = "my_new_event"
```

**Step 2:** Add typed emitter function

```python
# app/coordination/event_emitters.py
async def emit_my_new_event(
    config_key: str,
    some_data: str,
    **kwargs,
) -> bool:
    """Emit MY_NEW_EVENT.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        some_data: Description of data

    Returns:
        True if event was published successfully
    """
    return await _emit_data_event(
        DataEventType.MY_NEW_EVENT,
        {
            "config_key": config_key,
            "some_data": some_data,
            **kwargs,
        }
    )
```

**Step 3:** Update `__all__` in event_emitters.py

```python
__all__ = [
    # ... existing exports ...
    "emit_my_new_event",
]
```

**Step 4:** Document in EVENT_REFERENCE.md

### Adding a Subscriber

**Step 1:** Subscribe in constructor/start method

```python
class MyHandler:
    def __init__(self):
        self._subscribed = False

    async def start(self):
        if not self._subscribed:
            router = get_router()
            router.subscribe(DataEventType.SOME_EVENT, self._on_some_event)
            self._subscribed = True

    async def stop(self):
        if self._subscribed:
            router = get_router()
            router.unsubscribe(DataEventType.SOME_EVENT, self._on_some_event)
            self._subscribed = False
```

**Step 2:** Implement handler method

```python
async def _on_some_event(self, event: dict) -> None:
    """Handle SOME_EVENT.

    Args:
        event: Event payload with keys: config_key, some_data, timestamp
    """
    config_key = event.get("config_key", "")
    some_data = event.get("some_data")

    try:
        await self._process_event(config_key, some_data)
    except Exception as e:
        logger.error(f"Error handling SOME_EVENT: {e}")
```

### Verification

```bash
# Check event is emitted
grep -rn "emit_my_new_event" app/

# Check event has subscribers
grep -rn "MY_NEW_EVENT" app/coordination/

# Run event integration tests
pytest tests/integration/coordination/test_event_flow.py -v -k "my_new_event"
```

---

## 2. Daemon Integration

### Adding a New Daemon

**Step 1:** Create daemon class

```python
# app/coordination/my_daemon.py
from app.coordination.handler_base import HandlerBase

class MyDaemon(HandlerBase):
    """My new daemon description.

    Subscribes to: EVENT_A, EVENT_B
    Emits: EVENT_C
    """

    _instance: Optional["MyDaemon"] = None

    def __init__(self):
        super().__init__(name="MyDaemon", cycle_interval=60.0)

    @classmethod
    def get_instance(cls) -> "MyDaemon":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _run_cycle(self) -> None:
        """Main daemon loop body."""
        pass

    def _get_event_subscriptions(self) -> dict:
        """Return events to subscribe to."""
        return {
            DataEventType.EVENT_A: self._on_event_a,
            DataEventType.EVENT_B: self._on_event_b,
        }

    def health_check(self) -> HealthCheckResult:
        """Return daemon health status."""
        return HealthCheckResult(
            healthy=self._running,
            status=CoordinatorStatus.RUNNING if self._running else CoordinatorStatus.STOPPED,
            message="Healthy" if self._running else "Stopped",
            details={
                "cycles": self._cycles_completed,
                "errors": self._errors_count,
            }
        )
```

**Step 2:** Add to DaemonType enum

```python
# app/coordination/daemon_types.py
class DaemonType(Enum):
    # ... existing types ...
    MY_DAEMON = "my_daemon"
```

**Step 3:** Add to daemon registry

```python
# app/coordination/daemon_registry.py
DAEMON_REGISTRY = {
    # ... existing entries ...
    DaemonType.MY_DAEMON: DaemonSpec(
        runner_name="create_my_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
        auto_restart=True,
        health_check_interval=30.0,
    ),
}
```

**Step 4:** Add factory function

```python
# app/coordination/daemon_runners.py
async def create_my_daemon() -> None:
    """Create and run MyDaemon."""
    try:
        from app.coordination.my_daemon import MyDaemon

        daemon = MyDaemon.get_instance()
        await daemon.start()
        while daemon._running:
            await asyncio.sleep(1)
    except ImportError as e:
        logger.error(f"MyDaemon not available: {e}")
        raise
```

**Step 5:** Register in daemon_runners.py RUNNER_REGISTRY

```python
RUNNER_REGISTRY = {
    # ... existing entries ...
    DaemonType.MY_DAEMON: create_my_daemon,
}
```

### Verification

```bash
# Verify daemon type exists
python -c "from app.coordination.daemon_types import DaemonType; print(DaemonType.MY_DAEMON)"

# Verify registry entry
python -c "
from app.coordination.daemon_registry import DAEMON_REGISTRY
from app.coordination.daemon_types import DaemonType
print(DAEMON_REGISTRY[DaemonType.MY_DAEMON])
"

# Verify factory works
python -c "
from app.coordination.daemon_runners import get_runner
from app.coordination.daemon_types import DaemonType
print(get_runner(DaemonType.MY_DAEMON))
"

# Run daemon tests
pytest tests/unit/coordination/test_my_daemon.py -v
```

---

## 3. Health Check Integration

### Implementing health_check()

Every coordinator/daemon should implement:

```python
from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

def health_check(self) -> HealthCheckResult:
    """Return health status for DaemonManager integration."""
    error_rate = self._errors_count / max(self._cycles_completed, 1)

    if error_rate > 0.5:
        status = CoordinatorStatus.ERROR
        healthy = False
        message = f"High error rate: {error_rate:.1%}"
    elif error_rate > 0.1:
        status = CoordinatorStatus.DEGRADED
        healthy = True
        message = f"Elevated errors: {error_rate:.1%}"
    else:
        status = CoordinatorStatus.RUNNING
        healthy = True
        message = "Healthy"

    return HealthCheckResult(
        healthy=healthy,
        status=status,
        message=message,
        details={
            "cycles": self._cycles_completed,
            "errors": self._errors_count,
            "error_rate": error_rate,
            "uptime_seconds": time.time() - self._start_time,
        }
    )
```

### Using HealthCheckMixin

For simpler cases, use the mixin:

```python
from app.coordination.mixins.health_check_mixin import HealthCheckMixin

class MyDaemon(HealthCheckMixin):
    """Daemon with automatic health_check() implementation."""

    # Override thresholds if needed
    UNHEALTHY_THRESHOLD = 0.3  # 30% error rate = unhealthy
    DEGRADED_THRESHOLD = 0.05  # 5% error rate = degraded

    def __init__(self):
        self._running = True
        self._cycles_completed = 0
        self._errors_count = 0
        self._start_time = time.time()
```

---

## 4. Testing Requirements

### Unit Tests (Required)

```python
# tests/unit/coordination/test_my_daemon.py

class TestMyDaemon:
    """Tests for MyDaemon."""

    def test_creation(self):
        """Test daemon instantiation."""
        daemon = MyDaemon()
        assert daemon is not None

    def test_singleton(self):
        """Test singleton pattern."""
        d1 = MyDaemon.get_instance()
        d2 = MyDaemon.get_instance()
        assert d1 is d2

    def test_health_check(self):
        """Test health check returns valid result."""
        daemon = MyDaemon()
        result = daemon.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test daemon lifecycle."""
        daemon = MyDaemon()
        await daemon.start()
        assert daemon._running is True
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test event subscription on start."""
        with patch("app.coordination.my_daemon.get_router") as mock_get:
            mock_router = MagicMock()
            mock_get.return_value = mock_router

            daemon = MyDaemon()
            await daemon.start()

            mock_router.subscribe.assert_called()
```

### Integration Tests (Recommended)

```python
# tests/integration/coordination/test_my_daemon_integration.py

@pytest.mark.asyncio
async def test_event_flow():
    """Test daemon responds to events correctly."""
    daemon = MyDaemon.get_instance()
    await daemon.start()

    try:
        # Emit event
        await emit_event_a(config_key="hex8_2p")

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify daemon processed event
        assert daemon.processed_count > 0
    finally:
        await daemon.stop()
```

---

## 5. Documentation Updates

### Files to Update

| Document                  | Update Required          |
| ------------------------- | ------------------------ |
| `docs/DAEMON_REGISTRY.md` | Add new daemon entry     |
| `docs/EVENT_REFERENCE.md` | Add new events           |
| `ai-service/CLAUDE.md`    | Add to relevant sections |
| Module docstring          | Describe purpose, events |

### Module Docstring Template

```python
"""My Daemon Module.

December 2025: Created for [purpose].

Events:
    Subscribes to:
        - EVENT_A: When [condition]
        - EVENT_B: When [condition]

    Emits:
        - EVENT_C: After [action]

Usage:
    from app.coordination.my_daemon import MyDaemon

    daemon = MyDaemon.get_instance()
    await daemon.start()

Configuration:
    Environment variables:
        - RINGRIFT_MY_SETTING: Description (default: value)

See Also:
    - docs/DAEMON_REGISTRY.md
    - docs/EVENT_REFERENCE.md
"""
```

---

## 6. Pre-Merge Checklist

Copy and complete before merging:

```markdown
## Feature: [Name]

### Event Integration

- [ ] New events added to DataEventType
- [ ] Typed emitters created in event_emitters.py
- [ ] Events documented in EVENT_REFERENCE.md
- [ ] Subscribers registered correctly

### Daemon Integration (if applicable)

- [ ] DaemonType enum entry added
- [ ] DAEMON_REGISTRY entry added
- [ ] Factory function in daemon_runners.py
- [ ] RUNNER_REGISTRY entry added
- [ ] Startup order verified (consumers before emitters)

### Health & Monitoring

- [ ] health_check() implemented
- [ ] Returns HealthCheckResult
- [ ] Includes relevant metrics

### Testing

- [ ] Unit tests added (>80% coverage)
- [ ] Integration tests added
- [ ] All tests pass locally

### Documentation

- [ ] Module docstring complete
- [ ] DAEMON_REGISTRY.md updated
- [ ] EVENT_REFERENCE.md updated
- [ ] CLAUDE.md updated (if major feature)

### Verification Commands Run

- [ ] `pytest tests/unit/coordination/test_*.py -v`
- [ ] `pytest tests/integration/coordination/ -v`
- [ ] `python -c "from app.coordination import MyDaemon"`
```

---

## Common Mistakes to Avoid

1. **Starting emitters before subscribers**
   - Always start event-consuming daemons first

2. **String literals instead of enum values**
   - Use `DataEventType.MY_EVENT.value`, not `"MY_EVENT"`

3. **Missing health_check()**
   - DaemonManager can't monitor daemon health

4. **Forgetting to unsubscribe on stop**
   - Can cause memory leaks and duplicate handlers

5. **Not handling callback errors**
   - Wrap callback logic in try/except to avoid breaking event loop

6. **Missing singleton reset for tests**
   - Add `reset_instance()` method for test isolation
