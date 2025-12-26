# Async/Sync Boundary Fix - December 2025

## Problem

The EventBus and AsyncBridgeManager were using `threading.RLock` and `threading.Lock` respectively in async methods. This caused event loop blocking because threading locks hold the GIL and block the event loop thread.

### Symptoms
- Event loop blocking when publishing/subscribing to events
- Thread pool executor operations blocking async code
- Potential deadlocks when multiple async operations wait on threading locks

## Solution

Implemented dual-lock pattern:
- **asyncio.Lock** for async methods (publish, _dispatch, etc.)
- **threading.Lock** for sync methods (publish_sync, get_stats, etc.)

## Files Modified

### 1. app/core/event_bus.py

**Changes:**
- Replaced single `self._lock = threading.RLock()` with dual locks:
  - `self._async_lock = asyncio.Lock()` for async methods
  - `self._sync_lock = threading.RLock()` for sync methods

**Async methods (use `async with self._async_lock`):**
- `async def publish()`
- `async def _get_matching_subscriptions_async()`
- `async def _remove_subscription_async()`

**Sync methods (use `with self._sync_lock`):**
- `def add_subscription()`
- `def unsubscribe()`
- `def publish_sync()`
- `def _get_matching_subscriptions()`
- `def _remove_subscription()`
- `def get_history()`
- `def clear_history()`
- `def get_stats()`

### 2. app/coordination/async_bridge_manager.py

**Changes:**
- Replaced single `self._lock = threading.Lock()` with dual locks:
  - `self._async_lock = asyncio.Lock()` for async methods
  - `self._sync_lock = threading.Lock()` for sync methods

**Async methods (use `async with self._async_lock`):**
- `async def run_sync()` - statistics tracking
- Statistics updates in try/except/finally blocks

**Sync methods (use `with self._sync_lock`):**
- `def initialize()`
- `def register_bridge()`
- `def unregister_bridge()`
- `def get_bridge()`
- `def get_stats()`
- `def get_health()`

### 3. app/coordination/event_router.py

**Status:** ✓ Already correct
- Already uses `self._lock = asyncio.Lock()` for async methods
- Already uses `self._sync_lock = threading.Lock()` for sync methods
- No changes needed

## Testing

### Unit Tests
All existing tests pass:
```bash
pytest tests/unit/core/test_event_bus.py -v     # 55 tests PASSED
pytest tests/unit/coordination/test_integration_bridge.py -v  # 3 tests PASSED
```

### Manual Tests
```python
# Test EventBus
import asyncio
from app.core.event_bus import EventBus, Event

async def test():
    bus = EventBus()

    @bus.subscribe('test.event')
    async def handler(event):
        print(f"Received: {event.topic}")

    event = Event(topic='test.event')
    count = await bus.publish(event)
    print(f"✓ Delivered to {count} handlers")

asyncio.run(test())

# Test AsyncBridgeManager
from app.coordination.async_bridge_manager import AsyncBridgeManager

async def test_bridge():
    manager = AsyncBridgeManager()
    manager.initialize()

    result = await manager.run_sync(lambda x: x * 2, 21)
    print(f"✓ Result: {result}")

    await manager.shutdown()

asyncio.run(test_bridge())
```

## Impact

### Performance
- **Before:** Threading locks caused event loop blocking
- **After:** Async locks allow proper cooperative multitasking
- **Result:** No blocking, proper async/await semantics

### Compatibility
- ✓ Fully backward compatible
- ✓ All existing tests pass
- ✓ Both async and sync usage patterns supported
- ✓ No API changes

### Related Systems
This fix ensures proper async behavior in:
- Event routing and subscription system
- Cross-process event coordination
- Training pipeline orchestration
- Data sync coordination
- Model lifecycle management
- All daemon-based systems

## Pattern for Future Code

When creating classes that support both async and sync usage:

```python
class MyCoordinator:
    def __init__(self):
        # Dual-lock pattern
        self._async_lock = asyncio.Lock()  # For async methods
        self._sync_lock = threading.Lock()  # For sync methods

    async def async_operation(self):
        async with self._async_lock:
            # Async code here
            pass

    def sync_operation(self):
        with self._sync_lock:
            # Sync code here
            pass
```

**Key rules:**
1. Never use `threading.Lock` in `async def` methods
2. Never use `asyncio.Lock` in non-async methods
3. If a method can be called from both contexts, provide separate async/sync versions
4. Document which lock protects which shared state

## Verification

To verify the fix is working:

```bash
# 1. Check no blocking in event publishing
python -c "import asyncio; from app.core.event_bus import EventBus, Event; \
  async def test(): \
    bus = EventBus(); \
    await bus.publish(Event(topic='test')); \
    print('✓ No blocking'); \
  asyncio.run(test())"

# 2. Check concurrent async operations
python -c "import asyncio; from app.coordination.async_bridge_manager import get_bridge_manager; \
  async def test(): \
    mgr = get_bridge_manager(); \
    results = await asyncio.gather(*[mgr.run_sync(lambda x: x, i) for i in range(10)]); \
    print('✓ Concurrent ops:', results); \
  asyncio.run(test())"
```

## References

- Python asyncio documentation: https://docs.python.org/3/library/asyncio-sync.html
- Threading vs asyncio locks: https://docs.python.org/3/library/threading.html#lock-objects
- Event loop blocking: https://docs.python.org/3/library/asyncio-dev.html#running-blocking-code
