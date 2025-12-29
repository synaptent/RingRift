# Event Debugging Runbook

This runbook provides procedures for debugging issues with the RingRift event system.

**Last Updated**: December 2025

## Overview

The event system consists of three layers:

1. **EventBus** (`data_events.py`) - In-memory async pub/sub
2. **StageEventBus** (`stage_events.py`) - Pipeline stage events
3. **CrossProcessEventQueue** (`cross_process_events.py`) - SQLite-backed cross-process

The `UnifiedEventRouter` (`event_router.py`) bridges all three.

---

## Quick Diagnostics

### Check Event Router Status

```python
from app.coordination.event_router import get_router

router = get_router()
health = router.health_check()
print(f"Healthy: {health.healthy}")
print(f"Status: {health.status}")
print(f"Details: {health.details}")
```

### Check Subscription Count

```python
from app.coordination.event_router import get_router

router = get_router()
stats = router.get_stats()
print(f"Subscriptions: {stats.get('subscription_count')}")
print(f"Events published: {stats.get('events_published')}")
print(f"Events failed: {stats.get('events_failed')}")
```

### List All Subscribers

```python
from app.coordination.event_router import get_router

router = get_router()
for event_type, handlers in router._subscriptions.items():
    print(f"{event_type}: {len(handlers)} handlers")
```

---

## Common Issues

### 1. Events Not Being Delivered

**Symptoms**:

- Handler not receiving expected events
- Pipeline stages not triggering

**Diagnostic Steps**:

```bash
# Check if event is being published
grep "emit.*TRAINING_COMPLETED" logs/coordination.log

# Check if handler is subscribed
grep "subscribed to TRAINING_COMPLETED" logs/coordination.log
```

**Common Causes**:

| Cause                 | Fix                                               |
| --------------------- | ------------------------------------------------- |
| Handler not started   | Call `await handler.start()`                      |
| Wrong event type name | Use `DataEventType.TRAINING_COMPLETED` not string |
| Deduplication         | Check `_is_duplicate_event()` TTL (default 5 min) |
| Handler exception     | Check `get_recent_errors()` on handler            |

**Verification Script**:

```python
from app.coordination.event_router import get_router, DataEventType

router = get_router()

# Check if event type has subscribers
event_type = DataEventType.TRAINING_COMPLETED
subs = router._subscriptions.get(str(event_type.value), [])
print(f"Subscribers for {event_type}: {len(subs)}")

# Test publish
async def test_handler(event):
    print(f"Received: {event}")

unsub = router.subscribe(event_type, test_handler)
await router.publish(event_type, {"test": True}, source="debug")
unsub()
```

---

### 2. Event Handlers Timing Out

**Symptoms**:

- "Handler timeout" errors in logs
- Events pile up in queue

**Diagnostic Steps**:

```bash
# Check for timeout errors
grep "Handler.*timeout" logs/coordination.log

# Check handler execution times
grep "handler took" logs/coordination.log
```

**Common Causes**:

| Cause                         | Fix                        |
| ----------------------------- | -------------------------- |
| Slow database query           | Add indexing or cache      |
| Blocking I/O in async handler | Use `asyncio.to_thread()`  |
| Network call in handler       | Add timeout or retry logic |
| Handler raises exception      | Add try/except, check DLQ  |

**Fix Pattern**:

```python
# Bad: Blocking in async handler
async def _on_event(self, event):
    result = sync_database_query()  # Blocks event loop!

# Good: Run blocking code in thread
async def _on_event(self, event):
    result = await asyncio.to_thread(sync_database_query)
```

---

### 3. Cross-Process Events Not Delivered

**Symptoms**:

- Events work in-process but not across processes
- P2P events not reaching training scripts

**Diagnostic Steps**:

```bash
# Check cross-process queue
sqlite3 data/coordination/coordination.db "SELECT COUNT(*) FROM cross_process_events WHERE acked = 0"

# Check for polling errors
grep "CrossProcessPoller" logs/coordination.log
```

**Common Causes**:

| Cause                         | Fix                                 |
| ----------------------------- | ----------------------------------- |
| Event not in bridge map       | Add to `DATA_TO_CROSS_PROCESS_MAP`  |
| Poller not running            | Start `CROSS_PROCESS_POLLER` daemon |
| SQLite locked                 | Check for long transactions         |
| Event acked but not processed | Check handler subscription          |

**Check Bridge Mapping**:

```python
from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

# Check if event type is bridged
event_name = "training_completed"
is_bridged = event_name in DATA_TO_CROSS_PROCESS_MAP.values()
print(f"{event_name} is bridged: {is_bridged}")

# List all bridged events
print(f"Total bridged events: {len(DATA_TO_CROSS_PROCESS_MAP)}")
```

**Manual Event Inspection**:

```bash
# View pending cross-process events
sqlite3 data/coordination/coordination.db \
  "SELECT id, event_type, created_at, acked FROM cross_process_events ORDER BY created_at DESC LIMIT 10"
```

---

### 4. Duplicate Events

**Symptoms**:

- Same event processed multiple times
- Duplicate data/operations

**Diagnostic Steps**:

```python
# Check deduplication stats
handler = MyHandler.get_instance()
print(f"Events deduplicated: {handler.stats.events_deduplicated}")
```

**Common Causes**:

| Cause                | Fix                                   |
| -------------------- | ------------------------------------- |
| Missing dedup check  | Add `_is_duplicate_event()` call      |
| TTL too short        | Increase `DEDUP_TTL_SECONDS`          |
| Different key fields | Use consistent `key_fields` parameter |
| Event emitted twice  | Check emitter for redundant calls     |

**Fix Pattern**:

```python
async def _on_event(self, event):
    # Always check for duplicate first
    if self._is_duplicate_event(event, key_fields=["config_key", "model_path"]):
        return

    # Process event...
```

---

### 5. Dead Letter Queue (DLQ) Filling Up

**Symptoms**:

- DLQ retry daemon running frequently
- Failed events not recovering

**Diagnostic Steps**:

```bash
# Check DLQ size
sqlite3 data/coordination/coordination.db "SELECT COUNT(*) FROM dead_letter_queue"

# Check recent failures
sqlite3 data/coordination/coordination.db \
  "SELECT event_type, error, created_at FROM dead_letter_queue ORDER BY created_at DESC LIMIT 10"
```

**Common Causes**:

| Cause                        | Fix                                 |
| ---------------------------- | ----------------------------------- |
| Handler consistently failing | Fix handler bug, check logs         |
| Resource exhaustion          | Scale resources or add backpressure |
| Poison message               | Delete specific event from DLQ      |
| Retry not configured         | Ensure `DLQ_RETRY` daemon running   |

**Clear Specific Events**:

```bash
# Delete old failed events (> 7 days)
sqlite3 data/coordination/coordination.db \
  "DELETE FROM dead_letter_queue WHERE created_at < datetime('now', '-7 days')"
```

---

### 6. Event Loop Blocked

**Symptoms**:

- Events queue up
- System becomes unresponsive

**Diagnostic Steps**:

```python
import asyncio

# Check pending tasks
pending = asyncio.all_tasks()
print(f"Pending tasks: {len(pending)}")

# Check for long-running tasks
for task in pending:
    if not task.done():
        print(f"Running: {task.get_name()}")
```

**Common Causes**:

| Cause                        | Fix                         |
| ---------------------------- | --------------------------- |
| Sync code in async handler   | Use `asyncio.to_thread()`   |
| Infinite loop in handler     | Add timeout/break condition |
| Database deadlock            | Check SQLite locks          |
| Too many concurrent handlers | Add semaphore/rate limiting |

---

## Event Flow Tracing

### Enable Debug Logging

```python
import logging
logging.getLogger("app.coordination.event_router").setLevel(logging.DEBUG)
logging.getLogger("app.distributed.data_events").setLevel(logging.DEBUG)
```

### Trace Event Flow

```python
from app.coordination.event_router import get_router

router = get_router()

# Enable flow auditing
router._audit_enabled = True

# Later, check audit log
for entry in router._audit_log[-10:]:
    print(f"{entry['timestamp']}: {entry['event_type']} -> {entry['result']}")
```

### Add Tracing to Handler

```python
from app.core.tracing import traced, get_trace_id

class MyHandler(HandlerBase):
    @traced("my_handler.on_event")
    async def _on_event(self, event):
        trace_id = get_trace_id()
        logger.info(f"[{trace_id}] Processing event: {event.get('type')}")
        # Process...
```

---

## Event System Health Check

### Full System Check

```python
async def check_event_system_health():
    """Run full event system health check."""
    issues = []

    # 1. Check event router
    from app.coordination.event_router import get_router
    router = get_router()
    health = router.health_check()
    if not health.healthy:
        issues.append(f"Router unhealthy: {health.message}")

    # 2. Check subscription count
    stats = router.get_stats()
    if stats.get('subscription_count', 0) == 0:
        issues.append("No active subscriptions")

    # 3. Check cross-process queue
    try:
        from app.coordination.cross_process_events import get_event_queue
        queue = get_event_queue()
        pending = queue.count_pending()
        if pending > 1000:
            issues.append(f"Cross-process queue backlog: {pending}")
    except ImportError:
        pass

    # 4. Check DLQ
    try:
        import sqlite3
        conn = sqlite3.connect("data/coordination/coordination.db")
        dlq_count = conn.execute("SELECT COUNT(*) FROM dead_letter_queue").fetchone()[0]
        if dlq_count > 100:
            issues.append(f"DLQ has {dlq_count} events")
        conn.close()
    except Exception:
        pass

    return issues

# Run check
issues = await check_event_system_health()
if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Event system healthy")
```

---

## Recovery Procedures

### 1. Restart Event Router

```python
from app.coordination.event_router import get_router

router = get_router()
await router.shutdown()

# Clear singleton to force re-initialization
from app.coordination import event_router
event_router._router_instance = None

# Get fresh router
router = get_router()
```

### 2. Clear Cross-Process Queue

```bash
# Backup first
cp data/coordination/coordination.db data/coordination/coordination.db.bak

# Clear pending events
sqlite3 data/coordination/coordination.db "DELETE FROM cross_process_events WHERE acked = 0"
```

### 3. Resubscribe All Handlers

```python
from app.coordination.daemon_manager import get_daemon_manager

dm = get_daemon_manager()

# Stop and restart daemons
await dm.stop_all()
await dm.start_all()
```

### 4. Reset Event Deduplication

```python
# Clear dedup cache on specific handler
handler = MyHandler.get_instance()
handler._seen_events.clear()
handler.stats.events_deduplicated = 0
```

---

## Monitoring Queries

### Event Throughput

```sql
-- Events per minute in last hour
SELECT
    strftime('%Y-%m-%d %H:%M', created_at) as minute,
    COUNT(*) as count
FROM cross_process_events
WHERE created_at > datetime('now', '-1 hour')
GROUP BY minute
ORDER BY minute;
```

### Failed Events by Type

```sql
-- Top failing event types
SELECT event_type, COUNT(*) as failures
FROM dead_letter_queue
WHERE created_at > datetime('now', '-24 hours')
GROUP BY event_type
ORDER BY failures DESC
LIMIT 10;
```

### Queue Depth Over Time

```sql
-- Pending events by hour
SELECT
    strftime('%Y-%m-%d %H:00', created_at) as hour,
    COUNT(*) as pending
FROM cross_process_events
WHERE acked = 0
GROUP BY hour
ORDER BY hour;
```

---

## Event Types Quick Reference

### Critical Events (Must Be Monitored)

| Event                       | Purpose          | Alert If       |
| --------------------------- | ---------------- | -------------- |
| `DAEMON_PERMANENTLY_FAILED` | Daemon crash     | Any occurrence |
| `REGRESSION_CRITICAL`       | Model regression | Any occurrence |
| `SPLIT_BRAIN_DETECTED`      | Multiple leaders | Any occurrence |
| `DATA_SYNC_FAILED`          | Sync failure     | > 3 in 1 hour  |
| `TRAINING_FAILED`           | Training crash   | Any occurrence |

### Pipeline Events (Must Flow)

```
NEW_GAMES_AVAILABLE
    ↓
DATA_SYNC_COMPLETED
    ↓
NPZ_EXPORT_COMPLETE
    ↓
TRAINING_COMPLETED
    ↓
EVALUATION_COMPLETED
    ↓
MODEL_PROMOTED
    ↓
MODEL_DISTRIBUTION_COMPLETE
```

### Verify Pipeline Flow

```python
async def verify_pipeline_flow(config_key: str, hours: int = 24):
    """Verify all pipeline events fired for a config."""
    import sqlite3

    events_needed = [
        "new_games_available",
        "data_sync_completed",
        "training_completed",
        "evaluation_completed",
        "model_promoted",
    ]

    conn = sqlite3.connect("data/coordination/coordination.db")

    for event_type in events_needed:
        result = conn.execute(
            f"""
            SELECT COUNT(*) FROM cross_process_events
            WHERE event_type = ?
            AND payload LIKE ?
            AND created_at > datetime('now', '-{hours} hours')
            """,
            (event_type, f'%{config_key}%')
        ).fetchone()

        count = result[0]
        status = "✓" if count > 0 else "✗"
        print(f"{status} {event_type}: {count}")

    conn.close()
```

---

## See Also

- `docs/EVENT_SYSTEM_REFERENCE.md` - Complete event type reference
- `docs/DAEMON_REGISTRY_REFERENCE.md` - Daemon specifications
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Daemon troubleshooting
- `app/coordination/event_router.py` - Event router implementation
