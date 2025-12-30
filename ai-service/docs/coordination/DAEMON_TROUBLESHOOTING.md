# Daemon Troubleshooting Guide

This runbook covers common daemon issues in the RingRift AI training pipeline
and their resolutions.

## Quick Diagnostics

```bash
# Check all daemon status
python scripts/launch_daemons.py --status

# View P2P cluster health
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} peers")
print(f"Is Leader: {d.get(\"is_leader\")}")
'

# Check daemon logs
tail -f logs/daemons.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

## Common Issues

### 1. Daemon Won't Start

**Symptoms:**

- Daemon exits immediately after starting
- "Dependencies not available" error
- Import errors

**Diagnosis:**

```bash
# Test daemon import
python -c "from app.coordination.daemon_manager import DaemonManager; print('OK')"

# Check specific daemon
python -c "from app.coordination.auto_sync_daemon import AutoSyncDaemon; print('OK')"
```

**Common Causes & Fixes:**

| Cause                    | Fix                                       |
| ------------------------ | ----------------------------------------- |
| Missing dependency       | `pip install -r requirements.txt`         |
| Wrong PYTHONPATH         | `export PYTHONPATH=.` from ai-service dir |
| Circular import          | Check import order in `__init__.py`       |
| EVENT_ROUTER not started | Start EVENT_ROUTER first (dependency)     |

**Resolution:**

```bash
# Start with dependencies in order
python scripts/launch_daemons.py --daemons EVENT_ROUTER
sleep 2
python scripts/launch_daemons.py --daemons AUTO_SYNC
```

### 2. Daemon Keeps Restarting

**Symptoms:**

- Daemon restarts every few minutes
- "Task failed" or "Task exception" in logs
- DAEMON_WATCHDOG keeps restarting daemons

**Diagnosis:**

```bash
# Check watchdog restarts
grep "Restarting daemon" logs/daemons.log

# Check for repeated failures
grep -c "failed" logs/daemons.log
```

**Common Causes:**

| Cause               | Symptoms           | Fix                              |
| ------------------- | ------------------ | -------------------------------- |
| Resource exhaustion | OOM errors         | Reduce batch sizes, check memory |
| Network failures    | SSH timeout errors | Check node connectivity          |
| Database locked     | SQLite lock errors | Kill stuck processes             |
| Infinite loop       | 100% CPU           | Add `asyncio.sleep()` in loops   |

**Resolution - Database Lock:**

```bash
# Find processes holding locks
lsof data/work_queue.db

# Kill stuck process
kill -9 <PID>

# Verify database integrity
sqlite3 data/work_queue.db "PRAGMA integrity_check"
```

### 3. Events Not Being Delivered

**Symptoms:**

- Subscribers not receiving events
- Events appear in router stats but not in handlers
- "Event loop is closed" errors

**Diagnosis:**

```bash
# Check event router stats
python -c "
from app.coordination.event_router import get_router
router = get_router()
print(router.get_stats())
"
```

**Common Causes:**

| Cause                       | Fix                              |
| --------------------------- | -------------------------------- |
| Router not started          | Call `await router.start()`      |
| Wrong event type            | Check `DataEventType` enum value |
| Handler exception           | Add try/except in handler        |
| Fire-and-forget task failed | Check error callback             |

**Resolution - Debug Event Flow:**

```python
# Add debug logging to event router
import logging
logging.getLogger("app.coordination.event_router").setLevel(logging.DEBUG)
```

### 4. P2P Leader Issues

**Symptoms:**

- Multiple nodes think they're leader
- Leader-only daemons running on wrong node
- "Not leader" skips in JobReaper

**Diagnosis:**

```bash
# Check leader on multiple nodes
for host in node1 node2 node3; do
  echo "=== $host ==="
  ssh $host "curl -s http://localhost:8770/status | jq '.leader_id, .is_leader'"
done
```

**Common Causes:**

| Cause             | Symptoms                        | Fix                          |
| ----------------- | ------------------------------- | ---------------------------- |
| Network partition | Different leaders per partition | Check Tailscale/network      |
| Stale state file  | Old leader ID persisted         | Delete `/tmp/p2p_state.json` |
| Clock skew        | Election timeouts wrong         | Sync NTP                     |
| Voter quorum lost | No leader elected               | Check voter nodes alive      |

**Resolution - Force Re-Election:**

```bash
# On all nodes, restart P2P with clean state
pkill -f p2p_orchestrator
rm /tmp/p2p_state.json
python scripts/p2p_orchestrator.py &
```

### 5. Sync Failures

**Symptoms:**

- "Sync failed" events
- Data not propagating to nodes
- Stale data on some nodes

**Diagnosis:**

```bash
# Check sync status
python -c "
from app.coordination.auto_sync_daemon import get_auto_sync_daemon
daemon = get_auto_sync_daemon()
print(daemon.get_status())
"

# Check sync history
sqlite3 data/sync_history.db "SELECT * FROM syncs ORDER BY timestamp DESC LIMIT 10"
```

**Common Causes:**

| Cause                  | Symptoms             | Fix                       |
| ---------------------- | -------------------- | ------------------------- |
| SSH key issues         | "Permission denied"  | Check ~/.ssh/id_cluster   |
| Bandwidth exhausted    | Slow/stuck transfers | Reduce parallel syncs     |
| Disk full              | "No space left"      | Clean old data            |
| rsync version mismatch | Protocol errors      | Update rsync on all nodes |

**Resolution - Force Manual Sync:**

```bash
# Sync specific database to node
rsync -avz --progress data/games/selfplay.db \
  user@target-node:~/ringrift/ai-service/data/games/
```

### 6. Work Queue Starvation

**Symptoms:**

- Nodes are idle but queue is empty
- QueuePopulator not adding items
- "Backpressure active" messages

**Diagnosis:**

```bash
# Check queue status
python -c "
from app.coordination.work_queue import get_work_queue
queue = get_work_queue()
print(queue.get_queue_status())
"

# Check populator status
python -c "
from app.coordination.unified_queue_populator import get_queue_populator
populator = get_queue_populator()
print(populator.get_status())
"
```

**Common Causes:**

| Cause                 | Symptoms              | Fix                       |
| --------------------- | --------------------- | ------------------------- |
| All targets met       | All Elo >= 2000       | Raise target Elo          |
| Backpressure          | Queue > 50 pending    | Wait for jobs to complete |
| No work queue set     | `_work_queue is None` | Set queue in populator    |
| Cluster health factor | Dead nodes detected   | Check P2P health          |

**Resolution:**

```bash
# Force populate queue
python -c "
from app.coordination.unified_queue_populator import get_queue_populator
populator = get_queue_populator()
populator.config.enabled = True
added = populator.populate()
print(f'Added {added} items')
"
```

### 7. Training Jobs Timeout

**Symptoms:**

- Jobs marked as TIMEOUT
- JobReaper killing processes
- Blacklisted nodes

**Diagnosis:**

```bash
# Check JobReaper stats
python -c "
from app.coordination.job_reaper import JobReaperDaemon
# Need to get running instance or check logs
"

# Check blacklisted nodes
grep "blacklisted" logs/daemons.log
```

**Common Causes:**

| Cause             | Symptoms                 | Fix                        |
| ----------------- | ------------------------ | -------------------------- |
| Timeout too short | Good jobs killed         | Increase timeout in config |
| GPU OOM           | Process killed by system | Reduce batch size          |
| Stuck process     | 100% CPU, no progress    | Check for infinite loops   |
| Network I/O       | Waiting for data         | Check data server          |

**Resolution - Adjust Timeouts:**

```python
# In job_reaper.py, update timeouts
self.job_timeouts = {
    "training": 21600,  # 6 hours instead of 4
    "gumbel_selfplay": 14400,  # 4 hours instead of 3
}
```

### 8. Memory Leaks in Long-Running Daemons

**Symptoms:**

- Memory usage grows over time
- OOM after hours/days of operation
- Daemon slows down over time

**Diagnosis:**

```bash
# Monitor memory usage
watch -n 10 'ps aux | grep -E "daemon|p2p" | awk "{print \$4, \$11}"'

# Check for leaked objects
python -c "
import gc
gc.collect()
print(f'Garbage: {len(gc.garbage)}')
"
```

**Common Causes:**

| Cause                 | Fix                                  |
| --------------------- | ------------------------------------ |
| Event history growing | Set `max_history` in router config   |
| Unbounded caches      | Add LRU eviction                     |
| Task references held  | Use `asyncio.create_task()` properly |
| Queued work IDs set   | Prune old entries periodically       |

**Resolution - Add Periodic Cleanup:**

```python
# Add to daemon loop
async def _cleanup_memory(self):
    """Periodic memory cleanup."""
    # Clear old entries
    self._old_entries.clear()
    # Force garbage collection
    import gc
    gc.collect()
```

## Daemon Health Checks

### Liveness Check

```bash
curl -s http://localhost:8765/health/live
# Expected: {"status": "ok"}
```

### Readiness Check

```bash
curl -s http://localhost:8765/health/ready
# Expected: {"status": "ready", "daemons": {...}}
```

## Emergency Recovery

### Stop All Daemons

```bash
python scripts/launch_daemons.py --stop-all
pkill -f "p2p_orchestrator"
pkill -f "daemon_manager"
```

### Clean Restart

```bash
# Kill everything
pkill -f ringrift
pkill -f python

# Clean state
rm /tmp/p2p_state.json
rm data/work_queue.db  # WARNING: loses pending work

# Fresh start
cd ai-service
python scripts/launch_daemons.py --all
```

### Recover Corrupted Database

```bash
# Backup first
cp data/work_queue.db data/work_queue.db.bak

# Check integrity
sqlite3 data/work_queue.db "PRAGMA integrity_check"

# If corrupted, recover
sqlite3 data/work_queue.db ".dump" > dump.sql
rm data/work_queue.db
sqlite3 data/work_queue.db < dump.sql
```

## Monitoring Best Practices

1. **Set up alerts** for:
   - Daemon restart count > 3 per hour
   - Queue depth = 0 for > 10 minutes
   - No new games for > 30 minutes
   - Leader changes > 3 per hour

2. **Log rotation**:

   ```bash
   # Add to crontab
   0 0 * * * find logs/ -name "*.log" -mtime +7 -delete
   ```

3. **Health dashboard**:
   ```bash
   # Quick status command
   alias rr-status='curl -s http://localhost:8770/status | python3 -m json.tool'
   ```

## Related Documentation

- [DAEMON_REGISTRY.md](../DAEMON_REGISTRY.md) - Canonical daemon catalog
- [EVENT_CATALOG.md](../EVENT_CATALOG.md) - Event types reference
- [../AI_SERVICE_ENVIRONMENT_REFERENCE.md](../AI_SERVICE_ENVIRONMENT_REFERENCE.md) - Environment variables
- [ALERTING_THRESHOLDS.md](../../../docs/operations/ALERTING_THRESHOLDS.md) - Alerting setup
