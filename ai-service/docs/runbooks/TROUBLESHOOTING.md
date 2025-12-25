# Troubleshooting Runbook

Quick reference for diagnosing and resolving common issues in the RingRift AI Service.

## Table of Contents

1. [Node Unreachable](#node-unreachable)
2. [Training Job Stuck](#training-job-stuck)
3. [Circuit Breaker Open](#circuit-breaker-open)
4. [Model Promotion Failed](#model-promotion-failed)
5. [Database Locked](#database-locked)
6. [Task Timeout](#task-timeout)
7. [DLQ Events Accumulating](#dlq-events-accumulating)

---

## Node Unreachable

### Symptoms

- Health checks failing for a node
- Tasks not being assigned to node
- Node shows as "evicted" in cluster status

### Diagnosis

```bash
# Check node health status
python -c "
from app.coordination.node_health_monitor import get_node_health_monitor
monitor = get_node_health_monitor()
node = monitor.get_node_health('lambda-h100')
print(f'Status: {node.status}')
print(f'Consecutive failures: {node.consecutive_failures}')
print(f'Last error: {node.last_error}')
"

# Try direct health check
curl -s --connect-timeout 5 http://<node-ip>:8765/health
```

### Resolution

1. **Check network connectivity:**

   ```bash
   ping <node-ip>
   ssh -i ~/.ssh/id_cluster ubuntu@<node-ip> "echo OK"
   ```

2. **Check if service is running:**

   ```bash
   ssh ubuntu@<node-ip> "ps aux | grep python"
   ssh ubuntu@<node-ip> "curl localhost:8765/health"
   ```

3. **Restart service if needed:**

   ```bash
   ssh ubuntu@<node-ip> "pkill -f 'python.*app.main'; sleep 2; cd ~/ringrift/ai-service && nohup python -m app.main &"
   ```

4. **Force recover node after fix:**
   ```python
   from app.coordination.node_health_monitor import get_node_health_monitor
   monitor = get_node_health_monitor()
   monitor.force_recover("lambda-h100")
   ```

---

## Training Job Stuck

### Symptoms

- Training progress not updating
- GPU utilization at 0%
- No new checkpoints being saved

### Diagnosis

```bash
# Check training status
python -c "
from app.coordination.facade import get_coordination_facade
facade = get_coordination_facade()
status = facade.get_training_status('hex8', 2)
print(f'Status: {status}')
"

# Check for active tasks
python -c "
from app.coordination.facade import get_coordination_facade
facade = get_coordination_facade()
tasks = facade.get_active_tasks()
for t in tasks:
    print(f'{t.task_id}: {t.task_type} on {t.node_id} ({t.runtime_seconds:.0f}s)')
"
```

### Resolution

1. **Check logs on training node:**

   ```bash
   ssh ubuntu@<node-ip> "tail -100 ~/ringrift/ai-service/logs/train.log"
   ```

2. **Check GPU status:**

   ```bash
   ssh ubuntu@<node-ip> "nvidia-smi"
   ```

3. **Cancel stuck job:**

   ```python
   from app.coordination.facade import get_coordination_facade
   facade = get_coordination_facade()
   facade.stop_training("hex8", 2)
   ```

4. **Kill process manually if needed:**
   ```bash
   ssh ubuntu@<node-ip> "pkill -f 'train.*hex8'"
   ```

---

## Circuit Breaker Open

### Symptoms

- Operations failing immediately without retry
- Logs showing "Circuit breaker open"
- Cascading failures across services

### Diagnosis

```python
from app.distributed.circuit_breaker import get_circuit_breaker_status

status = get_circuit_breaker_status()
for name, state in status.items():
    print(f"{name}: {state['state']} (failures: {state['failure_count']})")
```

### Resolution

1. **Wait for automatic recovery** (default: 60 seconds half-open period)

2. **Force reset if underlying issue is fixed:**

   ```python
   from app.distributed.circuit_breaker import reset_circuit_breaker
   reset_circuit_breaker("training_pipeline")
   ```

3. **Check and fix underlying service before reset**

---

## Model Promotion Failed

### Symptoms

- Gauntlet evaluation passed but model not promoted
- Old model still serving
- Promotion script errors

### Diagnosis

```bash
# Check promotion logs
cat logs/promotion.log | tail -50

# Check model registry
python -c "
from app.training.model_registry import ModelRegistry
registry = ModelRegistry()
models = registry.list_models('hex8', 2)
for m in models:
    print(f'{m.version}: {m.status} ({m.policy_accuracy:.1%})')
"
```

### Resolution

1. **Check gauntlet results:**

   ```bash
   python scripts/auto_promote.py --gauntlet --model <path> --board-type hex8 --num-players 2 --dry-run
   ```

2. **Force promotion if evaluation passed:**

   ```python
   from app.training.model_registry import ModelRegistry
   registry = ModelRegistry()
   registry.promote_model("hex8", 2, "<version>")
   ```

3. **Sync to cluster:**
   ```bash
   python scripts/auto_promote.py --sync-to-cluster --model models/hex8_2p/best.pth
   ```

---

## Database Locked

### Symptoms

- "database is locked" errors
- SQLite timeout errors
- Slow database operations

### Diagnosis

```bash
# Check for processes holding locks
lsof data/games/*.db 2>/dev/null | head -20

# Check database integrity
sqlite3 data/games/selfplay.db "PRAGMA integrity_check;"
```

### Resolution

1. **Wait for lock to release** (usually resolves in <30s)

2. **If process is stuck, identify and kill:**

   ```bash
   lsof data/games/selfplay.db
   kill <pid>
   ```

3. **Run WAL checkpoint if needed:**

   ```bash
   sqlite3 data/games/selfplay.db "PRAGMA wal_checkpoint(TRUNCATE);"
   ```

4. **As last resort, copy database:**
   ```bash
   sqlite3 data/games/selfplay.db ".backup data/games/selfplay_backup.db"
   mv data/games/selfplay_backup.db data/games/selfplay.db
   ```

---

## Task Timeout

### Symptoms

- Tasks marked as "timed_out"
- Long-running tasks cancelled
- Incomplete results

### Diagnosis

```python
from app.coordination.task_coordinator import get_task_coordinator

coord = get_task_coordinator()
timed_out = coord.registry.get_timed_out_tasks()
for t in timed_out:
    print(f"{t.task_id}: ran for {t.runtime_seconds():.0f}s (timeout: {t.timeout_seconds}s)")
```

### Resolution

1. **Increase timeout for long tasks:**

   ```python
   # When spawning task
   facade.spawn_task("training", "lambda-h100", timeout_seconds=7200)  # 2 hours
   ```

2. **Check why task is slow:**
   - GPU memory issues
   - Network bottleneck
   - Data loading problems

3. **Clean up timed-out task state:**
   ```python
   coord.registry.update_task_status(task_id, "cancelled")
   ```

---

## DLQ Events Accumulating

### Symptoms

- Dead letter queue growing
- Events not being processed
- Retry failures

### Diagnosis

```python
from app.coordination.dead_letter_queue import get_dead_letter_queue

dlq = get_dead_letter_queue()
stats = dlq.get_stats()
print(f"Pending: {stats['pending']}")
print(f"Recovered: {stats['recovered']}")
print(f"Abandoned: {stats['abandoned']}")
print(f"By type: {stats['by_event_type']}")
```

### Resolution

1. **Check DLQ daemon is running:**

   ```python
   from app.coordination.dead_letter_queue import DLQRetryDaemon
   daemon = DLQRetryDaemon()
   # Start if not running
   import asyncio
   asyncio.run(daemon.start())
   ```

2. **Manually retry specific events:**

   ```python
   import asyncio
   asyncio.run(dlq.retry_event("<event_id>"))
   ```

3. **Purge old abandoned events:**

   ```python
   dlq.purge_old_events(days=7)
   ```

4. **Check handler errors in logs:**
   ```bash
   grep "DLQ" logs/*.log | tail -50
   ```

---

## General Debugging Tips

### Enable Debug Logging

```bash
export RINGRIFT_LOG_LEVEL=DEBUG
python -m app.main
```

### Check System Resources

```bash
# Memory
free -h

# Disk
df -h /path/to/ai-service

# GPU
nvidia-smi

# Network
ss -tulpn | grep 8765
```

### Restart All Services

```bash
# Stop all
pkill -f "python.*app.main"
pkill -f "python.*selfplay"
pkill -f "python.*train"

# Start fresh
cd ai-service && python -m app.main &
```

### Contact

For issues not covered here, check:

- `logs/` directory for detailed error messages
- `data/coordination/` for state files
- GitHub issues for known problems
