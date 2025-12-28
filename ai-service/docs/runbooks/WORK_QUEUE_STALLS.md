# Work Queue Stalls Runbook

This runbook covers diagnosis and resolution of work queue stalls - when the distributed work queue stops processing jobs.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: High

---

## Overview

The work queue system distributes:

- Selfplay jobs across GPU nodes
- Training jobs to high-memory nodes
- Evaluation jobs for gauntlet
- Tournament games

Work queue stalls prevent the cluster from generating new training data.

---

## Detection Methods

### Method 1: Check Queue Status

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()
stats = queue.get_stats()

print(f"Pending: {stats['pending']}")
print(f"In progress: {stats['in_progress']}")
print(f"Completed (last hour): {stats['completed_hour']}")
print(f"Failed (last hour): {stats['failed_hour']}")
print(f"Queue age (oldest): {stats['oldest_item_age_seconds']}s")
```

### Method 2: P2P Leader Status

```bash
# Check via P2P leader
curl -s http://localhost:8770/work_queue_status | python3 -c '
import sys, json
data = json.load(sys.stdin)
print(f"Pending: {data.get(\"pending\", 0)}")
print(f"Active workers: {data.get(\"active_workers\", 0)}")
print(f"Stalled jobs: {data.get(\"stalled\", 0)}")
'
```

### Method 3: Event Monitoring

```bash
# Check for queue events
grep -E "(WORK_QUEUED|WORK_CLAIMED|WORK_COMPLETED|WORK_TIMEOUT)" logs/coordination.log | tail -30

# Check for stall indicators
grep -i "stall\|stuck\|timeout" logs/work_queue.log | tail -10
```

---

## Stall Patterns

### Pattern 1: No Workers Claiming Jobs

**Symptom**: Jobs sit in queue, `in_progress = 0`.

**Diagnosis**:

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()
print(f"Pending: {queue.get_pending_count()}")
print(f"In progress: {queue.get_in_progress_count()}")

# Check for active workers
workers = queue.get_active_workers()
print(f"Active workers: {len(workers)}")
for w in workers:
    print(f"  {w['node']}: last_heartbeat={w['last_heartbeat']}")
```

**Common Causes**:

- All nodes offline
- Worker loop not running
- Job requirements don't match available nodes

**Fix**:

```bash
# Check P2P cluster health
curl -s http://localhost:8770/status | jq '.alive_peers'

# Restart worker loops on nodes
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "
  pkill -f 'p2p_orchestrator'
  cd ~/ringrift/ai-service
  nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &
"
```

---

### Pattern 2: Jobs Stuck In Progress

**Symptom**: `in_progress` count high, jobs never complete.

**Diagnosis**:

```python
queue = get_work_queue()
stuck = queue.get_stuck_jobs(threshold_seconds=3600)  # 1 hour
for job in stuck:
    print(f"{job['id']}: claimed by {job['worker']} at {job['claimed_at']}")
```

**Common Causes**:

- Worker crashed without releasing job
- Job running longer than timeout
- Deadlock in job execution

**Fix**:

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()

# Release stuck jobs back to pending
for job in queue.get_stuck_jobs(threshold_seconds=3600):
    queue.release_job(job['id'], reason="stuck_timeout")
    print(f"Released: {job['id']}")
```

---

### Pattern 3: All Jobs Failing

**Symptom**: High failure rate, queue drains but nothing succeeds.

**Diagnosis**:

```python
queue = get_work_queue()
recent_failures = queue.get_recent_failures(limit=20)

for f in recent_failures:
    print(f"{f['id']}: {f['error']}")
```

**Common Causes**:

- Model file missing on nodes
- Database connection issues
- GPU out of memory

**Fix**:

```bash
# Check common failure causes
grep -oP 'error": "\K[^"]+' logs/work_queue.log | sort | uniq -c | sort -rn | head -10

# Fix based on error type
# For "model not found":
python scripts/sync_models.py --distribute

# For "CUDA OOM":
# Reduce batch size in selfplay config
```

---

### Pattern 4: Queue Populator Not Running

**Symptom**: Queue empties and never refills.

**Diagnosis**:

```python
from app.coordination.unified_queue_populator import get_queue_populator

pop = get_queue_populator()
health = pop.health_check()

print(f"Healthy: {health.is_healthy}")
print(f"Last populate: {health.details.get('last_populate_time')}")
print(f"Items added (hour): {health.details.get('items_added_hour', 0)}")
```

**Fix**:

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
await dm.restart_daemon(DaemonType.QUEUE_POPULATOR)
```

---

## Recovery Procedures

### Option 1: Clear Stuck Jobs

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()

# Release all stuck jobs
stuck = queue.get_stuck_jobs(threshold_seconds=1800)  # 30 min
for job in stuck:
    queue.release_job(job['id'])
print(f"Released {len(stuck)} stuck jobs")
```

### Option 2: Reset Worker Registrations

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()

# Clear stale worker registrations
queue.clear_inactive_workers(threshold_seconds=300)
```

### Option 3: Purge Failed Jobs

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()

# Remove permanently failed jobs (exceeded retry limit)
purged = queue.purge_failed_jobs(max_age_hours=24)
print(f"Purged {purged} failed jobs")
```

### Option 4: Force Queue Repopulate

```python
from app.coordination.unified_queue_populator import get_queue_populator

pop = get_queue_populator()

# Force immediate repopulation
await pop.populate_all_queues(force=True)
```

### Option 5: Full Queue Reset

For severe issues:

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()

# Clear all queued work (WARNING: loses pending jobs)
queue.clear_all()

# Repopulate from scratch
from app.coordination.unified_queue_populator import get_queue_populator
pop = get_queue_populator()
await pop.populate_all_queues(force=True)
```

---

## Job Reaper Loop

The JobReaperLoop automatically handles stuck jobs:

```python
from scripts.p2p.loops.job_loops import JobReaperLoop

# Check reaper status
reaper = JobReaperLoop.get_instance()
print(f"Running: {reaper.is_running}")
print(f"Last run: {reaper.last_run_time}")
print(f"Jobs reaped: {reaper.stats.get('jobs_reaped', 0)}")
```

### Configure Reaper

```python
from scripts.p2p.loops.job_loops import JobReaperConfig

config = JobReaperConfig(
    stale_job_threshold_seconds=1800,  # 30 min
    stuck_job_threshold_seconds=3600,  # 1 hour
    max_jobs_to_reap_per_cycle=10,
)
```

---

## Queue Health Monitoring

### Key Metrics

| Metric                | Alert Threshold     | Source    |
| --------------------- | ------------------- | --------- |
| queue_depth           | > 1000 for > 1 hour | WorkQueue |
| stuck_jobs            | > 10                | WorkQueue |
| failure_rate          | > 50%               | WorkQueue |
| time_since_completion | > 30 min            | WorkQueue |

### Health Check Query

```python
from app.coordination.work_queue import get_work_queue

queue = get_work_queue()
health = queue.health_check()

print(f"Healthy: {health.is_healthy}")
if not health.is_healthy:
    print(f"Issues: {health.message}")
    print(f"Details: {health.details}")
```

### Dashboard Query

```bash
# Quick queue status
watch -n 10 'curl -s http://localhost:8770/work_queue_status | jq'
```

---

## P2P Work Distribution

### Check Worker Pull Loop

```python
# On worker node
from scripts.p2p.loops.job_loops import WorkerPullLoop

loop = WorkerPullLoop.get_instance()
print(f"Running: {loop.is_running}")
print(f"Jobs pulled: {loop.stats.get('jobs_pulled', 0)}")
print(f"Pull failures: {loop.stats.get('pull_failures', 0)}")
```

### Check Leader Work Assignment

```bash
# On leader node
curl -s http://localhost:8770/work_assignment_status | python3 -c '
import sys, json
data = json.load(sys.stdin)
for worker, info in data.items():
    print(f"{worker}: assigned={info[\"assigned\"]}, completed={info[\"completed\"]}")
'
```

---

## Prevention

### 1. Job Timeout Configuration

```bash
# Set reasonable timeouts
export RINGRIFT_SELFPLAY_TIMEOUT=7200    # 2 hours
export RINGRIFT_TRAINING_TIMEOUT=86400   # 24 hours
export RINGRIFT_EVALUATION_TIMEOUT=3600  # 1 hour
```

### 2. Worker Health Monitoring

```python
# In health check daemon
async def check_worker_health():
    from app.coordination.work_queue import get_work_queue

    queue = get_work_queue()
    workers = queue.get_active_workers()

    for w in workers:
        age = time.time() - w['last_heartbeat']
        if age > 300:  # 5 min without heartbeat
            emit_alert(f"Worker {w['node']} may be dead (no heartbeat for {age}s)")
```

### 3. Queue Depth Alerting

```python
# In monitoring loop
async def check_queue_depth():
    from app.coordination.work_queue import get_work_queue

    queue = get_work_queue()
    depth = queue.get_pending_count()

    if depth > 500:
        emit_alert(f"Work queue depth high: {depth} pending jobs")
```

---

## Related Documentation

- [CLUSTER_GPU_STUCK.md](CLUSTER_GPU_STUCK.md) - GPU process issues
- [P2P_ORCHESTRATOR_OPERATIONS.md](P2P_ORCHESTRATOR_OPERATIONS.md) - P2P operations
- [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md) - Daemon issues
