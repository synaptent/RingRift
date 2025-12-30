# Daemon Failure Recovery Runbook

This runbook covers detection, diagnosis, and recovery procedures for daemon failures in the RingRift AI training infrastructure.

**Created**: December 27, 2025
**Version**: 1.0

## Overview

The RingRift AI service runs 85 daemon types managed by `DaemonManager`. This runbook covers:

1. How to detect daemon failures
2. Common failure patterns and causes
3. Recovery procedures by daemon type
4. Restart patterns and exponential backoff

---

## Detecting Daemon Failures

### Method 1: Health Server Endpoints

The DaemonManager exposes HTTP health endpoints on port 8790:

```bash
# Check liveness (all critical daemons running)
curl http://localhost:8790/health

# Check readiness (system ready for traffic)
curl http://localhost:8790/ready

# Get detailed metrics
curl http://localhost:8790/metrics
```

### Method 2: Programmatic Health Check

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Check specific daemon health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(f"Status: {health.status}, Healthy: {health.healthy}")

# Check all daemons
all_health = dm.get_all_daemon_health()
for dtype, health in all_health.items():
    if not health.healthy:
        print(f"UNHEALTHY: {dtype} - {health.message}")
```

### Method 3: CLI Status

```bash
# Check daemon status via launcher
python scripts/launch_daemons.py --status

# Check via master loop
python scripts/master_loop.py --watch
```

---

## Common Failure Patterns

### Pattern 1: Startup Failure

**Symptoms**: Daemon never enters RUNNING state

**Common Causes**:

- Missing dependencies (import errors)
- Configuration errors
- Port conflicts

**Diagnosis**:

```bash
# Check logs for import errors
grep -r "ImportError\|ModuleNotFound" logs/

# Check for configuration issues
grep -r "Config\|ValueError" logs/coordination.log
```

**Recovery**: Fix the underlying issue and restart via DaemonManager.

---

### Pattern 2: Crash Loop

**Symptoms**: Daemon starts but crashes repeatedly

**Common Causes**:

- Database connection failures
- Network timeouts
- Resource exhaustion

**Diagnosis**:

```python
dm = get_daemon_manager()
info = dm.get_daemon_info(DaemonType.AUTO_SYNC)
print(f"Restart count: {info.restart_count}")
print(f"Last error: {info.last_error}")
```

**Recovery**: Check resource availability, then:

```python
# Reset restart counter and retry
await dm.stop_daemon(DaemonType.AUTO_SYNC)
await dm.start_daemon(DaemonType.AUTO_SYNC)
```

---

### Pattern 3: Health Check Failure

**Symptoms**: Daemon running but health_check() returns unhealthy

**Common Causes**:

- High error rate
- Stale data
- Lost connections

**Diagnosis**:

```python
health = await dm.get_daemon_health(DaemonType.DATA_PIPELINE)
print(f"Message: {health.message}")
print(f"Details: {health.details}")
```

**Recovery**: Address the specific issue in health.details.

---

### Pattern 4: Event Subscription Loss

**Symptoms**: Daemon running but not processing events

**Common Causes**:

- EventRouter restarted without re-subscribing
- Subscription exception during startup

**Diagnosis**:

```bash
export RINGRIFT_LOG_LEVEL=DEBUG
# Check if events are being delivered
grep "Delivered to" logs/coordination.log | grep -v "YOUR_DAEMON"
```

**Recovery**: Restart the daemon to re-subscribe:

```python
await dm.restart_daemon(DaemonType.FEEDBACK_LOOP)
```

---

## Recovery Procedures by Daemon Type

### Critical Daemons (Auto-Restart Enabled)

| Daemon Type     | Recovery Time | Action                                |
| --------------- | ------------- | ------------------------------------- |
| `EVENT_ROUTER`  | 5s            | Auto-restart, all others re-subscribe |
| `DATA_PIPELINE` | 10s           | Auto-restart, may miss some events    |
| `AUTO_SYNC`     | 15s           | Auto-restart, sync resumes            |
| `FEEDBACK_LOOP` | 10s           | Auto-restart, feedback state reset    |

### Non-Critical Daemons

| Daemon Type       | Impact if Down    | Manual Recovery                               |
| ----------------- | ----------------- | --------------------------------------------- |
| `IDLE_SHUTDOWN`   | GPU costs         | `dm.start_daemon(DaemonType.IDLE_SHUTDOWN)`   |
| `QUALITY_MONITOR` | No quality alerts | `dm.start_daemon(DaemonType.QUALITY_MONITOR)` |
| `ELO_SYNC`        | Stale Elo data    | `dm.start_daemon(DaemonType.ELO_SYNC)`        |

---

## Exponential Backoff

DaemonManager uses exponential backoff for restarts:

| Restart # | Delay     | Total Time |
| --------- | --------- | ---------- |
| 1         | 1s        | 1s         |
| 2         | 2s        | 3s         |
| 3         | 4s        | 7s         |
| 4         | 8s        | 15s        |
| 5         | 16s (max) | 31s        |

After 5 restarts, the daemon enters FAILED state and requires manual intervention.

### Resetting a Failed Daemon

```python
dm = get_daemon_manager()

# Reset and restart
dm.reset_daemon_state(DaemonType.FAILED_DAEMON)
await dm.start_daemon(DaemonType.FAILED_DAEMON)
```

---

## Full System Recovery

If multiple daemons are failing, perform a full restart:

```bash
# Stop all daemons gracefully
python scripts/launch_daemons.py --stop-all

# Wait for cleanup
sleep 10

# Restart with correct order
python scripts/launch_daemons.py --all

# Or use master loop for full orchestration
python scripts/master_loop.py
```

---

## Monitoring Alerts

### Set Up Alerting

```python
from app.coordination.daemon_manager import get_daemon_manager

dm = get_daemon_manager()

# Add alert callback
def on_daemon_failure(dtype, error):
    # Send to your alerting system
    print(f"ALERT: {dtype} failed: {error}")

dm.set_failure_callback(on_daemon_failure)
```

### Key Metrics to Monitor

| Metric                | Alert Threshold | Source            |
| --------------------- | --------------- | ----------------- |
| restart_count         | > 3 in 1 hour   | DaemonInfo        |
| health_check failures | > 2 consecutive | HealthCheckResult |
| error_rate            | > 50%           | health.details    |

---

## Troubleshooting Checklist

1. **Check logs**: `grep "ERROR\|CRITICAL" logs/coordination.log`
2. **Check restart count**: Is daemon in crash loop?
3. **Check dependencies**: Are required services running (P2P, DB)?
4. **Check resources**: Disk space, memory, connections
5. **Check event subscriptions**: Is EventRouter healthy?
6. **Check startup order**: Did dependencies start first?

---

## Related Documentation

- [EVENT_SYSTEM_REFERENCE.md](../EVENT_SYSTEM_REFERENCE.md) - Event subscription order
- [CLUSTER_CONNECTIVITY.md](CLUSTER_CONNECTIVITY.md) - Network troubleshooting
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General troubleshooting
