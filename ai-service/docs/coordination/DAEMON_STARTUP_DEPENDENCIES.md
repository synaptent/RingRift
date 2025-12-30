# Daemon Startup Dependencies

This document explains how daemon startup ordering and dependencies work in the RingRift AI service coordination infrastructure.

**Last Updated**: December 30, 2025

## Overview

The daemon manager starts 85 daemon types with specific dependencies. Incorrect startup order can cause race conditions where daemons try to use resources that aren't ready.

There are two canonical dependency layers:

1. **`daemon_types.py`**: `DAEMON_STARTUP_ORDER` (explicit production startup order)
2. **`daemon_registry.py`**: Declarative `depends_on` specifications (canonical dependencies)

`daemon_types.py:DAEMON_DEPENDENCIES` remains a legacy validation map for startup checks.

## Key Concepts

### Startup Order (daemon_types.py)

The `DAEMON_STARTUP_ORDER` defines the startup sequence used in production:

```
Position | Daemon Type                | Dependencies                             | Why This Position?
---------|----------------------------|------------------------------------------|--------------------
1        | EVENT_ROUTER               | (None - first)                           | Event bus for all others
2        | DAEMON_WATCHDOG            | EVENT_ROUTER                             | Self-healing for daemon crashes
3        | DATA_PIPELINE              | EVENT_ROUTER                             | Must receive DATA_SYNC_COMPLETED events
4        | FEEDBACK_LOOP              | EVENT_ROUTER                             | Must receive TRAINING_COMPLETED events
5        | AUTO_SYNC                  | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP | Sync events need subscribers
6        | QUEUE_POPULATOR            | EVENT_ROUTER, SELFPLAY_COORDINATOR       | Work queue maintenance
7        | WORK_QUEUE_MONITOR         | EVENT_ROUTER, QUEUE_POPULATOR            | Queue visibility after populator
8        | COORDINATOR_HEALTH_MONITOR | EVENT_ROUTER                             | Coordinator lifecycle visibility
9        | IDLE_RESOURCE              | EVENT_ROUTER                             | GPU utilization monitoring
10       | TRAINING_TRIGGER           | EVENT_ROUTER, AUTO_EXPORT                | Training trigger after export
11       | CLUSTER_MONITOR            | EVENT_ROUTER                             | Cluster monitoring
12       | NODE_HEALTH_MONITOR        | EVENT_ROUTER                             | Node health checks (deprecated, still ordered)
13       | HEALTH_SERVER              | EVENT_ROUTER                             | Health endpoints
14       | CLUSTER_WATCHDOG           | EVENT_ROUTER, CLUSTER_MONITOR            | Cluster watchdog depends on monitor
15       | NODE_RECOVERY              | EVENT_ROUTER                             | Node recovery actions
16       | QUALITY_MONITOR            | EVENT_ROUTER                             | Quality monitoring before evaluation
17       | DISTILLATION               | EVENT_ROUTER                             | Distillation readiness
18       | EVALUATION                 | EVENT_ROUTER                             | Model evaluation
19       | UNIFIED_PROMOTION          | EVENT_ROUTER                             | Promotion controller wiring
20       | AUTO_PROMOTION             | EVENT_ROUTER, EVALUATION                 | Auto-promotion after evaluation
21       | MODEL_DISTRIBUTION         | EVENT_ROUTER, EVALUATION, AUTO_PROMOTION | Distribute promoted models
```

**Critical Ordering Rule**: Event _subscribers_ (DATA*PIPELINE, FEEDBACK_LOOP) must start
before event \_emitters* (AUTO_SYNC) to avoid lost events.

### Registry-Based Dependencies (daemon_registry.py)

The canonical source of dependencies is `DAEMON_REGISTRY` in `daemon_registry.py`:

```python
from app.coordination.daemon_registry import DAEMON_REGISTRY, DaemonSpec

# Example entries
DAEMON_REGISTRY = {
    DaemonType.AUTO_SYNC: DaemonSpec(
        runner_name="create_auto_sync",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.FEEDBACK_LOOP),
        category="sync",
    ),
    DaemonType.TRAINING_TRIGGER: DaemonSpec(
        runner_name="create_training_trigger",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AUTO_EXPORT),
        category="pipeline",
    ),
    # ... 70+ entries
}
```

**Key fields in DaemonSpec:**

| Field                   | Type                     | Description                         |
| ----------------------- | ------------------------ | ----------------------------------- |
| `runner_name`           | `str`                    | Function name in daemon_runners.py  |
| `depends_on`            | `tuple[DaemonType, ...]` | Daemons that must start first       |
| `category`              | `str`                    | Grouping: sync, event, health, etc. |
| `auto_restart`          | `bool`                   | Restart on failure (default: True)  |
| `health_check_interval` | `float \| None`          | Custom health check interval        |
| `deprecated`            | `bool`                   | Marked for removal                  |

### Runtime Dependency Enforcement

When a daemon starts, DaemonManager checks dependencies:

1. All daemons in `depends_on` must be RUNNING
2. If any dependency is not running, startup is deferred
3. DAEMON_WATCHDOG retries deferred daemons periodically

### Readiness Protocol

Daemons signal readiness via `mark_daemon_ready(DaemonType)`:

```python
# In daemon implementation
async def start(self):
    await self._initialize()
    mark_daemon_ready(DaemonType.MY_DAEMON)  # Signal ready
    await self._run()
```

Other daemons can wait for dependencies:

```python
# In DaemonManager
ready_event = self._daemons[DaemonType.EVENT_ROUTER].ready_event
await asyncio.wait_for(ready_event.wait(), timeout=30.0)
```

## Critical Dependencies

### EVENT_ROUTER (Position 1)

**Must start first**. All event-driven daemons depend on it:

- DAEMON_WATCHDOG
- DATA_PIPELINE
- FEEDBACK_LOOP
- AUTO_SYNC
- All monitoring daemons

### DATA_PIPELINE (Position 3)

Must start **before** sync daemons to receive DATA_SYNC_COMPLETED events:

- If DATA_PIPELINE starts after AUTO_SYNC, sync completion events are lost
- Training pipeline won't trigger automatically

### FEEDBACK_LOOP (Position 4)

Must start before sync daemons for curriculum integration:

- Receives TRAINING_COMPLETED events
- Emits EXPLORATION_BOOST for selfplay adjustment

## Validation

Startup order consistency is validated at runtime:

```python
from app.coordination.daemon_types import validate_startup_order_or_raise

# Raises ValueError if dependencies come AFTER their dependents
validate_startup_order_or_raise()
```

Example violation:

```
DATA_PIPELINE (pos 3) depends on AUTO_SYNC (pos 5), but AUTO_SYNC starts AFTER DATA_PIPELINE
```

## Adding New Daemons

1. **Add DaemonType enum** in `daemon_types.py`
2. **Add runner function** in `daemon_runners.py`
3. **Add to DAEMON_STARTUP_ORDER** at the correct position
4. **Add dependencies** to `DAEMON_DEPENDENCIES` if needed
5. **Implement mark_daemon_ready()** call in the daemon

### Example

```python
# daemon_types.py
class DaemonType(Enum):
    MY_NEW_DAEMON = "my_new_daemon"

DAEMON_DEPENDENCIES[DaemonType.MY_NEW_DAEMON] = {
    DaemonType.EVENT_ROUTER,
    DaemonType.DATA_PIPELINE,
}

# Insert in DAEMON_STARTUP_ORDER after DATA_PIPELINE
```

## Troubleshooting

### "Daemon X failed to start - dependency Y not running"

1. Check `DAEMON_STARTUP_ORDER` - is Y before X?
2. Check if Y failed to start (look at logs)
3. Verify `DAEMON_DEPENDENCIES[X]` includes Y

### "Events not received after sync"

Likely cause: DATA_PIPELINE or FEEDBACK_LOOP started after AUTO_SYNC.

Fix: Ensure positions 3-4 (DATA_PIPELINE, FEEDBACK_LOOP) come before position 5 (AUTO_SYNC).

### "Daemon stuck waiting for ready signal"

```bash
# Check which daemons are waiting
grep "Waiting for" logs/daemon_manager.log
```

The `ready_timeout` (default 30s) prevents infinite waits.

## Daemon Categories

Categories from `daemon_registry.py` (use `get_daemons_by_category(category)`):

| Category     | Daemons                                             | Description                  |
| ------------ | --------------------------------------------------- | ---------------------------- |
| event        | EVENT_ROUTER, CROSS_PROCESS_POLLER, DLQ_RETRY       | Event bus infrastructure     |
| health       | NODE_HEALTH_MONITOR, CLUSTER_MONITOR, HEALTH_SERVER | Monitoring and health checks |
| sync         | AUTO_SYNC, ELO_SYNC, S3_NODE_SYNC, SYNC_PUSH        | Data synchronization         |
| pipeline     | DATA_PIPELINE, TRAINING_TRIGGER, AUTO_EXPORT        | Training pipeline stages     |
| evaluation   | EVALUATION, AUTO_PROMOTION, GAUNTLET_FEEDBACK       | Model evaluation & promotion |
| distribution | MODEL_DISTRIBUTION, NPZ_DISTRIBUTION, DATA_SERVER   | Model/data distribution      |
| feedback     | FEEDBACK_LOOP, CURRICULUM_INTEGRATION               | Training feedback signals    |
| queue        | QUEUE_POPULATOR, JOB_SCHEDULER                      | Work queue management        |
| resource     | IDLE_RESOURCE, NODE_RECOVERY, UTILIZATION_OPTIMIZER | Resource management          |
| recovery     | DISK_SPACE_MANAGER, MAINTENANCE, ORPHAN_DETECTION   | Cleanup and recovery         |
| provider     | MULTI_PROVIDER                                      | Cloud provider coordination  |

```python
# Query daemons by category
from app.coordination.daemon_registry import get_daemons_by_category, get_categories

sync_daemons = get_daemons_by_category("sync")
all_categories = get_categories()  # Returns sorted list
```

## Validation Commands

```bash
# Validate registry consistency (at startup)
python -c "from app.coordination.daemon_registry import validate_registry_or_raise; validate_registry_or_raise()"

# Check for deprecated daemons
python -c "from app.coordination.daemon_registry import get_deprecated_daemons; print('\n'.join(f'{d[0].name}: {d[1]}' for d in get_deprecated_daemons()))"

# Validate startup order consistency
python -c "from app.coordination.daemon_types import validate_startup_order_or_raise; validate_startup_order_or_raise()"

# Check registry health
python -c "from app.coordination.daemon_registry import check_registry_health; r = check_registry_health(); print(f'Healthy: {r.healthy}, {r.message}')"
```

## Related Files

- `app/coordination/daemon_types.py` - DaemonType enum, DaemonState, DaemonInfo
- `app/coordination/daemon_manager.py` - Lifecycle management (~2,000 LOC)
- `app/coordination/daemon_runners.py` - 70 runner functions (~1,100 LOC)
- `app/coordination/daemon_registry.py` - Declarative daemon specs
- `app/coordination/daemon_lifecycle.py` - Lifecycle operations
- `scripts/master_loop.py` - Production startup order

## References

- [DAEMON_REGISTRY.md](../DAEMON_REGISTRY.md) - Full daemon reference
- [DAEMON_FAILURE_RECOVERY.md](../runbooks/DAEMON_FAILURE_RECOVERY.md) - Recovery procedures
- [DAEMON_MANAGER_OPERATIONS.md](../runbooks/DAEMON_MANAGER_OPERATIONS.md) - Operational commands
