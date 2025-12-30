# Coordination System

The coordination system provides distributed resource management, task scheduling, and safety mechanisms across the RingRift AI cluster.

## Architecture

```
                        COORDINATION SYSTEM

    +-------------------+     +-------------------+
    | master_loop       |     | p2p_orchestrator  |
    +--------+----------+     +--------+----------+
             |                         |
             +------------+------------+
                          |
                          v
    +-----------------------------------------------------+
    |                 COORDINATION LAYER                   |
    +-----------------------------------------------------+
    |                                                     |
    |  +---------------+  +---------------+  +---------+  |
    |  | Task          |  | Resource      |  | Safe-   |  |
    |  | Coordinator   |  | Optimizer     |  | guards  |  |
    |  +---------------+  +---------------+  +---------+  |
    |                                                     |
    |  +---------------+  +---------------+  +---------+  |
    |  | Bandwidth     |  | Duration      |  | Resource|  |
    |  | Manager       |  | Scheduler     |  | Targets |  |
    |  +---------------+  +---------------+  +---------+  |
    |                                                     |
    +-----------------------------------------------------+
                          |
                          v
    +-----------------------------------------------------+
    |                   DATA LAYER                         |
    +-----------------------------------------------------+
    |  data/coordination/resource_state.db                 |
    |  data/coordination/tasks.db                          |
    |  /tmp/ringrift_coordinator/                          |
    +-----------------------------------------------------+
```

## Core Abstractions

### CoordinatorBase (`coordinator_base.py`)

Abstract base class providing unified lifecycle management for all coordinators.

**Key Features:**

- Status management (INITIALIZING → READY → RUNNING → DRAINING → STOPPED)
- Async lifecycle methods (initialize, start, stop, shutdown)
- Dependency injection via setters
- Stats/metrics tracking
- Error recording

**Usage:**

```python
from app.coordination.coordinator_base import CoordinatorBase, CoordinatorStatus

class MyCoordinator(CoordinatorBase):
    def __init__(self):
        super().__init__(name="MyCoordinator")

    async def _do_start(self) -> None:
        # Custom start logic
        pass

    async def _do_stop(self) -> None:
        # Custom stop logic
        pass

    async def get_stats(self) -> Dict[str, Any]:
        stats = await super().get_stats()
        stats["custom_metric"] = self._my_metric
        return stats
```

**Status States:**
| Status | Description |
|--------------|----------------------------------------|
| INITIALIZING | Initial state before setup |
| READY | Initialized, ready to start |
| RUNNING | Actively processing |
| PAUSED | Temporarily suspended |
| DRAINING | Finishing in-progress work |
| ERROR | Error state |
| STOPPED | Shutdown complete |

### Mixins

**SQLitePersistenceMixin:**
Thread-local SQLite connections with WAL mode for concurrent access.

```python
class MyManager(CoordinatorBase, SQLitePersistenceMixin):
    def __init__(self, db_path: Path):
        super().__init__()
        self.init_db(db_path)

    def _get_schema(self) -> str:
        return '''CREATE TABLE IF NOT EXISTS my_table (...)'''
```

**SingletonMixin:**
Ensures only one instance per class exists.

```python
class MyManager(CoordinatorBase, SingletonMixin):
    @classmethod
    def get_instance(cls) -> 'MyManager':
        return cls._get_or_create_instance()
```

**CallbackMixin:**
Event-driven extensibility via callbacks.

```python
class MyCoordinator(CoordinatorBase, CallbackMixin):
    async def on_job_complete(self, job_id: str):
        await self.invoke_callbacks("job_complete", job_id)

# Register handlers
coordinator.register_callback("job_complete", my_handler)
```

### CoordinatorRegistry

Singleton registry for centralized coordinator management and graceful shutdown.

**Key Features:**

- Register/unregister coordinators
- Orchestrated shutdown with timeouts
- Signal handler integration (SIGTERM/SIGINT)
- Health summary aggregation

**Usage:**

```python
from app.coordination.coordinator_base import (
    CoordinatorRegistry,
    get_coordinator_registry,
    shutdown_all_coordinators,
)

# Get the singleton registry
registry = get_coordinator_registry()

# Register coordinators
registry.register(my_coordinator, shutdown_priority=0)

# Install signal handlers for graceful shutdown
registry.install_signal_handlers()

# Get health summary
health = registry.get_health_summary()

# Manual shutdown (with 30s timeout per coordinator)
results = await shutdown_all_coordinators(timeout=30.0)
```

**Graceful Shutdown Flow:**

```
SIGTERM/SIGINT received
       │
       ▼
┌─────────────────────┐
│ shutdown_all()      │
│ (reverse reg order) │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌────────┐  ┌────────┐
│Coord 1 │  │Coord 2 │  ...
│shutdown│  │shutdown│
└────────┘  └────────┘
           │
           ▼
    Shutdown complete
```

## Core Modules

### 1. Task Coordinator (`task_coordinator.py`)

Global task registry with hard limits to prevent runaway task spawning.

**Key Features:**

- Per-node and cluster-wide task limits
- Rate limiting with token bucket algorithm
- SQLite-based task persistence
- Gauntlet worker reservation
- Emergency stop capability

**Usage:**

```python
from app.coordination.task_coordinator import (
    TaskCoordinator,
    TaskType,
    CoordinatedTask,
    get_coordinator,
)

coordinator = get_coordinator()

# Check if spawn is allowed
allowed, reason = coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")
if allowed:
    coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1", pid=1234)
    # ... run task ...
    coordinator.unregister_task(task_id)

# Context manager approach
async with CoordinatedTask(TaskType.TRAINING, "node-1") as task:
    if task.allowed:
        await run_training()
```

**Task Limits:**
| Resource | Per-Node | Cluster-Wide |
|-------------------|----------|--------------|
| Selfplay | 32 | 500 |
| Training | 1 | 3 |
| CMA-ES | - | 1 |
| Tournament | - | 2 |
| Pipeline | - | 1 |
| Improvement Loop | - | 1 |

**Resource Types:**

- `CPU` - Selfplay, tournament, evaluation
- `GPU` - Training, CMA-ES
- `HYBRID` - Hybrid selfplay, pipeline
- `IO` - Sync, export

### 2. Resource Optimizer (`resource_optimizer.py`)

PID-controlled workload adjustment targeting 60-80% utilization.

**Key Features:**

- PID controller for smooth utilization targeting
- Predictive scaling with trend analysis
- Rate negotiation between orchestrators
- Config-weighted selfplay distribution
- Hardware-aware selfplay limits

**Usage:**

```python
from app.coordination.resource_optimizer import (
    get_resource_optimizer,
    should_scale_up,
    should_scale_down,
    negotiate_selfplay_rate,
    get_max_selfplay_for_node,
)

optimizer = get_resource_optimizer()

# Check scaling decisions
if should_scale_up("gpu"):
    # Increase GPU selfplay jobs
    pass

# Negotiate selfplay rate
approved = negotiate_selfplay_rate(
    requested_rate=2000,
    reason="momentum_accelerating",
    requestor="unified_loop",
)

# Get hardware-aware limits
max_jobs = get_max_selfplay_for_node(
    node_id="vast-h100",
    gpu_count=1,
    gpu_name="H100",
    cpu_count=32,
    memory_gb=64,
    has_gpu=True,
)
```

**Utilization Targets:**
| Threshold | Value | Action |
|------------------|-------|---------------------|
| Scale Up | <55% | Add more jobs |
| Target Min | 60% | Lower bound |
| Target Optimal | 70% | Sweet spot |
| Target Max | 80% | Upper bound |
| Scale Down | >85% | Reduce jobs |

**PID Controller:**

- Kp (proportional): 0.3
- Ki (integral): 0.05
- Kd (derivative): 0.1
- Gain scheduling for large/small errors
- Output smoothing with EMA

**Hardware-Aware Limits:**
| GPU Type | Max Jobs/GPU | Notes |
|--------------|--------------|--------------------------|
| GH200 | ~48 | Limited by CPU cores |
| H100/H200 | 24 | 80GB VRAM |
| A100/L40 | 16 | 40-80GB VRAM |
| RTX 4090 | 10 | 24GB VRAM |
| RTX 4080 | 8 | 16GB VRAM |
| RTX 3070 | 4 | 8GB VRAM |

### 3. Safeguards (`safeguards.py`)

Circuit breakers and emergency controls for spawn protection.

**Key Features:**

- Circuit breakers per node and task type
- Backpressure rate limiting
- Resource monitoring
- Emergency halt file
- Auto-OOM detection

**Usage:**

```python
from app.coordination.safeguards import Safeguards, check_before_spawn

safeguards = Safeguards.get_instance()

# Check before spawning
if safeguards.allow_spawn("selfplay", "node-1"):
    safeguards.record_spawn("selfplay", "node-1")
    # ... spawn task ...
else:
    print(f"Blocked: {safeguards.get_block_reason()}")

# Record failures
safeguards.record_failure("selfplay", "node-1", "OOM")

# Emergency controls
safeguards.activate_emergency()
safeguards.deactivate_emergency()
```

**Circuit Breaker States:**

- `CLOSED` - Normal operation
- `OPEN` - Blocking all calls (after failures)
- `HALF_OPEN` - Testing recovery

**Resource Thresholds (enforced 2025-12-16):**
| Resource | Warning | Critical |
|----------|---------|----------|
| Disk | 65% | 70% |
| Memory | 70% | 80% |
| CPU | 70% | 80% |
| GPU | 70% | 80% |
| Load Avg | - | 1.5x CPUs|

### 4. Bandwidth Manager (`bandwidth_manager.py`)

Network bandwidth allocation for data synchronization.

**Key Features:**

- Per-host bandwidth limits
- Priority-based bandwidth factors
- Concurrent transfer limits
- Historical throughput tracking

**Usage:**

```python
from app.coordination.bandwidth_manager import BandwidthManager

bw_mgr = BandwidthManager()

# Check bandwidth availability
if bw_mgr.can_transfer("runpod-a100-1", "models"):
    bw_mgr.start_transfer("runpod-a100-1", transfer_id, "models")
    # ... transfer data ...
    bw_mgr.end_transfer("runpod-a100-1", transfer_id, bytes_transferred)
```

**Bandwidth Limits:**
| Host Type | Bandwidth |
|------------|------------|
| Runpod | 1 Gbps |
| GH200 | 2.5 Gbps |
| AWS | 500 Mbps |
| Default | 100 Mbps |

**Priority Factors:**
| Priority | Factor |
|------------|--------|
| Critical | 1.0 |
| High | 0.5 |
| Normal | 0.25 |
| Background | 0.1 |

### 5. Duration Scheduler (`duration_scheduler.py`)

Task duration tracking and intelligent scheduling.

**Key Features:**

- Historical duration learning
- Peak hours avoidance
- Resource availability prediction
- Stale task detection

**Usage:**

```python
from app.coordination.duration_scheduler import DurationScheduler

scheduler = DurationScheduler()

# Get expected duration
duration = scheduler.get_expected_duration("training", "square8_2p")

# Check if task can complete before peak hours
can_start = scheduler.can_start_before_peak("training", "square8_2p")

# Record actual duration
scheduler.record_duration("training", "square8_2p", actual_seconds=14400)
```

**Default Durations:**
| Task Type | Duration |
|------------|----------|
| Selfplay | 1 hour |
| Training | 4 hours |
| CMA-ES | 8 hours |
| Pipeline | 6 hours |

**Peak Hours:**

- UTC 14:00-22:00 (avoid starting long tasks)

### 6. Resource Targets (`resource_targets.py`)

Centralized utilization targets and thresholds.

**Usage:**

```python
from app.coordination.resource_targets import (
    get_resource_targets,
    UtilizationTargets,
)

targets = get_resource_targets()
print(f"CPU target: {targets.cpu_min}-{targets.cpu_max}%")
print(f"GPU target: {targets.gpu_min}-{targets.gpu_max}%")
```

## Integration Points

### Master Loop Integration

```python
# In master_loop.py
from app.coordination.task_coordinator import get_coordinator, TaskType
from app.coordination.resource_optimizer import (
    get_resource_optimizer,
    apply_feedback_adjustment,
)

coordinator = get_coordinator()
optimizer = get_resource_optimizer()

# Before spawning selfplay
if coordinator.can_spawn_task(TaskType.SELFPLAY, node_id)[0]:
    coordinator.register_task(task_id, TaskType.SELFPLAY, node_id)

# Periodic utilization adjustment
apply_feedback_adjustment("unified_loop")
```

### P2P Orchestrator Integration

```python
# In p2p_orchestrator.py
from app.coordination.safeguards import Safeguards, check_before_spawn

safeguards = Safeguards.get_instance()

# Before spawning
allowed, reason = check_before_spawn("selfplay", node_id)
if not allowed:
    logger.warning(f"Spawn blocked: {reason}")
    return
```

## Configuration

### Environment Variables

| Variable                   | Default  | Description                |
| -------------------------- | -------- | -------------------------- |
| `RINGRIFT_TARGET_UTIL_MIN` | 60       | Minimum utilization target |
| `RINGRIFT_TARGET_UTIL_MAX` | 80       | Maximum utilization target |
| `RINGRIFT_PID_KP`          | 0.3      | PID proportional gain      |
| `RINGRIFT_PID_KI`          | 0.05     | PID integral gain          |
| `RINGRIFT_PID_KD`          | 0.1      | PID derivative gain        |
| `RINGRIFT_COORDINATOR_DIR` | /tmp/... | Coordinator data directory |

### YAML Configuration

In `config/unified_loop.yaml`:

```yaml
resource_targets:
  cpu_min: 60
  cpu_max: 80
  cpu_target: 70
  gpu_min: 60
  gpu_max: 80
  gpu_target: 70
  pid:
    kp: 0.3
    ki: 0.05
    kd: 0.1
    gain_scheduling: true
```

## Monitoring

### Prometheus Metrics

```
# Resource utilization
ringrift_cluster_cpu_utilization 0.72
ringrift_cluster_gpu_utilization 0.68
ringrift_cluster_gpu_memory_utilization 0.55

# Target tracking
ringrift_cpu_in_target_range 1
ringrift_gpu_in_target_range 1

# Optimization state
ringrift_optimization_action "none"
ringrift_optimization_confidence 0.95

# Node counts
ringrift_cluster_gpu_nodes 5
ringrift_cluster_cpu_nodes 8
ringrift_cluster_total_jobs 45
```

### Debug Commands

```bash
# Check coordinator status
python -c "
from app.coordination.task_coordinator import get_coordinator
import json
print(json.dumps(get_coordinator().get_stats(), indent=2))
"

# Check safeguards
python -c "
from app.coordination.safeguards import Safeguards
import json
print(json.dumps(Safeguards.get_instance().get_stats(), indent=2, default=str))
"

# Check resource optimizer
python -c "
from app.coordination.resource_optimizer import get_utilization_status
import json
print(json.dumps(get_utilization_status(), indent=2))
"
```

## Troubleshooting

### Tasks Not Spawning

1. Check coordinator state:

   ```python
   coordinator = get_coordinator()
   print(coordinator.get_state())  # Should be RUNNING
   ```

2. Check resource limits:

   ```python
   allowed, reason = coordinator.can_spawn_task(TaskType.SELFPLAY, node_id)
   print(f"Allowed: {allowed}, Reason: {reason}")
   ```

3. Check safeguards:
   ```python
   safeguards = Safeguards.get_instance()
   print(safeguards.get_stats())
   ```

### High/Low Utilization

1. Check current utilization:

   ```python
   optimizer = get_resource_optimizer()
   status = optimizer.get_utilization_status()
   print(status)
   ```

2. Apply feedback adjustment:
   ```python
   from app.coordination.resource_optimizer import apply_feedback_adjustment
   new_rate = apply_feedback_adjustment()
   print(f"New rate: {new_rate}")
   ```

### Circuit Breaker Open

1. Check circuit state:

   ```python
   stats = safeguards.get_stats()
   print(stats["circuit_breakers"])
   ```

2. Wait for recovery timeout (default 5 minutes)

3. Or manually reset:
   ```python
   # Not recommended unless issue is resolved
   safeguards._circuit_breakers.clear()
   ```

### Emergency Halt Active

1. Check emergency file:

   ```bash
   cat /tmp/ringrift_coordination/EMERGENCY_HALT
   ```

2. Clear emergency:
   ```bash
   rm /tmp/ringrift_coordination/EMERGENCY_HALT
   # Or:
   python -c "from app.coordination.safeguards import Safeguards; Safeguards.get_instance().deactivate_emergency()"
   ```

## Data Files

| Path                                        | Purpose                   |
| ------------------------------------------- | ------------------------- |
| `data/coordination/resource_state.db`       | Resource optimizer state  |
| `data/coordination/tasks.db`                | Task registry (fallback)  |
| `/tmp/ringrift_coordinator/tasks.db`        | Task coordinator registry |
| `/tmp/ringrift_locks/`                      | Orchestrator file locks   |
| `/tmp/ringrift_coordination/EMERGENCY_HALT` | Emergency halt flag file  |

## Related Documentation

- [TRAINING_OPTIMIZATIONS.md](../training/TRAINING_OPTIMIZATIONS.md) - Training pipeline optimizations
- [FEEDBACK_ACCELERATOR.md](../training/FEEDBACK_ACCELERATOR.md) - Elo momentum tracking
- [scripts/README.md](../../scripts/README.md) - Resource guard usage

### Monitoring

- **Grafana Dashboard**: `monitoring/grafana/dashboards/coordinators.json`
- **Prometheus Alerts**: `monitoring/prometheus/rules/coordinator_alerts.yml`
- **Admin Endpoints**:
  - `GET /admin/health/coordinators` - Coordinator health status
  - `GET /admin/health/full` - Full system health check
  - `GET /admin/sync/status` - Sync coordinator status
  - `POST /admin/sync/trigger` - Trigger a manual sync (games/training/models)

---

## December 2025 Consolidation Status

### Event System - Fully Consolidated

Three event buses unified under a single router:

```
┌─────────────────────────────────────────┐
│     UnifiedEventRouter (event_router.py) │
├─────────────────────────────────────────┤
│  • SHA256 content deduplication         │
│  • Cross-process bridging               │
│  • Automatic type mapping               │
└─────────────────────────────────────────┘
        ↓          ↓          ↓
   DataEventBus  StageEventBus  CrossProcessQueue
```

### Feedback Loop Wiring - 100% Complete

All 8 critical AI training events fully wired:

| Event                  | Publishers | Subscribers | Status |
| ---------------------- | ---------- | ----------- | ------ |
| TRAINING_COMPLETED     | 5          | 16          | ✅     |
| EVALUATION_COMPLETED   | 4          | 14          | ✅     |
| SELFPLAY_COMPLETE      | 1          | 5           | ✅     |
| SELFPLAY_RATE_CHANGED  | 1          | 4           | ✅     |
| PLATEAU_DETECTED       | 4          | 2           | ✅     |
| MODEL_PROMOTED         | 7          | 14          | ✅     |
| HYPERPARAMETER_UPDATED | 4          | 6           | ✅     |
| NEW_GAMES_AVAILABLE    | 7          | 6           | ✅     |

### Daemon Manager - 100% Factory Coverage

- **89 DaemonType** enum values defined (78 active, 11 deprecated)
- **89 runner functions** implemented in `daemon_runners.py`
- **Daemon categories**: See `../DAEMON_REGISTRY.md` for the current list

### Cluster Infrastructure

- **28 nodes** configured across 6 providers
- **P2P voter quorum**: 5 stable nodes
- **Multi-transport sync**: aria2, SSH/rsync, HTTP P2P
- **Write-ahead logs** for crash recovery

### Quick Reference - New Components (Dec 2025)

| Component                | File                              | Purpose                                        |
| ------------------------ | --------------------------------- | ---------------------------------------------- |
| Unified Event Router     | `event_router.py`                 | Single event entry point                       |
| Auto-Promotion Daemon    | `auto_promotion_daemon.py`        | Automatic model promotion                      |
| Ephemeral Sync           | `auto_sync_daemon.py`             | 5-second sync for Vast.ai (strategy=ephemeral) |
| Feedback Loop Controller | `feedback_loop_controller.py`     | Training feedback signals                      |
| Gauntlet Feedback        | `gauntlet_feedback_controller.py` | Eval→training bridge                           |
| Job Scheduler            | `job_scheduler.py`                | Fair allocation, starvation prevention         |

### Remaining Consolidation Work

1. **Orchestrator variants** (18 files) → target 3 focused variants
2. **Checkpoint variants** (4 files) → checkpoint_unified.py is canonical
3. **Top-level coordination files** (117) → package into core/cluster/training/resources
