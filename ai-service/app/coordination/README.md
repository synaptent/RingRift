# Coordination Package

Cluster-wide coordination infrastructure for the RingRift AI training pipeline.

**67 modules** providing event-driven orchestration, resource management, and fault tolerance.

## Quick Start

```python
from app.coordination import (
    initialize_all_coordinators,
    get_system_health,
    shutdown_all_coordinators,
)

# Bootstrap all coordinators
status = initialize_all_coordinators(auto_trigger_pipeline=True)
print(f"Initialized: {status['_instances']}")

# Check system health
health = get_system_health()
print(f"Health: {health['status']} ({health['overall_health']:.0%})")

# Graceful shutdown
await shutdown_all_coordinators()
```

## Architecture Overview

```
                              ┌─────────────────────────────────────┐
                              │     initialize_all_coordinators()   │
                              │         (Single Entry Point)        │
                              └───────────────────┬─────────────────┘
                                                  │
              ┌───────────────────────────────────┼───────────────────────────────────┐
              │                                   │                                   │
              ▼                                   ▼                                   ▼
   ┌─────────────────────┐            ┌─────────────────────┐           ┌─────────────────────┐
   │     Layer 1:        │            │      Layer 2:       │           │      Layer 3:       │
   │   Foundational      │            │        Core         │           │    Application      │
   ├─────────────────────┤            ├─────────────────────┤           ├─────────────────────┤
   │ task_lifecycle      │ ────────▶  │ selfplay_orch       │ ────────▶ │ optimization_coord  │
   │ resource_monitoring │            │ data_pipeline_orch  │           │ metrics_analysis    │
   │ cache_coord_orch    │            │                     │           │                     │
   └─────────────────────┘            └─────────────────────┘           └─────────────────────┘
              │                                   │                                   │
              └───────────────────────────────────┼───────────────────────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────────────────┐
                              │        Three Event Buses            │
                              ├─────────────────────────────────────┤
                              │ DataEventBus     (in-memory async)  │
                              │ StageEventBus    (pipeline stages)  │
                              │ CrossProcessQueue (cross-node)      │
                              └─────────────────────────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────────────────┐
                              │        UnifiedEventRouter           │
                              │    (Bridges all three buses)        │
                              └─────────────────────────────────────┘
```

## Module Categories

### Core Orchestrators

| Module                               | Purpose                             | Key Functions                                           |
| ------------------------------------ | ----------------------------------- | ------------------------------------------------------- |
| `data_pipeline_orchestrator.py`      | Tracks and triggers pipeline stages | `wire_pipeline_events()`, `get_pipeline_status()`       |
| `selfplay_orchestrator.py`           | Selfplay task coordination          | `wire_selfplay_events()`, `emit_selfplay_completion()`  |
| `optimization_coordinator.py`        | Hyperparameter optimization         | `trigger_cmaes()`, `trigger_nas()`                      |
| `metrics_analysis_orchestrator.py`   | Training metrics analysis           | `analyze_metrics()`, `record_metric()`                  |
| `task_lifecycle_coordinator.py`      | Task state tracking                 | `wire_task_events()`, `get_task_stats()`                |
| `resource_monitoring_coordinator.py` | Resource utilization                | `wire_resource_events()`, `check_resource_thresholds()` |
| `cache_coordination_orchestrator.py` | Distributed cache management        | `wire_cache_events()`, `invalidate_model_caches()`      |

### Event System

| Module                    | Purpose                    | Key Classes/Functions                                  |
| ------------------------- | -------------------------- | ------------------------------------------------------ |
| `event_router.py`         | Unified event routing      | `UnifiedEventRouter`, `publish()`, `subscribe()`       |
| `stage_events.py`         | Pipeline stage events      | `StageEventBus`, `StageCompletionResult`               |
| `cross_process_events.py` | Inter-process events       | `CrossProcessEventQueue`, `publish_event()`            |
| `event_emitters.py`       | Centralized event emission | `emit_selfplay_complete()`, `emit_training_complete()` |
| `dead_letter_queue.py`    | Failed event recovery      | `DeadLetterQueue`, `run_retry_daemon()`                |

### Task Coordination

| Module                     | Purpose                      | Key Classes/Functions                                |
| -------------------------- | ---------------------------- | ---------------------------------------------------- |
| `task_coordinator.py`      | Task registration and limits | `TaskCoordinator`, `can_spawn()`                     |
| `orchestrator_registry.py` | Role-based mutual exclusion  | `acquire_orchestrator_role()`, `OrchestratorRole`    |
| `job_scheduler.py`         | Priority-based scheduling    | `PriorityJobScheduler`, `select_curriculum_config()` |
| `duration_scheduler.py`    | Task duration estimation     | `estimate_task_duration()`, `can_schedule_task()`    |
| `task_decorators.py`       | Lifecycle decorators         | `@coordinate_task`, `@task_context`                  |

### Resource Management

| Module                  | Purpose                      | Key Classes/Functions                                  |
| ----------------------- | ---------------------------- | ------------------------------------------------------ |
| `resource_optimizer.py` | PID-controlled optimization  | `ResourceOptimizer`, `get_optimal_concurrency()`       |
| `resource_targets.py`   | Utilization targets          | `get_host_targets()`, `should_scale_up()`              |
| `queue_monitor.py`      | Queue depth monitoring       | `check_backpressure()`, `should_throttle_production()` |
| `bandwidth_manager.py`  | Network bandwidth allocation | `request_bandwidth()`, `release_bandwidth()`           |
| `safeguards.py`         | Circuit breakers             | `check_before_spawn()`, `Safeguards`                   |

### Sync & Transfer

| Module                     | Purpose                         | Key Classes/Functions                              |
| -------------------------- | ------------------------------- | -------------------------------------------------- |
| `sync_coordinator.py`      | Sync scheduling                 | `SyncScheduler`, `get_sync_recommendations()`      |
| `sync_mutex.py`            | Cross-process sync locks        | `sync_lock()`, `acquire_sync_lock()`               |
| `sync_bandwidth.py`        | Bandwidth-coordinated transfers | (integrated with bandwidth_manager)                |
| `transfer_verification.py` | Checksum verification           | `verify_transfer()`, `compute_file_checksum()`     |
| `transaction_isolation.py` | ACID merge operations           | `merge_transaction()`, `begin_merge_transaction()` |

### Daemons & Lifecycle

| Module                      | Purpose                        | Key Classes/Functions                        |
| --------------------------- | ------------------------------ | -------------------------------------------- |
| `daemon_manager.py`         | Unified daemon lifecycle       | `DaemonManager`, `DaemonType`                |
| `daemon_adapters.py`        | Daemon wrappers                | (sync, promotion, distillation adapters)     |
| `unified_health_manager.py` | Error recovery                 | `UnifiedHealthManager`, `RecoveryAction`     |
| `ephemeral_data_guard.py`   | Ephemeral host data protection | `checkpoint_games()`, `request_evacuation()` |

### Configuration & Persistence

| Module                        | Purpose                   | Key Classes/Functions                       |
| ----------------------------- | ------------------------- | ------------------------------------------- |
| `coordinator_config.py`       | Centralized configuration | `get_config()`, `CoordinatorConfig`         |
| `coordinator_persistence.py`  | State snapshots           | `SnapshotCoordinator`, `StateSnapshot`      |
| `coordinator_base.py`         | Base classes              | `CoordinatorBase`, `SQLitePersistenceMixin` |
| `coordinator_dependencies.py` | Dependency graph          | (initialization ordering)                   |

### Distributed Communication

| Module                 | Purpose                       | Key Classes/Functions                         |
| ---------------------- | ----------------------------- | --------------------------------------------- |
| `cluster_transport.py` | Multi-transport communication | `ClusterTransport`, `get_cluster_transport()` |
| `p2p_backend.py`       | P2P REST client               | `P2PBackend`, `discover_p2p_leader_url()`     |
| `distributed_lock.py`  | Redis + file locking          | `training_lock()`, `acquire_training_lock()`  |
| `tracing.py`           | Distributed tracing           | `@traced`, `span()`, `get_trace_id()`         |

### Model Lifecycle

| Module                           | Purpose                  | Key Classes/Functions                                    |
| -------------------------------- | ------------------------ | -------------------------------------------------------- |
| `model_lifecycle_coordinator.py` | Model state tracking     | `ModelLifecycleCoordinator`, `get_production_model_id()` |
| `training_coordinator.py`        | Training slot management | `request_training_slot()`, `training_slot()`             |
| `async_training_bridge.py`       | Async training wrapper   | `async_request_training()`, `async_can_train()`          |

### Utilities

| Module                  | Purpose              | Key Functions                          |
| ----------------------- | -------------------- | -------------------------------------- |
| `helpers.py`            | Safe wrappers        | `*_safe()` variants with try/except    |
| `utils.py`              | Reusable patterns    | `BoundedHistory`, `CallbackRegistry`   |
| `dynamic_thresholds.py` | Adaptive thresholds  | `DynamicThreshold`, `ThresholdManager` |
| `handler_resilience.py` | Exception boundaries | `make_handlers_resilient()`            |

## Key Concepts

### 1. Event-Driven Pipeline

The training pipeline flows through stages, each emitting events:

```
SELFPLAY_COMPLETE → DATA_SYNC_COMPLETED → NPZ_EXPORT_COMPLETE → TRAINING_COMPLETED → MODEL_PROMOTED
         │                │                  │                    │                   │
         ▼                ▼                  ▼                    ▼                   ▼
    SelfplayOrch    SyncScheduler    DataPipelineOrch    TrainingCoord    CurriculumFeedback
```

Enable auto-triggering:

```python
initialize_all_coordinators(auto_trigger_pipeline=True)
```

### 2. Dead Letter Queue

Failed event handlers are captured and retried automatically:

```python
from app.coordination.dead_letter_queue import get_dead_letter_queue, run_retry_daemon

dlq = get_dead_letter_queue()
stats = dlq.get_stats()
print(f"Pending: {stats['pending']}, Recovered: {stats['recovered']}")

# Start background retry
await run_retry_daemon(interval_seconds=60)
```

### 3. Layered Initialization

Coordinators initialize in dependency order:

- **Layer 1 (Foundational)**: task_lifecycle, resources, cache
- **Layer 2 (Core)**: selfplay, pipeline
- **Layer 3 (Application)**: optimization, metrics

Failed dependencies skip dependent coordinators.

### 4. Health Monitoring

Aggregated health across all coordinators:

```python
health = get_system_health()
# Returns:
# {
#     "overall_health": 0.95,
#     "status": "healthy" | "degraded" | "unhealthy",
#     "coordinators": {"selfplay": 1.0, "pipeline": 0.8, ...},
#     "issues": ["pipeline: backpressure active", ...],
#     "handler_health": {"success_rate": 0.99, ...},
# }
```

### 5. Backpressure

Multi-level backpressure prevents resource exhaustion:

```python
from app.coordination import check_backpressure, QueueType, BackpressureLevel

level = check_backpressure(QueueType.TRAINING_DATA)
if level >= BackpressureLevel.HIGH:
    # Slow down data production
    pass
```

### 6. Distributed Tracing

Track requests across nodes:

```python
from app.coordination import traced, span, get_trace_id

@traced("selfplay_game")
async def play_game(config):
    with span("model_inference"):
        # ...
    with span("save_to_db"):
        # ...

# Trace ID propagated across process boundaries
trace_id = get_trace_id()
```

## Common Patterns

### Emitting Pipeline Events

```python
from app.coordination.event_emitters import (
    emit_selfplay_complete,
    emit_training_complete,
)

# After selfplay batch
await emit_selfplay_complete(
    config_key="hex8_2p",
    games_generated=1000,
    db_path="/data/games/selfplay.db",
)

# After training epoch
await emit_training_complete(
    config_key="hex8_2p",
    model_path="/models/hex8_2p_epoch_10.pth",
    metrics={"loss": 0.5, "accuracy": 0.75},
)
```

### Task Coordination

```python
from app.coordination import TaskCoordinator, TaskType, can_spawn

coordinator = TaskCoordinator.get_instance()

# Check if we can spawn
ok, reason = can_spawn(TaskType.SELFPLAY, "node-1")
if ok:
    task_id = coordinator.register_task(
        task_id="selfplay-001",
        task_type=TaskType.SELFPLAY,
        node_id="node-1",
        pid=os.getpid(),
    )
```

### Sync Scheduling

```python
from app.coordination import (
    get_sync_recommendations,
    execute_priority_sync,
    SyncPriority,
)

# Get sync recommendations
recommendations = get_sync_recommendations()
for rec in recommendations:
    if rec.priority >= SyncPriority.HIGH:
        await execute_priority_sync(rec)
```

## Environment Variables

| Variable                            | Default                                  | Description                           |
| ----------------------------------- | ---------------------------------------- | ------------------------------------- |
| `COORDINATOR_AUTO_TRIGGER_PIPELINE` | `false`                                  | Enable auto-trigger downstream stages |
| `RINGRIFT_COORDINATOR_DB`           | `data/coordination/coordinator.db`       | Coordinator SQLite path               |
| `RINGRIFT_DLQ_PATH`                 | `data/coordination/dead_letter_queue.db` | Dead letter queue path                |
| `RINGRIFT_TRACE_DEBUG`              | `false`                                  | Enable trace debug logging            |

## Testing

```bash
# Run coordination tests
PYTHONPATH=. python -m pytest mutants/tests/test_coordination_integration.py -v
PYTHONPATH=. python -m pytest mutants/tests/test_dead_letter_queue.py -v
```

## See Also

- `CLAUDE.md` - Project overview and AI context
- `../distributed/README.md` - Distributed layer documentation
- `../training/README.md` - Training pipeline documentation
- `docs/planning/CONSOLIDATION_ROADMAP.md` - Consolidation status
