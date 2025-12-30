# Daemon Registry Reference

> **Deprecated:** This document is a legacy snapshot and is no longer maintained. Use `DAEMON_REGISTRY.md` and `app/coordination/daemon_registry.py` for canonical daemon listings.

This document provides a comprehensive reference for all daemon types in the RingRift AI service coordination infrastructure.

**Last Updated**: December 2025

## Overview

The RingRift AI service uses 73 daemon types organized into 12 categories. Daemons are managed by `DaemonManager` with declarative configuration in `daemon_registry.py`.

### Architecture

```
daemon_types.py          → DaemonType enum (73 types)
daemon_registry.py       → DAEMON_REGISTRY (declarative specs)
daemon_runners.py        → Runner functions (async factories)
daemon_manager.py        → Lifecycle management
```

### Key Files

| File                                  | Purpose                                                    | LOC    |
| ------------------------------------- | ---------------------------------------------------------- | ------ |
| `app/coordination/daemon_types.py`    | DaemonType enum, DAEMON_DEPENDENCIES, DAEMON_STARTUP_ORDER | ~800   |
| `app/coordination/daemon_registry.py` | Declarative DaemonSpec configurations                      | ~870   |
| `app/coordination/daemon_runners.py`  | Async runner functions for each daemon                     | ~1,100 |
| `app/coordination/daemon_manager.py`  | Lifecycle, health checks, auto-restart                     | ~2,000 |

---

## Daemon Categories

### Sync (12 daemons)

Data synchronization across cluster nodes.

| Daemon Type             | Runner                         | Dependencies                               | Health Interval | Status         |
| ----------------------- | ------------------------------ | ------------------------------------------ | --------------- | -------------- |
| `AUTO_SYNC`             | `create_auto_sync`             | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP | 30s             | Active         |
| `HIGH_QUALITY_SYNC`     | `create_high_quality_sync`     | EVENT_ROUTER                               | 30s             | Active         |
| `ELO_SYNC`              | `create_elo_sync`              | EVENT_ROUTER                               | 30s             | Active         |
| `GOSSIP_SYNC`           | `create_gossip_sync`           | EVENT_ROUTER                               | 30s             | Active         |
| `TRAINING_NODE_WATCHER` | `create_training_node_watcher` | EVENT_ROUTER, DATA_PIPELINE                | 30s             | Active         |
| `TRAINING_DATA_SYNC`    | `create_training_data_sync`    | EVENT_ROUTER                               | 30s             | Active         |
| `S3_NODE_SYNC`          | `create_s3_node_sync`          | EVENT_ROUTER                               | 30s             | Active         |
| `S3_CONSOLIDATION`      | `create_s3_consolidation`      | EVENT_ROUTER, S3_NODE_SYNC                 | 30s             | Active         |
| `SYNC_PUSH`             | `create_sync_push`             | EVENT_ROUTER                               | 60s             | Active         |
| `UNIFIED_DATA_PLANE`    | `create_unified_data_plane`    | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP | 60s             | Active         |
| `SYNC_COORDINATOR`      | `create_sync_coordinator`      | EVENT_ROUTER                               | 30s             | **Deprecated** |
| `EPHEMERAL_SYNC`        | `create_ephemeral_sync`        | EVENT_ROUTER, DATA_PIPELINE                | 30s             | **Deprecated** |

### Event (3 daemons)

Event routing and processing infrastructure.

| Daemon Type            | Runner                        | Dependencies | Health Interval | Status       |
| ---------------------- | ----------------------------- | ------------ | --------------- | ------------ |
| `EVENT_ROUTER`         | `create_event_router`         | None         | 30s             | **Critical** |
| `CROSS_PROCESS_POLLER` | `create_cross_process_poller` | EVENT_ROUTER | 30s             | Active       |
| `DLQ_RETRY`            | `create_dlq_retry`            | EVENT_ROUTER | 30s             | Active       |

### Health (11 daemons)

Health monitoring and watchdog services.

| Daemon Type                  | Runner                              | Dependencies                      | Health Interval | Status         |
| ---------------------------- | ----------------------------------- | --------------------------------- | --------------- | -------------- |
| `DAEMON_WATCHDOG`            | `create_daemon_watchdog`            | EVENT_ROUTER                      | 30s             | **Critical**   |
| `CLUSTER_MONITOR`            | `create_cluster_monitor`            | EVENT_ROUTER                      | 30s             | Active         |
| `CLUSTER_WATCHDOG`           | `create_cluster_watchdog`           | EVENT_ROUTER, CLUSTER_MONITOR     | 30s             | Active         |
| `QUALITY_MONITOR`            | `create_quality_monitor`            | EVENT_ROUTER                      | 30s             | Active         |
| `MODEL_PERFORMANCE_WATCHDOG` | `create_model_performance_watchdog` | EVENT_ROUTER                      | 30s             | Active         |
| `COORDINATOR_HEALTH_MONITOR` | `create_coordinator_health_monitor` | EVENT_ROUTER                      | 30s             | Active         |
| `WORK_QUEUE_MONITOR`         | `create_work_queue_monitor`         | EVENT_ROUTER, QUEUE_POPULATOR     | 30s             | Active         |
| `QUEUE_MONITOR`              | `create_queue_monitor`              | EVENT_ROUTER                      | 30s             | Active         |
| `HEALTH_SERVER`              | `create_health_server`              | EVENT_ROUTER                      | 30s             | Active         |
| `AVAILABILITY_NODE_MONITOR`  | `create_availability_node_monitor`  | EVENT_ROUTER                      | 30s             | Active         |
| `HEALTH_CHECK`               | `create_health_check`               | None                              | 30s             | **Deprecated** |
| `NODE_HEALTH_MONITOR`        | `create_node_health_monitor`        | EVENT_ROUTER                      | 30s             | **Deprecated** |
| `SYSTEM_HEALTH_MONITOR`      | `create_system_health_monitor`      | EVENT_ROUTER, NODE_HEALTH_MONITOR | 30s             | **Deprecated** |

### Pipeline (7 daemons)

Training pipeline orchestration.

| Daemon Type                | Runner                            | Dependencies                | Health Interval | Status       |
| -------------------------- | --------------------------------- | --------------------------- | --------------- | ------------ |
| `DATA_PIPELINE`            | `create_data_pipeline`            | EVENT_ROUTER                | 30s             | **Critical** |
| `SELFPLAY_COORDINATOR`     | `create_selfplay_coordinator`     | EVENT_ROUTER                | 30s             | Active       |
| `TRAINING_TRIGGER`         | `create_training_trigger`         | EVENT_ROUTER, AUTO_EXPORT   | 30s             | Active       |
| `AUTO_EXPORT`              | `create_auto_export`              | EVENT_ROUTER                | 30s             | Active       |
| `TOURNAMENT_DAEMON`        | `create_tournament_daemon`        | EVENT_ROUTER                | 30s             | Active       |
| `CONTINUOUS_TRAINING_LOOP` | `create_continuous_training_loop` | EVENT_ROUTER                | 30s             | Active       |
| `DATA_CONSOLIDATION`       | `create_data_consolidation`       | EVENT_ROUTER, DATA_PIPELINE | 30s             | Active       |
| `NPZ_COMBINATION`          | `create_npz_combination`          | EVENT_ROUTER, DATA_PIPELINE | 60s             | Active       |

### Evaluation (4 daemons)

Model evaluation and promotion.

| Daemon Type         | Runner                     | Dependencies             | Health Interval | Status |
| ------------------- | -------------------------- | ------------------------ | --------------- | ------ |
| `EVALUATION`        | `create_evaluation_daemon` | EVENT_ROUTER             | 30s             | Active |
| `AUTO_PROMOTION`    | `create_auto_promotion`    | EVENT_ROUTER, EVALUATION | 30s             | Active |
| `UNIFIED_PROMOTION` | `create_unified_promotion` | EVENT_ROUTER             | 30s             | Active |
| `GAUNTLET_FEEDBACK` | `create_gauntlet_feedback` | EVENT_ROUTER             | 30s             | Active |

### Distribution (4 daemons)

Model and data distribution across cluster.

| Daemon Type          | Runner                      | Dependencies                             | Health Interval | Status         |
| -------------------- | --------------------------- | ---------------------------------------- | --------------- | -------------- |
| `MODEL_SYNC`         | `create_model_sync`         | EVENT_ROUTER                             | 30s             | Active         |
| `MODEL_DISTRIBUTION` | `create_model_distribution` | EVENT_ROUTER, EVALUATION, AUTO_PROMOTION | 30s             | Active         |
| `DATA_SERVER`        | `create_data_server`        | EVENT_ROUTER                             | 30s             | Active         |
| `NPZ_DISTRIBUTION`   | `create_npz_distribution`   | EVENT_ROUTER                             | 30s             | **Deprecated** |

### Replication (2 daemons)

Data replication monitoring and repair.

| Daemon Type           | Runner                       | Dependencies | Health Interval | Status         |
| --------------------- | ---------------------------- | ------------ | --------------- | -------------- |
| `REPLICATION_MONITOR` | `create_replication_monitor` | EVENT_ROUTER | 30s             | **Deprecated** |
| `REPLICATION_REPAIR`  | `create_replication_repair`  | EVENT_ROUTER | 30s             | **Deprecated** |

### Resource (8 daemons)

GPU/CPU resource management.

| Daemon Type                     | Runner                                 | Dependencies                                | Health Interval | Status       |
| ------------------------------- | -------------------------------------- | ------------------------------------------- | --------------- | ------------ |
| `IDLE_RESOURCE`                 | `create_idle_resource`                 | EVENT_ROUTER                                | 30s             | **Critical** |
| `NODE_RECOVERY`                 | `create_node_recovery`                 | EVENT_ROUTER                                | 30s             | Active       |
| `RESOURCE_OPTIMIZER`            | `create_resource_optimizer`            | EVENT_ROUTER, JOB_SCHEDULER                 | 30s             | Active       |
| `UTILIZATION_OPTIMIZER`         | `create_utilization_optimizer`         | EVENT_ROUTER, IDLE_RESOURCE                 | 30s             | Active       |
| `ADAPTIVE_RESOURCES`            | `create_adaptive_resources`            | EVENT_ROUTER, CLUSTER_MONITOR               | 30s             | Active       |
| `NODE_AVAILABILITY`             | `create_node_availability`             | EVENT_ROUTER                                | 30s             | Active       |
| `AVAILABILITY_CAPACITY_PLANNER` | `create_availability_capacity_planner` | EVENT_ROUTER                                | 30s             | Active       |
| `AVAILABILITY_PROVISIONER`      | `create_availability_provisioner`      | EVENT_ROUTER, AVAILABILITY_CAPACITY_PLANNER | 30s             | Active       |

### Provider (3 daemons)

Cloud provider-specific management.

| Daemon Type      | Runner                  | Dependencies                  | Health Interval | Status         |
| ---------------- | ----------------------- | ----------------------------- | --------------- | -------------- |
| `MULTI_PROVIDER` | `create_multi_provider` | EVENT_ROUTER, CLUSTER_MONITOR | 30s             | Active         |
| `LAMBDA_IDLE`    | `create_lambda_idle`    | EVENT_ROUTER, CLUSTER_MONITOR | 30s             | **Deprecated** |
| `VAST_IDLE`      | `create_vast_idle`      | EVENT_ROUTER, CLUSTER_MONITOR | 30s             | **Deprecated** |

### Queue (2 daemons)

Work queue and job scheduling.

| Daemon Type       | Runner                   | Dependencies                       | Health Interval | Status       |
| ----------------- | ------------------------ | ---------------------------------- | --------------- | ------------ |
| `QUEUE_POPULATOR` | `create_queue_populator` | EVENT_ROUTER, SELFPLAY_COORDINATOR | 30s             | **Critical** |
| `JOB_SCHEDULER`   | `create_job_scheduler`   | EVENT_ROUTER                       | 30s             | Active       |

### Feedback (2 daemons)

Training feedback and curriculum.

| Daemon Type              | Runner                          | Dependencies | Health Interval | Status       |
| ------------------------ | ------------------------------- | ------------ | --------------- | ------------ |
| `FEEDBACK_LOOP`          | `create_feedback_loop`          | EVENT_ROUTER | 30s             | **Critical** |
| `CURRICULUM_INTEGRATION` | `create_curriculum_integration` | EVENT_ROUTER | 30s             | Active       |

### Recovery (8 daemons)

Recovery, maintenance, and data integrity.

| Daemon Type                    | Runner                                | Dependencies                            | Health Interval | Status |
| ------------------------------ | ------------------------------------- | --------------------------------------- | --------------- | ------ |
| `RECOVERY_ORCHESTRATOR`        | `create_recovery_orchestrator`        | EVENT_ROUTER, NODE_HEALTH_MONITOR       | 30s             | Active |
| `CACHE_COORDINATION`           | `create_cache_coordination`           | EVENT_ROUTER, CLUSTER_MONITOR           | 30s             | Active |
| `MAINTENANCE`                  | `create_maintenance`                  | EVENT_ROUTER                            | 30s             | Active |
| `ORPHAN_DETECTION`             | `create_orphan_detection`             | EVENT_ROUTER                            | 30s             | Active |
| `DATA_CLEANUP`                 | `create_data_cleanup`                 | EVENT_ROUTER                            | 30s             | Active |
| `DISK_SPACE_MANAGER`           | `create_disk_space_manager`           | EVENT_ROUTER                            | 30s             | Active |
| `COORDINATOR_DISK_MANAGER`     | `create_coordinator_disk_manager`     | EVENT_ROUTER, DISK_SPACE_MANAGER        | 30s             | Active |
| `INTEGRITY_CHECK`              | `create_integrity_check`              | EVENT_ROUTER                            | 3600s           | Active |
| `AVAILABILITY_RECOVERY_ENGINE` | `create_availability_recovery_engine` | EVENT_ROUTER, AVAILABILITY_NODE_MONITOR | 30s             | Active |

### Misc (7 daemons)

Miscellaneous daemons.

| Daemon Type           | Runner                       | Dependencies                     | Health Interval | Status         |
| --------------------- | ---------------------------- | -------------------------------- | --------------- | -------------- |
| `S3_BACKUP`           | `create_s3_backup`           | EVENT_ROUTER, MODEL_DISTRIBUTION | 30s             | Active         |
| `DISTILLATION`        | `create_distillation`        | EVENT_ROUTER                     | 30s             | Active         |
| `EXTERNAL_DRIVE_SYNC` | `create_external_drive_sync` | EVENT_ROUTER                     | 30s             | Active         |
| `VAST_CPU_PIPELINE`   | `create_vast_cpu_pipeline`   | EVENT_ROUTER                     | 30s             | Active         |
| `P2P_BACKEND`         | `create_p2p_backend`         | EVENT_ROUTER                     | 30s             | Active         |
| `P2P_AUTO_DEPLOY`     | `create_p2p_auto_deploy`     | EVENT_ROUTER                     | 30s             | Active         |
| `METRICS_ANALYSIS`    | `create_metrics_analysis`    | EVENT_ROUTER                     | 30s             | Active         |
| `CLUSTER_DATA_SYNC`   | `create_cluster_data_sync`   | EVENT_ROUTER                     | 30s             | **Deprecated** |

---

## Critical Daemons

These daemons are essential for cluster operation and have faster health check intervals:

```python
CRITICAL_DAEMONS = {
    DaemonType.EVENT_ROUTER,      # Core event bus
    DaemonType.DAEMON_WATCHDOG,   # Self-healing
    DaemonType.DATA_PIPELINE,     # Pipeline processing
    DaemonType.AUTO_SYNC,         # Data synchronization
    DaemonType.QUEUE_POPULATOR,   # Work queue maintenance
    DaemonType.IDLE_RESOURCE,     # GPU utilization
    DaemonType.FEEDBACK_LOOP,     # Training feedback
}
```

---

## Startup Order

Daemons are started in this order to prevent race conditions:

```python
DAEMON_STARTUP_ORDER = [
    # Core infrastructure (1-4)
    DaemonType.EVENT_ROUTER,           # 1. Event system must be first
    DaemonType.DAEMON_WATCHDOG,        # 2. Self-healing
    DaemonType.DATA_PIPELINE,          # 3. Pipeline (before sync!)
    DaemonType.FEEDBACK_LOOP,          # 4. Feedback (before sync!)

    # Sync and queue (5-10)
    DaemonType.AUTO_SYNC,              # 5. Data sync (emits events)
    DaemonType.QUEUE_POPULATOR,        # 6. Work queue
    DaemonType.WORK_QUEUE_MONITOR,     # 7. Queue visibility
    DaemonType.COORDINATOR_HEALTH_MONITOR,  # 8. Coordinator visibility
    DaemonType.IDLE_RESOURCE,          # 9. GPU utilization
    DaemonType.TRAINING_TRIGGER,       # 10. Training trigger

    # Monitoring (11-15)
    DaemonType.CLUSTER_MONITOR,        # 11. Cluster monitoring
    DaemonType.NODE_HEALTH_MONITOR,    # 12. Node health
    DaemonType.HEALTH_SERVER,          # 13. Health endpoints
    DaemonType.CLUSTER_WATCHDOG,       # 14. Cluster watchdog
    DaemonType.NODE_RECOVERY,          # 15. Node recovery

    # Quality and training (16-17)
    DaemonType.QUALITY_MONITOR,        # 16. Quality monitoring
    DaemonType.DISTILLATION,           # 17. Distillation

    # Evaluation and promotion (18-21)
    DaemonType.EVALUATION,             # 18. Model evaluation
    DaemonType.UNIFIED_PROMOTION,      # 19. Unified promotion
    DaemonType.AUTO_PROMOTION,         # 20. Auto-promotion
    DaemonType.MODEL_DISTRIBUTION,     # 21. Model distribution
]
```

---

## Deprecated Daemons (Removal: Q2 2026)

| Deprecated Daemon       | Replacement           | Migration                                                |
| ----------------------- | --------------------- | -------------------------------------------------------- |
| `SYNC_COORDINATOR`      | `AUTO_SYNC`           | Use `AutoSyncDaemon`                                     |
| `EPHEMERAL_SYNC`        | `AUTO_SYNC`           | Use `AutoSyncDaemon(strategy="ephemeral")`               |
| `HEALTH_CHECK`          | `NODE_HEALTH_MONITOR` | Use `health_check_orchestrator.py`                       |
| `NODE_HEALTH_MONITOR`   | `HEALTH_SERVER`       | Use `health_check_orchestrator.py`                       |
| `SYSTEM_HEALTH_MONITOR` | `HEALTH_SERVER`       | Use `unified_health_manager.py`                          |
| `REPLICATION_MONITOR`   | `UNIFIED_REPLICATION` | Use `unified_replication_daemon.py`                      |
| `REPLICATION_REPAIR`    | `UNIFIED_REPLICATION` | Use `unified_replication_daemon.py`                      |
| `NPZ_DISTRIBUTION`      | `MODEL_DISTRIBUTION`  | Use `unified_distribution_daemon.py` with `DataType.NPZ` |
| `LAMBDA_IDLE`           | `VAST_IDLE`           | Use `unified_idle_shutdown_daemon.py`                    |
| `VAST_IDLE`             | -                     | Use `unified_idle_shutdown_daemon.py`                    |
| `CLUSTER_DATA_SYNC`     | `AUTO_SYNC`           | Use `AutoSyncDaemon(strategy="broadcast")`               |

---

## Dependencies Graph

Key dependency relationships:

```
EVENT_ROUTER (root - no dependencies)
├── DAEMON_WATCHDOG
├── DATA_PIPELINE
│   └── AUTO_SYNC
│   └── TRAINING_NODE_WATCHER
│   └── DATA_CONSOLIDATION
│   └── NPZ_COMBINATION
├── FEEDBACK_LOOP
│   └── AUTO_SYNC
│   └── UNIFIED_DATA_PLANE
│   └── CURRICULUM_INTEGRATION
├── CLUSTER_MONITOR
│   └── CLUSTER_WATCHDOG
│   └── ADAPTIVE_RESOURCES
│   └── CACHE_COORDINATION
├── SELFPLAY_COORDINATOR
│   └── QUEUE_POPULATOR
│       └── WORK_QUEUE_MONITOR
├── EVALUATION
│   └── AUTO_PROMOTION
│       └── MODEL_DISTRIBUTION
└── JOB_SCHEDULER
    └── RESOURCE_OPTIMIZER
```

---

## Usage Examples

### Query Daemons by Category

```python
from app.coordination.daemon_registry import get_daemons_by_category

# Get all sync daemons
sync_daemons = get_daemons_by_category("sync")
print(f"Sync daemons: {[d.name for d in sync_daemons]}")

# Get all health daemons
health_daemons = get_daemons_by_category("health")
```

### Check Daemon Health

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
print(f"AUTO_SYNC health: {health}")
```

### Validate Registry at Startup

```python
from app.coordination.daemon_registry import validate_registry_or_raise

try:
    validate_registry_or_raise()
except ValueError as e:
    print(f"Registry validation failed: {e}")
```

### Get Deprecated Daemons

```python
from app.coordination.daemon_registry import get_deprecated_daemons

for daemon_type, message in get_deprecated_daemons():
    print(f"{daemon_type.name}: {message}")
```

---

## Configuration

### DaemonSpec Fields

```python
@dataclass(frozen=True)
class DaemonSpec:
    runner_name: str               # Runner function in daemon_runners.py
    depends_on: tuple[DaemonType, ...]  # Dependencies that must be running
    health_check_interval: float | None  # Custom interval (None = default 30s)
    auto_restart: bool = True      # Auto-restart on failure
    max_restarts: int = 5          # Max restart attempts
    category: str = "misc"         # Category for documentation
    deprecated: bool = False       # Deprecated flag
    deprecated_message: str = ""   # Migration guidance
```

### DaemonInfo Runtime State

```python
@dataclass
class DaemonInfo:
    daemon_type: DaemonType
    state: DaemonState             # STOPPED, STARTING, RUNNING, FAILED, etc.
    task: asyncio.Task | None
    start_time: float
    restart_count: int
    last_error: str | None
    health_check_interval: float
    auto_restart: bool
    max_restarts: int
    startup_grace_period: float = 60.0
    depends_on: list[DaemonType]
    ready_event: asyncio.Event | None
    instance: Any | None           # Daemon instance for health_check()
```

---

## Troubleshooting

### Daemon Won't Start

1. Check dependencies are running:

   ```python
   from app.coordination.daemon_types import validate_daemon_dependencies
   ok, missing = validate_daemon_dependencies(DaemonType.AUTO_SYNC, running_daemons)
   if not ok:
       print(f"Missing deps: {missing}")
   ```

2. Check for import errors:

   ```bash
   python -c "from app.coordination.daemon_runners import create_auto_sync"
   ```

3. Check startup order:
   ```python
   from app.coordination.daemon_types import get_daemon_startup_position
   pos = get_daemon_startup_position(DaemonType.AUTO_SYNC)
   print(f"AUTO_SYNC starts at position {pos}")
   ```

### Health Check Failures

1. Query health directly:

   ```python
   daemon = get_auto_sync_daemon()
   health = daemon.health_check()
   print(health.details)
   ```

2. Check for stale state:
   ```bash
   curl http://localhost:8790/health | jq '.daemons.AUTO_SYNC'
   ```

### Deprecated Daemon Warnings

If you see deprecation warnings:

```python
# Old (deprecated)
from app.coordination.replication_monitor import ReplicationMonitor

# New (active)
from app.coordination.unified_replication_daemon import UnifiedReplicationDaemon
```

---

## See Also

- `docs/EVENT_SYSTEM_REFERENCE.md` - Event types and subscriptions
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Daemon troubleshooting guide
- `docs/audits/CIRCULAR_DEPENDENCY_MAP.md` - Import dependency graph
- `app/coordination/COORDINATOR_GUIDE.md` - Coordination module documentation
