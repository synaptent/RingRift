# Daemon Index

This document catalogs all daemons managed by the `DaemonManager` in the RingRift AI training pipeline.

## Overview

The `DaemonManager` (`app/coordination/daemon_manager.py`) provides lifecycle management for 40+ background daemons. Daemons are organized by function and can declare dependencies on other daemons.

## Daemon Categories

### Core Infrastructure

| Daemon               | Type                   | Dependencies | Description                                             |
| -------------------- | ---------------------- | ------------ | ------------------------------------------------------- |
| Event Router         | `EVENT_ROUTER`         | None         | Central event bus for all inter-component communication |
| Cross-Process Poller | `CROSS_PROCESS_POLLER` | None         | Polls cross-process events from persistence             |
| DLQ Retry            | `DLQ_RETRY`            | EVENT_ROUTER | Retries failed events from dead-letter queue            |
| Daemon Watchdog      | `DAEMON_WATCHDOG`      | EVENT_ROUTER | Monitors daemon health, restarts stuck daemons          |
| Health Server        | `HEALTH_SERVER`        | None         | HTTP endpoints for liveness/readiness probes            |

### Data Synchronization

| Daemon              | Type                  | Dependencies | Description                                    |
| ------------------- | --------------------- | ------------ | ---------------------------------------------- |
| Sync Coordinator    | `SYNC_COORDINATOR`    | None         | _Deprecated_ - Use AUTO_SYNC instead           |
| Auto Sync           | `AUTO_SYNC`           | EVENT_ROUTER | Automated P2P data sync with push + gossip     |
| High Quality Sync   | `HIGH_QUALITY_SYNC`   | None         | Priority sync for high-quality game data       |
| Elo Sync            | `ELO_SYNC`            | None         | Syncs Elo ratings across cluster               |
| Gossip Sync         | `GOSSIP_SYNC`         | None         | Gossip protocol for data propagation           |
| Ephemeral Sync      | `EPHEMERAL_SYNC`      | EVENT_ROUTER | Aggressive 5s sync for Vast.ai ephemeral hosts |
| Cluster Data Sync   | `CLUSTER_DATA_SYNC`   | EVENT_ROUTER | Full cluster-wide synchronization              |
| NPZ Distribution    | `NPZ_DISTRIBUTION`    | EVENT_ROUTER | Distributes NPZ training data after export     |
| Model Sync          | `MODEL_SYNC`          | EVENT_ROUTER | Syncs model files across nodes                 |
| Model Distribution  | `MODEL_DISTRIBUTION`  | EVENT_ROUTER | Distributes promoted models to cluster         |
| External Drive Sync | `EXTERNAL_DRIVE_SYNC` | None         | Backup to external drives (stub)               |

### Training Pipeline

| Daemon                   | Type                       | Dependencies              | Description                                    |
| ------------------------ | -------------------------- | ------------------------- | ---------------------------------------------- |
| Continuous Training Loop | `CONTINUOUS_TRAINING_LOOP` | EVENT_ROUTER              | Main training orchestration loop               |
| Training Trigger         | `TRAINING_TRIGGER`         | EVENT_ROUTER, AUTO_EXPORT | Triggers training when data threshold reached  |
| Auto Export              | `AUTO_EXPORT`              | EVENT_ROUTER              | Automatically exports NPZ when games available |
| Data Pipeline            | `DATA_PIPELINE`            | EVENT_ROUTER              | Orchestrates pipeline stages                   |
| Selfplay Coordinator     | `SELFPLAY_COORDINATOR`     | EVENT_ROUTER              | Coordinates selfplay across cluster            |
| Curriculum Integration   | `CURRICULUM_INTEGRATION`   | EVENT_ROUTER              | Bridges feedback loops for curriculum          |

### Evaluation & Promotion

| Daemon                     | Type                         | Dependencies             | Description                                |
| -------------------------- | ---------------------------- | ------------------------ | ------------------------------------------ |
| Evaluation                 | `EVALUATION`                 | EVENT_ROUTER             | Runs gauntlet evaluations                  |
| Tournament Daemon          | `TOURNAMENT_DAEMON`          | EVENT_ROUTER             | Runs model tournaments for Elo updates     |
| Auto Promotion             | `AUTO_PROMOTION`             | EVENT_ROUTER, EVALUATION | Promotes models passing gauntlet           |
| Unified Promotion          | `UNIFIED_PROMOTION`          | EVENT_ROUTER             | Unified promotion controller               |
| Gauntlet Feedback          | `GAUNTLET_FEEDBACK`          | EVENT_ROUTER             | Adjusts training based on gauntlet results |
| Model Performance Watchdog | `MODEL_PERFORMANCE_WATCHDOG` | EVENT_ROUTER             | Monitors model win rates                   |
| Distillation               | `DISTILLATION`               | EVENT_ROUTER             | Distills large models to NNUE format       |

### Cluster Management

| Daemon                | Type                    | Dependencies                      | Description                         |
| --------------------- | ----------------------- | --------------------------------- | ----------------------------------- |
| P2P Backend           | `P2P_BACKEND`           | EVENT_ROUTER                      | P2P mesh network backend            |
| P2P Auto Deploy       | `P2P_AUTO_DEPLOY`       | EVENT_ROUTER                      | Ensures P2P runs on all nodes       |
| Cluster Monitor       | `CLUSTER_MONITOR`       | EVENT_ROUTER                      | Real-time cluster status monitoring |
| Cluster Watchdog      | `CLUSTER_WATCHDOG`      | None                              | Watchdog for cluster-wide issues    |
| Node Health Monitor   | `NODE_HEALTH_MONITOR`   | EVENT_ROUTER                      | Unified node health monitoring      |
| Node Recovery         | `NODE_RECOVERY`         | EVENT_ROUTER                      | Auto-recovers terminated nodes      |
| System Health Monitor | `SYSTEM_HEALTH_MONITOR` | EVENT_ROUTER, NODE_HEALTH_MONITOR | System-wide health scoring          |

### Quality & Feedback

| Daemon                | Type                    | Dependencies | Description                            |
| --------------------- | ----------------------- | ------------ | -------------------------------------- |
| Quality Monitor       | `QUALITY_MONITOR`       | EVENT_ROUTER | Continuous data quality monitoring     |
| Feedback Loop         | `FEEDBACK_LOOP`         | EVENT_ROUTER | Training feedback signals              |
| Training Node Watcher | `TRAINING_NODE_WATCHER` | EVENT_ROUTER | Detects active training, triggers sync |
| Metrics Analysis      | `METRICS_ANALYSIS`      | EVENT_ROUTER | Analyzes training metrics trends       |

### Resource Management

| Daemon                | Type                    | Dependencies                       | Description                                 |
| --------------------- | ----------------------- | ---------------------------------- | ------------------------------------------- |
| Queue Monitor         | `QUEUE_MONITOR`         | None                               | Monitors queue depths, applies backpressure |
| Queue Populator       | `QUEUE_POPULATOR`       | EVENT_ROUTER, SELFPLAY_COORDINATOR | Maintains work queue until Elo targets met  |
| Job Scheduler         | `JOB_SCHEDULER`         | EVENT_ROUTER                       | PID-based resource allocation               |
| Idle Resource         | `IDLE_RESOURCE`         | EVENT_ROUTER                       | Monitors idle GPUs, spawns selfplay         |
| Utilization Optimizer | `UTILIZATION_OPTIMIZER` | EVENT_ROUTER, IDLE_RESOURCE        | Matches GPU capabilities to board sizes     |
| Adaptive Resources    | `ADAPTIVE_RESOURCES`    | EVENT_ROUTER, CLUSTER_MONITOR      | Adapts resources based on training phase    |
| Lambda Idle           | `LAMBDA_IDLE`           | EVENT_ROUTER, CLUSTER_MONITOR      | Lambda idle detection and shutdown          |
| Vast Idle             | `VAST_IDLE`             | EVENT_ROUTER, CLUSTER_MONITOR      | Vast.ai idle detection and shutdown         |
| Vast CPU Pipeline     | `VAST_CPU_PIPELINE`     | None                               | CPU-only jobs on Vast.ai (stub)             |

### Data Integrity

| Daemon              | Type                  | Dependencies | Description                            |
| ------------------- | --------------------- | ------------ | -------------------------------------- |
| Replication Monitor | `REPLICATION_MONITOR` | EVENT_ROUTER | Monitors data replication health       |
| Replication Repair  | `REPLICATION_REPAIR`  | EVENT_ROUTER | Repairs under-replicated data          |
| Orphan Detection    | `ORPHAN_DETECTION`    | EVENT_ROUTER | Detects unregistered game databases    |
| Data Cleanup        | `DATA_CLEANUP`        | EVENT_ROUTER | Auto-quarantine poor quality data      |
| Health Check        | `HEALTH_CHECK`        | None         | _Deprecated_ - Use NODE_HEALTH_MONITOR |

### Backup & Recovery

| Daemon                | Type                    | Dependencies                      | Description                       |
| --------------------- | ----------------------- | --------------------------------- | --------------------------------- |
| S3 Backup             | `S3_BACKUP`             | EVENT_ROUTER, MODEL_DISTRIBUTION  | Backs up models to S3             |
| Recovery Orchestrator | `RECOVERY_ORCHESTRATOR` | EVENT_ROUTER, NODE_HEALTH_MONITOR | Orchestrates recovery procedures  |
| Cache Coordination    | `CACHE_COORDINATION`    | EVENT_ROUTER, CLUSTER_MONITOR     | Coordinates caches across cluster |
| Maintenance           | `MAINTENANCE`           | None                              | Scheduled maintenance tasks       |

### Multi-Provider

| Daemon         | Type             | Dependencies                  | Description                        |
| -------------- | ---------------- | ----------------------------- | ---------------------------------- |
| Multi Provider | `MULTI_PROVIDER` | EVENT_ROUTER, CLUSTER_MONITOR | Manages multi-cloud provider nodes |
| Data Server    | `DATA_SERVER`    | None                          | HTTP server for data transfers     |

## Daemon Lifecycle

### States

- **PENDING**: Registered but not started
- **STARTING**: Initialization in progress
- **RUNNING**: Active and healthy
- **STOPPING**: Graceful shutdown in progress
- **STOPPED**: Cleanly stopped
- **FAILED**: Crashed or error state
- **RESTARTING**: Restart in progress

### Dependency Resolution

Daemons with `depends_on` will wait for dependencies to reach RUNNING state before starting. The `EVENT_ROUTER` is the most common dependency as most daemons emit events.

### Starting Daemons

```python
from app.coordination.daemon_manager import DaemonManager, DaemonType

manager = DaemonManager()

# Start specific daemon
await manager.start(DaemonType.AUTO_SYNC)

# Start all daemons (respects dependencies)
await manager.start_all()

# Start recommended production set
await manager.start_production_daemons()
```

### Monitoring

```bash
# Check daemon status
python scripts/launch_daemons.py --status

# Start all daemons
python scripts/launch_daemons.py --all

# Start specific daemons
python scripts/launch_daemons.py --daemons AUTO_SYNC,CLUSTER_MONITOR
```

## Event Integration

All daemons with `EVENT_ROUTER` dependency emit events via the unified event system:

```python
from app.coordination.event_router import get_router, DataEventType

router = get_router()
await router.publish(DataEventType.DATA_SYNC_COMPLETED, {...})
```

See `docs/coordination/EVENT_CATALOG.md` for the complete list of event types.

## Adding New Daemons

1. Create daemon class with `async def start()` and `async def stop()` methods
2. Add `DaemonType` enum value
3. Register factory in `DaemonManager._register_all_factories()`
4. Declare dependencies if daemon emits events
5. Add to this index

Example:

```python
# In daemon_manager.py
self.register_factory(
    DaemonType.MY_DAEMON,
    self._create_my_daemon,
    depends_on=[DaemonType.EVENT_ROUTER],
)

async def _create_my_daemon(self) -> None:
    from app.coordination.my_daemon import MyDaemon
    daemon = MyDaemon()
    await daemon.start()
```

## Deprecated Daemons

| Daemon             | Replacement           | Removal |
| ------------------ | --------------------- | ------- |
| `SYNC_COORDINATOR` | `AUTO_SYNC`           | Q2 2026 |
| `HEALTH_CHECK`     | `NODE_HEALTH_MONITOR` | Q2 2026 |

## Key Files

| File                                            | Purpose                           |
| ----------------------------------------------- | --------------------------------- |
| `app/coordination/daemon_manager.py`            | Main DaemonManager implementation |
| `app/coordination/daemon_adapters.py`           | Adapters for existing daemons     |
| `scripts/launch_daemons.py`                     | CLI for daemon management         |
| `docs/coordination/EVENT_CATALOG.md`            | Event types reference             |
| `docs/coordination/RESILIENT_TRANSFER_GUIDE.md` | Transfer daemon guide             |
