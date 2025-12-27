# Cluster Integration Guide

**Last Updated:** 2025-12-27
**Status:** Current (reflects December 2025 consolidation)

---

## Overview

This guide documents how RingRift's cluster components integrate together. After the December 2025 consolidation, the architecture follows a modular pattern with clear separation of concerns.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    P2P Orchestrator (26.5K LOC)                 │
│                   scripts/p2p_orchestrator.py                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ LoopManager │  │ EventRouter │  │ DaemonManager (62)      │ │
│  │ (6 loops)   │  │ (unified)   │  │ Lifecycle management    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    P2P Managers (scripts/p2p/managers/)         │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐ │
│  │ StateManager  │ │ JobManager    │ │ SelfplayScheduler     │ │
│  │ 629 LOC       │ │ 663 LOC       │ │ 737 LOC               │ │
│  │ - SQLite      │ │ - Spawning    │ │ - Config priority     │ │
│  │ - Epochs      │ │ - Lifecycle   │ │ - Curriculum weights  │ │
│  └───────────────┘ └───────────────┘ └───────────────────────┘ │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐ │
│  │ SyncPlanner   │ │ NodeSelector  │ │ TrainingCoordinator   │ │
│  │ 704 LOC       │ │ 330 LOC       │ │ 734 LOC               │ │
│  │ - Manifests   │ │ - Ranking     │ │ - Job dispatch        │ │
│  │ - Sync plan   │ │ - Selection   │ │ - Model promotion     │ │
│  └───────────────┘ └───────────────┘ └───────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     Core Modules                                │
│                                                                 │
│  ┌───────────────────┐  ┌─────────────────────────────────────┐│
│  │ app/core/         │  │ app/coordination/                   ││
│  │ - ssh.py          │  │ - event_router.py (unified events)  ││
│  │ - node.py         │  │ - daemon_manager.py (66 daemons)    ││
│  │                   │  │ - auto_sync_daemon.py               ││
│  └───────────────────┘  │ - sync_facade.py                    ││
│                         │ - unified_distribution_daemon.py    ││
│                         └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## P2P Manager Architecture

The P2P orchestrator delegates to 6 specialized managers:

| Manager               | Responsibility                     | Key Methods                                                            |
| --------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| `StateManager`        | SQLite persistence, cluster epochs | `save_state()`, `load_state()`, `get_cluster_epoch()`                  |
| `JobManager`          | Job spawning and lifecycle         | `run_gpu_selfplay_job()`, `run_training()`, `cleanup_completed_jobs()` |
| `SelfplayScheduler`   | Config priority, curriculum        | `pick_weighted_config()`, `get_hybrid_job_targets()`                   |
| `SyncPlanner`         | Manifest collection, sync plans    | `generate_sync_plan()`, `execute_sync_plan()`                          |
| `NodeSelector`        | Node ranking and selection         | `select_best_node()`, `rank_nodes()`                                   |
| `TrainingCoordinator` | Training dispatch, promotion       | `dispatch_training_job()`, `handle_training_completion()`              |

### Using Managers

```python
# Managers are accessed through the orchestrator instance
self.job_manager.run_gpu_selfplay_job(node, config)
self.selfplay_scheduler.pick_weighted_config(node)
self.training_coordinator.dispatch_training_job(config_key, node)
self.sync_planner.generate_sync_plan(cluster_manifest)
```

---

## Event System

All events flow through the unified `EventRouter`:

```python
from app.coordination.event_router import get_event_router

router = get_event_router()

# Subscribe to events
router.subscribe(DataEventType.SELFPLAY_COMPLETE, my_handler)

# Publish events
router.publish(DataEventType.TRAINING_STARTED, payload)
```

### Key Event Types

| Event                 | Emitted By          | Consumed By                             |
| --------------------- | ------------------- | --------------------------------------- |
| `SELFPLAY_COMPLETE`   | GPU selfplay jobs   | AutoSyncDaemon, TrainingCoordinator     |
| `TRAINING_COMPLETED`  | Training scripts    | AutoEvaluationDaemon, ModelDistribution |
| `MODEL_PROMOTED`      | PromotionController | UnifiedDistributionDaemon               |
| `DATA_SYNC_COMPLETED` | SyncCoordinator     | TrainingFreshness, PipelineOrchestrator |

---

## Sync Infrastructure

### Sync Flow

```
┌───────────────┐     ┌────────────────┐     ┌─────────────────┐
│ SyncFacade    │────▶│ AutoSyncDaemon │────▶│ SyncCoordinator │
│ (entry point) │     │ (scheduling)   │     │ (execution)     │
└───────────────┘     └────────────────┘     └─────────────────┘
                              │
                              ▼
                      ┌────────────────┐
                      │ SyncRouter     │
                      │ (node select)  │
                      └────────────────┘
```

### Programmatic Sync

```python
from app.coordination.sync_facade import SyncFacade

facade = SyncFacade()
result = await facade.sync_games(config_key="hex8_2p")
result = await facade.sync_models(model_path="models/canonical_hex8_2p.pth")
```

### Strategies

| Strategy    | Use Case                              |
| ----------- | ------------------------------------- |
| `HYBRID`    | Default: push-from-generator + gossip |
| `EPHEMERAL` | Aggressive 5s sync for Vast.ai        |
| `BROADCAST` | Leader pushes to all nodes            |
| `AUTO`      | Auto-detect based on node type        |

---

## Daemon Management

### Unified Daemons (December 2025)

| Unified Daemon              | Replaces                                 | Savings    |
| --------------------------- | ---------------------------------------- | ---------- |
| `UnifiedDistributionDaemon` | model_distribution + npz_distribution    | ~1,100 LOC |
| `UnifiedIdleShutdownDaemon` | lambda_idle + vast_idle                  | ~318 LOC   |
| `UnifiedReplicationDaemon`  | replication_monitor + repair             | ~500 LOC   |
| `UnifiedQueuePopulator`     | queue_populator + queue_populator_daemon | ~1,158 LOC |

### Daemon Lifecycle

```python
from app.coordination.daemon_manager import get_daemon_manager, DaemonType

manager = get_daemon_manager()

# Start all daemons
await manager.start_all()

# Start specific daemon
await manager.start(DaemonType.AUTO_SYNC)

# Check status
status = manager.get_daemon_status(DaemonType.AUTO_SYNC)
```

---

## Background Loops

The LoopManager runs 6 extracted loops:

| Loop                 | Interval | Purpose                          |
| -------------------- | -------- | -------------------------------- |
| `EloSyncLoop`        | 60s      | Sync Elo ratings across cluster  |
| `IdleDetectionLoop`  | 30s      | Detect and reclaim idle GPUs     |
| `AutoScalingLoop`    | 120s     | Scale nodes based on queue depth |
| `JobReaperLoop`      | 60s      | Clean up stale/completed jobs    |
| `QueuePopulatorLoop` | 30s      | Maintain work queue targets      |
| `ValidationLoop`     | 300s     | Queue validation for new models  |

### Feature Flag

```bash
# Enable extracted loops (default: true)
export RINGRIFT_EXTRACTED_LOOPS=true
```

---

## Common Integration Patterns

### 1. Adding Event Handler

```python
# In your daemon/coordinator:
from app.distributed.data_events import DataEventType, get_data_event_bus

bus = get_data_event_bus()
bus.subscribe(DataEventType.YOUR_EVENT, self._on_your_event)

async def _on_your_event(self, event):
    payload = event.payload
    # Handle event
```

### 2. Triggering Sync on Completion

```python
# After selfplay completion
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon.get_instance()
await daemon.trigger_urgent_sync(config_key)
```

### 3. Checking Node Health

```python
from app.coordination.health_check_orchestrator import get_health_check_orchestrator

hco = get_health_check_orchestrator()
health = await hco.check_node_health(node_id)
```

---

## Configuration

### Key Environment Variables

| Variable                   | Default  | Description              |
| -------------------------- | -------- | ------------------------ |
| `RINGRIFT_NODE_ID`         | hostname | Unique node identifier   |
| `RINGRIFT_IS_COORDINATOR`  | false    | Run as coordinator       |
| `RINGRIFT_P2P_PORT`        | 8770     | P2P orchestrator port    |
| `RINGRIFT_EXTRACTED_LOOPS` | true     | Use LoopManager          |
| `RINGRIFT_SWIM_ENABLED`    | false    | SWIM membership protocol |
| `RINGRIFT_RAFT_ENABLED`    | false    | Raft consensus           |

### Cluster Config

See `config/distributed_hosts.yaml` for full node configuration.

---

## Deprecated Modules

These modules have been superseded and emit deprecation warnings:

| Deprecated                                      | Replacement                       |
| ----------------------------------------------- | --------------------------------- |
| `app/coordination/sync_coordinator.py`          | `SyncFacade` + `AutoSyncDaemon`   |
| `app/coordination/node_health_monitor.py`       | `HealthCheckOrchestrator`         |
| `app/coordination/model_distribution_daemon.py` | `UnifiedDistributionDaemon`       |
| `app/coordination/npz_distribution_daemon.py`   | `UnifiedDistributionDaemon`       |
| `app/coordination/cluster/p2p.py`               | `app/coordination/p2p_backend.py` |

---

## See Also

- `docs/CLUSTER_TRAINING_GUIDE.md` - Training workflows
- `docs/COORDINATION_ARCHITECTURE.md` - Event system details
- `docs/DAEMON_REGISTRY.md` - Full daemon catalog
- `scripts/p2p/managers/README.md` - Manager architecture
