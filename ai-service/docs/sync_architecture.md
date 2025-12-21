# Sync Architecture

This document clarifies the sync-related modules and their responsibilities.

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SyncOrchestrator                         │
│   app/distributed/sync_orchestrator.py                      │
│   - Unified facade for ALL sync operations                  │
│   - Single entry point: data, models, Elo, registry         │
│   - Coordinates initialization, shutdown, health monitoring │
└──────────────────────────┬──────────────────────────────────┘
                           │ wraps
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│    SyncScheduler      │       │ DistributedSyncCoord. │
│ app/coordination/     │       │ app/distributed/      │
│ sync_coordinator.py   │       │ sync_coordinator.py   │
│ ───────────────────── │       │ ────────────────────  │
│ SCHEDULING layer      │       │ EXECUTION layer       │
│ - When/what to sync   │       │ - Performs syncs      │
│ - Data freshness      │       │ - aria2/SSH/P2P       │
│ - Priority scheduling │       │ - Transport selection │
│ - Recommendations     │       │ - Circuit breakers    │
└───────────────────────┘       └───────────────────────┘
```

## Which module should I use?

| Use Case                                | Module                       | Import                                                                |
| --------------------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| **Most cases**: Unified sync operations | `SyncOrchestrator`           | `from app.distributed.sync_orchestrator import get_sync_orchestrator` |
| Check when/what to sync                 | `SyncScheduler`              | `from app.coordination.sync_coordinator import get_sync_scheduler`    |
| Low-level sync execution                | `DistributedSyncCoordinator` | `from app.coordination import DistributedSyncCoordinator`             |

## Detailed Responsibilities

### SyncOrchestrator (Recommended Entry Point)

The unified facade wrapping all sync components:

```python
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.initialize()
result = await orchestrator.sync_all()
status = orchestrator.get_status()
await orchestrator.shutdown()
```

Features:

- Unified initialization and shutdown
- Coordinated sync scheduling
- Cross-component health monitoring
- Event-driven sync triggers

### SyncScheduler (Scheduling Layer)

Decides **when** and **what** to sync:

```python
from app.coordination.sync_coordinator import (
    get_sync_scheduler,
    get_cluster_data_status,
    schedule_priority_sync,
    get_sync_recommendations,
)

status = get_cluster_data_status()
recommendations = get_sync_recommendations()
await schedule_priority_sync()
```

Features:

- Data freshness tracking across all hosts
- Priority-based sync scheduling
- Bandwidth-aware transfer balancing
- Cluster-wide data state visibility

### DistributedSyncCoordinator (Execution Layer)

Performs **actual sync operations**:

```python
from app.coordination import DistributedSyncCoordinator

coordinator = DistributedSyncCoordinator.get_instance()
await coordinator.sync_training_data()
await coordinator.sync_models(model_ids=["model_v42"])
stats = await coordinator.full_cluster_sync()
```

Features:

- Multiple transport backends (aria2, SSH/rsync, P2P HTTP, Gossip)
- Automatic transport selection based on capabilities
- NFS optimization (skip sync when storage is shared)
- Circuit breaker integration

## Exports from app.coordination

Both layers are exported for convenience:

```python
from app.coordination import (
    SyncScheduler,              # Scheduling layer (app/coordination/sync_coordinator.py)
    DistributedSyncCoordinator, # Execution layer (app/distributed/sync_coordinator.py)
)
```

## Migration Notes

The naming evolved over time:

- `SyncCoordinator` in `app/distributed/` is the original execution layer
- `SyncCoordinator` in `app/coordination/` was renamed to `SyncScheduler` in Dec 2025
- `SyncOrchestrator` is the recommended unified entry point (Dec 2025)
