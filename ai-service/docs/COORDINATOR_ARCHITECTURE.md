# Coordinator Architecture

This document provides an overview of the coordination and distributed systems architecture in RingRift.

## Overview

The codebase has two main coordination layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER CODE                                    │
│  (scripts/unified_ai_loop.py, training loops, tournament runners)   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Orchestrators │    │ Coordinators  │    │   Services    │
│ (High-level)  │    │ (Mid-level)   │    │ (Low-level)   │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                                    │
│     (SQLite, SSH, aria2, Prometheus, Redis, NFS, etc.)              │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

### `app/coordination/` - Scheduling & Orchestration

This directory contains modules for **deciding what to do and when**:

| Module                          | Responsibility                          |
| ------------------------------- | --------------------------------------- |
| `coordinator_base.py`           | Abstract base class for coordinators    |
| `coordinator_registry.py`       | Central coordinator dispatch            |
| `sync_coordinator.py`           | **SyncScheduler** - When/what to sync   |
| `training_coordinator.py`       | Training job coordination               |
| `unified_scheduler.py`          | Unified scheduling across concerns      |
| `leadership_coordinator.py`     | Leader election for distributed systems |
| `error_recovery_coordinator.py` | Error handling and recovery             |
| `bandwidth_manager.py`          | Network bandwidth allocation            |
| `job_scheduler.py`              | Generic job scheduling                  |
| `lock_manager.py`               | Distributed lock management             |
| `dynamic_thresholds.py`         | Adaptive threshold configuration        |

### `app/distributed/` - Execution & Transport

This directory contains modules for **actually performing distributed operations**:

| Module                   | Responsibility                                 |
| ------------------------ | ---------------------------------------------- |
| `sync_coordinator.py`    | **DistributedSyncCoordinator** - Execute syncs |
| `sync_orchestrator.py`   | **SyncOrchestrator** - Unified sync facade     |
| `cluster_coordinator.py` | Cluster membership and state                   |
| `aria2_transport.py`     | aria2-based file transfers                     |
| `ssh_transport.py`       | SSH/rsync transfers                            |
| `gossip_sync.py`         | P2P gossip protocol                            |
| `circuit_breaker.py`     | Fault tolerance                                |
| `host_classification.py` | Host capability detection                      |
| `health_checks.py`       | Distributed health monitoring                  |

## Key Patterns

### 1. Orchestrator → Coordinator → Service

```python
# Orchestrator (highest level) - use this in most cases
from app.distributed.sync_orchestrator import get_sync_orchestrator
orchestrator = get_sync_orchestrator()
await orchestrator.sync_all()

# Coordinator (mid level) - for specific operations
from app.coordination.training_coordinator import get_training_coordinator
coordinator = get_training_coordinator()
can_start = coordinator.can_train(config_key)

# Service (low level) - for infrastructure operations
from app.distributed.aria2_transport import Aria2Transport
transport = Aria2Transport()
await transport.download(url, dest)
```

### 2. Coordinator Base Pattern

All coordinators inherit from `CoordinatorBase`:

```python
from app.coordination.coordinator_base import CoordinatorBase

class MyCoordinator(CoordinatorBase):
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    def get_status(self) -> dict: ...
```

### 3. Event Bus Integration

Coordinators emit events via the centralized EventBus:

```python
from app.events import EventBus, TrainingCompletedEvent

bus = EventBus.get_instance()
await bus.emit(TrainingCompletedEvent(config_key="square8_2p"))
```

## Which Module to Use?

| Task                            | Module                  | Import                                                                       |
| ------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| Sync data/models across cluster | `SyncOrchestrator`      | `from app.distributed.sync_orchestrator import get_sync_orchestrator`        |
| Check training coordination     | `TrainingCoordinator`   | `from app.coordination.training_coordinator import get_training_coordinator` |
| Schedule jobs                   | `JobScheduler`          | `from app.coordination.job_scheduler import JobScheduler`                    |
| Manage distributed locks        | `LockManager`           | `from app.coordination.lock_manager import get_lock_manager`                 |
| Leader election                 | `LeadershipCoordinator` | `from app.coordination.leadership_coordinator import LeadershipCoordinator`  |
| Health monitoring               | `HealthRegistry`        | `from app.distributed.health_registry import HealthRegistry`                 |

## Ownership Matrix

| Concern              | Owner Module           | Layer        |
| -------------------- | ---------------------- | ------------ |
| **Data Sync**        | SyncOrchestrator       | Distributed  |
| **Model Sync**       | SyncOrchestrator       | Distributed  |
| **Elo Sync**         | EloSyncManager         | Training     |
| **Training Jobs**    | TrainingCoordinator    | Coordination |
| **Evaluation**       | TournamentOrchestrator | Tournament   |
| **Host Health**      | HealthRegistry         | Distributed  |
| **Circuit Breakers** | CircuitBreaker         | Distributed  |
| **Bandwidth**        | BandwidthManager       | Coordination |
| **Locks**            | LockManager            | Coordination |
| **Leadership**       | LeadershipCoordinator  | Coordination |

## Related Documentation

- [Sync Architecture](sync_architecture.md) - Detailed sync layer documentation
- [Tournament Consolidation](TOURNAMENT_CONSOLIDATION.md) - Tournament system analysis

## Module Count by Directory

```
app/coordination/    ~61 modules (scheduling, orchestration, policies)
app/distributed/     ~34 modules (execution, transport, health)
app/training/        ~45 modules (training-specific coordination)
app/tournament/      ~15 modules (tournament-specific logic)
```

## Best Practices

1. **Use orchestrators for high-level operations** - They handle initialization, shutdown, and cross-component coordination.

2. **Use coordinators for specific concerns** - When you need fine-grained control over a single system (training, sync, etc.).

3. **Avoid using low-level transports directly** - Let orchestrators choose the appropriate transport.

4. **Emit events for cross-system communication** - Don't create direct dependencies between unrelated coordinators.

5. **Register with the coordinator registry** - For dynamic instantiation and lifecycle management.
