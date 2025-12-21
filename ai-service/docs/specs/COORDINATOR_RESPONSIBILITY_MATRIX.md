# Coordinator Responsibility Matrix

**Created:** December 2025
**Purpose:** Document coordinator class responsibilities and prevent overlap
**Status:** ACTIVE - Reference for architecture decisions

---

## Overview

The RingRift AI service uses multiple coordinator classes to manage different aspects
of the training pipeline. This document clarifies responsibilities, prevents duplication,
and guides future consolidation.

---

## 1. Core Coordinator Infrastructure

| Class                        | Location                                       | Responsibility                                                                    |
| ---------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `CoordinatorBase`            | `app/coordination/coordinator_base.py`         | Abstract base for all coordinators. Provides lifecycle, stats, SQLite persistence |
| `CoordinatorProtocol`        | `app/coordination/coordinator_base.py`         | Protocol for structural typing of coordinators                                    |
| `CoordinatorRegistry`        | `app/coordination/coordinator_base.py`         | Registry tracking all coordinator instances                                       |
| `CoordinatorDependencyGraph` | `app/coordination/coordinator_dependencies.py` | Tracks dependencies between coordinators                                          |
| `CoordinatorConfig`          | `app/coordination/coordinator_config.py`       | Centralized configuration for coordinators                                        |

---

## 2. Event Coordination

| Class                     | Location                                        | Responsibility                                              |
| ------------------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| `UnifiedEventCoordinator` | `app/coordination/unified_event_coordinator.py` | Bridges DataEventBus, StageEventBus, CrossProcessEventQueue |
| `EventRouter`             | `app/coordination/event_router.py`              | Routes events between bus types                             |

**Single Source of Truth:** `UnifiedEventCoordinator` is the canonical event bridge.

---

## 3. Task Management

| Class                      | Location                                         | Responsibility                                      |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| `TaskCoordinator`          | `app/coordination/task_coordinator.py`           | Queue-based task execution with parallelism control |
| `TaskLifecycleCoordinator` | `app/coordination/task_lifecycle_coordinator.py` | Tracks task state transitions and history           |

**Boundary:** `TaskCoordinator` handles execution, `TaskLifecycleCoordinator` handles state/history.

---

## 4. Model Lifecycle

| Class                       | Location                                          | Responsibility                                          |
| --------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| `ModelLifecycleCoordinator` | `app/coordination/model_lifecycle_coordinator.py` | Central model state machine (dev→staging→prod→archived) |
| `ModelSyncCoordinator`      | `app/integration/model_lifecycle.py`              | Syncs model state across cluster nodes                  |

**Boundary:** `ModelLifecycleCoordinator` owns state, `ModelSyncCoordinator` handles distribution.

---

## 5. Training Coordination

| Class                     | Location                                   | Responsibility                 | Status        |
| ------------------------- | ------------------------------------------ | ------------------------------ | ------------- |
| `TrainingCoordinator`     | `app/coordination/training_coordinator.py` | Orchestrates training workflow | **CANONICAL** |
| `TrainingCoordinator`     | `app/integration/p2p_integration.py`       | P2P training coordination      | DEPRECATED    |
| `TrainingDataCoordinator` | `app/training/data_coordinator.py`         | Manages training data loading  | Active        |

**CONSOLIDATION NEEDED:** The P2P `TrainingCoordinator` should be merged into canonical one.

---

## 6. Cluster & Distribution

| Class                   | Location                                     | Responsibility                              |
| ----------------------- | -------------------------------------------- | ------------------------------------------- |
| `ClusterCoordinator`    | `app/distributed/cluster_coordinator.py`     | Cluster membership, leader election, health |
| `SyncCoordinator`       | `app/distributed/sync_coordinator.py`        | Data synchronization between nodes          |
| `LeadershipCoordinator` | `app/coordination/leadership_coordinator.py` | Leader election protocol                    |

**Boundary:**

- `ClusterCoordinator`: Cluster membership and topology
- `LeadershipCoordinator`: Election logic
- `SyncCoordinator`: Data replication

---

## 7. Selfplay & Evaluation

| Class                   | Location                             | Responsibility                      |
| ----------------------- | ------------------------------------ | ----------------------------------- |
| `SelfplayCoordinator`   | `app/integration/p2p_integration.py` | Coordinates selfplay across workers |
| `EvaluationCoordinator` | `app/integration/p2p_integration.py` | Coordinates evaluation matches      |

---

## 8. Optimization

| Class                     | Location                                       | Responsibility                     |
| ------------------------- | ---------------------------------------------- | ---------------------------------- |
| `OptimizationCoordinator` | `app/coordination/optimization_coordinator.py` | CMA-ES, NAS, PBT optimization runs |

---

## 9. Error & Recovery

| Class                       | Location                                         | Responsibility                            |
| --------------------------- | ------------------------------------------------ | ----------------------------------------- |
| `ErrorRecoveryCoordinator`  | `app/coordination/error_recovery_coordinator.py` | Error classification, recovery strategies |
| `ResilientCoordinatorMixin` | `app/coordination/handler_resilience.py`         | Adds retry/fallback to coordinators       |

---

## 10. Resource & Health

| Class                           | Location                                              | Responsibility                               |
| ------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| `ResourceMonitoringCoordinator` | `app/coordination/resource_monitoring_coordinator.py` | GPU/CPU/memory monitoring                    |
| `HealthRegistry`                | `app/distributed/health_registry.py`                  | Aggregates health checks (not a coordinator) |

---

## 11. Persistence & Snapshots

| Class                   | Location                                      | Responsibility                            |
| ----------------------- | --------------------------------------------- | ----------------------------------------- |
| `SnapshotCoordinator`   | `app/coordination/coordinator_persistence.py` | Periodic state snapshots for recovery     |
| `StatePersistenceMixin` | `app/coordination/coordinator_persistence.py` | SQLite state persistence for coordinators |

---

## Interaction Rules

### 1. Event Flow

```
StageEventBus → UnifiedEventCoordinator → DataEventBus
                        ↓
               CrossProcessEventQueue
```

### 2. Training Pipeline

```
SelfplayCoordinator → TrainingDataCoordinator → TrainingCoordinator → ModelLifecycleCoordinator
                                                       ↓
                                              EvaluationCoordinator
```

### 3. Cluster Coordination

```
ClusterCoordinator (membership) → LeadershipCoordinator (election) → SyncCoordinator (replication)
```

---

## Anti-Patterns to Avoid

1. **Multiple coordinators writing same state** - Use single source of truth
2. **Bypassing event coordinator** - Always route cross-process events through UnifiedEventCoordinator
3. **Duplicate coordinator classes** - Merge functionality into canonical coordinator
4. **Direct inter-coordinator calls** - Prefer event-based communication

---

## Consolidation Roadmap

### Completed

- [x] Event mappings centralized in `event_mappings.py`
- [x] `CoordinatorBase` provides unified lifecycle

### In Progress

- [ ] Merge `integration/p2p_integration.py:TrainingCoordinator` into canonical

### Future

- [ ] Create coordinator composition patterns (vs deep inheritance)
- [ ] Add coordinator dependency validation on startup
- [ ] Implement coordinator health dashboard

---

## Adding New Coordinators

Before creating a new coordinator:

1. **Check this matrix** - Ensure no existing coordinator handles this
2. **Extend CoordinatorBase** - Use the common base class
3. **Register dependencies** - Add to `CoordinatorDependencyGraph`
4. **Document here** - Add entry to appropriate section
5. **Use events** - Communicate via UnifiedEventCoordinator

---

## References

- `docs/CONSOLIDATION_ROADMAP.md` - Overall consolidation plan
- `app/coordination/coordinator_base.py` - Base class implementation
- `app/coordination/event_mappings.py` - Event type mappings
