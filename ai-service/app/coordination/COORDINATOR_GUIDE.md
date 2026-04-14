# Coordinator/Manager/Orchestrator Guide

**Created**: December 2025
**Purpose**: Clarify class roles and consolidation path

This guide is about in-process class responsibilities. For supported trainer
canaries, operators should use `ai-service/scripts/deploy_minimal_loops.sh`
instead of treating these classes as the fleet rollout contract.

## Overview

The codebase has 125+ Coordinator/Manager/Orchestrator classes. This guide categorizes them and provides consolidation recommendations.

## Naming Conventions

| Suffix         | Purpose                                          | Lifecycle            | Example                       |
| -------------- | ------------------------------------------------ | -------------------- | ----------------------------- |
| `Orchestrator` | Complex multi-step workflows                     | Long-running process | `UnifiedTrainingOrchestrator` |
| `Coordinator`  | Communication/synchronization between components | On-demand            | `TrainingCoordinator`         |
| `Manager`      | Resource lifecycle (create/update/destroy)       | Singleton            | `CheckpointManager`           |

## Core Classes (KEEP)

These are the essential classes that should be maintained:

### Training

| Class                         | File                                   | Purpose                       |
| ----------------------------- | -------------------------------------- | ----------------------------- |
| `UnifiedTrainingOrchestrator` | `training/unified_orchestrator.py`     | Primary training orchestrator |
| `TrainingCoordinator`         | `coordination/training_coordinator.py` | Cluster-wide job coordination |
| `TrainingLifecycleManager`    | `training/lifecycle_integration.py`    | Service lifecycle             |

### Coordination Infrastructure

| Class                  | File                                    | Purpose                         |
| ---------------------- | --------------------------------------- | ------------------------------- |
| `CoordinatorBase`      | `coordination/coordinator_base.py`      | Base class for all coordinators |
| `CoordinatorRegistry`  | `coordination/coordinator_base.py`      | Registry of active coordinators |
| `OrchestratorRegistry` | `coordination/orchestrator_registry.py` | Registry with role-based access |

### Data Management

| Class                      | File                                         | Purpose                  |
| -------------------------- | -------------------------------------------- | ------------------------ |
| `TrainingDataCoordinator`  | `training/data_coordinator.py`               | Training data management |
| `DataPipelineOrchestrator` | `coordination/data_pipeline_orchestrator.py` | Data pipeline workflows  |

### Model Lifecycle

| Class                       | File                                          | Purpose                  |
| --------------------------- | --------------------------------------------- | ------------------------ |
| `ModelLifecycleCoordinator` | `coordination/model_lifecycle_coordinator.py` | Model promotion/rollback |
| `ModelVersionManager`       | `training/model_versioning.py`                | Version tracking         |

### Cluster/Distributed

| Class                   | File                                     | Purpose                   |
| ----------------------- | ---------------------------------------- | ------------------------- |
| `ClusterCoordinator`    | `distributed/cluster_coordinator.py`     | Cluster node coordination |
| `SyncCoordinator`       | `distributed/sync_coordinator.py`        | Data synchronization      |
| `LeadershipCoordinator` | `coordination/leadership_coordinator.py` | Leader election           |

### Tournament/Evaluation

| Class                    | File                         | Purpose              |
| ------------------------ | ---------------------------- | -------------------- |
| `TournamentOrchestrator` | `tournament/orchestrator.py` | Tournament execution |
| `TournamentRunner`       | `tournament/runner.py`       | Match execution      |

## Deprecated Classes (PHASE OUT)

These classes have been superseded:

| Class                       | File                                                   | Replacement                   |
| --------------------------- | ------------------------------------------------------ | ----------------------------- |
| `TrainingOrchestrator`      | `archive/deprecated_training/orchestrated_training.py` | `UnifiedTrainingOrchestrator` |
| `IntegratedTrainingManager` | `training/integrated_enhancements.py`                  | `UnifiedTrainingOrchestrator` |
| `P2PTrainingBridge`\*       | `integration/p2p_integration.py`                       | Use for P2P only              |
| `P2PSelfplayBridge`\*       | `integration/p2p_integration.py`                       | Use for P2P only              |
| `P2PEvaluationBridge`\*     | `integration/p2p_integration.py`                       | Use for P2P only              |

*Not deprecated, but renamed from `*Coordinator` to avoid confusion.

## Specialized Classes (KEEP but scope-limited)

These are specialized and should remain but with clear scope:

### Health Monitoring

- `UnifiedHealthManager` - Primary health orchestrator
- `NodeHealthOrchestrator` - Node-specific health
- `ResourceMonitoringCoordinator` - Resource tracking

### Caching/Optimization

- `CacheCoordinationOrchestrator` - Cache management
- `BandwidthManager` - Network bandwidth
- `AdaptiveResourceManager` - Dynamic resource allocation

### Selfplay

- `SelfplayOrchestrator` - Selfplay coordination
- `GPUMCTSSelfplayRunner` - GPU selfplay execution

## Mixin Usage Guide

```python
# Basic coordinator with SQLite persistence
class MyCoordinator(CoordinatorBase, SQLitePersistenceMixin):
    def __init__(self, db_path: Path):
        super().__init__()
        self.init_db(db_path)

# Preferred singleton pattern
from app.coordination.singleton_mixin import singleton

@singleton
class GlobalManager(CoordinatorBase):
    pass

# Event-driven coordinator
class MyMonitor(CoordinatorBase, EventDrivenMonitorMixin):
    def __init__(self):
        super().__init__()
        self._handlers = {}

# Resilient coordinator with error handling
class RobustOrchestrator(CoordinatorBase, ResilientCoordinatorMixin):
    pass
```

## Available Mixins

| Mixin                       | Purpose                      | Use When                                         |
| --------------------------- | ---------------------------- | ------------------------------------------------ |
| `SQLitePersistenceMixin`    | SQLite database persistence  | Need persistent state                            |
| `StatePersistenceMixin`     | Extended state persistence   | Need checkpoint/restore                          |
| `SingletonMixin`            | Legacy single-instance mixin | Existing compatibility only; prefer `@singleton` |
| `CallbackMixin`             | Callback registration        | Event-based coordination                         |
| `EventDrivenMonitorMixin`   | Event-driven monitoring      | Continuous monitoring                            |
| `ResilientCoordinatorMixin` | Error handling/retry         | Network operations                               |

Prefer the `@singleton` decorator for new single-instance classes. Keep
`SingletonMixin` only when you are extending older coordination classes that
already depend on its behavior.

## Consolidation Recommendations

### High Priority

1. **Merge health orchestrators** → Single `HealthOrchestrator`
   - `UnifiedHealthOrchestrator`
   - `NodeHealthOrchestrator`
   - `UnifiedHealthManager`

2. **Merge sync coordinators** → Single `SyncCoordinator`
   - `SyncOrchestrator`
   - `SyncCoordinator`
   - `SyncScheduler`

### Medium Priority

3. **Simplify tournament** → `TournamentOrchestrator` only
   - Remove `TournamentRunner` (inline into orchestrator)
   - Remove `TournamentScheduleManager` (inline)

4. **Simplify checkpoint** → `UnifiedCheckpointManager` only
   - Remove `SmartCheckpointManager`
   - Remove `_LegacyCheckpointManager`

### Low Priority

5. **Document remaining 50+ specialized classes**
   - Many are domain-specific and justified
   - Focus on removing truly redundant ones

## Anti-Patterns to Avoid

1. **Don't create new `*Coordinator` for simple tasks**
   - Use functions for stateless operations
   - Use existing coordinators with callbacks

2. **Don't duplicate coordination patterns**
   - Check if similar class exists
   - Extend existing rather than create new

3. **Don't mix responsibilities**
   - Coordinators: communication
   - Managers: lifecycle
   - Orchestrators: workflows

## Migration Checklist

When consolidating classes:

- [ ] Identify all usages of deprecated class
- [ ] Add deprecation warning to class docstring
- [ ] Create migration guide in docstring
- [ ] Add backward compatibility alias if needed
- [ ] Update imports in dependent files
- [ ] Run tests to verify compatibility
- [ ] Schedule removal after grace period
