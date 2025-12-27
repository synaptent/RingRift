# ADR-007: P2P Orchestrator Decomposition

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift AI Team

## Context

The P2P orchestrator (`scripts/p2p_orchestrator.py`) grew to 27,000+ lines with 100+ methods handling:

- Cluster membership and leader election
- Job scheduling and dispatch
- Training coordination
- Data sync planning
- Selfplay resource allocation
- Node selection and scoring

This monolithic design caused:

- Difficult testing (too many interdependencies)
- Hard to understand code flow
- High cognitive load for modifications
- Circular dependencies within the file

## Decision

Decompose the P2P orchestrator into **6 focused manager classes** at `scripts/p2p/managers/`:

### StateManager (~630 LOC)

**Responsibility**: SQLite persistence and cluster state

```python
from scripts.p2p.managers import StateManager

state_mgr = StateManager(db_path, node_id)
state_mgr.increment_cluster_epoch()
cluster_stats = state_mgr.get_cluster_stats()
```

Methods: `persist_state()`, `load_state()`, `get_cluster_stats()`, `increment_cluster_epoch()`

### NodeSelector (~330 LOC)

**Responsibility**: Node ranking and capability matching

```python
from scripts.p2p.managers import NodeSelector

selector = NodeSelector(alive_peers)
best_node = selector.select_best_node_for_job(job_type="training", requirements={"gpu_memory": 40000})
```

Methods: `select_best_node_for_job()`, `get_node_score()`, `filter_by_capability()`

### SyncPlanner (~700 LOC)

**Responsibility**: Manifest collection and sync planning

```python
from scripts.p2p.managers import SyncPlanner

planner = SyncPlanner(cluster_manifest, sync_router)
plan = planner.create_sync_plan(source_nodes, target_nodes)
```

Methods: `collect_manifests()`, `create_sync_plan()`, `execute_sync_plan()`, `get_sync_targets()`

### JobManager (~660 LOC)

**Responsibility**: Job lifecycle and dispatch

```python
from scripts.p2p.managers import JobManager

job_mgr = JobManager(state_manager, node_selector)
job_id = await job_mgr.spawn_job(node, command, job_type)
await job_mgr.cleanup_old_completed_jobs()
```

Methods: `spawn_job()`, `run_distributed_selfplay()`, `run_training()`, `export_training_data()`

### SelfplayScheduler (~740 LOC)

**Responsibility**: Priority-based selfplay allocation

```python
from scripts.p2p.managers import P2PSelfplayScheduler

scheduler = P2PSelfplayScheduler(coordinator_scheduler)
config = scheduler.pick_weighted_selfplay_config(node_capabilities)
target_jobs = scheduler.get_target_jobs_for_node(node_info)
```

Methods: `pick_weighted_selfplay_config()`, `get_elo_based_priority_boost()`, `get_hybrid_job_targets()`

### TrainingCoordinator (~730 LOC)

**Responsibility**: Training dispatch and model promotion

```python
from scripts.p2p.managers import TrainingCoordinator

coordinator = TrainingCoordinator(state_manager, job_manager)
await coordinator.dispatch_training_job(config_key, node)
await coordinator.handle_training_job_completion(job)
```

Methods: `dispatch_training_job()`, `handle_training_job_completion()`, `run_post_training_gauntlet()`

## Integration Pattern

Managers use **dependency injection** for testability:

```python
class P2POrchestrator:
    def __init__(self):
        self.state_manager = StateManager(self.db_path, self.node_id)
        self.node_selector = NodeSelector(lambda: self._alive_peers)
        self.job_manager = JobManager(
            state_manager=self.state_manager,
            node_selector=self.node_selector,
            run_ssh_command_fn=self._run_ssh_command,
        )
        # ... other managers
```

## Migration Status (December 2025)

| Manager             | Status   | Methods Delegated | LOC Removed |
| ------------------- | -------- | ----------------- | ----------- |
| StateManager        | Complete | 7/7 (100%)        | ~200        |
| TrainingCoordinator | Complete | 5/5 (100%)        | ~450        |
| JobManager          | Complete | 7/7 (100%)        | ~400        |
| SyncPlanner         | Complete | 4/4 (100%)        | ~60         |
| SelfplayScheduler   | Partial  | 4/7 (57%)         | ~200        |
| NodeSelector        | Complete | 6/6 (100%)        | ~50         |

Total reduction: ~1,360 LOC removed from p2p_orchestrator.py

## Background Loop Extraction

In addition to managers, background loops were extracted to `scripts/p2p/loops/`:

- `EloSyncLoop` - Periodic Elo synchronization
- `IdleDetectionLoop` - GPU idle resource detection
- `AutoScalingLoop` - Dynamic node scaling
- `JobReaperLoop` - Stale job cleanup
- `QueuePopulatorLoop` - Work queue maintenance
- `ValidationLoop` - Queues model validation work for newly trained models

Controlled via `LoopManager.start_all()` and feature flag `RINGRIFT_EXTRACTED_LOOPS=true`.

## Consequences

### Positive

- Each manager is independently testable
- Clear separation of concerns
- Reduced file complexity (27K -> 26K LOC, targeting 20K)
- Dependency injection enables mocking
- Type hints improve IDE support

### Negative

- More files to navigate
- Callback-based API for remaining integrations
- Some orchestrator logic still tightly coupled

## Remaining Work (Q1 2026)

1. Complete SelfplayScheduler delegation (3 methods, ~150 LOC)
2. Extract remaining state sync logic (~200 LOC)
3. Move background job monitoring to dedicated manager
4. Consider extracting election logic to separate module

## Related ADRs

- ADR-001: Event-Driven Architecture (managers emit events)
- ADR-002: Daemon Lifecycle Management (orchestrator is a daemon)
- ADR-006: Cluster Manifest Decomposition (SyncPlanner uses manifest)
