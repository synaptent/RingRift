# P2P Manager Architecture

**Last Updated**: December 28, 2025

## Overview

The P2P orchestrator has been decomposed from a ~30,000 LOC monolith into modular manager classes. This document provides an overview of the architecture and links to detailed documentation.

**Current Status**: All 7 managers fully delegated (100% coverage), ~1,990 LOC removed from p2p_orchestrator.py.

**Detailed Documentation**: [scripts/p2p/managers/README.md](../scripts/p2p/managers/README.md)

---

## Manager Summary

| Manager                 | Purpose                      | Key Methods                                                       |
| ----------------------- | ---------------------------- | ----------------------------------------------------------------- |
| **StateManager**        | SQLite persistence, epochs   | `load_state()`, `save_state()`, `get_epoch()`                     |
| **NodeSelector**        | Node ranking for jobs        | `get_best_gpu_node_for_training()`, `get_training_nodes_ranked()` |
| **SyncPlanner**         | Data sync planning           | `collect_manifest()`, `create_sync_plan()`                        |
| **JobManager**          | Job spawning/lifecycle       | `run_gpu_selfplay_job()`, `spawn_training()`                      |
| **SelfplayScheduler**   | Priority-based scheduling    | `pick_weighted_config()`, `get_target_jobs()`                     |
| **TrainingCoordinator** | Training workflow            | `dispatch_training_job()`, `check_readiness()`                    |
| **LoopManager**         | Background loop coordination | `start_all()`, `stop_all()`, `health_check()`                     |

---

## Architecture Diagram

```
P2POrchestrator
├── StateManager (SQLite persistence)
│   ├── Peers table
│   ├── Jobs table
│   └── State/Config tables
│
├── NodeSelector (Node ranking)
│   ├── GPU power scoring
│   └── Node filtering
│
├── SyncPlanner (Data sync)
│   ├── Manifest collection
│   └── Sync job dispatch
│
├── JobManager (Job lifecycle)
│   ├── Selfplay spawning
│   ├── Training spawning
│   └── Job tracking
│
├── SelfplayScheduler (Priority scheduling)
│   ├── Config weighting
│   ├── Curriculum integration
│   └── Diversity tracking
│
├── TrainingCoordinator (Training workflow)
│   ├── Readiness checking
│   ├── Job dispatch
│   └── Gauntlet/promotion
│
└── LoopManager (Background loops)
    ├── JobReaperLoop
    ├── IdleDetectionLoop
    ├── EloSyncLoop
    ├── QueuePopulatorLoop
    └── SelfHealingLoop
```

---

## Background Loops

| Loop                     | Interval | Purpose                            |
| ------------------------ | -------- | ---------------------------------- |
| `JobReaperLoop`          | 5 min    | Clean stale/stuck jobs             |
| `IdleDetectionLoop`      | 30 sec   | Detect idle GPUs, trigger selfplay |
| `WorkerPullLoop`         | 30 sec   | Workers poll leader for work       |
| `EloSyncLoop`            | 5 min    | Elo rating synchronization         |
| `QueuePopulatorLoop`     | 1 min    | Work queue maintenance             |
| `SelfHealingLoop`        | 5 min    | Recover stuck processes            |
| `ManifestCollectionLoop` | 1 min    | Collect peer manifests             |

---

## Dependency Injection Pattern

All managers follow a consistent pattern for testability:

```python
class Manager:
    def __init__(
        self,
        get_peers: Callable[[], dict[str, NodeInfo]],  # State callbacks
        get_self_info: Callable[[], NodeInfo],
        peers_lock: threading.Lock,
        config: ManagerConfig | None = None,           # Configuration
    ):
        self._get_peers = get_peers
        # ...
```

**Benefits**:

- Managers can be unit tested with mock callbacks
- No direct imports of orchestrator internals
- Composable and flexible

---

## Health Check Integration

All managers implement `health_check()` for DaemonManager integration:

```python
def health_check(self) -> HealthCheckResult:
    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="Manager operational",
        details={...},
    )
```

Health is aggregated at the P2P `/status` endpoint.

---

## Event Integration

Managers use `EventSubscriptionMixin` for event handling:

```python
class MyManager(EventSubscriptionMixin):
    def _get_event_subscriptions(self) -> dict:
        return {
            "HOST_OFFLINE": self._on_host_offline,
            "TRAINING_COMPLETED": self._on_training_completed,
        }
```

---

## File Locations

| Component      | Path                             |
| -------------- | -------------------------------- |
| Manager README | `scripts/p2p/managers/README.md` |
| Managers       | `scripts/p2p/managers/*.py`      |
| Loops          | `scripts/p2p/loops/*.py`         |
| Base mixins    | `scripts/p2p/p2p_mixin_base.py`  |
| Orchestrator   | `scripts/p2p_orchestrator.py`    |

---

## Quick Start

```python
# Import managers
from scripts.p2p.managers import (
    StateManager,
    NodeSelector,
    JobManager,
    SelfplayScheduler,
    TrainingCoordinator,
)

# Import loops
from scripts.p2p.loops import (
    LoopManager,
    JobReaperLoop,
    IdleDetectionLoop,
)
```

---

## Testing

```bash
# Run P2P manager tests
PYTHONPATH=. pytest tests/unit/p2p/ -v

# Specific manager
PYTHONPATH=. pytest tests/unit/p2p/test_node_selector.py -v
```

---

## See Also

- [Detailed Manager README](../scripts/p2p/managers/README.md) - Comprehensive documentation
- [SELFPLAY_SCHEDULER_USAGE.md](../scripts/p2p/managers/SELFPLAY_SCHEDULER_USAGE.md) - Scheduler guide
- [DAEMON_REGISTRY.md](./DAEMON_REGISTRY.md) - Daemon types and lifecycle
- [EVENT_SYSTEM_REFERENCE.md](./EVENT_SYSTEM_REFERENCE.md) - Event catalog
