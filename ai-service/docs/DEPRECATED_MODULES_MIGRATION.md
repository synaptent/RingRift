# Deprecated Modules Migration Catalog

**Last Updated**: December 28, 2025
**Removal Target**: Q2 2026

This document consolidates all deprecated modules across the RingRift AI service, their replacements, and migration instructions.

---

## Quick Reference Table

### Coordination Layer (`app/coordination/`)

| Deprecated Module                    | Replacement                                         | Status    | Removal |
| ------------------------------------ | --------------------------------------------------- | --------- | ------- |
| `auto_evaluation_daemon.py`          | `evaluation_daemon.py` + `auto_promotion_daemon.py` | Archived  | Q2 2026 |
| `cluster_data_sync.py`               | `AutoSyncDaemon(strategy="broadcast")`              | Warning   | Q2 2026 |
| `cross_process_events.py`            | `event_router.py`                                   | Warning   | Q2 2026 |
| `ephemeral_sync.py`                  | `AutoSyncDaemon(strategy="ephemeral")`              | Warning   | Q2 2026 |
| `event_emitters.py`                  | `event_router.py`                                   | Warning   | Q2 2026 |
| `health_check_orchestrator.py` (old) | `cluster/health.py` → `UnifiedHealthManager`        | Warning   | Q2 2026 |
| `lambda_idle_daemon.py`              | `unified_idle_shutdown_daemon.py`                   | Archived  | Q2 2026 |
| `model_distribution_daemon.py`       | `unified_distribution_daemon.py`                    | Archived  | Q2 2026 |
| `node_health_monitor.py`             | `health_check_orchestrator.py`                      | Archived  | Q2 2026 |
| `npz_distribution_daemon.py`         | `unified_distribution_daemon.py`                    | Archived  | Q2 2026 |
| `queue_populator.py` (original)      | `unified_queue_populator.py`                        | Re-export | Q2 2026 |
| `queue_populator_daemon.py`          | `unified_queue_populator.py`                        | Archived  | Q2 2026 |
| `replication_monitor.py`             | `unified_replication_daemon.py`                     | Archived  | Q2 2026 |
| `replication_repair_daemon.py`       | `unified_replication_daemon.py`                     | Archived  | Q2 2026 |
| `sync_coordination_core.py`          | `auto_sync_daemon.py` + `sync_router.py`            | Archived  | Q2 2026 |
| `sync_coordinator.py`                | `auto_sync_daemon.py` + `sync_facade.py`            | Archived  | Q2 2026 |
| `system_health_monitor.py`           | `unified_health_manager.py`                         | Warning   | Q2 2026 |
| `unified_event_coordinator.py`       | `event_router.py`                                   | Archived  | Q2 2026 |
| `vast_idle_daemon.py`                | `unified_idle_shutdown_daemon.py`                   | Archived  | Q2 2026 |

### Distributed Layer (`app/distributed/`)

| Deprecated Module      | Replacement                                                             | Status  | Removal |
| ---------------------- | ----------------------------------------------------------------------- | ------- | ------- |
| `unified_data_sync.py` | `sync_facade.py` (programmatic) or `scripts/unified_data_sync.py` (CLI) | Warning | Q2 2026 |

### Training Layer (`app/training/`)

| Deprecated Module                         | Replacement                            | Status     | Removal |
| ----------------------------------------- | -------------------------------------- | ---------- | ------- |
| `integrated_enhancements.py`              | `unified_orchestrator.py`              | Deprecated | Q2 2026 |
| `orchestrated_training.py`                | `unified_orchestrator.py`              | Deprecated | Q2 2026 |
| `training_enhancements.DataQualityScorer` | `unified_quality.UnifiedQualityScorer` | Deprecated | Q2 2026 |

### AI Layer (`app/ai/`)

| Deprecated Module | Replacement                   | Status     | Removal |
| ----------------- | ----------------------------- | ---------- | ------- |
| `ebmo_ai.py`      | Legacy, explicit imports only | Deprecated | Q2 2026 |
| `gmo_ai.py`       | Legacy, explicit imports only | Deprecated | Q2 2026 |
| `gmo_v2.py`       | Legacy, explicit imports only | Deprecated | Q2 2026 |

### Scripts (`scripts/`)

| Deprecated Script             | Replacement                                | Status  |
| ----------------------------- | ------------------------------------------ | ------- |
| `cluster_automation.py`       | `p2p_orchestrator.py`                      | Removed |
| `cluster_control.py`          | `p2p_orchestrator.py`                      | Removed |
| `cluster_manager.py`          | `p2p_orchestrator.py`                      | Removed |
| `cluster_monitor.py`          | `p2p_orchestrator.py` + `scripts/monitor/` | Removed |
| `continuous_training_loop.py` | `p2p_orchestrator.py`                      | Removed |
| `job_scheduler.py`            | `p2p_orchestrator.py`                      | Removed |
| `training_orchestrator.py`    | `p2p_orchestrator.py`                      | Removed |

---

## Status Definitions

| Status         | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| **Archived**   | Moved to `archive/deprecated_coordination/`, no imports remain |
| **Warning**    | Still in place, emits `DeprecationWarning` on import           |
| **Re-export**  | Thin wrapper that re-exports from replacement with warning     |
| **Removed**    | Deleted from codebase (use git history if needed)              |
| **Deprecated** | Marked deprecated but no warning yet                           |

---

## Detailed Migration Guides

### 1. Event System Migration

**Old:**

```python
from app.coordination.event_emitters import emit_training_completed, emit_model_promoted
from app.coordination.cross_process_events import publish_event, poll_events
```

**New:**

```python
from app.coordination.event_router import (
    publish,
    subscribe,
    get_router,
    DataEventType,
    emit_training_completed,  # Still available for convenience
    emit_model_promoted,
)

# Typed publish
await publish(DataEventType.TRAINING_COMPLETED, payload={"config_key": "hex8_2p"})

# Subscriptions
router = get_router()
router.subscribe(DataEventType.MODEL_PROMOTED, my_handler)
```

### 2. Health Monitoring Migration

**Old:**

```python
from app.coordination.health_check_orchestrator import HealthCheckOrchestrator, get_health_orchestrator
from app.coordination.system_health_monitor import get_system_health_score
from app.coordination.node_health_monitor import NodeHealthMonitor
```

**New:**

```python
from app.coordination.unified_health_manager import (
    UnifiedHealthManager,
    get_unified_health_manager,
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
)

# Or use the facade
from app.coordination.health_facade import (
    get_node_health,
    get_healthy_nodes,
    get_cluster_health_summary,
)
```

### 3. Data Sync Migration

**Old:**

```python
from app.coordination.cluster_data_sync import ClusterDataSync
from app.coordination.ephemeral_sync import EphemeralSync

sync = ClusterDataSync()
await sync.sync_to_nodes(targets=["node1", "node2"])
```

**New:**

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

# For broadcast sync (push to all nodes)
daemon = AutoSyncDaemon(strategy="broadcast")
await daemon.start()

# For ephemeral hosts (aggressive 5s interval)
daemon = AutoSyncDaemon(strategy="ephemeral")
await daemon.start()

# Or for programmatic one-time sync
from app.coordination.sync_facade import sync
await sync("games", targets=["node1", "node2"])
```

### 4. Distribution Daemon Migration

**Old:**

```python
from app.coordination.model_distribution_daemon import ModelDistributionDaemon
from app.coordination.npz_distribution_daemon import NPZDistributionDaemon

model_daemon = ModelDistributionDaemon()
await model_daemon.distribute_model("models/new.pth")
```

**New:**

```python
from app.coordination.unified_distribution_daemon import (
    UnifiedDistributionDaemon,
    DataType,
    wait_for_model_distribution,
    check_model_availability,
)

# Unified daemon handles both models and NPZ
daemon = UnifiedDistributionDaemon()
await daemon.start()

# Distribute specific files
await daemon.distribute(DataType.MODEL, "models/new.pth")
await daemon.distribute(DataType.NPZ, "data/training/hex8_2p.npz")
```

### 5. Queue Populator Migration

**Old:**

```python
from app.coordination.queue_populator import QueuePopulator

populator = QueuePopulator()
await populator.populate_work_queue()
```

**New:**

```python
from app.coordination.unified_queue_populator import (
    UnifiedQueuePopulator,
    QueuePopulatorConfig,
)

config = QueuePopulatorConfig(
    selfplay_ratio=0.6,
    training_ratio=0.3,
    tournament_ratio=0.1,
)
populator = UnifiedQueuePopulator(config)
await populator.start()
```

### 6. Replication Daemon Migration

**Old:**

```python
from app.coordination.replication_monitor import ReplicationMonitor
from app.coordination.replication_repair_daemon import ReplicationRepairDaemon

monitor = ReplicationMonitor()
repair = ReplicationRepairDaemon()
```

**New:**

```python
from app.coordination.unified_replication_daemon import (
    UnifiedReplicationDaemon,
    ReplicationConfig,
)

# Single daemon handles both monitoring and repair
config = ReplicationConfig(
    target_replicas=3,
    repair_interval=300,
    emergency_sync_threshold=1,
)
daemon = UnifiedReplicationDaemon(config)
await daemon.start()
```

### 7. Idle Shutdown Migration

**Old:**

```python
from app.coordination.lambda_idle_daemon import LambdaIdleDaemon
from app.coordination.vast_idle_daemon import VastIdleDaemon

lambda_daemon = LambdaIdleDaemon()
vast_daemon = VastIdleDaemon()
```

**New:**

```python
from app.coordination.unified_idle_shutdown_daemon import (
    UnifiedIdleShutdownDaemon,
    IdleShutdownConfig,
    create_lambda_idle_daemon,
    create_vast_idle_daemon,
    create_runpod_idle_daemon,
)

# Factory functions for backward compatibility
lambda_daemon = create_lambda_idle_daemon()
vast_daemon = create_vast_idle_daemon()
runpod_daemon = create_runpod_idle_daemon()  # NEW!
```

### 8. Evaluation Daemon Migration

**Old:**

```python
from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon

daemon = AutoEvaluationDaemon()
await daemon.evaluate_and_promote("models/new.pth")
```

**New:**

```python
from app.coordination.evaluation_daemon import EvaluationDaemon
from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

# Separate concerns - evaluation and promotion are now distinct
eval_daemon = EvaluationDaemon()
await eval_daemon.start()  # Evaluates models, emits EVALUATION_COMPLETED

promo_daemon = AutoPromotionDaemon()
await promo_daemon.start()  # Listens for eval results, promotes if passing
```

---

## Environment Variable Changes

| Old Variable                       | New Variable                       | Default |
| ---------------------------------- | ---------------------------------- | ------- |
| `RINGRIFT_CLUSTER_SYNC_INTERVAL`   | `RINGRIFT_AUTO_SYNC_INTERVAL`      | 60      |
| `RINGRIFT_EPHEMERAL_SYNC_INTERVAL` | `RINGRIFT_EPHEMERAL_SYNC_INTERVAL` | 5       |
| `RINGRIFT_MODEL_DIST_WORKERS`      | `RINGRIFT_DIST_WORKERS`            | 4       |
| `RINGRIFT_REPLICATION_TARGET`      | `RINGRIFT_REPLICATION_REPLICAS`    | 3       |

---

## Archive Locations

Archived modules are preserved for reference in:

```
archive/
├── deprecated_coordination/
│   ├── README.md                      # Archive documentation
│   ├── MIGRATION_GUIDE.md             # Detailed migration guide
│   ├── _deprecated_auto_evaluation_daemon.py
│   ├── _deprecated_cross_process_events.py
│   ├── _deprecated_event_emitters.py
│   ├── _deprecated_health_check_orchestrator.py
│   ├── _deprecated_host_health_policy.py
│   ├── _deprecated_lambda_idle_daemon.py
│   ├── _deprecated_model_distribution_daemon.py
│   ├── _deprecated_node_health_monitor.py
│   ├── _deprecated_npz_distribution_daemon.py
│   ├── _deprecated_queue_populator.py
│   ├── _deprecated_queue_populator_daemon.py
│   ├── _deprecated_replication_monitor.py
│   ├── _deprecated_replication_repair_daemon.py
│   ├── _deprecated_sync_coordination_core.py
│   ├── _deprecated_sync_coordinator.py
│   ├── _deprecated_system_health_monitor.py
│   ├── _deprecated_unified_event_coordinator.py
│   └── _deprecated_vast_idle_daemon.py
└── deprecated_scripts/
    └── README.md                      # Scripts archive documentation
```

---

## Consolidation Summary

### December 2025 Wave

| Consolidation                          | Original LOC | New LOC    | Savings    |
| -------------------------------------- | ------------ | ---------- | ---------- |
| Idle daemons (Lambda + Vast)           | ~600         | ~318       | ~282       |
| Distribution daemons (Model + NPZ)     | ~2,617       | ~750       | ~1,867     |
| Replication daemons (Monitor + Repair) | ~1,334       | ~750       | ~584       |
| Queue populator variants               | ~1,967       | ~809       | ~1,158     |
| Health monitors (Node + System)        | ~986         | ~200       | ~786       |
| Event system (3 modules)               | ~1,500       | ~600       | ~900       |
| **Total**                              | **~9,004**   | **~3,427** | **~5,577** |

---

## Verification Commands

```bash
# Find files still importing deprecated modules
grep -r "from app.coordination.cluster_data_sync import" --include="*.py" .
grep -r "from app.coordination.event_emitters import" --include="*.py" .
grep -r "from app.coordination.cross_process_events import" --include="*.py" .

# Count deprecated imports
git grep -l "deprecated_" app/ | wc -l

# Verify deprecation warnings work
python -c "import warnings; warnings.filterwarnings('error'); from app.coordination.queue_populator import QueuePopulator"
```

---

## Timeline

| Phase  | Date     | Action                                            |
| ------ | -------- | ------------------------------------------------- |
| Wave 1 | Dec 2025 | Initial consolidation, deprecation warnings added |
| Wave 2 | Q1 2026  | Update remaining imports, add linting rules       |
| Wave 3 | Q2 2026  | Remove deprecated modules (breaking change)       |

---

## Related Documentation

- `archive/deprecated_coordination/README.md` - Detailed archive documentation
- `archive/deprecated_coordination/MIGRATION_GUIDE.md` - Step-by-step migration
- `docs/archive/coordination_consolidation_2025_12/DEPRECATED_MODULE_USAGE.md` - Import tracking
- `scripts/DEPRECATED.md` - Scripts deprecation manifest
- `app/DEPRECATION_AUDIT.md` - Migration status tracking

---

## Contact

For migration questions:

- See `CLAUDE.md` for module documentation
- See individual module docstrings for usage examples
- Check test files for working examples
