# Deprecated Coordination Modules - Migration Guide

**Last Updated:** December 27, 2025
**Removal Date:** Q2 2026

This guide documents deprecated coordination modules and their replacements.

---

## Quick Reference

| Deprecated Module              | Replacement                                         | Status     |
| ------------------------------ | --------------------------------------------------- | ---------- |
| `cluster_data_sync.py`         | `AutoSyncDaemon(strategy="broadcast")`              | Deprecated |
| `ephemeral_sync.py`            | `AutoSyncDaemon(strategy="ephemeral")`              | Deprecated |
| `node_health_monitor.py`       | `health_check_orchestrator.py`                      | Deprecated |
| `system_health_monitor.py`     | `unified_health_manager.py`                         | Deprecated |
| `queue_populator.py`           | `unified_queue_populator.py`                        | Deprecated |
| `model_distribution_daemon.py` | `unified_distribution_daemon.py`                    | Deprecated |
| `npz_distribution_daemon.py`   | `unified_distribution_daemon.py`                    | Deprecated |
| `replication_monitor.py`       | `unified_replication_daemon.py`                     | Deprecated |
| `replication_repair_daemon.py` | `unified_replication_daemon.py`                     | Deprecated |
| `auto_evaluation_daemon.py`    | `evaluation_daemon.py` + `auto_promotion_daemon.py` | Deprecated |

---

## Detailed Migration Guide

### 1. cluster_data_sync.py

**Old usage:**

```python
from app.coordination.cluster_data_sync import ClusterDataSync

sync = ClusterDataSync()
await sync.sync_to_nodes(targets=["node1", "node2"])
```

**New usage:**

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

# For broadcast sync (push to all nodes)
daemon = AutoSyncDaemon(strategy="broadcast")
await daemon.start()

# Or for one-time sync
from app.coordination.sync_facade import sync
await sync("games", targets=["node1", "node2"])
```

---

### 2. ephemeral_sync.py

**Old usage:**

```python
from app.coordination.ephemeral_sync import EphemeralSync

sync = EphemeralSync(interval=5)
await sync.run()
```

**New usage:**

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

# For ephemeral hosts (aggressive 5s interval)
daemon = AutoSyncDaemon(strategy="ephemeral")
await daemon.start()
```

---

### 3. node_health_monitor.py

**Old usage:**

```python
from app.coordination.node_health_monitor import NodeHealthMonitor

monitor = NodeHealthMonitor()
status = monitor.get_node_status("node-1")
```

**New usage:**

```python
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    get_health_orchestrator,
)

orchestrator = get_health_orchestrator()
status = orchestrator.get_node_health("node-1")
```

---

### 4. system_health_monitor.py

**Old usage:**

```python
from app.coordination.system_health_monitor import (
    SystemHealthMonitor,
    get_system_health_score,
)

score = get_system_health_score()
```

**New usage:**

```python
from app.coordination.unified_health_manager import (
    UnifiedHealthManager,
    get_unified_health_manager,
    get_system_health_score,
    get_system_health_level,
)

manager = get_unified_health_manager()
score = get_system_health_score()
level = get_system_health_level()
```

---

### 5. queue_populator.py

**Old usage:**

```python
from app.coordination.queue_populator import QueuePopulator

populator = QueuePopulator()
await populator.populate_work_queue()
```

**New usage:**

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

---

### 6. model_distribution_daemon.py / npz_distribution_daemon.py

**Old usage:**

```python
from app.coordination.model_distribution_daemon import ModelDistributionDaemon
from app.coordination.npz_distribution_daemon import NPZDistributionDaemon

model_daemon = ModelDistributionDaemon()
await model_daemon.distribute_model("models/new.pth")

npz_daemon = NPZDistributionDaemon()
await npz_daemon.distribute_npz("data/training/hex8_2p.npz")
```

**New usage:**

```python
from app.coordination.unified_distribution_daemon import (
    UnifiedDistributionDaemon,
    DataType,
    DistributionConfig,
    wait_for_model_distribution,
    check_model_availability,
)

# Unified daemon handles both models and NPZ
daemon = UnifiedDistributionDaemon()
await daemon.start()

# Distribute specific files
await daemon.distribute(DataType.MODEL, "models/new.pth")
await daemon.distribute(DataType.NPZ, "data/training/hex8_2p.npz")

# Wait for model to be available on nodes
available = await wait_for_model_distribution(
    "models/new.pth",
    required_nodes=5,
    timeout=300,
)
```

---

### 7. replication_monitor.py / replication_repair_daemon.py

**Old usage:**

```python
from app.coordination.replication_monitor import ReplicationMonitor
from app.coordination.replication_repair_daemon import ReplicationRepairDaemon

monitor = ReplicationMonitor()
monitor.check_replication_status()

repair = ReplicationRepairDaemon()
repair.repair_missing_replicas()
```

**New usage:**

```python
from app.coordination.unified_replication_daemon import (
    UnifiedReplicationDaemon,
    ReplicationConfig,
    create_replication_monitor,  # Backward compat
    create_replication_repair_daemon,  # Backward compat
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

---

### 8. auto_evaluation_daemon.py

**Old usage:**

```python
from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon

daemon = AutoEvaluationDaemon()
await daemon.evaluate_and_promote("models/new.pth")
```

**New usage:**

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

## Backward Compatibility

Most deprecated modules have shim imports that emit `DeprecationWarning`. These will continue to work until Q2 2026:

```python
# These still work but emit warnings
from app.coordination.cluster_data_sync import ClusterDataSync  # DeprecationWarning
from app.coordination.queue_populator import QueuePopulator  # DeprecationWarning
```

To suppress warnings during migration:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="app.coordination")
```

---

## Environment Variables

Some deprecated modules had environment-variable-based configuration. Here are the new equivalents:

| Old Variable                       | New Variable                       | Default |
| ---------------------------------- | ---------------------------------- | ------- |
| `RINGRIFT_CLUSTER_SYNC_INTERVAL`   | `RINGRIFT_AUTO_SYNC_INTERVAL`      | 60      |
| `RINGRIFT_EPHEMERAL_SYNC_INTERVAL` | `RINGRIFT_EPHEMERAL_SYNC_INTERVAL` | 5       |
| `RINGRIFT_MODEL_DIST_WORKERS`      | `RINGRIFT_DIST_WORKERS`            | 4       |
| `RINGRIFT_REPLICATION_TARGET`      | `RINGRIFT_REPLICATION_REPLICAS`    | 3       |

---

## Questions?

See the main CLAUDE.md for module documentation or file an issue in the repository.
