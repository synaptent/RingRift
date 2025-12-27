# Deprecated Coordination Modules - Migration Guide

This guide provides a timeline and migration paths for deprecated coordination modules.

## Timeline

| Phase       | Date          | Action                                          |
| ----------- | ------------- | ----------------------------------------------- |
| Deprecation | December 2025 | Modules deprecated with warnings                |
| Transition  | Q1 2026       | Use new APIs, deprecated imports still work     |
| Archival    | Q2 2026       | Move remaining deprecated modules to archive/   |
| Removal     | Q3 2026       | Delete archived files if no regression reported |

## Migration Matrix

| Deprecated Module                    | Replacement                            | Status                          | Migration Priority |
| ------------------------------------ | -------------------------------------- | ------------------------------- | ------------------ |
| `cluster_data_sync.py`               | `AutoSyncDaemon(strategy="broadcast")` | Deprecated - migrate by Q2 2026 | HIGH               |
| `ephemeral_sync.py`                  | `AutoSyncDaemon(strategy="ephemeral")` | Deprecated - migrate by Q2 2026 | HIGH               |
| `system_health_monitor.py`           | `health_facade.get_system_health()`    | Deprecated - migrate by Q2 2026 | MEDIUM             |
| `node_health_monitor.py`             | `health_check_orchestrator.py`         | Deprecated - migrate by Q2 2026 | LOW                |
| `sync_coordinator.py` (coordination) | `AutoSyncDaemon`, `SyncFacade`         | Archived Dec 2025               | COMPLETE           |
| `unified_event_coordinator.py`       | `event_router.py`                      | Archived Dec 2025               | COMPLETE           |
| `queue_populator.py` (old)           | `unified_queue_populator.py`           | Archived Dec 2025               | COMPLETE           |
| `model_distribution_daemon.py`       | `unified_distribution_daemon.py`       | Archived Dec 2025               | COMPLETE           |
| `npz_distribution_daemon.py`         | `unified_distribution_daemon.py`       | Archived Dec 2025               | COMPLETE           |
| `replication_monitor.py`             | `unified_replication_daemon.py`        | Archived Dec 2025               | COMPLETE           |
| `replication_repair_daemon.py`       | `unified_replication_daemon.py`        | Archived Dec 2025               | COMPLETE           |
| `lambda_idle_daemon.py`              | `unified_idle_shutdown_daemon.py`      | Archived Dec 2025               | COMPLETE           |
| `vast_idle_daemon.py`                | `unified_idle_shutdown_daemon.py`      | Archived Dec 2025               | COMPLETE           |

## Migration Examples

### Sync Modules

**cluster_data_sync.py -> AutoSyncDaemon**

```python
# OLD (deprecated)
from app.coordination.cluster_data_sync import get_training_node_watcher
watcher = get_training_node_watcher()

# NEW
from app.coordination.auto_sync_daemon import (
    AutoSyncDaemon,
    AutoSyncConfig,
    SyncStrategy,
)

config = AutoSyncConfig.from_config_file()
config.strategy = SyncStrategy.BROADCAST
daemon = AutoSyncDaemon(config=config)
await daemon.start()
```

**ephemeral_sync.py -> AutoSyncDaemon**

```python
# OLD (deprecated)
from app.coordination.ephemeral_sync import get_ephemeral_sync_daemon
daemon = get_ephemeral_sync_daemon()

# NEW
from app.coordination.auto_sync_daemon import (
    AutoSyncDaemon,
    AutoSyncConfig,
    SyncStrategy,
)

config = AutoSyncConfig.from_config_file()
config.strategy = SyncStrategy.EPHEMERAL
daemon = AutoSyncDaemon(config=config)
await daemon.start()
```

### Health Modules

**system_health_monitor.py -> health_facade**

```python
# OLD (deprecated)
from app.coordination.system_health_monitor import SystemHealthMonitor
monitor = SystemHealthMonitor()
health = monitor.get_health()

# NEW
from app.coordination.health_facade import (
    get_system_health_score,
    should_pause_pipeline,
    get_system_health,  # Backward compat - emits DeprecationWarning
)

score = get_system_health_score()
if should_pause_pipeline():
    # Pause training
    pass
```

**node_health_monitor.py -> health_check_orchestrator**

```python
# OLD (deprecated)
from app.coordination.node_health_monitor import (
    NodeHealthMonitor,
    get_node_health_monitor,
)
monitor = get_node_health_monitor()
health = monitor.get_node_health("runpod-h100")

# NEW
from app.coordination.health_facade import (
    get_health_orchestrator,
    get_node_health,
    get_healthy_nodes,
    get_unhealthy_nodes,
)

orchestrator = get_health_orchestrator()
health = get_node_health("runpod-h100")
healthy = get_healthy_nodes()
```

### Distribution Daemons

**model_distribution_daemon.py + npz_distribution_daemon.py -> unified_distribution_daemon**

```python
# OLD (deprecated)
from app.coordination.model_distribution_daemon import ModelDistributionDaemon
from app.coordination.npz_distribution_daemon import NPZDistributionDaemon
model_daemon = ModelDistributionDaemon()
npz_daemon = NPZDistributionDaemon()

# NEW
from app.coordination.unified_distribution_daemon import (
    UnifiedDistributionDaemon,
    DataType,
    DistributionConfig,
    # Factory functions for backward compat
    create_model_distribution_daemon,
    create_npz_distribution_daemon,
)

# Unified daemon handles both
config = DistributionConfig(data_types={DataType.MODEL, DataType.NPZ})
daemon = UnifiedDistributionDaemon(config=config)

# Or use factory functions
model_daemon = create_model_distribution_daemon()
npz_daemon = create_npz_distribution_daemon()
```

### Idle Shutdown Daemons

**lambda_idle_daemon.py + vast_idle_daemon.py -> unified_idle_shutdown_daemon**

```python
# OLD (deprecated)
from app.coordination.lambda_idle_daemon import LambdaIdleDaemon
from app.coordination.vast_idle_daemon import VastIdleDaemon

# NEW
from app.coordination.unified_idle_shutdown_daemon import (
    create_lambda_idle_daemon,
    create_vast_idle_daemon,
    create_runpod_idle_daemon,
    UnifiedIdleShutdownDaemon,
    IdleShutdownConfig,
)

# Factory functions (backward compat)
lambda_daemon = create_lambda_idle_daemon()
vast_daemon = create_vast_idle_daemon()
runpod_daemon = create_runpod_idle_daemon()  # NEW!

# Or configure manually
config = IdleShutdownConfig.for_provider("vast")
daemon = UnifiedIdleShutdownDaemon(provider=VastProvider(), config=config)
```

### Event System

**unified_event_coordinator.py -> event_router**

```python
# OLD (deprecated)
from app.coordination.unified_event_coordinator import (
    get_event_coordinator,
    start_coordinator,
)
coord = get_event_coordinator()
await start_coordinator()

# NEW
from app.coordination.event_router import (
    get_router,
    UnifiedEventRouter,
)
router = get_router()
await router.start()
```

### Queue Populator

**queue_populator.py (old) -> unified_queue_populator**

```python
# OLD (deprecated)
from app.coordination.queue_populator import QueuePopulator
populator = QueuePopulator()

# NEW
from app.coordination.unified_queue_populator import (
    UnifiedQueuePopulator,
    QueuePopulatorConfig,
)
config = QueuePopulatorConfig()
populator = UnifiedQueuePopulator(config=config)
```

## Deprecation Warning Behavior

All deprecated modules emit `DeprecationWarning` on import:

```python
import warnings

# To see warnings in production
warnings.filterwarnings("default", category=DeprecationWarning)

# To silence warnings temporarily (not recommended)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## Verification Commands

```bash
# Check for deprecated imports in your code
grep -r "from app.coordination.cluster_data_sync import" --include="*.py" .
grep -r "from app.coordination.ephemeral_sync import" --include="*.py" .
grep -r "from app.coordination.system_health_monitor import" --include="*.py" .
grep -r "from app.coordination.node_health_monitor import" --include="*.py" .

# Run tests to verify migration
PYTHONPATH=. pytest tests/unit/coordination/ -v

# Check import graph for deprecated usages
python -c "
from app.coordination import *
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)
"
```

## Contact

For migration assistance or questions:

- Check the main README.md for detailed per-module migration docs
- Review test files for usage examples
- Raise issues in the project tracker

---

_Last updated: December 27, 2025_
