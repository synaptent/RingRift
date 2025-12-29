# Migration Guide: December 2025 Consolidations

This document summarizes all major module consolidations completed in December 2025 and provides migration paths for deprecated code.

> Status: Historical snapshot (Dec 2025). Kept for reference; consult `ai-service/docs/README.md` for current guidance.

## Quick Reference Table

| Category        | Deprecated Module                                            | Replacement                             | LOC Saved |
| --------------- | ------------------------------------------------------------ | --------------------------------------- | --------- |
| Idle Shutdown   | `lambda_idle_daemon.py`, `vast_idle_daemon.py`               | `unified_idle_shutdown_daemon.py`       | 318       |
| Distribution    | `model_distribution_daemon.py`, `npz_distribution_daemon.py` | `unified_distribution_daemon.py`        | ~1,100    |
| Replication     | `replication_monitor.py`, `replication_repair_daemon.py`     | `unified_replication_daemon.py`         | ~580      |
| Queue Populator | `queue_populator_daemon.py`, `queue_populator.py`            | `unified_queue_populator.py`            | ~1,100    |
| Events          | `unified_event_coordinator.py`                               | `event_router.py`                       | ~400      |
| Health          | `system_health_monitor.py`, `health_check_orchestrator.py`   | `unified_health_manager.py`             | ~600      |
| Sync            | `sync_coordinator.py` (coordination)                         | `auto_sync_daemon.py`, `sync_facade.py` | ~1,400    |

**Total LOC Saved**: ~5,500+ lines through consolidation

## Module Consolidations

### 1. Idle Shutdown Daemon

Provider-specific idle daemons consolidated into a single provider-agnostic implementation.

```python
# OLD - Provider-specific
from app.coordination.lambda_idle_daemon import LambdaIdleDaemon
from app.coordination.vast_idle_daemon import VastIdleDaemon

# NEW - Unified with factory functions
from app.coordination.unified_idle_shutdown_daemon import (
    create_lambda_idle_daemon,
    create_vast_idle_daemon,
    create_runpod_idle_daemon,  # NEW! Added RunPod support
)

daemon = create_vast_idle_daemon()
await daemon.start()
```

**Key improvements**:

- Provider-agnostic design using CloudProvider interface
- Per-provider configurable thresholds (Lambda: 30min, Vast: 15min, RunPod: 20min)
- Pending work check before termination
- Minimum node retention for cluster capacity

---

### 2. Distribution Daemon

Model and NPZ distribution consolidated into single daemon with smart transport selection.

```python
# OLD - Separate daemons
from app.coordination.model_distribution_daemon import ModelDistributionDaemon
from app.coordination.npz_distribution_daemon import NPZDistributionDaemon

# NEW - Unified with DataType enum
from app.coordination.unified_distribution_daemon import (
    UnifiedDistributionDaemon,
    DataType,
    wait_for_model_distribution,  # Backward compat
)

daemon = UnifiedDistributionDaemon()
await daemon.distribute(DataType.MODEL, "/path/to/model.pth")
```

**Key improvements**:

- Single daemon handles both model and NPZ distribution
- Smart transport selection: BitTorrent > HTTP > rsync
- SHA256 checksum verification
- Per-node delivery tracking

---

### 3. Replication Daemon

Replication monitoring and repair consolidated into single daemon.

```python
# OLD - Separate monitoring and repair
from app.coordination.replication_monitor import ReplicationMonitor
from app.coordination.replication_repair_daemon import ReplicationRepairDaemon

# NEW - Unified monitoring + repair
from app.coordination.unified_replication_daemon import (
    UnifiedReplicationDaemon,
    create_replication_monitor,       # Backward compat factory
    create_replication_repair_daemon,  # Backward compat factory
)

daemon = UnifiedReplicationDaemon()
await daemon.start()  # Handles both monitoring and repair
```

**Key improvements**:

- Single daemon for monitoring and repair
- Priority repair queue
- Emergency sync for critical gaps
- Alert generation on prolonged issues

---

### 4. Queue Populator

Queue population logic consolidated with Elo-based targeting.

```python
# OLD
from app.coordination.queue_populator import QueuePopulator
from app.coordination.queue_populator_daemon import QueuePopulatorDaemon

# NEW
from app.coordination.unified_queue_populator import (
    UnifiedQueuePopulator,
    QueuePopulatorConfig,
)

populator = UnifiedQueuePopulator()
populator.populate()
```

**Key improvements**:

- Unified work queue maintenance
- Elo-based completion targets
- Config-specific thresholds
- 60/30/10 ratio (selfplay/training/tournament)

---

### 5. Event System

Event coordination consolidated into single router with deduplication.

```python
# OLD
from app.coordination.unified_event_coordinator import (
    emit_selfplay_complete,
    get_event_coordinator,
)

# NEW
from app.coordination.event_router import (
    emit_selfplay_complete,  # Same API
    get_event_bus,
    subscribe,
    publish,
)
```

**Key improvements**:

- Content-based deduplication (SHA256)
- Unified subscribe/publish API
- Event history tracking
- Cross-process support via SQLite

---

### 6. Health Management

System health monitoring consolidated with health scoring.

```python
# OLD
from app.coordination.system_health_monitor import get_system_health
from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

# NEW
from app.coordination.unified_health_manager import (
    get_health_manager,
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
)

manager = get_health_manager()
score = get_system_health_score()  # SystemHealthScore dataclass
level = get_system_health_level()  # HEALTHY, DEGRADED, CRITICAL, etc.
```

**Key improvements**:

- Consolidated health scoring
- Health level classification (HEALTHY, DEGRADED, CRITICAL, EMERGENCY)
- Pipeline pause recommendations
- Configurable thresholds

---

### 7. Sync Coordination

Sync scheduling layer deprecated in favor of specialized modules.

```python
# OLD - Monolithic scheduler
from app.coordination.sync_coordinator import (
    SyncScheduler,
    get_sync_scheduler,
    execute_priority_sync,
)

# NEW - Specialized modules
# For automated background sync:
from app.coordination.auto_sync_daemon import AutoSyncDaemon
daemon = AutoSyncDaemon()
await daemon.start()

# For manual sync operations:
from app.coordination.sync_facade import sync
await sync('games', priority='high')

# For routing decisions:
from app.coordination.sync_router import get_sync_router
router = get_sync_router()
targets = router.get_sync_targets('games')
```

**Key improvements**:

- Separation of concerns (scheduling vs execution vs routing)
- AutoSyncDaemon for fire-and-forget automation
- SyncFacade for simple programmatic sync
- SyncRouter for intelligent source selection

---

## Deprecation Timeline

| Status               | Modules                                | Removal Date  |
| -------------------- | -------------------------------------- | ------------- |
| Deprecated           | All modules listed above               | Q2 2026       |
| Archived             | Modules in `archive/deprecated_*/`     | Already moved |
| Active with warnings | `app/coordination/sync_coordinator.py` | Q2 2026       |

## Environment Variables

Many unified modules support environment variable configuration:

```bash
# Idle shutdown (per provider)
export LAMBDA_IDLE_THRESHOLD=1800  # 30 minutes
export VAST_IDLE_THRESHOLD=900     # 15 minutes
export RUNPOD_IDLE_THRESHOLD=1200  # 20 minutes

# Replication
export RINGRIFT_REPLICATION_CHECK_INTERVAL=300  # 5 minutes

# Queue population
export RINGRIFT_QUEUE_POPULATOR_INTERVAL=60  # 1 minute
```

## Test Migration

Existing tests may need updates:

```python
# OLD import
from app.coordination.model_distribution_daemon import ModelDistributionDaemon

# NEW import
from app.coordination.unified_distribution_daemon import UnifiedDistributionDaemon
```

Most factory functions provide backward compatibility:

- `create_lambda_idle_daemon()` wraps `UnifiedIdleShutdownDaemon`
- `create_replication_monitor()` wraps `UnifiedReplicationDaemon`
- `wait_for_model_distribution()` uses `UnifiedDistributionDaemon`

## Detailed Documentation

For detailed migration guides per module category, see:

- [`archive/deprecated_coordination/README.md`](archive/deprecated_coordination/README.md) - Coordination module migrations
- [`archive/deprecated_distributed/README.md`](archive/deprecated_distributed/README.md) - Distributed module migrations
- [`archive/deprecated_scripts/README.md`](archive/deprecated_scripts/README.md) - Script migrations
- [`archive/deprecated_ai/README.md`](archive/deprecated_ai/README.md) - AI module migrations
- [`archive/deprecated_training/README.md`](archive/deprecated_training/README.md) - Training module migrations

---

## December 29, 2025 Updates

### Centralized Port Configuration

Fixed 6 files with hardcoded port 8770 references to use centralized `app/config/ports.py`:

| File                                                | Change                                          |
| --------------------------------------------------- | ----------------------------------------------- |
| `app/coordination/availability/capacity_planner.py` | Uses `get_p2p_status_url()`                     |
| `app/routes/cluster.py`                             | Uses `get_p2p_status_url()`                     |
| `app/integration/__init__.py`                       | Uses `get_local_p2p_url()`                      |
| `app/coordination/unified_data_plane_daemon.py`     | Uses `get_p2p_status_url()`, `P2P_DEFAULT_PORT` |
| `app/monitoring/keepalive_dashboard.py`             | Uses `get_p2p_status_url()`                     |
| `app/tournament/distributed_gauntlet.py`            | Uses `get_local_p2p_url()`                      |
| `app/utils/env_config.py`                           | Uses `get_local_p2p_url()`                      |

**Migration**:

```python
# OLD - Hardcoded
P2P_STATUS_URL = "http://localhost:8770/status"

# NEW - Centralized
from app.config.ports import get_p2p_status_url
P2P_STATUS_URL = get_p2p_status_url()
```

---

### train.py Decomposition Status (6,003 LOC)

**Analysis Summary**:

The main training function `train_model()` spans ~5,250 lines with:

- 116 parameters (lines 657-772)
- Inline validation duplicating `train_validation.py` (~200 LOC)
- Model initialization not using `train_model_init.py` (~400 LOC)
- Training loop not using `train_loop.py` (~2,000 LOC)

**Existing Extracted Modules (NOT YET INTEGRATED)**:

| Module                | Purpose                   | LOC  | Status      |
| --------------------- | ------------------------- | ---- | ----------- |
| `train_validation.py` | Data validation           | 21KB | ✅ Imported |
| `train_loop.py`       | Training loop             | 15KB | ❌ Not used |
| `train_model_init.py` | Model initialization      | 12KB | ❌ Not used |
| `train_setup.py`      | Training setup utilities  | 17KB | ❌ Not used |
| `train_config.py`     | Configuration dataclasses | 11KB | ✅ Imported |

**Recommended Phased Integration**:

1. **Phase 1**: Integrate `train_model_init.py` for model creation (~400 LOC reduction)
2. **Phase 2**: Integrate `train_setup.py` for optimizer/scheduler setup (~300 LOC reduction)
3. **Phase 3**: Refactor 116-parameter signature to use `FullTrainingConfig` dataclass
4. **Phase 4**: Extract training loop to use `train_loop.py` (~2,000 LOC reduction)

**Priority**: Medium (deferred to Q1 2026 due to complexity)

**Note**: The extracted modules exist but full integration was deferred due to:

- Risk of breaking cluster training jobs
- Extensive parameter passing complexity
- Need for comprehensive test coverage

---

## Verification Commands

Check for remaining deprecated imports:

```bash
# Check for deprecated coordination imports
grep -r "from app.coordination.lambda_idle_daemon import" --include="*.py" .
grep -r "from app.coordination.model_distribution_daemon import" --include="*.py" .
grep -r "from app.coordination.replication_monitor import" --include="*.py" .

# Check for deprecated distributed imports
grep -r "from app.distributed.unified_data_sync import" --include="*.py" .
```

Run deprecation warning tests:

```bash
# Enable deprecation warnings
python3 -W default::DeprecationWarning -m pytest tests/unit/coordination/ -v
```
