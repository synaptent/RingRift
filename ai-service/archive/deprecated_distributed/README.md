# Deprecated Distributed Infrastructure Modules

This directory contains deprecated distributed infrastructure modules that have been superseded by the unified coordination framework.

## Archived Modules

### sync_orchestrator.py

**Status**: Deprecated December 2025, archived December 26, 2025

**Deprecated Since**: December 2025 (PendingDeprecationWarning added lines 70-80)

**Archive Date**: December 26, 2025

**Will Be Removed**: Q2 2026

#### Why It Was Deprecated

`SyncOrchestrator` was a complex, monolithic orchestrator that wrapped 5+ separate sync components:

- SyncCoordinator (data sync)
- SyncScheduler (scheduling) - itself deprecated
- UnifiedDataSync - deprecated
- EloSyncManager
- RegistrySyncManager

The module attempted to provide a unified API but became overly complex with:

1. **Multiple Initialization Modes**: Lazy component loading, singleton management, complex config objects
2. **State Tracking Complexity**: Tracking last sync times, error counts, component statuses across multiple subsystems
3. **Event Subscription Maze**: Complex event wiring for quality events, stage events, sync triggers
4. **Poor Composability**: Required full initialization even for simple sync operations
5. **Configuration Burden**: `SyncOrchestratorConfig` with 10+ parameters for each component

#### What Replaced It

**SyncFacade** (`app/coordination/sync_facade.py`) provides a simpler, unified interface:

```python
from app.coordination.sync_facade import sync

# Sync all data types
await sync("all")

# Sync specific data with routing
await sync("games", board_type="hex8", priority="high")

# Sync models
await sync("models", model_ids=["canonical_hex8_2p"])
```

**Key improvements:**

- Single function call instead of orchestrator instance management
- Automatic routing to appropriate sync subsystems
- Simplified configuration (no complex config objects)
- Better error handling and logging
- Event emission built-in

#### Migration Guide

**Before (deprecated)**:

```python
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.initialize()
result = await orchestrator.sync_all()
status = orchestrator.get_status()
await orchestrator.shutdown()
```

**After (recommended)**:

```python
from app.coordination.sync_facade import sync

# Simple one-liner
result = await sync("all")

# Or specific targets
await sync("games", board_type="hex8")
await sync("models")
await sync("elo")
await sync("registry")
```

**For scheduling** (instead of `orchestrator.run_scheduler()`):

```python
from app.coordination.daemon_manager import DaemonManager, DaemonType

manager = DaemonManager()
await manager.start_daemon(DaemonType.AUTO_SYNC)
```

**For quality-driven sync prioritization**:

```python
from app.coordination.sync_router import SyncRouter

router = SyncRouter()
priority_configs = router.get_priority_sync_order()
```

#### Breaking Changes

1. **No singleton orchestrator:** Use `sync()` function directly
2. **No explicit initialization:** Subsystems initialize on-demand
3. **No state tracking:** Each sync call is independent
4. **No scheduler loop:** Use daemon manager for background sync
5. **Simplified events:** Events emitted automatically via `sync_facade`

#### Feature Mapping

| Old Feature         | New Location                                       | Notes                |
| ------------------- | -------------------------------------------------- | -------------------- |
| `sync_all()`        | `sync("all")`                                      | Single function call |
| `sync_data()`       | `sync("games")` or `sync("training")`              | More specific        |
| `sync_models()`     | `sync("models")`                                   | Same functionality   |
| `sync_elo()`        | `sync("elo")`                                      | Same functionality   |
| `sync_registry()`   | `sync("registry")`                                 | Same functionality   |
| `run_scheduler()`   | `DaemonManager.start_daemon(DaemonType.AUTO_SYNC)` | Background daemon    |
| `needs_sync()`      | Handled internally by sync router                  | Automatic            |
| `get_status()`      | `SyncRouter.get_sync_status()`                     | Focused status       |
| Event subscriptions | Built into `sync_facade`                           | Automatic            |
| Quality integration | `SyncRouter` with `UnifiedQualityScorer`           | Cleaner separation   |

#### Timeline

- **Dec 24, 2025:** `SyncFacade` introduced in coordination consolidation
- **Dec 26, 2025:** `SyncOrchestrator` archived (this file)
- **Jan 2026 (planned):** Remove from `app/distributed/` after final migration check
- **Q2 2026:** Complete removal from codebase

#### Compatibility

The deprecated module will remain functional at `app/distributed/sync_orchestrator.py` until Q1-Q2 2026. However:

- **No new features** will be added
- **Bug fixes** only for critical issues
- **Import warnings** will be shown (PendingDeprecationWarning)
- **Migration assistance** available via this README

#### See Also

- `app/coordination/sync_facade.py` - Replacement implementation
- `app/coordination/sync_router.py` - Intelligent sync routing
- `app/coordination/daemon_manager.py` - Background sync scheduling
- `archive/deprecated_coordination/README.md` - Other deprecated coordination modules
- `docs/CONSOLIDATION_STATUS_2025_12_19.md` - Architecture decisions

---

### unified_data_sync.py

**Status**: Deprecated December 2025, archived December 26, 2025

**Deprecated Since**: December 2025 (deprecation warnings added)

**Archive Date**: December 26, 2025

**Will Be Removed**: Q2 2026

#### Why It Was Deprecated

`UnifiedDataSyncService` was a monolithic 2170-line module that attempted to consolidate all data synchronization functionality into a single service. While this approach seemed logical at the time, it created several problems:

1. **Tight Coupling**: Combined transport logic (SSH, P2P, Aria2), manifest management, WAL, deduplication, quality extraction, and aggregation into one class
2. **Hard to Test**: Monolithic design made unit testing difficult - mocking required extensive setup
3. **Poor Separation of Concerns**: Mixed different responsibilities (sync, validation, quality scoring, aggregation)
4. **Configuration Complexity**: Required massive YAML configs with nested sections for every feature
5. **Difficult to Extend**: Adding new sync backends or features required modifying the core class

#### What Replaced It

The functionality has been split into focused, composable modules in `app/coordination/`:

**Primary Replacement**:

- **`AutoSyncDaemon`** (`app/coordination/auto_sync_daemon.py`): Automated P2P data synchronization
  - Push-from-generator model: Games synced immediately after generation
  - Gossip replication: Peer-to-peer eventual consistency
  - Excludes coordinator nodes automatically
  - NFS-aware: Skips unnecessary sync when shared storage available

**Supporting Components**:

- **`SyncCoordinator`** (`app/distributed/sync_coordinator.py`): Direct sync API (`sync_games()`, `sync_models()`, `sync_training_data()`)
- **`SyncFacade`** (`app/coordination/sync_facade.py`): Simplified sync interface
- **`DataManifest`** (`app/distributed/unified_manifest.py`): Standalone manifest tracking
- **`UnifiedWAL`** (`app/distributed/unified_wal.py`): Crash-safe write-ahead log
- **`ContentDeduplicator`** (`app/distributed/content_deduplication.py`): Hash-based deduplication
- **`ManifestReplicator`** (`app/distributed/manifest_replication.py`): Distributed manifest replication
- **`UnifiedQualityScorer`** (`app/coordination/unified_quality.py`): Quality-based prioritization

#### Migration Guide

**Before (deprecated)**:

```python
from app.distributed.unified_data_sync import UnifiedDataSyncService

service = UnifiedDataSyncService.from_config(
    config_path=Path("config/unified_loop.yaml")
)
await service.run()
```

**After (recommended)**:

```python
# For automated continuous sync
from app.coordination import AutoSyncDaemon

daemon = AutoSyncDaemon()
await daemon.start()
```

**For one-time sync operations**:

```python
from app.coordination.sync_facade import sync

# Sync everything
await sync("all")

# Sync specific categories
await sync("games")
await sync("models")
await sync("training_data")
```

**For programmatic sync control**:

```python
from app.distributed.sync_coordinator import SyncCoordinator

coordinator = SyncCoordinator()
stats = await coordinator.sync_games(
    source_hosts=["runpod-h100", "nebius-h100-1"],
    target_dir=Path("data/games/synced")
)
```

#### Configuration Migration

**Old YAML config** (`config/unified_loop.yaml`):

```yaml
data_ingestion:
  poll_interval_seconds: 60
  enable_p2p_fallback: true
  enable_gossip_sync: true
  enable_quality_extraction: true
  # ... 50+ more options
```

**New approach** (environment variables + smart defaults):

```bash
# AutoSyncDaemon uses intelligent defaults
export RINGRIFT_SYNC_INTERVAL=60
export RINGRIFT_ENABLE_GOSSIP=true
export RINGRIFT_COORDINATOR_NODE=nebius-backbone-1

python -m app.coordination.auto_sync_daemon
```

Or use the programmatic API with minimal config:

```python
from app.coordination import AutoSyncDaemon

daemon = AutoSyncDaemon(
    poll_interval=60,
    enable_gossip=True
)
await daemon.start()
```

#### CLI Migration

**Old CLI**:

```bash
python -m app.distributed.unified_data_sync \
  --config config/unified_loop.yaml \
  --hosts config/remote_hosts.yaml
```

**New CLI** (via daemon launcher):

```bash
python scripts/launch_daemons.py --sync-only
```

Or directly:

```bash
python -m app.coordination.auto_sync_daemon
```

#### Feature Mapping

| Old Feature           | New Location                   | Notes                   |
| --------------------- | ------------------------------ | ----------------------- |
| SSH/rsync sync        | `SyncCoordinator.sync_games()` | Simplified API          |
| P2P HTTP fallback     | `AutoSyncDaemon`               | Built-in fallback chain |
| Gossip sync           | `AutoSyncDaemon`               | Default enabled         |
| WAL for crash safety  | `UnifiedWAL`                   | Standalone module       |
| Content deduplication | `ContentDeduplicator`          | Standalone, reusable    |
| Quality extraction    | `UnifiedQualityScorer`         | Unified scoring         |
| Manifest replication  | `ManifestReplicator`           | Fault-tolerant manifest |
| Aggregation mode      | `DataPipelineOrchestrator`     | Pipeline-based          |
| Aria2 transport       | `Aria2Transport`               | Standalone transport    |
| Circuit breaker       | `CircuitBreaker`               | Reusable across modules |
| Bandwidth manager     | `sync_bandwidth.py`            | Adaptive bandwidth      |

#### Testing Migration

**Old tests** (hard to write):

```python
# Required extensive mocking
def test_unified_sync():
    mock_manifest = Mock()
    mock_replicator = Mock()
    # ... 20 more mocks
    service = UnifiedDataSyncService(config, hosts, manifest, ...)
    # Test one small piece
```

**New tests** (focused, composable):

```python
def test_auto_sync_daemon():
    daemon = AutoSyncDaemon()
    # Test just the daemon logic

def test_sync_coordinator():
    coordinator = SyncCoordinator()
    # Test just sync operations

def test_content_deduplicator():
    dedup = ContentDeduplicator()
    # Test just deduplication
```

#### Benefits of New Architecture

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Composable**: Mix and match components as needed
3. **Testable**: Small, focused modules are easy to unit test
4. **Extensible**: Add new sync backends without modifying core code
5. **Configuration**: Smart defaults reduce YAML configuration by 80%
6. **Performance**: Push-from-generator model reduces latency vs. polling
7. **Reliability**: Gossip replication provides eventual consistency without central coordination

#### Deprecation Timeline

- **Dec 2025**: Deprecation warnings added to all entry points
- **Jan 2026**: New coordination framework becomes default
- **Mar 2026**: Documentation updated to recommend new framework
- **Q2 2026**: `unified_data_sync.py` removed from codebase

#### Compatibility

The deprecated module will remain functional until Q2 2026. However:

- **No new features** will be added
- **Bug fixes** only for critical issues
- **Migration assistance** available via CLAUDE.md

#### Questions?

See:

- `app/coordination/README.md` - Coordination framework overview
- `docs/CONSOLIDATION_STATUS_2025_12_19.md` - Architecture decisions
- `CLAUDE.md` - Migration examples and patterns

---

**Note**: This module is preserved for reference and to support legacy deployments during the migration period. Do not use for new code.
