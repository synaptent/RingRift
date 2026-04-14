# Deprecation Tracker

**Last Updated:** 2025-12-29
**Total Active Deprecations:** 83
**Target Removal:** Q2 2026 (April-June 2026)

This document provides a consolidated, actionable view of all deprecated modules, functions, and patterns in the RingRift AI service.

---

## Quick Summary

| Category              | Count  | Status           |
| --------------------- | ------ | ---------------- |
| Coordination Modules  | 18     | Deprecated/Wired |
| Training Modules      | 12     | Deprecated       |
| AI/Neural Net Modules | 8      | Deprecated       |
| Sync Modules          | 10     | Deprecated       |
| Distributed Modules   | 7      | Deprecated       |
| Daemon Types          | 7      | Deprecated       |
| Factory Functions     | 12     | Wired            |
| Re-Export Modules     | 9      | Deprecated       |
| **Total**             | **83** | Mixed            |

---

## Priority 1: High-Traffic Deprecations (Action Required)

These are actively imported and emit warnings frequently:

| Module/Function          | Location                               | Calls/Day | Replacement                  |
| ------------------------ | -------------------------------------- | --------- | ---------------------------- |
| `queue_populator.py`     | `app/coordination/queue_populator.py`  | High      | `unified_queue_populator.py` |
| `SmartCheckpointManager` | `app/training/advanced_training.py`    | Medium    | `UnifiedCheckpointManager`   |
| `SelfPlayConfig`         | `app/training/config.py`               | Medium    | `SelfplayConfig`             |
| `get_sync_coordinator()` | `app/coordination/sync_coordinator.py` | Medium    | `auto_sync_daemon.py`        |
| `cluster/sync.py`        | `app/coordination/cluster/sync.py`     | Medium    | `sync_facade.py`             |
| `data_events.py` aliases | `app/distributed/data_events.py`       | High      | Direct enum imports          |

### Verification Commands

```bash
# Find all imports of deprecated modules
cd ai-service
grep -r "from app.coordination.queue_populator import" --include="*.py" | grep -v archive
grep -r "from app.training.config import SelfPlayConfig" --include="*.py" | grep -v archive
grep -r "from app.coordination.sync_coordinator import" --include="*.py" | grep -v archive
```

---

## Priority 2: Archived Modules (Q2 2026 Removal)

These modules have been archived to `archive/deprecated_*/` but still exist for backward compatibility:

### Coordination Archive (`archive/deprecated_coordination/`)

| Module                                  | Archived Date | Replacement                       |
| --------------------------------------- | ------------- | --------------------------------- |
| `_deprecated_queue_populator_daemon.py` | Dec 27, 2025  | `unified_queue_populator.py`      |
| `_deprecated_replication_monitor.py`    | Dec 27, 2025  | `unified_replication_daemon.py`   |
| `_deprecated_replication_repair.py`     | Dec 27, 2025  | `unified_replication_daemon.py`   |
| `_deprecated_model_distribution.py`     | Dec 27, 2025  | `unified_distribution_daemon.py`  |
| `_deprecated_npz_distribution.py`       | Dec 27, 2025  | `unified_distribution_daemon.py`  |
| `_deprecated_vast_idle.py`              | Dec 26, 2025  | `unified_idle_shutdown_daemon.py` |
| `_deprecated_lambda_idle.py`            | Dec 26, 2025  | `unified_idle_shutdown_daemon.py` |
| `_deprecated_auto_evaluation.py`        | Dec 27, 2025  | `evaluation_daemon.py`            |
| `_deprecated_sync_safety.py`            | Dec 2025      | `sync_integrity.py`               |
| `_deprecated_base_event_handler.py`     | Dec 2025      | `handler_base.py`                 |
| `_deprecated_base_handler.py`           | Dec 2025      | `handler_base.py`                 |

### P2P Archive (`archive/deprecated_p2p/`)

| Module                          | Archived Date | Replacement                |
| ------------------------------- | ------------- | -------------------------- |
| `_deprecated_gossip_metrics.py` | Dec 28, 2025  | `gossip_protocol_mixin.py` |

### Training Archive (`archive/deprecated_training/`)

| Module                              | Archived Date | Replacement                |
| ----------------------------------- | ------------- | -------------------------- |
| `_deprecated_incremental_export.py` | Dec 2025      | `export_replay_dataset.py` |

### AI Archive (`archive/deprecated_ai/`)

| Module                   | Archived Date | Replacement           |
| ------------------------ | ------------- | --------------------- |
| `_neural_net_legacy.py`  | Dec 2025      | `app/ai/neural_net/*` |
| `_game_engine_legacy.py` | Dec 2025      | `app/rules/fsm.py`    |
| `_deprecated_legacy.py`  | Dec 2025      | Various replacements  |

---

## Priority 3: Deprecated Daemon Types

| DaemonType           | Status     | Replacement                      |
| -------------------- | ---------- | -------------------------------- |
| `SYNC_COORDINATOR`   | Deprecated | `AUTO_SYNC`                      |
| `HEALTH_CHECK`       | Deprecated | `NODE_HEALTH_MONITOR`            |
| `CLUSTER_DATA_SYNC`  | Deprecated | `AUTO_SYNC` (strategy=broadcast) |
| `EPHEMERAL_SYNC`     | Deprecated | `AUTO_SYNC` (strategy=ephemeral) |
| `MODEL_DISTRIBUTION` | Deprecated | `UNIFIED_DISTRIBUTION`           |
| `NPZ_DISTRIBUTION`   | Deprecated | `UNIFIED_DISTRIBUTION`           |
| `REPLICATION_REPAIR` | Deprecated | `UNIFIED_REPLICATION`            |

---

## Priority 4: Factory Function Deprecations

These factory functions emit warnings but redirect to unified implementations:

### `app/coordination/unified_distribution_daemon.py`

```python
# Deprecated (emit warning, still work)
create_model_distribution_daemon()  # → unified
create_npz_distribution_daemon()    # → unified
get_distribution_manager()          # → unified

# Canonical
UnifiedDistributionDaemon()
```

### `app/coordination/unified_idle_shutdown_daemon.py`

```python
# Deprecated (emit warning, still work)
create_lambda_idle_daemon()  # → UnifiedIdleShutdownDaemon(provider=lambda)
create_vast_idle_daemon()    # → UnifiedIdleShutdownDaemon(provider=vast)

# Canonical
UnifiedIdleShutdownDaemon(provider=..., config=...)
```

### `app/coordination/unified_replication_daemon.py`

```python
# Deprecated (emit warning, still work)
create_replication_monitor()       # → unified
create_replication_repair_daemon() # → unified

# Canonical
UnifiedReplicationDaemon()
```

### `app/coordination/auto_sync_daemon.py`

```python
# Deprecated (emit warning, still work)
get_sync_coordinator()   # → auto_sync_daemon
reset_sync_coordinator() # → auto_sync_daemon.reset()

# Canonical
AutoSyncDaemon()
```

---

## Priority 5: Re-Export Modules (Shims)

These modules exist only to re-export from canonical locations with deprecation warnings:

| Re-Export Module                               | Canonical Location                    |
| ---------------------------------------------- | ------------------------------------- |
| `app/coordination/queue_populator.py`          | `unified_queue_populator.py`          |
| `app/coordination/cluster/sync.py`             | `sync_facade.py`                      |
| `app/coordination/bandwidth_manager.py`        | `sync_bandwidth.py`                   |
| `app/coordination/sync_safety.py`              | `sync_integrity.py`                   |
| `app/coordination/base_handler.py`             | `handler_base.py`                     |
| `app/coordination/event_subscription_mixin.py` | `handler_base.py`                     |
| `app/core/singleton_mixin.py`                  | `app/coordination/singleton_mixin.py` |
| `app/distributed/cluster_monitor.py`           | `cluster_status_monitor.py`           |
| `scripts/p2p/gossip_metrics.py`                | `gossip_protocol_mixin.py`            |

---

## Priority 6: Training Module Deprecations

| Module                                                 | Location                            | Replacement                                                        |
| ------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------ |
| `archive/deprecated_training/orchestrated_training.py` | `archive/deprecated_training/`      | `unified_orchestrator.py` + `app.training` compatibility re-export |
| `integrated_enhancements.py`                           | `app/training/`                     | `unified_orchestrator.py`                                          |
| `data_pipeline_controller.py`                          | `app/training/`                     | `data_pipeline_orchestrator.py`                                    |
| `checkpointing.py`                                     | `app/training/`                     | `checkpoint_unified.py`                                            |
| `train_checkpointing.py`                               | `app/training/`                     | `checkpoint_unified.py`                                            |
| `fault_tolerance.retry_with_backoff`                   | `app/training/fault_tolerance.py`   | `app.core.error_handler.retry`                                     |
| `distributed.DistributedTrainer`                       | `app/training/distributed.py`       | `distributed_unified.py`                                           |
| `advanced_training.SmartCheckpoint`                    | `app/training/advanced_training.py` | `UnifiedCheckpointManager`                                         |
| `config.SelfPlayConfig`                                | `app/training/config.py`            | `SelfplayConfig`                                                   |

---

## Priority 7: AI Module Deprecations

| Module       | Location  | Replacement               | Notes               |
| ------------ | --------- | ------------------------- | ------------------- |
| `gmo_ai.py`  | `app/ai/` | `gumbel_search_engine.py` | Wildcard import fix |
| `gmo_v2.py`  | `app/ai/` | `gumbel_search_engine.py` | Wildcard import fix |
| `ebmo_ai.py` | `app/ai/` | `gumbel_search_engine.py` | Wildcard import fix |
| `ig_gmo.py`  | `app/ai/` | `gumbel_search_engine.py` | Explicit imports    |
| `zobrist.py` | `app/ai/` | `app/rules/zobrist.py`    | Moved to rules      |

---

## Migration Checklist

### Before Q2 2026 Removal

- [ ] Update all `queue_populator` imports to `unified_queue_populator`
- [ ] Update all `SelfPlayConfig` to `SelfplayConfig`
- [ ] Update all `sync_coordinator` to `auto_sync_daemon`
- [ ] Update all factory functions to direct instantiation
- [ ] Run deprecation warning scan (see commands below)
- [ ] Delete archive directories after verification

### Verification Commands

```bash
# Run with deprecation warnings visible
cd ai-service
python -W default::DeprecationWarning -c "import app.coordination"

# Count deprecation warnings
python -W error::DeprecationWarning -c "import app.coordination" 2>&1 | wc -l

# Find remaining deprecated imports
grep -r "from app.coordination.queue_populator import" --include="*.py" | grep -v archive | grep -v test

# Verify archived modules have no imports
for f in archive/deprecated_*/*.py; do
  name=$(basename "$f" .py | sed 's/_deprecated_//')
  grep -r "import.*$name" --include="*.py" | grep -v archive | head -1
done
```

---

## Cleanup Progress

### Completed (Dec 2025)

- [x] Archived 11 coordination modules
- [x] Archived 2 P2P modules
- [x] Archived 3 training modules
- [x] Archived 4 AI modules
- [x] Created re-export shims with warnings
- [x] Updated DaemonManager factory
- [x] Updated CLAUDE.md documentation

### Pending (Q1 2026)

- [ ] Update remaining callers of deprecated factory functions
- [ ] Convert re-export modules to hard errors (warnings → ImportError)
- [ ] Remove archived modules from git history

### Q2 2026 Removal

- [ ] Delete `archive/deprecated_*` directories
- [ ] Remove re-export shim modules
- [ ] Update all `__init__.py` exports
- [ ] Final grep verification

---

## Adding New Deprecations

When deprecating a module:

1. Add deprecation warning at module level:

   ```python
   import warnings
   warnings.warn(
       "module_name is deprecated. Use replacement_name instead. "
       "Scheduled for removal in Q2 2026.",
       DeprecationWarning,
       stacklevel=2,
   )
   ```

2. Add entry to this tracker with:
   - Module path
   - Deprecation date
   - Replacement module
   - Estimated removal date

3. Create re-export shim if needed for backward compatibility

4. Update CLAUDE.md with migration guide

---

## See Also

- `DEPRECATION_ROADMAP.md` - Detailed roadmap with migration guides
- `DEPRECATION_TIMELINE.md` - Timeline with phase planning
- `archive/deprecated_coordination/README.md` - Archive documentation
- `CLAUDE.md` - Main documentation with deprecation notes
