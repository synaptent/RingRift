# Deprecation Audit - December 2025

This document tracks deprecated modules and their replacements as part of the consolidation effort.

## Status Legend

- `DEPRECATED` - Module still exists but should not be used
- `SUPERSEDED` - Replaced by a unified module
- `REMOVE_SOON` - Scheduled for removal in next cleanup phase

## Deprecated Modules

### Data Sync Layer

| Module                                     | Status      | Replacement                                          | Notes                               |
| ------------------------------------------ | ----------- | ---------------------------------------------------- | ----------------------------------- |
| `app/distributed/data_sync.py`             | **REMOVED** | `UnifiedDataSyncService` from `unified_data_sync.py` | Deleted December 2025               |
| `app/distributed/data_sync_robust.py`      | **REMOVED** | `UnifiedDataSyncService` from `unified_data_sync.py` | Deleted December 2025               |
| `app/training/data_pipeline_controller.py` | DEPRECATED  | Direct use of data loading APIs                      | Emits deprecation warning on import |

### Cluster Coordination

| Module                                   | Status     | Replacement                  | Notes            |
| ---------------------------------------- | ---------- | ---------------------------- | ---------------- |
| `app/distributed/cluster_coordinator.py` | SUPERSEDED | `app/coordination/*` modules | Being phased out |

### Configuration

| Module                                         | Status     | Replacement                    | Notes               |
| ---------------------------------------------- | ---------- | ------------------------------ | ------------------- |
| `app/training/config.py`                       | SUPERSEDED | `app/config/unified_config.py` | Old training config |
| `app/config/training_config.py` (some classes) | SUPERSEDED | `UnifiedTrainingConfig`        | Legacy classes      |

### Quality Scoring

| Module                                | Status     | Replacement                      | Notes                      |
| ------------------------------------- | ---------- | -------------------------------- | -------------------------- |
| `app/training/game_quality_scorer.py` | DEPRECATED | `app/quality/unified_quality.py` | Use `UnifiedQualityScorer` |

### Write-Ahead Log

| Module                           | Status     | Replacement  | Notes         |
| -------------------------------- | ---------- | ------------ | ------------- |
| `WriteAheadLog` (unified_wal.py) | DEPRECATED | `UnifiedWAL` | Wrapper class |
| `IngestionWAL` (unified_wal.py)  | DEPRECATED | `UnifiedWAL` | Wrapper class |

### Training Infrastructure

| Module                                               | Status     | Replacement              | Notes                    |
| ---------------------------------------------------- | ---------- | ------------------------ | ------------------------ |
| `app/training/distributed.py`                        | DEPRECATED | `distributed_unified.py` | Old distributed training |
| `app/training/advanced_training.py` (some functions) | PARTIAL    | Newer training modules   | Check specific functions |

### Neural Network

| Module                         | Status     | Replacement                 | Notes                                     |
| ------------------------------ | ---------- | --------------------------- | ----------------------------------------- |
| `app/ai/_neural_net_legacy.py` | DEPRECATED | `nnue.py`, `nnue_policy.py` | CNN policy net, being phased out for NNUE |
| `app/ai/neural_net/` (package) | DEPRECATED | `nnue_policy.py`            | Re-exports from legacy module             |

### Tournament/Elo

| Module                             | Status | Replacement       | Notes               |
| ---------------------------------- | ------ | ----------------- | ------------------- |
| `app/tournament/unified_elo_db.py` | CHECK  | May be superseded | Needs investigation |

## New Unified Components (December 2025)

These new facades consolidate multiple older modules:

| New Component             | Location                                     | Purpose                                        |
| ------------------------- | -------------------------------------------- | ---------------------------------------------- |
| `UnifiedDataValidator`    | `app/training/unified_data_validator.py`     | Consolidates all data validation               |
| `UnifiedModelStore`       | `app/training/unified_model_store.py`        | Consolidates model registry/versioning/loading |
| `DataQualityOrchestrator` | `app/quality/data_quality_orchestrator.py`   | Centralized quality event monitoring           |
| `NodeHealthOrchestrator`  | `app/monitoring/node_health_orchestrator.py` | Centralized health event monitoring            |
| `PEROrchestrator`         | `app/training/per_orchestrator.py`           | PER buffer event monitoring                    |
| `SubscriptionRegistry`    | `app/distributed/subscription_registry.py`   | Event subscription tracking                    |

## Migration Guide

### Before (Deprecated - REMOVED December 2025)

```python
# This module has been DELETED - use unified_quality directly
from app.training.game_quality_scorer import GameQualityScorer  # No longer exists
scorer = GameQualityScorer()
quality = scorer.compute_game_quality(game)
```

### After (Current)

```python
from app.quality.unified_quality import get_quality_scorer
scorer = get_quality_scorer()
quality = scorer.compute_game_quality(game_data)
```

### Before (Deprecated - REMOVED December 2025)

```python
# This module has been DELETED - use unified_data_sync directly
from app.distributed.data_sync import RobustDataSync  # No longer exists
sync = RobustDataSync()
```

### After (Current)

```python
from app.distributed.unified_data_sync import UnifiedDataSyncService
sync = UnifiedDataSyncService.get_instance()
```

## Cleanup Tasks

1. **Remove deprecated warning code** after all callers migrated
2. **Delete deprecated modules** after verification period
3. **Update tests** to use new unified modules
4. **Update documentation** to reflect current architecture

## Files to Review for Removal

Priority order for cleanup:

1. `app/training/game_quality_scorer.py` - Simple replacement available
2. `app/distributed/data_sync.py` - Wrapper module only
3. `app/distributed/data_sync_robust.py` - Wrapper module only
4. `app/training/distributed.py` - Complex, needs careful migration
