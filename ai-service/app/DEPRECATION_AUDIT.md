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

| Module                             | Status     | Replacement                   | Notes                            |
| ---------------------------------- | ---------- | ----------------------------- | -------------------------------- |
| `app/tournament/unified_elo_db.py` | DEPRECATED | `app/training/elo_service.py` | Emits warning on import, Q2 2026 |

### Health/Recovery Coordination

| Module                                           | Status     | Replacement                                  | Notes                               |
| ------------------------------------------------ | ---------- | -------------------------------------------- | ----------------------------------- |
| `app/coordination/error_recovery_coordinator.py` | DEPRECATED | `app/coordination/unified_health_manager.py` | Use `get_health_manager()`, Q2 2026 |
| `app/coordination/recovery_manager.py`           | DEPRECATED | `app/coordination/unified_health_manager.py` | Use `wire_health_events()`, Q2 2026 |

## New Unified Components (December 2025)

These new facades consolidate multiple older modules:

| New Component             | Location                                     | Purpose                                             |
| ------------------------- | -------------------------------------------- | --------------------------------------------------- |
| `UnifiedDataValidator`    | `app/training/unified_data_validator.py`     | Consolidates all data validation                    |
| `UnifiedModelStore`       | `app/training/unified_model_store.py`        | Consolidates model registry/versioning/loading      |
| `UnifiedHealthManager`    | `app/coordination/unified_health_manager.py` | Consolidates error recovery and health management   |
| `DataQualityOrchestrator` | `app/quality/data_quality_orchestrator.py`   | Centralized quality event monitoring                |
| `NodeHealthOrchestrator`  | `app/monitoring/node_health_orchestrator.py` | Centralized health event monitoring                 |
| `PEROrchestrator`         | `app/training/per_orchestrator.py`           | PER buffer event monitoring                         |
| `SubscriptionRegistry`    | `app/distributed/subscription_registry.py`   | Event subscription tracking                         |
| `EloService`              | `app/training/elo_service.py`                | Consolidates Elo tracking with training integration |

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

## Legacy Module Migration Plan (11,366 LOC)

### Assessment (December 2025)

| Module                         | LOC   | Classes/Functions | Import Sites     |
| ------------------------------ | ----- | ----------------- | ---------------- |
| `app/ai/_neural_net_legacy.py` | 6,931 | 33                | 154 (via facade) |
| `app/_game_engine_legacy.py`   | 4,435 | 4                 | 154 (via facade) |

### Migration Strategy

**Phase 1: Split neural_net_legacy (High Priority)**

1. Extract constants → `app/ai/neural_net/constants.py` (already exists)
2. Extract CNN blocks → `app/ai/neural_net/blocks.py` (already exists)
3. Extract architectures → `app/ai/neural_net/architectures.py`
4. Extract encoding → `app/ai/neural_net/encoding.py`
5. Keep NeuralNetAI class until NNUE migration complete

**Phase 2: Consolidate game_engine_legacy (Medium Priority)**

1. Already has GameEngine as single class
2. Extract PhaseRequirement types → `app/game_engine/phase_requirements.py`
3. Consider splitting GameEngine methods into mixins

**Phase 3: Update facades**

1. Update `app/ai/neural_net/__init__.py` to import from new modules
2. Update `app/game_engine/__init__.py` to import from new modules
3. Keep re-exports for backwards compatibility

**Phase 4: Remove legacy files**

1. Verify all imports resolve to new modules
2. Run full test suite
3. Delete `_neural_net_legacy.py` and `_game_engine_legacy.py`

### Blockers

- NNUE migration must complete before NeuralNetAI can be removed
- 154 import sites need verification after each phase

## Files to Review for Removal

Priority order for cleanup:

1. `app/training/game_quality_scorer.py` - Simple replacement available
2. `app/distributed/data_sync.py` - Wrapper module only
3. `app/distributed/data_sync_robust.py` - Wrapper module only
4. `app/training/distributed.py` - Complex, needs careful migration

## Additional Migration Guides

### Health Coordination Migration

**Before (Deprecated)**

```python
# OLD: Using deprecated ErrorRecoveryCoordinator
from app.coordination.error_recovery_coordinator import ErrorRecoveryCoordinator
coordinator = ErrorRecoveryCoordinator()
coordinator.track_error("training", "model_failure", ErrorSeverity.ERROR)
if coordinator.is_circuit_broken("training"):
    handle_failure()

# OLD: Using deprecated RecoveryManager
from app.coordination.recovery_manager import RecoveryManager
manager = RecoveryManager()
await manager.recover_stuck_job(work_item, expected_timeout=300)
```

**After (Current - December 2025)**

```python
# NEW: Using UnifiedHealthManager
from app.coordination.unified_health_manager import (
    get_health_manager,
    wire_health_events,
)

# Wire health events at startup
manager = wire_health_events()

# Track errors (unified interface)
manager.track_error("training", "model_failure", ErrorSeverity.ERROR)

# Check circuit breaker
if manager.is_circuit_broken("training"):
    handle_failure()

# Recover stuck job
result = await manager.recover_stuck_job(work_item, expected_timeout=300)

# Get unified health stats
stats = manager.get_health_stats()
```

### Elo Database Migration

**Before (Deprecated)**

```python
# OLD: Using deprecated unified_elo_db
from app.tournament.unified_elo_db import get_elo_database
db = get_elo_database()
db.record_match_and_update(...)
```

**After (Current - December 2025)**

```python
# NEW: Using EloService (SSoT for training integration)
from app.training.elo_service import get_elo_service
elo = get_elo_service()
elo.record_match(
    participant_a=...,
    participant_b=...,
    winner=...,
    board_type=...,
    num_players=...,
)
```
