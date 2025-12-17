# Consolidation Roadmap

> **Last Updated**: 2025-12-17
> **Status**: In Progress

This document outlines identified consolidation opportunities in the RingRift AI Service codebase and provides a roadmap for addressing them.

## Overview

The codebase has evolved with multiple contributors and use cases, resulting in some duplication and parallel implementations. This roadmap prioritizes consolidation work based on impact and risk.

## Recent Progress (December 2025)

### Completed Quick Wins

1. ✅ **Unified thresholds.py** - `app/config/thresholds.py` created with all canonical threshold values
2. ✅ **IntegratedEnhancementsConfig** - Added to canonical `app/config/unified_config.py`
3. ✅ **Factory functions** - `create_training_manager()` added to unified_config
4. ✅ **Both import paths** - Verified working: `app.config.unified_config` and `scripts.unified_loop.config`
5. ✅ **Cross-references** - Header comments added pointing to canonical locations

### Regression Detection Consolidation (Priority 3) ✅ COMPLETE

**Created:** `app/training/regression_detector.py`

A unified regression detection component consolidating logic from:

- `rollback_manager.py` (Elo drop, win rate drop, error rate)
- `promotion_controller.py` (consecutive regression tracking)
- `feedback_accelerator.py` (momentum-based regression state)

**Features:**

- `RegressionSeverity` enum (MINOR, MODERATE, SEVERE, CRITICAL)
- `RegressionConfig` with canonical thresholds from `app/config/thresholds.py`
- `RegressionEvent` dataclass for detected events
- `RegressionListener` protocol for pub/sub pattern
- Consecutive regression tracking with automatic severity escalation
- Cooldown and minimum games requirements

**Usage:**

```python
from app.training import RegressionDetector, get_regression_detector

detector = get_regression_detector()
detector.set_baseline('model_v42', elo=1500, win_rate=0.55)
event = detector.check_regression('model_v42', current_elo=1440, games_played=100)
if event:
    print(f"Regression: {event.severity.name} - {event.reason}")
```

### Dead Code Cleanup ✅ COMPLETE

**Removed Files:**

- `app/training/reanalyze.py` - Deprecated stub (pointed to scripts/reanalyze_replay_dataset.py)
- `app/training/test_overfit.py` - Manual test script
- `app/training/distributed_training.py` - Superseded by `distributed_unified.py`

**Removed Orphan Directories:**

- `app/training/ai-service/` - Empty nested structure
- `app/training/app/` - Empty artifact
- `app/training/logs/` - Stale training data

**Code Savings:** ~1200 lines removed

### Distributed Module Consolidation ✅ COMPLETE

- Updated `optimized_pipeline.py` to import from `distributed_unified.py`
- Added backward-compatible aliases in `distributed_unified.py`
- Added missing helper functions to `distributed.py`:
  - `is_distributed()`, `get_local_rank()`, `synchronize()`
  - `reduce_tensor()`, `all_gather_object()`, `broadcast_object()`
  - `get_device_for_rank()`
- Updated `DistributedMetrics` with `.add()` and `.reduce_and_reset()` methods
- Fixed `scale_learning_rate()` to support optional `world_size`
- All 52 distributed training tests pass

### Threshold Constant Migration (Phase 1)

Updated core modules to import from `app/config/thresholds.py`:

| Module                                 | Constants Migrated                                  | Status      |
| -------------------------------------- | --------------------------------------------------- | ----------- |
| `app/tournament/elo.py`                | INITIAL_ELO_RATING                                  | ✅ Complete |
| `app/training/elo_service.py`          | INITIAL_ELO_RATING, ELO_K_FACTOR, MIN_GAMES_FOR_ELO | ✅ Complete |
| `app/training/auto_tournament.py`      | INITIAL_ELO_RATING, ELO_K_FACTOR                    | ✅ Complete |
| `app/training/training_triggers.py`    | INITIAL*ELO_RATING, TRAINING*\* constants           | ✅ Complete |
| `app/training/background_eval.py`      | INITIAL_ELO_RATING, ELO_DROP_ROLLBACK               | ✅ Complete |
| `app/training/feedback_accelerator.py` | INITIAL_ELO_RATING                                  | ✅ Complete |
| `app/training/unified_orchestrator.py` | INITIAL_ELO_RATING                                  | ✅ Complete |
| `app/config/unified_config.py`         | INITIAL_ELO_RATING, ELO_K_FACTOR, etc.              | ✅ Complete |

**Pattern Used:**

```python
try:
    from app.config.thresholds import INITIAL_ELO_RATING
except ImportError:
    INITIAL_ELO_RATING = 1500.0  # Fallback for standalone usage
```

### Health Function Analysis

Analyzed 4 `get_health_summary()` implementations:

- **`app/main.py`** - Stateless HTTP endpoint (no init required)
- **`app/routes/health.py`** - FastAPI router version
- **`app/services/model_registry.py`** - ModelRegistry.get_health_summary()
- **`app/training/model_lifecycle.py`** - ModelLifecycleManager.get_health_summary()

**Conclusion:** These serve different purposes (stateless vs registry-based) and are not candidates for consolidation. Documented as parallel implementations for different contexts.

## Priority 1: Configuration Consolidation

> **Status**: Partially Complete (Quick wins done, full merge pending)

### Issue: Duplicate Config Classes

**Files:**

- `app/config/unified_config.py` (~970 lines) - Canonical location, 10+ importers
- `scripts/unified_loop/config.py` (~1080 lines) - Extended version, 2 importers

**Duplicated Classes:**
| Class | app/config | scripts/unified_loop |
|-------|------------|---------------------|
| DataIngestionConfig | ✓ Basic | ✓ Extended (WAL, P2P) |
| TrainingConfig | ✓ Basic | ✓ Extended (SWA, EMA, focal) |
| EvaluationConfig | ✓ | ✓ |
| PromotionConfig | ✓ | ✓ |
| CurriculumConfig | ✓ | ✓ |

**Unique Classes:**

- `app/config`: SafeguardsConfig, BoardConfig, RegressionConfig, AlertingConfig, SafetyConfig, PlateauDetectionConfig, ReplayBufferConfig, ClusterConfig, SSHConfig, SelfplayConfig
- `scripts/unified_loop`: PBTConfig, NASConfig, PERConfig, FeedbackConfig, P2PClusterConfig, ModelPruningConfig, IntegratedEnhancementsConfig, UnifiedLoopConfig, DataEventType, DataEvent

**Consolidation Plan:**

1. Merge extended fields from `scripts/unified_loop/config.py` into `app/config/unified_config.py`
2. Add missing classes (PBTConfig, NASConfig, etc.) to canonical location
3. Create re-export shim in `scripts/unified_loop/config.py`:
   ```python
   # Backward compatibility - import from canonical location
   from app.config.unified_config import (
       DataIngestionConfig,
       TrainingConfig,
       # ... all classes
   )
   ```
4. Update tests to verify both import paths work

**Risk:** Medium - many files import these configs
**Impact:** High - single source of truth for all configuration
**Effort:** 4-6 hours

---

## Priority 2: Training Decision Logic

### Issue: Multiple Competing Training Trigger Systems

**Systems Identified:**
| System | File | Purpose |
|--------|------|---------|
| TrainingTriggers | `training_triggers.py` | Simplified 3-signal system |
| FeedbackAccelerator | `feedback_accelerator.py` | Elo momentum + intensity |
| ModelLifecycleManager | `model_lifecycle.py` | Embedded trigger logic |
| PromotionController | `promotion_controller.py` | Regression detection |
| OptimizedPipeline | `optimized_pipeline.py` | Partial training logic |

**Problem:** These systems independently evaluate similar conditions:

- All track "games since training"
- All monitor Elo metrics
- All implement different decision thresholds
- Risk of conflicting decisions

**Consolidation Plan:**

1. Create `UnifiedTrainingSignal` class that computes all signals once
2. Adapt `TrainingTriggers` to use it (keep as stable API)
3. Refactor `FeedbackAccelerator` to build on unified signals
4. Extract common signal computation from other systems

**Risk:** High - training decisions affect model quality
**Impact:** High - cleaner architecture, consistent decisions
**Effort:** 8-12 hours

---

## Priority 3: Regression/Rollback Detection ✅ COMPLETE

> **Status**: Complete - See "Regression Detection Consolidation" in Recent Progress

### Completed Work

Created `app/training/regression_detector.py` as a unified component:

1. ✅ Extracted `RegressionDetector` as standalone component
2. ✅ Unified thresholds via `RegressionConfig` + `app/config/thresholds.py`
3. ✅ Event-based architecture with `RegressionListener` protocol
4. ✅ Exported via `app/training/__init__.py`

**Next Steps (Optional):**

- Integrate `RollbackManager` to subscribe to detector events
- Refactor `PromotionController` to use unified detector
- Remove inline regression checks from `FeedbackAccelerator`

---

## Priority 4: Model Sync Systems

### Issue: Duplicate Sync Implementations

**Systems:**

- `RegistrySyncManager` - Multi-transport failover, circuit breaker
- `ModelLifecycleManager.ModelSyncCoordinator` - HTTP push/pull

**Problem:** Similar retry/failover logic duplicated

**Consolidation Plan:**

1. Extract common transport layer
2. Unify into single `ClusterSyncManager`
3. Support both model registry and Elo DB sync

**Risk:** Low
**Impact:** Medium - reduced code duplication
**Effort:** 3-4 hours

---

## Priority 5: Promotion Pipeline

### Issue: Multiple Promotion Entry Points

**Entry Points:**

- `ModelRegistry.AutoPromoter` - Stage transitions
- `PromotionController` - A/B testing + auto-promotion
- `ModelLifecycleManager.PromotionGate` - Multi-criteria evaluation
- `CMAESRegistryIntegration` - Heuristic weight promotion

**Problem:** Promotion logic fragmented across classes

**Consolidation Plan:**

1. Create `PromotionPipeline` orchestrator
2. Consolidate criteria into `PromotionEvaluator`
3. Centralize webhook/notification dispatch

**Risk:** Medium
**Impact:** Medium - clearer promotion flow
**Effort:** 6-8 hours

---

## Implementation Order

Based on risk/impact analysis:

| Phase | Task                 | Priority | Risk | Effort |
| ----- | -------------------- | -------- | ---- | ------ |
| 1     | Config consolidation | P1       | Med  | 4-6h   |
| 2     | Regression detection | P3       | Med  | 4-6h   |
| 3     | Model sync           | P4       | Low  | 3-4h   |
| 4     | Training signals     | P2       | High | 8-12h  |
| 5     | Promotion pipeline   | P5       | Med  | 6-8h   |

**Total Estimated Effort:** 25-36 hours

---

## Quick Wins (Low Risk, Immediate Value)

### 1. Add Config Cross-References

Add comments pointing to canonical location:

```python
# scripts/unified_loop/config.py
"""
NOTE: For new projects, prefer importing from app.config.unified_config
This file provides backward compatibility and extended options.
"""
```

### 2. Unify Threshold Constants

Create `app/config/thresholds.py`:

```python
# Training thresholds
TRAINING_TRIGGER_GAMES = 500
TRAINING_MIN_INTERVAL_SECONDS = 1200

# Regression thresholds
ELO_DROP_ROLLBACK = 50
WIN_RATE_DROP_ROLLBACK = 0.10

# Promotion thresholds
ELO_IMPROVEMENT_PROMOTE = 20
MIN_GAMES_PROMOTE = 100
```

### 3. Document Current Architecture

Update ARCHITECTURE_OVERVIEW.md with:

- Current state of each system
- Which file to use for what purpose
- Migration notes for future consolidation

---

## Validation Checklist

Before any consolidation PR:

- [ ] All existing tests pass
- [ ] Both import paths work (backward compat)
- [ ] Config values unchanged (verify with diff)
- [ ] No runtime errors in unified loop
- [ ] Cluster coordination still works

## See Also

- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Training Triggers](TRAINING_TRIGGERS.md)
- [Unified AI Loop](UNIFIED_AI_LOOP.md)
