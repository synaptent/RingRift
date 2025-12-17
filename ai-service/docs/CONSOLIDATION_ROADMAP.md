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

### Checkpoint Versioning System ✅ COMPLETE

**Verified existing comprehensive system** in `app/training/model_versioning.py`:

**Features:**

- `ModelMetadata` dataclass with version, config, checksums
- `ModelVersionManager` for save/load/migrate operations
- All model classes have `ARCHITECTURE_VERSION` attributes:
  - RingRiftCNN_v2/v3/v4, HexNeuralNet_v2/v3
- Version validation on load (strict/non-strict modes)
- Checksum integrity verification
- Legacy checkpoint migration
- 37 comprehensive tests

**Integration with checkpoint_unified.py:**

- Added architecture metadata fields to `CheckpointMetadata`
- `save_checkpoint()` now accepts `architecture_version`, `model_class`, `model_config`
- `load_checkpoint()` supports `expected_version`, `expected_class`, `strict_version`
- Stores `_versioning_metadata` key compatible with `model_versioning.py`

**Batch Migration Script** created: `scripts/migrate_checkpoints.py`

- Scans directories for legacy checkpoints
- Infers model class from filename patterns
- Supports dry-run, in-place, and output directory modes
- Generates JSON migration reports

**Usage:**

```python
# Save with versioning
manager.save_checkpoint(
    model_state=model.state_dict(),
    progress=progress,
    architecture_version='v2.0.0',
    model_class='RingRiftCNN_v2',
    model_config={'num_filters': 128},
)

# Load with validation
checkpoint = manager.load_checkpoint(
    expected_version='v2.0.0',
    strict_version=True,
)
```

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
| `app/config/unified_config.py`         | INITIAL_ELO_RATING, ELO_K_FACTOR, ELO_DROP_ROLLBACK | ✅ Complete |
| `scripts/unified_loop/config.py`       | INITIAL_ELO_RATING, ELO_DROP_ROLLBACK               | ✅ Complete |
| `scripts/p2p/constants.py`             | INITIAL_ELO_RATING, ELO_K_FACTOR                    | ✅ Complete |
| `scripts/p2p_orchestrator.py`          | Uses constants from p2p/constants.py                | ✅ Complete |

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

## Priority 1: Configuration Consolidation ✅ SUBSTANTIALLY COMPLETE

> **Status**: Complete (2025-12-17) - Core configs migrated, re-export shim active

### Completed Work (2025-12-17)

1. ✅ **Added missing classes to canonical `app/config/unified_config.py`:**
   - `PBTConfig` - Population-Based Training
   - `NASConfig` - Neural Architecture Search
   - `PERConfig` - Prioritized Experience Replay
   - `FeedbackConfig` - Pipeline feedback controller
   - `P2PClusterConfig` - P2P cluster integration
   - `ModelPruningConfig` - Automated model pruning

2. ✅ **Created re-export shim in `scripts/unified_loop/config.py`:**
   - Imports migrated classes from canonical location
   - Backward compatibility maintained (both import paths work)
   - Local definitions removed, replaced with canonical imports

3. ✅ **Verified both import paths work:**
   - `from app.config.unified_config import PBTConfig` ✓
   - `from scripts.unified_loop.config import PBTConfig` ✓ (re-export)
   - Classes are identical (`PBTConfig is CanonicalPBT` = True)

4. ✅ **UnifiedConfig updated:**
   - Version bumped to 1.2
   - New fields: `pbt`, `nas`, `per`, `feedback`, `p2p`, `model_pruning`
   - `_from_dict()` updated to load from YAML

### Remaining (Low Priority - Deferred)

**Extended TrainingConfig fields:** The scripts version has 150+ experimental training fields. These remain in `scripts/unified_loop/config.py` as an extended version. Full merge deferred due to:

- High risk of breakage
- Many fields are experimental
- Current setup provides sufficient consolidation

**Files:**

- `app/config/unified_config.py` - Canonical location (now ~1100 lines)
- `scripts/unified_loop/config.py` - Extended version with re-exports (2 importers)

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

## Priority 4: Model Sync Systems ✅ COMPLETE

> **Status**: Complete (2025-12-17) - Unified transport layer extracted

### Completed Work

**Created:** `app/coordination/cluster_transport.py`

A unified transport layer for cluster communication, extracted from RegistrySyncManager:

**Features:**

- `ClusterTransport` class with multi-transport failover (Tailscale -> SSH -> HTTP)
- `CircuitBreaker` class for fault tolerance (now shared across modules)
- `NodeConfig` dataclass for node configuration
- `TransportResult` dataclass for operation results
- Async-first design with configurable timeouts
- Singleton pattern via `get_cluster_transport()`

**Integration:**

- `RegistrySyncManager` now imports `CircuitBreaker` from `cluster_transport`
- All 24 registry sync tests pass
- Exported via `app/coordination/__init__.py`

**Usage:**

```python
from app.coordination import (
    ClusterTransport,
    CircuitBreaker,
    NodeConfig,
    get_cluster_transport,
)

transport = get_cluster_transport()
result = await transport.transfer_file(
    local_path=Path("data/model.pth"),
    remote_path="models/latest.pth",
    node=NodeConfig(hostname="lambda-h100"),
)
```

**Additional Enhancement (2025-12-17):**

Created `app/coordination/sync_base.py` - Abstract base class for sync managers:

- `SyncManagerBase` - Base class with common patterns
- `SyncState` - Unified state dataclass with JSON serialization
- `SimpleCircuitBreaker` - Lightweight circuit breaker for per-node fault tolerance
- `try_transports()` - Helper for transport failover orchestration

This provides a foundation for future sync manager consolidation.

**Remaining (Optional):**

- Refactor `ModelSyncCoordinator` to use `ClusterTransport` (not widely used)
- Migrate existing sync managers to inherit from `SyncManagerBase`
- Add ClusterTransport tests

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
