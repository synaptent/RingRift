# Consolidation Roadmap

> **Last Updated**: 2025-12-17
> **Status**: ✅ All Priorities Complete

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

### Training Signal Consolidation (Priority 2) ✅ COMPLETE

**Created:** `app/training/unified_signals.py`

Central signal computation engine consolidating logic from 5 systems:

- `TrainingTriggers` - Now delegates to `get_signal_computer()`
- `FeedbackAccelerator` - Uses `_signal_computer` for intensity mapping
- `ModelLifecycleManager` - Integrated with unified signals
- `PromotionController` - Uses `get_signal_computer()` for decisions
- `OptimizedPipeline` - Imports unified signal computation

**Key Components:**

- `UnifiedSignalComputer` - Thread-safe computation with 5s caching
- `TrainingSignals` - Immutable snapshot of all computed metrics
- `TrainingUrgency` enum (CRITICAL, HIGH, NORMAL, LOW, NONE)
- Urgency-to-intensity mapping for FeedbackAccelerator

**Test Coverage (2025-12-17):** 31 tests in `tests/unit/training/test_unified_signals.py`

- TrainingUrgency enum values and ordering
- TrainingSignals dataclass defaults, summary, and serialization
- UnifiedSignalComputer initialization, computation, caching, thread safety
- Urgency computation logic (NONE → CRITICAL based on thresholds)
- Elo trend computation (positive/negative/stable)
- Singleton pattern verification
- Convenience function tests (should_train, get_urgency, get_training_intensity)
- Integration verification with TrainingTriggers and FeedbackAccelerator

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

### Threshold Constant Migration (Phase 3) ✅ COMPLETE

Updated additional scripts to import from `app/config/thresholds.py`:

| Module                                         | Constants Migrated                     | Status      |
| ---------------------------------------------- | -------------------------------------- | ----------- |
| `scripts/run_distributed_tournament.py`        | INITIAL_ELO_RATING (TierStats default) | ✅ Complete |
| `scripts/run_p2p_elo_tournament.py`            | INITIAL_ELO_RATING, ELO_K_FACTOR       | ✅ Complete |
| `scripts/launch_distributed_elo_tournament.py` | INITIAL_ELO_RATING, ELO_K_FACTOR       | ✅ Complete |
| `scripts/fix_elo_database.py`                  | INITIAL_RATING, K_FACTOR               | ✅ Complete |
| `scripts/unified_loop/training.py`             | INITIAL_ELO_RATING (method defaults)   | ✅ Complete |
| `scripts/unified_loop/selfplay.py`             | INITIAL_ELO_RATING (method defaults)   | ✅ Complete |

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

## Priority 2: Training Decision Logic ✅ COMPLETE

> **Status**: Complete (2025-12-17) - UnifiedSignalComputer created and integrated

### Completed Work

**Created:** `app/training/unified_signals.py`

Central signal computation engine that all training decision systems now use:

**Features:**

- `UnifiedSignalComputer` - Thread-safe central computation engine
- `TrainingSignals` - Immutable dataclass with all computed metrics
- `TrainingUrgency` enum (CRITICAL, HIGH, NORMAL, LOW, NONE)
- Per-config state tracking with `ConfigTrainingState`
- Elo trend computation via linear regression
- Short-term caching with 5s TTL
- Convenience functions: `should_train()`, `get_urgency()`, `get_training_intensity()`

**Systems Integrated:**

| System                | File                      | Integration                              |
| --------------------- | ------------------------- | ---------------------------------------- |
| TrainingTriggers      | `training_triggers.py`    | ✅ Delegates to `get_signal_computer()`  |
| FeedbackAccelerator   | `feedback_accelerator.py` | ✅ Uses `_signal_computer` for intensity |
| ModelLifecycleManager | `model_lifecycle.py`      | ✅ Uses unified signals                  |
| PromotionController   | `promotion_controller.py` | ✅ Uses `get_signal_computer()`          |
| OptimizedPipeline     | `optimized_pipeline.py`   | ✅ Imports `get_signal_computer`         |

**Usage:**

```python
from app.training.unified_signals import get_signal_computer, TrainingUrgency

computer = get_signal_computer()
signals = computer.compute_signals(
    current_games=10000,
    current_elo=1650.0,
    config_key='square8_2p',
)

if signals.should_train:
    print(f"Training triggered: {signals.reason}")
    print(f"Urgency: {signals.urgency.value}")
    print(f"Priority: {signals.priority}")
```

**Urgency Mapping:**

- `CRITICAL` → HOT_PATH (2.0x intensity) - Regression detected
- `HIGH` → ACCELERATED (1.5x intensity) - Threshold exceeded significantly
- `NORMAL` → NORMAL (1.0x intensity) - Threshold met
- `LOW` → REDUCED (0.75x intensity) - Approaching threshold
- `NONE` → PAUSED (0.5x intensity) - No training needed

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

## Priority 5: Promotion Pipeline ✅ COMPLETE

> **Status**: Complete (2025-12-17) - Unified criteria across promotion systems

### Completed Work

**Unified thresholds via `PromotionCriteria`:**

All promotion systems now use `PromotionCriteria` from `promotion_controller.py`:

1. **AutoPromoter** (`app/training/model_registry.py`):
   - Updated `__init__` to accept optional `criteria: PromotionCriteria`
   - Falls back to `PromotionCriteria()` defaults when no explicit values provided
   - Backward compatible - explicit `min_elo_improvement`, `min_games`, `min_win_rate` still work
   - 17 tests pass

2. **PromotionGate** (`app/integration/model_lifecycle.py`):
   - Updated `__init__` to accept optional `criteria: UnifiedCriteria`
   - Added property getters: `min_elo_improvement`, `min_games_for_production`, `min_win_rate`
   - Uses `LifecycleConfig` values via criteria bridge

3. **PromotionController** - Already the canonical source of `PromotionCriteria`

**Unified Default Values:**
| Parameter | Value | Source |
|-----------|-------|--------|
| min_elo_improvement | 25.0 | PromotionCriteria |
| min_games_played | 50 | PromotionCriteria |
| min_win_rate | 0.52 | PromotionCriteria |
| confidence_threshold | 0.95 | PromotionCriteria |

**Usage:**

```python
from app.training.promotion_controller import PromotionCriteria
from app.training.model_registry import AutoPromoter

# Use unified defaults
promoter = AutoPromoter(registry)  # Uses PromotionCriteria defaults

# Or customize
criteria = PromotionCriteria(min_elo_improvement=30.0)
promoter = AutoPromoter(registry, criteria=criteria)
```

**Remaining (Optional):**

- Add webhook/notification dispatch to PromotionController
- Migrate CMAESRegistryIntegration to use PromotionCriteria

---

## Implementation Order

Based on risk/impact analysis:

| Phase | Task                 | Priority | Risk | Status      |
| ----- | -------------------- | -------- | ---- | ----------- |
| 1     | Config consolidation | P1       | Med  | ✅ Complete |
| 2     | Regression detection | P3       | Med  | ✅ Complete |
| 3     | Model sync           | P4       | Low  | ✅ Complete |
| 4     | Training signals     | P2       | High | ✅ Complete |
| 5     | Promotion pipeline   | P5       | Med  | ✅ Complete |

**All consolidation priorities complete as of 2025-12-17.**

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
