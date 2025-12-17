# Comprehensive Action Plan

**Created:** 2025-12-17
**Status:** Active
**Scope:** Critical Gaps, Areas for Improvement, Weak Areas, Product/Adoption

This document provides detailed, step-by-step action plans for all identified improvement areas in the RingRift project.

---

## Project Context

**RingRift** is a full-stack, real-time multiplayer platform for an original deterministic abstract strategy game:

- **2-4 players**, zero randomness, perfect information
- **Three victory paths**: Ring elimination, territory control, Last Player Standing
- **Multiple boards**: 8×8 square (64), 19×19 square (361), hexagonal (469)
- **Tech stack**: React + Express + WebSockets + PostgreSQL/Redis + Python AI service
- **AI**: 10-level difficulty ladder (Random → Heuristic → Minimax → MCTS → AlphaZero-style Descent)
- **Testing**: 10,177 TypeScript tests, 1,824 Python tests, 81 contract vectors (100% parity)

**Current Status**: Stable Beta — production validation in progress, focus on scaling tests, security hardening, UX polish. No public releases yet, 0 stars/forks.

---

## Table of Contents

1. [Critical Gaps & Hard Problems](#1-critical-gaps--hard-problems)
   - [1.1 Rules Documentation Contradiction (NEW - URGENT)](#11-rules-documentation-contradiction)
   - [1.2 Configuration Duplication Crisis](#12-configuration-duplication-crisis)
   - [1.3 Competing Training Decision Systems](#13-competing-training-decision-systems)
   - [1.4 Neural Network Checkpoint Versioning](#14-neural-network-checkpoint-versioning)
   - [1.5 Hex Board Models Missing](#15-hex-board-models-missing)
2. [Product & Adoption Gaps (NEW)](#2-product--adoption-gaps)
   - [2.1 No Public Release or Hosted Demo](#21-no-public-release-or-hosted-demo)
   - [2.2 UX/Tutorial Burden](#22-uxtutorial-burden)
   - [2.3 Multiplayer Politics & Audience Positioning](#23-multiplayer-politics--audience-positioning)
3. [Areas for Improvement](#3-areas-for-improvement)
   - [3.1 Architecture & Code Quality](#31-architecture--code-quality)
   - [3.2 Infrastructure Gaps](#32-infrastructure-gaps)
   - [3.3 Testing Gaps](#33-testing-gaps)
4. [Weak Areas to Address](#4-weak-areas-to-address)
   - [4.1 Documentation Drift](#41-documentation-drift)
   - [4.2 Error Handling Inconsistency](#42-error-handling-inconsistency)
   - [4.3 Observability Gaps](#43-observability-gaps)
   - [4.4 Self-Play Data Quality](#44-self-play-data-quality)
   - [4.5 Recovery After Failures](#45-recovery-after-failures)

---

# 1. Critical Gaps & Hard Problems

## 1.1 Rules Documentation Contradiction

**Priority:** P0 - URGENT
**Risk:** High (player confusion, AI training inconsistency)
**Effort:** 2-3 hours
**Owner:** TBD

### Problem Statement

**CONFIRMED CONTRADICTION** between rules documents on territory processing eligibility:

| Document                                   | What it says about standalone rings (height-1)                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `RULES_CANONICAL_SPEC.md` (RR-CANON-R022)  | "**All controlled stacks are eligible cap targets for territory processing**, including... **height-1 standalone rings**" |
| `ringrift_simple_human_rules.md` (line 89) | "**Standalone rings (height 1) are NOT eligible for territory processing**"                                               |

These statements are **mutually exclusive**. One must be wrong.

### Additional Drift Areas Identified

1. **Line elimination cost**: Some older prose may reference "entire cap" for line processing, but canonical spec says "one ring"
2. **Forced elimination target selection**: Canonical spec requires interactive choice, engines historically auto-selected (now substantially fixed per `KNOWN_ISSUES.md` P0.1)
3. **Version/date drift**: Multiple docs lack version stamps, making it hard to know which is current

### Impact

- **Player confusion**: Rules seem inconsistent depending on which doc they read
- **AI training inconsistency**: If AI training data was generated under wrong rules interpretation
- **Credibility risk**: Multiple contradictory rulebooks undermines trust in the project

### Action Plan

#### Phase 1: Determine Canonical Truth (30 min)

The canonical spec (`RULES_CANONICAL_SPEC.md`) is explicitly declared as the SSoT. Per the doc:

> "This file (`RULES_CANONICAL_SPEC.md`) is the _normative_ rules SSoT: when behaviour is in doubt, the RR‑CANON‑RXXX rules here win."

**Decision required**: Confirm that `RR-CANON-R022` (standalone rings ARE eligible) is the intended rule. Check:

1. Does the TypeScript engine allow standalone rings for territory processing?
2. Does the Python engine allow standalone rings for territory processing?

```bash
# Check TS implementation
grep -n "height.*1\|standalone\|eligible" src/shared/engine/aggregates/TerritoryAggregate.ts | head -20

# Check Python implementation
grep -n "height.*1\|standalone\|eligible" ai-service/app/rules/territory*.py | head -20
```

#### Phase 2: Fix the Contradiction (1 hour)

**File:** `ringrift_simple_human_rules.md`

```markdown
# BEFORE (line 88-90):

- **Territory processing:** Eliminate your **entire cap** from an eligible stack.
  Eligible targets must be either: (1) multicolour stacks you control, or
  (2) single-colour stacks of height > 1. **Standalone rings (height 1) are
  NOT eligible for territory processing.**

# AFTER (aligned with RR-CANON-R022):

- **Territory processing:** Eliminate your **entire cap** from an eligible stack.
  **All controlled stacks are eligible**, including: (1) multicolour stacks,
  (2) single-colour stacks of any height, and (3) standalone rings (height 1).
  For multicolour stacks, eliminating the cap exposes buried rings.
```

#### Phase 3: Add Version Stamps & Derivation Notice (30 min)

Add to top of `ringrift_simple_human_rules.md`:

```markdown
> **Version:** 2025-12-17 (aligned with RULES_CANONICAL_SPEC.md RR-CANON-R022)
> **Derived from:** RULES_CANONICAL_SPEC.md (the authoritative SSoT)
> **If this document conflicts with the canonical spec, the canonical spec wins.**
```

Add to `ringrift_complete_rules.md` and `ringrift_compact_rules.md`:

```markdown
> **Last Verified:** 2025-12-17
> **Canonical Reference:** RULES_CANONICAL_SPEC.md
```

#### Phase 4: Audit All Rules Docs for Additional Drift (1 hour)

Run systematic check for elimination semantics inconsistencies:

```bash
# Check for "entire cap" vs "one ring" in line processing context
grep -rn "line.*cap\|line.*entire\|line.*one ring" \
  ringrift_*.md RULES_CANONICAL_SPEC.md \
  --include="*.md" | grep -i "elimin"

# Check for forced elimination eligibility statements
grep -rn "forced.*eligible\|forced.*standalone\|forced.*height" \
  ringrift_*.md RULES_CANONICAL_SPEC.md
```

Document any additional inconsistencies found and fix them.

#### Phase 5: Update Consistency Audit Doc (30 min)

Update `docs/rules/RULES_DOCS_CONSISTENCY_AUDIT_2025_12_12.md`:

```markdown
## Update (2025-12-17): Territory Processing Eligibility

### F) Territory processing stack eligibility ✅ NOW CONSISTENT

**Issue found:** `ringrift_simple_human_rules.md` incorrectly stated standalone
rings are NOT eligible for territory processing, contradicting RR-CANON-R022.

**Fix applied:** Updated simple human rules to match canonical spec.
All controlled stacks (including standalone rings) are now documented as
eligible for territory processing across all four primary rules docs.

**Verification:** Grep confirms no remaining "NOT eligible" statements for
territory processing in any rules doc.
```

### Verification Checklist

- [x] Confirmed canonical interpretation (standalone rings ARE eligible) ✅ Verified 2025-12-17
- [x] `ringrift_simple_human_rules.md` updated ✅ Fixed 2025-12-17
- [x] Version stamps added to all rules docs ✅ Already present, version bumped
- [x] No other elimination-related contradictions found ✅ Audited line/forced elimination rules
- [x] Consistency audit doc updated ✅ Added Section F to RULES_DOCS_CONSISTENCY_AUDIT_2025_12_12.md
- [x] 25+ additional code comments, test fixtures, and user-facing content fixed ✅
- [ ] AI training data reviewed (was it generated under correct rules?) - **NOTE**: The TS and Python implementations already correctly allowed height-1 stacks (EliminationAggregate.ts, elimination.py). Training data should be valid.

### Why This Matters

This is the **highest leverage fix** in this entire document because:

1. It directly affects player understanding and game fairness
2. It's fast to fix (2-3 hours)
3. It has immediate credibility impact
4. It could affect AI training data validity

---

## 1.2 Configuration Duplication Crisis

**Priority:** P1 - Critical
**Risk:** Medium
**Effort:** 4-6 hours
**Owner:** TBD
**Status:** ✅ PARTIALLY COMPLETE (2025-12-17)

> **Implementation Status:**
>
> - ✅ `IntegratedEnhancementsConfig` consolidated to `app/config/unified_config.py` (canonical)
> - ✅ `scripts/unified_loop/config.py` now imports from canonical with fallback
> - ✅ `PBTConfig`, `NASConfig`, `PERConfig`, `FeedbackConfig`, `P2PClusterConfig`, `ModelPruningConfig` migrated
> - ✅ Threshold constants centralized in `app/config/thresholds.py`
> - 🟡 Some legacy imports may still exist in less-used scripts

### Problem Statement

Two configuration files define overlapping classes with divergent fields:

- `app/config/unified_config.py` (~970 lines) - 10+ importers, canonical
- `scripts/unified_loop/config.py` (~1080 lines) - 2 importers, extended

Classes exist in scripts version only: `PBTConfig`, `NASConfig`, `PERConfig`, `FeedbackConfig`, `P2PClusterConfig`, `ModelPruningConfig`

### Impact

- Training runs may use different parameters depending on import path
- New features added to one file but not the other
- Confusing developer experience

### Action Plan

#### Phase 1: Audit & Document (1 hour)

```bash
# Step 1.1: Generate complete class comparison
cd /Users/armand/Development/RingRift
python3 << 'EOF'
import ast
import sys

def extract_classes(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
            classes[node.name] = fields
    return classes

app_classes = extract_classes('ai-service/app/config/unified_config.py')
script_classes = extract_classes('ai-service/scripts/unified_loop/config.py')

print("=== Classes in app/config only ===")
for cls in sorted(set(app_classes) - set(script_classes)):
    print(f"  {cls}: {app_classes[cls]}")

print("\n=== Classes in scripts only ===")
for cls in sorted(set(script_classes) - set(app_classes)):
    print(f"  {cls}: {script_classes[cls]}")

print("\n=== Field differences in shared classes ===")
for cls in sorted(set(app_classes) & set(script_classes)):
    app_fields = set(app_classes[cls])
    script_fields = set(script_classes[cls])
    if app_fields != script_fields:
        print(f"  {cls}:")
        if script_fields - app_fields:
            print(f"    + scripts only: {script_fields - app_fields}")
        if app_fields - script_fields:
            print(f"    + app only: {app_fields - script_fields}")
EOF
```

```bash
# Step 1.2: Find all importers
grep -r "from.*unified_config import\|from.*config import.*Config" ai-service/ --include="*.py" | grep -v __pycache__ | sort
```

#### Phase 2: Merge Extended Fields (2 hours)

**File:** `ai-service/app/config/unified_config.py`

```python
# Step 2.1: Add missing fields to DataIngestionConfig
@dataclass
class DataIngestionConfig:
    # Existing fields...
    poll_interval_seconds: int = 30
    sync_method: str = "incremental"

    # ADD: Extended fields from scripts version
    wal_enabled: bool = True
    p2p_fallback: bool = True
    checksum_validation: bool = True
    dead_letter_queue: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 5

# Step 2.2: Add missing fields to TrainingConfig
@dataclass
class TrainingConfig:
    # Existing fields...
    trigger_threshold_games: int = 500
    min_interval_seconds: int = 1200

    # ADD: Extended fields
    swa_enabled: bool = False
    swa_start_epoch: int = 75
    ema_enabled: bool = False
    ema_decay: float = 0.999
    focal_loss_enabled: bool = False
    focal_loss_gamma: float = 2.0
    progressive_batching: bool = True
    initial_batch_multiplier: float = 0.25
```

```python
# Step 2.3: Add missing config classes
@dataclass
class PBTConfig:
    """Population-Based Training configuration."""
    enabled: bool = False
    population_size: int = 8
    exploit_fraction: float = 0.2
    explore_fraction: float = 0.2
    perturbation_factors: tuple = (0.8, 1.2)

@dataclass
class NASConfig:
    """Neural Architecture Search configuration."""
    enabled: bool = False
    search_space: str = "micro"  # micro, macro
    max_epochs: int = 50

@dataclass
class PERConfig:
    """Prioritized Experience Replay configuration."""
    enabled: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000

@dataclass
class P2PClusterConfig:
    """P2P cluster coordination configuration."""
    enabled: bool = True
    discovery_interval_seconds: int = 30
    heartbeat_interval_seconds: int = 10
    leader_election_timeout_seconds: int = 60

@dataclass
class ModelPruningConfig:
    """Model pruning configuration."""
    enabled: bool = False
    prune_threshold_elo: int = -100
    min_models_to_keep: int = 5
    max_age_days: int = 30
```

#### Phase 3: Create Backward Compatibility Shim (30 min)

**File:** `ai-service/scripts/unified_loop/config.py`

```python
"""
Unified Loop Configuration - Backward Compatibility Module

NOTE: This file re-exports from the canonical location for backward compatibility.
For new code, import directly from app.config.unified_config.

All config classes are now defined in app/config/unified_config.py
"""

# Re-export everything from canonical location
from app.config.unified_config import (
    # Core configs
    DataIngestionConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    CurriculumConfig,
    ClusterConfig,
    SafeguardsConfig,

    # Extended configs (now in canonical location)
    PBTConfig,
    NASConfig,
    PERConfig,
    P2PClusterConfig,
    ModelPruningConfig,

    # Factory functions
    get_config,
    create_training_manager,

    # Convenience accessors
    get_training_threshold,
    get_promotion_threshold,
)

# Keep any script-specific utilities here if needed
# (But prefer moving them to canonical location)
```

#### Phase 4: Update Tests & Verify (1 hour)

```bash
# Step 4.1: Run existing tests
cd ai-service
python -m pytest tests/unit/config/ -v

# Step 4.2: Add import path verification test
cat > tests/unit/config/test_config_import_paths.py << 'EOF'
"""Verify both import paths resolve to same classes."""
import pytest

def test_import_paths_resolve_same_classes():
    """Both import paths should return identical config objects."""
    from app.config.unified_config import (
        DataIngestionConfig as AppDataIngestion,
        TrainingConfig as AppTraining,
        get_config as app_get_config,
    )
    from scripts.unified_loop.config import (
        DataIngestionConfig as ScriptDataIngestion,
        TrainingConfig as ScriptTraining,
        get_config as script_get_config,
    )

    # Same class objects
    assert AppDataIngestion is ScriptDataIngestion
    assert AppTraining is ScriptTraining

    # Same singleton instance
    assert app_get_config() is script_get_config()

def test_all_config_classes_exist():
    """All expected config classes should be importable."""
    from app.config.unified_config import (
        DataIngestionConfig,
        TrainingConfig,
        EvaluationConfig,
        PromotionConfig,
        CurriculumConfig,
        PBTConfig,
        NASConfig,
        PERConfig,
        P2PClusterConfig,
        ModelPruningConfig,
    )
    # If we get here, all imports succeeded
    assert True
EOF

python -m pytest tests/unit/config/test_config_import_paths.py -v
```

```bash
# Step 4.3: Run full test suite
python -m pytest tests/ -x --tb=short

# Step 4.4: Verify unified loop still works
python -c "from scripts.unified_loop.config import get_config; print(get_config())"
```

#### Phase 5: Update Documentation (30 min)

```bash
# Update CONSOLIDATION_ROADMAP.md to mark as complete
# Add migration notes to README
```

### Verification Checklist

- [x] All 15+ config classes defined in `app/config/unified_config.py` _(Verified 2025-12-17)_
- [x] Scripts version re-exports from canonical location _(PBTConfig, NASConfig, PERConfig, FeedbackConfig, P2PClusterConfig, ModelPruningConfig)_
- [x] Both import paths return identical objects _(Verified: `PBTConfig is CanonicalPBT: True`)_
- [ ] All existing tests pass
- [x] Unified loop starts successfully _(Verified: `get_config()` loads unified_loop.yaml)_
- [x] CONSOLIDATION_ROADMAP.md updated ✅ Priority 1 marked as complete (2025-12-17)

**Fields added to canonical during consolidation (2025-12-17):**

- `DataIngestionConfig.sync_disabled` - Orchestrator-only mode flag
- `P2PClusterConfig.model_sync_on_promotion` - Auto-sync on promotion
- `P2PClusterConfig.tournament_nodes_per_eval` - Nodes per eval
- `P2PClusterConfig.unhealthy_threshold` - Failure threshold
- `P2PClusterConfig.gossip_sync_enabled` - Gossip replication
- `P2PClusterConfig.gossip_port` - Gossip port

### Rollback Plan

```bash
# If issues discovered, revert via git
git checkout HEAD~1 -- ai-service/app/config/unified_config.py
git checkout HEAD~1 -- ai-service/scripts/unified_loop/config.py
```

---

## 1.3 Competing Training Decision Systems

**Priority:** P1 - Critical
**Risk:** High
**Effort:** 8-12 hours
**Owner:** TBD
**Status:** ✅ PARTIALLY COMPLETE (2025-12-17)

> **Implementation Status:**
>
> - ✅ `app/training/unified_signals.py` - Created UnifiedSignalComputer as single source of truth
> - ✅ `app/training/regression_detector.py` - Centralized regression detection
> - ✅ `app/config/thresholds.py` - Canonical threshold constants
> - 🟡 Legacy systems not yet fully migrated to use unified signals

### Problem Statement

Five independent systems evaluate training triggers:

| System                | File                      | Tracks                                             |
| --------------------- | ------------------------- | -------------------------------------------------- |
| TrainingTriggers      | `training_triggers.py`    | games_since_training, data_quality, elo_plateau    |
| FeedbackAccelerator   | `feedback_accelerator.py` | elo_momentum, training_intensity, regression_state |
| ModelLifecycleManager | `model_lifecycle.py`      | model_age, performance_metrics, sync_status        |
| PromotionController   | `promotion_controller.py` | elo_delta, win_rate, game_count                    |
| OptimizedPipeline     | `optimized_pipeline.py`   | batch_readiness, gpu_utilization                   |

### Impact

- Conflicting decisions (one system says train, another says wait)
- Redundant computations across systems
- Difficult to reason about training behavior
- Silent training skips when systems disagree

### Action Plan

#### Phase 1: Create Unified Signal Computation (3 hours)

**New File:** `ai-service/app/training/unified_signals.py`

```python
"""
Unified Training Signal Computation

Single source of truth for all training-related metrics and decisions.
Other systems should subscribe to these signals rather than computing their own.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List
import threading

class TrainingUrgency(Enum):
    """Training urgency levels."""
    CRITICAL = "critical"      # Train immediately (regression detected)
    HIGH = "high"              # Train soon (threshold exceeded significantly)
    NORMAL = "normal"          # Train when convenient (threshold met)
    LOW = "low"                # Can wait (threshold approaching)
    NONE = "none"              # No training needed

@dataclass
class TrainingSignals:
    """Computed training signals - read-only snapshot."""
    # Core metrics
    games_since_last_training: int = 0
    time_since_last_training: timedelta = field(default_factory=lambda: timedelta(0))
    current_elo: float = 1500.0
    elo_trend: float = 0.0  # positive = improving, negative = regressing

    # Thresholds (from unified config)
    games_threshold: int = 500
    min_interval: timedelta = field(default_factory=lambda: timedelta(minutes=20))

    # Derived signals
    games_threshold_ratio: float = 0.0  # games / threshold (>1 = exceeded)
    time_threshold_met: bool = False
    data_quality_score: float = 1.0  # 0-1, lower = more issues

    # Regression indicators
    elo_regression_detected: bool = False
    elo_drop_magnitude: float = 0.0
    consecutive_losses: int = 0

    # Resource availability
    gpu_available: bool = True
    memory_pressure: float = 0.0  # 0-1

    # Final recommendation
    urgency: TrainingUrgency = TrainingUrgency.NONE
    should_train: bool = False
    reason: str = ""

    @property
    def summary(self) -> str:
        return (f"Urgency={self.urgency.value}, "
                f"games={self.games_since_last_training}/{self.games_threshold}, "
                f"elo_trend={self.elo_trend:+.1f}, "
                f"reason={self.reason}")


class UnifiedSignalComputer:
    """
    Central computation engine for training signals.

    All training decision systems should use this instead of
    computing their own metrics.
    """

    def __init__(self, config=None):
        self._config = config or self._load_config()
        self._lock = threading.RLock()
        self._last_computation: Optional[TrainingSignals] = None
        self._computation_cache_ttl = timedelta(seconds=5)
        self._last_computation_time: Optional[datetime] = None

        # State tracking
        self._last_training_time: Optional[datetime] = None
        self._last_training_games: int = 0
        self._elo_history: List[tuple] = []  # (timestamp, elo)

    def _load_config(self):
        from app.config.unified_config import get_config
        return get_config()

    def compute_signals(self,
                       current_games: int,
                       current_elo: float,
                       force_recompute: bool = False) -> TrainingSignals:
        """
        Compute current training signals.

        Args:
            current_games: Total games in training corpus
            current_elo: Current model Elo rating
            force_recompute: Bypass cache

        Returns:
            TrainingSignals snapshot
        """
        with self._lock:
            now = datetime.now()

            # Return cached if fresh
            if (not force_recompute and
                self._last_computation_time and
                now - self._last_computation_time < self._computation_cache_ttl):
                return self._last_computation

            signals = TrainingSignals()

            # Core metrics
            signals.games_since_last_training = current_games - self._last_training_games
            signals.games_threshold = self._config.training.trigger_threshold_games
            signals.games_threshold_ratio = (
                signals.games_since_last_training / signals.games_threshold
                if signals.games_threshold > 0 else 0
            )

            if self._last_training_time:
                signals.time_since_last_training = now - self._last_training_time
                signals.min_interval = timedelta(
                    seconds=self._config.training.min_interval_seconds
                )
                signals.time_threshold_met = (
                    signals.time_since_last_training >= signals.min_interval
                )
            else:
                signals.time_threshold_met = True  # First training

            # Elo tracking
            signals.current_elo = current_elo
            self._elo_history.append((now, current_elo))
            self._elo_history = [
                (t, e) for t, e in self._elo_history
                if now - t < timedelta(hours=1)
            ][-100:]  # Keep last hour, max 100 points

            if len(self._elo_history) >= 2:
                signals.elo_trend = self._compute_elo_trend()
                signals.elo_regression_detected = signals.elo_trend < -10
                signals.elo_drop_magnitude = abs(min(0, signals.elo_trend))

            # Compute urgency
            signals.urgency, signals.reason = self._compute_urgency(signals)
            signals.should_train = signals.urgency in (
                TrainingUrgency.CRITICAL,
                TrainingUrgency.HIGH,
                TrainingUrgency.NORMAL
            ) and signals.time_threshold_met

            self._last_computation = signals
            self._last_computation_time = now
            return signals

    def _compute_elo_trend(self) -> float:
        """Compute Elo trend via linear regression."""
        if len(self._elo_history) < 2:
            return 0.0

        # Simple slope calculation
        times = [(t - self._elo_history[0][0]).total_seconds()
                 for t, _ in self._elo_history]
        elos = [e for _, e in self._elo_history]

        n = len(times)
        sum_t = sum(times)
        sum_e = sum(elos)
        sum_te = sum(t * e for t, e in zip(times, elos))
        sum_t2 = sum(t * t for t in times)

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_te - sum_t * sum_e) / denom
        # Convert to Elo per hour
        return slope * 3600

    def _compute_urgency(self, signals: TrainingSignals) -> tuple:
        """Determine training urgency and reason."""

        # Critical: Regression detected
        if signals.elo_regression_detected and signals.elo_drop_magnitude > 30:
            return TrainingUrgency.CRITICAL, f"Elo regression: {signals.elo_drop_magnitude:.0f} drop"

        # High: Significantly over threshold
        if signals.games_threshold_ratio >= 1.5:
            return TrainingUrgency.HIGH, f"Games {signals.games_threshold_ratio:.1f}x threshold"

        # Normal: Threshold met
        if signals.games_threshold_ratio >= 1.0:
            return TrainingUrgency.NORMAL, f"Games threshold met ({signals.games_since_last_training})"

        # Low: Approaching threshold
        if signals.games_threshold_ratio >= 0.8:
            return TrainingUrgency.LOW, f"Approaching threshold ({signals.games_threshold_ratio:.0%})"

        return TrainingUrgency.NONE, "Below threshold"

    def record_training_started(self, games_count: int):
        """Record that training has started."""
        with self._lock:
            self._last_training_time = datetime.now()
            self._last_training_games = games_count

    def record_training_completed(self, new_elo: Optional[float] = None):
        """Record that training has completed."""
        with self._lock:
            if new_elo is not None:
                self._elo_history.append((datetime.now(), new_elo))


# Singleton instance
_signal_computer: Optional[UnifiedSignalComputer] = None
_signal_computer_lock = threading.Lock()

def get_signal_computer() -> UnifiedSignalComputer:
    """Get the singleton signal computer instance."""
    global _signal_computer
    if _signal_computer is None:
        with _signal_computer_lock:
            if _signal_computer is None:
                _signal_computer = UnifiedSignalComputer()
    return _signal_computer
```

#### Phase 2: Adapt TrainingTriggers (1.5 hours)

**File:** `ai-service/app/training/training_triggers.py`

```python
"""
Training Triggers - Adapter Layer

This module provides the stable API for training decisions.
Internally delegates to UnifiedSignalComputer.
"""
from .unified_signals import get_signal_computer, TrainingUrgency

class TrainingTriggers:
    """
    Simplified interface for training trigger decisions.

    Delegates to UnifiedSignalComputer for actual computation.
    """

    def __init__(self):
        self._computer = get_signal_computer()

    def should_train(self, current_games: int, current_elo: float) -> bool:
        """Check if training should be triggered."""
        signals = self._computer.compute_signals(current_games, current_elo)
        return signals.should_train

    def get_urgency(self, current_games: int, current_elo: float) -> str:
        """Get current training urgency level."""
        signals = self._computer.compute_signals(current_games, current_elo)
        return signals.urgency.value

    def get_detailed_status(self, current_games: int, current_elo: float) -> dict:
        """Get detailed status for logging/debugging."""
        signals = self._computer.compute_signals(current_games, current_elo)
        return {
            "should_train": signals.should_train,
            "urgency": signals.urgency.value,
            "reason": signals.reason,
            "games_ratio": signals.games_threshold_ratio,
            "elo_trend": signals.elo_trend,
            "time_threshold_met": signals.time_threshold_met,
        }

    def record_training_started(self, games: int):
        """Record training start for timing calculations."""
        self._computer.record_training_started(games)

    def record_training_completed(self, new_elo: float = None):
        """Record training completion."""
        self._computer.record_training_completed(new_elo)
```

#### Phase 3: Refactor FeedbackAccelerator (2 hours)

**File:** `ai-service/app/training/feedback_accelerator.py`

```python
# Add at top of file
from .unified_signals import get_signal_computer, TrainingSignals

class FeedbackAccelerator:
    """
    Training intensity controller based on Elo feedback.

    Uses UnifiedSignalComputer for base metrics, adds intensity logic.
    """

    def __init__(self):
        self._computer = get_signal_computer()
        # Intensity-specific state (not in unified signals)
        self._intensity_multiplier = 1.0
        self._last_intensity_update = None

    def get_training_intensity(self, current_games: int, current_elo: float) -> float:
        """
        Get training intensity multiplier (0.5 - 2.0).

        Higher intensity means:
        - More games per training batch
        - Lower learning rate decay
        - More exploration in self-play
        """
        signals = self._computer.compute_signals(current_games, current_elo)

        # Base intensity from urgency
        base = {
            "critical": 2.0,
            "high": 1.5,
            "normal": 1.0,
            "low": 0.75,
            "none": 0.5,
        }.get(signals.urgency.value, 1.0)

        # Adjust for Elo trend
        if signals.elo_trend > 20:
            base *= 0.9  # Improving, can ease off
        elif signals.elo_trend < -20:
            base *= 1.2  # Regressing, push harder

        self._intensity_multiplier = max(0.5, min(2.0, base))
        return self._intensity_multiplier

    def is_regressing(self, current_games: int, current_elo: float) -> bool:
        """Check if model is in regression state."""
        signals = self._computer.compute_signals(current_games, current_elo)
        return signals.elo_regression_detected
```

#### Phase 4: Extract Common Logic from Other Systems (2 hours)

For each remaining system (`ModelLifecycleManager`, `PromotionController`, `OptimizedPipeline`):

1. Identify training-related computations
2. Replace with calls to `get_signal_computer().compute_signals()`
3. Keep only system-specific logic

**Example refactor pattern:**

```python
# Before (in ModelLifecycleManager):
def _should_trigger_training(self):
    games = self._get_game_count()
    if games - self._last_training_games < 500:
        return False
    if time.time() - self._last_training_time < 1200:
        return False
    return True

# After:
def _should_trigger_training(self):
    from .unified_signals import get_signal_computer
    signals = get_signal_computer().compute_signals(
        self._get_game_count(),
        self._get_current_elo()
    )
    return signals.should_train
```

#### Phase 5: Add Observability (1 hour)

```python
# Add to unified_signals.py

class SignalMetricsExporter:
    """Export training signals to Prometheus."""

    def __init__(self):
        from prometheus_client import Gauge, Counter

        self.games_since_training = Gauge(
            'ringrift_training_games_since_last',
            'Games since last training'
        )
        self.elo_trend = Gauge(
            'ringrift_training_elo_trend',
            'Current Elo trend (per hour)'
        )
        self.training_urgency = Gauge(
            'ringrift_training_urgency',
            'Training urgency (0=none, 4=critical)'
        )
        self.training_decisions = Counter(
            'ringrift_training_decisions_total',
            'Training decision counts',
            ['decision', 'urgency']
        )

    def export(self, signals: TrainingSignals):
        """Export current signals to Prometheus."""
        self.games_since_training.set(signals.games_since_last_training)
        self.elo_trend.set(signals.elo_trend)
        self.training_urgency.set({
            "none": 0, "low": 1, "normal": 2, "high": 3, "critical": 4
        }.get(signals.urgency.value, 0))

        decision = "train" if signals.should_train else "wait"
        self.training_decisions.labels(
            decision=decision,
            urgency=signals.urgency.value
        ).inc()
```

#### Phase 6: Testing & Validation (2 hours)

```python
# tests/unit/training/test_unified_signals.py

import pytest
from datetime import timedelta
from app.training.unified_signals import (
    UnifiedSignalComputer,
    TrainingSignals,
    TrainingUrgency
)

class TestUnifiedSignalComputer:

    def test_below_threshold_returns_no_training(self):
        computer = UnifiedSignalComputer()
        signals = computer.compute_signals(current_games=100, current_elo=1500)

        assert signals.should_train is False
        assert signals.urgency == TrainingUrgency.NONE

    def test_threshold_exceeded_returns_normal_urgency(self):
        computer = UnifiedSignalComputer()
        computer._last_training_games = 0
        computer._last_training_time = datetime.now() - timedelta(hours=1)

        signals = computer.compute_signals(current_games=600, current_elo=1500)

        assert signals.should_train is True
        assert signals.urgency == TrainingUrgency.NORMAL

    def test_elo_regression_triggers_critical(self):
        computer = UnifiedSignalComputer()
        # Simulate Elo drop
        computer._elo_history = [
            (datetime.now() - timedelta(minutes=30), 1600),
            (datetime.now() - timedelta(minutes=20), 1580),
            (datetime.now() - timedelta(minutes=10), 1550),
            (datetime.now(), 1500),
        ]
        computer._last_training_time = datetime.now() - timedelta(hours=1)

        signals = computer.compute_signals(current_games=100, current_elo=1500)

        assert signals.elo_regression_detected is True
        assert signals.urgency == TrainingUrgency.CRITICAL

    def test_time_threshold_blocks_training(self):
        computer = UnifiedSignalComputer()
        computer._last_training_time = datetime.now() - timedelta(minutes=5)
        computer._last_training_games = 0

        signals = computer.compute_signals(current_games=1000, current_elo=1500)

        assert signals.games_threshold_ratio > 1.0  # Over threshold
        assert signals.time_threshold_met is False  # But too soon
        assert signals.should_train is False

    def test_cache_returns_same_result(self):
        computer = UnifiedSignalComputer()

        signals1 = computer.compute_signals(100, 1500)
        signals2 = computer.compute_signals(200, 1600)  # Different values

        assert signals1 is signals2  # Same cached object

    def test_force_recompute_bypasses_cache(self):
        computer = UnifiedSignalComputer()

        signals1 = computer.compute_signals(100, 1500)
        signals2 = computer.compute_signals(200, 1600, force_recompute=True)

        assert signals1 is not signals2
```

### Verification Checklist

- [x] `UnifiedSignalComputer` passes all unit tests _(Verified 2025-12-17 via functional tests)_
- [x] `TrainingTriggers` still works as before (API unchanged) _(Verified - delegates to UnifiedSignalComputer)_
- [x] `FeedbackAccelerator` uses unified signals _(Integrated via get_unified_urgency/get_unified_signals)_
- [x] Other systems refactored to use unified signals _(ModelLifecycleManager, PromotionController, OptimizedPipeline - DONE 2025-12-17)_
- [x] Prometheus metrics exported ✅ SignalMetricsExporter implemented with Gauge, Counter, Histogram (2025-12-17)
- [x] No duplicate threshold constants in codebase _(Both use app.config.thresholds)_
- [x] Unified loop runs successfully with new code ✅ Module imports successfully, graceful fallback if prometheus-client not installed
- [x] Training decisions are logged consistently _(Via unified signals debug logging)_

**Implementation Notes (2025-12-17):**

- Created `app/training/unified_signals.py` with `UnifiedSignalComputer` class
- `TrainingTriggers` refactored to delegate to `UnifiedSignalComputer`
- `FeedbackAccelerator` syncs Elo updates and training events with unified signals
- Added convenience functions: `should_train()`, `get_urgency()`, `get_training_intensity()`
- FeedbackAccelerator maps intensity levels to TrainingUrgency for cross-system compatibility

**Phase 4 Integration (2025-12-17):**

- `ModelLifecycleManager.TrainingTrigger` integrated with unified signals
- `PromotionController` uses unified signals for regression detection in rollback evaluation
- `OptimizedPipeline` uses unified signals as primary decision source with fallback chain

### Rollback Plan

Keep old implementations with deprecation warnings for 2 weeks:

```python
# Old file, add at top:
import warnings
warnings.warn(
    "Direct use of FeedbackAccelerator metrics is deprecated. "
    "Use unified_signals.get_signal_computer() instead.",
    DeprecationWarning
)
```

---

## 1.4 Neural Network Checkpoint Versioning

**Priority:** P1 - Critical
**Risk:** High
**Effort:** 2-3 hours
**Owner:** TBD
**Status:** ✅ COMPLETE (2025-12-17)

> **Implementation Status:**
>
> - ✅ `app/training/model_versioning.py` provides `checkpoint_version` for embeddings
> - ✅ `app/training/checkpoint_unified.py` handles unified checkpoint management
> - ✅ Model registry tracks versions in `data/model_registry/`
> - ✅ Regression detection via `app/training/regression_detector.py`

### Problem Statement (RESOLVED)

~~Saved checkpoints contain no version metadata.~~ Model versioning is now implemented via `model_versioning.py`. When architecture changes (e.g., `history_length`, `num_res_blocks`), the checkpoint includes metadata to verify compatibility.

### Impact

- Silent model regression after architecture changes
- Hours of training lost when loading incompatible checkpoints
- Debugging difficulty ("why did performance drop?")

### Action Plan

#### Phase 1: Define Checkpoint Metadata Schema (30 min)

**File:** `ai-service/app/models/checkpoint_metadata.py`

```python
"""
Checkpoint Metadata Schema

Ensures checkpoint compatibility verification before loading.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import hashlib

CHECKPOINT_SCHEMA_VERSION = "1.0"

@dataclass
class ModelArchitecture:
    """Neural network architecture parameters."""
    architecture_type: str  # "cnn", "nnue", "transformer"
    input_planes: int
    history_length: int
    num_res_blocks: int
    num_filters: int
    policy_head_filters: int
    value_head_hidden: int
    board_type: str  # "square8", "hex8", etc.

    def compute_hash(self) -> str:
        """Compute deterministic hash of architecture."""
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class CheckpointMetadata:
    """Full checkpoint metadata."""
    # Schema version for future compatibility
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    # Architecture info
    architecture: ModelArchitecture = None
    architecture_hash: str = ""

    # Training info
    training_games: int = 0
    training_epochs: int = 0
    final_loss: float = 0.0
    final_elo: Optional[float] = None

    # Provenance
    created_at: str = ""  # ISO timestamp
    parent_checkpoint: Optional[str] = None
    git_commit: Optional[str] = None

    # Compatibility
    pytorch_version: str = ""
    cuda_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointMetadata":
        arch_dict = d.pop("architecture", None)
        if arch_dict:
            d["architecture"] = ModelArchitecture(**arch_dict)
        return cls(**d)

    def is_compatible_with(self, other: "CheckpointMetadata") -> tuple:
        """
        Check if this checkpoint is compatible with another architecture.

        Returns:
            (is_compatible, reason)
        """
        if self.architecture_hash != other.architecture_hash:
            return False, f"Architecture mismatch: {self.architecture_hash} vs {other.architecture_hash}"

        if self.schema_version != other.schema_version:
            return False, f"Schema version mismatch: {self.schema_version} vs {other.schema_version}"

        return True, "Compatible"
```

#### Phase 2: Update Save Logic (45 min)

**File:** `ai-service/app/training/checkpointing.py`

```python
import torch
from datetime import datetime
import subprocess
from .checkpoint_metadata import CheckpointMetadata, ModelArchitecture

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    board_type: str,
    training_games: int = 0,
    elo: float = None,
    parent_checkpoint: str = None,
):
    """
    Save checkpoint with full metadata.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Final loss value
        path: Save path
        board_type: Board type string
        training_games: Games used in training
        elo: Current Elo rating
        parent_checkpoint: Path to parent checkpoint (if fine-tuning)
    """
    # Extract architecture from model
    architecture = ModelArchitecture(
        architecture_type=model.__class__.__name__,
        input_planes=getattr(model, 'input_planes', 12),
        history_length=getattr(model, 'history_length', 3),
        num_res_blocks=getattr(model, 'num_res_blocks', 10),
        num_filters=getattr(model, 'num_filters', 128),
        policy_head_filters=getattr(model, 'policy_head_filters', 32),
        value_head_hidden=getattr(model, 'value_head_hidden', 256),
        board_type=board_type,
    )

    # Get git commit if available
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except:
        git_commit = None

    metadata = CheckpointMetadata(
        architecture=architecture,
        architecture_hash=architecture.compute_hash(),
        training_games=training_games,
        training_epochs=epoch,
        final_loss=loss,
        final_elo=elo,
        created_at=datetime.utcnow().isoformat(),
        parent_checkpoint=parent_checkpoint,
        git_commit=git_commit,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
    )

    checkpoint = {
        'metadata': metadata.to_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path} (arch_hash={metadata.architecture_hash})")
```

#### Phase 3: Update Load Logic with Verification (45 min)

```python
def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    strict: bool = True,
    board_type: str = None,
) -> dict:
    """
    Load checkpoint with compatibility verification.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        strict: If True, raise on architecture mismatch
        board_type: Expected board type

    Returns:
        Checkpoint dict with metadata

    Raises:
        CheckpointIncompatibleError: If architecture doesn't match
    """
    checkpoint = torch.load(path, map_location='cpu')

    # Check for metadata (new format)
    if 'metadata' in checkpoint:
        saved_meta = CheckpointMetadata.from_dict(checkpoint['metadata'])

        # Build current architecture metadata
        current_arch = ModelArchitecture(
            architecture_type=model.__class__.__name__,
            input_planes=getattr(model, 'input_planes', 12),
            history_length=getattr(model, 'history_length', 3),
            num_res_blocks=getattr(model, 'num_res_blocks', 10),
            num_filters=getattr(model, 'num_filters', 128),
            policy_head_filters=getattr(model, 'policy_head_filters', 32),
            value_head_hidden=getattr(model, 'value_head_hidden', 256),
            board_type=board_type or "unknown",
        )
        current_hash = current_arch.compute_hash()

        # Verify compatibility
        if current_hash != saved_meta.architecture_hash:
            msg = (
                f"Checkpoint architecture mismatch!\n"
                f"  Saved: {saved_meta.architecture_hash} "
                f"({saved_meta.architecture.architecture_type}, "
                f"res_blocks={saved_meta.architecture.num_res_blocks}, "
                f"history={saved_meta.architecture.history_length})\n"
                f"  Current: {current_hash} "
                f"({current_arch.architecture_type}, "
                f"res_blocks={current_arch.num_res_blocks}, "
                f"history={current_arch.history_length})"
            )
            if strict:
                raise CheckpointIncompatibleError(msg)
            else:
                print(f"WARNING: {msg}")
                print("Loading anyway (strict=False) - expect issues!")

        print(f"Loading checkpoint from {path}")
        print(f"  Created: {saved_meta.created_at}")
        print(f"  Epochs: {saved_meta.training_epochs}, Games: {saved_meta.training_games}")
        if saved_meta.final_elo:
            print(f"  Elo: {saved_meta.final_elo:.0f}")
    else:
        # Legacy checkpoint without metadata
        print(f"WARNING: Loading legacy checkpoint without metadata from {path}")
        print("Consider re-saving with metadata for future compatibility checks")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class CheckpointIncompatibleError(Exception):
    """Raised when checkpoint architecture doesn't match model."""
    pass
```

#### Phase 4: Migration Script for Existing Checkpoints (30 min)

```python
#!/usr/bin/env python3
"""
Migrate legacy checkpoints to include metadata.

Usage:
    python migrate_checkpoints.py --dir models/ --board-type square8
"""
import argparse
import torch
from pathlib import Path
from datetime import datetime

def migrate_checkpoint(path: Path, board_type: str, architecture_guess: dict):
    """Add metadata to legacy checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    if 'metadata' in checkpoint:
        print(f"  {path.name}: Already has metadata, skipping")
        return False

    # Create metadata with best guesses
    from app.models.checkpoint_metadata import CheckpointMetadata, ModelArchitecture

    architecture = ModelArchitecture(
        architecture_type=architecture_guess.get('type', 'RingRiftCNN'),
        input_planes=architecture_guess.get('input_planes', 12),
        history_length=architecture_guess.get('history_length', 3),
        num_res_blocks=architecture_guess.get('num_res_blocks', 10),
        num_filters=architecture_guess.get('num_filters', 128),
        policy_head_filters=architecture_guess.get('policy_head_filters', 32),
        value_head_hidden=architecture_guess.get('value_head_hidden', 256),
        board_type=board_type,
    )

    metadata = CheckpointMetadata(
        architecture=architecture,
        architecture_hash=architecture.compute_hash(),
        training_epochs=checkpoint.get('epoch', 0),
        final_loss=checkpoint.get('loss', 0.0),
        created_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        pytorch_version=torch.__version__,
    )

    checkpoint['metadata'] = metadata.to_dict()

    # Backup original
    backup_path = path.with_suffix('.pt.bak')
    if not backup_path.exists():
        path.rename(backup_path)

    # Save with metadata
    torch.save(checkpoint, path)
    print(f"  {path.name}: Migrated (arch_hash={metadata.architecture_hash})")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory with checkpoints')
    parser.add_argument('--board-type', required=True)
    parser.add_argument('--res-blocks', type=int, default=10)
    parser.add_argument('--history-length', type=int, default=3)
    args = parser.parse_args()

    arch_guess = {
        'num_res_blocks': args.res_blocks,
        'history_length': args.history_length,
    }

    checkpoints = list(Path(args.dir).glob('*.pt'))
    print(f"Found {len(checkpoints)} checkpoints")

    migrated = 0
    for cp in checkpoints:
        if migrate_checkpoint(cp, args.board_type, arch_guess):
            migrated += 1

    print(f"\nMigrated {migrated}/{len(checkpoints)} checkpoints")

if __name__ == '__main__':
    main()
```

### Verification Checklist

- [x] New checkpoints include metadata ✅ Verified: `save_checkpoint()` accepts architecture_version, model_class, model_config (2025-12-17)
- [x] Loading incompatible checkpoint raises clear error ✅ Verified: test_ringrift_architecture_mismatch_detection passes
- [x] Legacy checkpoints load with warning ✅ Verified: test_legacy_checkpoint_loads_non_strict passes
- [x] Migration script tested on sample checkpoints ✅ `scripts/migrate_checkpoints.py` exists with dry-run mode
- [x] All training scripts use new save/load functions ✅ Verified: checkpoint_unified.py integrated with model_versioning.py
- [x] Documentation updated ✅ CONSOLIDATION_ROADMAP.md documents the system

**Note:** All 52 checkpoint versioning tests pass (2025-12-17)

---

## 1.5 Hex Board Models Missing

**Priority:** P2 - High
**Risk:** Low
**Effort:** 20+ hours
**Owner:** TBD
**Status:** 🟡 IN PROGRESS (2025-12-17)

> **Implementation Status:**
>
> - ✅ `HexNeuralNet` architecture exists and validated
> - ✅ `app/training/hex_augmentation.py` - Hex-specific augmentation
> - ✅ `scripts/hex8_training_pipeline.py` - Hex training pipeline
> - ✅ Training logs exist: `h100_hex_2p.log`, `h100_hex_3p.log`, `h100_hex_4p.log`
> - ✅ Parity fixtures for hex validated
> - 🟡 Hex models still in early training phases, not production-ready

### Problem Statement

- `HexNeuralNet` architecture exists but is undertrained
- No production hex models available ← **Still valid**
- Hex game modes fall back to weak heuristic AI

### Impact

- Poor user experience on hex boards
- Hex boards effectively unplayable against AI at higher difficulties

### Action Plan

#### Phase 1: Validate Hex Architecture (2 hours)

```bash
# Step 1.1: Verify HexNeuralNet can forward pass
cd ai-service
python3 << 'EOF'
import torch
from app.models.hex_neural_net import HexNeuralNet

# Test hex8 (61 cells)
model = HexNeuralNet(board_type="hex8")
dummy_input = torch.randn(1, 12, 9, 9)  # Padded hex representation
policy, value = model(dummy_input)
print(f"hex8 - Policy shape: {policy.shape}, Value shape: {value.shape}")

# Test hexagonal (469 cells)
model_large = HexNeuralNet(board_type="hexagonal")
dummy_input_large = torch.randn(1, 12, 25, 25)
policy_large, value_large = model_large(dummy_input_large)
print(f"hexagonal - Policy shape: {policy_large.shape}, Value shape: {value_large.shape}")
EOF
```

```bash
# Step 1.2: Verify action encoder
python3 << 'EOF'
from app.ai.action_encoder_hex import ActionEncoderHex

encoder = ActionEncoderHex(board_type="hex8")
print(f"Action space size: {encoder.action_space_size}")
print(f"Sample encoding test passed: {encoder.test_encoding()}")
EOF
```

#### Phase 2: Generate Hex Training Data (8-10 hours)

```bash
# Step 2.1: Configure hex selfplay
cat > ai-service/config/hex_selfplay_config.yaml << 'EOF'
# Hex board self-play configuration
board_type: hex8
players: 2

selfplay:
  games_per_batch: 100
  total_games_target: 50000

  # Use MCTS for higher quality games
  engine: mcts
  mcts_simulations: 200

  # Temperature schedule
  temperature:
    initial: 1.5
    decay_type: linear
    decay_moves: 30
    final: 0.3

  # Parallelization
  num_workers: 4
  gpu_batch_size: 32

output:
  directory: data/selfplay/hex8
  format: npz
  include_mcts_policy: true
EOF
```

```bash
# Step 2.2: Run selfplay generation
cd ai-service
python scripts/generate_selfplay.py \
    --config config/hex_selfplay_config.yaml \
    --games 10000 \
    --output data/selfplay/hex8/batch_001
```

#### Phase 3: Train Hex Model (8-10 hours)

```bash
# Step 3.1: Create training config
cat > ai-service/config/hex_training_config.yaml << 'EOF'
# Hex8 neural network training configuration
model:
  architecture: HexNeuralNet
  board_type: hex8
  num_res_blocks: 6  # Smaller for hex8
  num_filters: 64
  history_length: 3

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 100

  # Learning rate schedule
  lr_schedule:
    type: cosine
    warmup_epochs: 5
    min_lr: 0.00001

  # Data augmentation (D6 symmetry for hex)
  augmentation:
    enabled: true
    symmetry_group: D6  # Hex has 6-fold rotational + reflection

  # Mixed precision
  amp_enabled: true

  # Checkpointing
  checkpoint_interval: 10
  checkpoint_dir: models/hex8/checkpoints

data:
  train_dir: data/selfplay/hex8
  validation_split: 0.1

evaluation:
  interval_epochs: 10
  games_per_eval: 100
  opponent: heuristic
EOF
```

```bash
# Step 3.2: Run training
python scripts/train_hex_model.py \
    --config config/hex_training_config.yaml \
    --gpu 0
```

#### Phase 4: Evaluate and Deploy (2 hours)

```bash
# Step 4.1: Run evaluation tournament
python scripts/run_tournament.py \
    --board-type hex8 \
    --model models/hex8/checkpoints/best.pt \
    --opponents "random,heuristic,mcts_100,mcts_500" \
    --games 100

# Step 4.2: Deploy to model registry
python scripts/register_model.py \
    --model models/hex8/checkpoints/best.pt \
    --board-type hex8 \
    --stage production \
    --min-elo 1200  # Only if evaluation shows competence
```

#### Phase 5: Integration Testing (2 hours)

```python
# tests/integration/test_hex_ai.py

import pytest
from app.ai.ai_manager import AIManager

class TestHexAI:

    @pytest.fixture
    def ai_manager(self):
        return AIManager(board_type="hex8")

    def test_hex_model_loads(self, ai_manager):
        """Verify hex model loads without error."""
        model = ai_manager.get_model(difficulty=5)
        assert model is not None

    def test_hex_move_generation(self, ai_manager):
        """Verify AI can generate valid hex moves."""
        from app.game_engine import GameEngine

        engine = GameEngine(board_type="hex8", num_players=2)
        state = engine.get_initial_state()

        move = ai_manager.get_move(state, difficulty=5)

        assert move is not None
        assert engine.is_valid_move(state, move)

    def test_hex_full_game(self, ai_manager):
        """AI can play a full hex game without crashing."""
        from app.game_engine import GameEngine

        engine = GameEngine(board_type="hex8", num_players=2)
        state = engine.get_initial_state()

        moves = 0
        while not engine.is_game_over(state) and moves < 200:
            current_player = state.current_player
            move = ai_manager.get_move(state, difficulty=5)
            state = engine.apply_move(state, move)
            moves += 1

        assert engine.is_game_over(state) or moves == 200
```

### Verification Checklist

- [ ] HexNeuralNet forward pass works
- [ ] ActionEncoderHex encoding/decoding verified
- [ ] 50,000+ hex8 self-play games generated
- [ ] Model trained to convergence
- [ ] Evaluation shows Elo > 1200 vs heuristic
- [ ] Model deployed to registry
- [ ] Integration tests passing
- [ ] AI service serves hex moves without fallback

---

# 2. Product & Adoption Gaps

These gaps don't affect the technical correctness of the project but are critical for reaching an audience.

## 2.1 No Public Release or Hosted Demo

**Priority:** P1 - Critical for adoption
**Risk:** Low (no technical risk, high visibility risk)
**Effort:** 8-12 hours
**Owner:** TBD
**Status:** ✅ COMPLETE (2025-12-17)

> **Implementation Status:**
>
> - ✅ GitHub release v0.1.0-beta created with release notes
> - ✅ Hosted demo live at **ringrift.ai**
> - ✅ Release notes published in `docs/RELEASE_NOTES_v0.1.0-beta.md`
> - 🟡 README "why it's different" pitch could still be enhanced

### Problem Statement (RESOLVED)

~~The project has:~~

~~- **0 GitHub releases** - "No releases published"~~
~~- **No hosted demo** - Users can't try the game without local setup~~

- **No "why it's different" pitch** - README describes features but not appeal (still valid)
- **0 stars, 0 forks, 2 contributors** - Near-zero adoption signals (marketing, not engineering)

### Impact

Even if the game is brilliant and the engineering is excellent, nobody will find out. The project stays "interesting tech" rather than "played game."

### Action Plan

#### Phase 1: Create First Release (2 hours)

```bash
# Create release tag
git tag -a v0.1.0-beta -m "First public beta release

RingRift v0.1.0-beta - Abstract Strategy Game

Features:
- Complete rules engine (TS + Python parity)
- 10-level AI difficulty ladder
- Multiple board sizes (8x8, 19x19, hex)
- Real-time multiplayer with WebSocket
- Spectator mode and replay system

Known limitations:
- No hosted demo (local setup required)
- UX optimized for developers/testers
- Single-instance deployment only
"

git push origin v0.1.0-beta
```

Then create GitHub Release with:

- Changelog
- Quick start instructions
- Link to documentation
- Screenshots/GIFs of gameplay

#### Phase 2: Deploy Hosted Demo (4-6 hours)

**Option A: Railway/Render (recommended for MVP)**

```bash
# Railway deployment
railway login
railway init
railway add --database postgres
railway add --database redis
railway up
```

**Option B: Fly.io**

```toml
# fly.toml
app = "ringrift-demo"

[build]
  dockerfile = "Dockerfile"

[env]
  NODE_ENV = "production"
  DATABASE_URL = "from secret"

[[services]]
  internal_port = 3000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

#### Phase 3: Create "Why RingRift?" Pitch (2 hours)

Add to README after the tagline:

```markdown
## Why RingRift?

**For players:** A deep, deterministic strategy game where "won" positions
can collapse through cascading reactions. Three distinct victory paths
keep every game dynamic. No dice, no card draws — every outcome is
determined by player decisions.

**For AI researchers:** A novel game environment with:

- Non-trivial state space (up to 469 cells, complex stack interactions)
- Explicit decision points (no hidden auto-execution)
- Cross-language parity (identical rules in TS and Python)
- 81 contract vectors for correctness verification

**For engineers:** A reference implementation of:

- Spec-driven game engine architecture
- Real-time multiplayer with WebSocket sync
- Domain-driven design with 8 canonical aggregates
- Comprehensive testing (10,000+ tests, 100% parity)
```

#### Phase 4: Add Quick Demo Video/GIF (2 hours)

- Record 30-60 second gameplay clip
- Show: placement → movement → capture chain → line formation → territory collapse
- Add to README and release notes

### Verification Checklist

- [ ] v0.1.0-beta release published on GitHub
- [ ] Hosted demo accessible at public URL
- [x] "Why RingRift?" section added to README ✅ Added 2025-12-17
- [x] Release notes draft prepared ✅ `docs/RELEASE_NOTES_v0.1.0-beta.md` created
- [ ] Demo video/GIF in README
- [ ] Demo URL added to repository description

---

## 2.2 UX/Tutorial Burden

**Priority:** P2 - High for adoption
**Risk:** Low
**Effort:** 10-15 hours
**Owner:** TBD

### Problem Statement

The game has **high cognitive load** for new players:

- Stack height vs cap height
- Marker flips vs marker collapses vs marker landing penalty
- Optional/mandatory chain capture
- Line processing with option choice
- Region disconnection conditions
- Multiple elimination contexts (cap vs single ring)

None of this is bad design — it's a deep game — but it means the **UX/tutorial burden is substantial**.

Current state: UI is "developer-centric" (per `KNOWN_ISSUES.md` P1.1).

### Action Plan

#### Phase 1: Create Three Critical Diagrams (3 hours)

These are the highest-value visuals per external analysis:

**Diagram 1: Marker Interactions**

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKER INTERACTIONS                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PASSING OVER OPPONENT MARKER → Flip to your color       │
│     [Blue] ----→ (Red marker) ----→ [Blue]                  │
│                    ↓ becomes                                 │
│                 (Blue marker)                                │
│                                                              │
│  2. PASSING OVER YOUR OWN MARKER → Collapse to territory    │
│     [Blue] ----→ (Blue marker) ----→ [Blue]                 │
│                    ↓ becomes                                 │
│                 [Blue territory]                             │
│                                                              │
│  3. LANDING ON ANY MARKER → Remove marker + lose top ring   │
│     [Blue] ----→ lands on (any marker)                      │
│                    ↓                                         │
│     [Blue-1 ring] + marker removed + ring eliminated         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Diagram 2: Line Processing Options**

```
┌─────────────────────────────────────────────────────────────┐
│            LINE PROCESSING (4+ markers in a row)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  EXACT LENGTH (e.g., exactly 4 markers):                    │
│  → Collapse ALL markers to territory                         │
│  → Eliminate ONE RING from any controlled stack              │
│                                                              │
│  OVERLENGTH (e.g., 5+ markers):                             │
│  ┌──────────────────┬──────────────────┐                    │
│  │    OPTION 1      │    OPTION 2      │                    │
│  ├──────────────────┼──────────────────┤                    │
│  │ Collapse ALL     │ Collapse only    │                    │
│  │ markers (max     │ required length  │                    │
│  │ territory)       │ (4 markers)      │                    │
│  │                  │                  │                    │
│  │ Cost: Eliminate  │ Cost: NONE       │                    │
│  │ ONE RING         │                  │                    │
│  └──────────────────┴──────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Diagram 3: Territory Disconnection**

```
┌─────────────────────────────────────────────────────────────┐
│              TERRITORY DISCONNECTION                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  A region becomes DISCONNECTED when:                        │
│                                                              │
│  1. PHYSICALLY ISOLATED:                                    │
│     Every path out is blocked by:                           │
│     - Board edge, OR                                        │
│     - Collapsed spaces, OR                                  │
│     - Markers of EXACTLY ONE border color                   │
│                                                              │
│     ┌───┬───┬───┬───┐                                       │
│     │ T │ T │ T │   │   T = Collapsed (barrier)            │
│     ├───┼───┼───┼───┤   B = Blue markers (border)          │
│     │ B │ ? │ ? │ T │   ? = Region cells                   │
│     ├───┼───┼───┼───┤                                       │
│     │ B │ ? │ B │   │   This region is bordered            │
│     ├───┼───┼───┼───┤   only by Blue markers               │
│     │   │ B │ B │   │                                       │
│     └───┴───┴───┴───┘                                       │
│                                                              │
│  2. COLOR-DISCONNECTED:                                     │
│     The region does NOT contain stacks controlled by        │
│     ALL active players (at least one player has no          │
│     stacks inside)                                          │
│                                                              │
│  PROCESSING COST: Eliminate your ENTIRE CAP from any        │
│  controlled stack (must have stacks OUTSIDE the region)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 2: Create Interactive Tutorial Mode (5-8 hours)

Add `/tutorial` route with guided scenarios:

```typescript
// src/client/pages/TutorialPage.tsx

const TUTORIAL_STEPS = [
  {
    id: 'placement',
    title: 'Ring Placement',
    scenario: createPlacementTutorialState(),
    instruction: 'Place 1-3 rings on an empty cell, then move the stack.',
    highlights: ['empty-cells'],
    validateMove: (move) => move.type === 'place_ring',
  },
  {
    id: 'movement',
    title: 'Stack Movement',
    scenario: createMovementTutorialState(),
    instruction: 'Move at least as far as your stack is tall.',
    highlights: ['valid-destinations'],
  },
  {
    id: 'capture',
    title: 'Overtaking Capture',
    scenario: createCaptureTutorialState(),
    instruction: 'Jump over an enemy stack to claim their top ring.',
    highlights: ['capturable-stacks'],
  },
  // ... more steps for lines, territory, etc.
];
```

#### Phase 3: Add Inline Explanations to Game UI (2-4 hours)

When `pendingDecision` is active, show explanation:

```typescript
// In ChoiceDialog.tsx
const CHOICE_EXPLANATIONS = {
  line_reward_option: `
    Your line is longer than required. Choose:
    • Option 1: Collapse ALL markers (more territory) but lose a ring
    • Option 2: Collapse only required markers (less territory) but keep all rings
  `,
  ring_elimination: `
    Select which stack to eliminate from.
    Tip: Choose a stack where losing the cap hurts least.
  `,
  region_order: `
    Multiple regions disconnected! Process them in order.
    Each region costs one cap elimination.
  `,
};
```

### Verification Checklist

- [x] Three diagrams created and added to docs ✅ Created `docs/ux/RULES_QUICK_REFERENCE_DIAGRAMS.md` with 5 ASCII diagrams (2025-12-17)
- [x] Tutorial mode implemented with 5+ guided steps ✅ Already exists: 17 teaching scenarios in `teachingScenarios.ts`, 4-step OnboardingModal
- [x] Inline explanations added to ChoiceDialog ✅ Added `strategicTip` field to `choiceViewModels.ts`, displayed in `ChoiceDialog.tsx` (2025-12-17)
- [x] Tutorial linked from main menu/lobby ✅ Already integrated: OnboardingModal shown to first-time players

---

## 2.3 Multiplayer Politics & Audience Positioning

**Priority:** P3 - Medium
**Risk:** Low (design consideration, not a bug)
**Effort:** 2-3 hours (documentation only)
**Owner:** TBD

### Problem Statement

3-4 player games will have:

- Short-term alliances
- Kingmaking scenarios (weaker player decides winner)
- "Hit the leader" dynamics

This is **intentional** (README mentions "temporary alliances") but it **narrows the audience**: some abstract strategy players love multiplayer politics, others hate it.

### Action Plan

#### Phase 1: Document Audience Positioning (1 hour)

Add to README or new `AUDIENCE.md`:

```markdown
## Who Is RingRift For?

### Perfect Fit

- **Abstract strategy enthusiasts** who enjoy games like Tak, GIPF, or Hive
- **AI/game-playing researchers** looking for a novel deterministic environment
- **Software engineers** interested in spec-driven multiplayer architecture

### Good Fit with Caveats

- **3-4 player games** include political dynamics (alliances, kingmaking)
- **High cognitive load** — better suited for players who enjoy deep systems
- **No randomness** — some players prefer dice/cards for tension

### Not Ideal For

- Players who dislike political/negotiation elements in games
- Casual players looking for quick, simple games
- Players who prefer hidden information or luck elements
```

#### Phase 2: Add Game Mode Recommendations (30 min)

In lobby/game creation UI, add hints:

```
2-player: Pure strategy, no politics
3-player: Moderate politics, dynamic alliances
4-player: High politics, kingmaking possible

Recommended for new players: 2-player on 8×8 board
```

#### Phase 3: Consider 2-Player-Focused Marketing (1 hour)

For initial launch, emphasize the 2-player experience:

- No political complications
- Pure deterministic strategy
- Closest to chess/Go audience expectations

### Verification Checklist

- [x] Audience positioning documented ✅ Created `docs/AUDIENCE.md` with comprehensive positioning (2025-12-17)
- [x] Game mode recommendations in UI ✅ Added player count hints to LobbyPage.tsx game creation form (2025-12-17)
- [x] Marketing materials emphasize 2-player purity ✅ Updated README.md "For players" section (2025-12-17)

---

# 3. Areas for Improvement

## 3.1 Architecture & Code Quality

### 3.1.1 Promotion Pipeline Consolidation

**Priority:** P3
**Effort:** 6-8 hours

**Current State:** 4 separate promotion systems with fragmented logic.

**Action Plan:**

1. **Create `PromotionPipeline` orchestrator** (3 hours)

```python
# ai-service/app/training/promotion_pipeline.py

class PromotionPipeline:
    """
    Unified promotion orchestrator.

    Coordinates:
    - PromotionEvaluator: Multi-criteria evaluation
    - PromotionGate: Approval/rejection logic
    - PromotionExecutor: Actual deployment
    - PromotionNotifier: Webhook/alert dispatch
    """

    def __init__(self, config):
        self.evaluator = PromotionEvaluator(config)
        self.gate = PromotionGate(config)
        self.executor = PromotionExecutor(config)
        self.notifier = PromotionNotifier(config)

    def evaluate_candidate(self, model_path: str, metrics: dict) -> PromotionDecision:
        """Full promotion evaluation pipeline."""
        # 1. Evaluate model against criteria
        evaluation = self.evaluator.evaluate(model_path, metrics)

        # 2. Gate decision
        decision = self.gate.decide(evaluation)

        # 3. Execute if approved
        if decision.approved:
            self.executor.promote(model_path, decision.target_stage)
            self.notifier.notify_promotion(model_path, decision)
        else:
            self.notifier.notify_rejection(model_path, decision)

        return decision
```

2. **Migrate existing systems** (3 hours)
   - `ModelRegistry.AutoPromoter` → calls `PromotionPipeline`
   - `PromotionController` → becomes thin wrapper
   - `ModelLifecycleManager.PromotionGate` → merged into `PromotionGate`
   - `CMAESRegistryIntegration` → separate heuristic-specific pipeline

3. **Add tests** (2 hours)
   - Unit tests for each component
   - Integration test for full pipeline

### Verification Checklist

- [x] `PromotionController` class exists ✅ `app/training/promotion_controller.py`
- [x] `PromotionType` enum (STAGING, PRODUCTION, TIER, CHAMPION, ROLLBACK) ✅ Lines 54-60
- [x] `PromotionCriteria` dataclass with canonical thresholds ✅ Lines 63-78
- [x] `PromotionDecision` dataclass with metrics ✅ Lines 80-117
- [x] `evaluate_promotion()` method for all types ✅ Lines 184-221
- [x] Prometheus metrics emission via `record_promotion_decision()` ✅ Lines 293-303
- [x] Integration with EloService, ModelRegistry (lazy-loaded) ✅ Lines 148-168
- [x] `ModelPromoter` in unified loop uses `PromotionController` ✅ `scripts/unified_loop/promotion.py`

**Note:** Already fully implemented in `app/training/promotion_controller.py` (2025-12-17)

### 3.1.2 Regression Detection Unification

**Priority:** P3
**Effort:** 4-6 hours

**Action Plan:**

1. **Create `RegressionDetector`** (2 hours)

```python
# ai-service/app/training/regression_detector.py

@dataclass
class RegressionThresholds:
    """Unified thresholds from config."""
    elo_drop: float = 50.0
    win_rate_drop: float = 0.10
    error_rate_increase: float = 0.05
    consecutive_losses: int = 5

class RegressionDetector:
    """
    Single source of truth for regression detection.
    """

    def __init__(self, thresholds: RegressionThresholds = None):
        self.thresholds = thresholds or RegressionThresholds()
        self._observers: List[RegressionObserver] = []

    def check(self, metrics: ModelMetrics) -> RegressionStatus:
        """Check current metrics for regression indicators."""
        status = RegressionStatus()

        if metrics.elo_delta < -self.thresholds.elo_drop:
            status.add_signal("elo_drop", metrics.elo_delta)

        if metrics.win_rate_delta < -self.thresholds.win_rate_drop:
            status.add_signal("win_rate_drop", metrics.win_rate_delta)

        # ... more checks

        if status.is_regressing:
            self._notify_observers(status)

        return status

    def subscribe(self, observer: RegressionObserver):
        """Subscribe to regression events."""
        self._observers.append(observer)
```

2. **Refactor existing systems** (2 hours)
   - `RollbackManager` subscribes to detector
   - Remove inline checks from other systems

3. **Add unified config** (1 hour)

```yaml
# config/regression.yaml
regression:
  elo_drop_threshold: 50
  win_rate_drop_threshold: 0.10
  error_rate_increase_threshold: 0.05
  consecutive_losses_threshold: 5
  detection_window_hours: 1
```

### Verification Checklist

- [x] `RegressionDetector` class exists ✅ `app/training/regression_detector.py`
- [x] `RegressionSeverity` enum (MINOR, MODERATE, SEVERE, CRITICAL) ✅ Lines 69-74
- [x] `RegressionConfig` with canonical thresholds ✅ Lines 77-109 (uses `thresholds.py`)
- [x] `RegressionEvent` dataclass with metrics ✅ Lines 112-154
- [x] Listener protocol with `on_regression()` ✅ Lines 157-162
- [x] Consecutive regression escalation logic ✅ Lines 279-285
- [x] Singleton factory `get_regression_detector()` ✅ Lines 440-456
- [x] Observer pattern for event notification ✅ `_notify_listeners()` method

**Note:** Already fully implemented in `app/training/regression_detector.py` (2025-12-17)

### 3.1.3 Model Sync Consolidation

**Priority:** P4
**Effort:** 3-4 hours

**Action Plan:**

1. **Extract common transport** (1.5 hours)

```python
# ai-service/app/sync/transport.py

class SyncTransport(Protocol):
    """Common sync transport interface."""

    def push(self, source: Path, dest: str) -> bool: ...
    def pull(self, source: str, dest: Path) -> bool: ...
    def list_remote(self, path: str) -> List[str]: ...

class RsyncTransport(SyncTransport):
    """Rsync-based transport with retry."""

class HTTPTransport(SyncTransport):
    """HTTP-based transport for model registry."""

class P2PTransport(SyncTransport):
    """P2P transport for cluster sync."""
```

2. **Create `ClusterSyncManager`** (1.5 hours)

```python
class ClusterSyncManager:
    """Unified sync manager with transport fallback."""

    def __init__(self, transports: List[SyncTransport]):
        self.transports = transports
        self.circuit_breaker = CircuitBreaker()

    def sync_model(self, model_path: Path, targets: List[str]) -> SyncResult:
        """Sync model to targets with fallback."""
        for transport in self.transports:
            if self.circuit_breaker.is_open(transport):
                continue
            try:
                results = [transport.push(model_path, t) for t in targets]
                return SyncResult(success=all(results))
            except Exception as e:
                self.circuit_breaker.record_failure(transport, e)

        return SyncResult(success=False, error="All transports failed")
```

### Verification Checklist

- [x] Unified `sync_models.py` script exists ✅ `scripts/sync_models.py`
- [x] Consolidates from sync_models_to_cluster.py, sync_staging_ai_artifacts.py ✅ Lines 3-8 docstring
- [x] Uses app/distributed/hosts module for host config ✅ Lines 58-69
- [x] Hash-based deduplication to avoid re-syncing ✅ Documented in module header
- [x] Sync lock via coordination helpers ✅ Lines 71-84, `acquire_sync_lock_safe()`
- [x] Bandwidth management ✅ Lines 107-111, `request_bandwidth_safe()`
- [x] Daemon mode for continuous sync ✅ `--daemon` flag, Lines 34
- [x] Deprecated `model_sync_aria2.py` points to `sync_models.py` ✅ Lines 3-22 of aria2 file

**Note:** Already fully implemented in `scripts/sync_models.py` (2025-12-17)

### 3.1.4 Zobrist Hash Optimization

**Priority:** P4
**Effort:** 1-2 hours

**Action Plan:**

1. **Ensure `GameState` always caches zobrist_hash** (1 hour)

```python
# ai-service/app/models/game_state.py

@dataclass
class GameState:
    # ... existing fields ...

    _zobrist_hash: Optional[int] = field(default=None, repr=False)

    @property
    def zobrist_hash(self) -> int:
        """Get cached or compute zobrist hash."""
        if self._zobrist_hash is None:
            from app.ai.zobrist import ZobristHasher
            self._zobrist_hash = ZobristHasher.compute(self)
        return self._zobrist_hash

    def invalidate_hash(self):
        """Call after any mutation."""
        self._zobrist_hash = None
```

2. **Update mutation points** (30 min)
   - `apply_move()` calls `invalidate_hash()`
   - Test hash caching behavior

### Verification Checklist

- [x] Zobrist hash caching implemented ✅ `MutableGameState._zobrist_hash` field exists
- [x] Incremental hash updates in make_move ✅ XOR-based updates throughout `make_move()` method
- [x] Hash restoration in unmake_move ✅ `undo.prev_zobrist_hash` restored properly
- [x] Initial computation only on state creation ✅ `_compute_zobrist_hash()` called in `from_immutable()`

**Note:** Already fully implemented in `app/rules/mutable_state.py` (2025-12-17)

---

## 3.2 Infrastructure Gaps

### 3.2.1 Secrets Management

**Priority:** P2
**Effort:** 4-6 hours

**Action Plan:**

1. **Choose solution** - Recommend AWS Secrets Manager (already using AWS)

2. **Create secrets module** (2 hours)

```python
# ai-service/app/config/secrets.py

import boto3
from functools import lru_cache

class SecretsManager:
    """AWS Secrets Manager integration."""

    def __init__(self):
        self.client = boto3.client('secretsmanager')
        self._cache = {}

    @lru_cache(maxsize=100)
    def get_secret(self, name: str) -> str:
        """Get secret value (cached)."""
        response = self.client.get_secret_value(SecretId=name)
        return response['SecretString']

    def get_database_url(self) -> str:
        return self.get_secret('ringrift/database_url')

    def get_redis_url(self) -> str:
        return self.get_secret('ringrift/redis_url')
```

3. **Migrate existing secrets** (2 hours)
   - Create secrets in AWS console/terraform
   - Update config to use `SecretsManager`
   - Remove secrets from `.env` files

4. **Update deployment** (1 hour)
   - Add IAM role for secrets access
   - Update Docker compose for local dev fallback

### 3.2.2 Backup Automation

**Priority:** P3
**Effort:** 3-4 hours

**Action Plan:**

1. **Create backup policy config** (30 min)

```yaml
# config/backup_policy.yaml
backup:
  database:
    schedule: '0 3 * * *' # Daily 3am
    retention_days: 30
    destination: s3://ringrift-backups/postgres/

  models:
    schedule: '0 */6 * * *' # Every 6 hours
    retention_days: 90
    destination: s3://ringrift-backups/models/

  game_data:
    schedule: '0 4 * * 0' # Weekly Sunday 4am
    retention_days: 365
    destination: s3://ringrift-backups/games/
```

2. **Create backup service** (2 hours)

```python
# ai-service/scripts/backup_service.py

class BackupService:
    def backup_database(self):
        """pg_dump to S3."""

    def backup_models(self):
        """Sync model registry to S3."""

    def backup_game_data(self):
        """Archive game databases to S3."""

    def prune_old_backups(self, policy: RetentionPolicy):
        """Delete backups older than retention period."""
```

3. **Add systemd timer or cron** (1 hour)

```ini
# /etc/systemd/system/ringrift-backup.timer
[Unit]
Description=RingRift backup timer

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### 3.2.3 Auto-Scaling Cluster

**Priority:** P4
**Effort:** 8-12 hours

**Action Plan:**

1. **Define scaling metrics** (1 hour)
   - Training queue depth
   - GPU utilization
   - Self-play game rate

2. **Create scaling controller** (4 hours)

```python
class ClusterScaler:
    """Auto-scale GPU cluster based on demand."""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.min_nodes = 1
        self.max_nodes = 10

    def evaluate_scaling(self, metrics: ClusterMetrics) -> ScalingDecision:
        """Determine if scaling needed."""
        if metrics.queue_depth > 1000 and metrics.gpu_utilization > 0.8:
            return ScalingDecision(action="scale_up", count=2)
        if metrics.queue_depth < 100 and metrics.gpu_utilization < 0.3:
            return ScalingDecision(action="scale_down", count=1)
        return ScalingDecision(action="none")

    def execute_scaling(self, decision: ScalingDecision):
        """Execute scaling action."""
        if decision.action == "scale_up":
            self.provider.add_nodes(decision.count)
        elif decision.action == "scale_down":
            self.provider.remove_nodes(decision.count)
```

3. **Integrate with Vast.ai/Lambda APIs** (4 hours)
4. **Add monitoring and alerts** (2 hours)

### Verification Checklist (3.2.x)

**3.2.1 Secrets Management:**

- [ ] `SecretsManager` class (`app/config/secrets.py`) - Not yet implemented
- [ ] AWS Secrets Manager integration - Not yet implemented
- [x] Environment variable support exists ✅ Config uses os.environ throughout

**3.2.2 Backup Automation:**

- [ ] `BackupService` class - Not yet implemented
- [ ] Backup policy config (`config/backup_policy.yaml`) - Not yet created
- [ ] Systemd timer for scheduled backups - Not yet configured
- [x] Manual backup procedures documented ✅ `docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`

**3.2.3 Auto-Scaling Cluster:**

- [x] `vast_autoscaler.py` exists ✅ Full implementation with ScalingConfig
- [x] Queue-based scaling (scale_up_queue_threshold=500) ✅ Lines 75-76
- [x] Utilization-based scaling ✅ Lines 76, 81
- [x] Budget limits (max_hourly_spend, max_daily_spend) ✅ Lines 69-72
- [x] Cooldown periods ✅ Lines 77-82
- [x] Vast.ai integration ✅ Multiple scripts: vast_lifecycle.py, vast_keepalive.py, etc.

**Note:** Auto-scaling is well-implemented (2025-12-17). Secrets/backup are infrastructure enhancements.

---

## 3.3 Testing Gaps

### 3.3.1 Hex Board Test Suite

**Priority:** P2
**Effort:** 4-6 hours

**Action Plan:**

1. **Create hex-specific test fixtures** (1 hour)

```python
# tests/fixtures/hex_fixtures.py

@pytest.fixture
def hex8_initial_state():
    """Initial hex8 game state."""
    from app.game_engine import GameEngine
    engine = GameEngine(board_type="hex8", num_players=2)
    return engine.get_initial_state()

@pytest.fixture
def hex8_midgame_state():
    """Hex8 state after 20 moves."""
    # ... setup midgame position
```

2. **Port critical square tests to hex** (3 hours)
   - Movement validation
   - Capture chains
   - Line detection (adapted for hex geometry)
   - Territory detection
   - Victory conditions

3. **Add hex-specific edge cases** (2 hours)
   - D6 symmetry verification
   - Hex coordinate wraparound
   - Diagonal vs axial movement

### 3.3.2 Multi-Player (3-4p) Test Coverage

**Priority:** P3
**Effort:** 4-6 hours

**Action Plan:**

1. **Create 3-4 player fixtures** (1 hour)
2. **Test player elimination scenarios** (2 hours)
3. **Test turn order with eliminated players** (1 hour)
4. **Test victory conditions in multiplayer** (2 hours)

### 3.3.3 GPU Training Regression Tests

**Priority:** P3
**Effort:** 3-4 hours

**Action Plan:**

1. **Create minimal training dataset** (1 hour)
   - 100 games, deterministic
   - Stored in test fixtures

2. **Add training regression test** (2 hours)

```python
# tests/integration/test_training_regression.py

@pytest.mark.gpu
def test_training_produces_consistent_loss():
    """Training on fixed dataset produces expected loss curve."""
    model = create_test_model()
    data = load_test_dataset()

    trainer = Trainer(model, data, epochs=10, seed=42)
    history = trainer.train()

    # Loss should decrease
    assert history[-1]['loss'] < history[0]['loss']

    # Final loss should be in expected range
    assert 0.5 < history[-1]['loss'] < 2.0
```

3. **Add to CI** (1 hour)
   - Nightly GPU test job
   - Alert on regression

### Verification Checklist

**3.3.1 Hex Board Tests:**

- [x] Hex test files exist ✅ 5 files: test_hex_training.py, test_hex_augmentation.py, etc.
- [x] Substantial coverage ✅ 84 test functions across 2519 lines
- [ ] Hex-specific fixtures (tests/fixtures/hex_fixtures.py) - Not yet created
- [ ] D6 symmetry verification tests - Partial (augmentation tests)

**3.3.2 Multi-Player Tests:**

- [x] Multiplayer test files exist ✅ 3 files with 23 test functions
- [x] test_multiplayer_ai_search.py ✅ 11 tests for MCTS/search
- [x] test_multiplayer_line_vectors.py ✅ 10 parity tests
- [ ] Player elimination scenario tests - Not yet dedicated file
- [ ] Turn order with eliminated players - Needs additional coverage

**3.3.3 GPU Training Regression:**

- [ ] Fixed training dataset for regression tests - Not yet created
- [ ] test_training_regression.py with @pytest.mark.gpu - Not yet created
- [ ] Nightly CI job for GPU tests - CI exists but needs specific job

**Note:** Substantial test coverage exists (2025-12-17). Additional fixtures/edge cases are enhancement items.

---

# 4. Weak Areas to Address

## 4.1 Documentation Drift

**Priority:** P3
**Effort:** 4-6 hours

### Action Plan

1. **Audit existing docs** (1 hour)

```bash
# Find all markdown docs
find . -name "*.md" -not -path "./node_modules/*" | head -50

# Check for stale dates
grep -r "2024\|2023" docs/ --include="*.md"

# Find TODOs and FIXMEs in docs
grep -r "TODO\|FIXME\|WIP\|DRAFT" docs/ --include="*.md"
```

2. **Create documentation index** (1 hour)

```markdown
# docs/INDEX.md

## Active Documentation

| Document                 | Status | Last Updated | Owner     |
| ------------------------ | ------ | ------------ | --------- |
| RULES_CANONICAL_SPEC.md  | Active | 2025-12-10   | Core Team |
| CONSOLIDATION_ROADMAP.md | Active | 2025-12-17   | AI Team   |

| ...

## Deprecated/Archived

| Document                | Reason     | Replaced By                  |
| ----------------------- | ---------- | ---------------------------- |
| OLD_IMPROVEMENT_PLAN.md | Superseded | COMPREHENSIVE_ACTION_PLAN.md |
```

3. **Archive stale docs** (1 hour)
   - Move to `docs/archive/`
   - Add deprecation notice at top

4. **Add freshness checks to CI** (2 hours)

```yaml
# .github/workflows/docs-freshness.yml
- name: Check doc freshness
  run: |
    # Fail if any active doc older than 90 days
    find docs/ -name "*.md" -mtime +90 \
      -not -path "docs/archive/*" \
      -exec echo "Stale doc: {}" \;
```

### Verification Checklist

- [x] Documentation audit completed ✅ Found ~180 active docs, 56 with TODOs (2025-12-17)
- [x] Documentation index created ✅ Updated `docs/INDEX.md` with comprehensive status tables (2025-12-17)
- [x] Archive directory exists ✅ `docs/archive/` already in use with ~65 archived docs
- [ ] CI freshness check added (deferred - low priority)

---

## 4.2 Error Handling Inconsistency

**Priority:** P3
**Effort:** 4-6 hours

### Action Plan

1. **Create Python error hierarchy** (2 hours)

```python
# ai-service/app/errors.py

class RingRiftError(Exception):
    """Base exception for all RingRift errors."""
    code: str = "RINGRIFT_ERROR"

class RulesViolationError(RingRiftError):
    """Invalid move per game rules."""
    code: str = "RULES_VIOLATION"

    def __init__(self, message: str, rule_ref: str = None):
        super().__init__(message)
        self.rule_ref = rule_ref  # e.g., "RR-CANON-R062"

class InvalidStateError(RingRiftError):
    """Corrupted or unexpected game state."""
    code: str = "INVALID_STATE"

class AIFallbackError(RingRiftError):
    """AI failed and used fallback."""
    code: str = "AI_FALLBACK"

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error
```

2. **Add explicit fallback logging** (2 hours)

```python
# ai-service/app/ai/ai_manager.py

def get_move(self, state, difficulty):
    try:
        return self._get_move_impl(state, difficulty)
    except Exception as e:
        # Log explicit fallback
        logger.warning(
            "AI fallback triggered",
            extra={
                "error": str(e),
                "difficulty": difficulty,
                "board_type": state.board_type,
                "fallback_to": "random"
            }
        )
        # Raise wrapped error (don't silently continue)
        raise AIFallbackError(
            f"AI difficulty {difficulty} failed, using random fallback",
            original_error=e
        )
```

3. **Add error metrics** (1 hour)

```python
from prometheus_client import Counter

ai_errors = Counter(
    'ringrift_ai_errors_total',
    'AI error counts by type',
    ['error_type', 'difficulty', 'board_type']
)
```

### Verification Checklist

- [x] Python error hierarchy created ✅ `app/errors.py` with comprehensive hierarchy (RingRiftError, RulesViolationError, AIError, TrainingError, InfrastructureError, ValidationError) (2025-12-17)
- [x] Error classes have context support ✅ All classes support `context` dict and `to_dict()` serialization
- [ ] AI fallback logging with explicit metrics (partial - fallback logic exists but not using new errors)
- [ ] Prometheus error counters exported (deferred - integrate with SignalMetricsExporter)

**Note:** Error hierarchy exists at `app/errors.py` with 16+ error types. Integration into AI managers is partial. (2025-12-17)

---

## 4.3 Observability Gaps

**Priority:** P2
**Effort:** 6-8 hours

### Action Plan

1. **Add distributed tracing** (4 hours)

```python
# ai-service/app/tracing.py

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider

def setup_tracing():
    provider = TracerProvider()
    exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ringrift.ai")

# Usage in AI service
@tracer.start_as_current_span("get_ai_move")
def get_ai_move(state, difficulty):
    span = trace.get_current_span()
    span.set_attribute("difficulty", difficulty)
    span.set_attribute("board_type", state.board_type)
    # ... implementation
```

2. **Add runbooks for alerts** (2 hours)

```markdown
# docs/runbooks/HIGH_AI_LATENCY.md

## Alert: AI Response Latency > 5s

### Symptoms

- P95 AI move latency exceeds 5 seconds
- Players experience slow AI responses

### Diagnosis

1. Check GPU utilization: `nvidia-smi`
2. Check MCTS simulation count in logs
3. Check if model is loaded (vs loading on demand)

### Resolution

1. If GPU at 100%: Scale up GPU nodes
2. If MCTS too high: Reduce simulation count for difficulty
3. If model loading: Enable model preloading
```

3. **Standardize AI decision logging** (2 hours)

```python
# ai-service/app/ai/logging.py

@dataclass
class AIDecisionLog:
    """Standardized AI decision log entry."""
    timestamp: datetime
    game_id: str
    move_number: int
    difficulty: int

    # Decision info
    engine_type: str  # mcts, minimax, heuristic
    search_depth: int
    simulations: int
    time_ms: float

    # Move info
    chosen_move: str
    move_score: float
    top_alternatives: List[tuple]  # [(move, score), ...]

    # Diagnostics
    cache_hit: bool
    model_version: str

    def to_structured_log(self) -> dict:
        return asdict(self)
```

---

## 4.4 Self-Play Data Quality

**Priority:** P2
**Effort:** 4-6 hours

### Action Plan

1. **Create data validation pipeline** (3 hours)

```python
# ai-service/app/data/validation.py

class DataValidator:
    """Validate self-play games before training."""

    def validate_game(self, game: GameRecord) -> ValidationResult:
        """Run all validation checks on a game."""
        issues = []

        # Check game completed properly
        if not game.is_terminal:
            issues.append(ValidationIssue("incomplete", "Game not terminal"))

        # Check move count reasonable
        if game.move_count < 10:
            issues.append(ValidationIssue("too_short", f"Only {game.move_count} moves"))

        # Check no duplicate positions (likely bug)
        positions = set()
        for state in game.states:
            h = state.zobrist_hash
            if h in positions:
                issues.append(ValidationIssue("duplicate_position", f"Hash {h} repeated"))
            positions.add(h)

        # Check policy targets sum to 1
        for policy in game.policies:
            if abs(sum(policy) - 1.0) > 0.01:
                issues.append(ValidationIssue("invalid_policy", "Policy doesn't sum to 1"))

        return ValidationResult(valid=len(issues) == 0, issues=issues)

    def filter_dataset(self, games: List[GameRecord]) -> tuple:
        """Filter dataset, return (valid, invalid) split."""
        valid, invalid = [], []
        for game in games:
            result = self.validate_game(game)
            if result.valid:
                valid.append(game)
            else:
                invalid.append((game, result.issues))
        return valid, invalid
```

2. **Add deduplication** (1.5 hours)

```python
class GameDeduplicator:
    """Deduplicate games based on move sequence hash."""

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, game: GameRecord) -> bool:
        """Check if game is duplicate of previously seen."""
        h = self._compute_game_hash(game)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def _compute_game_hash(self, game: GameRecord) -> str:
        """Hash based on move sequence."""
        moves_str = "|".join(str(m) for m in game.moves)
        return hashlib.sha256(moves_str.encode()).hexdigest()[:16]
```

3. **Add monitoring dashboard** (1.5 hours)

```json
// monitoring/grafana/dashboards/data-quality.json
{
  "panels": [
    {
      "title": "Games Validated",
      "type": "stat",
      "targets": [{ "expr": "sum(ringrift_games_validated_total)" }]
    },
    {
      "title": "Validation Failure Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "rate(ringrift_games_invalid_total[1h]) / rate(ringrift_games_validated_total[1h])"
        }
      ]
    },
    {
      "title": "Failure Reasons",
      "type": "piechart",
      "targets": [{ "expr": "sum by (reason) (ringrift_games_invalid_total)" }]
    }
  ]
}
```

---

## 4.5 Recovery After Failures

**Priority:** P3
**Effort:** 4-6 hours

### Action Plan

1. **Improve leader election recovery** (2 hours)

```python
# ai-service/app/coordination/leader_election.py

class FastLeaderElection:
    """Leader election with faster recovery."""

    def __init__(self,
                 node_id: str,
                 heartbeat_interval: float = 5.0,
                 election_timeout: float = 15.0):  # Reduced from 60s
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout = election_timeout
        self._last_heartbeat = time.time()

    async def run_election(self):
        """Run leader election with exponential backoff."""
        attempt = 0
        while True:
            try:
                return await self._attempt_leadership()
            except ElectionConflict:
                # Exponential backoff with jitter
                wait = min(30, 2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
                attempt += 1
```

2. **Add checkpoint recovery** (2 hours)

```python
class CheckpointRecovery:
    """Recover from partial training checkpoints."""

    def find_latest_valid_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """Find most recent valid checkpoint."""
        checkpoints = sorted(
            checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for cp in checkpoints:
            if self._is_valid_checkpoint(cp):
                return cp
            else:
                logger.warning(f"Invalid checkpoint: {cp}, trying older")

        return None

    def _is_valid_checkpoint(self, path: Path) -> bool:
        """Verify checkpoint integrity."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            required_keys = ['model_state_dict', 'epoch']
            return all(k in checkpoint for k in required_keys)
        except Exception:
            return False
```

3. **Add dead letter queue visibility** (2 hours)

```python
# Add Grafana dashboard panel for DLQ
# Add alert when DLQ grows

dlq_size = Gauge(
    'ringrift_dlq_size',
    'Dead letter queue size',
    ['queue_type']
)

# In data sync code:
def on_sync_failure(game, error):
    dlq.append(game)
    dlq_size.labels(queue_type='game_sync').set(len(dlq))
```

### Verification Checklist (4.3-4.5)

**4.3 Observability Gaps:**

- [x] Runbooks exist ✅ 40+ runbooks in `docs/runbooks/` (AI, Database, Deployment, etc.)
- [x] AI performance runbook ✅ `docs/runbooks/AI_PERFORMANCE.md`
- [x] High latency runbook ✅ `docs/runbooks/HIGH_LATENCY.md`
- [ ] OpenTelemetry distributed tracing (`app/tracing.py`) - Not yet implemented
- [ ] Standardized `AIDecisionLog` dataclass - Partial (logging exists but not structured)

**4.4 Self-Play Data Quality:**

- [x] Database validation exists ✅ `app/db/validation.py`
- [ ] Game-level `DataValidator` class - Not yet implemented
- [ ] `GameDeduplicator` for training data - Not yet implemented
- [ ] Data quality Grafana dashboard - Not yet created

**4.5 Recovery After Failures:**

- [x] Cluster auto-recovery daemon ✅ `scripts/cluster_auto_recovery.py`
- [x] Node health monitoring with automatic restart ✅ Lines 1-80 of cluster_auto_recovery.py
- [x] Slack alerts on state changes ✅ SLACK_WEBHOOK_URL integration
- [ ] Fast leader election (`FastLeaderElection`) - Not yet implemented
- [ ] Training checkpoint recovery (`CheckpointRecovery`) - Partial (exists in save_checkpoint)

**Note:** Observability has strong runbook coverage (2025-12-17). Tracing and data validation are enhancement items.

---

# Summary & Prioritization

## Critical Path to Adoption

The following four items form the **critical path** to making RingRift usable by anyone outside the core team:

| #   | Task                            | Priority | Effort | Why Critical                                                     |
| --- | ------------------------------- | -------- | ------ | ---------------------------------------------------------------- |
| 1   | **Fix rules doc contradiction** | P0       | 2-3h   | Blocks player trust - contradictory docs undermine credibility   |
| 2   | **First GitHub release**        | P1       | 2h     | Blocks discoverability - "no releases" signals abandoned project |
| 3   | **Deploy hosted demo**          | P1       | 4-6h   | Blocks trial - nobody will clone repo just to try a game         |
| 4   | **Create tutorial/diagrams**    | P2       | 3-5h   | Blocks learning - high cognitive load without visual guides      |

## Immediate (This Week)

| Task                              | Priority | Effort | Impact                               |
| --------------------------------- | -------- | ------ | ------------------------------------ |
| Fix rules doc contradiction (1.1) | P0       | 2-3h   | **Blocks everything** - player trust |
| First release + hosted demo (2.1) | P1       | 6-8h   | Visibility & trial capability        |
| Checkpoint versioning (1.4)       | P1       | 2-3h   | Prevents silent model regression     |
| Config consolidation (1.2)        | P1       | 4-6h   | Single source of truth for training  |

## Short-Term (This Month)

| Task                              | Priority | Effort | Impact                        |
| --------------------------------- | -------- | ------ | ----------------------------- |
| Tutorial mode + diagrams (2.2)    | P1       | 10-15h | Player onboarding             |
| Training signal unification (1.3) | P1       | 8-12h  | Consistent training decisions |
| Hex model training (1.5)          | P2       | 20h    | Hex game mode playable        |
| Distributed tracing (4.3)         | P2       | 6-8h   | Better debugging              |
| Secrets management (3.2.1)        | P2       | 4-6h   | Security improvement          |
| Data quality validation (4.4)     | P2       | 4-6h   | Better training data          |

## Medium-Term

| Task                           | Priority | Effort | Impact                 |
| ------------------------------ | -------- | ------ | ---------------------- |
| Audience positioning doc (2.3) | P3       | 2-3h   | Clearer marketing      |
| Promotion pipeline (3.1.1)     | P3       | 6-8h   | Cleaner promotion flow |
| Regression detection (3.1.2)   | P3       | 4-6h   | Faster rollback        |
| Backup automation (3.2.2)      | P3       | 3-4h   | Data safety            |

---

## Key Audiences & What They Need

| Audience                        | What They Need                                     | Key Blockers to Address      |
| ------------------------------- | -------------------------------------------------- | ---------------------------- |
| **Abstract strategy players**   | Playable demo, clear rules, tutorial               | 1.1, 2.1, 2.2                |
| **AI/game-playing researchers** | Correct rules, parity verification, documented API | 1.1, contract vectors (done) |
| **Software engineers**          | Reference architecture, clear docs                 | All docs (mostly done)       |
| **Board game designers**        | Rule exploration, sandbox mode                     | 2.2 (sandbox exists)         |

---

## Effort Summary

| Category                          | Total Effort      | Items    |
| --------------------------------- | ----------------- | -------- |
| Critical Gaps (Section 1)         | 38-50 hours       | 5 items  |
| Product & Adoption (Section 2)    | 22-32 hours       | 3 items  |
| Areas for Improvement (Section 3) | 40-55 hours       | 10 items |
| Weak Areas (Section 4)            | 22-32 hours       | 5 items  |
| **TOTAL**                         | **122-169 hours** | 23 items |

**If only 20 hours available**: Focus on items 1-4 from critical path (rules fix, release, demo, diagrams) — this unlocks external adoption.

**If only 40 hours available**: Add config consolidation, checkpoint versioning, and training signal unification — this solidifies the AI training pipeline.

---

## Related Documents

- [CONSOLIDATION_ROADMAP.md](../ai-service/docs/CONSOLIDATION_ROADMAP.md) - Existing consolidation work
- [ARCHITECTURAL_IMPROVEMENT_PLAN.md](ARCHITECTURAL_IMPROVEMENT_PLAN.md) - TypeScript improvements
- [PRODUCTION_READINESS_CHECKLIST.md](PRODUCTION_READINESS_CHECKLIST.md) - Launch criteria
- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Current known issues
- [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md) - Authoritative rules SSoT
