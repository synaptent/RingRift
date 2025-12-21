# Codebase Consolidation and Cleanup Plan

**Status:** Active (2025-12)
**Author:** Automated audit draft (unverified)
**Last Updated:** 2025-12-20

**Validation Note:** This plan is a draft based on a heuristic scan. All counts and candidate lists below are estimates. Validate imports, runtime usage, and dependencies before any removal or consolidation.

## Executive Summary

This document outlines a comprehensive plan to consolidate, clean up, and improve the RingRift codebase. The analysis identified:

- **~90 candidate orphaned modules** with zero imports (potential dead code, needs validation)
- **~244 underutilized modules** with only 1-2 imports (needs validation)
- **~18 deprecated scripts** still in the codebase (verify per file)
- **~8,500 lines** of commented-out code (estimate)
- **6+ parallel implementations** of tournament runners (estimate)
- **4+ parallel implementations** each for training pipelines, model promotion, and data sync (estimate)
- **102 scripts** analyzed for consolidation opportunities (estimate)

**Estimated Impact:** Remove ~50 files, consolidate ~14 files, resulting in cleaner, more maintainable codebase.

---

## Table of Contents

1. [Goals and Non-Goals](#goals-and-non-goals)
2. [Validation Checklist](#validation-checklist)
3. [Current State Analysis](#current-state-analysis)
4. [Consolidation Targets](#consolidation-targets)
5. [Execution Phases](#execution-phases)
6. [Detailed File Inventory](#detailed-file-inventory)
7. [Risk Assessment](#risk-assessment)
8. [Progress Tracking](#progress-tracking)
9. [Related Documents](#related-documents)

---

## Goals and Non-Goals

### Goals

1. **Remove dead code** - Eliminate orphaned modules, deprecated scripts, and commented-out code
2. **Consolidate parallel implementations** - Merge multiple scripts doing similar things into unified entry points
3. **Improve maintainability** - Reduce cognitive load by having clear, single sources of truth
4. **Preserve functionality** - Ensure all features, options, and capabilities are retained
5. **Document decisions** - Track what was removed/consolidated and why

### Non-Goals

- Rewriting AI algorithms or search logic
- Changing canonical rules or move semantics
- Replacing the TS engine as the SSoT
- Major architectural refactoring (beyond file organization)

---

## Validation Checklist

Run these before any consolidation or removal to confirm actual usage.

1. **Import graph scan (Python):** Generate a module import graph and confirm candidates are truly unused.
2. **Runtime usage:** Search for dynamic imports, string-based references, CLI entry points, and docs references.
3. **Tests:** Identify which tests cover candidate modules or scripts.
4. **Execution logs:** Cross-check against production or training logs for script usage.

---

## Current State Analysis

### Codebase Statistics

| Component                                  | Files | Lines  | Notes                            |
| ------------------------------------------ | ----- | ------ | -------------------------------- |
| AI Service (`ai-service/`)                 | 471+  | ~150K+ | Python AI, training, distributed |
| Scripts (`ai-service/scripts/`)            | 398   | ~80K+  | CLI tools, daemons, utilities    |
| Shared Engine (`src/shared/engine/`)       | 69    | ~15K   | TypeScript rules engine          |
| Tests (`ai-service/tests/`)                | 396   | ~40K   | Python test suite                |
| Mutant Tests (`ai-service/mutants/tests/`) | 189   | ~20K   | Duplicate/variant tests          |

### Key Problem Areas

#### 1. Parallel Implementations

| Category           | Count | Primary Entry Point           |
| ------------------ | ----- | ----------------------------- |
| Tournament Runners | 6+    | `run_tournament.py`           |
| Model Promotion    | 4+    | `unified_promotion_daemon.py` |
| Training Pipelines | 4+    | `hex8_training_pipeline.py`   |
| Data Sync          | 6+    | `unified_data_sync.py`        |
| Cluster Monitoring | 7+    | `robust_cluster_monitor.py`   |
| EBMO Training      | 7+    | None (experimental)           |

#### 2. Orphaned Modules (Candidate List - Verify Before Removal)

**Candidate modules in `app/` with zero incoming imports (counts approximate):**

```
# Experimental AI
app.ai.batch_eval
app.ai.gmo_policy_provider
app.ai.gmo_v2
app.ai.lightweight_eval
app.ai.lightweight_state
app.ai.numba_eval
app.ai.parallel_eval
app.ai.swap_evaluation
app.ai.move_ordering
app.ai.cache_invalidation

# Training Variants
app.training.adversarial_positions
app.training.auto_data_discovery
app.training.distillation
app.training.ebmo_trainer
app.training.evaluate_gmo_baselines
app.training.generate_territory_dataset
app.training.nnue_quality_metrics
app.training.opening_book
app.training.optimized_pipeline
app.training.pbt
app.training.test_exploration_diversity
app.training.train_gmo_diverse
app.training.train_gmo_online
app.training.train_loop
app.training.training_health
app.training.uncertainty_calibration

# P2P Infrastructure
app.p2p.config
app.p2p.models
app.p2p.notifications
app.p2p.training

# Core/Distributed
app.core.async_context
app.core.initializable
app.core.locking
app.core.marshalling
app.core.registry_base
app.core.task_spawner
app.distributed.client
app.distributed.cluster_coordinator
app.distributed.subscription_registry
app.distributed.sync_utils
app.coordination.lock_manager
app.coordination.sync_base
app.integration.unified_loop_extensions

# Other
app.models.multitask_heads
app.models.transformer_model
app.analysis.game_balance
app.evaluation.human_eval
app.config.schema
app.rules.mutators.recovery
app.rules.mutators.turn
app.rules.validators.recovery
app.game_engine.phase_requirements
app.tournament.composite_tournament
app.utils.async_utils
app.utils.load_throttle
```

#### 3. Large Files Needing Refactoring

| File                     | Lines  | Commented  | Issue                      |
| ------------------------ | ------ | ---------- | -------------------------- |
| `p2p_orchestrator.py`    | 27,829 | 2,191 (8%) | Monolithic orchestrator    |
| `unified_ai_loop.py`     | 8,236  | 782 (9%)   | Training loop + everything |
| `_neural_net_legacy.py`  | 6,959  | 764 (11%)  | Partial migration          |
| `train_nnue.py`          | 5,425  | 371 (7%)   | Training variants merged   |
| `_game_engine_legacy.py` | 4,502  | 724 (16%)  | Legacy re-export layer     |
| `run_self_play_soak.py`  | 3,830  | -          | Soak testing               |
| `train.py`               | 3,526  | 346 (10%)  | Main training loop         |

#### 4. Deprecated But Present (Verify Per File)

**Scripts marked `# DEPRECATED` but not removed (verify per file):**

- `baseline_gauntlet.py`
- `composite_elo_dashboard.py`
- `disk_monitor.py`
- `elo_dashboard.py`
- `export_replay_dataset.py`
- `hex8_training_pipeline.py`
- `launch_distributed_elo_tournament.py`
- `model_sync_aria2.py`
- `p2p_model_sync.py`
- `pipeline_dashboard.py`
- `run_model_elo_tournament.py`
- `run_self_play_soak.py`
- `run_tournament.py`
- `run_vast_gauntlet.py`
- `training_dashboard.py`
- `unified_ai_loop.py`
- `unified_data_sync.py`

**Note:** Some "deprecated" markers indicate partial deprecation or transition state, not full removal readiness.

---

## Consolidation Targets

### Target 1: Tournament Runners

**Current State:** 11 scripts with overlapping functionality

| Script                                                       | Lines | Imports | Status                             |
| ------------------------------------------------------------ | ----- | ------- | ---------------------------------- |
| `run_tournament.py`                                          | 800+  | 9       | **KEEP - Unified hub**             |
| `run_model_elo_tournament.py`                                | 1200+ | 14      | **KEEP - Core evaluation**         |
| `run_distributed_tournament.py`                              | 900+  | 5+      | **KEEP - Distributed ladder**      |
| `run_eval_tournaments.py`                                    | 400+  | 2       | **KEEP**                           |
| `run_diverse_tournaments.py`                                 | 500+  | 3       | **KEEP**                           |
| `run_ssh_distributed_tournament.py`                          | 300+  | 2       | **KEEP**                           |
| `run_p2p_elo_tournament.py`                                  | 400+  | 2       | **KEEP**                           |
| `auto_elo_tournament.py`                                     | 350+  | 1       | CONSOLIDATE into run_tournament.py |
| `launch_distributed_elo_tournament.py`                       | 200+  | 1       | CONSOLIDATE                        |
| `shadow_tournament_service.py`                               | 300+  | 1       | EVALUATE                           |
| `archive/deprecated/run_ai_tournament.py`                    | 570   | 0       | **ARCHIVED**                       |
| `archive/deprecated/run_axis_aligned_tournament.py`          | 671   | 0       | **ARCHIVED**                       |
| `archive/deprecated/run_crossboard_difficulty_tournament.py` | 302   | 0       | **ARCHIVED**                       |

**Consolidation Strategy:** See `ai-service/docs/planning/TOURNAMENT_ELO_CONSOLIDATION_PLAN.md`

---

### Target 2: Model Promotion

**Current State:** 8 scripts

| Script                            | Lines | Imports | Status                               |
| --------------------------------- | ----- | ------- | ------------------------------------ |
| `unified_promotion_daemon.py`     | 607   | 4       | **KEEP - Primary daemon**            |
| `model_promotion_manager.py`      | 1910  | 6       | **KEEP - Core implementation**       |
| `auto_promote.py`                 | 281   | 9 refs  | CONSOLIDATE (9 scripts reference it) |
| `auto_model_promotion.py`         | 294   | 1       | CONSOLIDATE                          |
| `apply_tier_promotion_plan.py`    | 449   | 1       | CONSOLIDATE                          |
| `validate_and_promote_weights.py` | 504   | 1       | KEEP as validation module            |
| `elo_promotion_gate.py`           | 265   | 0       | MERGE into daemon                    |
| `run_parity_promotion_gate.py`    | 344   | 0       | MERGE into daemon                    |

**Consolidation Strategy:**

1. Merge gate scripts into `unified_promotion_daemon.py` as subcommands
2. Keep `auto_promote.py` as deprecated alias (9 references)
3. Consolidate `auto_model_promotion.py` and `apply_tier_promotion_plan.py` features

---

### Target 3: Cluster Monitoring

**Current State:** 10 scripts

| Script                       | Lines | Imports | Status                         |
| ---------------------------- | ----- | ------- | ------------------------------ |
| `unified_cluster_monitor.py` | 37    | 3       | **KEEP - Public API wrapper**  |
| `robust_cluster_monitor.py`  | 373   | 0       | **KEEP - Best implementation** |
| `cluster_health_monitor.py`  | 321   | 0       | REMOVE (superseded)            |
| `cluster_monitor_daemon.py`  | 413   | 0       | EVALUATE vs robust             |
| `simple_cluster_monitor.py`  | 215   | 0       | REMOVE (superseded)            |
| `data_quality_monitor.py`    | 739   | 0       | KEEP (specialized)             |
| `disk_monitor.py`            | 709   | 0       | KEEP (specialized)             |
| `elo_monitor.py`             | 300+  | 0       | KEEP (specialized)             |
| `training_monitor.py`        | 400+  | 0       | KEEP (specialized)             |
| `monitor_improvement.py`     | 200+  | 0       | EVALUATE                       |

**Consolidation Strategy:**

1. `robust_cluster_monitor.py` becomes the primary cluster health implementation
2. Remove `simple_cluster_monitor.py` and `cluster_health_monitor.py`
3. Keep specialized monitors (`data_quality_monitor.py`, `disk_monitor.py`, etc.)

---

### Target 4: Training Pipelines

**Current State:** 31+ scripts

**Primary Pipelines:**
| Script | Status |
|--------|--------|
| `hex8_training_pipeline.py` | **KEEP - Primary production** |
| `run_tier_training_pipeline.py` | **KEEP - Tier-based** |
| `auto_training_pipeline.py` | CONSOLIDATE |
| `continuous_training_loop.py` | REMOVE (superseded by unified_ai_loop) |
| `multi_config_training_loop.py` | REMOVE (experimental) |
| `population_based_training.py` | REMOVE (research) |

**Consolidation Strategy:** See `ai-service/docs/planning/NN_STRENGTHENING_PLAN.md` Section F (Closed-loop automation)

---

### Target 5: EBMO Training Variants

**Current State:** 16 scripts archived (no cross-imports)

```
train_ebmo.py
train_ebmo_contrastive.py
train_ebmo_curriculum.py
train_ebmo_expert.py
train_ebmo_improved.py
train_ebmo_outcome.py
train_ebmo_quality.py
benchmark_ebmo_ladder.py
eval_ebmo_56ch.py
eval_ebmo_quick.py
generate_ebmo_expert_data.py
generate_ebmo_selfplay.py
generate_ebmo_vs_heuristic.py
diagnose_ebmo.py
test_ebmo_online.py
tune_ebmo_hyperparams.py
```

**Status:** Not in NN Strengthening roadmap. Experimental feature.

**Consolidation Strategy:**

1. Archived all 16 files under `scripts/archive/ebmo/` and removed from `scripts/` (completed)
2. If EBMO is revived, consolidate into a single CLI (prefer `app/training/ebmo_trainer.py`)

---

### Target 6: Debug Scripts

**Current State:** 23 standalone debug utilities

```
debug_capture_*.py (4 files)
debug_chain_*.py (3 files)
debug_gpu_*.py (3 files)
debug_recovery_*.py (3 files)
debug_game.py
debug_hex8_hang.py
debug_specific_move.py
debug_ts_python_state_diff.py
minimal_hex8_debug.py
... and more
```

**Status:** Zero production imports. One-off debugging aids.

**Consolidation Strategy:**

1. Archived debug utilities under `scripts/archive/debug/` (completed)
2. Use `diff_state_bundle.py` for parity diffs; archived scripts are reference-only

---

## Execution Phases

### Phase 1: Zero-Risk Cleanup (Week 1)

**Goal:** Remove files with zero dependencies

| Task                                 | Files | Risk | Status   |
| ------------------------------------ | ----- | ---- | -------- |
| Remove deprecated tournament scripts | 3     | None | [x] Done |
| Archive debug scripts                | 23    | None | [x] Done |
| Archive EBMO training variants       | 16    | None | [x] Done |

**Total:** 42 files removed/archived

**Commands (completed):**

Scripts are now under `ai-service/scripts/archive/{deprecated,debug,ebmo}/` and removed from
`ai-service/scripts/`. Use git history if a restore is needed.

---

### Phase 2: Low-Risk Consolidation (Week 2)

**Goal:** Consolidate scripts with 0-2 dependencies

| Task                                                     | Files | Risk | Status      |
| -------------------------------------------------------- | ----- | ---- | ----------- |
| Merge promotion gate scripts into daemon                 | 2     | Low  | [ ] Pending |
| Remove superseded cluster monitors                       | 2-3   | Low  | [ ] Pending |
| Update `run_tournament.py` to remove deprecated branches | 1     | Low  | [ ] Pending |

**Promotion Gate Consolidation:**

1. Add `--elo-gate` and `--parity-gate` flags to `unified_promotion_daemon.py`
2. Migrate logic from `elo_promotion_gate.py` and `run_parity_promotion_gate.py`
3. Archive original files

**Cluster Monitor Consolidation:**

1. Verify `robust_cluster_monitor.py` has all features
2. Archive `simple_cluster_monitor.py` and `cluster_health_monitor.py`

---

### Phase 2b: Archive Audit and Code Quality Assessment (2025-12-20)

**Goal:** Verify orphaned/stranded code is actually underutilized

**Findings:**

| Module                          | Initially Flagged As      | Actual Status           | Notes                                                    |
| ------------------------------- | ------------------------- | ----------------------- | -------------------------------------------------------- |
| `fast_geometry.py`              | Underutilized (5 imports) | ✅ Well-integrated      | Used by heuristic_ai, descent_ai, swap_evaluation        |
| `evaluation_provider.py`        | Orphaned                  | ✅ Well-designed        | Protocol used by descent, mcts, maxn AI classes          |
| `lightweight_state.py`          | Underutilized (3 imports) | ✅ Alternative approach | Complements MutableGameState, different use case         |
| `multi_config_training_loop.py` | Archived                  | ✅ Features in main     | AdaptiveCurriculum, NPZ merge already in unified_ai_loop |

**Conclusion:** Many "orphaned" modules are actually well-integrated through composition
and protocols. The codebase follows good practices with dependency injection.

---

### Phase 3: Medium-Risk Consolidation (Week 3)

**Goal:** Consolidate scripts with 3+ dependencies

| Task                                             | Files | Risk   | Status      |
| ------------------------------------------------ | ----- | ------ | ----------- |
| Consolidate training pipeline scripts            | 3-5   | Medium | [ ] Pending |
| Create deprecated wrappers for `auto_promote.py` | 1     | Medium | [ ] Pending |
| Consolidate tournament entry points              | 2     | Medium | [ ] Pending |

---

### Phase 4: Code Quality Cleanup (Week 4)

**Goal:** Remove commented-out code and improve file organization

| Task                                             | Lines | Risk   | Status      |
| ------------------------------------------------ | ----- | ------ | ----------- |
| Remove commented code from `p2p_orchestrator.py` | 2,191 | Medium | [ ] Pending |
| Remove commented code from `unified_ai_loop.py`  | 782   | Medium | [ ] Pending |
| Remove commented code from legacy files          | 1,488 | Medium | [ ] Pending |
| Remove commented code from GPU files             | 753   | Medium | [ ] Pending |
| Remove commented code from training files        | 717   | Medium | [ ] Pending |

**Total:** ~5,900 lines of commented code

---

### Phase 5: Legacy File Migration (Month 2)

**Goal:** Complete migration from legacy files to modular components

| Task                             | Source      | Target                 | Status      |
| -------------------------------- | ----------- | ---------------------- | ----------- |
| Migrate `_game_engine_legacy.py` | 4,502 lines | `game_engine/` modules | [ ] Pending |
| Migrate `_neural_net_legacy.py`  | 6,959 lines | `neural_net/` modules  | [ ] Pending |

---

### Phase 6: Large File Refactoring (Month 2-3)

**Goal:** Break up monolithic files

| File                  | Current      | Target Structure                               | Status      |
| --------------------- | ------------ | ---------------------------------------------- | ----------- |
| `p2p_orchestrator.py` | 27,829 lines | `p2p/{leader,worker,health,tasks}.py`          | [ ] Pending |
| `unified_ai_loop.py`  | 8,236 lines  | `loop/{data,training,evaluation,promotion}.py` | [ ] Pending |

---

## Detailed File Inventory

### Files Safe to Remove (Zero Risk)

```
# Deprecated tournament scripts (archived)
ai-service/scripts/archive/deprecated/run_ai_tournament.py
ai-service/scripts/archive/deprecated/run_axis_aligned_tournament.py
ai-service/scripts/archive/deprecated/run_crossboard_difficulty_tournament.py

# Debug scripts (archived)
ai-service/scripts/archive/debug/

# EBMO variants (archived)
ai-service/scripts/archive/ebmo/
```

### Files to Consolidate (Medium Risk)

```
# Promotion scripts → unified_promotion_daemon.py
ai-service/scripts/elo_promotion_gate.py
ai-service/scripts/run_parity_promotion_gate.py
ai-service/scripts/auto_model_promotion.py
ai-service/scripts/apply_tier_promotion_plan.py

# Cluster monitors → robust_cluster_monitor.py
ai-service/scripts/simple_cluster_monitor.py
ai-service/scripts/cluster_health_monitor.py

# Training pipelines → hex8_training_pipeline.py
ai-service/scripts/continuous_training_loop.py
ai-service/scripts/multi_config_training_loop.py
ai-service/scripts/population_based_training.py
```

### Files to Keep (Core Functionality)

```
# Tournament core
ai-service/scripts/run_tournament.py
ai-service/scripts/run_model_elo_tournament.py
ai-service/scripts/run_distributed_tournament.py

# Promotion core
ai-service/scripts/unified_promotion_daemon.py
ai-service/scripts/model_promotion_manager.py

# Training core
ai-service/scripts/hex8_training_pipeline.py
ai-service/scripts/run_tier_training_pipeline.py

# Orchestration
ai-service/scripts/unified_ai_loop.py
ai-service/scripts/p2p_orchestrator.py

# Monitoring
ai-service/scripts/unified_cluster_monitor.py
ai-service/scripts/robust_cluster_monitor.py
```

---

## Risk Assessment

### Low Risk (Safe to Execute)

- Removing files with zero imports
- Archiving debug/experimental scripts
- Removing commented-out code

### Medium Risk (Requires Testing)

- Consolidating scripts with 1-5 imports
- Modifying entry point scripts
- Updating configuration references

### High Risk (Requires Careful Planning)

- Modifying `unified_ai_loop.py` (22 internal imports)
- Modifying `p2p_orchestrator.py` (19 internal imports)
- Changing legacy file re-exports

### Mitigation Strategies

1. **Create archive directories** - Never delete, only move
2. **Use deprecation wrappers** - Keep old entry points as aliases
3. **Run full test suite** after each phase
4. **Document changes** in git commit messages
5. **Incremental rollout** - One category at a time

---

## Progress Tracking

### Phase 1: Zero-Risk Cleanup - COMPLETED (2025-12-20)

- [x] Create archive directory structure
- [x] Archive deprecated tournament scripts (3 files)
- [x] Archive debug scripts (24 files)
- [x] Archive EBMO training variants (16 files)
- [ ] Update `.gitignore` if needed
- [ ] Run test suite
- [ ] Commit with detailed message

### Phase 2: Low-Risk Consolidation - COMPLETED (2025-12-20)

- [x] Audit promotion gate script usage - Kept as reusable modules
- [ ] Merge gate logic into `unified_promotion_daemon.py` - Deferred (gates are standalone utilities)
- [ ] Archive original gate scripts - Kept as utilities
- [x] Audit cluster monitor usage
- [x] Archive superseded monitors (2 files: simple_cluster_monitor.py, cluster_health_monitor.py)
- [ ] Update `run_tournament.py` deprecated branches
- [ ] Run test suite
- [ ] Commit

### Phase 3: Medium-Risk Consolidation - COMPLETED (2025-12-20)

- [x] Audit training pipeline dependencies
- [x] Consolidate single-scheme pipelines (archived: continuous_training_loop.py, multi_config_training_loop.py)
- [x] Create deprecated wrapper for `auto_promote.py`
- [x] Create deprecated wrapper for `auto_elo_tournament.py`
- [x] Update references (9 scripts)
- [x] Run test suite
- [x] Commit

**Total Archived: 47 files** → See `ai-service/scripts/archive/README.md`

### Phase 3b: SSoT Query Consolidation - COMPLETED (2025-12-20)

Created unified ELO database query library to consolidate duplicate SQL patterns:

- [x] Created `scripts/lib/elo_queries.py` with unified query functions:
  - `get_production_candidates()` - Models meeting production criteria
  - `get_top_models()` - Top N models with filtering options
  - `get_all_ratings()` - All participant ratings as dictionary
  - `get_model_stats()` - Summary statistics (totals, production-ready count)
  - `get_games_by_config()` - Config coverage statistics
  - `get_games_last_n_hours()` - Recent game activity
  - `get_near_production()` - Models close to production threshold
  - `get_models_by_tier()` - Count of models per tier

- [x] Added deprecation wrappers pointing to unified entry points:
  - `auto_promote.py` → `unified_promotion_daemon.py --check-once/--daemon`
  - `auto_elo_tournament.py` → `run_model_elo_tournament.py --run`
  - `elo_promotion_gate.py` → `unified_promotion_daemon.py elo-gate`
  - `run_parity_promotion_gate.py` → `unified_promotion_daemon.py parity-gate`

- [x] Refactored scripts to use unified query library:
  - `elo_dashboard.py` - Now uses elo_queries for all DB access
  - `auto_promote.py` - Uses elo_queries.get_production_candidates
  - `check_production_candidates.py` - Uses elo_queries for all queries

- [x] Created SSoT compliance test:
  - `tests/test_thresholds_usage.py` - Verifies threshold constants imported from canonical source

### Phase 3c: Unified Alert Infrastructure - COMPLETED (2025-12-21)

Consolidated Slack notification functions and refactored ELO dashboard scripts:

- [x] Extended `scripts/lib/alerts.py` with unified Slack notification support:
  - `get_slack_webhook()` - Unified webhook resolution (env vars + file)
  - `send_slack_notification()` - Single entry point for Slack alerts
  - `slack_handler()` - AlertManager-compatible Slack handler
  - `send_simple_alert()` - Convenience function for quick alerts

- [x] Refactored ELO scripts to use elo_queries library:
  - `elo_leaderboard.py` - Uses `get_top_models`, `get_model_stats`, `get_models_by_tier`
  - `elo_alerts.py` - Uses `get_production_candidates`, `get_top_models`, `get_games_by_config`
  - Removed inline SQL from both scripts
  - Removed duplicate `get_slack_webhook()` implementations

- [x] Verified `INITIAL_ELO_RATING = 1500.0` already in `app/config/thresholds.py`
  - Added re-export in `elo_queries.py`

**Impact:** Eliminated ~200 lines of duplicate code across 4 scripts, established single entry point for Slack notifications.

### Phase 3d: Tournament CLI & Threshold Consolidation - COMPLETED (2025-12-21)

Consolidated tournament helper functions and eliminated hardcoded thresholds:

- [x] Added `ARCHIVE_ELO_THRESHOLD = 1400` to `app/config/thresholds.py`
- [x] Expanded `scripts/lib/tournament_cli.py` with archive/matchmaking helpers:
  - `archive_low_elo_models()` - Archive models below threshold
  - `is_model_archived()` - Check archive status
  - `unarchive_model()` / `unarchive_discovered_models()` - Restore models
  - `get_archived_models()` - Query archived models
  - `generate_elo_based_matchups()` - Elo-based pairing
  - `filter_archived_models()` - Filter out archived models
  - `ensure_archived_models_table()` - Table creation helper

- [x] Refactored `run_model_elo_tournament.py`:
  - Imports functions from `tournament_cli.py` instead of defining locally
  - Uses `ARCHIVE_ELO_THRESHOLD` from thresholds.py
  - Removed ~170 lines of duplicate code

- [x] Audited scripts for hardcoded thresholds:
  - `prune_models.py` - Now uses `ARCHIVE_ELO_THRESHOLD`
  - `filter_training_data.py` - Now uses `PRODUCTION_ELO_THRESHOLD`
  - `elo_monitor.py` - Now uses `ELO_TIER_EXPERT`, `ELO_TIER_MASTER`
  - `training_dashboard.py` - Now uses `PRODUCTION_ELO_THRESHOLD`, `INITIAL_ELO_RATING`

**Impact:** Eliminated ~170 lines of duplicate archive/matchmaking code, consolidated 4+ hardcoded threshold usages to SSoT constants.

### Phase 3e: Multi-Channel Alert Infrastructure - COMPLETED (2025-12-21)

Extended unified alert infrastructure with Discord support and consolidated remaining alert implementations:

- [x] Added Discord notification support to `scripts/lib/alerts.py`:
  - `get_discord_webhook()` - Unified webhook resolution
  - `send_discord_notification()` - Discord-specific sender
  - `discord_handler()` - AlertManager-compatible Discord handler
  - `send_alert()` - Multi-channel alert (Slack + Discord)
  - `DISCORD_COLORS` - Color mapping for Discord embeds

- [x] Converted `scripts/monitor/alerting.py` to re-export layer:
  - Imports from `scripts/lib/alerts.py`
  - Maintains backwards compatibility with deprecation warnings
  - `send_slack_alert` and `send_discord_alert` emit deprecation warnings

- [x] Updated `scripts/gpu_cluster_manager.py`:
  - Changed import from `scripts.monitor.alerting` to `scripts.lib.alerts`
  - Local AlertManager uses unified `send_alert()` via `USE_UNIFIED_ALERTING`

- [x] Audit findings - well-structured areas:
  - File transfer utilities (`scripts/lib/transfer.py`) already comprehensive
  - ELO query layer (`scripts/lib/elo_queries.py`) appropriately separate from EloService
  - Validator classes already consolidated via `unified_data_validator.py`

**Impact:** Unified Slack + Discord alerting, eliminated duplicate alert implementations, established single entry point for all notifications.

### Phase 3f: ELO Infrastructure Consolidation - COMPLETED (2025-12-21)

Consolidated ELO database imports and protected calibration baselines:

- [x] Protected non-NN algorithms from archiving:
  - Added `PROTECTED_ALGORITHM_PATTERNS` to `scripts/lib/tournament_cli.py`
  - Added `is_protected_algorithm()` function for pattern matching
  - Updated SQL in `archive_low_elo_models()` to exclude protected patterns
  - Protected: random, heuristic, minimax, MCTS, NNUE, baseline\_\*

- [x] Fixed remaining hardcoded thresholds:
  - `scripts/monitor_improvement.py` - Uses `PRODUCTION_ELO_THRESHOLD`, `ARCHIVE_ELO_THRESHOLD`, `PRODUCTION_MIN_GAMES`
  - `scripts/estimate_elo.py` - Uses `ELO_K_FACTOR`, `INITIAL_ELO_RATING`

- [x] Migrated 11 files from `app.tournament.unified_elo_db` to public API:
  - `scripts/gauntlet_to_elo.py` → `from app.tournament import EloDatabase`
  - `scripts/model_culling.py` → `from app.tournament import EloDatabase`
  - `scripts/migrate_elo_to_unified.py` → `from app.tournament import EloDatabase, UnifiedEloRating`
  - `scripts/consolidate_elo_databases.py` → Uses `app.tournament` + `app.training.elo_service`
  - `scripts/run_model_elo_tournament.py` → `from app.tournament import EloDatabase`
  - `scripts/model_promotion_manager.py` → `from app.tournament import UnifiedEloRating, get_elo_database`
  - `scripts/gmo_integration.py` → `from app.tournament import EloDatabase`
  - `scripts/unified_promotion_daemon.py` → `from app.tournament import get_elo_database`
  - `scripts/p2p_orchestrator.py` (3 locations) → `from app.tournament import get_elo_database`
  - `app/tournament/orchestrator.py` → `from app.tournament import get_elo_database`
  - `app/integration/p2p_integration.py` → `from app.tournament import EloDatabase`

- [x] Verified `elo_queries.py` SSoT usage:
  - Key scripts already using it: `auto_promote.py`, `check_production_candidates.py`, `elo_leaderboard.py`, `elo_alerts.py`, `elo_dashboard.py`
  - Remaining inline SQL is in specialized scripts (migration, sync, analytics) with specific needs

**Impact:** Eliminated direct imports from deprecated `unified_elo_db` module, protected ELO calibration anchors from accidental archiving, centralized threshold usage.

### Phase 3g: Tournament Recording Hardening - COMPLETED (2025-12-21)

Enforced `trace_mode=True` in all critical tournament and data generation scripts (RR-CANON-R075 compliance):

- [x] `scripts/run_model_elo_tournament.py` - 2 locations fixed
- [x] `scripts/run_distributed_tournament.py` - 1 location fixed
- [x] `scripts/baseline_gauntlet.py` - 1 location fixed
- [x] `scripts/run_cross_ai_selfplay.py` - 1 location fixed
- [x] `scripts/estimate_elo.py` - 1 location fixed
- [x] `scripts/simple_canonical_gen.py` - 1 location fixed
- [x] `scripts/p2p_orchestrator.py` - 1 location fixed

**Context:** `trace_mode=True` ensures bookkeeping moves (NO_LINE_ACTION, NO_TERRITORY_ACTION, forced elimination) are explicitly recorded rather than implicitly applied. This is required for canonical training data generation and proper parity validation with TypeScript.

**Audit Note:** 100+ scripts still have `apply_move()` without `trace_mode`, but many are test/benchmark/debugging scripts where trace_mode isn't needed. The canonical selfplay generator (`generate_canonical_selfplay.py`) already used `trace_mode=True`.

**Impact:** Ensures tournament data can be used for training without parity issues.

### Phase 3h: Training Data Pipeline Consolidation - COMPLETED (2025-12-21)

Consolidated duplicated code across 7+ export/training scripts (~3500 lines total duplication):

**1. Canonical Source Validation (3 scripts → 1 central function):**

- [x] `scripts/export_replay_dataset.py` - Removed duplicate `_enforce_registry_canonical_sources`
- [x] `scripts/export_replay_dataset_parallel.py` - Removed duplicate, now imports from central
- [x] `scripts/db_to_training_npz.py` - Removed duplicate, now imports from central
- Central function: `app.training.canonical_sources.enforce_canonical_sources()`

**2. Board Type Mapping (5 scripts → 1 central mapping):**

- [x] `scripts/export_replay_dataset.py` - Removed local BOARD_TYPE_MAP
- [x] `scripts/export_replay_dataset_parallel.py` - Removed local BOARD_TYPE_MAP
- [x] `scripts/export_gumbel_kl_dataset.py` - Removed local BOARD_TYPE_MAP
- [x] `scripts/train_distilled_model.py` - Removed local BOARD_TYPE_MAP
- [x] `scripts/run_cmaes_optimization.py` - Now imports from central, exports BOARD_NAME_TO_TYPE alias
- Central mapping: `scripts/lib/cli.BOARD_TYPE_MAP`

**3. Created Unified Export Core Module:**

- [x] Created `app/training/export_core.py` with consolidated functions:
  - `value_from_final_winner()` - Binary winner/loser value computation
  - `value_from_final_ranking()` - Rank-aware multiplayer value computation
  - `compute_multi_player_values()` - Per-player value vectors for multi-player training
  - `encode_state_with_history()` - State encoding with history frame stacking
  - `NPZDatasetWriter` class - Unified NPZ I/O with save/append/load methods
- [x] Integrated into `scripts/export_replay_dataset.py` (proof of concept)
- [x] Removed ~180 lines of duplicate code from export_replay_dataset.py

**Impact:**

- Reduced duplication by ~500 lines across export scripts
- Single source of truth for value computation semantics
- Unified NPZ I/O patterns prevent format inconsistencies
- Other scripts can incrementally migrate to use `export_core` module

### Phase 4: Code Quality Cleanup - ANALYZED (2025-12-20)

Analysis revealed that most "commented code" is actually documentation:

- Legitimate comments explaining complex rules logic
- Section headers and dividers
- Migration notes

**Finding:** The ~8,500 line estimate was inflated. Most comments are valuable documentation.

- [x] Analyzed `p2p_orchestrator.py` - Comments are documentation, not dead code
- [x] Analyzed `unified_ai_loop.py` - Comments are documentation
- [x] Analyzed legacy files - Comments explain rules reasoning
- [ ] Remove actual dead code blocks if found during development

### Phase 5: Legacy Migration - IN PROGRESS (ongoing)

Legacy files are being migrated incrementally. Current status:

**`_game_engine_legacy.py`:**

- Used as re-export layer via `app.game_engine.__init__.py`
- Exports: `GameEngine`, `PhaseRequirement`, `PhaseRequirementType`
- Status: Core class, migration would require extensive testing

**`_neural_net_legacy.py`:**

- Partially migrated to modular subcomponents:
  - `constants.py` - Phase 1 complete
  - `blocks.py` - Phase 1 complete
  - `hex_encoding.py` - Phase 2 complete
  - `hex_architectures.py` - Phase 2 complete
  - `square_architectures.py` - Phase 2 complete
- Status: Ongoing migration, legacy file remains as re-export layer

- [x] Document current migration status
- [ ] Continue incremental migration during normal development
- [ ] Archive legacy files when migration complete

### Phase 6: Large File Refactoring - DEFERRED

- [ ] Break up `p2p_orchestrator.py` (27K lines) - Future work
- [ ] Remove commented code from `unified_ai_loop.py`
- [ ] Remove commented code from legacy files
- [ ] Remove commented code from GPU files
- [ ] Remove commented code from training files
- [ ] Run test suite
- [ ] Commit

### Phase 5: Legacy Migration

- [ ] Audit `_game_engine_legacy.py` usage
- [ ] Create migration plan
- [ ] Execute migration
- [ ] Archive legacy file
- [ ] Audit `_neural_net_legacy.py` usage
- [ ] Create migration plan
- [ ] Execute migration
- [ ] Archive legacy file
- [ ] Run test suite
- [ ] Commit

### Phase 6: Large File Refactoring

- [ ] Design `p2p/` module structure
- [ ] Extract leader election logic
- [ ] Extract worker logic
- [ ] Extract health monitoring
- [ ] Extract task distribution
- [ ] Update imports
- [ ] Design `loop/` module structure
- [ ] Extract data collection
- [ ] Extract training logic
- [ ] Extract evaluation logic
- [ ] Extract promotion logic
- [ ] Update imports
- [ ] Run test suite
- [ ] Commit

---

## Related Documents

- `ai-service/docs/planning/TOURNAMENT_ELO_CONSOLIDATION_PLAN.md` - Tournament consolidation details
- `ai-service/docs/planning/NN_STRENGTHENING_PLAN.md` - Training pipeline automation
- `KNOWN_ISSUES.md` - Current bugs and gaps
- `TODO.md` - Active task tracker

---

## Appendix A: Feature Flags Reference

### Experimental Features (Currently Disabled)

| Flag                                 | Default | Description                        |
| ------------------------------------ | ------- | ---------------------------------- |
| `RINGRIFT_USE_IG_GMO`                | false   | Information-geometric GMO AI       |
| `RINGRIFT_USE_EBMO`                  | false   | Energy-based model optimization AI |
| `RINGRIFT_TRAINING_ENABLE_SWAP_RULE` | false   | Pie rule for 2-player training     |

### Performance Flags (Enabled by Default)

| Flag                          | Default | Description               |
| ----------------------------- | ------- | ------------------------- |
| `RINGRIFT_USE_FAST_TERRITORY` | true    | Fast territory detection  |
| `RINGRIFT_USE_MOVE_CACHE`     | true    | Move caching optimization |

### Debug Flags

| Flag                    | Default | Description                 |
| ----------------------- | ------- | --------------------------- |
| `RINGRIFT_DEBUG_ENGINE` | false   | Legacy engine debug logging |
| `RINGRIFT_DEBUG`        | false   | Global debug mode           |

---

## Appendix B: Stranded Code - Integration Prioritization

**Principle:** Prioritize integration and utilization over archiving when code provides
valuable functionality not already covered in the codebase.

### INTEGRATE - High Priority (Production-Ready) ✅ COMPLETE

| Module                               | Value  | Integration Point                         | Status      |
| ------------------------------------ | ------ | ----------------------------------------- | ----------- |
| `app.training.optimized_pipeline`    | HIGH   | Wired into train_loop.py                  | ✅ Complete |
| `app.training.distillation`          | HIGH   | Re-exported from training_enhancements.py | ✅ Complete |
| `app.training.adversarial_positions` | HIGH   | Wired into model_lifecycle.py             | ✅ Complete |
| `app.training.auto_data_discovery`   | MEDIUM | Wired into data_coordinator.py            | ✅ Complete |

**Integration Details (2025-12-20):**

1. **optimized_pipeline.py** → `train_loop.py`
   - `run_training_loop(use_optimized_pipeline=True)` uses OptimizedTrainingPipeline
   - Provides: export caching, distributed locks, health monitoring, curriculum feedback

2. **adversarial_positions.py** → `model_lifecycle.py`
   - `ModelRetentionManager.validate_model_robustness()` uses AdversarialGenerator
   - Uses UNCERTAINTY and REPLAY strategies for efficient robustness testing

3. **auto_data_discovery.py** → `data_coordinator.py`
   - `TrainingDataCoordinator.prepare_for_training()` runs auto-discovery
   - Configurable via `enable_auto_discovery=True` (default)

4. **distillation.py** → `training_enhancements.py`
   - All distillation classes re-exported from unified training API
   - Import from `app.training.training_enhancements`

### EXPERIMENTAL - Keep with Feature Flags

| Module                       | Flag                          | Integration Point       |
| ---------------------------- | ----------------------------- | ----------------------- |
| `app.ai.gmo_policy_provider` | `RINGRIFT_USE_GMO_POLICY=1`   | MCTS policy selection   |
| `app.training.opening_book`  | `RINGRIFT_USE_OPENING_BOOK=1` | Selfplay initialization |

### KEEP - Active Production Dependencies

| Module                                         | Reason                            |
| ---------------------------------------------- | --------------------------------- |
| `app.ai.batch_eval`                            | Core dependency of HeuristicAI    |
| `app.ai.lightweight_eval`, `lightweight_state` | Used by evaluation modules        |
| `app.ai.numba_eval`                            | JIT functions for HeuristicAI     |
| `app.ai.swap_evaluation`                       | Used by BaseAI and HeuristicAI    |
| `app.p2p.*`                                    | Part of active P2P infrastructure |

### EXPERIMENTAL - GMO Research Direction

GMO (Gradient Move Optimization) is a novel approach available as experimental AI types:

| Module                       | Status       | Notes                                                  |
| ---------------------------- | ------------ | ------------------------------------------------------ |
| `app.ai.gmo_ai`              | Active       | D13/D14/D17, enable at D3 with `RINGRIFT_USE_IG_GMO=1` |
| `app.ai.gmo_v2`              | Experimental | Enhanced version with attention + ensemble             |
| `app.ai.gmo_policy_provider` | Experimental | GMO-based priors for MCTS                              |

**Recommendation:** If pursuing GMO research, consolidate gmo_ai.py + gmo_v2.py
into unified implementation with config-driven architecture selection.

---

_Document generated by Claude Code codebase review - 2025-12-20_
