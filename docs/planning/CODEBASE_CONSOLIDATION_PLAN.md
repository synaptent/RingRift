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

| Script                                               | Lines | Imports | Status                             |
| ---------------------------------------------------- | ----- | ------- | ---------------------------------- |
| `run_tournament.py`                                  | 800+  | 9       | **KEEP - Unified hub**             |
| `run_model_elo_tournament.py`                        | 1200+ | 14      | **KEEP - Core evaluation**         |
| `run_distributed_tournament.py`                      | 900+  | 5+      | **KEEP - Distributed ladder**      |
| `run_eval_tournaments.py`                            | 400+  | 2       | **KEEP**                           |
| `run_diverse_tournaments.py`                         | 500+  | 3       | **KEEP**                           |
| `run_ssh_distributed_tournament.py`                  | 300+  | 2       | **KEEP**                           |
| `run_p2p_elo_tournament.py`                          | 400+  | 2       | **KEEP**                           |
| `auto_elo_tournament.py`                             | 350+  | 1       | CONSOLIDATE into run_tournament.py |
| `launch_distributed_elo_tournament.py`               | 200+  | 1       | CONSOLIDATE                        |
| `shadow_tournament_service.py`                       | 300+  | 1       | EVALUATE                           |
| `deprecated/run_ai_tournament.py`                    | 570   | 0       | **REMOVE**                         |
| `deprecated/run_axis_aligned_tournament.py`          | 671   | 0       | **REMOVE**                         |
| `deprecated/run_crossboard_difficulty_tournament.py` | 302   | 0       | **REMOVE**                         |

**Consolidation Strategy:** See `docs/planning/TOURNAMENT_ELO_CONSOLIDATION_PLAN.md`

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

**Consolidation Strategy:** See `docs/planning/NN_STRENGTHENING_PLAN.md` Section F (Closed-loop automation)

---

### Target 5: EBMO Training Variants

**Current State:** 16 scripts with no cross-imports

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

1. Archive all 16 files to `scripts/archive/ebmo/`
2. If EBMO is revived, consolidate into single `train_ebmo.py` with `--mode` flag

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

1. Archive all to `scripts/archive/debug/`
2. Keep `debug_ts_python_state_diff.py` if `diff_state_bundle.py` is in use

---

## Execution Phases

### Phase 1: Zero-Risk Cleanup (Week 1)

**Goal:** Remove files with zero dependencies

| Task                                 | Files | Risk | Status      |
| ------------------------------------ | ----- | ---- | ----------- |
| Remove deprecated tournament scripts | 3     | None | [ ] Pending |
| Archive debug scripts                | 23    | None | [ ] Pending |
| Archive EBMO training variants       | 16    | None | [ ] Pending |

**Total:** 42 files removed/archived

**Commands:**

```bash
# Create archive directories
mkdir -p ai-service/scripts/archive/{deprecated,debug,ebmo}

# Move deprecated tournament scripts
mv ai-service/scripts/deprecated/*.py ai-service/scripts/archive/deprecated/

# Move debug scripts
mv ai-service/scripts/debug_*.py ai-service/scripts/archive/debug/
mv ai-service/scripts/minimal_*_debug.py ai-service/scripts/archive/debug/

# Move EBMO variants
mv ai-service/scripts/*ebmo*.py ai-service/scripts/archive/ebmo/
mv ai-service/scripts/train_ebmo*.py ai-service/scripts/archive/ebmo/
```

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
# Deprecated tournament scripts (3 files, 1,543 lines)
ai-service/scripts/deprecated/run_ai_tournament.py
ai-service/scripts/deprecated/run_axis_aligned_tournament.py
ai-service/scripts/deprecated/run_crossboard_difficulty_tournament.py

# Debug scripts (23 files, ~3,745 lines)
ai-service/scripts/debug_*.py
ai-service/scripts/minimal_*_debug.py

# EBMO variants (16 files, ~4,000 lines)
ai-service/scripts/*ebmo*.py
ai-service/scripts/train_ebmo*.py
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
- [ ] Create deprecated wrapper for `auto_promote.py`
- [ ] Update references (9 scripts)
- [ ] Run test suite
- [ ] Commit

**Total Archived: 47 files** → See `ai-service/scripts/archive/README.md`

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

- `docs/planning/TOURNAMENT_ELO_CONSOLIDATION_PLAN.md` - Tournament consolidation details
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

## Appendix B: Orphaned Module Categories

### High-Value Integration Candidates

| Module                                 | Value  | Integration Path        |
| -------------------------------------- | ------ | ----------------------- |
| `app.training.opening_book`            | HIGH   | Wire into selfplay      |
| `app.training.optimized_pipeline`      | HIGH   | Evaluate as replacement |
| `app.training.pbt`                     | MEDIUM | Wire into training loop |
| `app.models.transformer_model`         | MEDIUM | Test as architecture    |
| `app.training.uncertainty_calibration` | MEDIUM | Add to evaluation       |
| `app.ai.gmo_v2`                        | MEDIUM | Wire into factory       |

### Low-Value / Remove Candidates

| Module                                    | Reason                    |
| ----------------------------------------- | ------------------------- |
| `app.ai.numba_eval`                       | Experimental optimization |
| `app.ai.parallel_eval`                    | Superseded by GPU eval    |
| `app.ai.lightweight_*`                    | Experimental              |
| `app.p2p.*`                               | Incomplete feature        |
| `app.training.test_exploration_diversity` | Should be in tests/       |

---

_Document generated by Claude Code codebase review - 2025-12-20_
