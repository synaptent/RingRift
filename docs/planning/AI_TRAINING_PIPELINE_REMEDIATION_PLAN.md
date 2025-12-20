# AI Training Pipeline Remediation Plan

> **Doc Status (2025-12-20): Active Draft**
>
> **Purpose:** Comprehensive remediation plan to establish a fully gated, cross-language AI training data pipeline for RingRift.
>
> **Owner:** TBD  
> **Scope:** AI service, training infrastructure, canonical databases, neural network models
>
> **References:**
>
> - [`ai-service/TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md) - Data classification and provenance
> - [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md`](../ai/AI_TRAINING_ASSESSMENT_FINAL.md) - Infrastructure assessment
> - [`docs/planning/NEXT_AREAS_EXECUTION_PLAN.md`](./NEXT_AREAS_EXECUTION_PLAN.md) - Overall execution context
> - [`ai-service/app/db/game_replay.py`](../../ai-service/app/db/game_replay.py) - Database schema
> - [`ai-service/scripts/generate_canonical_selfplay.py`](../../ai-service/scripts/generate_canonical_selfplay.py) - Self-play generator

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Current State Assessment](#1-current-state-assessment)
- [2. Problem Analysis](#2-problem-analysis)
- [3. Remediation Subtasks](#3-remediation-subtasks)
- [4. Dependency Diagram](#4-dependency-diagram)
- [5. Phase Ordering](#5-phase-ordering)
- [6. Risk Assessment](#6-risk-assessment)
- [7. Success Criteria](#7-success-criteria)
- [Revision History](#revision-history)

---

## Executive Summary

### Goal

Establish a fully gated, cross-language training data pipeline that produces canonical datasets for all board types (square8, square19, hexagonal) and supports neural network training at scale.

### Current State

The AI Training Pipeline was identified as the **weakest aspect** in the comprehensive project assessment post-Production Validation. Key blockers include:

| Issue                      | Current State                                                         | Target State                       |
| -------------------------- | --------------------------------------------------------------------- | ---------------------------------- |
| Schema completeness        | Large-board DBs now include `game_moves`, but parity gate still fails | All DBs gateable with full schema  |
| Training data volume       | ~100 games total                                                      | 10,000+ games per board type       |
| Neural network performance | 75% win rate vs random                                                | ≥90% win rate (matching heuristic) |
| Parity gating              | Blocked by phase invariant violations on large boards                 | All DBs pass TS↔Python parity      |

### Primary Risk

Cannot train production-quality neural models until the canonical data pipeline is unblocked. The heuristic AI remains the only reliable option for production use.

### Success Definition

1. All canonical DBs (square8, square19, hexagonal) pass parity + canonical history gates
2. 500+ canonical games per board type in the training pool
3. Neural network achieves ≥85% win rate vs random after extended training
4. Minimax/MCTS wired into production difficulty ladder

---

## 1. Current State Assessment

### 1.1 Database Schema Status

| Database                  | Board Type | Games | `game_moves` Table | Parity Gate | Status           |
| ------------------------- | ---------- | ----- | ------------------ | ----------- | ---------------- |
| `canonical_square8_2p.db` | square8    | 200   | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square8_3p.db` | square8    | 2     | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square8_4p.db` | square8    | 2     | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square19.db`   | square19   | 1     | ✅ Present         | ❌ FAIL     | **pending_gate** |
| `canonical_hexagonal.db`  | hexagonal  | 1     | ✅ Present         | ❌ FAIL     | **pending_gate** |

**Root Cause:** The large-board DBs (square19, hexagonal) are schema-complete, but parity gating fails due to phase invariant violations in generated move histories (for example forced elimination moves recorded while still in `territory_processing`). A suspected contributor was Python territory-processing eligibility rejecting height-1 stacks outside a region (RR-CANON-R145), which has now been aligned to the canonical elimination rules.

### 1.2 Training Data Volume

| Board Type   | Current         | Target (Baseline) | Target (Training) | Gap        |
| ------------ | --------------- | ----------------- | ----------------- | ---------- |
| square8 (2p) | 200             | ≥200              | ≥1,000            | 800 games  |
| square8 (3p) | 2               | ≥32               | ≥500              | 498 games  |
| square8 (4p) | 2               | ≥32               | ≥500              | 498 games  |
| square19     | 1 (failed gate) | ≥200              | ≥1,000            | 999+ games |
| hexagonal    | 1 (failed gate) | ≥200              | ≥1,000            | 999+ games |

### 1.3 Neural Network Performance

From [`AI_TRAINING_ASSESSMENT_FINAL.md`](../ai/AI_TRAINING_ASSESSMENT_FINAL.md):

| Metric                  | Neural Network | Heuristic      | Target  |
| ----------------------- | -------------- | -------------- | ------- |
| Win rate vs random      | 75%            | 90%            | ≥90%    |
| 95% CI                  | [53.1%, 88.8%] | [69.9%, 97.2%] | -       |
| Effect size (Cohen's h) | 0.52 (medium)  | 0.93 (large)   | ≥0.78   |
| Training epochs         | 5              | N/A            | 50+     |
| Training games          | ~100           | N/A            | 10,000+ |

**Conclusion:** Neural network is undertrained due to insufficient data and epochs. Infrastructure is ready for extended training.

### 1.4 Blocked Pipeline Components

```
                    ┌─────────────────────────────────┐
                    │   Canonical Self-Play Generator  │
                    │ generate_canonical_selfplay.py   │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │     GameReplayDB (Schema v9)     │
                    │  - game_moves table REQUIRED     │
                    │  - snapshots table               │
                    │  - games table                   │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┬───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ square8.db  │ │ square19.db │ │hexagonal.db │
            │  ✅ PASS    │ │ ❌ PARITY   │ │ ❌ PARITY   │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    │               X               X
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────────────────────────────────────┐
            │         TS↔Python Parity Gate               │
            │ check_ts_python_replay_parity.py            │
            │ check_canonical_phase_history.py            │
            └─────────────────────────────────────────────┘
                                    │
                                    X (blocked by phase invariant errors)
                                    │
                                    ▼
            ┌─────────────────────────────────────────────┐
            │           Training Data Export              │
            │              *.npz datasets                 │
            └─────────────────────────────────────────────┘
                                    │
                                    X
                                    │
                                    ▼
            ┌─────────────────────────────────────────────┐
            │        Neural Network Training              │
            │  ringrift_v2_square19/hex.pth              │
            └─────────────────────────────────────────────┘
```

---

## 2. Problem Analysis

### 2.1 Parity Gate Failures (Phase Invariant Violations)

**Background:** The parity gate now runs on large-board DBs (schema is complete), but TS replay fails due to phase/move invariant violations. Recent failures show `forced_elimination` moves applied while the `currentPhase` is still `territory_processing`, and canonical history validation caught `no_placement_action` in `territory_processing`.

**Impact:**

- canonical_square19.db: parity gate fails on a forced-elimination move recorded in the wrong phase.
- canonical_hexagonal.db: canonical history validation fails; parity gate also fails on forced elimination in `territory_processing`.
- These datasets remain non-canonical and cannot be used for training until regeneration produces valid phase histories.

**Root Cause Analysis (Hypothesis):**

1. Python territory-processing eligibility rejected height-1 stacks outside a region (RR-CANON-R145), leading to no_territory_action → forced_elimination sequences without the canonical phase transition; fixed by delegating eligibility to the elimination helper.
2. Self-play generator/engine emits forced elimination without entering the `forced_elimination` phase.
3. Phase transitions may be skipped or recorded out of order under long-running large-board games.
4. The generator does not fail fast on invalid phase/move sequences, allowing invalid games into the DB.

### 2.2 Insufficient Training Data

**Background:** The neural network was trained on ~100 games with only 5 epochs, which is insufficient for the model to learn complex board positions.

**Impact:**

- Neural network performs 15% worse than heuristic AI
- Head-to-head vs heuristic: 30-70 (not statistically significant, but trend is clear)
- Cannot deploy neural-based difficulty levels

**Recommendation from Assessment:**

> Focus future training efforts on **neural network self-play data generation (10,000+ games)** and **extended training (50+ epochs)** rather than heuristic weight optimization.

### 2.3 Production AI Integration Gap

**Background:** The production AI ladder currently uses:

- Levels 1-2: RandomAI / HeuristicAI (local TypeScript)
- Levels 3-6: MinimaxAI (available but not wired to production)
- Levels 7-8: MCTSAI (available but not wired to production)
- Levels 9-10: DescentAI (available but not wired to production)

**Impact:**

- Players only experience heuristic-based AI
- No progression to stronger search-based AI
- Neural network inference not available in production

---

## 3. Remediation Subtasks

### Phase 1: Schema & DB Fix

#### AI-01: Diagnose Parity/Phase Invariant Failures

| Attribute               | Value                                                                                                                                                                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-01                                                                                                                                                                                                                                                                                                                   |
| **Title**               | Diagnose parity/phase invariant failures in large-board DBs                                                                                                                                                                                                                                                             |
| **Description**         | Inspect canonical_square19.db and canonical_hexagonal.db to confirm schema completeness and identify the failing move/phase invariants reported by parity + canonical history gates. Capture the exact move type + phase mismatch and decide whether the fix is in self-play generation, engine rules, or gate filters. |
| **Acceptance Criteria** | <ul><li>Schema version confirmed for each DB</li><li>Failing move/phase invariants identified (e.g., forced_elimination in territory_processing)</li><li>Canonical history failure examples captured</li><li>Decision made: fix generator vs engine vs gate filter</li></ul>                                            |
| **Key Files**           | <ul><li>`ai-service/app/db/game_replay.py`</li><li>`ai-service/data/games/canonical_square19.db`</li><li>`ai-service/data/games/canonical_hexagonal.db`</li></ul>                                                                                                                                                       |
| **Diagnostic Commands** | `bash<br>cd ai-service<br>sqlite3 data/games/canonical_square19.db ".schema"<br>sqlite3 data/games/canonical_hexagonal.db ".schema"<br>cat data/games/db_health.canonical_square19.json<br>cat data/games/db_health.canonical_hexagonal.json<br>`                                                                       |
| **Dependencies**        | None                                                                                                                                                                                                                                                                                                                    |
| **Recommended Mode**    | debug                                                                                                                                                                                                                                                                                                                   |

#### AI-02: Fix Phase Invariants and Regenerate Canonical DBs

| Attribute               | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Title**               | Fix phase invariants and regenerate canonical DBs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Description**         | Address the phase invariant bug(s) in self-play generation (forced elimination or no-action moves recorded in the wrong phase). Regenerate canonical_square19.db and canonical_hexagonal.db with the canonical self-play generator. Archive existing DBs before regeneration. Start with a small number of games (32-64) to verify parity + canonical history gates pass.                                                                                                                                                                                                                                                                                                                                                                             |
| **Acceptance Criteria** | <ul><li>Existing DBs archived with timestamp</li><li>New DBs created with `generate_canonical_selfplay.py`</li><li>`game_moves` table present in new DBs</li><li>At least 32 games per board type generated</li><li>Parity gate passes for both DBs</li><li>Canonical history validation passes</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Key Files**           | <ul><li>`ai-service/scripts/generate_canonical_selfplay.py`</li><li>`ai-service/TRAINING_DATA_REGISTRY.md`</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Commands**            | `bash<br>cd ai-service<br># Archive existing DBs<br>mv data/games/canonical_square19.db data/games/canonical_square19.db.pre_regen_$(date +%Y%m%d)<br>mv data/games/canonical_hexagonal.db data/games/canonical_hexagonal.db.pre_regen_$(date +%Y%m%d)<br><br># Regenerate with proper schema<br>PYTHONPATH=. python scripts/generate_canonical_selfplay.py \<br>  --board-type square19 \<br>  --num-games 32 \<br>  --db data/games/canonical_square19.db \<br>  --summary logs/db_health.canonical_square19.json<br><br>PYTHONPATH=. python scripts/generate_canonical_selfplay.py \<br>  --board-type hexagonal \<br>  --num-games 32 \<br>  --db data/games/canonical_hexagonal.db \<br>  --summary logs/db_health.canonical_hexagonal.json<br>` |
| **Dependencies**        | AI-01                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Recommended Mode**    | code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

#### AI-03: Run Parity Gate on Regenerated DBs

| Attribute               | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Task ID**             | AI-03                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Title**               | Run parity gate on regenerated DBs                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Description**         | Execute TS↔Python parity validation and canonical phase history checks on the newly regenerated square19 and hexagonal DBs. Verify `passed_canonical_parity_gate: true` and `canonical_ok: true` in the gate summaries.                                                                                                                                                                                                                                                        |
| **Acceptance Criteria** | <ul><li>Parity gate executed for both DBs</li><li>`games_with_semantic_divergence: 0` for both</li><li>`passed_canonical_parity_gate: true` for both</li><li>Canonical history validation passes</li><li>Gate summary JSONs saved alongside registry</li><li>TRAINING_DATA_REGISTRY.md updated with Status = canonical</li></ul>                                                                                                                                               |
| **Key Files**           | <ul><li>`ai-service/scripts/check_ts_python_replay_parity.py`</li><li>`ai-service/scripts/check_canonical_phase_history.py`</li><li>`ai-service/TRAINING_DATA_REGISTRY.md`</li></ul>                                                                                                                                                                                                                                                                                           |
| **Commands**            | `bash<br>cd ai-service<br># Run parity gate (included in generate_canonical_selfplay.py)<br># Or run standalone:<br>PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \<br>  --db data/games/canonical_square19.db \<br>  --summary logs/parity_summary.canonical_square19.json<br><br>PYTHONPATH=. python scripts/check_canonical_phase_history.py \<br>  --db data/games/canonical_square19.db \<br>  --summary logs/history_summary.canonical_square19.json<br>` |
| **Dependencies**        | AI-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Recommended Mode**    | code                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

---

### Phase 2: Data Scaling

#### AI-04: Scale Self-Play to 500 Games per Board Type

| Attribute               | Value                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-04                                                                                                                                                                                                                                                                                                                                                                        |
| **Title**               | Scale self-play to 500 games per board type                                                                                                                                                                                                                                                                                                                                  |
| **Description**         | After the parity gate passes, scale up canonical self-play generation to produce at least 500 games per board type (square8_2p, square8_3p, square8_4p, square19, hexagonal). Use distributed self-play across SSH hosts if available. Run parity gates on the scaled DBs.                                                                                                   |
| **Acceptance Criteria** | <ul><li>≥500 games per board type in canonical DBs</li><li>All DBs pass parity + canonical history gate</li><li>NPZ datasets exported for training</li><li>Training samples count documented per board</li></ul>                                                                                                                                                             |
| **Key Files**           | <ul><li>`ai-service/scripts/generate_canonical_selfplay.py`</li><li>`ai-service/data/training/*.npz`</li></ul>                                                                                                                                                                                                                                                               |
| **Volume Targets**      | <table><tr><th>Board Type</th><th>Target Games</th><th>Est. Samples</th></tr><tr><td>square8_2p</td><td>500</td><td>~30,000</td></tr><tr><td>square8_3p</td><td>500</td><td>~35,000</td></tr><tr><td>square8_4p</td><td>500</td><td>~40,000</td></tr><tr><td>square19</td><td>500</td><td>~100,000</td></tr><tr><td>hexagonal</td><td>500</td><td>~120,000</td></tr></table> |
| **Commands**            | `bash<br>cd ai-service<br># Use distributed self-play for large-board types<br>PYTHONPATH=. python scripts/generate_canonical_selfplay.py \<br>  --board-type square19 \<br>  --num-games 500 \<br>  --db data/games/canonical_square19.db \<br>  --summary logs/db_health.canonical_square19.json \<br>  --hosts lambda1,lambda2,lambda3<br>`                               |
| **Dependencies**        | AI-03                                                                                                                                                                                                                                                                                                                                                                        |
| **Recommended Mode**    | code                                                                                                                                                                                                                                                                                                                                                                         |

#### AI-05: Update TRAINING_DATA_REGISTRY.md with Gate Summaries

| Attribute               | Value                                                                                                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-05                                                                                                                                                                                                                                  |
| **Title**               | Update TRAINING_DATA_REGISTRY.md with gate summaries                                                                                                                                                                                   |
| **Description**         | After scaling, update the training data registry with the new game counts, gate summary references, and status changes. Ensure all canonical DBs are documented with their provenance and parity gate results.                         |
| **Acceptance Criteria** | <ul><li>All canonical DBs listed with correct game counts</li><li>Gate summary JSON paths documented</li><li>Status = canonical for all passing DBs</li><li>Volume targets table updated</li><li>NPZ export paths documented</li></ul> |
| **Key Files**           | <ul><li>`ai-service/TRAINING_DATA_REGISTRY.md`</li><li>`ai-service/logs/db_health.*.json`</li></ul>                                                                                                                                    |
| **Dependencies**        | AI-04                                                                                                                                                                                                                                  |
| **Recommended Mode**    | architect                                                                                                                                                                                                                              |

---

### Phase 3: Neural Network Training

#### AI-06: Extended Neural Network Training (50+ Epochs)

| Attribute                  | Value                                                                                                                                                                                                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**                | AI-06                                                                                                                                                                                                                                                    |
| **Title**                  | Extended neural network training (50+ epochs)                                                                                                                                                                                                            |
| **Description**            | Train the neural network on the scaled canonical datasets for at least 50 epochs. Use the existing training infrastructure with appropriate hyperparameters. Monitor for loss convergence and early stopping. Generate checkpoints at regular intervals. |
| **Acceptance Criteria**    | <ul><li>Training runs for ≥50 epochs (or early stopping)</li><li>Final loss < 0.5 (from 0.7 at epoch 5)</li><li>Checkpoints saved at epochs 10, 20, 30, 40, 50</li><li>Training logs with loss curves saved</li><li>Best checkpoint identified</li></ul> |
| **Key Files**              | <ul><li>`ai-service/app/training/train.py`</li><li>`ai-service/checkpoints/*.pth`</li><li>`ai-service/data/training/*.npz`</li></ul>                                                                                                                     |
| **Training Configuration** | `python<br>TrainConfig(<br>  batch_size=128,<br>  epochs=50,<br>  learning_rate=0.001,<br>  early_stopping_patience=5,<br>  checkpoint_interval=10,<br>  validation_split=0.1<br>)<br>`                                                                  |
| **Commands**               | `bash<br>cd ai-service<br>PYTHONPATH=. python -m app.training.train \<br>  --epochs 50 \<br>  --batch-size 128 \<br>  --checkpoint-interval 10 \<br>  --data data/training/canonical_square8_2p.npz<br>`                                                 |
| **Dependencies**           | AI-04                                                                                                                                                                                                                                                    |
| **Recommended Mode**       | code                                                                                                                                                                                                                                                     |

#### AI-07: Evaluate Neural Network Improvement

| Attribute               | Value                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Task ID**             | AI-07                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Title**               | Evaluate neural network improvement                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Description**         | Evaluate the trained neural network against the baselines (random, heuristic) using the statistical evaluation framework. Run at least 50 games per matchup for 80% statistical power. Document win rates with confidence intervals.                                                                                                                                                                                                             |
| **Acceptance Criteria** | <ul><li>Neural vs random: ≥85% win rate</li><li>Neural vs heuristic: ≥40% win rate (competitive)</li><li>50+ games per matchup</li><li>Statistical report with CIs and p-values</li><li>Effect sizes documented</li></ul>                                                                                                                                                                                                                        |
| **Key Files**           | <ul><li>`ai-service/scripts/evaluate_ai_models.py`</li><li>`ai-service/scripts/generate_statistical_report.py`</li><li>`ai-service/results/*.json`</li></ul>                                                                                                                                                                                                                                                                                     |
| **Commands**            | `bash<br>cd ai-service<br># Evaluate neural vs random<br>PYTHONPATH=. python scripts/evaluate_ai_models.py \<br>  --player1 neural \<br>  --player2 random \<br>  --games 50 \<br>  --board square8 \<br>  --output results/neural_vs_random.json<br><br># Generate statistical report<br>PYTHONPATH=. python scripts/generate_statistical_report.py \<br>  --results-dir results/ \<br>  --output results/statistical_analysis_report.json<br>` |
| **Dependencies**        | AI-06                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Recommended Mode**    | debug                                                                                                                                                                                                                                                                                                                                                                                                                                            |

#### AI-08: Compare Neural vs Heuristic at Difficulty Levels

| Attribute               | Value                                                                                                                                                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-08                                                                                                                                                                                                                                   |
| **Title**               | Compare neural vs heuristic at difficulty levels                                                                                                                                                                                        |
| **Description**         | Create a difficulty ladder comparison matrix showing neural network performance at different difficulty levels compared to heuristic AI. This informs the production AI ladder configuration.                                           |
| **Acceptance Criteria** | <ul><li>Comparison matrix for difficulty levels 1-10</li><li>Recommended difficulty mappings for neural AI</li><li>Decision: which levels should use neural vs heuristic</li><li>Documentation of performance characteristics</li></ul> |
| **Key Files**           | <ul><li>`ai-service/scripts/evaluate_ai_models.py`</li><li>`docs/ai/AI_DIFFICULTY_ANALYSIS.md`</li></ul>                                                                                                                                |
| **Dependencies**        | AI-07                                                                                                                                                                                                                                   |
| **Recommended Mode**    | debug                                                                                                                                                                                                                                   |

---

### Phase 4: Production Integration

#### AI-09: Wire Minimax/MCTS into Production Ladder

| Attribute               | Value                                                                                                                                                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Task ID**             | AI-09                                                                                                                                                                                                                                            |
| **Title**               | Wire Minimax/MCTS into production difficulty ladder                                                                                                                                                                                              |
| **Description**         | Update the production AI difficulty ladder to include MinimaxAI and MCTSAI at appropriate difficulty levels. Ensure the backend correctly routes AI requests to the Python AI service for these types. Verify with integration tests.            |
| **Acceptance Criteria** | <ul><li>Difficulty levels 3-6 use MinimaxAI</li><li>Difficulty levels 7-8 use MCTSAI</li><li>Backend routes requests to Python AI service</li><li>Integration tests pass for all difficulty levels</li><li>Response latency meets SLOs</li></ul> |
| **Key Files**           | <ul><li>`src/server/game/ai/AIEngine.ts`</li><li>`ai-service/app/main.py`</li><li>`tests/integration/ai-difficulty-ladder.test.ts`</li></ul>                                                                                                     |
| **Dependencies**        | AI-07                                                                                                                                                                                                                                            |
| **Recommended Mode**    | code                                                                                                                                                                                                                                             |

#### AI-10: Add AI Fallback and Latency Monitoring

| Attribute               | Value                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-10                                                                                                                                                                                                                                                                                                                           |
| **Title**               | Add AI fallback and latency monitoring                                                                                                                                                                                                                                                                                          |
| **Description**         | Implement robust fallback behavior when the Python AI service is unavailable or slow. Add Prometheus metrics for AI response latency, fallback rate, and error rate. Configure alerts for SLO breaches.                                                                                                                         |
| **Acceptance Criteria** | <ul><li>Fallback to heuristic AI when Python service unavailable</li><li>Fallback to heuristic AI when response exceeds timeout</li><li>Prometheus metrics: `ai_response_latency_seconds`, `ai_fallback_total`, `ai_errors_total`</li><li>Grafana dashboard panel for AI metrics</li><li>Alert for fallback rate > 5%</li></ul> |
| **Key Files**           | <ul><li>`src/server/game/ai/AIEngine.ts`</li><li>`src/server/services/AIServiceClient.ts`</li><li>`prometheus/alerts.yml`</li></ul>                                                                                                                                                                                             |
| **Dependencies**        | AI-09                                                                                                                                                                                                                                                                                                                           |
| **Recommended Mode**    | code                                                                                                                                                                                                                                                                                                                            |

#### AI-11: Document AI Strength Progression

| Attribute               | Value                                                                                                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-11                                                                                                                                                                                                                                       |
| **Title**               | Document AI strength progression                                                                                                                                                                                                            |
| **Description**         | Create user-facing and developer documentation for the AI difficulty ladder, including expected strength at each level, AI type used, and what players can expect. Update the calibration guide with neural network results.                |
| **Acceptance Criteria** | <ul><li>AI difficulty ladder documented in `docs/ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`</li><li>User-facing difficulty descriptions</li><li>Developer reference for AI type routing</li><li>Calibration methodology documented</li></ul> |
| **Key Files**           | <ul><li>`docs/ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`</li><li>`docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md`</li></ul>                                                                                                                           |
| **Dependencies**        | AI-09, AI-10                                                                                                                                                                                                                                |
| **Recommended Mode**    | architect                                                                                                                                                                                                                                   |

---

## 4. Dependency Diagram

```
                    ┌──────────────────────────────────┐
                    │     Phase 1: Schema & DB Fix      │
                    └──────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               │
              ┌─────────┐                           │
              │  AI-01  │                           │
              │Diagnose │                           │
              │ Schema  │                           │
              └─────────┘                           │
                    │                               │
                    ▼                               │
              ┌─────────┐                           │
              │  AI-02  │                           │
              │  Regen  │                           │
              │   DBs   │                           │
              └─────────┘                           │
                    │                               │
                    ▼                               │
              ┌─────────┐                           │
              │  AI-03  │                           │
              │ Parity  │                           │
              │  Gate   │                           │
              └─────────┘                           │
                    │                               │
                    ▼                               │
      ┌──────────────────────────────────┐          │
      │     Phase 2: Data Scaling         │          │
      └──────────────────────────────────┘          │
                    │                               │
        ┌───────────┴───────────┐                   │
        │                       │                   │
        ▼                       ▼                   │
  ┌─────────┐             ┌─────────┐               │
  │  AI-04  │────────────▶│  AI-05  │               │
  │ Scale   │             │Registry │               │
  │Self-Play│             │ Update  │               │
  └─────────┘             └─────────┘               │
        │                                           │
        ▼                                           │
┌──────────────────────────────────┐                │
│  Phase 3: Neural Network Training │                │
└──────────────────────────────────┘                │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-06  │                                       │
  │Extended │                                       │
  │Training │                                       │
  └─────────┘                                       │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-07  │                                       │
  │Evaluate │                                       │
  │  Model  │                                       │
  └─────────┘                                       │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-08  │                                       │
  │ Compare │                                       │
  │ Levels  │                                       │
  └─────────┘                                       │
        │                                           │
        ▼                                           │
┌──────────────────────────────────┐                │
│  Phase 4: Production Integration  │                │
└──────────────────────────────────┘                │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-09  │                                       │
  │  Wire   │                                       │
  │ Ladder  │                                       │
  └─────────┘                                       │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-10  │                                       │
  │Fallback │                                       │
  │Monitor  │                                       │
  └─────────┘                                       │
        │                                           │
        ▼                                           │
  ┌─────────┐                                       │
  │  AI-11  │                                       │
  │Document │                                       │
  └─────────┘                                       │
```

**Dependency Summary:**

| Task  | Depends On   |
| ----- | ------------ |
| AI-01 | None         |
| AI-02 | AI-01        |
| AI-03 | AI-02        |
| AI-04 | AI-03        |
| AI-05 | AI-04        |
| AI-06 | AI-04        |
| AI-07 | AI-06        |
| AI-08 | AI-07        |
| AI-09 | AI-07        |
| AI-10 | AI-09        |
| AI-11 | AI-09, AI-10 |

---

## 5. Phase Ordering

### Phase 1: Schema & DB Fix (Critical Path - Highest Priority)

| Order | Task  | Rationale                            |
| ----- | ----- | ------------------------------------ |
| 1     | AI-01 | Must diagnose before fixing          |
| 2     | AI-02 | Unblocks parity gate                 |
| 3     | AI-03 | Validates fix, enables data pipeline |

### Phase 2: Data Scaling

| Order | Task  | Rationale              |
| ----- | ----- | ---------------------- |
| 4     | AI-04 | Produces training data |
| 5     | AI-05 | Documents provenance   |

### Phase 3: Neural Network Training

| Order | Task  | Rationale                        |
| ----- | ----- | -------------------------------- |
| 6     | AI-06 | Extended training on scaled data |
| 7     | AI-07 | Validates training success       |
| 8     | AI-08 | Informs production config        |

### Phase 4: Production Integration

| Order | Task  | Rationale                             |
| ----- | ----- | ------------------------------------- |
| 9     | AI-09 | Enables search-based AI in production |
| 10    | AI-10 | Ensures resilience                    |
| 11    | AI-11 | Completes documentation               |

---

## 6. Risk Assessment

### High Risk

| Risk                                                               | Probability | Impact | Mitigation                                                      |
| ------------------------------------------------------------------ | ----------- | ------ | --------------------------------------------------------------- |
| Large-board DBs cannot be regenerated due to infrastructure issues | Medium      | High   | Start with small batch (32 games), verify schema before scaling |
| Neural network does not improve with more data                     | Low         | High   | Monitor loss curves, try multiple architectures/hyperparameters |
| Python AI service latency exceeds SLOs                             | Medium      | Medium | Implement timeout and fallback before production integration    |

### Medium Risk

| Risk                                     | Probability | Impact | Mitigation                                                |
| ---------------------------------------- | ----------- | ------ | --------------------------------------------------------- |
| Distributed self-play fails on SSH hosts | Medium      | Medium | Test single-host first, have fallback to local generation |
| Parity gate finds new divergences        | Low         | Medium | Fix divergences before scaling; budget time for debugging |
| Training takes longer than expected      | Medium      | Low    | Use early stopping, checkpoints; can incrementally deploy |

### Low Risk

| Risk                            | Probability | Impact | Mitigation                                      |
| ------------------------------- | ----------- | ------ | ----------------------------------------------- |
| Documentation falls out of sync | Low         | Low    | Update registry immediately after each phase    |
| Heuristic weights regress       | Low         | Low    | Run tier gate before changing production config |

---

## 7. Success Criteria

### Phase 1 Exit Criteria

- [ ] AI-01: Parity/phase invariant issues diagnosed and documented
- [ ] AI-02: Both large-board DBs regenerated with schema-complete tables and no phase invariant violations
- [ ] AI-03: Both DBs pass parity gate with `canonical_ok: true`

### Phase 2 Exit Criteria

- [ ] AI-04: ≥500 canonical games per board type
- [ ] AI-05: TRAINING_DATA_REGISTRY.md updated with all DBs and gate summaries

### Phase 3 Exit Criteria

- [ ] AI-06: Neural network trained for 50+ epochs
- [ ] AI-07: Neural network achieves ≥85% win rate vs random
- [ ] AI-08: Difficulty ladder comparison documented

### Phase 4 Exit Criteria

- [ ] AI-09: Minimax/MCTS wired into production at levels 3-8
- [ ] AI-10: Fallback and monitoring in place with alerts
- [ ] AI-11: AI strength progression documented

### Overall Success Definition

The AI Training Pipeline Remediation is complete when:

1. ✅ All canonical DBs (square8, square19, hexagonal) pass parity gates
2. ✅ ≥500 canonical games exist per board type
3. ✅ Neural network achieves ≥85% win rate vs random
4. ✅ Production difficulty ladder includes search-based AI (Minimax/MCTS)
5. ✅ Fallback and monitoring are operational
6. ✅ All documentation is updated

---

## Revision History

| Version | Date       | Changes                                                                              |
| ------- | ---------- | ------------------------------------------------------------------------------------ |
| 1.0     | 2025-12-20 | Initial remediation plan created                                                     |
| 1.1     | 2025-12-20 | AI-02 (hexagonal): Schema regenerated, parity blocker identified                     |
| 1.2     | 2025-12-20 | Updated large-board status: schema complete, parity failures due to phase invariants |
| 1.3     | 2025-12-20 | Aligned Python territory eligibility with canonical elimination rules (height-1 ok)  |

---

## Task Execution Log

### AI-02: Align territory eligibility to canonical elimination rules (2025-12-20)

**Status:** ✅ CODE FIXED (awaiting regeneration + gate re-run)

**Context:** Parity failures showed `forced_elimination` moves recorded while still in `territory_processing`. A likely contributor was Python rejecting height-1 outside stacks as valid territory self-elimination targets (RR-CANON-R145), which would cause `no_territory_action` to be recorded instead of processing a region.

**Actions Completed:**

1. ✅ Updated `GameEngine._can_process_disconnected_region` to delegate to `is_stack_eligible_for_elimination` for both recovery and territory contexts.
2. ✅ Removed singleton-stack height gating and aligned eligibility with canonical rules (height-1 controlled stacks are valid).

**Files Modified:**

- `ai-service/app/_game_engine_legacy.py`

### AI-02: Regenerate canonical_hexagonal.db (2025-12-20)

**Status:** ⚠️ PARTIAL SUCCESS - Schema fixed, parity bug blocking further generation

**Actions Completed:**

1. ✅ Removed 0-byte placeholder file at `ai-service/data/canonical_hexagonal.db`
2. ✅ Archived malformed DB to `ai-service/data/games/canonical_hexagonal.db.malformed.bak`
3. ✅ Ran `generate_canonical_selfplay.py --board-type hexagonal --num-games 50`
4. ✅ New DB created with correct schema v9 (all 9 tables present)
5. ⚠️ Only 1 game recorded due to parity divergence on first game

**Schema Verification:**

```
$ sqlite3 ai-service/data/games/canonical_hexagonal.db ".tables"
game_choices          game_moves            game_state_snapshots
game_history_entries  game_nnue_features    games
game_initial_state    game_players          schema_metadata
```

**Data Verification:**

- Games: 1
- Moves: 1,104 (in `game_moves` table)
- Health summary: `ai-service/data/games/db_health.canonical_hexagonal.json`

**Blocking Issue Identified:**
The parity gate fails with a **TS↔Python phase/move invariant violation**:

```
Phase/move invariant violated: cannot apply move type no_placement_action in phase territory_processing
```

**Root Cause:** Python's GameEngine emits `no_placement_action` moves during `territory_processing` phase. The TS engine rejects this as invalid - `no_placement_action` is only allowed in `ring_placement` phase.

**Relevant Error (from health summary):**

```json
{
  "canonical_history": {
    "games_checked": 1,
    "non_canonical_games": 1,
    "sample_issues": {
      "dff66903-cc1a-4807-8df4-1ecc3ae67d1e": [
        {
          "move_number": 865,
          "move_type": "no_placement_action",
          "phase_before": "territory_processing",
          "reason": "RuntimeError: Phase/move invariant violated: cannot apply move type no_placement_action in phase territory_processing"
        }
      ]
    }
  }
}
```

**Next Steps (AI-03 Blocker):**
The hexagonal parity gate is blocked by a cross-language phase/move invariant bug. Before AI-03 can proceed for hexagonal:

1. Investigate Python GameEngine phase transition logic for hexagonal boards
2. Fix the `no_placement_action` emission in `territory_processing` phase
3. Re-run AI-02 for hexagonal after fix

**Files Modified/Created:**

- `ai-service/data/games/canonical_hexagonal.db.malformed.bak` (archived)
- `ai-service/data/games/canonical_hexagonal.db` (new, with correct schema)
- `ai-service/data/games/db_health.canonical_hexagonal.json` (health summary)
- `ai-service/data/games/canonical_hexagonal.db.parity_gate.json` (parity gate output)
- `ai-service/data/games/canonical_hexagonal.db.parity_summary.json` (parity summary)

### AI-02: Regenerate canonical_square19.db (2025-12-20)

**Status:** ⚠️ PARTIAL SUCCESS - Schema complete, parity bug blocking further generation

**Actions Recorded (db_health.canonical_square19.json):**

1. ✅ Archived prior DB to `ai-service/data/games/canonical_square19.db.archived_20251220_052509`
2. ✅ Ran `generate_canonical_selfplay.py --board-type square19 --num-games 200`
3. ✅ New DB created with schema v9 (all 9 tables present)
4. ⚠️ Only 1 game recorded due to parity divergence on first game

**Data Verification:**

- Games: 1
- Moves: 733 (in `game_moves` table)
- Health summary: `ai-service/data/games/db_health.canonical_square19.json`

**Blocking Issue Identified:**
The parity gate fails with a **TS↔Python phase/move invariant violation**:

```
[PHASE_MOVE_INVARIANT] Cannot apply move type 'forced_elimination' in phase 'territory_processing'
```

**Root Cause:** Self-play/engine emitted a `forced_elimination` move without transitioning into the `forced_elimination` phase.

**Next Steps (AI-03 Blocker):**
Before AI-03 can proceed for square19:

1. Fix forced-elimination phase transition in self-play generation
2. Re-run AI-02 for square19 after fix
