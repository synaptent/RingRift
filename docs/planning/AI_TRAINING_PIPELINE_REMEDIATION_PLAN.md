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

| Issue                      | Current State                                  | Target State                       |
| -------------------------- | ---------------------------------------------- | ---------------------------------- |
| Schema completeness        | Missing `game_moves` tables in large-board DBs | All DBs gateable with full schema  |
| Training data volume       | ~100 games total                               | 10,000+ games per board type       |
| Neural network performance | 75% win rate vs random                         | ≥90% win rate (matching heuristic) |
| Parity gating              | Blocked for large boards                       | All DBs pass TS↔Python parity      |

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

| Database                  | Board Type | Games  | `game_moves` Table | Parity Gate | Status           |
| ------------------------- | ---------- | ------ | ------------------ | ----------- | ---------------- |
| `canonical_square8_2p.db` | square8    | 200    | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square8_3p.db` | square8    | 2      | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square8_4p.db` | square8    | 2      | ✅ Present         | ✅ PASS     | **canonical**    |
| `canonical_square19.db`   | square19   | 16,081 | ❌ Missing         | ❌ BLOCKED  | **pending_gate** |
| `canonical_hexagonal.db`  | hexagonal  | 25,203 | ❌ Missing         | ❌ BLOCKED  | **pending_gate** |

**Root Cause:** The large-board DBs (square19, hexagonal) were generated before the `game_moves` table was required for parity gating. The schema version is current (v9), but the table was not populated during generation.

### 1.2 Training Data Volume

| Board Type   | Current     | Target (Baseline) | Target (Training) | Gap          |
| ------------ | ----------- | ----------------- | ----------------- | ------------ |
| square8 (2p) | 200         | ≥200              | ≥1,000            | 800 games    |
| square8 (3p) | 2           | ≥32               | ≥500              | 498 games    |
| square8 (4p) | 2           | ≥32               | ≥500              | 498 games    |
| square19     | 0 (blocked) | ≥200              | ≥1,000            | 1,000+ games |
| hexagonal    | 0 (blocked) | ≥200              | ≥1,000            | 1,000+ games |

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
            │  ✅ PASS    │ │  ❌ BLOCKED │ │  ❌ BLOCKED │
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
                                    X (blocked for large boards)
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

### 2.1 Missing `game_moves` Table

**Background:** The parity gate (`check_ts_python_replay_parity.py`) requires the `game_moves` table to replay games move-by-move and compare TS vs Python engine states. Without this table, the gate cannot execute.

**Impact:**

- Cannot validate canonical_square19.db (16,081 games)
- Cannot validate canonical_hexagonal.db (25,203 games)
- These datasets are unusable for training until regenerated

**Root Cause Analysis:**

1. Large-board self-play was run using an older generator configuration
2. The `game_moves` table requires `GameWriter.record_move()` to be called
3. The generator may have used snapshot-only mode or an older schema

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

#### AI-01: Diagnose Canonical DB Schema Issues

| Attribute               | Value                                                                                                                                                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-01                                                                                                                                                                                                                                                |
| **Title**               | Diagnose canonical DB schema issues                                                                                                                                                                                                                  |
| **Description**         | Inspect canonical_square19.db and canonical_hexagonal.db to identify exactly why the `game_moves` table is missing. Determine if the DBs can be patched or must be regenerated. Check schema version, table structures, and generator configuration. |
| **Acceptance Criteria** | <ul><li>Schema version confirmed for each DB</li><li>Missing tables documented</li><li>Generator configuration that created the DBs identified</li><li>Decision made: patch vs regenerate</li></ul>                                                  |
| **Key Files**           | <ul><li>`ai-service/app/db/game_replay.py`</li><li>`ai-service/data/games/canonical_square19.db`</li><li>`ai-service/data/games/canonical_hexagonal.db`</li></ul>                                                                                    |
| **Diagnostic Commands** | `bash<br>cd ai-service<br>sqlite3 data/games/canonical_square19.db ".schema"<br>sqlite3 data/games/canonical_square19.db "SELECT name FROM sqlite_master WHERE type='table'"<br>`                                                                    |
| **Dependencies**        | None                                                                                                                                                                                                                                                 |
| **Recommended Mode**    | debug                                                                                                                                                                                                                                                |

#### AI-02: Fix Schema and Regenerate Canonical DBs

| Attribute               | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task ID**             | AI-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Title**               | Fix schema and regenerate canonical DBs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Description**         | Regenerate canonical_square19.db and canonical_hexagonal.db using the canonical self-play generator with proper `game_moves` table population. Archive existing DBs before regeneration. Start with a small number of games (32-64) to verify the schema is correct.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Acceptance Criteria** | <ul><li>Existing DBs archived with timestamp</li><li>New DBs created with `generate_canonical_selfplay.py`</li><li>`game_moves` table present in new DBs</li><li>At least 32 games per board type generated</li><li>Schema v9 verified</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
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

- [ ] AI-01: Schema issues diagnosed and documented
- [ ] AI-02: Both large-board DBs regenerated with `game_moves` table
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

| Version | Date       | Changes                          |
| ------- | ---------- | -------------------------------- |
| 1.0     | 2025-12-20 | Initial remediation plan created |
