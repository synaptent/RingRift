# Training Data Volume Remediation Plan (Phase 3)

> **Doc Status (2025-12-20): Active**
>
> **Purpose:** Scale canonical training datasets to target volumes for neural network training.
>
> **Owner:** TBD  
> **Scope:** AI self-play generation, canonical database scaling, neural network training
>
> **References:**
>
> - [`docs/planning/AI_TRAINING_PIPELINE_REMEDIATION_PLAN.md`](./AI_TRAINING_PIPELINE_REMEDIATION_PLAN.md) - Parent remediation plan
> - [`ai-service/TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md) - Data classification and provenance
> - [`ai-service/scripts/generate_canonical_selfplay.py`](../../ai-service/scripts/generate_canonical_selfplay.py) - Self-play generator

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Current State Analysis](#1-current-state-analysis)
- [2. Remediation Subtasks](#2-remediation-subtasks)
- [3. Acceptance Criteria](#3-acceptance-criteria)
- [4. Risk Mitigation](#4-risk-mitigation)
- [5. Commands Reference](#5-commands-reference)
- [6. Dependency Diagram](#6-dependency-diagram)
- [Revision History](#revision-history)

---

## Executive Summary

### Objective

Scale canonical training datasets from current volumes (~210 games total) to target volumes (4,000+ games total) to enable effective neural network training across all board types.

### Timeline

- **Week 1:** Complete square8 scaling (all player counts) and begin large-board generation
- **Week 2:** Complete square19/hexagonal scaling, run neural network training, evaluate results

### Dependencies (COMPLETE)

| Task ID | Description                                              | Status                               |
| ------- | -------------------------------------------------------- | ------------------------------------ |
| AI-01   | Diagnose parity/phase invariant failures                 | ✅ COMPLETE                          |
| AI-02   | Fix phase invariants and regenerate DBs                  | ✅ COMPLETE                          |
| AI-03   | Run canonical gate (parity + history) on regenerated DBs | ✅ COMPLETE                          |
| AI-04   | Diagnose LPS victory detection parity                    | ✅ COMPLETE                          |
| AI-04b  | Fix TS LPS victory detection timing                      | ✅ COMPLETE                          |
| AI-04c  | Validate parity fixes                                    | ✅ COMPLETE                          |
| AI-06   | Free disk space for scale-up                             | ✅ COMPLETE (359GB available)        |
| AI-07   | Initial scale-up attempt                                 | ⏳ Partial (sporadic edge-case bugs) |

### Key Constraint

The canonical gate pipeline (parity + canonical history + FE/territory fixtures + ANM invariants)
has been stable with **0 semantic divergences** on recent runs. Throughput remains the primary constraint:

| Board Type | Time per Game  | Games per Hour |
| ---------- | -------------- | -------------- |
| square8    | ~10-15 seconds | ~240-360       |
| square19   | ~45-55 seconds | ~65-80         |
| hexagonal  | ~2+ minutes    | ~25-30         |

---

## 1. Current State Analysis

### 1.1 Database Volumes

| Database                  | Board Type | Current Games | Target | Gap   | Gate Status |
| ------------------------- | ---------- | ------------- | ------ | ----- | ----------- |
| `canonical_square8_2p.db` | square8    | 200           | 1,000  | 80%   | ✅ PASS     |
| `canonical_square8_3p.db` | square8    | 2             | 500    | 99.6% | ✅ PASS     |
| `canonical_square8_4p.db` | square8    | 2             | 500    | 99.6% | ✅ PASS     |
| `canonical_square19.db`   | square19   | 26            | 1,000  | 97.4% | ✅ PASS     |
| `canonical_hexagonal.db`  | hexagonal  | 14            | 1,000  | 98.6% | ✅ PASS     |

**Total Current:** ~244 games  
**Total Target:** ~4,000 games  
**Overall Gap:** ~94%

### 1.2 Throughput Constraints

Based on AI-07 scale-up attempts, observed throughput rates are:

```
┌─────────────────────────────────────────────────────────────┐
│         Self-Play Throughput by Board Type                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  square8   ████████████████████████████████████  ~15 sec/game│
│                                                              │
│  square19  ████████████████████████████████████  ~50 sec/game│
│            ████████████████████████                          │
│                                                              │
│  hexagonal ████████████████████████████████████  ~120 sec/gm │
│            ████████████████████████████████████              │
│            ████████████████████████████████████              │
│            ████████████████                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Edge-Case Bug Rates

From AI-07 observations, approximately **5% of generated games** encounter edge-case bugs:

| Bug Type                 | Affected Board | Description                                    |
| ------------------------ | -------------- | ---------------------------------------------- |
| `recovery_slide` invalid | hexagonal      | "No buried ring at position" errors            |
| `state_hash` divergence  | square19       | Line processing parity edge cases              |
| AI out-of-bounds         | hexagonal      | Invalid placement positions (e.g., `(0, -12)`) |

**Workaround:** Games with bugs are deleted after generation; remaining games pass parity + canonical history validation.

### 1.4 Infrastructure Status

| Resource      | Current State         | Constraint                               |
| ------------- | --------------------- | ---------------------------------------- |
| Disk Space    | 359GB available       | ✅ No constraint                         |
| Memory        | Adequate              | ✅ No constraint                         |
| CPU           | Single-host limited   | ⚠️ Multi-host preferred for large boards |
| Cluster Nodes | Lambda/VAST available | ✅ Can distribute                        |

---

## 2. Remediation Subtasks

### Phase P0: Immediate (Local Single-Host)

These tasks can run on the development machine using `nohup` for long-running processes.

Note: `canonical_ok` in `db_health.*.json` already gates parity, canonical history, FE/territory fixtures,
and ANM invariants. Treat it as the go/no-go signal for promoting a DB. For
square19/hex runs, `generate_canonical_selfplay.py` defaults to
`RINGRIFT_USE_MAKE_UNMAKE=true` and `RINGRIFT_USE_FAST_TERRITORY=false` unless
overridden, so explicit env flags are optional.

#### VOL-01: Square8 2-Player Scale-Up

| Attribute             | Value                                        |
| --------------------- | -------------------------------------------- |
| **Task ID**           | VOL-01                                       |
| **Title**             | Scale canonical_square8_2p.db to 1,000 games |
| **Current**           | 200 games                                    |
| **Target**            | 1,000 games                                  |
| **Games to Generate** | 800                                          |
| **Estimated Time**    | ~4 hours                                     |
| **Depends On**        | None                                         |

**Command:**

```bash
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 2 \
    --num-games 100 \
    --min-recorded-games 1000 \
    --max-soak-attempts 10 \
    --difficulty-band light \
    --db data/games/canonical_square8_2p.db \
    --summary data/games/db_health.canonical_square8_2p.json' \
  > logs/vol01_square8_2p.log 2>&1 &
```

**Success Metric:**

- `canonical_ok: true` in summary JSON
- `games_total >= 1000` in db_stats
- `games_with_semantic_divergence: 0`

---

#### VOL-02: Square8 3-Player Scale-Up

| Attribute             | Value                                      |
| --------------------- | ------------------------------------------ |
| **Task ID**           | VOL-02                                     |
| **Title**             | Scale canonical_square8_3p.db to 500 games |
| **Current**           | 2 games                                    |
| **Target**            | 500 games                                  |
| **Games to Generate** | 498                                        |
| **Estimated Time**    | ~6 hours                                   |
| **Depends On**        | None (can run in parallel with VOL-01)     |

**Command:**

```bash
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 3 \
    --num-games 100 \
    --min-recorded-games 500 \
    --max-soak-attempts 6 \
    --difficulty-band light \
    --db data/games/canonical_square8_3p.db \
    --summary data/games/db_health.canonical_square8_3p.json' \
  > logs/vol02_square8_3p.log 2>&1 &
```

**Success Metric:**

- `canonical_ok: true` in summary JSON
- `games_total >= 500` in db_stats

---

#### VOL-03: Square8 4-Player Scale-Up

| Attribute             | Value                                      |
| --------------------- | ------------------------------------------ |
| **Task ID**           | VOL-03                                     |
| **Title**             | Scale canonical_square8_4p.db to 500 games |
| **Current**           | 2 games                                    |
| **Target**            | 500 games                                  |
| **Games to Generate** | 498                                        |
| **Estimated Time**    | ~8 hours                                   |
| **Depends On**        | None (can run in parallel)                 |

**Command:**

```bash
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  python scripts/generate_canonical_selfplay.py \
    --board square8 \
    --num-players 4 \
    --num-games 100 \
    --min-recorded-games 500 \
    --max-soak-attempts 6 \
    --difficulty-band light \
    --db data/games/canonical_square8_4p.db \
    --summary data/games/db_health.canonical_square8_4p.json' \
  > logs/vol03_square8_4p.log 2>&1 &
```

**Success Metric:**

- `canonical_ok: true` in summary JSON
- `games_total >= 500` in db_stats

---

#### VOL-04: Square19 Scale-Up

| Attribute             | Value                                      |
| --------------------- | ------------------------------------------ |
| **Task ID**           | VOL-04                                     |
| **Title**             | Scale canonical_square19.db to 1,000 games |
| **Current**           | 26 games                                   |
| **Target**            | 1,000 games                                |
| **Games to Generate** | 974                                        |
| **Estimated Time**    | ~15 hours (single host)                    |
| **Depends On**        | None                                       |

**Command:**

```bash
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  RINGRIFT_USE_MAKE_UNMAKE=true \
  python scripts/generate_canonical_selfplay.py \
    --board square19 \
    --num-players 2 \
    --num-games 100 \
    --min-recorded-games 1000 \
    --max-soak-attempts 15 \
    --difficulty-band light \
    --db data/games/canonical_square19.db \
    --summary data/games/db_health.canonical_square19.json' \
  > logs/vol04_square19.log 2>&1 &
```

**Success Metric:**

- `canonical_ok: true` in summary JSON
- `games_total >= 1000` in db_stats

---

#### VOL-05: Hexagonal Scale-Up

| Attribute             | Value                                       |
| --------------------- | ------------------------------------------- |
| **Task ID**           | VOL-05                                      |
| **Title**             | Scale canonical_hexagonal.db to 1,000 games |
| **Current**           | 14 games                                    |
| **Target**            | 1,000 games                                 |
| **Games to Generate** | 986                                         |
| **Estimated Time**    | ~35 hours (single host)                     |
| **Depends On**        | None                                        |

**Command:**

```bash
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 \
  RINGRIFT_USE_MAKE_UNMAKE=true \
  python scripts/generate_canonical_selfplay.py \
    --board hexagonal \
    --num-players 2 \
    --num-games 50 \
    --min-recorded-games 1000 \
    --max-soak-attempts 25 \
    --difficulty-band light \
    --db data/games/canonical_hexagonal.db \
    --summary data/games/db_health.canonical_hexagonal.json' \
  > logs/vol05_hexagonal.log 2>&1 &
```

**Success Metric:**

- `canonical_ok: true` in summary JSON
- `games_total >= 1000` in db_stats

---

### Phase P1: Short-Term (Distributed/Cluster)

These tasks leverage cluster nodes for faster throughput.

#### VOL-06: Distributed Self-Play on Cluster

| Attribute       | Value                                                 |
| --------------- | ----------------------------------------------------- |
| **Task ID**     | VOL-06                                                |
| **Title**       | Run distributed self-play on Lambda/VAST cluster      |
| **Description** | Use SSH-based distributed generation for large boards |
| **Depends On**  | VOL-04, VOL-05 (if still below targets)               |

**Command (with cluster hosts):**

```bash
cd ai-service && PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
  --board square19 \
  --num-players 2 \
  --num-games 200 \
  --min-recorded-games 1000 \
  --max-soak-attempts 5 \
  --difficulty-band light \
  --db data/games/canonical_square19.db \
  --summary data/games/db_health.canonical_square19.json \
  --hosts lambda1,lambda2,lambda3,vast1
```

**Success Metric:**

- 3-5x throughput improvement
- All distributed DBs merged and pass canonical gate

---

#### VOL-07: Clean Up Edge-Case Games

| Attribute       | Value                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| **Task ID**     | VOL-07                                                                 |
| **Title**       | Remove games with edge-case bugs from canonical DBs                    |
| **Description** | Delete games that failed with recovery_slide or state_hash divergences |
| **Depends On**  | VOL-04, VOL-05                                                         |

**Verification Command:**

```bash
cd ai-service && PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hexagonal.db \
  --mode canonical \
  --view post_move \
  --progress-every 50 \
  --compact
```

**Cleanup Command (for individual games):**

```bash
cd ai-service && sqlite3 data/games/canonical_hexagonal.db \
  "DELETE FROM games WHERE game_id = '<problematic_game_id>';"
```

**Success Metric:**

- `games_with_semantic_divergence: 0`
- `games_with_structural_issues: 0`
- `canonical_history.non_canonical_games: 0`
- `canonical_ok: true` for remaining games

---

#### VOL-08: Update TRAINING_DATA_REGISTRY.md

| Attribute      | Value                                                 |
| -------------- | ----------------------------------------------------- |
| **Task ID**    | VOL-08                                                |
| **Title**      | Update registry with final volumes and gate summaries |
| **Depends On** | VOL-01 through VOL-07                                 |

**Actions:**

1. Update game counts in the Canonical database table
2. Link to updated gate summary JSONs
3. Update Volume Targets section with achieved counts
4. Document any edge-case workarounds in Gate Notes

**Success Metric:**

- All canonical DBs listed with accurate counts
- All gate summaries referenced

---

#### VOL-09: Run Neural Network Training

| Attribute       | Value                                                |
| --------------- | ---------------------------------------------------- |
| **Task ID**     | VOL-09                                               |
| **Title**       | Train neural network on scaled datasets              |
| **Description** | Run extended training (50+ epochs) on canonical data |
| **Depends On**  | VOL-08                                               |

**Export Training Data:**

```bash
cd ai-service && PYTHONPATH=. python scripts/export_replay_dataset_parallel.py \
  --input-db data/games/canonical_square8_2p.db \
  --output data/training/canonical_square8_2p.npz \
  --board-type square8 \
  --num-players 2
```

**Training Command:**

```bash
cd ai-service && PYTHONPATH=. python -m app.training.train \
  --epochs 50 \
  --batch-size 128 \
  --checkpoint-interval 10 \
  --data data/training/canonical_square8_2p.npz \
  --output checkpoints/ringrift_v2_square8.pth
```

**Success Metric:**

- Training loss < 0.5 (from ~0.7 at epoch 5)
- Checkpoints saved at epochs 10, 20, 30, 40, 50

---

### Phase P2: Medium-Term

#### VOL-10: Evaluate Neural Network Improvement

| Attribute      | Value                                             |
| -------------- | ------------------------------------------------- |
| **Task ID**    | VOL-10                                            |
| **Title**      | Evaluate trained neural network against baselines |
| **Depends On** | VOL-09                                            |

**Evaluation Command:**

```bash
cd ai-service && PYTHONPATH=. python scripts/evaluate_ai_models.py \
  --player1 neural \
  --player2 random \
  --games 50 \
  --board square8 \
  --output results/neural_vs_random_post_scaling.json
```

**Success Metric:**

- Neural vs random: ≥85% win rate
- Neural vs heuristic: ≥40% win rate (competitive)

---

#### VOL-11: Wire Neural Network into Production Ladder

| Attribute      | Value                                                      |
| -------------- | ---------------------------------------------------------- |
| **Task ID**    | VOL-11                                                     |
| **Title**      | Integrate neural network into production difficulty ladder |
| **Depends On** | VOL-10                                                     |

**Files to Modify:**

- [`src/server/game/ai/AIEngine.ts`](../../src/server/game/ai/AIEngine.ts)
- [`ai-service/app/main.py`](../../ai-service/app/main.py)

**Success Metric:**

- Difficulty levels 9-10 use neural-guided DescentAI
- Integration tests pass

---

#### VOL-12: Document AI Strength Progression

| Attribute      | Value                                         |
| -------------- | --------------------------------------------- |
| **Task ID**    | VOL-12                                        |
| **Title**      | Document final AI performance characteristics |
| **Depends On** | VOL-11                                        |

**Deliverables:**

- Update [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](../ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md)
- Create difficulty ladder progression chart
- Document training data requirements for future upgrades

**Success Metric:**

- Complete documentation of AI strength at each level
- Calibration methodology documented

---

## 3. Acceptance Criteria

### Overall Success Definition

| Criterion                                 | Target | Currently |
| ----------------------------------------- | ------ | --------- |
| square8_2p games                          | ≥1,000 | 200       |
| square8_3p games                          | ≥500   | 2         |
| square8_4p games                          | ≥500   | 2         |
| square19 games                            | ≥1,000 | 26        |
| hexagonal games                           | ≥1,000 | 14        |
| Total canonical games                     | ≥4,000 | ~244      |
| Canonical gate pass rate (`canonical_ok`) | 100%   | 100%      |
| Neural network win rate vs random         | ≥85%   | 75%       |

### Per-Task Verification

| Task   | Verification Command                                                                                                                                                      | Expected Output                                   |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| VOL-01 | `sqlite3 data/games/canonical_square8_2p.db "SELECT COUNT(*) FROM games"`                                                                                                 | ≥1000                                             |
| VOL-02 | `sqlite3 data/games/canonical_square8_3p.db "SELECT COUNT(*) FROM games"`                                                                                                 | ≥500                                              |
| VOL-03 | `sqlite3 data/games/canonical_square8_4p.db "SELECT COUNT(*) FROM games"`                                                                                                 | ≥500                                              |
| VOL-04 | `sqlite3 data/games/canonical_square19.db "SELECT COUNT(*) FROM games"`                                                                                                   | ≥1000                                             |
| VOL-05 | `sqlite3 data/games/canonical_hexagonal.db "SELECT COUNT(*) FROM games"`                                                                                                  | ≥1000                                             |
| VOL-07 | `cat data/games/db_health.*.json \| jq '{canonical_ok, divergences: .parity_gate.games_with_semantic_divergence, non_canonical: .canonical_history.non_canonical_games}'` | canonical_ok true; divergences 0; non_canonical 0 |
| VOL-10 | `cat results/neural_vs_random_post_scaling.json \| jq '.win_rate'`                                                                                                        | ≥0.85                                             |

---

## 4. Risk Mitigation

### Edge-Case Bugs (~5% of games)

**Risk:** Some games fail with recovery_slide or state_hash divergence bugs.

**Mitigation:**

- Use `--max-soak-attempts` to retry until target count reached
- Periodically run parity + canonical history gate and delete problematic games
- Document bug patterns in [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md)
- Track affected game IDs for future investigation

**Workaround Script:**

```bash
# Find and delete divergent games after generation
cd ai-service && PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hexagonal.db \
  --emit-fixtures-dir parity_fixtures \
  --mode canonical

# Review fixtures for problematic game_ids, then:
sqlite3 data/games/canonical_hexagonal.db \
  "DELETE FROM games WHERE game_id IN ('<id1>', '<id2>', ...);"

# Refresh db_health summary after deletions
PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
  --board hexagonal \
  --num-games 0 \
  --db data/games/canonical_hexagonal.db \
  --summary data/games/db_health.canonical_hexagonal.json
```

### Disk Space Exhaustion

**Risk:** Large-board DBs grow quickly; may exceed available space.

**Mitigation:**

- Monitor with `df -h` (currently 359GB available)
- Each game uses ~100KB in square19, ~150KB in hexagonal
- 1,000 games ≈ 100-150MB per board type
- Total requirement: <1GB (well within capacity)

**Monitoring Command:**

```bash
watch -n 60 'df -h /Users/armand/Development/RingRift/ai-service/data/games'
```

### Long-Running Process Failures

**Risk:** Generation processes may crash during overnight runs.

**Mitigation:**

- Use `nohup` for all long-running processes
- Set `--max-soak-attempts` to automatically retry on failure
- Redirect output to log files for debugging
- Use `screen` or `tmux` for persistent sessions

**Process Monitoring:**

```bash
# Check if generation is still running
ps aux | grep generate_canonical_selfplay

# Check progress
tail -f logs/vol04_square19.log

# Check current game count
sqlite3 data/games/canonical_square19.db "SELECT COUNT(*) FROM games"
```

### Distributed Generation Failures

**Risk:** SSH connections to cluster nodes may drop.

**Mitigation:**

- Use `--distributed-job-timeout-seconds` and `--distributed-fetch-timeout-seconds`
- Script automatically merges partial results
- Can fall back to local generation if cluster unavailable

---

## 5. Commands Reference

### Environment Setup

```bash
# Set up Python environment (if not already done)
cd ai-service && ./setup.sh

# Activate virtual environment
source venv/bin/activate

# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Quick Verification Commands

```bash
# Check all DB game counts
for db in canonical_square8_2p canonical_square8_3p canonical_square8_4p canonical_square19 canonical_hexagonal; do
  echo -n "$db: "
  sqlite3 data/games/${db}.db "SELECT COUNT(*) FROM games" 2>/dev/null || echo "N/A"
done

# Check canonical gate status for all DBs
for db in canonical_square8_2p canonical_square8_3p canonical_square8_4p canonical_square19 canonical_hexagonal; do
  echo "=== $db ==="
  if [ -f "data/games/db_health.${db}.json" ]; then
    jq '{canonical_ok, games_total: .db_stats.games_total, divergences: .parity_gate.games_with_semantic_divergence, non_canonical: .canonical_history.non_canonical_games, anm_ok, fe_territory_fixtures_ok}' \
      data/games/db_health.${db}.json
  else
    echo "No health summary found"
  fi
done

# Check disk space
df -h /Users/armand/Development/RingRift/ai-service/data
```

### All VOL Commands (Copy-Paste Ready)

```bash
# VOL-01: Square8 2-player (run in background)
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 python scripts/generate_canonical_selfplay.py --board square8 --num-players 2 --num-games 100 --min-recorded-games 1000 --max-soak-attempts 10 --difficulty-band light --db data/games/canonical_square8_2p.db --summary data/games/db_health.canonical_square8_2p.json' > logs/vol01_square8_2p.log 2>&1 &

# VOL-02: Square8 3-player (run in background)
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 python scripts/generate_canonical_selfplay.py --board square8 --num-players 3 --num-games 100 --min-recorded-games 500 --max-soak-attempts 6 --difficulty-band light --db data/games/canonical_square8_3p.db --summary data/games/db_health.canonical_square8_3p.json' > logs/vol02_square8_3p.log 2>&1 &

# VOL-03: Square8 4-player (run in background)
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 python scripts/generate_canonical_selfplay.py --board square8 --num-players 4 --num-games 100 --min-recorded-games 500 --max-soak-attempts 6 --difficulty-band light --db data/games/canonical_square8_4p.db --summary data/games/db_health.canonical_square8_4p.json' > logs/vol03_square8_4p.log 2>&1 &

# VOL-04: Square19 (run in background)
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 RINGRIFT_USE_MAKE_UNMAKE=true python scripts/generate_canonical_selfplay.py --board square19 --num-players 2 --num-games 100 --min-recorded-games 1000 --max-soak-attempts 15 --difficulty-band light --db data/games/canonical_square19.db --summary data/games/db_health.canonical_square19.json' > logs/vol04_square19.log 2>&1 &

# VOL-05: Hexagonal (run in background)
cd ai-service && nohup bash -c 'PYTHONPATH=. RINGRIFT_SKIP_RESOURCE_GUARD=1 RINGRIFT_USE_MAKE_UNMAKE=true python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 2 --num-games 50 --min-recorded-games 1000 --max-soak-attempts 25 --difficulty-band light --db data/games/canonical_hexagonal.db --summary data/games/db_health.canonical_hexagonal.json' > logs/vol05_hexagonal.log 2>&1 &
```

---

## 6. Dependency Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Phase P0: Immediate                         │
│                   (Local Single-Host)                         │
└──────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
 ┌──────────┐          ┌──────────┐          ┌──────────┐
 │  VOL-01  │          │  VOL-02  │          │  VOL-03  │
 │ square8  │          │ square8  │          │ square8  │
 │   2p     │          │   3p     │          │   4p     │
 │ 800 gms  │          │ 498 gms  │          │ 498 gms  │
 │  ~4 hrs  │          │  ~6 hrs  │          │  ~8 hrs  │
 └──────────┘          └──────────┘          └──────────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
       ┌──────────────────────┴──────────────────────┐
       │                                             │
       ▼                                             ▼
 ┌───────────┐                                ┌───────────┐
 │  VOL-04   │                                │  VOL-05   │
 │ square19  │                                │ hexagonal │
 │  974 gms  │                                │  986 gms  │
 │  ~15 hrs  │                                │  ~35 hrs  │
 └───────────┘                                └───────────┘
       │                                             │
       └──────────────────────┬──────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                   Phase P1: Short-Term                        │
│                   (Distributed/Cluster)                       │
└──────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
 ┌──────────┐          ┌──────────┐          ┌──────────┐
 │  VOL-06  │          │  VOL-07  │          │  VOL-08  │
 │Distribute│─────────▶│ Cleanup  │─────────▶│ Update   │
 │ Selfplay │          │ Edge-Cse │          │ Registry │
 └──────────┘          └──────────┘          └──────────┘
                                                   │
                              ┌─────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   Phase P2: Medium-Term                       │
│                   (Training & Production)                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────┐
                       │  VOL-09  │
                       │  Train   │
                       │   NN     │
                       └──────────┘
                              │
                              ▼
                       ┌──────────┐
                       │  VOL-10  │
                       │ Evaluate │
                       │   NN     │
                       └──────────┘
                              │
                              ▼
                       ┌──────────┐
                       │  VOL-11  │
                       │  Wire    │
                       │ Ladder   │
                       └──────────┘
                              │
                              ▼
                       ┌──────────┐
                       │  VOL-12  │
                       │ Document │
                       └──────────┘
```

---

## Revision History

| Version | Date       | Changes                                         |
| ------- | ---------- | ----------------------------------------------- |
| 1.0     | 2025-12-20 | Initial Phase 3 Volume Remediation Plan created |

---

## Appendix: Parallel Execution Strategy

For maximum efficiency, tasks can be parallelized as follows:

### Day 1 (Start All P0 Tasks)

```bash
# Morning: Start all square8 variants in parallel
# VOL-01, VOL-02, VOL-03 can run simultaneously

# Afternoon: Start large-board generation
# VOL-04, VOL-05 can run simultaneously
```

### Day 2-3 (Monitor and Verify)

```bash
# Monitor progress periodically
for log in logs/vol*.log; do echo "=== $log ==="; tail -5 "$log"; done

# Check game counts
for db in canonical_*.db; do echo "$db: $(sqlite3 $db 'SELECT COUNT(*) FROM games')"; done
```

### Day 4+ (Cleanup and Training)

```bash
# Run canonical gate verification on all DBs
# Execute VOL-07 cleanup if needed
# Proceed to VOL-08, VOL-09
```

**Note:** Large-board generation (VOL-04, VOL-05) will likely extend into Day 3+ if running single-host. Consider using cluster nodes (VOL-06) to accelerate if available.
