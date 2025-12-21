# Stranded Features Inventory

This document catalogs features that are **implemented but not fully utilized or integrated**.
These represent potential Elo gains or code quality improvements that can be activated with minimal additional work.

Last updated: 2025-12-21

## Status Legend

- **Implemented**: Code exists and works
- **Integrated**: Wired into training/inference pipelines
- **Validated**: Tested against baselines with documented results

---

## 1. GUMBEL MCTS OPTIMIZATIONS

### 1.1 GPU Batch Inference

- **Status**: Implemented, Integrated
- **File**: `app/ai/gumbel_mcts_ai.py` (lines 67-72, 205)
- **Control**: `RINGRIFT_GPU_GUMBEL_DISABLE=1` to disable (enabled by default)
- **Validation**: Shadow validation available via `RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1`
- **Expected Impact**: 5-50x speedup for selfplay generation

### 1.2 Soft Policy Targets from Visit Distribution

- **Status**: Implemented, Integrated
- **Files**:
  - `app/ai/gumbel_mcts_ai.py:1252` - `get_visit_distribution()`
  - `app/training/generate_data.py:1328-1369` - Integration for Gumbel
  - `app/training/parallel_selfplay.py:371-378` - Parallel integration
- **Expected Impact**: +30-70 Elo (richer policy learning signal)

### 1.3 Root Dirichlet Noise for Exploration

- **Status**: Implemented, Integrated
- **File**: `app/ai/gumbel_mcts_ai.py` (lines 227-234, 300-343)
- **Config**: `self_play=True`, `root_dirichlet_alpha`, `root_noise_fraction=0.25`
- **Expected Impact**: +20-40 Elo (more diverse training games)

### 1.4 Temperature Scheduling

- **Status**: Implemented, Integrated
- **File**: `app/training/temperature_scheduling.py`
- **Features**: Linear/exponential/cosine decay, adaptive, curriculum-based
- **Expected Impact**: +10-30 Elo (better exploration/exploitation balance)

---

## 2. EXPERIMENTAL AI ALGORITHMS (D12-D19)

### 2.1 EBMO AI (D12)

- **Status**: Implemented, Registered
- **File**: `app/ai/ebmo_ai.py` (~1500 lines)
- **Description**: Energy-Based Move Optimization using gradient descent
- **Integration Gap**: No training pipeline, no tournament validation
- **Next Step**: Run tournament against D9-D11 to validate strength

### 2.2 GMO AI (D13)

- **Status**: Implemented, Registered
- **File**: `app/ai/gmo_ai.py` (~649 lines)
- **Description**: Gradient Move Optimization with entropy-guided search
- **Performance**: 68.8% vs MCTS(D7), 100% vs Random
- **Integration Gap**: Training pipeline exists but not in tier promotion

### 2.3 IG-GMO AI (D14)

- **Status**: Implemented, Registered
- **File**: `app/ai/ig_gmo.py` (~677 lines)
- **Description**: Information-Gain GMO with GNN state encoder
- **Integration Gap**: Research-grade, no tests, no training pipeline
- **Recommendation**: Consider deprecation unless validated

### 2.4 GMO v2 AI (D18-D19)

- **Status**: Implemented, Registered
- **File**: `app/ai/gmo_v2.py` (~733 lines)
- **Description**: Enhanced GMO with attention + ensemble voting
- **Integration Gap**: No tests, no training pipeline
- **Next Step**: Benchmark against GMO v1

### 2.5 GMO-MCTS Hybrid (D17)

- **Status**: Implemented, Registered (experimental)
- **File**: `app/ai/gmo_mcts_hybrid.py`
- **Description**: GMO priors + MCTS tree search
- **Integration Gap**: Not validated, potential for strong play
- **Note**: Unarchived 2025-12-21 and wired into AI factory D17 tier

---

## 3. TRAINING ENHANCEMENTS

### 3.1 Multi-Task Learning Auxiliary Targets

- **Status**: Partially Implemented
- **File**: `app/training/datasets.py`
- **Features**: Game length, piece count, outcome prediction
- **Integration Gap**: RingRiftDataset supports it, but training loop needs wiring

### 3.2 Curriculum Feedback System

- **Status**: Implemented, Integrated
- **File**: `app/training/curriculum_feedback.py`
- **Features**: Dynamic weight adjustment based on Elo/plateau events
- **Integration**: Wired into `unified_orchestrator.py`

### 3.3 Improvement Optimizer

- **Status**: Implemented, Integrated
- **File**: `app/training/improvement_optimizer.py`
- **Features**: Positive feedback acceleration, dynamic thresholds
- **Integration**: Wired into `unified_orchestrator.py`

### 3.4 Reanalysis Pipeline

- **Status**: Implemented, Partially Integrated
- **File**: `app/training/reanalysis.py`
- **Description**: Re-evaluate old games with stronger model
- **Integration Gap**: Manual invocation only, not in automated pipeline

### 3.5 Knowledge Distillation

- **Status**: Implemented
- **File**: `app/training/distillation.py`
- **Features**: Teacher-student training, ensemble compression
- **Integration Gap**: No automated pipeline for checkpoint distillation

### 3.6 Canonical Action-Space Migration (Correctness + Strength)

- **Status**: Not Integrated
- **Files**:
  - `app/ai/neural_net/square_encoding.py`
  - `app/ai/neural_net/square_architectures.py`
  - `app/ai/mcts_ai.py`, `app/ai/minimax_ai.py`
  - `app/ai/nnue_policy.py`
- **Issue**: The policy/action encoding still references legacy action types
  (`line_formation`, `territory_claim`, `choose_line_reward`, `process_territory_region`).
  This is a hidden correctness risk and dilutes policy learning signal.
- **Next Step**: Introduce a canonical action-space adapter that maps
  legacy aliases to canonical move types at model I/O, then refactor
  encoders and policy heads to use only canonical action indices.
- **Expected Impact**: Higher policy consistency, fewer off-policy errors,
  and cleaner training data (quality + Elo).

---

## 4. SEARCH OPTIMIZATIONS

### 4.1 Incremental Search (Make/Unmake)

- **Status**: Implemented, Opt-in
- **Files**: `app/ai/minimax_ai.py`, `app/ai/mcts_ai.py`, `app/ai/descent_ai.py`
- **Control**: `use_incremental_search=True` in AIConfig
- **Expected Impact**: 2-5x speedup vs immutable state cloning

### 4.2 Transposition Tables

- **Status**: Implemented
- **File**: `app/ai/minimax_ai.py`, `app/ai/mcts_ai.py`
- **Features**: Zobrist hashing, bounded table, depth-aware replacement
- **Integration Gap**: Not all search modes use it consistently

### 4.3 Killer Heuristic

- **Status**: Implemented, Integrated
- **File**: `app/ai/minimax_ai.py` (lines 88-92, 653-732)
- **Description**: Move ordering optimization for alpha-beta pruning

### 4.4 Progressive Widening

- **Status**: Implemented, Integrated
- **File**: `app/ai/mcts_ai.py` (lines 3265-3299)
- **Description**: Gradually expand legal moves for large action spaces (Square19, Hex)
- **Config**: Auto-enabled for `BoardType.SQUARE19` and `BoardType.HEXAGONAL`
- **Formula**: `max_children = max(min_children, c * v^alpha)` where c=2.0, alpha=0.5
- **Integration**: Used in `_can_expand_node()` at lines 1396, 1413, 2393, 2488

---

## 5. ARCHITECTURE IMPROVEMENTS

### 5.1 Vector Value Head for Multi-Player

- **Status**: Implemented in CNN architectures
- **Files**: `app/ai/neural_net/square_architectures.py`, `hex_architectures.py`
- **Integration Gap**: Training doesn't always use multi-player value targets

### 5.2 Attention-Based Architectures (V4)

- **Status**: Implemented
- **File**: `app/ai/neural_net/square_architectures.py` (RingRiftCNN_v4)
- **Features**: NAS-optimized, attention blocks
- **Integration Gap**: Only for square boards, not hex

### 5.3 GMO Shared Components

- **Status**: Just Created (2025-12-21)
- **File**: `app/ai/gmo_shared.py`
- **Features**: Base classes for state/move encoders, value networks
- **Next Step**: Refactor GMO variants to use shared components

---

## 6. VALIDATION GAPS

### 6.1 Missing Tournament Data

- No systematic D12-D19 tournament results
- No cross-board-type strength comparison
- GMO vs Gumbel not documented

### 6.2 Missing Test Coverage

- `gumbel_mcts_ai.py` visit distribution tests
- `gmo_ai.py`, `gmo_v2.py`, `ig_gmo.py` unit tests
- GPU batching parity tests (shadow validation exists but not in CI)

---

## 7. QUICK WINS (Minimal Effort, High Value)

| Feature                              | Effort | Impact     | File                                    |
| ------------------------------------ | ------ | ---------- | --------------------------------------- |
| Run D12-D19 tournament               | 2h     | Discovery  | `scripts/run_distributed_tournament.py` |
| Enable multi-player value training   | 1h     | +10-20 Elo | `app/training/train_nn.py`              |
| Add reanalysis to automated pipeline | 4h     | +20-40 Elo | `app/training/reanalysis.py`            |
| Wire auxiliary tasks in training     | 2h     | +5-15 Elo  | `app/training/unified_orchestrator.py`  |

---

## 8. DEPRECATION CANDIDATES

| Module                            | Status     | Reason                            | Timeline |
| --------------------------------- | ---------- | --------------------------------- | -------- |
| `app/ai/_neural_net_legacy.py`    | Deprecated | Migrated to `neural_net/` package | Q1 2026  |
| `app/ai/ig_gmo.py`                | Review     | Research-grade, no validation     | Q1 2026  |
| Legacy search paths in AI classes | Refactor   | Move to LegacySearchMixin         | Q1 2026  |

---

## Next Actions

1. **Run experimental AI tournament** to discover if any D12-D19 beats production D9-D11
2. **Enable multi-player value training** for 3-4P games
3. **Add reanalysis to automated training** for data quality boost
4. **Migrate policy/action encoding to canonical move types** (legacy aliases only via adapters)
5. **Create LegacySearchMixin** to isolate legacy code from incremental search
6. **Implement progressive widening** for Square19/Hex
