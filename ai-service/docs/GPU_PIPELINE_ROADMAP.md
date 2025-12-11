# RingRift GPU Pipeline Roadmap

> **Document Status:** Living document for GPU acceleration strategy
> **Last Updated:** 2025-12-11
> **Author:** AI Assistant with human review

## Executive Summary

This document outlines an incremental strategy to achieve full GPU-accelerated game execution for RingRift AI training while maintaining 100% rules parity with the canonical CPU implementation. The approach travels through three phases, each delivering measurable value while building toward the final goal.

**Key Insight:** Rather than attempting a risky "big bang" GPU rewrite, we build incrementally with parity validation at each stage. The CPU rules engine remains the source of truth throughout, with GPU acceleration added as a verified optimization layer.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Architecture Problems](#2-architecture-problems)
3. [Rule Simplifications in Current GPU Code](#3-rule-simplifications-in-current-gpu-code)
4. [Effort vs Benefit Analysis](#4-effort-vs-benefit-analysis)
5. [Incremental Roadmap](#5-incremental-roadmap)
6. [Phase 1: Solidify Hybrid Foundation](#6-phase-1-solidify-hybrid-foundation)
7. [Phase 2: GPU-Accelerated Move Generation](#7-phase-2-gpu-accelerated-move-generation)
8. [Phase 3: Full GPU Rules with CPU Oracle](#8-phase-3-full-gpu-rules-with-cpu-oracle)
9. [Parity Testing Strategy](#9-parity-testing-strategy)
10. [Risk Mitigation](#10-risk-mitigation)
11. [Decision Gates](#11-decision-gates)
12. [Implementation Priorities](#12-implementation-priorities)
13. [Appendix: Code References](#appendix-code-references)

---

## 1. Current State Analysis

### 1.1 GPU File Inventory

| File                    | Lines  | Purpose                                    | Status                                         |
| ----------------------- | ------ | ------------------------------------------ | ---------------------------------------------- |
| `gpu_kernels.py`        | ~804   | Move generation kernels, evaluation kernel | Partial - Python loops defeat parallelism      |
| `gpu_batch.py`          | ~917   | GPUHeuristicEvaluator, GPUBatchEvaluator   | Complete - 45-weight CMA-ES evaluation         |
| `gpu_parallel_games.py` | ~2,615 | BatchGameState, ParallelGameRunner         | Partial - per-game loops, rule simplifications |
| `hybrid_gpu.py`         | ~822   | HybridGPUEvaluator - CPU rules + GPU eval  | Complete - safe path with 100% parity          |
| `cuda_rules.py`         | ~300   | CUDA kernels for territory (flood-fill)    | Stub - kernels exist but not integrated        |
| `numba_rules.py`        | ~900   | Numba JIT-compiled rule functions          | Complete - 10-50x faster than pure Python      |

### 1.2 Component Status Matrix

| Component                | Implementation    | Correctness                      | Performance              |
| ------------------------ | ----------------- | -------------------------------- | ------------------------ |
| **Placement Move Gen**   | JIT kernel exists | Correct                          | Good                     |
| **Movement Move Gen**    | Python loops      | Correct                          | Poor (serialized)        |
| **Capture Move Gen**     | Python loops      | Simplified (no chains)           | Poor                     |
| **Recovery Move Gen**    | Python loops      | Simplified                       | Poor                     |
| **Line Detection**       | Python loops      | **INCORRECT** (hardcoded length) | Poor                     |
| **Territory Processing** | Stub only         | Missing                          | N/A                      |
| **Heuristic Evaluation** | 45 weights        | Correct                          | Medium (adjacency loops) |
| **Victory Checking**     | Python iteration  | Correct                          | Poor                     |
| **FSM Phase Handling**   | Per-game loops    | Simplified                       | Poor                     |

### 1.3 Integration Points

```
run_gpu_selfplay.py
 └─ ParallelGameRunner (gpu_parallel_games.py)
    └─ Fully GPU: move gen + apply + evaluate
    └─ RISK: Rule simplifications affect training quality

run_hybrid_selfplay.py
 └─ HybridGPUEvaluator (hybrid_gpu.py)
    ├─ CPU: GameEngine for rules (100% parity)
    └─ GPU: GPUHeuristicEvaluator for scoring only
    └─ SAFE: Rules correctness guaranteed
```

---

## 2. Architecture Problems

### 2.1 False Parallelism

The GPU code claims "vectorized" operations but uses Python patterns that serialize execution:

```python
# gpu_kernels.py:180-230 - generate_normal_moves_vectorized()
# Despite the name, this runs SEQUENTIALLY on CPU:

for i in range(stack_positions.shape[0]):
    g = stack_positions[i, 0].item()      # .item() forces GPU→CPU sync
    from_y = stack_positions[i, 1].item()  # Each call blocks on GPU
    from_x = stack_positions[i, 2].item()
    height = stack_height[g, from_y, from_x].item()

    for d in range(8):                     # 8 directions - sequential
        for dist in range(height, board_size):  # Variable path length
            # Path validation - sequential per-cell checks
            for step in range(1, dist):
                check_y = from_y + dy * step
                check_x = from_x + dx * step
                cell_owner = stack_owner[g, check_y, check_x].item()  # Another sync!
```

**Impact:** Each `.item()` call forces a GPU→CPU synchronization, completely negating any parallelism benefit. A batch of 1000 games with 10 stacks each and 8 directions results in ~80,000 synchronization points per move generation call.

### 2.2 Irregular Data Access Patterns

RingRift has inherently irregular access patterns that challenge GPU SIMD architecture:

| Pattern                  | Description                                       | GPU Challenge                            |
| ------------------------ | ------------------------------------------------- | ---------------------------------------- |
| **Variable move counts** | Each stack has different number of legal moves    | Requires parallel prefix scan or padding |
| **Path validation**      | Must check each cell along movement path          | Sequential dependency chain              |
| **Chain captures**       | Recursive continuation until no captures remain   | Inherently sequential                    |
| **Territory flood-fill** | Wavefront expansion with unpredictable boundaries | Requires iterative kernel launches       |
| **Line detection**       | Variable-length lines in 8 directions             | Parallel scan possible but complex       |

### 2.3 Per-Game Loop Anti-Pattern

The `ParallelGameRunner._step_movement_phase()` method (gpu_parallel_games.py:2027-2155) iterates over games individually:

```python
def _step_movement_phase(self, mask, weights_list):
    # ...
    for g in range(self.batch_size):        # Sequential over games!
        if not games_with_stacks[g]:
            continue

        capture_start = capture_moves.move_offsets[g].item()
        capture_count = capture_moves.moves_per_game[g].item()

        if capture_count > 0:
            for i in range(capture_count):  # Sequential over moves!
                # ... scoring logic
```

**Impact:** Even with batch sizes of 1000+ games, the code processes them one at a time, achieving no parallelism.

---

## 3. Rule Simplifications in Current GPU Code

### 3.1 Critical Rule Deviations

| Rule                            | Canonical Spec                                                | GPU Implementation   | Impact                                            |
| ------------------------------- | ------------------------------------------------------------- | -------------------- | ------------------------------------------------- |
| **Line length (8×8)**           | 4 for 2-player, 3 for 3-4 player (RR-CANON-R120)              | Hardcoded to 4       | **CRITICAL:** 3-4 player games detect wrong lines |
| **Overlength line choice**      | Option 1 (all + eliminate) or Option 2 (subset, no eliminate) | Always Option 1      | Training learns suboptimal line play              |
| **Chain capture continuation**  | Must continue until no captures available                     | Single capture only  | Misses multi-capture sequences                    |
| **Cap eligibility (territory)** | Multicolor OR single-color height>1; NOT height-1             | Any controlled stack | Allows invalid eliminations                       |
| **Cap eligibility (line)**      | Any controlled stack including height-1                       | Same as territory    | Correct by accident                               |
| **Cap eligibility (forced)**    | Any controlled stack including height-1                       | Same                 | Correct                                           |
| **Recovery cascade**            | Territory processing after recovery line                      | Not implemented      | Misses territory gains                            |
| **Marker removal on landing**   | Remove marker, eliminate top ring                             | Simplified           | May miss eliminations                             |

### 3.2 Locations of Rule Simplifications

```
gpu_parallel_games.py:
  Line 1152-1216: detect_lines_batch()
    - Hardcodes line length to 4 (line 1209: `if len(line) >= 4`)
    - Should use get_effective_line_length(board_type, num_players)

  Line 2104-2125: _step_movement_phase() capture handling
    - Applies single capture only
    - No chain continuation logic

  Line 1219-1280: process_lines_batch()
    - No Option 1/Option 2 choice
    - Always collapses all markers

gpu_kernels.py:
  Line 471-476: evaluate_positions_batch()
    - Line detection for scoring uses hardcoded patterns
    - Fixed as of recent changes but needs verification
```

### 3.3 Correctness Status by Board Type

| Board         | 2-Player                                  | 3-Player                                  | 4-Player                                  |
| ------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| **square8**   | Line length wrong (uses 4, should be 4) ✓ | Line length wrong (uses 4, should be 3) ✗ | Line length wrong (uses 4, should be 3) ✗ |
| **square19**  | Correct (4) ✓                             | Correct (4) ✓                             | Correct (4) ✓                             |
| **hexagonal** | Correct (4) ✓                             | Correct (4) ✓                             | Correct (4) ✓                             |

---

## 4. Effort vs Benefit Analysis

### 4.1 Full GPU Completion Effort Estimate

| Task                              | Effort          | Complexity | Dependencies       |
| --------------------------------- | --------------- | ---------- | ------------------ |
| Fix line length rule              | 1 day           | Low        | None               |
| Fix cap eligibility               | 2-3 days        | Medium     | None               |
| Vectorize placement (already JIT) | 2 days          | Low        | None               |
| Vectorize movement (custom CUDA)  | 2-3 weeks       | High       | CUDA expertise     |
| Vectorize capture (custom CUDA)   | 2 weeks         | High       | Movement done      |
| Chain capture continuation        | 1-2 weeks       | Very High  | Capture done       |
| Implement overlength choices      | 3-4 days        | Medium     | Line detection     |
| Territory flood-fill integration  | 1 week          | Medium     | CUDA kernel exists |
| Recovery cascade processing       | 1 week          | High       | Territory done     |
| FSM phase batching                | 2 weeks         | Very High  | All above          |
| Parity test suite                 | 1 week          | Medium     | None (do first)    |
| **Total**                         | **10-14 weeks** |            |                    |

### 4.2 Expected Performance Gains

| Optimization Level         | Speedup vs CPU     | Speedup vs Hybrid | Notes                           |
| -------------------------- | ------------------ | ----------------- | ------------------------------- |
| Current GPU code           | 0.5-1x             | 0.3-0.5x          | Python loops negate GPU benefit |
| Phase 1 (Hybrid optimized) | 3-5x               | 1x (baseline)     | GPU eval only                   |
| Phase 2 (Partial GPU)      | 5-10x              | 2-3x              | GPU placement + movement        |
| Phase 3 (Full GPU)         | 10-20x theoretical | 3-5x realistic    | Irregular access limits gains   |

### 4.3 Why Not 100x Speedup?

GPUs excel at regular, data-parallel workloads. RingRift has structural limitations:

1. **Game tree search is sequential:** Alpha-beta pruning requires evaluating nodes in order
2. **Move selection is inherently serial:** Must pick one move, then see result
3. **Batch size limits:** GPU memory constrains batch size (~1000 games for 8×8)
4. **Synchronization overhead:** Move application requires CPU coordination
5. **Irregular access:** Path validation, flood-fill have unpredictable memory patterns

**Realistic expectation:** 5-10x speedup for selfplay, with CPU remaining faster for single-game analysis.

---

## 5. Incremental Roadmap

### 5.1 Overview

```
                    Phase 1                 Phase 2                 Phase 3
                   (1-2 weeks)             (4-6 weeks)             (6-8 weeks)
                       │                       │                       │
                       ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   CPU Rules    ──────►  CPU Rules    ──────►  CPU Oracle                     │
│   (100%)               (complex only)         (validation)                   │
│       │                     │                      │                         │
│       ▼                     ▼                      ▼                         │
│   GPU Eval     ──────►  GPU Eval +   ──────►  Full GPU                       │
│   (scoring)             Move Gen              Pipeline                       │
│                         (simple)                                             │
│                                                                              │
│   Speedup:              Speedup:              Speedup:                       │
│   3-5x                  5-10x                 10-20x                         │
│                                                                              │
│   Risk: Low             Risk: Medium          Risk: High                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Value Delivery at Each Phase

| Phase       | Deliverable                           | Training Impact                  | Risk             |
| ----------- | ------------------------------------- | -------------------------------- | ---------------- |
| **Phase 1** | Optimized hybrid evaluation           | 3-5x faster eval, same quality   | Low              |
| **Phase 2** | GPU move generation for simple moves  | 5-10x faster selfplay            | Medium           |
| **Phase 3** | Full GPU pipeline with CPU validation | 10-20x faster, validated quality | High (mitigated) |

---

## 6. Phase 1: Solidify Hybrid Foundation

### 6.1 Goals

1. Establish comprehensive parity testing infrastructure
2. Fix known GPU evaluation bugs
3. Optimize GPU evaluation (eliminate Python loops)
4. Validate training data quality matches CPU-only path
5. Establish baseline metrics for future phases

### 6.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Evaluator (Phase 1)                                 │
│                                                             │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │ CPU Rules Engine        │  │ GPU Evaluation           │  │
│  │ (GameEngine)            │  │ (GPUHeuristicEvaluator)  │  │
│  │                         │  │                          │  │
│  │ - Move generation       │  │ - Position scoring       │  │
│  │ - Move application      │  │ - Neural network eval    │  │
│  │ - FSM transitions       │  │ - Batch state encoding   │  │
│  │ - Victory checking      │  │                          │  │
│  │ - Line/territory proc   │  │                          │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
│             │                            ▲                  │
│             │      State transfer        │                  │
│             └────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Deliverables

| Deliverable                 | Description                              | Acceptance Criteria                 |
| --------------------------- | ---------------------------------------- | ----------------------------------- |
| **Parity test suite**       | Comprehensive tests comparing GPU vs CPU | 100% pass rate                      |
| **Line length fix**         | Player-count-aware line detection        | Correct for all board/player combos |
| **Adjacency vectorization** | Eliminate Python loops in scoring        | No `.item()` calls in hot path      |
| **Benchmark suite**         | Measure CPU vs Hybrid performance        | Documented speedup metrics          |
| **Training validation**     | Compare model quality                    | Equivalent Elo after training       |

### 6.4 Implementation Plan

#### Week 1: Parity Infrastructure

| Day | Task                                      | Output                  |
| --- | ----------------------------------------- | ----------------------- |
| 1   | Create `tests/gpu/test_gpu_cpu_parity.py` | Test file with fixtures |
| 2   | Add placement move parity tests           | 10+ test cases          |
| 3   | Add movement move parity tests            | 15+ test cases          |
| 4   | Add capture move parity tests             | 15+ test cases          |
| 5   | Add evaluation score parity tests         | 20+ test cases          |

#### Week 2: Optimization & Integration

| Day | Task                                  | Output                         |
| --- | ------------------------------------- | ------------------------------ |
| 1   | Fix line length in GPU evaluation     | Updated `gpu_kernels.py`       |
| 2   | Vectorize adjacency calculation       | No Python loops                |
| 3   | Integrate GPU eval into hybrid runner | Updated `hybrid_gpu.py`        |
| 4   | Create benchmark script               | `scripts/benchmark_gpu_cpu.py` |
| 5   | Run benchmarks, document results      | Performance report             |

### 6.5 Code Changes Required

#### 6.5.1 Fix Line Length Detection

```python
# gpu_parallel_games.py - detect_lines_batch()
# Current (WRONG):
if len(line) >= 4:  # Hardcoded!

# Fixed:
from app.rules.core import get_effective_line_length
line_length = get_effective_line_length(board_type, num_players)
if len(line) >= line_length:
```

#### 6.5.2 Vectorize Adjacency Calculation

```python
# gpu_parallel_games.py:1491-1503 - Current (slow):
for g in range(batch_size):
    adj_count = 0.0
    ps = player_stacks[g]
    for y in range(board_size):
        for x in range(board_size):
            if ps[y, x]:
                if x + 1 < board_size and ps[y, x + 1]:
                    adj_count += 1.0
                if y + 1 < board_size and ps[y + 1, x]:
                    adj_count += 1.0
    adjacency_score[g] = adj_count

# Fixed (vectorized):
# Right neighbors
right_adj = (player_stacks[:, :, :-1] & player_stacks[:, :, 1:]).sum(dim=(1, 2))
# Down neighbors
down_adj = (player_stacks[:, :-1, :] & player_stacks[:, 1:, :]).sum(dim=(1, 2))
adjacency_score = (right_adj + down_adj).float()
```

---

## 7. Phase 2: GPU-Accelerated Move Generation

### 7.1 Goals

1. Move placement and simple movement to GPU
2. Maintain CPU fallback for complex operations
3. Implement shadow validation for drift detection
4. Achieve 5-10x speedup over Phase 1

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Hybrid+ Evaluator (Phase 2)                                        │
│                                                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ CPU Rules Engine        │  │ GPU Accelerated                  │  │
│  │ (Complex Operations)    │  │ (Simple Operations)              │  │
│  │                         │  │                                  │  │
│  │ - Chain captures        │  │ - Placement move gen             │  │
│  │ - Territory processing  │  │ - Simple movement gen            │  │
│  │ - Recovery cascades     │  │ - Single capture gen             │  │
│  │ - FSM edge cases        │  │ - Position evaluation            │  │
│  │ - Victory validation    │  │ - Move application (simple)      │  │
│  └─────────────────────────┘  └──────────────────────────────────┘  │
│             ▲                            │                          │
│             │      Shadow Validation     │                          │
│             └────────────────────────────┘                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Shadow Validator                                            │    │
│  │ - Samples 1-5% of GPU-generated moves                       │    │
│  │ - Validates against CPU rules engine                        │    │
│  │ - Logs divergence statistics                                │    │
│  │ - Halts if divergence exceeds threshold                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Deliverables

| Deliverable              | Description                | Acceptance Criteria      |
| ------------------------ | -------------------------- | ------------------------ |
| **JIT placement kernel** | Fully vectorized placement | No `.item()` calls       |
| **JIT movement kernel**  | Vectorized path validation | Custom CUDA or torch.jit |
| **Shadow validator**     | Runtime parity checking    | <0.1% divergence rate    |
| **Capture generation**   | GPU single-capture gen     | Matches CPU output       |
| **Fallback mechanism**   | Automatic CPU fallback     | Seamless for complex ops |

### 7.4 Technical Approach

#### 7.4.1 Vectorized Path Validation

The key challenge is validating movement paths without Python loops. Approach:

```python
@torch.jit.script
def validate_paths_vectorized(
    stack_owner: torch.Tensor,      # (batch, board, board)
    from_positions: torch.Tensor,   # (N, 3) - [game, y, x]
    directions: torch.Tensor,       # (N,) - direction index 0-7
    distances: torch.Tensor,        # (N,) - move distance
    current_player: torch.Tensor,   # (batch,)
) -> torch.Tensor:
    """Validate all paths in parallel.

    Returns:
        Boolean tensor (N,) - True if path is valid
    """
    # Pre-compute direction offsets
    dy = torch.tensor([-1, -1, -1, 0, 0, 1, 1, 1], device=stack_owner.device)
    dx = torch.tensor([-1, 0, 1, -1, 1, -1, 0, 1], device=stack_owner.device)

    max_dist = distances.max().item()
    N = from_positions.shape[0]

    # Build all intermediate positions at once
    # Shape: (N, max_dist) for each coordinate
    steps = torch.arange(1, max_dist + 1, device=stack_owner.device)

    # Broadcast: from_y + dy[dir] * steps
    # ... (vectorized path construction)

    # Check blocking in parallel using gather operations
    # ...
```

#### 7.4.2 Shadow Validation Implementation

```python
class ShadowValidator:
    """Validates GPU moves against CPU rules engine."""

    def __init__(self, sample_rate: float = 0.05, threshold: float = 0.001):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.divergence_count = 0
        self.total_validated = 0

    def validate_moves(
        self,
        gpu_moves: List[Move],
        game_state: GameState,
        cpu_engine: GameEngine,
    ) -> bool:
        """Validate a sample of GPU-generated moves.

        Returns:
            True if validation passes, False if divergence detected
        """
        if random.random() > self.sample_rate:
            return True

        cpu_moves = cpu_engine.generate_legal_moves(game_state)
        cpu_move_set = set((m.type, m.from_pos, m.to_pos) for m in cpu_moves)
        gpu_move_set = set((m.type, m.from_pos, m.to_pos) for m in gpu_moves)

        if cpu_move_set != gpu_move_set:
            self.divergence_count += 1
            logger.warning(f"Move divergence: CPU={len(cpu_moves)}, GPU={len(gpu_moves)}")
            logger.warning(f"Missing in GPU: {cpu_move_set - gpu_move_set}")
            logger.warning(f"Extra in GPU: {gpu_move_set - cpu_move_set}")

        self.total_validated += 1
        divergence_rate = self.divergence_count / self.total_validated

        if divergence_rate > self.threshold:
            raise RuntimeError(f"GPU divergence rate {divergence_rate:.4f} exceeds threshold")

        return True
```

---

## 8. Phase 3: Full GPU Rules with CPU Oracle

### 8.1 Goals

1. Complete GPU implementation of all game rules
2. Maintain CPU oracle for validation
3. Achieve 10-20x speedup for selfplay
4. Ensure training data quality through statistical validation

### 8.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Full GPU Pipeline (Phase 3)                                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ GPU Rules Engine                                            │    │
│  │                                                             │    │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │    │
│  │ │ Move Gen    │ │ Move Apply  │ │ Post-Processing         │ │    │
│  │ │             │ │             │ │                         │ │    │
│  │ │ - Placement │ │ - Stack ops │ │ - Line detection        │ │    │
│  │ │ - Movement  │ │ - Markers   │ │ - Territory flood-fill  │ │    │
│  │ │ - Capture   │ │ - Captures  │ │ - Victory checking      │ │    │
│  │ │ - Recovery  │ │             │ │ - FSM transitions       │ │    │
│  │ └─────────────┘ └─────────────┘ └─────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ CPU Oracle                                                  │    │
│  │                                                             │    │
│  │ - Validates 1% of moves (configurable)                      │    │
│  │ - Full validation on victory/endgame                        │    │
│  │ - Deterministic replay capability                           │    │
│  │ - Statistical divergence tracking                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Key Technical Challenges

#### 8.3.1 Chain Capture Parallelization

Chain captures are inherently sequential (must complete one capture to see if another is available). Approach:

```
Option A: Warp-Cooperative Processing
- Assign one warp (32 threads) per game
- Threads cooperate on single chain
- Sequential within chain, parallel across games

Option B: Speculative Execution
- Pre-compute all possible capture sequences
- Select valid sequence after evaluation
- Higher memory cost, better parallelism

Recommendation: Option A for correctness, Option B as optimization later
```

#### 8.3.2 Territory Flood-Fill

The existing `cuda_rules.py` has a flood-fill kernel. Integration steps:

1. Fix boundary condition handling (current kernel has edge cases)
2. Add proper region labeling for multi-region detection
3. Integrate with main game state tensor format
4. Add cascade processing for territory-triggered territory

#### 8.3.3 FSM Phase Batching

Rather than processing games one at a time, batch by phase:

```python
def step_batch_by_phase(self, states: BatchGameState):
    """Process games grouped by current phase."""

    # Group games by phase
    placement_mask = (states.phase == Phase.PLACEMENT)
    movement_mask = (states.phase == Phase.MOVEMENT)
    line_mask = (states.phase == Phase.LINE_PROCESSING)
    territory_mask = (states.phase == Phase.TERRITORY_PROCESSING)

    # Process each phase group in parallel
    if placement_mask.any():
        self._step_placement_batch(states, placement_mask)

    if movement_mask.any():
        self._step_movement_batch(states, movement_mask)

    # ... etc
```

### 8.4 Deliverables

| Deliverable                      | Description                     | Acceptance Criteria       |
| -------------------------------- | ------------------------------- | ------------------------- |
| **Chain capture kernel**         | Warp-cooperative capture chains | Matches CPU chain results |
| **Territory kernel integration** | Flood-fill with cascade         | Correct region detection  |
| **Full FSM on GPU**              | All phases batched              | No per-game Python loops  |
| **CPU oracle system**            | Statistical validation          | Configurable sample rate  |
| **Deterministic replay**         | GPU games replayable on CPU     | Bit-exact match possible  |

---

## 9. Parity Testing Strategy

### 9.1 Test Categories

| Category                   | Description                     | Test Count | Run Frequency |
| -------------------------- | ------------------------------- | ---------- | ------------- |
| **Unit: Move Generation**  | Compare move lists              | 50+        | Every commit  |
| **Unit: Move Application** | Compare state after apply       | 30+        | Every commit  |
| **Unit: Evaluation**       | Compare scores (with tolerance) | 40+        | Every commit  |
| **Integration: Full Game** | Same seed → same game           | 20+        | Daily         |
| **Statistical: Training**  | Model quality comparison        | 5+         | Weekly        |

### 9.2 Test Fixtures

```python
# tests/gpu/fixtures/parity_positions.py

PARITY_FIXTURES = {
    "early_game": [
        # Move 5-10 positions with various stack configurations
    ],
    "mid_game": [
        # Positions with lines forming, captures available
    ],
    "late_game": [
        # Territory processing, near-victory states
    ],
    "edge_cases": [
        # Chain captures, recovery, forced elimination
    ],
    "multi_player": [
        # 3-player and 4-player specific positions
    ],
}
```

### 9.3 Parity Test Implementation

```python
# tests/gpu/test_gpu_cpu_parity.py

import pytest
from app.ai.gpu_parallel_games import generate_placement_moves_batch
from app.game_engine import GameEngine

class TestMoveGenerationParity:
    """Verify GPU move generation matches CPU exactly."""

    @pytest.fixture
    def cpu_engine(self):
        return GameEngine()

    @pytest.fixture
    def gpu_state(self, game_state):
        return BatchGameState.from_game_state(game_state)

    @pytest.mark.parametrize("fixture_name,fixture_data", PARITY_FIXTURES.items())
    def test_placement_moves_match(self, fixture_name, fixture_data, cpu_engine):
        """GPU and CPU generate identical placement moves."""
        for state in fixture_data:
            cpu_moves = set(cpu_engine.generate_placement_moves(state))

            gpu_batch = BatchGameState.from_game_state(state)
            gpu_moves_raw = generate_placement_moves_batch(gpu_batch)
            gpu_moves = set(convert_gpu_moves_to_comparable(gpu_moves_raw))

            assert cpu_moves == gpu_moves, (
                f"Placement mismatch in {fixture_name}: "
                f"CPU={len(cpu_moves)}, GPU={len(gpu_moves)}"
            )

    def test_evaluation_scores_close(self, game_states, cpu_engine):
        """GPU and CPU evaluation scores within tolerance."""
        for state in game_states:
            cpu_score = cpu_engine.evaluate(state)
            gpu_score = gpu_eval.evaluate(state)

            assert abs(cpu_score - gpu_score) < 0.01, (
                f"Evaluation mismatch: CPU={cpu_score:.4f}, GPU={gpu_score:.4f}"
            )


class TestFullGameParity:
    """Verify complete games produce same results."""

    def test_deterministic_selfplay(self):
        """Same seed produces identical game on CPU and GPU."""
        seed = 42

        cpu_game = run_cpu_selfplay(seed=seed, max_moves=200)
        gpu_game = run_gpu_selfplay(seed=seed, max_moves=200)

        assert len(cpu_game.moves) == len(gpu_game.moves)
        for i, (cpu_move, gpu_move) in enumerate(zip(cpu_game.moves, gpu_game.moves)):
            assert cpu_move == gpu_move, f"Move {i} differs: CPU={cpu_move}, GPU={gpu_move}"
```

---

## 10. Risk Mitigation

### 10.1 Risk Matrix

| Risk                              | Likelihood | Impact   | Mitigation                      |
| --------------------------------- | ---------- | -------- | ------------------------------- |
| Subtle rule divergence            | High       | Critical | Shadow validation, parity tests |
| Training data quality degradation | Medium     | High     | A/B model comparison            |
| GPU memory exhaustion             | Medium     | Medium   | Adaptive batch sizing           |
| Performance not meeting targets   | Low        | Medium   | Fall back to hybrid path        |
| Maintenance burden                | Medium     | Medium   | Keep CPU as reference impl      |

### 10.2 Mitigation Strategies

#### 10.2.1 Subtle Rule Divergence

**Detection:**

- Shadow validation samples 1-5% of moves
- Full validation on game end
- Statistical tracking of divergence rate

**Response:**

- Automatic halt if divergence > threshold
- Detailed logging for debugging
- Deterministic replay for reproduction

#### 10.2.2 Training Data Quality

**Detection:**

- A/B test: train identical models on CPU vs GPU games
- Compare Elo ratings after N training steps
- Track win rate against baseline

**Response:**

- If GPU-trained model underperforms by >50 Elo: investigate
- If >100 Elo: halt GPU training, fix issues

#### 10.2.3 Memory Management

```python
class AdaptiveBatchRunner:
    """Automatically adjusts batch size based on available memory."""

    def __init__(self, target_memory_fraction: float = 0.8):
        self.target_fraction = target_memory_fraction
        self.current_batch_size = 1000
        self.min_batch_size = 100

    def run_with_adaptive_batching(self, total_games: int):
        """Run games with automatic batch size adjustment."""
        completed = 0

        while completed < total_games:
            try:
                batch = min(self.current_batch_size, total_games - completed)
                self._run_batch(batch)
                completed += batch

                # Try increasing batch size if successful
                self.current_batch_size = min(
                    self.current_batch_size * 1.1,
                    2000  # max batch
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    self.min_batch_size
                )
                logger.warning(f"OOM, reducing batch to {self.current_batch_size}")
```

---

## 11. Decision Gates

### 11.1 Phase 1 → Phase 2 Gate

**Criteria:**

- [ ] All parity tests pass (100%)
- [ ] Speedup ≥ 3x over CPU-only
- [ ] No regression in model training quality
- [ ] Benchmark results documented

**Decision:**

- If all criteria met: Proceed to Phase 2
- If speedup < 2x: Optimize CPU instead, reconsider GPU approach
- If parity failures: Fix before proceeding

### 11.2 Phase 2 → Phase 3 Gate

**Criteria:**

- [ ] Shadow validation divergence < 0.1%
- [ ] Speedup ≥ 5x over Phase 1
- [ ] GPU move generation matches CPU for simple moves
- [ ] Complex operations cleanly fall back to CPU

**Decision:**

- If all criteria met: Proceed to Phase 3
- If divergence > 0.5%: Stop, fix GPU rules
- If speedup < 3x: Evaluate if Phase 3 worth the effort

### 11.3 Phase 3 Completion Gate

**Criteria:**

- [ ] CPU oracle validation passing
- [ ] Speedup ≥ 8x over Phase 1
- [ ] A/B training shows equivalent model quality
- [ ] Tournament validation on CPU matches GPU predictions

**Decision:**

- If all criteria met: GPU pipeline production-ready
- If model quality differs: Investigate, may need to keep CPU for final training
- Maintain CPU oracle permanently regardless

---

## 12. Implementation Priorities

### 12.1 Immediate (Phase 1 Start)

| Priority | Task                              | Effort | Impact                        |
| -------- | --------------------------------- | ------ | ----------------------------- |
| P0       | Create parity test infrastructure | 2 days | Foundation for all work       |
| P0       | Fix line length in GPU code       | 1 day  | Critical correctness bug      |
| P1       | Vectorize adjacency calculation   | 1 day  | Removes hot-path Python loops |
| P1       | Add hex board support to GPU eval | 1 day  | Completes board coverage      |
| P2       | Benchmark CPU vs Hybrid           | 1 day  | Establish baseline metrics    |

### 12.2 Short-term (Phase 1 Completion)

| Priority | Task                             | Effort | Impact               |
| -------- | -------------------------------- | ------ | -------------------- |
| P1       | Integration tests for full games | 3 days | Validates end-to-end |
| P1       | Document benchmark results       | 1 day  | Decision support     |
| P2       | Training quality A/B test setup  | 2 days | Quality validation   |

### 12.3 Medium-term (Phase 2)

| Priority | Task                         | Effort  | Impact            |
| -------- | ---------------------------- | ------- | ----------------- |
| P1       | JIT-compile placement kernel | 2 days  | First GPU speedup |
| P1       | Vectorized path validation   | 2 weeks | Movement on GPU   |
| P1       | Shadow validation system     | 3 days  | Safety net        |
| P2       | Capture move generation      | 1 week  | More moves on GPU |

---

## Appendix: Code References

### A.1 Key Files

| File                    | Purpose              | Key Functions                                                            |
| ----------------------- | -------------------- | ------------------------------------------------------------------------ |
| `gpu_kernels.py`        | Move generation      | `generate_placement_mask_kernel()`, `generate_normal_moves_vectorized()` |
| `gpu_parallel_games.py` | Batch game state     | `BatchGameState`, `ParallelGameRunner`, `evaluate_positions_batch()`     |
| `gpu_batch.py`          | Heuristic evaluation | `GPUHeuristicEvaluator`, `GPUBatchEvaluator`                             |
| `hybrid_gpu.py`         | Safe hybrid path     | `HybridGPUEvaluator`, `game_state_to_gpu_arrays()`                       |
| `cuda_rules.py`         | CUDA kernels         | `_parallel_flood_fill_kernel()`                                          |
| `numba_rules.py`        | JIT CPU rules        | `detect_lines_numba()`, `check_victory_numba()`                          |

### A.2 Configuration Points

```python
# Batch size configuration
DEFAULT_BATCH_SIZE = 1000  # gpu_parallel_games.py
MAX_BATCH_SIZE = 2000      # Memory-limited

# Shadow validation
SHADOW_SAMPLE_RATE = 0.05  # 5% of moves validated
DIVERGENCE_THRESHOLD = 0.001  # 0.1% max divergence

# Performance targets
PHASE_1_SPEEDUP_TARGET = 3.0  # vs CPU-only
PHASE_2_SPEEDUP_TARGET = 5.0  # vs Phase 1
PHASE_3_SPEEDUP_TARGET = 10.0  # vs Phase 1
```

### A.3 Test Locations

```
tests/
├── gpu/
│   ├── test_gpu_cpu_parity.py      # Move generation parity
│   ├── test_evaluation_parity.py   # Scoring parity
│   ├── test_full_game_parity.py    # End-to-end games
│   ├── test_shadow_validation.py   # Shadow validator tests
│   └── fixtures/
│       ├── parity_positions.py     # Test game states
│       └── expected_moves.py       # Expected move lists
├── integration/
│   └── test_training_quality.py    # A/B training comparison
└── benchmarks/
    └── benchmark_gpu_cpu.py        # Performance measurement
```

---

## Document History

| Date       | Version | Changes                        |
| ---------- | ------- | ------------------------------ |
| 2025-12-11 | 1.0     | Initial comprehensive analysis |

---

_This document should be reviewed and updated as implementation progresses. Each phase completion should include updates to reflect actual results vs. estimates._
