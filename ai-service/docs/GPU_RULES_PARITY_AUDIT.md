# GPU Rules Parity Audit & Improvement Plan

> **Document Status:** Active Tracking Document
> **Created:** 2025-12-11
> **Last Updated:** 2025-12-11
> **Purpose:** Track GPU implementation gaps vs canonical rules and improvement progress

---

## Executive Summary

This document provides a comprehensive audit of the GPU implementation against canonical RingRift rules (RULES_CANONICAL_SPEC.md), identifying parity gaps and prioritizing improvements for achieving full rules parity.

**Current State:**

- Phase 1 COMPLETE: 6.56x speedup achieved on CUDA (A10 GPU)
- Phase 2 IN PROGRESS: Shadow validation infrastructure complete
- ~65% rules parity (critical mechanics missing)

**Target:** 100% rules parity while maintaining 5-10x speedup over CPU-only path

---

## Table of Contents

1. [Canonical Rules Reference](#1-canonical-rules-reference)
2. [Parity Gap Analysis](#2-parity-gap-analysis)
3. [Performance Anti-Patterns](#3-performance-anti-patterns)
4. [Prioritized Improvement Plan](#4-prioritized-improvement-plan)
5. [Implementation Details](#5-implementation-details)
6. [Progress Tracking](#6-progress-tracking)
7. [Testing Strategy](#7-testing-strategy)

---

## 1. Canonical Rules Reference

### 1.1 Critical Rules for GPU Parity

| Rule ID       | Rule Name                   | Canonical Spec Reference                                       |
| ------------- | --------------------------- | -------------------------------------------------------------- |
| RR-CANON-R080 | Chain Capture Continuation  | Must continue until no captures available                      |
| RR-CANON-R120 | Line Length Requirements    | 4 for 2p, 3 for 3-4p on square8                                |
| RR-CANON-R130 | Overlength Line Options     | Option 1 (all + eliminate) or Option 2 (subset, no eliminate)  |
| RR-CANON-R140 | Territory Processing        | Flood-fill from collapsed spaces                               |
| RR-CANON-R150 | Cap Eligibility (Territory) | Multicolor OR single-color height>1; NOT height-1              |
| RR-CANON-R110 | Recovery Mechanics          | Marker slide + territory processing                            |
| RR-CANON-R061 | Victory Threshold           | round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))         |
| RR-CANON-R200 | Victory Conditions          | Ring Elimination, Territory (>50%), LPS (2 consecutive rounds) |

### 1.2 FSM Phase Order

Per canonical rules:

```
RING_PLACEMENT → MOVEMENT → LINE_PROCESSING → TERRITORY_PROCESSING → (next player)
```

---

## 2. Parity Gap Analysis

### 2.1 Critical Gaps (Training Quality Impact: HIGH)

| Gap                        | Canonical Behavior                        | GPU Implementation          | Impact                                                           | Location                                          |
| -------------------------- | ----------------------------------------- | --------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| **Chain Captures**         | Must continue until no captures available | Single capture only         | Misses multi-capture sequences, AI learns suboptimal play        | `gpu_parallel_games.py:2104-2125`                 |
| **Territory Processing**   | Flood-fill from collapsed spaces          | Stub only (not implemented) | Territory victory path broken, AI can't learn territory strategy | `gpu_parallel_games.py:process_territory_batch()` |
| **Overlength Line Choice** | Player chooses Option 1 or Option 2       | Always Option 1             | Training learns suboptimal line processing                       | `gpu_parallel_games.py:1219-1280`                 |
| **Cap Eligibility**        | Different rules for line/territory/FE     | Same for all contexts       | Allows invalid eliminations in some cases                        | `gpu_parallel_games.py:process_lines_batch()`     |

### 2.2 Medium Gaps (Training Quality Impact: MEDIUM)

| Gap                           | Canonical Behavior                       | GPU Implementation | Impact                               | Location                                       |
| ----------------------------- | ---------------------------------------- | ------------------ | ------------------------------------ | ---------------------------------------------- |
| **Recovery Cascade**          | Territory processing after recovery line | Not implemented    | Misses territory gains from recovery | `gpu_parallel_games.py:_step_recovery_phase()` |
| **Marker Removal on Landing** | Remove marker, eliminate top ring        | Simplified         | May miss eliminations                | `gpu_parallel_games.py:apply_movement_batch()` |
| **LPS Victory**               | 2 consecutive rounds without progress    | Not tracked        | LPS victory path broken              | `gpu_parallel_games.py:check_victory_batch()`  |

### 2.3 Fixed Gaps (Completed)

| Gap                        | Fix Date   | Notes                                    |
| -------------------------- | ---------- | ---------------------------------------- |
| Line length (square8 3-4p) | 2025-12-11 | Now player-count-aware per RR-CANON-R120 |
| Adjacency calculation      | 2025-12-11 | Vectorized using tensor operations       |
| Victory threshold formula  | 2025-12-11 | Uses RR-CANON-R061 formula               |
| Buried rings tracking      | 2025-12-11 | Fixed indexing bugs                      |

---

## 3. Performance Anti-Patterns

### 3.1 Per-Game Python Loops (Serialization)

**Problem:** GPU code uses Python `for g in range(batch_size)` loops that execute sequentially, negating parallelism benefits.

**Locations:**

1. `gpu_parallel_games.py:2015-2052` - Line potential calculation
2. `gpu_parallel_games.py:2054-2101` - Opponent threat metrics
3. `gpu_parallel_games.py:2103-2158` - Vulnerability and overtake metrics
4. `gpu_parallel_games.py:2160-2178` - Territory metrics

**Impact:** Achieves ~0.5-1x speedup instead of target 10-100x

### 3.2 `.item()` Calls (GPU→CPU Sync)

**Problem:** Each `.item()` call forces GPU→CPU synchronization, blocking execution.

**Example from `gpu_parallel_games.py:2116`:**

```python
my_height = state.stack_height[g, y, x].item()  # Forces sync
```

**Solution:** Use masked tensor operations instead of per-element access.

### 3.3 Variable-Length Data Handling

**Problem:** RingRift has inherently irregular data (variable move counts, path lengths, chain captures).

**GPU Challenge:** SIMD architecture requires uniform work per thread.

**Recommended Approach:**

- Parallel prefix scan for move counting
- Padding with sentinel values
- Warp-cooperative processing for chains

---

## 4. Prioritized Improvement Plan

### 4.1 Priority Matrix

| Priority | Improvement                | Effort  | Impact | ROI              |
| -------- | -------------------------- | ------- | ------ | ---------------- |
| **P0**   | Chain capture continuation | 2 weeks | HIGH   | Training quality |
| **P0**   | Territory flood-fill       | 1 week  | HIGH   | Victory path     |
| **P1**   | Vectorize evaluation loops | 1 week  | HIGH   | 5-10x speedup    |
| **P1**   | Overlength line choice     | 3 days  | MEDIUM | Strategic play   |
| **P2**   | Recovery cascade           | 1 week  | MEDIUM | Rare mechanic    |
| **P2**   | LPS victory tracking       | 2 days  | LOW    | Rare victory     |
| **P3**   | Cap eligibility contexts   | 3 days  | LOW    | Edge cases       |

### 4.2 Recommended Implementation Order

**Sprint 1 (Week 1-2): Chain Captures**

1. Design warp-cooperative capture chain algorithm
2. Implement iterative capture loop with convergence check
3. Add parity tests against CPU capture chains
4. Benchmark performance impact

**Sprint 2 (Week 3): Territory Processing**

1. Port flood-fill algorithm to GPU tensors
2. Implement cascade detection
3. Add territory victory condition checking
4. Test with replay validation

**Sprint 3 (Week 4): Performance**

1. Vectorize evaluation metrics
2. Remove `.item()` calls from hot paths
3. Batch FSM transitions by phase
4. Re-benchmark overall throughput

**Sprint 4 (Week 5): Polish**

1. Overlength line choice (stochastic Option 1/2)
2. Recovery cascade integration
3. LPS victory tracking
4. Comprehensive parity test suite

---

## 5. Implementation Details

### 5.1 Chain Capture Algorithm (P0)

**Approach:** Warp-cooperative iterative processing

```python
def apply_chain_captures_batch(state: BatchGameState) -> None:
    """Apply chain captures iteratively until no captures remain.

    Algorithm:
    1. Generate all single-step captures for all games
    2. Apply captures in parallel
    3. Check for new capture opportunities
    4. Repeat until convergence (no new captures)

    Convergence is guaranteed because:
    - Each capture removes rings from the board
    - Finite rings means finite captures
    - Maximum iterations = max_total_rings
    """
    max_iterations = state.num_players * state.rings_per_player

    for _ in range(max_iterations):
        # Generate capture moves for all games
        capture_moves = generate_capture_moves_batch(state)

        # Check if any game has captures available
        games_with_captures = capture_moves.moves_per_game > 0
        if not games_with_captures.any():
            break  # Convergence: no more captures

        # Select best capture per game (heuristic or first valid)
        selected_captures = select_captures_vectorized(capture_moves, state)

        # Apply captures in parallel
        apply_captures_vectorized(state, selected_captures, games_with_captures)

        # Update game state for next iteration
        # (stack heights, ownership, markers flipped)
```

### 5.2 Territory Flood-Fill Algorithm (P0)

**Approach:** Iterative wavefront expansion

```python
def flood_fill_territory_batch(state: BatchGameState) -> None:
    """Flood-fill territory from collapsed spaces.

    Algorithm:
    1. Initialize frontier from newly collapsed spaces
    2. For each frontier cell, check 4-neighbors
    3. If neighbor is empty AND not owned, add to territory
    4. Expand frontier to newly claimed cells
    5. Repeat until frontier empty

    Tensor representation:
    - territory_owner: (batch, board, board) - 0=unclaimed, 1-4=player
    - frontier: (batch, board, board) - bool mask of active cells
    """
    batch_size = state.batch_size
    board_size = state.board_size
    device = state.device

    # Initialize frontier from newly collapsed spaces
    frontier = state.newly_collapsed.clone()

    max_iterations = board_size * board_size  # Maximum flood distance

    for _ in range(max_iterations):
        if not frontier.any():
            break

        # Get territory owner of frontier cells
        frontier_owners = state.territory_owner * frontier

        # For each direction, check if neighbor can be claimed
        new_claims = torch.zeros_like(frontier)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Shift frontier to check neighbors
            shifted = torch.roll(torch.roll(frontier, dy, 1), dx, 2)
            shifted_owner = torch.roll(torch.roll(frontier_owners, dy, 1), dx, 2)

            # Claimable: empty cell, not already owned, not blocked
            claimable = (
                (state.stack_owner == 0) &  # No stack
                (state.territory_owner == 0) &  # Not already owned
                shifted &  # Adjacent to frontier
                (shifted_owner > 0)  # Owner propagates
            )

            # Assign owner from adjacent territory
            state.territory_owner = torch.where(
                claimable,
                shifted_owner,
                state.territory_owner
            )

            new_claims |= claimable

        # Update frontier for next iteration
        frontier = new_claims

    # Update territory counts
    for p in range(1, state.num_players + 1):
        state.territory_count[:, p] = (state.territory_owner == p).sum(dim=(1, 2))
```

### 5.3 Vectorized Evaluation Metrics (P1)

**Current (Slow):**

```python
for g in range(batch_size):
    for y in range(board_size):
        for x in range(board_size):
            if player_stacks[g, y, x]:
                # Per-cell computation
```

**Optimized (Vectorized):**

```python
# Compute all metrics in parallel using tensor operations
# Example: Two-in-row detection

# Horizontal pairs: stack at (y,x) and stack at (y,x+1)
h_pairs = player_stacks[:, :, :-1] & player_stacks[:, :, 1:]
# Vertical pairs
v_pairs = player_stacks[:, :-1, :] & player_stacks[:, 1:, :]
# Diagonal pairs (both directions)
d1_pairs = player_stacks[:, :-1, :-1] & player_stacks[:, 1:, 1:]
d2_pairs = player_stacks[:, :-1, 1:] & player_stacks[:, 1:, :-1]

two_in_row = (h_pairs.sum(dim=(1,2)) + v_pairs.sum(dim=(1,2)) +
              d1_pairs.sum(dim=(1,2)) + d2_pairs.sum(dim=(1,2))).float()
```

---

## 6. Progress Tracking

### 6.1 Sprint Progress

| Sprint   | Task                   | Status      | Notes |
| -------- | ---------------------- | ----------- | ----- |
| Sprint 1 | Chain capture design   | NOT_STARTED |       |
| Sprint 1 | Chain capture impl     | NOT_STARTED |       |
| Sprint 1 | Chain capture tests    | NOT_STARTED |       |
| Sprint 2 | Territory flood-fill   | NOT_STARTED |       |
| Sprint 2 | Territory tests        | NOT_STARTED |       |
| Sprint 3 | Vectorize evaluation   | NOT_STARTED |       |
| Sprint 3 | Remove .item() calls   | NOT_STARTED |       |
| Sprint 4 | Overlength line choice | NOT_STARTED |       |
| Sprint 4 | Recovery cascade       | NOT_STARTED |       |

### 6.2 Metrics

| Metric               | Baseline (Phase 1) | Current | Target (Phase 3) |
| -------------------- | ------------------ | ------- | ---------------- |
| CUDA Speedup         | 6.56x              | 6.56x   | 10-20x           |
| MPS Speedup          | 1.75x (batch≥500)  | 1.75x   | 3-5x             |
| Rules Parity         | ~65%               | ~65%    | 100%             |
| Parity Tests Passing | 32/32              | 32/32   | 100+             |

### 6.3 Known Issues

| Issue                      | Severity | Status | Notes                    |
| -------------------------- | -------- | ------ | ------------------------ |
| Chain captures missing     | HIGH     | OPEN   | Affects training quality |
| Territory processing stub  | HIGH     | OPEN   | Territory victory broken |
| Per-game loops             | MEDIUM   | OPEN   | Performance bottleneck   |
| Overlength always Option 1 | MEDIUM   | OPEN   | Suboptimal line play     |

---

## 7. Testing Strategy

### 7.1 Parity Test Categories

| Category             | Test Count | Coverage                                 |
| -------------------- | ---------- | ---------------------------------------- |
| Move Generation      | 50+        | Placement, movement, capture, recovery   |
| Move Application     | 30+        | State changes, marker flipping, captures |
| Line Processing      | 20+        | Detection, collapse, overlength          |
| Territory Processing | 15+        | Flood-fill, cascade, victory             |
| Victory Conditions   | 15+        | Ring elimination, territory, LPS         |
| Full Game Replay     | 20+        | End-to-end parity                        |

### 7.2 Test File Locations

- `tests/gpu/test_gpu_cpu_parity.py` - Core parity tests
- `tests/gpu/test_gpu_cpu_replay_parity.py` - Full game replay tests
- `tests/gpu/test_chain_captures.py` - Chain capture specific tests (to create)
- `tests/gpu/test_territory_processing.py` - Territory specific tests (to create)

### 7.3 Replay-Based Validation

For high-confidence parity validation:

1. Replay canonical game recordings on GPU
2. Compare state at each move
3. Flag any divergence for investigation
4. Target: 100% replay accuracy

---

## Appendix A: File References

| File                    | Purpose          | Key Functions                                |
| ----------------------- | ---------------- | -------------------------------------------- |
| `gpu_parallel_games.py` | Main GPU engine  | `ParallelGameRunner`, `BatchGameState`       |
| `gpu_batch.py`          | Batch evaluation | `GPUHeuristicEvaluator`, `GPUBatchEvaluator` |
| `gpu_kernels.py`        | Move gen kernels | `generate_*_moves_vectorized()`              |
| `shadow_validation.py`  | Parity checking  | `ShadowValidator`                            |
| `hybrid_gpu.py`         | Hybrid CPU+GPU   | `HybridGPUEvaluator`                         |

## Appendix B: Canonical Rules Files

| File                         | Size  | Purpose                |
| ---------------------------- | ----- | ---------------------- |
| `RULES_CANONICAL_SPEC.md`    | 110KB | Formal specification   |
| `ringrift_compact_rules.md`  | 45KB  | Implementation-focused |
| `ringrift_complete_rules.md` | 185KB | Full narrative rules   |

---

## Document History

| Date       | Version | Changes                        |
| ---------- | ------- | ------------------------------ |
| 2025-12-11 | 1.0     | Initial audit document created |
