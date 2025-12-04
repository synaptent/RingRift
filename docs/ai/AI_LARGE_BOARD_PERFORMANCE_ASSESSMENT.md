# AI Large Board Performance Assessment

> **SSOT**: This document is the canonical reference for understanding and addressing
> AI performance issues on large boards (Square19, Hexagonal).
> Last updated: 2025-11-30

## Executive Summary

The `probe_plateau_diagnostics.py` script runs for hours when evaluating Square19 and
Hexagonal boards because the bottleneck is **move generation and application**, not
heuristic feature evaluation. The existing "light" `eval_mode` only skips Tier-2
heuristic features but does not address the core computational expense.

### Root Cause Analysis

| Component           | Bottleneck                                          | Complexity                                              |
| ------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| `get_valid_moves()` | Enumerates ALL legal moves via GameEngine           | O(stacks × directions × ray_length × landing_positions) |
| `apply_move()`      | Creates full state copies with expensive validation | O(board_size²) per move                                 |
| Capture enumeration | Walks ALL capture rays from ALL stacks              | O(stacks × 6_directions × max_ray_length)               |

For a 19×19 board with ~50 stacks and rays up to 18 cells long, each call to
`get_valid_moves()` can touch **thousands** of candidate moves. With:

- 8 games per evaluation
- Up to 200 moves per game
- 3 boards × 8+ candidate profiles

This results in **hundreds of thousands** of expensive move generation calls.

---

## 1. Immediate Workarounds

### 1.1 Run on Square8 Only for Quick Diagnostics

```bash
# Fast smoke test (~minutes instead of hours)
python scripts/probe_plateau_diagnostics.py \
  --boards square8 \
  --games-per-eval 4 \
  --eval-mode multi-start \
  --export-dir logs/plateau_probe/quick_smoke
```

**Rationale**: Square8 is ~10× smaller than Square19 and evaluation completes in
reasonable time. Use this for rapid iteration on heuristic weight profiles.

### 1.2 Reduce Games Per Evaluation

```bash
# Fewer games = faster iteration at cost of statistical significance
python scripts/probe_plateau_diagnostics.py \
  --boards square8,square19,hex \
  --games-per-eval 2 \
  --max-moves 50 \
  --export-dir logs/plateau_probe/reduced_load
```

**Trade-off**: Lower `--games-per-eval` and `--max-moves` reduce signal quality but
provide faster feedback during development.

### 1.3 Progressive Board Unlocking

For training purposes, consider a staged approach:

1. **Phase 1**: Train/tune on Square8 only until weights stabilize
2. **Phase 2**: Validate on Square19 with reduced games
3. **Phase 3**: Final validation on Hexagonal

---

## 2. Short-Term Optimizations (1-2 days)

### 2.1 Move Sampling Instead of Exhaustive Enumeration

Add a `max_moves_sample` parameter to the evaluation harness that:

1. Generates all valid moves once
2. Randomly samples up to N moves for evaluation
3. Always includes "obviously good" moves (captures, territory-completing placements)

```python
# Proposed addition to HeuristicAI.select_move
def select_move(self, game_state: GameState) -> Optional[Move]:
    valid_moves = self.get_valid_moves(game_state)

    if len(valid_moves) > self.max_moves_to_evaluate:
        # Prioritize captures and smart placements
        prioritized = self._prioritize_moves(valid_moves, game_state)
        valid_moves = prioritized[:self.max_moves_to_evaluate]

    # Evaluate only the sampled subset
    ...
```

### 2.2 Early Termination in Self-Play

If the evaluation harness detects a "runaway" game (one side clearly winning),
terminate early with a victory rather than playing to the 200-move limit:

```python
# In self-play loop
if abs(heuristic_score) > EARLY_TERMINATION_THRESHOLD:
    # One side has overwhelming advantage, end game early
    break
```

### 2.3 Board-Specific Move Limits

```python
# Add to env.py
TRAINING_MAX_MOVES_BY_BOARD: Dict[BoardType, int] = {
    BoardType.SQUARE8: 200,
    BoardType.SQUARE19: 100,  # Shorter games
    BoardType.HEXAGONAL: 80,   # Even shorter to manage complexity
}
```

---

## 3. Medium-Term Optimizations (1-2 weeks)

### 3.1 Implement SearchBoard with Make/Unmake Pattern

The AI Improvement Backlog §4.1 already specifies this optimization:

```python
class SearchBoard:
    """Lightweight board representation for fast AI search."""

    def __init__(self, game_state: GameState):
        # Extract only fields needed for search
        self.stacks: Dict[str, StackInfo] = ...
        self.markers: Dict[str, int] = ...  # player number only
        self.collapsed: Set[str] = ...

    def make_move(self, move: Move) -> UndoInfo:
        """Apply move in-place, return undo info for unmake."""
        # O(1) incremental update instead of full state copy
        ...

    def unmake_move(self, undo: UndoInfo):
        """Restore previous state from undo info."""
        ...
```

**Expected speedup**: 10-50× for tree-search AIs (Minimax, MCTS, Descent).
For HeuristicAI (single-depth), speedup is smaller but still significant.

### 3.2 Lazy Move Generation

Instead of generating ALL moves upfront, use a generator that yields moves
incrementally. Stop evaluation as soon as a sufficiently good move is found:

```python
def get_valid_moves_lazy(self, game_state: GameState) -> Generator[Move, None, None]:
    """Yield moves one at a time, allowing early termination."""
    # Yield captures first (usually best)
    yield from self._enumerate_captures(game_state)
    # Then placements
    yield from self._enumerate_placements(game_state)
    # Then regular moves
    yield from self._enumerate_movements(game_state)
```

### 3.3 Precomputed Geometry Tables

For fixed board sizes, precompute:

- Adjacent position lookup tables
- Line-of-sight direction vectors
- Distance-from-center values

```python
# One-time precomputation at startup
SQUARE19_ADJACENCY: Dict[str, List[str]] = precompute_adjacency(BoardType.SQUARE19)
SQUARE19_LOS_RAYS: Dict[str, Dict[Direction, List[str]]] = precompute_rays(BoardType.SQUARE19)
```

---

## 4. Long-Term Optimizations (Future)

### 4.1 Neural Network for Move Ordering

Train a lightweight policy network that predicts "promising" moves. Use its output
to order moves so that alpha-beta pruning cuts off poor branches early:

```python
# Move ordering via policy network
policy_probs = self.policy_net.predict(game_state)
sorted_moves = sorted(valid_moves, key=lambda m: policy_probs[encode(m)], reverse=True)
```

### 4.2 Heuristic-Guided Move Generation

Instead of enumerating all moves and then scoring, integrate the heuristic into
move generation to filter obviously bad moves:

```python
def get_promising_moves(self, game_state: GameState) -> List[Move]:
    """Return only moves that pass a quick heuristic filter."""
    for move in self.get_valid_moves_lazy(game_state):
        if self._quick_heuristic_filter(move, game_state):
            yield move
```

### 4.3 Parallel Evaluation

Use Python multiprocessing to evaluate different games/candidates in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(evaluate_candidate, candidate, baseline, board)
        for candidate in candidates
        for board in boards
    ]
    results = [f.result() for f in futures]
```

---

## 5. Recommended Approach

### For Immediate Work

1. **Run diagnostics on Square8 only** for rapid iteration
2. **Use `--games-per-eval 4` and `--max-moves 100`** for faster feedback

### For This Week

1. Implement move sampling (§2.1) to cap evaluation cost per position
2. Add board-specific max-move limits (§2.3) to `env.py`

### For Next Sprint

1. Implement SearchBoard with make/unmake pattern (§3.1)
2. Refactor HeuristicAI and MinimaxAI to use SearchBoard internally

---

## 6. Configuration Reference

### Current Configuration (env.py)

```python
TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD: Dict[BoardType, str] = {
    BoardType.SQUARE8: "full",    # Full structural heuristics
    BoardType.SQUARE19: "light",  # Skip Tier-2 features
    BoardType.HEXAGONAL: "light", # Skip Tier-2 features
}
```

**Note**: "light" mode only affects heuristic FEATURE evaluation, not move generation.

### Recommended New Configuration

```python
# Proposed addition
TRAINING_MOVE_SAMPLE_LIMIT_BY_BOARD: Dict[BoardType, int] = {
    BoardType.SQUARE8: 0,      # No sampling (evaluate all)
    BoardType.SQUARE19: 50,    # Sample up to 50 moves
    BoardType.HEXAGONAL: 40,   # Sample up to 40 moves
}

TRAINING_MAX_MOVES_BY_BOARD: Dict[BoardType, int] = {
    BoardType.SQUARE8: 200,
    BoardType.SQUARE19: 100,
    BoardType.HEXAGONAL: 80,
}
```

---

## 7. Related Documentation

- [`AI_IMPROVEMENT_BACKLOG.md`](./supplementary/AI_IMPROVEMENT_BACKLOG.md) §4.1 - SearchBoard design
- [`ai-service/docs/MAKE_UNMAKE_DESIGN.md`](../ai-service/docs/MAKE_UNMAKE_DESIGN.md) - Make/unmake pattern
- [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md) - Overall AI system design
- [`AI_TRAINING_AND_DATASETS.md`](./AI_TRAINING_AND_DATASETS.md) - Training pipeline

---

## 8. Action Items

| Priority | Task                                               | Owner | Status       |
| -------- | -------------------------------------------------- | ----- | ------------ |
| P0       | Run Square8-only diagnostics for immediate results | -     | Pending      |
| P1       | Add move sampling to HeuristicAI                   | -     | Backlog      |
| P1       | Add board-specific max-move limits                 | -     | Backlog      |
| P2       | Implement SearchBoard with make/unmake             | -     | Backlog §4.1 |
| P3       | Parallel evaluation harness                        | -     | Future       |
