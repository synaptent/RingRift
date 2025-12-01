# Make/Unmake Pattern Extension Analysis

> **Doc Status (2025-11-27): Active (analysis, Python AI host only)**
>
> **Role:** Analysis and prioritisation of where to extend the Python AI service's make/unmake move pattern (`MutableGameState` + `MoveUndo`) beyond `MinimaxAI`—covering MCTS, DescentAI, self-play data generation, tournaments, RL environments, and heuristic AI.
>
> **Not a semantics SSoT:** This document does not define core game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`). Lifecycle semantics are owned by `docs/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`. The Python engine, its make/unmake implementation, and the extensions analysed here are **hosts/adapters** that must match the TS SSoT via the parity backbone and contract vectors.
>
> **Related docs:** `AI_ARCHITECTURE.md`, `docs/AI_TRAINING_AND_DATASETS.md`, `docs/AI_TRAINING_PREPARATION_GUIDE.md`, `docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/STRICT_INVARIANT_SOAKS.md`, `ai-service/docs/MAKE_UNMAKE_DESIGN.md`, `ai-service/AI_ASSESSMENT_REPORT.md`, `ai-service/AI_IMPROVEMENT_PLAN.md`, and `DOCUMENTATION_INDEX.md`.
>
> **Date:** 2025-11-27
>
> **Purpose:** Identify which AI components and training infrastructure would benefit from extending the make/unmake pattern currently implemented in MinimaxAI.

---

## Executive Summary

The make/unmake pattern implemented in [`mutable_state.py`](../app/rules/mutable_state.py) has achieved **12-25x improvement in nodes/second** for MinimaxAI. This document analyzes the applicability of this pattern to other AI components and training infrastructure to identify further optimization opportunities.

### Key Findings

| Component                   | Benefit Potential | Priority | Expected Speedup | Status          |
| --------------------------- | ----------------- | -------- | ---------------- | --------------- |
| **MCTS AI**                 | **High**          | **1**    | 5-15x            | ✅ **COMPLETE** |
| **Descent AI**              | **High**          | **2**    | 5-15x            | ✅ **COMPLETE** |
| **Training Self-Play**      | **High**          | **3**    | 3-8x             | Auto-benefits   |
| **Tournament System**       | Medium            | 4        | 2-5x             | Auto-benefits   |
| **RL Environment**          | Medium            | 5        | 2-4x             | Pending         |
| **Heuristic AI**            | Low               | 6        | 1.5-3x           | Pending         |
| **Neural Network Training** | None              | N/A      | N/A              | N/A             |

---

## Component Analysis

### 1. MCTS AI

**File:** [`ai-service/app/ai/mcts_ai.py`](../app/ai/mcts_ai.py)

#### Current Pattern

```python
# Lines 444-451: Selection phase creates new states via apply_move
while not node.untried_moves and node.children:
    node = node.uct_select_child()
    state = self.rules_engine.apply_move(state, node.move)
    played_moves.append(node.move)

# Expansion
if node.untried_moves:
    m = cast(Move, self.get_random_element(node.untried_moves))
    state = self.rules_engine.apply_move(state, m)
```

#### Analysis

| Question                    | Answer                                                                    |
| --------------------------- | ------------------------------------------------------------------------- |
| Does tree search?           | ✅ Yes - extensive tree exploration with selection, expansion, simulation |
| Needs backtracking?         | ✅ Yes - backpropagation phase needs original state values                |
| State copying bottleneck?   | ✅ Yes - creates new GameState for every node expansion                   |
| Would benefit from Zobrist? | ✅ Yes - transposition table already uses state hashing                   |

#### Benefit Potential: **HIGH**

MCTS explores thousands of nodes per search iteration. Each node expansion currently creates a full GameState copy. The make/unmake pattern would:

1. **Eliminate allocation overhead** during selection descent
2. **Enable efficient backtracking** during backpropagation
3. **Maintain Zobrist hash consistency** for transposition table lookups
4. **Reduce memory pressure** during long searches

#### Integration Complexity: **Medium**

Challenges:

- MCTS stores `game_state` in each `MCTSNode` for tree reuse
- Need to either store undo chains or convert to lazy state reconstruction
- Simulation/rollout phase also uses `apply_move` repeatedly

#### Expected Speedup: **5-15x**

The speedup will be most pronounced during:

- Selection phase (descending the tree)
- Expansion phase (evaluating new positions)
- Simulation/rollout phase (lightweight playouts)

#### Integration Plan

```python
# Proposed modification in MCTSAI.select_move_and_policy()
mutable_state = MutableGameState.from_immutable(game_state)
undo_stack = []

# Selection with make/unmake
while not node.untried_moves and node.children:
    node = node.uct_select_child()
    undo = mutable_state.make_move(node.move)
    undo_stack.append(undo)

# Expansion
if node.untried_moves:
    m = self.get_random_element(node.untried_moves)
    undo = mutable_state.make_move(m)
    undo_stack.append(undo)
    # Store reference to mutable state for evaluation

# Backpropagation - unwind the stack
for undo in reversed(undo_stack):
    mutable_state.unmake_move(undo)
```

---

### 2. Descent AI

**File:** [`ai-service/app/ai/descent_ai.py`](../app/ai/descent_ai.py)

#### Current Pattern

```python
# Line 279: Creates new state for each descent step
next_state = self.rules_engine.apply_move(state, best_move)
val = self._descent_iteration(next_state, depth + 1, deadline=deadline)

# Line 445: Expansion creates states for all children
for move in valid_moves:
    next_state = self.rules_engine.apply_move(state, move)
```

#### Analysis

| Question                    | Answer                                               |
| --------------------------- | ---------------------------------------------------- |
| Does tree search?           | ✅ Yes - iterative deepening with best-first descent |
| Needs backtracking?         | ✅ Yes - values propagate back up the tree           |
| State copying bottleneck?   | ✅ Yes - each descent step creates new state         |
| Would benefit from Zobrist? | ✅ Yes - uses transposition table extensively        |

#### Benefit Potential: **HIGH**

Descent AI performs unbounded best-first search similar to MCTS. Each iteration descends to a leaf, then backpropagates values. The make/unmake pattern would:

1. **Eliminate state cloning** during descent iterations
2. **Enable efficient state restoration** after backpropagation
3. **Support existing transposition table** with maintained Zobrist hashes
4. **Reduce GC pressure** during long searches

#### Integration Complexity: **Medium**

Challenges:

- Recursive descent structure needs careful undo management
- Transposition table stores values, not states (good - no change needed)
- Need to maintain state consistency across recursive calls

#### Expected Speedup: **5-15x**

Similar to MinimaxAI since Descent shares the tree-search paradigm.

#### Integration Plan

```python
def _descent_iteration_mutable(
    self,
    state: MutableGameState,
    depth: int = 0,
    deadline: Optional[float] = None,
) -> float:
    """Descent iteration using make/unmake pattern."""
    # ... check terminal conditions ...

    # Select best child to descend
    best_move = children_values[best_move_key][0]

    # Make move, descend, unmake
    undo = state.make_move(best_move)
    val = self._descent_iteration_mutable(state, depth + 1, deadline=deadline)
    state.unmake_move(undo)

    # Update values
    children_values[best_move_key] = (best_move, val, ...)
    return new_best_val
```

---

### 3. Heuristic AI

**File:** [`ai-service/app/ai/heuristic_ai.py`](../app/ai/heuristic_ai.py)

#### Current Pattern

```python
# Lines 112-114: Single-depth evaluation loop
for move in valid_moves:
    next_state = self.rules_engine.apply_move(game_state, move)
    score = self.evaluate_position(next_state)
```

#### Analysis

| Question                    | Answer                                   |
| --------------------------- | ---------------------------------------- |
| Does tree search?           | ❌ No - single-depth evaluation only     |
| Needs backtracking?         | ✅ Minimal - just iterate through moves  |
| State copying bottleneck?   | ⚠️ Some - depends on branching factor    |
| Would benefit from Zobrist? | ❌ No - doesn't use transposition tables |

#### Benefit Potential: **LOW**

HeuristicAI only evaluates one move at a time at depth 1. The overhead of state copying is present but:

- Branching factor is typically 20-50 moves
- Each evaluation is relatively cheap
- No tree traversal means minimal compounding benefit

#### Integration Complexity: **Easy**

Simple loop-based pattern:

```python
mutable = MutableGameState.from_immutable(game_state)
for move in valid_moves:
    undo = mutable.make_move(move)
    score = self.evaluate_position_mutable(mutable)
    mutable.unmake_move(undo)
    if score > best_score:
        best_score = score
        best_move = move
```

#### Expected Speedup: **1.5-3x**

Modest improvement due to:

- Reduced memory allocation per move
- Better cache locality
- However, evaluation dominates runtime, not state copying

---

### 4. Training Infrastructure

#### 4.1 RL Environment

**File:** [`ai-service/app/training/env.py`](../app/training/env.py)

##### Current Pattern

```python
# Line 89: Step function uses apply_move
def step(self, move: Move) -> Tuple[GameState, float, bool, Dict[str, Any]]:
    self._state = GameEngine.apply_move(self.state, move)
```

##### Analysis

| Question                    | Answer                              |
| --------------------------- | ----------------------------------- |
| Does tree search?           | ❌ No - sequential game progression |
| Needs backtracking?         | ❌ No - forward-only execution      |
| State copying bottleneck?   | ⚠️ Low - one copy per step          |
| Would benefit from Zobrist? | ⚠️ Potentially for state caching    |

##### Benefit Potential: **Medium**

The RL environment processes games sequentially. Benefits would come from:

- Reduced per-step allocation overhead
- Better memory efficiency for long episodes
- Potential for state checkpointing/rollback features

##### Integration Complexity: **Easy**

Could maintain internal `MutableGameState` and convert to immutable only when returning to caller.

##### Expected Speedup: **2-4x**

Modest improvement since each step is independent.

---

#### 4.2 Self-Play Data Generation

**File:** [`ai-service/app/training/generate_data.py`](../app/training/generate_data.py)

##### Current Pattern

```python
# Line 596: Game loop uses env.step which calls apply_move
state, _, done, _ = env.step(move)

# Lines 437-445: AI selection also triggers apply_move in descent
move = ai.select_move(state)
```

##### Analysis

| Question                    | Answer                            |
| --------------------------- | --------------------------------- |
| Does tree search?           | ✅ Yes - via DescentAI selection  |
| Needs backtracking?         | ✅ Yes - in AI search phase       |
| State copying bottleneck?   | ✅ Yes - AI search + game steps   |
| Would benefit from Zobrist? | ✅ Yes - for AI search efficiency |

##### Benefit Potential: **HIGH**

Self-play data generation is dominated by AI move selection time (DescentAI). Integrating make/unmake in the AI layer would:

- Speed up each move selection significantly
- Allow faster game completion
- Enable higher quality data with deeper searches

##### Integration Complexity: **Medium**

Primary benefit comes from integrating make/unmake into DescentAI (see above). The game loop itself is sequential.

##### Expected Speedup: **3-8x**

Most of the speedup comes from faster AI search, not the game loop itself.

---

#### 4.3 Tournament System

**File:** [`ai-service/app/training/tournament.py`](../app/training/tournament.py)

##### Current Pattern

```python
# Line 227: Game loop uses apply_move
state = GameEngine.apply_move(state, move)
```

##### Analysis

| Question                    | Answer                            |
| --------------------------- | --------------------------------- |
| Does tree search?           | ✅ Yes - via DescentAI selection  |
| Needs backtracking?         | ✅ Yes - in AI search phase       |
| State copying bottleneck?   | ✅ Yes - similar to self-play     |
| Would benefit from Zobrist? | ✅ Yes - for AI search efficiency |

##### Benefit Potential: **Medium**

Similar to self-play, but tournaments focus on evaluation rather than high-volume data generation.

##### Integration Complexity: **Easy**

Primarily benefits from AI layer integration.

##### Expected Speedup: **2-5x**

---

#### 4.4 Neural Network Training

**File:** [`ai-service/app/training/train.py`](../app/training/train.py)

##### Analysis

This module loads pre-generated training data and runs gradient descent. It does **not** perform any game state manipulation during training.

##### Benefit Potential: **None**

The make/unmake pattern is not applicable to this component.

---

## Priority-Ranked Integration Plan

### Phase 1: Tree-Search AIs (Highest Impact)

| Priority | Component  | Effort   | Impact              | Dependencies |
| -------- | ---------- | -------- | ------------------- | ------------ |
| **1**    | MCTS AI    | 3-5 days | 5-15x faster search | None         |
| **2**    | Descent AI | 2-4 days | 5-15x faster search | None         |

**Rationale:** These components have the highest node throughput and will see the greatest benefit from avoiding allocation overhead in tree traversal.

### Phase 2: Training Infrastructure (High Volume)

| Priority | Component                 | Effort   | Impact                  | Dependencies         |
| -------- | ------------------------- | -------- | ----------------------- | -------------------- |
| **3**    | Self-Play Data Generation | 1-2 days | 3-8x faster data gen    | Phase 1 (Descent AI) |
| **4**    | Tournament System         | 1 day    | 2-5x faster tournaments | Phase 1              |

**Rationale:** These components run many games and will benefit from faster AI search. Primary benefit comes from Phase 1 integration.

### Phase 3: Sequential Execution (Lower Priority)

| Priority | Component      | Effort    | Impact               | Dependencies |
| -------- | -------------- | --------- | -------------------- | ------------ |
| **5**    | RL Environment | 0.5-1 day | 2-4x faster episodes | None         |
| **6**    | Heuristic AI   | 0.5-1 day | 1.5-3x faster        | None         |

**Rationale:** These components are sequential and see less compounding benefit from make/unmake.

---

## Effort Estimates Summary

| Phase     | Components            | Total Effort  | Expected Value |
| --------- | --------------------- | ------------- | -------------- |
| 1         | MCTS AI, Descent AI   | 5-9 days      | Very High      |
| 2         | Self-Play, Tournament | 2-3 days      | High           |
| 3         | RL Env, Heuristic AI  | 1-2 days      | Medium         |
| **Total** | All                   | **8-14 days** | **High**       |

---

## Technical Considerations

### MutableGameState Reuse

For tree-search algorithms, create a single `MutableGameState` at the root and maintain it throughout the search:

```python
def select_move(self, game_state: GameState) -> Optional[Move]:
    mutable = MutableGameState.from_immutable(game_state)
    # Use mutable throughout search
    best_move = self._search(mutable, ...)
    return best_move
```

### Undo Stack Management

For recursive algorithms (MCTS, Descent), maintain an explicit or implicit undo stack:

```python
# Explicit stack for iterative implementation
undo_stack: List[MoveUndo] = []

# Implicit stack for recursive implementation
def recurse(state, depth):
    undo = state.make_move(move)
    result = recurse(state, depth - 1)
    state.unmake_move(undo)
    return result
```

### Zobrist Hash Consistency

The make/unmake pattern maintains Zobrist hash incrementally. Components should leverage this for transposition table lookups:

```python
# Hash is automatically updated during make/unmake
state_hash = mutable.zobrist_hash
cached = self.transposition_table.get(state_hash)
```

### Evaluation Function Adaptation

Evaluation functions that currently take `GameState` will need variants that work with `MutableGameState`:

```python
def evaluate_position(self, game_state: GameState) -> float:
    """Evaluate immutable state."""
    ...

def evaluate_position_mutable(self, state: MutableGameState) -> float:
    """Evaluate mutable state directly."""
    # Option 1: Convert to immutable (slower but compatible)
    return self.evaluate_position(state.to_immutable())

    # Option 2: Direct evaluation (faster but requires adaptation)
    return self._evaluate_mutable(state)
```

---

## Expected Total Performance Improvement

### Scenario: Full Tournament Run

Current time for 100-game tournament: ~60 minutes

With make/unmake integration:

- Phase 1 (MCTS/Descent): ~12-15 minutes (4-5x faster)
- Phase 2 (Infra): ~10-12 minutes (5-6x faster)

### Scenario: Self-Play Data Generation

Current time for 1000 games: ~8-10 hours

With make/unmake integration:

- Phase 1: ~2-3 hours (3-4x faster)
- Combined: ~1.5-2 hours (5-6x faster)

### Scenario: Live AI Response Time

Current move selection time (Descent, difficulty 10): ~500ms

With make/unmake integration:

- Expected: ~100-150ms (3-5x faster)

---

## Recommendations

1. **Start with MCTS AI** - Highest complexity but also highest potential impact for neural network-based play.

2. **Follow with Descent AI** - Similar structure to MinimaxAI, can leverage existing integration patterns.

3. **Self-play benefits automatically** once underlying AIs are updated.

4. **Consider a unified `select_move_incremental()` interface** across all tree-search AIs for consistency.

5. **Add benchmarking infrastructure** to measure actual speedups in each component.

6. **Implement feature flags** (like `use_incremental_search` in MinimaxAI) to enable A/B testing and gradual rollout.

---

## References

- [`ai-service/docs/MAKE_UNMAKE_DESIGN.md`](./MAKE_UNMAKE_DESIGN.md) - Original design document
- [`ai-service/app/rules/mutable_state.py`](../app/rules/mutable_state.py) - Implementation
- [`ai-service/app/ai/minimax_ai.py`](../app/ai/minimax_ai.py) - Reference integration
- [`ai-service/scripts/benchmark_make_unmake.py`](../scripts/benchmark_make_unmake.py) - Benchmarking tools
