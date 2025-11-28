> **Doc Status (2025-11-27): Active (AI host improvement plan, Python service only)**
>
> - Role: prioritized technical improvement and performance plan for the Python AI microservice (agents, search patterns, training pipeline). It informs work on the AI host, but does not redefine game rules.
> - Not a semantics or lifecycle SSoT: for rules semantics and lifecycle / API contracts, defer to the shared TypeScript rules engine under `src/shared/engine/**`, the engine contracts under `src/shared/engine/contracts/**`, the v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`, [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md), [`ringrift_complete_rules.md`](../ringrift_complete_rules.md), [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md), [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md), and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - For current AI architecture and assessment, pair this with [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md), the technical assessment in [`AI_ASSESSMENT_REPORT.md`](./AI_ASSESSMENT_REPORT.md), and the training/meta docs [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md) and [`docs/AI_TRAINING_PREPARATION_GUIDE.md`](../docs/AI_TRAINING_PREPARATION_GUIDE.md).
> - Related docs: parity and invariants meta-docs such as [`docs/PYTHON_PARITY_REQUIREMENTS.md`](../docs/PYTHON_PARITY_REQUIREMENTS.md), [`docs/STRICT_INVARIANT_SOAKS.md`](../docs/STRICT_INVARIANT_SOAKS.md), [`tests/TEST_SUITE_PARITY_PLAN.md`](../tests/TEST_SUITE_PARITY_PLAN.md), [`docs/PARITY_SEED_TRIAGE.md`](../docs/PARITY_SEED_TRIAGE.md), and the historical plan in [`archive/AI_IMPROVEMENT_PLAN.md`](../archive/AI_IMPROVEMENT_PLAN.md).

# AI Service Improvement Plan

**Document Version:** 1.0  
**Date:** 2025-11-27  
**Status:** Analysis Complete, Ready for Implementation  
**Original Rating:** 3.2/5 (Per WEAKNESS_ASSESSMENT_REPORT.md)

---

## Executive Summary

This document provides a comprehensive analysis of the RingRift AI Service, identifies verified weaknesses after code review, and proposes prioritized improvements. Several issues originally documented have been addressed, while new findings emerged from direct code inspection.

---

## 1. Current State Assessment

### 1.1 Architecture Overview

The AI Service implements a multi-tier difficulty system with six AI implementations:

| AI Type                                 | Difficulty Levels | Key Features                           |
| --------------------------------------- | ----------------- | -------------------------------------- |
| [`RandomAI`](app/ai/random_ai.py)       | 1                 | Pure random move selection             |
| [`HeuristicAI`](app/ai/heuristic_ai.py) | 2                 | 17+ weighted evaluation factors        |
| [`MinimaxAI`](app/ai/minimax_ai.py)     | 3-6               | Alpha-beta + PVS + Quiescence search   |
| [`MCTSAI`](app/ai/mcts_ai.py)           | 7-8               | PUCT + RAVE + tree reuse               |
| [`DescentAI`](app/ai/descent_ai.py)     | 9-10              | UBFM-style search with neural guidance |
| [`NeuralNetAI`](app/ai/neural_net.py)   | Backend for 7-10  | ResNet CNN with policy/value heads     |

### 1.2 Verified Strengths (Issues Previously Resolved)

The following items were documented as issues but have been **verified as resolved**:

| Claimed Issue                        | Verification    | Evidence                                                                                                  |
| ------------------------------------ | --------------- | --------------------------------------------------------------------------------------------------------- |
| Minimax/MCTS not wired to difficulty | ✅ **RESOLVED** | [`main.py:683-754`](app/main.py:683) defines `_CANONICAL_DIFFICULTY_PROFILES` with proper AI type mapping |
| RNG uses global `random.seed(42)`    | ✅ **RESOLVED** | [`base.py:38-46`](app/ai/base.py:38) uses per-instance `random.Random(self.rng_seed)`                     |
| MCTS lacks tree reuse                | ✅ **RESOLVED** | [`mcts_ai.py:391-410`](app/ai/mcts_ai.py:391) implements subtree preservation via `last_root`             |
| Transposition tables unbounded       | ✅ **RESOLVED** | [`BoundedTranspositionTable`](app/ai/bounded_transposition_table.py) with memory limits                   |
| Minimax depth doesn't scale          | ✅ **RESOLVED** | [`minimax_ai.py:82-89`](app/ai/minimax_ai.py:82) scales `max_depth` 2→5 based on difficulty               |

### 1.3 Verified Weaknesses (Still Present)

The following issues were **verified as present** after code inspection:

#### 1.3.1 State Copying Bottleneck (Critical)

**Location:** [`minimax_ai.py:137-141`](app/ai/minimax_ai.py:137)

```python
# Use apply_move (which now returns a new state).
# This is a known bottleneck; in future we should
# switch to make_move/undo_move on a mutable board.
next_state = self.rules_engine.apply_move(game_state, move)
```

**Impact:** Every node in the search tree creates a full state copy. For Minimax at depth 5 with branching factor ~30, this means ~24M state copies per search.

**Current Workaround:** None. The code contains a TODO comment acknowledging this.

#### 1.3.2 Neural Network Architecture Mismatch

**Location:** [`neural_net.py:356-383`](app/ai/neural_net.py:356)

```python
try:
    self.model.load_state_dict(
        torch.load(model_path, map_location=self.device, weights_only=True)
    )
except RuntimeError as e:
    # Architecture mismatch
    print(f"Could not load model (architecture mismatch): {e}. Starting with fresh weights.")
```

**Impact:** When model architecture changes (e.g., history_length, num_res_blocks), saved weights become incompatible. The AI silently falls back to untrained random weights.

**Evidence:** The model uses `history_length=3` and `num_res_blocks=10`, but there's no version checking for saved checkpoints.

#### 1.3.3 Zobrist Hash Computation on Demand

**Location:** [`minimax_ai.py:161-165`](app/ai/minimax_ai.py:161), [`zobrist.py:101-126`](app/ai/zobrist.py:101)

```python
def _get_state_hash(self, game_state: GameState) -> int:
    if game_state.zobrist_hash is not None:
        return game_state.zobrist_hash
    return self.zobrist.compute_initial_hash(game_state)  # O(N) if not cached
```

**Impact:** When `zobrist_hash` is not pre-computed on the GameState, each hash computation iterates over all stacks, markers, and collapsed spaces (O(N)).

#### 1.3.4 Training Self-Play Diversity

**Location:** [`train_loop.py:79-106`](app/training/train_loop.py:79)

```python
ai1 = DescentAI(1, AIConfig(difficulty=5, think_time=500, randomness=0.1, rngSeed=config.seed))
ai2 = DescentAI(2, AIConfig(difficulty=5, think_time=500, randomness=0.1, rngSeed=config.seed))
```

**Impact:** Both self-play players use the same seed. While they play different sides, the deterministic behavior may limit exploration diversity.

#### 1.3.5 Hex-Specific Model Not Trained

**Location:** [`neural_net.py:1390-1519`](app/ai/neural_net.py:1390)

The `HexNeuralNet` class exists with proper architecture, but:

- No trained weights file exists for hex boards
- Self-play training in [`train_loop.py`](app/training/train_loop.py) defaults to square boards
- Hex-specific action encoder [`ActionEncoderHex`](app/ai/neural_net.py:1161) is implemented but unused in training

---

## 2. Top 5 Improvement Priorities

### Priority 1: Implement Make/Unmake Move Pattern (High Impact, Medium Effort)

**Current State:** Every search node creates a full GameState copy via `apply_move()`.

**Proposed Solution:**

1. Add `make_move(state, move)` and `unmake_move(state, move)` to the rules engine
2. Track move deltas (captured rings, placed markers, collapsed spaces) for efficient undo
3. Maintain incremental Zobrist hash updates during make/unmake

**Estimated Effort:** 3-5 days

**Expected Impact:**

- 10-50x speedup in search throughput
- Deeper search at same time budget
- Reduced memory allocation pressure

**Implementation Sketch:**

```python
class IncrementalRulesEngine:
    def make_move(self, state: GameState, move: Move) -> MoveDelta:
        delta = MoveDelta()
        # Apply move, record changes in delta
        # Update state.zobrist_hash incrementally
        return delta

    def unmake_move(self, state: GameState, delta: MoveDelta):
        # Undo changes from delta
        # Restore state.zobrist_hash from delta
```

---

### Priority 2: Neural Network Model Versioning (Medium Impact, Low Effort)

**Current State:** No version metadata in saved checkpoints; architecture changes silently break loading.

**Proposed Solution:**

1. Add model metadata to checkpoint files:
   ```python
   torch.save({
       'version': '1.0',
       'architecture': {'history_length': 3, 'num_res_blocks': 10, 'num_filters': 128},
       'state_dict': model.state_dict()
   }, path)
   ```
2. Validate architecture compatibility before loading
3. Support migration scripts for weight conversion when architecture changes

**Estimated Effort:** 1 day

**Expected Impact:**

- Prevents silent fallback to random weights
- Enables safe architecture evolution
- Better debugging of model loading failures

---

### Priority 3: Hex Board Training Pipeline (High Impact, Medium Effort)

**Current State:** Hex-specific neural network exists but is never trained.

**Proposed Solution:**

1. Extend [`train_loop.py`](app/training/train_loop.py) to support `board_type` parameter
2. Create hex-specific tournament and self-play configurations
3. Generate hex-specific training data with proper augmentation (6-fold rotational symmetry)
4. Train and validate hex model separately from square model

**Estimated Effort:** 3-4 days

**Expected Impact:**

- Enables neural-guided AI for hex boards
- Improves difficulty 7-10 play quality on hex
- Leverages existing [`HexNeuralNet`](app/ai/neural_net.py:1390) infrastructure

**Configuration Example:**

```python
# train_loop.py
TrainConfig(
    board_type=BoardType.HEXAGONAL,
    model_id="ringrift_hex_v1",
    episodes_per_iter=100,
)
```

---

### Priority 4: Self-Play Diversity Enhancement (Medium Impact, Low Effort)

**Current State:** Both self-play players share the same RNG seed.

**Proposed Solution:**

1. Use distinct seeds for each player: `seed + player_number`
2. Add temperature-based move sampling during self-play (not just tournament)
3. Implement strength asymmetry: occasional matches between different difficulties

**Estimated Effort:** 0.5 days

**Expected Impact:**

- More diverse training data
- Better exploration of game tree
- Reduced overfitting to specific play patterns

**Implementation:**

```python
ai1 = DescentAI(1, AIConfig(rngSeed=config.seed + 1))
ai2 = DescentAI(2, AIConfig(rngSeed=config.seed + 2))
```

---

### Priority 5: Incremental Zobrist Hash Maintenance (Medium Impact, Medium Effort)

**Current State:** [`apply_move()`](app/game_engine.py) creates new states but doesn't always pre-compute Zobrist hash.

**Proposed Solution:**

1. Ensure `GameEngine.apply_move()` always computes and caches `zobrist_hash` on returned state
2. Use incremental XOR updates instead of full recomputation:
   ```python
   new_hash = old_hash ^ get_old_piece_hash(pos) ^ get_new_piece_hash(pos)
   ```
3. Thread Zobrist updates through all state mutation paths

**Estimated Effort:** 2 days

**Expected Impact:**

- O(1) hash computation for transposition table lookups
- Improved cache hit rates in Minimax/MCTS
- More efficient Descent search

---

## 3. Implementation Roadmap

### Phase 1: Quick Wins (Week 1)

| Task                       | Priority     | Effort | Impact |
| -------------------------- | ------------ | ------ | ------ |
| Model versioning           | P2           | 1d     | Medium |
| Self-play diversity        | P4           | 0.5d   | Medium |
| Zobrist hash on apply_move | P5 (partial) | 1d     | Medium |

### Phase 2: Core Infrastructure (Week 2-3)

| Task                     | Priority | Effort | Impact |
| ------------------------ | -------- | ------ | ------ |
| Make/unmake move pattern | P1       | 3-5d   | High   |
| Full incremental Zobrist | P5       | 1d     | Medium |

### Phase 3: Training Pipeline (Week 3-4)

| Task                        | Priority | Effort | Impact |
| --------------------------- | -------- | ------ | ------ |
| Hex training pipeline       | P3       | 3-4d   | High   |
| Train initial hex model     | -        | 2-3d   | High   |
| Validate hex model strength | -        | 1d     | Medium |

---

## 4. Metrics for Success

### 4.1 Performance Metrics

- **Search throughput:** Nodes/second in Minimax (target: 10x improvement after P1)
- **Memory efficiency:** Peak memory during search (target: 50% reduction)
- **Cache hit rate:** Transposition table hits vs misses (target: >60%)

### 4.2 Playing Strength Metrics

- **Win rate vs baseline:** New model vs current model at same difficulty
- **Tournament ELO:** Rating in controlled tournament environment
- **Hex-specific strength:** Difficulty 10 win rate vs difficulty 7 (target: >80%)

### 4.3 Training Metrics

- **Data diversity:** Unique positions per self-play game (target: >50)
- **Model stability:** Win rate variance across training runs (target: <5%)
- **Checkpoint compatibility:** Zero silent fallbacks to random weights

---

## 5. Risk Assessment

| Risk                                 | Likelihood | Impact | Mitigation                                   |
| ------------------------------------ | ---------- | ------ | -------------------------------------------- |
| Make/unmake breaks state consistency | Medium     | High   | Extensive parity tests against apply_move    |
| Hex model training diverges          | Low        | Medium | Use proven square training hyperparameters   |
| Performance regression in live play  | Low        | High   | A/B testing with metrics before full rollout |
| Zobrist collisions increase          | Very Low   | Low    | Use 64-bit hashes, monitor collision rate    |

---

## 6. Conclusion

The AI Service is more robust than initially documented, with several claimed issues already resolved. The remaining improvements focus on:

1. **Performance:** Make/unmake pattern and incremental Zobrist (P1, P5)
2. **Reliability:** Model versioning (P2)
3. **Coverage:** Hex training pipeline (P3)
4. **Quality:** Self-play diversity (P4)

Implementing these changes would raise the AI Service rating from 3.2/5 to an estimated **4.2-4.5/5**, making it competitive with the strongest components in the RingRift system.

---

## Appendix: Verified Code References

| Component             | File                                                                      | Key Lines |
| --------------------- | ------------------------------------------------------------------------- | --------- |
| Difficulty ladder     | [`main.py`](app/main.py)                                                  | 683-754   |
| Per-instance RNG      | [`base.py`](app/ai/base.py)                                               | 38-46     |
| State copy bottleneck | [`minimax_ai.py`](app/ai/minimax_ai.py)                                   | 137-141   |
| Tree reuse            | [`mcts_ai.py`](app/ai/mcts_ai.py)                                         | 391-410   |
| Minimax depth scaling | [`minimax_ai.py`](app/ai/minimax_ai.py)                                   | 82-89     |
| Neural net loading    | [`neural_net.py`](app/ai/neural_net.py)                                   | 356-383   |
| Zobrist computation   | [`zobrist.py`](app/ai/zobrist.py)                                         | 101-126   |
| Self-play config      | [`train_loop.py`](app/training/train_loop.py)                             | 79-106    |
| Hex neural net        | [`neural_net.py`](app/ai/neural_net.py)                                   | 1390-1519 |
| Bounded transposition | [`bounded_transposition_table.py`](app/ai/bounded_transposition_table.py) | All       |
