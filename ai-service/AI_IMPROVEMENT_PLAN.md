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
| [`MinimaxAI`](app/ai/minimax_ai.py)     | 3–6               | Alpha-beta + PVS + Quiescence search   |
| [`MCTSAI`](app/ai/mcts_ai.py)           | 7–8               | PUCT + RAVE + tree reuse               |
| [`DescentAI`](app/ai/descent_ai.py)     | 9–10              | UBFM-style search with neural guidance |

### 1.3 MPS-Compatible Architecture (Apple Silicon)

For running on Apple Silicon (M1/M2/M3) Macs, an MPS-compatible variant of the neural network architecture is available. See [`docs/MPS_ARCHITECTURE.md`](./docs/MPS_ARCHITECTURE.md) for details.

| Architecture                                  | MPS Support | Pooling Method         | Use Case           |
| --------------------------------------------- | ----------- | ---------------------- | ------------------ |
| [`RingRiftCNN`](app/ai/neural_net.py:202)     | ❌ No       | `nn.AdaptiveAvgPool2d` | CUDA GPUs, CPU     |
| [`RingRiftCNN_MPS`](app/ai/neural_net.py:325) | ✅ Yes      | `torch.mean()`         | Apple Silicon GPUs |

**Usage:**

```bash
# Auto-select MPS architecture on Apple Silicon
export RINGRIFT_NN_ARCHITECTURE=auto
python scripts/run_self_play_soak.py ...
```

| [`NeuralNetAI`](app/ai/neural_net.py) | Backend for 7–10 | ResNet CNN with policy/value heads |

### 1.2 Canonical Difficulty Ladder & Product-Facing Profiles

The canonical 1–10 difficulty ladder is defined in `_CANONICAL_DIFFICULTY_PROFILES` in
`app/main.py` and mirrored by the TypeScript `AIEngine` presets. This section records
the intended _product-facing_ interpretation so lobby/matchmaking UIs and operators
have a shared understanding of what each band represents.

| Difficulty | Internal AI type(s)         | Profile ID (Python) | Suggested label / use case                                     |
| ---------- | --------------------------- | ------------------- | -------------------------------------------------------------- |
| 1          | `RandomAI`                  | `v1-random-1`       | **Beginner – Random**: sandbox/tutorial only, not for rating.  |
| 2          | `HeuristicAI`               | `v1-heuristic-2`    | **Beginner – Heuristic**: default “easy” AI vs human.          |
| 3–4        | `MinimaxAI`                 | `v1-minimax-3/4`    | **Intermediate – Minimax**: shallow search, casual play.       |
| 5–6        | `MinimaxAI`                 | `v1-minimax-5/6`    | **Challenging – Minimax**: deeper search, default “strong” AI. |
| 7–8        | `MCTSAI` + `NeuralNetAI`    | `v1-mcts-7/8`       | **Stronger Opponents – MCTS**: advanced/experimental ladder.   |
| 9–10       | `DescentAI` + `NeuralNetAI` | `v1-descent-9/10`   | **Stronger Opponents – Descent**: highest difficulty, beta.    |

Product-level guidance:

- Lobby / matchmaking UIs SHOULD:
  - Default AI games to **difficulty 2–5** for casual play.
  - Treat **7–10** as an **opt‑in “Stronger Opponents” band**, surfaced with
    explicit copy (e.g. “Experimental – may think slowly on large boards”).
- Operators SHOULD:
  - Avoid using 7–10 for time‑sensitive rating queues until aggregate latency
    and strength have been validated via scripts like
    `ai-service/scripts/evaluate_ai_models.py` and orchestrator soaks.

### 1.2 Verified Strengths (Issues Previously Resolved)

The following items were documented as issues but have been **verified as resolved**:

| Claimed Issue                        | Verification    | Evidence                                                                                                  |
| ------------------------------------ | --------------- | --------------------------------------------------------------------------------------------------------- |
| Minimax/MCTS not wired to difficulty | ✅ **RESOLVED** | [`main.py:683-754`](app/main.py:683) defines `_CANONICAL_DIFFICULTY_PROFILES` with proper AI type mapping |
| RNG uses global `random.seed(42)`    | ✅ **RESOLVED** | [`base.py:38-46`](app/ai/base.py:38) uses per-instance `random.Random(self.rng_seed)`                     |
| MCTS lacks tree reuse                | ✅ **RESOLVED** | [`mcts_ai.py:391-410`](app/ai/mcts_ai.py:391) implements subtree preservation via `last_root`             |
| Transposition tables unbounded       | ✅ **RESOLVED** | [`BoundedTranspositionTable`](app/ai/bounded_transposition_table.py) with memory limits                   |
| Minimax depth doesn't scale          | ✅ **RESOLVED** | [`minimax_ai.py:82-89`](app/ai/minimax_ai.py:82) scales `max_depth` 2→5 based on difficulty               |

### 1.3 Verified Weaknesses (Status Update)

The following issues have been addressed or are in progress:

#### 1.3.1 State Copying Bottleneck ✅ RESOLVED

**Status:** ✅ **COMPLETE** (as of 2025-11-30)

The make/unmake pattern has been fully implemented in:

- **MutableGameState** (`app/rules/mutable_state.py`): Full implementation with `make_move()`, `unmake_move()`, and incremental Zobrist hashing
- **MinimaxAI** (`app/ai/minimax_ai.py`): Uses `use_incremental_search` flag (default: True)
- **MCTSAI** (`app/ai/mcts_ai.py`): Uses `MCTSNodeLite` and `_search_incremental()`
- **DescentAI** (`app/ai/descent_ai.py`): Uses `_descent_iteration_mutable()`

**Impact:** 10-25x improvement in nodes/second, enabling deeper search at same time budget.

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

> \*\*Status (2025-11-30): Core make/unmake pattern implemented for MinimaxAI, MCTSAI, and DescentAI; remaining extension work is tracked in [`ai-service/docs/MAKE_UNMAKE_EXTENSION_ANALYSIS.md`](docs/MAKE_UNMAKE_EXTENSION_ANALYSIS.md). The original proposal below is retained as historical context and for guiding further extensions (RL environment, heuristic AI, and training infrastructure) rather than as a description of the current implementation gap.

**Current State:** The rules engine exposes a `MutableGameState` + `MoveUndo` make/unmake API, and the pattern has been integrated into MinimaxAI (initially) and extended to MCTSAI and DescentAI, yielding the documented speedups in [`MAKE_UNMAKE_EXTENSION_ANALYSIS.md`](docs/MAKE_UNMAKE_EXTENSION_ANALYSIS.md). Remaining opportunities are primarily:

- Extending make/unmake usage into the RL environment and selected training/self-play code paths.
- Evaluating whether HeuristicAI benefits from a lightweight make/unmake integration in high-branching positions.
- Tightening invariants and performance metrics around the existing make/unmake surfaces.

**Proposed Solution:**

1. Continue to rely on the existing `MutableGameState` + `MoveUndo` API as the canonical make/unmake surface.
2. Thread make/unmake through remaining high-value hosts (RL environment, selected training/self-play loops) where state copying is still a bottleneck.
3. Maintain and extend incremental Zobrist hash updates during make/unmake so that all search-oriented hosts share the same fast hashing path.

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

### Priority 6: Online Evaluation & Analysis Mode (Medium Impact, Medium Effort)

**Current State:**  
AI strength and latency are evaluated offline via `evaluate_ai_models.py` and related training scripts. A lightweight `/ai/evaluate_position` endpoint now exists in `ai-service/app/main.py` that returns per-player heuristic evaluations for a given `GameState`, and the Node backend can stream these as `position_evaluation` WebSocket events when analysis mode is enabled. Live games and replays still do not persist evaluation history, and the evaluation engine currently uses the heuristic profile rather than the strongest Descent/MCTS configurations.

**Goal:**  
Introduce an **opt-in Analysis Mode** that surfaces an evaluation panel in the client for selected games (primarily spectating and post-game review). The panel should:

- Show per-move evaluation history based on a strong engine (e.g. Descent+NN or deep Minimax).
- Expose current evaluation per player as an estimated win/loss margin (e.g. composite of territory advantage and eliminated-ring advantage).
- Use clear, color-coded indicators for which player is ahead and by how much, without affecting core matchmaking or rated game semantics.

**High-Level Plan:**

1. **Evaluation API in AI service**
   - Add `/ai/evaluate_position` in `ai-service/app/main.py` that:
     - Accepts a serialized `GameState` (using existing contract/serialization models).
     - Uses a fixed evaluation profile (e.g. DescentAI or MinimaxAI at bounded depth/think time) to compute per-player scores:
       - `totalEval` (scalar advantage), and optionally `territoryEval` and `ringEval`.
     - Returns a compact JSON structure suitable for streaming via WebSocket and persisting alongside move history.
   - Reuse `RingRiftEnv` and evaluation helpers already used by `evaluate_ai_models.py`, with deterministic RNG and hard time budgets for predictable runtime.

2. **Backend evaluation client & integration**
   - Add a thin client on the Node side (extend `AIServiceClient` or introduce a dedicated `PositionEvaluationClient`) that calls `/ai/evaluate_position` with:
     - Strict timeouts shorter than move-selection calls.
     - Concurrency caps so evaluation cannot starve gameplay-critical AI requests.
   - From `GameSession` / `GamePersistenceService`, trigger evaluation:
     - After each committed move (or on a sampled subset) for games flagged as “analysis-enabled”.
     - Store results keyed by `(gameId, moveNumber)` and optionally broadcast them via a new WebSocket event (e.g. `position_evaluation`).

3. **Persistence & history**
   - Persist evaluation snapshots per move using either:
     - A light DB table attached to move history, or
     - Redis/in-memory caches for short-lived analysis sessions.
   - Extend `/api/games/:gameId/history` to optionally return evaluation history so the client can reconstruct evaluation curves in replay views without re-querying the AI service.

4. **Client-side EvaluationPanel**
   - Extend `gameViewModels` / HUD view-models with:
     - `evaluationHistory: Array<{ moveNumber, evalByPlayer }>` and
     - `currentEvaluation: { perPlayer: { [playerNumber]: totalEval } }`.
   - Implement an `EvaluationPanel` React component that:
     - Renders a compact sparkline or bar chart over moves, color-coded by advantage.
     - Shows the current evaluation with breakdown (“P1 +3.2 (territory +2.0, rings +1.2)”).
     - Updates live when evaluation events arrive, and supports hover to inspect individual moves.
   - Integrate the panel into `BackendGameHost`:
     - Enabled by default for `/spectate/:gameId` and history/replay contexts.
     - Toggled for active players via an “Analysis” switch so it remains opt-in and non-distracting.

5. **Guardrails & observability**
   - Treat evaluation as best-effort:
     - If the AI evaluation service times out or fails, the panel should show “Analysis unavailable” without affecting gameplay.
   - Keep CI coverage focused and light:
     - Add Python tests validating `/ai/evaluate_position` determinism on fixed seeds.
     - Add a TS integration test that stubs evaluation responses and verifies EvaluationPanel wiring.
   - Monitor evaluation latency separately from `/ai/move` to avoid conflating analysis load with core move-generation health.

**Estimated Effort:** 4–7 days (Python + Node + client).  
**Expected Impact:**

- Stronger spectator and replay experience (especially for streams and coaching).
- Clearer visibility into the “Stronger Opponents” band by exposing AI assessments directly.
- A reusable evaluation primitive for future features (coach mode, hints, automated annotations).

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
