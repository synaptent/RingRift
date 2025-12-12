> **Doc Status (2025-12-04): Active (AI host improvement plan, Python service only)**
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

As of the P19B/P20 passes, a small set of **late-game heuristic validation tests** has been added (`ai-service/tests/test_heuristic_ai.py`) to lock in expected preferences around:

- Near-victory elimination and territory progress (`_evaluate_victory_proximity` and full `evaluate_position`).
- Chain-capture-style overtake potential and nearby territory structure ( `_evaluate_overtake_potential`, `_evaluate_territory_closure`, `_evaluate_territory_safety`).

These tests are designed to validate the existing v1 balanced profile rather than to redefine semantics. They serve as the safety net for future weight training/tuning runs.

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

#### 1.3.6 Self-Play Game Recording (Track 11)

**Status (2025-12-04):** ✅ **SUBSTANTIALLY COMPLETE** (see `IMPROVEMENT_PLAN.md` §10.3)

**Current State:** CMA‑ES optimisation and other training/self‑play harnesses now record games by default to `GameReplayDB` SQLite databases:

- `ai-service/app/db/recording.py` provides the canonical recording helpers:
  - Environment controls: `RINGRIFT_RECORD_SELFPLAY_GAMES` (global on/off, default enabled) and `RINGRIFT_SELFPLAY_DB_PATH` (default DB path when none is supplied).
  - Helpers: `should_record_games(...)`, `get_or_create_db(...)`, `record_completed_game(...)`, and `GameRecorder` for incremental recording.
- `run_cmaes_optimization.py`:
  - Enables recording by default (`record_games=True`) and gates it via `should_record_games(cli_no_record=not config.record_games)` and a `--no-record` flag.
  - Creates a per‑run DB at `{run_dir}/games.db` and records all evaluation games with rich metadata (`source="cmaes"`, board, num_players, run_id, generation, candidate index, and any extra tags from `recording_context`).
- Other self‑play / evaluation scripts (for example `run_self_play_soak.py` and multi‑player evaluation helpers) share the same `GameReplayDB` + `record_completed_game(...)` surface when recording is enabled.
- `ai-service/scripts/export_state_pool.py` extracts mid‑game states from one or more such DBs and writes JSONL evaluation pools that are consumed by `app.training.eval_pools.load_state_pool(...)` and the CMA‑ES/GA fitness harnesses.

**Remaining Gaps:**

- No first‑class `merge_game_dbs.py` utility yet for combining many small per‑run DBs into larger corpora.
- No centralised manifest/registry for long‑lived state pools (currently defined ad‑hoc via `POOL_PATHS` in `app.training.eval_pools`).

**Follow‑up Direction:** Keep using `GameReplayDB` + `export_state_pool.py` as the canonical self‑play recording and evaluation‑pool surface, and consider:

1. Adding a small `merge_game_dbs.py` utility for long‑running experiments and shared corpora.
2. Promoting a stable set of evaluation pools (paths + metadata) into configuration and documentation (see `docs/ai/AI_TRAINING_AND_DATASETS.md` §5.4 and §6).

#### 1.3.7 Canonical Self-Play + Parity Gate (New)

**Status (2025-12-05):** ✅ **INITIAL IMPLEMENTATION COMPLETE**

**Current State:** We now have an end-to-end driver that:

- Runs a small Python self‑play soak using the canonical GameEngine for a given board type.
- Records completed games into a fresh `GameReplayDB` SQLite file.
- Invokes the TS↔Python replay parity harness on that DB.
- Emits a concise JSON summary indicating whether the DB passes a **canonical parity gate**:
  - `games_with_structural_issues == 0`, and
  - `games_with_semantic_divergence == 0`.

**Key Script:**

- `ai-service/scripts/run_canonical_selfplay_parity_gate.py`
  - CLI:
    ```bash
    # From ai-service/
    PYTHONPATH=. python scripts/run_canonical_selfplay_parity_gate.py \
      --board-type square8 \
      --num-games 20 \
      --db data/games/selfplay_square8_parity_gate.db \
      --summary parity_gate.square8.json
    ```
  - Parameters:
    - `--board-type`: `square8`, `square19`, or `hexagonal`.
    - `--num-games`: number of self-play games to run (default: 20).
    - `--db`: path to the GameReplayDB to create/write.
    - `--seed`: base RNG seed (default: 42).
    - `--max-moves`: max moves per game before forced termination (default: 200).
    - `--summary`: optional path for the parity gate JSON summary; when omitted, the summary is printed to stdout only.
  - Behaviour:
    - Internally calls `scripts/run_self_play_soak.py` with:
      - `RINGRIFT_STRICT_NO_MOVE_INVARIANT=1`,
      - `--engine-mode mixed`, `--num-players 2`.
    - Then calls `scripts/check_ts_python_replay_parity.py --db <db>`.
    - Sets `passed_canonical_parity_gate = true` iff:
      - The parity harness returns successfully, and
      - Both `games_with_structural_issues` and `games_with_semantic_divergence` are zero.

**Recommended Usage in Training Pipelines:**

- Before training on any new self‑play corpus for a given board type:
  1. Run the canonical parity gate script with a modest `--num-games` (e.g. 20–50) and inspect the summary for:
     - Structural issues (should be 0).
     - Any unexpected semantic divergences.
  2. Only promote DB paths to the **training allowlist** if they either:
     - Pass the parity gate cleanly, or
     - Have known, locally-suppressed bookkeeping-only differences (e.g. agreed phase bookkeeping) that are explicitly documented.
- Over time, extend this gate to:
  - Cover multiple player counts (2p/3p/4p) per board type.
  - Run as part of CI for rules-engine changes that may affect self‑play.

#### 1.3.8 Training Data Hygiene & Registry (New)

**Status (2025-12-05):** ✅ **DOCUMENTED**

**Current State:** A **Training Data Registry** now exists at [`TRAINING_DATA_REGISTRY.md`](./TRAINING_DATA_REGISTRY.md) that:

- Classifies all game replay databases as **canonical**, **legacy_noncanonical**, or **pending_gate**.
- Documents model provenance: which training data each `.pth` checkpoint was trained on.
- Defines the **Training Data Allowlist** policy: new training must only use parity-gated DBs.

**Key Classifications:**

| Classification        | Meaning                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `canonical`           | Passes `run_canonical_selfplay_parity_gate.py`; safe for training |
| `legacy_noncanonical` | Pre-dates line-length/territory/parity fixes; DO NOT use          |
| `pending_gate`        | Not yet validated; requires parity gate before use                |

**Legacy Data (Not For New Training):**

All self-play databases generated before December 2025 parity fixes are **legacy_noncanonical**:

- `selfplay_square8_2p.db`, `selfplay_square19_*.db`, `selfplay_hexagonal_*.db`
- `selfplay.db`, `square8_2p.db`, `minimal_test.db`

These pre-date:

- Line-length validation fix (RR-CANON-R120)
- Explicit line decision flow (RR-CANON-R121-R122)
- All turn actions/skips explicit (RR-CANON-R074)
- Forced elimination player rotation fixes

**Canonical Data (For New Training):**

Fresh parity-gated DBs:

- `canonical_square8.db` (gate: `parity_gate.square8.json`)
- `canonical_square19.db` (gate: `parity_gate.square19.json`)
- `canonical_hex.db` — **removed (old radius-10 geometry)**; regenerate for radius-12 hex before gating.

**Action Items:**

1. Move legacy DBs to `data/games/legacy/` directory.
2. Move legacy models to `models/legacy/` directory.
3. Retrain v2 models from canonical DBs only.
4. Update training scripts to enforce allowlist checks.

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

---

## 7. Neural Network Training Pipeline Improvements

> **Status (2025-12-05):** Implementation plans finalized. These improvements target the training data generation and neural network architecture for better policy/value learning.

### 7.1 Already Implemented (v1.x, scalar value head)

The following pipeline improvements are implemented and in active use:

| Improvement                        | Status      | Files Modified                                              | Notes                                                            |
| ---------------------------------- | ----------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Rank-aware value targets           | ✅ COMPLETE | `scripts/export_replay_dataset.py`                          | Scalar in `[-1,1]` from current player POV, rank-aware for 3–4p. |
| Replay quality filtering           | ✅ COMPLETE | `scripts/export_replay_dataset.py`                          | Filters by `termination_reason` and move-count thresholds.       |
| Robust territory / choice encoding | ✅ COMPLETE | `app/ai/neural_net.py`                                      | Encodes pie rule, line/territory options; `POLICY_SIZE`=67k.     |
| Late-game / phase-aware sampling   | ✅ COMPLETE | `scripts/export_replay_dataset.py`, `app/training/train.py` | Weighted sampling via metadata + `WeightedRingRiftDataset`.      |

**Rank-aware value targets (scalar v1):** For multiplayer games (3–4 players), the exporter uses a rank-based scalar value from the current player’s perspective:

- 2-player: +1/-1 (winner/loser, unchanged)
- 3-player: +1/0/-1 (1st/2nd/3rd)
- 4-player: +1/+0.33/-0.33/-1 (1st/2nd/3rd/4th)

This is implemented in `value_from_final_ranking(...)` and is compatible with the existing scalar value head in `RingRiftCNN`.

**Robust territory / choice encoding and unified action space:**

- The generic policy head now covers:
  - Pie-rule `swap_sides` (meta-move),
  - Legacy line/territory moves (`line_formation`, `territory_claim`),
  - Canonical choice moves (`choose_line_option`, `choose_territory_option`) with extra structure (position, size bucket, controlling player).
- The module-level `POLICY_SIZE` constant is 67,000, and all policy heads use it consistently. Older 55k checkpoints are treated as legacy and require retraining.

**Late-game / phase-aware sampling:**

- `scripts/export_replay_dataset.py` adds per-sample metadata:
  - `move_numbers`, `total_game_moves`, `phases`.
- `WeightedRingRiftDataset` in `app/training/train.py` computes per-sample weights using this metadata for four strategies:
  - `uniform`, `late_game`, `phase_emphasis`, `combined`.
- Training CLI exposes `--sampling-weights` to select a strategy. This is a cheap, robust bias toward sharper, decision-heavy positions without changing the core loss.

### 7.2 Remaining Improvements – Future-Facing Design (v2.x+)

#### 7.2.1 Search-Based Training Targets (MCTS Visit Counts → Soft Policy)

**Priority:** HIGH | **Effort:** LOW | **Impact:** Better policy learning from search

**Current State:**

- For replay-based datasets (`export_replay_dataset.py`), policy targets are sparse 1‑hots derived from the actual move played.
- For search-based datasets (`generate_data.py`), the DescentAI path already builds soft targets from search statistics (transposition table) but is Descent‑specific and more complex than necessary.
- `MCTSAI` already exposes visit distributions via `get_visit_distribution()`, and `extract_mcts_visit_distribution(...)` exists in `app/training/generate_data.py`, but this is not yet wired into the main data generator.

**Goal:** Make MCTS search the canonical source for high-quality soft policy targets, while keeping DescentAI available as a complementary engine. Provide a clean flag in the data generator to choose the engine per run.

**Refined Design:**

1. **Engine selection flag in `generate_data` CLI** (`app/training/generate_data.py`):
   - Add `--engine {descent,mcts}` with:
     - Default: `descent` (matches current behaviour and uses DescentAI).
     - `mcts`: uses `MCTSAI` as the self-play engine.
   - Internally, `generate_dataset(...)` becomes engine-agnostic:
     - Creates either DescentAI or MCTSAI instances for each player, based on the flag.

2. **Canonical MCTS soft policy extraction** (already partially implemented):
   - Reuse `extract_mcts_visit_distribution(ai, state, encoder)`:
     - Calls `ai.get_visit_distribution()` (works for both legacy and incremental MCTS).
     - Encodes moves via `encoder.encode_move(...)` into sparse `policy_indices` and `policy_values` with probabilities proportional to visit counts.
   - When `--engine mcts`:
     - After `move = ai.select_move(state)`, call `extract_mcts_visit_distribution(...)` to generate policy targets for that root state.
     - Use the scalar value target as either:
       - Final outcome (v1‑style) for stability, or
       - A smoothed blend of root value estimate and final outcome (v2 option).

3. **DescentAI path as complementary search-based engine:**
   - Keep the current DescentAI path for:
     - Value‑only auxiliary targets derived from the search tree (`search_data`).
     - Soft policies based on child values where helpful.
   - Clearly separate the two modes in code and documentation:
     - `--engine descent`: search‑guided self-play with Descent-specific features.
     - `--engine mcts`: canonical AlphaZero-style pipeline with visit-based policies.

4. **Dataset format:** No schema change required:
   - Both engines produce sparse `policy_indices` + `policy_values`.
   - Soft policies (MCTS or Descent) and 1‑hots are all compatible with the existing `KLDivLoss` training path.

---

#### 7.2.2 Curriculum / Bootstrapping Training Loop

**Priority:** HIGH | **Effort:** MEDIUM | **Impact:** Continuous improvement loop

**Current State:** Training typically uses a single generation of self-play data (or a small number of manually curated datasets) without a formal notion of “generations” or model promotion criteria.

**Goal:** Introduce a lightweight but principled self-play curriculum:

1. Generate self-play data with the current “champion” model.
2. Train a candidate model on a mixture of recent and historical data.
3. Evaluate candidate vs. champion under a fixed match protocol.
4. Promote the candidate if it exceeds a target win rate (e.g. 55%).
5. Repeat for N generations, with logging and checkpointing.

**Implementation Steps:**

Status: ✅ A first implementation exists in `app/training/curriculum.py` with a
`CurriculumTrainer` class, JSON history, and a small CLI. The sketch below
captures the original design shape and remains a useful reference.

1. **Create curriculum training module** (`app/training/curriculum.py`):

   ```python
   @dataclass
   class CurriculumConfig:
       generations: int = 10
       games_per_generation: int = 1000
       training_epochs: int = 20
       eval_games: int = 100
       promotion_threshold: float = 0.55  # Win rate needed to promote
       data_retention: int = 3  # Generations of data to keep

   class CurriculumTrainer:
       def __init__(self, config: CurriculumConfig, base_model_path: str):
           self.config = config
           self.base_model_path = base_model_path
           self.current_generation = 0
           self.history = []

       def run_generation(self) -> dict:
           """Run one generation of curriculum training."""
           # 1. Generate self-play data
           data_path = self._generate_self_play_data()

           # 2. Combine with historical data
           combined_data = self._combine_with_history(data_path)

           # 3. Train candidate model
           candidate_path = self._train_candidate(combined_data)

           # 4. Evaluate against current best
           eval_result = self._evaluate_candidate(candidate_path)

           # 5. Promote if improved
           if eval_result['win_rate'] >= self.config.promotion_threshold:
               self._promote_candidate(candidate_path)
               self.current_generation += 1

           return {'generation': self.current_generation, ...}
   ```

2. **Add CLI integration** (`app/training/train.py`):

   ```python
   parser.add_argument('--curriculum', action='store_true')
   parser.add_argument('--curriculum-generations', type=int, default=10)
   ```

3. **Implement data mixing**:
   - Keep last N generations of data
   - Weight recent data higher (exponential decay)

4. **Implement model evaluation**:
   - Use existing `run_model_vs_model_eval()` or similar
   - Track win rate, draw rate, and average game length

---

#### 7.2.3 Per-Player Value Head (Vector Output for Multiplayer)

**Priority:** HIGH | **Effort:** MEDIUM | **Impact:** Fundamental for multiplayer games

**Current State:**

- v1 model (`RingRiftCNN`) outputs a single scalar in `[-1, +1]` from the current player’s perspective and is still the default for training.
- A v2 multi-player architecture (`RingRiftCNN_MultiPlayer`) and helper loss
  (`multi_player_value_loss`) already exist in `app/ai/neural_net.py`.
- Replay exporters compute richer, rank‑aware scalars per position today, and
  `compute_multi_player_values` in `scripts/export_replay_dataset.py` provides
  the per‑player vector encoding, but v2 datasets and training wiring have not
  yet been switched over to use vector value targets end‑to‑end.

**Goal:** Migrate to a vector value head that predicts an outcome or utility for _each_ player simultaneously, while keeping the v1 scalar head supported for backwards compatibility and incremental rollout.

**Implementation Steps:**

1. **Add a multi-player value variant** (`app/ai/neural_net.py`):

   ```python
   class RingRiftCNN_MultiPlayer(nn.Module):
       """CNN with per-player value head for N-player games."""

       ARCHITECTURE_VERSION = "v2.0.0"

       def __init__(
           self,
           max_players: int = 4,
           board_size: int = 8,
           in_channels: int = 10,
           global_features: int = 10,
           num_res_blocks: int = 10,
           num_filters: int = 128,
           history_length: int = 3,
       ):
           super().__init__()
           self.max_players = max_players
           # ... existing backbone ...

           # Multi-player value head
           self.value_head = nn.Linear(256, max_players)
           self.tanh = nn.Tanh()

       def forward(self, features, globals_vec):
           # ... backbone forward ...

           # Value head: (batch, max_players)
           values = self.tanh(self.value_head(x))

           # Policy head unchanged
           policy = self.policy_head(x)

           return values, policy
   ```

2. **Update training loss** (`app/training/train.py`):

   ```python
   def multi_player_value_loss(pred_values, target_values, num_players):
       """MSE loss only over active players."""
       mask = torch.zeros_like(target_values)
       mask[:, :num_players] = 1.0
       mse = ((pred_values - target_values) ** 2) * mask
       return mse.sum() / mask.sum()
   ```

3. **Update value target generation** (`scripts/export_replay_dataset.py`):

   ```python
   def compute_multi_player_values(final_state: GameState) -> List[float]:
       """Compute value for each player position."""
       values = [0.0] * 4  # max 4 players
       num_players = len(final_state.players)

       for player in final_state.players:
           rank = compute_rank(player, final_state)
           value = 1.0 - 2.0 * (rank - 1) / (num_players - 1)
           values[player.number - 1] = value

       return values
   ```

4. **Dataset format migration:**
   - For v2+ data, store values as shape `(N, max_players)` plus a per-sample `num_players` array.
   - Training code can then:
     - Use vector loss (masked) when `values.ndim == 2`.
     - Fall back to scalar loss when `values.ndim == 1`.
   - This allows mixed datasets (legacy scalar + new vector) during transition.

---

#### 7.2.4 Move Sampling Weights (Late-Game / Phase-Aware Sampling)

**Priority:** MEDIUM | **Effort:** LOW | **Impact:** Better data utilization

**Status:** ✅ COMPLETE for single-file (non-streaming) training.

**Current State / Design:**

- Exporter writes `move_numbers`, `total_game_moves`, `phases` alongside the usual tensors.
- `WeightedRingRiftDataset` encapsulates weighting logic and can be dropped into existing training flows.
- Training CLI exposes `--sampling-weights` with four options:
  - `uniform` (default), `late_game`, `phase_emphasis`, `combined`.

**Future Refinements:**

- Extend weighting to streaming mode (multiple files via `StreamingDataLoader`) by:
  - Either enriching the streaming path with the same metadata, or
  - Using per-file weighting heuristics (e.g., more weight to datasets from higher-quality self-play).
- Experimentally tune phase weights based on empirical sensitivity (e.g., emphasise capture/territory phases more on square19 than square8).

---

### 7.3 ROI Summary

| Rank | Improvement           | ROI  | Effort | Impact                              |
| ---- | --------------------- | ---- | ------ | ----------------------------------- |
| 1    | Search-based targets  | HIGH | LOW    | Stronger policy learning from MCTS  |
| 2    | Per-player value head | HIGH | MEDIUM | Correct multiplayer value semantics |
| 3    | Curriculum training   | HIGH | MEDIUM | Continuous, automated improvement   |
| 4    | Move sampling weights | MED  | LOW    | Better utilisation of existing data |

### 7.4 Implementation Order

1. **Search-based targets (MCTS-first, with engine flag)** – immediate, high leverage.
2. **Per-player value head + vector targets** – unlocks principled multiplayer training.
3. **Curriculum training loop** – wraps self-play, training, and eval into a repeatable pipeline.
4. **Move sampling weights** – already implemented for non-streaming; extend/tune as needed.

### 7.5 Files to Modify

| File                               | Changes                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `app/ai/mcts_ai.py`                | Keep visit distribution API stable; minor cleanup/docs only.                |
| `app/training/generate_data.py`    | Add `--engine {descent,mcts}` and wire `extract_mcts_visit_distribution`.   |
| `app/ai/neural_net.py`             | Introduce `RingRiftCNN_MultiPlayer` or equivalent, value-head refactor.     |
| `app/training/train.py`            | Add multi-player value loss, handle vector/scalar values, curriculum hooks. |
| `app/training/curriculum.py`       | New module implementing generation loop and model promotion.                |
| `scripts/export_replay_dataset.py` | Extend to vector value targets + `num_players` metadata for v2.             |

### 7.6 Backwards Compatibility Notes

- **POLICY_SIZE change (55000 → 67000):** Models trained with old policy size are incompatible. Retraining required.
- **Multi-player value head (future v2.x):** Introducing a vector value head will require a new architecture version (e.g. `v2.0.0`) and dedicated checkpoints. v1 scalar checkpoints remain usable on scalar-only training runs.
- **Dataset format:** New metadata fields (`move_numbers`, `total_game_moves`, `phases`) are optional; older datasets still train with uniform sampling. When vector values are introduced, training code must handle both scalar and vector `values` arrays during a transition period.

---

## 8. Board-Specific Neural Network Models

> **Status (2025-12-05):** ✅ **COMPLETE** - Board-specific policy sizes, model factory functions, and training configurations implemented.

### 8.1 Overview

Each board type has a distinct action space size, and using a single large policy head (67K parameters) for small boards wastes computational resources and may hinder learning. This section documents the board-specific model configurations that optimize for each board type.

### 8.2 Board-Specific Policy Sizes

| Board Type           | Spatial Size | Policy Size | Description                               |
| -------------------- | ------------ | ----------- | ----------------------------------------- |
| **SQUARE8** (8×8)    | 8            | **7,000**   | Compact action space for fast training    |
| **SQUARE19** (19×19) | 19           | **67,000**  | Full action space with territory encoding |
| **HEXAGONAL** (N=10) | 21           | **54,244**  | Hex-specific with 6-direction movement    |

**Policy Layout for 8×8:**

```
Placement:       3 * 8 * 8 = 192
Movement:        8 * 8 * 8 * 7 = 3,584  (8 directions, max 7 distance)
Line Formation:  8 * 8 * 4 = 256
Territory Claim: 8 * 8 = 64
Skip Placement:  1
Swap Sides:      1
Line Choice:     4
Territory Choice: 64 * 8 * 4 = 2,048
Total: ~6,150 → 7,000 (with padding)
```

**Policy Layout for 19×19:**

```
Placement:        3 * 19 * 19 = 1,083
Movement:         19 * 19 * 8 * 18 = 51,984
Line Formation:   19 * 19 * 4 = 1,444
Territory Claim:  19 * 19 = 361
Skip Placement:   1
Swap Sides:       1
Line Choice:      4
Territory Choice: 361 * 8 * 4 = 11,552
Total: 66,430 → 67,000 (with padding)
```

**Policy Layout for Hex (N=12):**

```
Placements:       25 × 25 × 3 = 1,875
Movement/capture: 25 × 25 × 6 × 24 = 90,000
Special:          1 (skip_placement)
Total: P_HEX = 91,876
```

### 8.3 Model Classes and Factory Functions

**Available Model Classes:**

| Class                     | Architecture           | MPS Compatible | Value Head        |
| ------------------------- | ---------------------- | -------------- | ----------------- |
| `RingRiftCNN`             | ResNet + AdaptivePool  | ❌             | Scalar            |
| `RingRiftCNN_MPS`         | ResNet + GlobalAvgPool | ✅             | Scalar            |
| `RingRiftCNN_MultiPlayer` | ResNet + AdaptivePool  | ❌             | Vector (4-player) |
| `HexNeuralNet`            | ResNet + MaskedPool    | ❌             | Scalar            |

**Factory Function** (`app/ai/neural_net.py`):

```python
from app.ai.neural_net import create_model_for_board, get_model_config_for_board
from app.models import BoardType

# Create optimal model for each board type
model_8x8 = create_model_for_board(BoardType.SQUARE8)
model_19x19 = create_model_for_board(BoardType.SQUARE19)
model_hex = create_model_for_board(BoardType.HEXAGONAL)

# Get recommended configuration
config = get_model_config_for_board(BoardType.SQUARE8)
# Returns: {'board_size': 8, 'policy_size': 7000, 'num_res_blocks': 6, ...}

# Create multiplayer model for specific board
model_mp = create_model_for_board(
    BoardType.SQUARE8,
    model_class="RingRiftCNN_MultiPlayer"
)
```

### 8.4 Board-Specific Training Configurations

**Training Config Presets** (`app/training/config.py`):

```python
from app.training.config import get_training_config_for_board
from app.models import BoardType

# Get optimized training config
config_8x8 = get_training_config_for_board(BoardType.SQUARE8)
# config_8x8.policy_size = 7000
# config_8x8.num_res_blocks = 6
# config_8x8.num_filters = 64
# config_8x8.batch_size = 64
# config_8x8.learning_rate = 2e-3
# config_8x8.model_id = "ringrift_8x8_v1"

config_19x19 = get_training_config_for_board(BoardType.SQUARE19)
# config_19x19.policy_size = 67000
# config_19x19.num_res_blocks = 10
# config_19x19.num_filters = 128
# config_19x19.batch_size = 32
# config_19x19.learning_rate = 1e-3
# config_19x19.model_id = "ringrift_19x19_v1"

config_hex = get_training_config_for_board(BoardType.HEXAGONAL)
# config_hex.policy_size = 91876
# config_hex.num_res_blocks = 8
# config_hex.num_filters = 128
# config_hex.model_id = "ringrift_hex_v1"
```

**Pre-defined Configurations** (accessed via `BOARD_TRAINING_CONFIGS`):

| Board Type | Res Blocks | Filters | Batch Size | LR   | Model ID          |
| ---------- | ---------- | ------- | ---------- | ---- | ----------------- |
| SQUARE8    | 6          | 64      | 64         | 2e-3 | ringrift_8x8_v1   |
| SQUARE19   | 10         | 128     | 32         | 1e-3 | ringrift_19x19_v1 |
| HEXAGONAL  | 8          | 128     | 32         | 1e-3 | ringrift_hex_v1   |

### 8.5 Parameter Count Comparison

Using board-specific models significantly reduces parameter count for smaller boards:

| Model                            | Board     | Policy Head Params      | Total Approx. Params |
| -------------------------------- | --------- | ----------------------- | -------------------- |
| RingRiftCNN (8x8, 64 filters)    | SQUARE8   | 256 × 7K = 1.8M         | ~2.5M                |
| RingRiftCNN (19x19, 128 filters) | SQUARE19  | 256 × 67K = 17.2M       | ~18M                 |
| HexNeuralNet (128 filters)       | HEXAGONAL | 128 × 441 × 54K = 13.9M | ~15M                 |

The 8×8 model is **~7× smaller** than the 19×19 model, enabling:

- Faster training iterations
- Lower memory usage
- More efficient inference on resource-constrained devices

### 8.6 Checkpoint Naming Convention

Board-specific checkpoints follow this naming pattern:

```
models/ringrift_<board>_<variant>_v<version>.pth
```

Examples:

- `models/ringrift_8x8_v1.pth` - Standard 8×8 model
- `models/ringrift_8x8_mps_v1.pth` - MPS-compatible 8×8 model
- `models/ringrift_19x19_v1.pth` - Standard 19×19 model
- `models/ringrift_hex_v1.pth` - Hex board model

### 8.7 Usage Examples

**Training a board-specific model:**

```bash
# Train 8×8 model
python -m app.training.train \
  --board-type square8 \
  --model-id ringrift_8x8_v1 \
  --epochs 100

# Train 19×19 model
python -m app.training.train \
  --board-type square19 \
  --model-id ringrift_19x19_v1 \
  --epochs 100

# Train hex model
python -m app.training.train \
  --board-type hexagonal \
  --model-id ringrift_hex_v1 \
  --epochs 100
```

**Loading board-specific models in inference:**

```python
from app.ai.neural_net import NeuralNetAI, create_model_for_board
from app.models import AIConfig, BoardType

# The NeuralNetAI automatically loads based on nn_model_id
ai_8x8 = NeuralNetAI(
    player_number=1,
    config=AIConfig(nn_model_id="ringrift_8x8_v1")
)

ai_19x19 = NeuralNetAI(
    player_number=1,
    config=AIConfig(nn_model_id="ringrift_19x19_v1")
)
```

### 8.8 Files Modified

| File                     | Changes                                                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/ai/neural_net.py`   | Added `POLICY_SIZE_8x8`, `POLICY_SIZE_19x19`, `BOARD_POLICY_SIZES`, `BOARD_SPATIAL_SIZES`, `get_policy_size_for_board()`, `get_spatial_size_for_board()`, `create_model_for_board()`, `get_model_config_for_board()`. Updated all model classes to accept `policy_size` parameter. |
| `app/training/config.py` | Added `num_res_blocks`, `num_filters`, `policy_size` to `TrainConfig`. Added `get_training_config_for_board()` and `BOARD_TRAINING_CONFIGS` preset dictionary.                                                                                                                     |

### 8.9 Compatibility Notes

- **Checkpoint incompatibility:** Models with different `policy_size` cannot share checkpoints. The policy head linear layer has different dimensions.
- **Architecture version:** Each model class has an `ARCHITECTURE_VERSION` attribute for checkpoint validation:
  - `RingRiftCNN`: v1.1.0
  - `RingRiftCNN_MPS`: v1.1.0-mps
  - `RingRiftCNN_MultiPlayer`: v2.0.0
  - `HexNeuralNet`: v1.0.0

---

## 9. Engine Mixing for Self-Play Data Diversity

> **Status (2025-12-05):** ✅ **COMPLETE** - Engine mixing implemented in `generate_data.py` and integrated with curriculum training.

### 9.1 Overview

To improve training data diversity and policy quality, the self-play data generation now supports engine mixing—the ability to randomize which search engine (Descent vs MCTS) is used during self-play games.

**Motivation:**

- **Descent AI** produces soft policies from transposition table values, which can be sharper but may overfit to Descent's specific search patterns.
- **MCTS AI** produces soft policies from visit counts, which better represents move uncertainty and exploration.
- Mixing engines in training data helps the neural network generalize across different play styles and reduces overfitting to a single search algorithm's biases.

### 9.2 Engine Mixing Modes

| Mode         | Description                                                                  | Use Case                                                     |
| ------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `single`     | All players use the same engine (specified by `--engine`)                    | Baseline/controlled experiments                              |
| `per_game`   | Randomly select engine per game (probability controlled by `--engine-ratio`) | Moderate diversity while keeping games internally consistent |
| `per_player` | Randomly select engine per player within each game                           | Maximum diversity, tests cross-engine play                   |

### 9.3 CLI Usage

**Basic data generation with engine selection:**

```bash
# Generate data with MCTS only
python -m app.training.generate_data \
  --num-games 1000 \
  --engine mcts

# Generate data with Descent only (default)
python -m app.training.generate_data \
  --num-games 1000 \
  --engine descent
```

**Engine mixing modes:**

```bash
# Per-game mixing: 50% MCTS, 50% Descent games
python -m app.training.generate_data \
  --num-games 1000 \
  --engine-mix per_game \
  --engine-ratio 0.5

# Per-player mixing: each player independently uses MCTS with 30% probability
python -m app.training.generate_data \
  --num-games 1000 \
  --engine-mix per_player \
  --engine-ratio 0.3

# Heavily MCTS-weighted generation
python -m app.training.generate_data \
  --num-games 1000 \
  --engine-mix per_game \
  --engine-ratio 0.8
```

### 9.4 Curriculum Training Integration

Engine mixing is fully integrated with the curriculum training loop via `CurriculumConfig`:

```python
from app.training.curriculum import CurriculumConfig, CurriculumTrainer
from app.models import BoardType

config = CurriculumConfig(
    board_type=BoardType.SQUARE8,
    generations=10,
    games_per_generation=1000,

    # Engine mixing configuration
    engine="descent",           # Base engine when mix="single"
    engine_mix="per_game",      # Mix mode: "single", "per_game", "per_player"
    engine_ratio=0.5,           # MCTS probability when mixing
)

trainer = CurriculumTrainer(config)
trainer.run()
```

**CLI integration via train.py:**

```bash
# Curriculum training with engine mixing
python -m app.training.train \
  --curriculum \
  --board-type square8 \
  --curriculum-generations 10 \
  --curriculum-games-per-gen 500 \
  --curriculum-engine-mix per_game \
  --curriculum-engine-ratio 0.5
```

### 9.5 Implementation Details

**Key files modified:**

| File                            | Changes                                                                                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `app/training/generate_data.py` | Added `engine_mix`, `engine_ratio`, `nn_model_id` parameters to `generate_dataset()`. Added `_make_ai()` and `_select_engine_for_player()` helper functions. |
| `app/training/curriculum.py`    | Added `engine`, `engine_mix`, `engine_ratio`, `nn_model_id` fields to `CurriculumConfig`. Updated `_generate_self_play_data()` to pass engine parameters.    |
| `app/training/train.py`         | Added `--curriculum-engine`, `--curriculum-engine-mix`, `--curriculum-engine-ratio` CLI arguments.                                                           |

**Engine selection logic** (`generate_data.py`):

```python
def _select_engine_for_player(player_num: int, game_engine: str) -> str:
    """Select engine type for a player based on mixing mode."""
    if engine_mix == "single":
        return game_engine
    elif engine_mix == "per_player":
        return "mcts" if random.random() < engine_ratio else "descent"
    else:  # per_game
        return game_engine
```

### 9.6 Dataset Metadata

When engine mixing is enabled, each game's history entries include the engine used:

```python
game_history = [
    {
        "state": state_dict,
        "move": encoded_move,
        "search_data": {...},
        "engine": "mcts",  # or "descent"
    },
    ...
]
```

This metadata can be used for:

- Filtering training data by engine type
- Analyzing engine-specific policy quality
- Debugging engine mixing distributions

### 9.7 Recommended Configurations

| Scenario                        | Mode         | Ratio | Notes                                            |
| ------------------------------- | ------------ | ----- | ------------------------------------------------ |
| **Initial training**            | `per_game`   | 0.5   | Balanced diversity for exploration               |
| **MCTS-focused fine-tuning**    | `per_game`   | 0.8   | Emphasize visit-based soft policies              |
| **Descent-focused fine-tuning** | `per_game`   | 0.2   | Emphasize value-based policies                   |
| **Maximum diversity**           | `per_player` | 0.5   | Cross-engine games, hardest for model to overfit |
| **Controlled baseline**         | `single`     | N/A   | Reproducible experiments                         |

### 9.8 Future Enhancements

- **Engine-specific evaluation:** Evaluate trained models against both Descent and MCTS opponents to measure generalization.
- **Adaptive mixing:** Adjust engine ratio dynamically based on training progress or model evaluation metrics.
- **Engine annotations in policy targets:** Weight policy targets differently based on source engine (e.g., MCTS visit counts may be more reliable than Descent values for certain positions).

---

## 10. Distributed AI Strength Evaluation

> **Status (2025-12-11):** ✅ **IMPLEMENTED** - New distributed tournament infrastructure for empirical AI strength validation.

### 10.1 Overview

A distributed tournament system has been implemented to empirically measure the relative strength of AI configurations across the canonical 1-10 difficulty ladder. The system:

- Runs AI-vs-AI matches in parallel across multiple workers
- Calculates Elo ratings from match results
- Supports checkpointing and resumption
- Generates comprehensive strength reports

### 10.2 Distributed Tournament Script

**Location:** `scripts/run_distributed_tournament.py`

**Features:**

- Round-robin matchups between all specified tiers
- Parallel game execution with configurable worker count
- Elo rating calculation with K-factor 32
- Head-to-head matrix generation
- JSON checkpointing for fault tolerance
- Resume capability from previous runs

**Usage:**

```bash
# Quick validation (D1-D4, 10 games per matchup)
python scripts/run_distributed_tournament.py --quick

# Full tournament (D1-D6, 50 games per matchup)
python scripts/run_distributed_tournament.py --games-per-matchup 50 --tiers D1,D2,D3,D4,D5,D6

# Resume from checkpoint
python scripts/run_distributed_tournament.py --resume results/tournaments/tournament_abc123.json

# Use specific board type
python scripts/run_distributed_tournament.py --board square19 --tiers D2,D4,D6,D8
```

### 10.3 Canonical Difficulty Ladder

The tournament validates the following canonical AI configurations:

| Tier | AI Type   | Algorithm                  | Think Time | Randomness | Neural |
| ---- | --------- | -------------------------- | ---------- | ---------- | ------ |
| D1   | Random    | Uniform random             | 150ms      | 50%        | No     |
| D2   | Heuristic | 45-weight evaluation       | 200ms      | 30%        | No     |
| D3   | Minimax   | Alpha-beta search          | 1.8s       | 15%        | No     |
| D4   | Minimax   | Alpha-beta + NNUE          | 2.8s       | 8%         | Yes    |
| D5   | MCTS      | Monte Carlo Tree Search    | 4.0s       | 5%         | No     |
| D6   | MCTS      | MCTS + neural value/policy | 5.5s       | 2%         | Yes    |
| D7   | MCTS      | Expert neural MCTS         | 7.5s       | 0%         | Yes    |
| D8   | MCTS      | Strong expert MCTS         | 9.6s       | 0%         | Yes    |
| D9   | Descent   | AlphaZero-style UBFM       | 12.6s      | 0%         | Yes    |
| D10  | Descent   | Grandmaster Descent        | 16.0s      | 0%         | Yes    |

### 10.4 AI Strength Assessment

Based on architectural analysis, canonical ladder configuration, and preliminary tournament data:

#### Theoretical Strength Hierarchy

| Tier | AI Type      | Algorithm           | Expected Elo Range | Key Strength Factors       |
| ---- | ------------ | ------------------- | ------------------ | -------------------------- |
| D1   | Random       | Uniform random      | 800-1000           | None - baseline            |
| D2   | Heuristic    | 45-weight eval      | 1200-1400          | Tuned evaluation function  |
| D3   | Minimax      | Alpha-beta (3-ply)  | 1350-1500          | Look-ahead search          |
| D4   | Minimax+NNUE | Alpha-beta + neural | 1450-1600          | Better position evaluation |
| D5   | MCTS         | Monte Carlo TS      | 1500-1700          | Statistical sampling       |
| D6   | MCTS+Neural  | Neural value/policy | 1650-1850          | Guided search              |
| D7   | MCTS Expert  | Deep neural MCTS    | 1800-2000          | Increased search budget    |
| D8   | MCTS Strong  | Extended budget     | 1900-2100          | More simulations           |
| D9   | Descent      | AlphaZero UBFM      | 2000-2200          | Best-first search          |
| D10  | Descent Max  | Maximum budget      | 2100-2400          | Deepest search             |

#### Key Observations from Architecture

1. **D1 → D2 Gap (~200-400 Elo):** The largest strength gap. Any strategy beats random.

2. **D2 → D3/D4 Gap (~150-200 Elo):** Minimax's lookahead provides significant advantage over pure heuristic, but only with sufficient think time.

3. **D4 NNUE Enhancement (~100-150 Elo over D3):** The NNUE neural evaluation provides more accurate position assessment than the 45-weight heuristic.

4. **D5 → D6 Neural Gap (~150-200 Elo):** Adding neural guidance to MCTS significantly improves move selection and value estimation.

5. **D7-D10 Diminishing Returns (~100-150 Elo per tier):** Higher tiers primarily differ in search budget (think time), showing diminishing returns.

#### Tournament Infrastructure Status

- **ZobristHash Fix:** ✅ Thread-safe singleton pattern implemented
- **Distributed Tournament Script:** ✅ Created (`scripts/run_distributed_tournament.py`)
- **Extended Tournament (D1-D5):** 🔄 Running in background (200 games, ~2 hours estimated)

#### Preliminary Tournament Data (Quick Run)

Tournament ID: `af0957de` | Board: Square 8×8 | Games per matchup: 10

| Matchup  | Result (W-L-D) | Notes                         |
| -------- | -------------- | ----------------------------- |
| D1 vs D2 | 0-0-10\*       | \*ZobristHash bug (now fixed) |
| D1 vs D3 | 3-7            | D3 wins as expected           |
| D1 vs D4 | 2-7            | D4 wins as expected           |
| D2 vs D3 | 7-3            | D2 surprisingly strong        |
| D2 vs D4 | 8-2            | D2 surprisingly strong        |
| D3 vs D4 | 4-6            | D4 slight edge as expected    |

**Analysis:** The D2 (Heuristic) outperformance over D3/D4 in the quick tournament is likely due to:

1. Capped think times favoring instant evaluation over search
2. Well-tuned 45-weight heuristic (CMA-ES optimized)
3. Small sample size (10 games per matchup)

Full tournament with proper think times will provide more accurate strength measurements.

### 10.5 Cluster Infrastructure Integration

The tournament system integrates with the existing cluster infrastructure:

**Distributed Hosts** (`config/distributed_hosts.yaml`):

- Local Mac cluster via Tailscale (mac-studio, mbp-16gb, mbp-64gb)
- AWS staging instance (r5.4xlarge, 128GB RAM)
- Lambda Labs GPU instance (A10, 222GB RAM)
- Vast.ai GPU instances (3090, 5090 configurations)

**Cluster Manager** (`scripts/cluster_manager.py`):

- SSH-based worker lifecycle management
- Health checking and memory monitoring
- Preloading state pools
- Running CMA-ES and self-play soaks

### 10.6 ZobristHash Thread-Safety Fix (2025-12-11)

**Issue:** The original `ZobristHash` singleton was not thread-safe. In multi-threaded tournament play, concurrent accesses could result in `'ZobristHash' object has no attribute 'table'` errors when one thread accessed the singleton before another thread completed initialization.

**Fix:** Applied double-checked locking pattern with `threading.Lock`:

```python
class ZobristHash:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(ZobristHash, cls).__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance
```

**File Modified:** `app/ai/zobrist.py`

### 10.7 Future Work

1. ~~**Fix RandomAI ZobristHash issue**~~ ✅ RESOLVED (thread-safe singleton)
2. **Extended tournament** - Run full D1-D10 tournament with 100+ games per matchup
3. **Cross-board validation** - Validate strength consistency across Square8, Square19, and Hex
4. **Cloud-distributed execution** - Leverage Lambda/Vast.ai for faster tournament completion
5. **Continuous strength monitoring** - Integrate tournament into CI for regression detection

---

## 11. MCTS + Descent Strength Maximization Roadmap (v2.x+)

> **Purpose:** Exhaustive, future‑facing list of upgrades to push top‑tier playing strength for `MCTSAI` (D7–D8) and `DescentAI` (D9–D10).  
> **Constraints:** All training/self‑play inputs must be canonical (`canonical_*.db` passing parity + history gates). Search must stay rules‑correct and TS↔Python‑parity‑aligned.

### 11.1 Shared Search Upgrades (applies to both MCTS and Descent)

**Throughput → strength**

- **Async + batched NN inference:** collect leaf states across workers/threads, evaluate as GPU batches, and feed results back into the tree (higher sims/sec for same wall‑clock).
- **Kernelized move generation:** keep pushing hot loops (valid‑move generation, line/territory filters) into Numba/CUDA paths once GPU parity is complete.
- **Phase‑aware move ordering:** reuse encoder + heuristics to order legal moves by phase (captures/line/territory moves first), improving both pruning and tree focus.
- **Unified transposition caching:** cache NN evals, legal moves, and terminal proofs keyed by canonical state hash; share across turns via subtree/TT reuse.

**Correctness‑preserving pruning**

- **Safe dominance pruning:** skip provably dominated moves within a phase (e.g., identical destinations after symmetry/rotation), guarded by parity fixtures.
- **Early terminal proofing:** propagate proven win/loss/draw up the tree immediately to avoid wasting sims in dead subtrees.

### 11.2 MCTS‑Specific Strength Upgrades

**Tree policy / math**

- **Dynamic `c_puct` and FPU:** make exploration constant depend on root visits/entropy; use phase‑specific FPU to prevent over‑exploring junk actions.
- **RAVE tapering:** decay or disable RAVE as NN priors strengthen; tune `rave_k` per board/phase.
- **Root Dirichlet noise for self‑play:** add `α`‑scaled Dirichlet noise only at root during self‑play; keep evaluation deterministic.
- **Temperature schedule:** high temperature early turn / early game, anneal to argmax late game to sharpen endgame play.
- **Progressive widening:** on large action spaces (square19/hex), expand only top‑K prior moves until node visit threshold is met.
- **Virtual loss + multi‑threading:** enable parallel rollouts on CPU/GPU without collapsing onto the same branch; lock‑free or fine‑grain locks on node stats.
- **Full transposition tree:** merge node statistics for identical states reached via different move orders (especially in chain/territory sequences).

**Backups**

- **Value discounting for long lines:** optionally discount value by ply to prefer faster wins / delay losses.
- **Multi‑player backup with vector values:** once v2 vector value head is canonical, back up per‑player utilities using principled multi‑player backups (MaxN as baseline; Paranoid remains default for robustness; optionally evaluate Best‑Reply Search as a cheaper compromise) instead of scalar sign‑flip hacks.

**Rollouts**

- **Heuristic playout fallback:** when NN is missing or low‑confidence, run cheap heuristic rollouts to stabilize Q estimates.

### 11.3 Descent‑Specific Strength Upgrades (UBFM / Best‑First)

**Selection**

- **Exploratory descent:** replace purely greedy best‑child descent with a soft/PUCT‑style selector to prevent local traps.
- **Uncertainty‑aware scores:** maintain `(mean, variance)` or confidence bounds per child and descend on optimistic bounds.

**Backups / proofs**

- **Terminal proof propagation:** promote `PROVEN_WIN/LOSS/DRAW` statuses aggressively (already scaffolded via `NodeStatus`) to cut search.
- **Vector Max‑N / Paranoid backup:** same as MCTS — switch to per‑player utilities once vector value head is live, with MaxN + optional Best‑Reply Search behind config flags and Paranoid kept as the safe default.

**Expansion**

- **Progressive widening:** mirror MCTS widening to focus on NN‑preferred candidates first.
- **TT sharing across turns:** preserve and reuse TT entries across consecutive moves to speed convergence.

### 11.4 Neural Network Upgrades to Support Search

**Targets / data**

- **Default to soft search targets:** always train policy on `mcts_visits` or `descent_softmax`; keep 1‑hots only for ablations.
- **Canonical reanalysis:** periodically re‑run search with the latest NN over recent canonical games to refresh policy/value targets (AlphaZero reanalysis).
- **Prioritized replay:** sample positions proportional to policy‑KL/value‑TD error + late‑game/rare‑phase boosts.
- **Stronger symmetry augmentation:** full dihedral rotations/reflections for square boards; 6‑fold rotational symmetries for hex, phase‑safe only.
- **Vector value targets (3–4p):** finish v2 dataset export + training + inference wiring and enable NN‑backed multiplayer search.
- **Optional rank‑distribution head:** auxiliary output predicting final rank histogram for richer multiplayer supervision.

**Architecture**

- **Consolidate around `RingRiftCNN_v3` backbone:** increase depth/width for square19/hex; keep lite variants for latency tiers.
- **Attention / SE blocks:** add lightweight channel/spatial attention to improve long‑range territory/line reasoning.
- **Phase embedding:** inject a learned phase token so policy/value can specialize per phase without overfitting.
- **Auxiliary heads:** territory‑potential / line‑completion probability heads to regularize and stabilize training.
- **Mixed precision + `torch.compile`:** speed up training and inference to unlock more sims and larger batches.

### 11.5 Self‑Play / Curriculum Strength Improvements

- **Engine‑mix curriculum:** start with Descent‑heavy exploration, shift toward MCTS‑heavy fine‑tuning as policy stabilizes.
- **Temperature + resign curriculum:** phase‑aware temperature decay; conservative resign thresholds early in training.
- **Opponent pools / league:** keep a rolling pool of past best models and sample opponents to prevent cycling.
- **Cross‑board warm starts:** pretrain on square8, then finetune square19/hex using shared backbone weights.

### 11.6 Evaluation / Promotion Tightening (to prevent false gains)

- **Significance‑gated promotion:** require Wilson/SPR T‑test confidence that win‑rate > threshold before promotion.
- **Cross‑engine eval:** every candidate must beat both Descent and MCTS baselines at matched budgets.
- **Cross‑board sanity:** periodic tournaments on square8/square19/hex to prevent board‑specific regressions.
- **Nightly regression tournaments:** automate “current best vs previous best” tournaments to catch strength drift.

### 11.7 Recommended Implementation Slices (Strength‑First)

| Slice | Goal                                                          | Priority | Dependencies                | Primary files                                                                                       |
| ----- | ------------------------------------------------------------- | -------- | --------------------------- | --------------------------------------------------------------------------------------------------- |
| S11‑A | Root Dirichlet noise + temperature schedules (self‑play only) | HIGH     | None                        | `app/ai/mcts_ai.py`, `scripts/run_self_play_soak.py`                                                |
| S11‑B | Progressive widening for square19/hex                         | HIGH     | Policy priors stable        | `app/ai/mcts_ai.py`, `app/ai/descent_ai.py`                                                         |
| S11‑C | Async/batched NN leaf evaluation                              | HIGH     | GPU parity + batching infra | `app/ai/gpu_batch.py`, `app/ai/mcts_ai.py`, `app/ai/descent_ai.py`                                  |
| S11‑D | Vector value head end‑to‑end + enable multiplayer NN search   | HIGH     | v2 datasets                 | `app/ai/neural_net.py`, `app/training/train.py`, `scripts/export_replay_dataset.py`, search engines |
| S11‑E | Canonical reanalysis pipeline                                 | MED      | S11‑C                       | `scripts/run_improvement_loop.py`, `scripts/export_replay_dataset.py`                               |
| S11‑F | Significance‑gated promotion in curriculum/tier gates         | MED      | Distributed tournaments     | `app/training/curriculum.py`, `scripts/run_tier_gate.py`                                            |
| S11‑G | RAVE tapering + dynamic c_puct/FPU                            | MED      | S11‑A                       | `app/ai/mcts_ai.py`                                                                                 |
| S11‑H | Uncertainty‑aware Descent selection                           | MED      | S11‑D                       | `app/ai/descent_ai.py`                                                                              |
