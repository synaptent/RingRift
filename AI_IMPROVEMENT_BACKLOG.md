# RingRift AI & Neural Net Improvement Backlog

This backlog captures concrete follow-up work for the RingRift AI system across TypeScript, Python, and documentation. It is intended as a ticket-ready index; architectural context and rationale live in **`AI_ARCHITECTURE.md`** (canonical overview) and historical deep dives:

- `AI In Depth Improvement Analysis.md` (historical, detailed review)
- `ai-service/AI_ASSESSMENT_REPORT.md` (Python AI service–focused review)

Status markers:

- `[ ]` not started
- `[~]` in progress / partially implemented
- `[x]` done (kept here for traceability)

---

## 1. Difficulty Ladder, Behaviour & UX

**Goal:** Single canonical difficulty ladder, consistent between TS and Python, exposed clearly in UI and docs.

Canonical mapping (see `AI_ARCHITECTURE.md` §1.2):

- 1 → `RandomAI`
- 2 → `HeuristicAI`
- 3–6 → `MinimaxAI`
- 7–8 → `MCTSAI`
- 9–10 → `DescentAI`

### 1.1 Ladder implementation & TS alignment

- [ ] **Align TS difficulty presets with canonical ladder**
  - Update `AI_DIFFICULTY_PRESETS` in `src/server/game/ai/AIEngine.ts` so that:
    - 1 → `AIType.RANDOM`
    - 2 → `AIType.HEURISTIC`
    - 3–6 → `AIType.MINIMAX`
    - 7–8 → `AIType.MCTS`
    - 9–10 → `AIType.DESCENT`
  - Ensure `selectAITypeForDifficulty` is a thin wrapper over `AI_DIFFICULTY_PRESETS`.
  - Add unit tests asserting the clamped mapping for all difficulties 1–10.

- [ ] **Verify Python ladder parity**
  - Ensure `_CANONICAL_DIFFICULTY_PROFILES` in `ai-service/app/main.py` matches the same mapping and `think_time_ms` / `randomness` ranges.
  - Add a short comment in both TS and Python pointing at each other as canonical mirrors.
  - Add `/ai/move` tests in `ai-service/tests/test_ai_creation.py` (or similar) that assert the created AI type for each difficulty matches the canonical ladder.

- [ ] **Wire `DescentAI` into production ladder explicitly**
  - Confirm `AIType.DESCENT` is exported and supported by `AIServiceClient`.
  - Add explicit tests that `/ai/move` for difficulty 9–10 returns `ai_type == 'descent'`.
  - Add a small integration test on the TS side ensuring `AIServiceClient` sends `ai_type: 'descent'` for 9–10.

### 1.2 UI & player-facing text

- [ ] **Update difficulty labels and tooltips**
  - Ensure lobby / HUD difficulty descriptions match the canonical ladder (e.g. 1 Beginner–Random, 2 Easy–Heuristic, 3–6 Minimax “thinker”, 7–8 MCTS expert, 9–10 Descent/UBFM).
  - Keep descriptions honest about actual strength (avoid promises like “perfect play”).

- [ ] **Expose AI type and persona to players where helpful**
  - In Game HUD / lobby, show both difficulty number and high-level type label (Random / Heuristic / Minimax / MCTS / Descent).
  - Optionally surface “persona” once implemented (see §5).

### 1.3 Legacy / fallback modes

- [ ] **Make AIControlMode (`service` vs `local_heuristic`) actually affect behaviour**
  - In `AIEngine.getAIMove` and choice methods, respect `AIConfig.mode`:
    - `service`: call Python first, fallback to local on failure.
    - `local_heuristic`: skip service entirely, use TS heuristics.
  - Add tests for both modes.

- [ ] **Strengthen TS local fallback AI for mid/high difficulties**
  - Introduce a simple TS-side heuristic that roughly mirrors `HeuristicAI` for:
    - Captures vs quiet moves.
    - Territory and line-forming opportunities.
  - Use this heuristic in `getLocalAIMove` for higher difficulties instead of pure bucket-random selection.

---

## 2. RNG, Determinism & Replayability

**Goal:** Deterministic, reproducible AI behaviour across TS and Python when a seed is provided, while retaining stochasticity in casual play.

### 2.1 TS RNG discipline

- [ ] **Eliminate bare `Math.random` from AI-critical paths**
  - Confirm all AI decisions (TS local fallback, sandbox AI, normalization) use injected `LocalAIRng` only.
  - Remaining bare calls should be limited to non-critical UX (e.g. cosmetic animations).

- [ ] **Normalize service placements deterministically**
  - In `AIEngine.normalizeServiceMove`, we already default empty-cell `place_ring` without `placementCount` to `1` ring and log a warning.
  - [ ] Add metrics / tests ensuring no `Math.random` is used, and consider promoting “missing placementCount” to a hard contract error in a future phase.

- [ ] **Per-game RNG for TS**
  - Introduce a `GameRng` abstraction (e.g. xorshift128+ implemented once in `src/shared/utils/rng.ts`).
  - Seed it from `(gameId, initialStateHash)` when a game is created.
  - Thread it into:
    - `AIEngine.getAIMove` (as `rng?: LocalAIRng`).
    - `AIEngine.getLocalFallbackMove` and sandbox AI entrypoints.

### 2.2 Python RNG discipline

- [ ] **Stop reseeding global RNG in `ZobristHash`**
  - Replace `random.seed(42)` in `ai-service/app/ai/zobrist.py` with a local RNG instance: `self._rng = random.Random(42)`.
  - Use `self._rng.getrandbits(64)` for table entries.
  - Replace Python’s salted `hash()` with a deterministic hash function over a canonical tuple or string.

- [ ] **Per-game RNG for all Python AIs**
  - Give `BaseAI` an instance `self.rng: random.Random`, seeded from an `rng_seed` in `AIConfig`.
  - Route **all** randomness (thinking delays, ε-greedy choices, MCTS expansions/rollouts) through this RNG.
  - Ensure `RandomAI`, `HeuristicAI`, `MinimaxAI`, `MCTSAI`, and `NeuralNetAI` no longer call `random.*` directly.

- [ ] **Extend `/ai/move` to accept an explicit RNG seed**
  - Add `seed: Optional[int]` to the MoveRequest model in `ai-service/app/main.py` (already supported on the TypeScript side via `AIServiceClient`).
  - When `seed` is provided, construct a fresh AI instance with that seed (no cross-game caching) to guarantee deterministic behaviour for that move.

### 2.3 Cross-language determinism and replay

- [ ] **Propagate seeds TS → Python consistently**
  - Ensure `AIServiceClient.getAIMove` uses `GameState.rngSeed` or an explicit debug seed to set `seed` in `/ai/move` calls.
  - Document the seed propagation path in `AI_ARCHITECTURE.md`.

- [ ] **Add AI-level determinism tests**
  - For each AI type (Random, Heuristic, Minimax, MCTS, Descent):
    - Seed TS and Python RNGs with a known value.
    - Run a fixed number of AI turns from a known initial `GameState`.
    - Assert that move sequences and `hashGameState`/`hash_game_state` stay identical across runs.

- [ ] **Include RNG metadata in debug traces**
  - Extend sandbox and backend trace entries with:
    - Game-level RNG seed.
    - Optional move-level RNG counter / offset.
  - Make it easy to reconstruct a full match from a “repro bundle” (initial state + seeds + human inputs).

---

## 3. Rules Parity (Python ↔ TypeScript)

**Goal:** Python rules engine (`ai-service/app/game_engine.py`) is a 1:1 semantic match to the TypeScript rules, including edge cases, so training and analysis are spec-accurate.

### 3.1 Stalemate, global stalemate & tie-breakers

- [ ] **Implement full stalemate / tie-break logic in Python**
  - Mirror the logic from TS `GameEngine` and `TurnEngine` for:
    - Global stalemate conditions.
    - “No stacks remaining” draw/tiebreak.
    - Forced-elimination chains leading into stalemate resolutions.
  - Add targeted fixtures and tests in `ai-service/tests/parity/` and TS counterparts.

### 3.2 Line processing (Option 2 granularity)

- [ ] **Expose and handle “which 3 markers to collapse” in overlength lines**
  - Extend Python move/choice representations so an overlength line can specify **which segment** of 3 markers is collapsed (Option 2), matching TS’s `choose_line_reward` moves and `LineRewardChoice`.
  - Update Python `GameEngine` and mutators to respect this choice.
  - Add parity tests that:
    - Use TS to generate overlength-line scenarios with different Option 2 choices.
    - Replay them in Python and compare `hash_game_state` and S-invariant.

### 3.3 Forced elimination behaviour

- [ ] **Align forced elimination multi-step flows**
  - Compare TS `TurnEngine` / `GameEngine` forced elimination logic with Python’s elimination flows.
  - Fix Python to:
    - Match TS ordering of candidate stacks.
    - Match sequence of eliminations across turns where multiple are required.
  - Add scenario tests for multi-step eliminations (e.g., `ForcedEliminationAndStalemate.test.ts` mirrored in Python).

### 3.4 Parity fixtures & CI

- [ ] **Expand TS→Python parity fixtures for new edge cases**
  - Generate new fixtures for:
    - Overlength lines with multiple valid Option 2 choices.
    - Multi-region territory disconnections with self-elimination.
    - Global stalemate endgames.
  - Wire them into `ai-service/tests/parity/` and TS tests.

- [ ] **Add CI gate on rules parity**
  - Ensure GitHub Actions (or equivalent) fails if any TS↔Python parity test regresses.

---

## 4. Search Performance & Engine Optimization

**Goal:** Make Minimax/MCTS/Descent fast and scalable by decoupling search from heavy, parity-oriented rules paths while preserving correctness.

### 4.1 Search-optimized board representation

- [ ] **Introduce `SearchBoard` (or equivalent) in Python**
  - Represent only the fields needed for search:
    - Stack heights, ring ownership, marker positions.
    - Collapsed spaces, eliminated rings (for victory detection).
  - Implement `make_move` / `unmake_move` operations with **O(1)** incremental updates.
  - Validate behaviour against canonical `GameEngine` using dedicated tests.

- [ ] **Refactor MinimaxAI to use `SearchBoard` internally**
  - Use `SearchBoard` for recursive nodes; only convert back to full `GameState` when needed (e.g., for /rules/evaluate_move or debug).
  - Measure and document depth increase vs previous DefaultRulesEngine-based search.

- [ ] **Refactor MCTSAI / DescentAI to use `SearchBoard`**
  - Drive tree expansion and rollouts on `SearchBoard`.
  - Keep DefaultRulesEngine only for top-level validation and parity tests.

### 4.2 Zobrist hashing & transposition tables

- [ ] **Make Zobrist hashing deterministic and process-independent**
  - See RNG tasks in §2.2.
  - Use `SearchBoard` state to derive a canonical 64-bit hash.

- [ ] **Review and correct TT flag logic in MinimaxAI**
  - Fix flag selection (`exact`, `lowerbound`, `upperbound`) to use original `alpha`/`beta` rather than updated values.
  - Add unit tests covering transposition flag behaviour.

### 4.3 MCTS tree reuse & memory bounds

- [ ] **Scope MCTS trees per game and guard reuse**
  - Store a `game_id` and `state_hash` on `MCTSNode` roots.
  - Only reuse `self.last_root` when:
    - `game_id` matches,
    - `state_hash` matches the expected pre-move state.

- [ ] **Introduce node and memory caps**
  - Limit total nodes per search and/or per game.
  - On exceeding caps, drop oldest subtrees or disable tree reuse for that game.

- [ ] **Cache NN evaluations in MCTS**
  - Use `hash_game_state` or `SearchBoard` hash as key.
  - Cache `(value, policy_probs)` for leaf states.
  - Add an LRU bound to keep memory under control.

---

## 5. Neural Net Robustness & Training Pipeline

**Goal:** Robust, versioned NN infrastructure with safe loading, clear failure modes, and solid training/eval loop.

### 5.1 Model loading, versioning & health

- [ ] **Introduce explicit model versioning & config**
  - Read model paths and tags from a config (env or `model_config.json`), not hard-coded `ringrift_v1.pth`.
  - Expose current model version/sha in a health/metadata endpoint.

- [ ] **Harden `NeuralNetAI` loading**
  - Use `torch.load(..., weights_only=True)` where possible.
  - On missing/mismatched weights:
    - In production: treat as a fatal error or cleanly disable NN-backed AIs (MCTS/Descent fall back to heuristic-only).
    - In dev: allow random models but log loudly.

- [ ] **Add health checks & NaN/Inf guards**
  - After each `evaluate_batch`, assert `np.isfinite(values).all()` and `np.isfinite(policy_probs).all()`.
  - On failure, log and return safe defaults (e.g., zero value, uniform policy) to avoid corrupting MCTS.

### 5.2 Move encoding & tests

- [ ] **Make encode/decode contracts explicit and tested**
  - Decide whether `decode_move` is:
    - A best-effort visualization helper; or
    - A true inverse used in training/self-play.
  - Update tests in `ai-service/tests/test_action_encoding.py` to cover:
    - Round-trip for all encodable move families.
    - Hex boards.
    - Expected failures (INVALID_MOVE_INDEX).

- [ ] **Plan for richer line/territory choice encoding**
  - Reserve sections of the policy vector for future line-reward and territory-region choices.
  - Document the mapping in comments and `AI_ARCHITECTURE.md`.

- [x] **Introduce a dedicated hex-board neural network (`HexNeuralNet`)**
  - **Status:** encoder + HexNeuralNet + unit tests implemented.
    - `ActionEncoderHex` encodes/decodes the canonical hex action space with
      policy dimension `P_hex = 54_244` (21×21×3 placements, 21×21×6×20
      movement/capture, and one skip-placement slot), per
      `AI_ARCHITECTURE.md`.
    - `HexNeuralNet` implements the documented ResNet-style backbone, masked
      value head, and hex policy head sized to `P_hex`, with architecture
      tests in `ai-service/tests/test_model_architecture.py`.
    - Hex action-encoding tests live in `ai-service/tests/test_action_encoding.py`.
  - [x] **Wire hex NN into search engines based on `boardType`**
    - `MCTSAI` now selects between `RingRiftCNN` + the canonical square
      encoder and `HexNeuralNet` + `ActionEncoderHex` based on
      `board.type`, with a dedicated hex-path smoke test in
      `ai-service/tests/test_mcts_ai.py::test_select_move_uses_hex_network_for_hex_board`.
    - `DescentAI` similarly branches on `board.type` and uses the shared
      feature encoder plus `HexNeuralNet` + `ActionEncoderHex` for
      hexagonal boards, with a smoke test in
      `ai-service/tests/test_descent_ai.py::TestDescentAIHex::test_select_move_uses_hex_network_for_hex_board`.
  - [ ] **Add richer hex-specific search parity/behaviour tests**
    - Add small hex scenarios to validate that MCTS/Descent with the hex NN
      preserve expected tactical behaviour and do not regress compared to the
      square-only fallback.
    - Extend `AI_ARCHITECTURE.md` with a short description of the
      `boardType`→(model, encoder) selection logic and link it from this
      backlog item.

### 5.3 Training loop & data pipeline

- [ ] **Scalable dataset storage**
  - Replace load-all-into-RAM `.npy` usage with a streaming dataset (e.g., HDF5, LMDB) or sharded `.npz` + index.

- [ ] **Tournament-based model promotion**
  - Implement a `Tournament` class that pits new checkpoints vs current best.
  - Promote models based on win-rate thresholds (e.g. >55% over N games).

- [ ] **Align training rules with production rules**
  - Ensure self-play uses the same rules mode as production (no lingering simplifications).
  - Add a regression test that replays a small training dataset through both TS and Python rules to confirm parity.

---

## 6. Behaviour Tuning & Personas

**Goal:** Make AI behaviour tunable by designers (not just engineers), with distinct playstyles at similar raw strength.

### 6.1 Wire Minimax into live service

- [ ] **Enable MinimaxAI in `ai-service/app/main.py`**
  - Un-comment and wire the `AIType.MINIMAX` branch in `_create_ai_instance`.
  - Adjust `_select_ai_type` to map 3–6 to `MINIMAX` per canonical ladder.
  - Add `/ai/move` tests asserting Minimax is used for those difficulties.

### 6.2 Personas via heuristic weights and NN variants

- [ ] **Define persona presets**
  - `balanced`: current heuristic weights.
  - `aggressive`: higher weights for captures and elimination, lower for territory safety.
  - `territorial`: higher territory and line/region weights, lower elimination focus.

- [ ] **Extend `AIProfile` / `AIConfig` with `persona`**
  - Thread `persona` through TS (`AIEngine`, `AIProfile`) and Python (`AIConfig`, AI constructors).
  - Use persona to select:
    - A heuristic weight profile for HeuristicAI/MinimaxAI.
    - A NN model variant (optional) for MCTS/Descent.

### 6.3 Sandbox & TS behaviour alignment

- [ ] **Expose production AI behaviour in sandbox (optional)**
  - Add a “remote AI” toggle in sandbox that calls `/ai/move` directly for moves.
  - Or port a thin, approximate version of `HeuristicAI` into TS so sandbox AI better matches low/mid difficulties.

### 6.4 HeuristicAI training & weight tuning

**Goal:** Treat Python `HeuristicAI` as a trainable model with versioned weight sets and clear personas.

- [ ] **Externalize heuristic weights into a config/profile module**
  - Move the scalar weights used in `evaluate_position` into a dedicated module or config file (e.g., `heuristic_weights.py`).
  - Define named profiles such as `heuristic_v1_balanced`, `heuristic_v1_aggressive`, `heuristic_v1_territorial`.
  - Ensure `HeuristicAI` reads weights from this config rather than hard-coded literals.

- [ ] **Define training targets and datasets**
  - Choose an initial training target:
    - Teacher-based: use stronger engines (MCTS+NN or deeper Minimax) to generate scalar evaluations for positions.
    - Or outcome-based: use full game outcomes to approximate `P(win | state)`.
  - Add utilities under `ai-service/app/training/` or `ai-service/scripts/` to:
    - Convert logged games and test traces into `(features, target)` pairs.
    - Include curated designer scenarios for critical tactical positions.

- [ ] **Implement a training script for heuristic weights**
  - Create `train_heuristic_weights.py` that:
    - Loads datasets of `(features, target)`.
    - Optimizes the weight vector using least squares or a simple gradient-based optimizer.
    - Writes updated weight profiles back to the config module.
  - Optionally support a tiny PyTorch model for non-linear corrections whose parameters are exported as scalars.

- [ ] **Versioning, personas and integration**
  - Extend `AIConfig` / `AIProfile` with `persona` and optional `heuristic_version`.
  - Map personas (`balanced`, `aggressive`, `territorial`) to specific weight profiles.
  - Thread persona and version through TS → Python so `/ai/move` can select the appropriate profile.
  - Document persona semantics and versions in `AI_ARCHITECTURE.md`.

- [ ] **Validation and regression tests**
  - Add regression tests in `ai-service/tests/test_heuristic_ai.py` that:
    - Cover known blunder positions and assert newer profiles avoid them.
    - Ensure behaviour remains sensible on reference opening/endgame scenarios.
  - Add a lightweight tournament-style evaluation harness that pits new heuristic profiles vs the current baseline and reports win-rate and key metrics.

---

## 7. Tooling, Tests & Observability

**Goal:** Make AI easy to test, debug, and monitor over time.

- [ ] **Extend AI tests in TS**
  - Add focused tests for:
    - Difficulty→AIType mapping in `AIEngine`.
    - Fallback paths (service failure → heuristic → random).
    - Non-move choice parity (line reward, region order, ring elimination) once Python side is fully wired.

- [ ] **Extend AI tests in Python**
  - Add explicit coverage for:
    - MCTS tree reuse gating by game ID.
    - NN evaluation caching correctness.
    - DescentAI correctness on small tactical scenarios.

- [ ] **Centralize AI logging & metrics**
  - Prefer `logger` over `console` in all TS AI code.
  - Add Prometheus-style metrics:
    - `ai_move_latency_ms{engine,difficulty}` (already begun).
    - `ai_service_failure_total{reason}`.
    - `ai_local_fallback_total{difficulty}`.

- [ ] **Standardize “repro bundle” tooling**
  - Script that, given a game ID and timestamp, dumps:
    - Initial seed and RNG metadata.
    - Full move history and relevant AI configs.
  - Provide a CLI to replay the match in both TS and Python.

---

## 8. Documentation & Governance

**Goal:** Single, current AI architecture + roadmap document, with historical references preserved but clearly marked.

- [ ] **Finalize `AI_ARCHITECTURE.md` as the canonical AI doc**
  - Ensure it:
    - Describes current ladder and AI types (TS + Python).
    - Summarizes key design/implementation details for each engine.
    - Includes a high-level roadmap linked to this backlog.

- [ ] **Mark older AI docs as historical / auxiliary**
  - At the top of `AI In Depth Improvement Analysis.md`, add a banner:
    - “Historical deep-dive; see `AI_ARCHITECTURE.md` and `AI_IMPROVEMENT_BACKLOG.md` for current behaviour and tasks.”
  - At the top of `ai-service/AI_ASSESSMENT_REPORT.md`, add a similar banner focused on the Python AI service.

- [ ] **Keep docs and code in lockstep**
  - For any future change to:
    - Difficulty mapping.
    - AI types or personas.
    - RNG/seed behaviour.
  - Require updates to:
    - `AI_ARCHITECTURE.md`.
    - `AI_IMPROVEMENT_BACKLOG.md` (if it affects roadmap).
    - Relevant comments in TS and Python code.
  - Use the following **difficulty-ladder governance checklist** whenever the 1–10 ladder
    is modified:
    - [ ] Update TS presets in `AI_DIFFICULTY_PRESETS` and `selectAITypeForDifficulty()`
          (`src/server/game/ai/AIEngine.ts`).
    - [ ] Update Python profiles in `_CANONICAL_DIFFICULTY_PROFILES` and `_select_ai_type()`
          (`ai-service/app/main.py`).
    - [ ] Update TS ladder tests in
          `tests/unit/AIEngine.serviceClient.test.ts`
          (both the `AI_DIFFICULTY_PRESETS` mapping test and the
          `getAIMove` mapping tests for representative difficulties).
    - [ ] Update Python ladder tests in
          `ai-service/tests/test_ai_creation.py`
          (both `_select_ai_type` and `_get_difficulty_profile` expectations).
    - [ ] Update the ladder table and narrative in `AI_ARCHITECTURE.md`
          (§“Difficulty-to-AI-Type Mapping”).
    - [ ] If UI text changes, update any player-facing difficulty descriptions
          (lobby, tooltips, HUD) to stay consistent with the new ladder.
