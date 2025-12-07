# Python Rules Implementation Plan (Updated for P0.6)

This document describes the current Python rules engine implementation in the AI-service and how it achieves behavioural parity with the TypeScript shared engine and backend. It supersedes earlier plans that proposed a separate `rules/validators` and `rules/mutators` directory structure.

The authoritative rules specification remains:

- `ringrift_complete_rules.md`
- `docs/rules/ringrift_compact_rules.md`

The canonical executable implementation is the TypeScript shared engine under `src/shared/engine/`, used by:

- `src/server/game/GameEngine.ts`
- `src/server/game/RuleEngine.ts`
- `src/client/sandbox/ClientSandboxEngine.ts`

The Python engine is an implementation of those same rules, designed for:

- Use by the AI-service (for AI search and evaluation).
- Use as a rules backend in shadow mode and, eventually, as an authoritative rules engine (`RINGRIFT_RULES_MODE=python`).

---

## 1. Current Python rules architecture

### 1.1 Core modules

The Python rules engine is implemented in:

- `ai-service/app/game_engine.py` – move generation, state transitions, turn engine, captures, lines, territory, termination.
- `ai-service/app/board_manager.py` – board-level utilities (hashing, S-invariant, line detection, territory regions, neighbors).
- `ai-service/app/rules/default_engine.py` – a validator/mutator–aware `RulesEngine` adapter that currently delegates canonical orchestration to `GameEngine` while routing key move types through dedicated mutators under _shadow contracts_.
- `ai-service/app/main.py` – FastAPI app exposing `/rules/evaluate_move` and AI-related endpoints.

Domain models (`GameState`, `Move`, `BoardState`, `BoardType`, `GamePhase`, `GameStatus`, `RingStack`, etc.) are defined in `ai-service/app/models/` and mirror the TS shared engine types, including the extended Move model (`continue_capture_segment`, `process_line`, `choose_line_reward`, `process_territory_region`, `eliminate_rings_from_stack`, etc.).

### 1.2 GameEngine responsibilities

`GameEngine` provides:

- `get_valid_moves(game_state, player_number)`:
  - Phase-based move generation:
    - `RING_PLACEMENT`: placement and optional `skip_placement` via `_get_ring_placement_moves` and `_get_skip_placement_moves`.
    - `MOVEMENT`, `CAPTURE`, `CHAIN_CAPTURE`: non-capture and capture-chain moves via `_get_movement_moves` and `_get_capture_moves`.
    - `LINE_PROCESSING`: line decision moves via `_get_line_processing_moves`.
    - `TERRITORY_PROCESSING`: region processing and explicit elimination moves via `_get_territory_processing_moves`.
  - Uses canonical directions and geometry via `BoardManager` helpers.

- `apply_move(game_state, move)`:
  - Clones the mutable parts of `GameState` (board, players, move history).
  - Applies the move via helper methods:
    - Placement: `_apply_place_ring`
    - Movement: `_apply_move_stack`
    - Captures: `_apply_chain_capture` (and `_apply_overtaking_capture` wrapper)
    - Lines: `_apply_line_formation`
    - Territory: `_apply_territory_claim`
    - Forced elimination: `_apply_forced_elimination`
  - Updates:
    - `move_history` and `last_move_at`
    - `must_move_from_stack_key` (per-turn must-move semantics)
    - `current_phase` and `current_player`
  - Calls `_check_victory` to apply ring-elimination, territory, and tiebreak-based terminal states.
  - Enforces S-invariant non-decrease via `BoardManager.compute_progress_snapshot`.

- Turn and forced elimination logic:
  - `_update_phase` and `_end_turn` mirror TS `TurnEngine` semantics:
    - Skip placement only when a player has rings in hand and at least one stack with a legal move/capture.
    - Forced elimination when a player has stacks but no legal placement, movement, or capture.
    - After forced elimination, re-check actions for the same player and either start movement or rotate the turn.

### 1.3 BoardManager responsibilities

`BoardManager` contains:

- `hash_game_state(state)`:
  - Canonical textual hash of a GameState used across TS and Python for trace parity and diagnostics.
  - The hash encodes:
    - Current player, phase, and game status.
    - Per-player rings-in-hand, eliminated-rings, and territory-spaces.
    - Board stacks, markers, and collapsed spaces.
    - **Python-only extension:** when `must_move_from_stack_key` is set, the Python hash appends a `:must_move=<posKey>` segment before the first `#`. TS hashes do **not** include this suffix.
    - Parity tests treat this `:must_move=` segment as an ignorable extension (see §3.3).

- `compute_progress_snapshot(game_state)`:
  - Computes markers, collapsed spaces, and eliminated rings, combining them into the S-invariant `S = M + C + E`.

- Line detection and territory helpers mirror the TS `BoardManager` as described in earlier revisions of this plan.

### 1.4 DefaultRulesEngine adapter and mutators

`ai-service/app/rules/default_engine.py` provides a `RulesEngine` implementation used by the FastAPI `/rules/evaluate_move` endpoint and by parity tests.

Responsibilities today:

- `get_valid_moves(state, player)`:
  - Delegates directly to `GameEngine.get_valid_moves` so that canonical
    move generation remains in one place.

- `validate_move(state, move)`:
  - Dispatches to specialised validators under `app.rules.validators.*`:
    - `PlacementValidator` for placement and skip-placement.
    - `MovementValidator` for movement.
    - `CaptureValidator` for overtaking / chain captures.
    - `LineValidator` for line-processing moves.
    - `TerritoryValidator` for territory-processing and explicit elimination.
  - These validators mirror the TS shared validators and are tested via
    `ai-service/tests/rules/test_validators.py`.

- `apply_move(state, move)` (shadow-contract orchestration):
  - **Canonical path (unchanged):**
    - Always calls `GameEngine.apply_move(state, move)` and returns the resulting `GameState`.
    - `GameEngine` remains the single source of truth for turn/phase/victory and hashing.

  - **Placement shadow contract (existing):**
    - For `MoveType.PLACE_RING`, `DefaultRulesEngine.apply_move` also:
      - Creates `mutator_state = state.model_copy(deep=True)`.
      - Applies `PlacementMutator().apply(mutator_state, move)`.
      - Asserts equality of:
        - `board.stacks`, `board.markers`, `board.collapsed_spaces`, `board.eliminated_rings`.
        - `players`.
      - Raises `RuntimeError` on any divergence.

  - **Movement shadow contract (existing):**
    - For `MoveType.MOVE_STACK`, the engine similarly:
      - Copies the input state.
      - Applies `MovementMutator().apply(mutator_state, move)`.
      - Asserts that board + players match the canonical `GameEngine.apply_move` result.

  - **Capture shadow contract (new in this task):**
    - For capture-segment move types:
      - `MoveType.OVERTAKING_CAPTURE`
      - `MoveType.CHAIN_CAPTURE`
      - `MoveType.CONTINUE_CAPTURE_SEGMENT`
    - `DefaultRulesEngine.apply_move` now:
      - Copies the input state.
      - Applies `CaptureMutator().apply(mutator_state, move)`, which wraps `GameEngine._apply_chain_capture`.
      - Asserts equality of board and players with the `GameEngine.apply_move` result.

  - **Line-processing shadow contract (new in this task):**
    - For line decision moves:
      - `MoveType.PROCESS_LINE`
      - `MoveType.CHOOSE_LINE_REWARD`
      - `MoveType.LINE_FORMATION` (legacy)
      - `MoveType.CHOOSE_LINE_OPTION` (legacy)
    - `DefaultRulesEngine.apply_move` now:
      - Copies the input state.
      - Applies `LineMutator().apply(mutator_state, move)`, which wraps `GameEngine._apply_line_formation`.
      - Asserts equality of board and players with the `GameEngine.apply_move` result.

  - **Territory-processing shadow contract (new in this task):**
    - For territory-region and elimination moves:
      - `MoveType.PROCESS_TERRITORY_REGION`
      - `MoveType.ELIMINATE_RINGS_FROM_STACK`
      - `MoveType.TERRITORY_CLAIM` (legacy)
      - `MoveType.CHOOSE_TERRITORY_OPTION` (legacy)
    - `DefaultRulesEngine.apply_move` now:
      - Copies the input state.
      - Applies `TerritoryMutator().apply(mutator_state, move)`, which dispatches to `GameEngine._apply_territory_claim` or `_apply_forced_elimination` depending on move type.
      - Asserts equality of board and players with the `GameEngine.apply_move` result.

  - All of these contracts are _shadow_ paths: failures raise `RuntimeError`, but in normal operation only the canonical `GameEngine.apply_move` result is returned to callers.

- Mutator registry:
  - `DefaultRulesEngine` instantiates `PlacementMutator`, `MovementMutator`, `CaptureMutator`, `LineMutator`, `TerritoryMutator`, and `TurnMutator` in `self.mutators`.
  - Today, `apply_move` exercises them only for the contracts described above; future work may use `self.mutators` to drive a purely mutator-based `apply_move` flow.

---

## 2. Python ↔ TypeScript parity mapping

For each major rules axis, the Python engine mirrors TS behaviour:

- **Movement and captures**:
  - Non-capture movement uses the same ray-based geometry, minimum-distance constraints (`distance >= stackHeight`), and path blocking semantics as the TS shared engine.
  - Capture chain logic replicates `validateCaptureSegmentOnBoard` geometry:
    - Attacker.capHeight ≥ target.capHeight.
    - Direction and distance constraints.
    - Clear paths along both legs of the capture.
    - Empty (or own marker) landing cell.
  - Chain capture segments update `chain_capture_state` analogously to TS `ChainCaptureState`.

- **Ring placement**:
  - `_get_ring_placement_moves` and `_apply_place_ring` / `PlacementMutator` behave as described in the previous revision; the contracts above ensure placement mutator semantics stay aligned with `GameEngine.apply_move`.

- **Lines**:
  - `find_all_lines`, `_get_line_processing_moves`, `_apply_line_formation`, and `LineMutator` mirror TS line-formation and reward semantics, including overlength-line option handling.

- **Territory processing**:
  - `find_disconnected_regions`, `_get_territory_processing_moves`, `_apply_territory_claim`, and `TerritoryMutator` mirror TS territory region processing, self-elimination prerequisites, and explicit elimination decisions.

- **Turn engine and victory**:
  - `_update_phase`, `_end_turn`, `_get_forced_elimination_moves`, `_perform_forced_elimination_for_player`, `_apply_forced_elimination`, and `_check_victory` are unchanged and continue to mirror TS behaviour.

These behaviours are validated by:

- Python unit and integration tests in `ai-service/tests/` (engine correctness, env parity, `/rules/evaluate_move`).
- TS-side parity suites such as `tests/unit/Python_vs_TS.traceParity.test.ts` that consume parity vectors generated by Python in `ai-service/tests/parity/generate_test_vectors.py`.
- Python parity tests in `ai-service/tests/parity/test_rules_parity_fixtures.py` that compare TS fixture hashes and S-invariants against Python `BoardManager` outputs.

### 2.1 Hash parity and the `must_move=` suffix

The Python hash intentionally includes `must_move_from_stack_key` via a `:must_move=<posKey>` suffix that does not exist in TS `stateHash` strings. To keep tests honest while acknowledging this extension:

- `BoardManager.hash_game_state` continues to emit the full, enriched hash including `:must_move=` when applicable.
- `tests/parity/test_rules_parity_fixtures.py` now defines `_normalise_hash_for_ts_comparison(py_hash: str)`, which strips the `:must_move=` segment (up to the next `#`, if present) before comparing against TS hashes.
- The following parity assertions use this normalisation:
  - `test_state_action_parity` – when comparing `BoardManager.hash_game_state(next_state)` to `tsNext.stateHash`.
  - `test_state_action_http_parity` – when comparing the `/rules/evaluate_move` response `state_hash` to `tsNext.stateHash`.
  - `test_replay_ts_trace_fixtures_and_assert_python_state_parity` – when comparing trace-step hashes.

This preserves the Python-only `must_move` visibility for debugging while making parity comparisons robust to the intentional extension.

---

## 3. New tests added in this task

To support the new contracts and ensure ongoing parity, the following tests were added or extended:

- `ai-service/tests/rules/test_mutators.py`:
  - Existing placement and movement mutator tests remain unchanged.
  - **New:** `test_capture_mutator_matches_game_engine_for_overtaking_capture`
    - Constructs a minimal overtaking-capture scenario (P1 attacker, P2 target, empty landing space).
    - Applies `GameEngine.apply_move` and `CaptureMutator.apply` to equivalent states and asserts equality of:
      - `board.stacks`, `board.markers`, `board.collapsed_spaces`, `board.eliminated_rings`.
      - `players`.
  - **New:** `test_line_mutator_matches_game_engine_for_process_line_board_and_players`
    - Uses a synthetic line (via `BoardManager.find_all_lines` monkeypatch) in `LINE_PROCESSING`.
    - Compares `LineMutator.apply` vs `GameEngine.apply_move` on board + players.
  - **New:** `test_territory_mutator_matches_game_engine_for_process_region_board_and_players`
    - Uses a synthetic disconnected region and border setup (via `BoardManager.find_disconnected_regions` and `get_border_marker_positions` monkeypatch) in `TERRITORY_PROCESSING`.
    - Compares `TerritoryMutator.apply` vs `GameEngine.apply_move` on board + players.

- `ai-service/tests/rules/test_default_engine_equivalence.py`:
  - Existing `test_default_engine_apply_move_matches_game_engine_for_place_ring` (unchanged).
  - **New:** `test_default_engine_apply_move_matches_game_engine_for_move_stack`
    - Uses `RingRiftEnv` to advance the game until a `MOVE_STACK` move appears.
    - Applies the move via `GameEngine.apply_move` and `DefaultRulesEngine.apply_move`.
    - Asserts equivalence of:
      - Board: stacks, markers, collapsed_spaces, eliminated_rings.
      - Players.
      - `current_player`, `current_phase`, `game_status`.

- `ai-service/tests/parity/test_rules_parity_fixtures.py`:
  - Introduced `_normalise_hash_for_ts_comparison` and updated hash comparisons as described in §2.1.

All of these tests currently pass, with the only remaining TS↔Python hash differences confined to the known, now-normalised `:must_move=` suffix.

---

## 4. Future refactor considerations

The original plan to create separate `ai-service/app/rules/validators/` and `ai-service/app/rules/mutators/` packages is now largely realised, and this task has tightened their contracts with `GameEngine`:

1. **Progressively promote mutator-driven orchestration**
   - For each move type, we now have mutator-level contracts and, for placement and movement (and now capture/line/territory), DefaultRulesEngine shadow contracts.
   - Once all move types are covered and stable across parity suites, we can:
     - Implement a mutator-first `DefaultRulesEngine.apply_move` that applies moves on a copied state using mutators.
     - Optionally call through to `GameEngine` only for high-level orchestration (phase/turn/victory) or retire that dependency entirely once everything is factored out.
   - Throughout this process, we will retain `GameEngine.apply_move` as a parity oracle in tests.

2. **Extend parity and unit tests further**
   - Add more scenario-based tests (e.g. multi-segment chain captures, complex line/territory interactions) that assert mutator ↔ engine equivalence.
   - Extend TS trace fixtures to cover additional decision phases and ensure `_normalise_hash_for_ts_comparison` continues to be the only tolerated difference.

3. **Keep documentation and contracts in sync**
   - As new move types and phases are wired into the mutator contracts, this document should be updated to:
     - List the move types covered by each shadow contract.
     - Reference the tests that enforce each contract.
     - Call out any intentional Python-only extensions (like `must_move=`) and how parity tests handle them.

Until those longer-term refactors are complete, the invariants remain:

- `GameEngine + BoardManager` are the canonical Python implementation.
- `DefaultRulesEngine` is a validator/mutator–aware adapter with strong contracts tying mutators back to `GameEngine`.
- TS↔Python parity tests treat the `:must_move=` hash suffix as an expected, explicit extension of the Python engine.
