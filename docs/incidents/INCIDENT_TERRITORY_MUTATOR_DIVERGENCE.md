# Incident: TerritoryMutator vs GameEngine Divergence on Forced Elimination

**Date discovered:** November 2025  
**Status:** Fixed and guarded by tests  
**Affected area:** Python rules engine shadow contracts, territory-processing moves, territory dataset generator

This document records the incident where the Python [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6) diverged from the canonical [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117) during territory-processing, causing crashes in the **territory dataset generator** CLI when running in mixed-engine mode.

---

## 1. Symptom

### 1.1 CLI failure in territory dataset generation

Running the territory dataset generator in **mixed** engine mode:

```bash
cd ai-service
python -m app.training.generate_territory_dataset \
  --num-games 10 \
  --output logs/debug.square8.mixed2p.10.jsonl \
  --board-type square8 \
  --engine-mode mixed \
  --num-players 2 \
  --max-moves 200 \
  --seed 42
```

would intermittently fail with a `RuntimeError` raised from [`DefaultRulesEngine.apply_move()`](ai-service/app/rules/default_engine.py:285) while exercising territory moves under mutator shadow contracts. The representative error was:

```text
RuntimeError: TerritoryMutator diverged from GameEngine.apply_move
for territory-processing move: board.stacks mismatch
(move=id='eliminate-5,5', type=ELIMINATE_RINGS_FROM_STACK,
 player=1, from=None, to=5,5)
```

Key characteristics:

- The crash occurred during internal shadow-contract checks in [`DefaultRulesEngine`](ai-service/app/rules/default_engine.py:23), not in the canonical [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117) result.
- The mismatch was specifically reported on `board.stacks` for an `ELIMINATE_RINGS_FROM_STACK` territory-processing move.
- When this runtime error surfaced inside the dataset generator, the CLI process exited non‑zero and no JSONL output was produced (or it was truncated).

### 1.2 Reproduction

The pattern is captured and reproducible via the focused test:

- [`test_territory_forced_elimination_divergence_pattern`](ai-service/tests/test_territory_forced_elimination_divergence.py:188)

This test constructs a synthetic `GameState` in `TERRITORY_PROCESSING` where:

- Player 1 (`P1`) has a single stack at `5,5` and no rings in hand.
- Player 2 (`P2`) has a single stack at `0,0`, fully surrounded by collapsed spaces so that `P2` has **no legal movement or capture**, and no rings in hand.
- The current player is `P1`, and the only territory-processing decision available is for `P1` to **self-eliminate** from their own stack via `ELIMINATE_RINGS_FROM_STACK`.

The test then compares:

- The canonical path: `GameEngine.apply_move(state, move)`.
- The mutator path: `TerritoryMutator().apply(mutator_state, move)`.

It asserts:

- The explicit elimination on `P1`’s stack at `5,5` matches on both paths.
- The canonical path has strictly **more** `total_rings_eliminated` and no remaining stack at `0,0` (due to an additional forced elimination on `P2`).
- The mutator path leaves `P2`’s stack at `0,0` intact.

---

## 2. Root cause

### 2.1 Rules-level behaviour

At the level of game rules and phases, the problematic sequence is:

1. The game is in `TERRITORY_PROCESSING` for `P1` (the moving player).
2. There are **no further disconnected regions** that satisfy the self-elimination prerequisite (Q23) for `P1`, but `P1` still controls at least one stack.
3. The territory-processing decision surface exposes **explicit self-elimination moves**:
   - In TS, via `eliminate_rings_from_stack` moves enumerated by [`enumerateTerritoryEliminationMoves()`](src/shared/engine/territoryDecisionHelpers.ts:402).
   - In Python, via `ELIMINATE_RINGS_FROM_STACK` moves enumerated by [`GameEngine._get_territory_processing_moves()`](ai-service/app/game_engine.py:1905).
4. `P1` chooses to eliminate from their own stack at `5,5`, paying the self-elimination cost required by Q23.
5. After applying this move, turn logic **advances to the next player**. If that next player (`P2`) controls stacks but has:
   - no legal placements,
   - no legal movements, and
   - no legal captures,

   then the rules require a **forced elimination** on that player, eliminating a cap from one of their own stacks (the host-level “blocked player must self-eliminate” rule).

In other words:

- The **explicit** decision is `P1`’s self-elimination (`ELIMINATE_RINGS_FROM_STACK`).
- The **implicit/host-level** consequence is a forced elimination of `P2`’s stack during end-of-turn processing for `P2`, because `P2` is blocked.

### 2.2 Code-level interplay

#### Canonical engine path (Python)

1. The canonical engine [`GameEngine`](ai-service/app/game_engine.py:33) enumerates territory decisions via:
   - Region and elimination enumeration: [`_get_territory_processing_moves()`](ai-service/app/game_engine.py:1905).
   - For `ELIMINATE_RINGS_FROM_STACK`:
     - The move id is of the form `eliminate-{pos.to_key()}`.
     - The move is applied inside [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117), which delegates to [`GameEngine._apply_forced_elimination()`](ai-service/app/game_engine.py:2988) when `move.type == MoveType.ELIMINATE_RINGS_FROM_STACK`.

2. After applying the explicit elimination, [`GameEngine._update_phase()`](ai-service/app/game_engine.py:471) is invoked:
   - For `MoveType.ELIMINATE_RINGS_FROM_STACK`, `_update_phase` calls [`GameEngine._end_turn()`](ai-service/app/game_engine.py:624) directly:
     - See the explicit branch at [`_update_phase` lines handling `ELIMINATE_RINGS_FROM_STACK`](ai-service/app/game_engine.py:563).

3. Inside [`_end_turn`](ai-service/app/game_engine.py:624):
   - The turn rotates to the next player in table order.
   - For each candidate next player:
     - Computes whether the player has stacks (`BoardManager.get_player_stacks`).
     - Computes whether the player has any legal placements/movements/captures via [`_has_valid_actions()`](ai-service/app/game_engine.py:1758).
   - If the next player controls at least one stack but has **no legal actions**, `_end_turn` calls [`_perform_forced_elimination_for_player()`](ai-service/app/game_engine.py:1817).

4. [`_perform_forced_elimination_for_player`](ai-service/app/game_engine.py:1817):
   - Enumerates forced elimination moves via [`_get_forced_elimination_moves()`](ai-service/app/game_engine.py:1775).
   - Applies the first such move via [`_apply_forced_elimination()`](ai-service/app/game_engine.py:2988).
   - Calls [`_check_victory()`](ai-service/app/game_engine.py:334) to see if victory thresholds have been reached.

Resulting behaviour in the synthetic test case:

- `P1`’s explicit `ELIMINATE_RINGS_FROM_STACK` at `5,5` is applied.
- `_end_turn` rotates to `P2`, detects that `P2` is blocked, and applies an additional forced elimination to `P2`’s stack at `0,0`.
- The canonical `GameEngine` state ends with:
  - `board.stacks` missing the `0,0` stack.
  - `total_rings_eliminated` strictly greater than the mutator-only path.

#### TerritoryMutator and DefaultRulesEngine

- [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6) is a thin façade around the engine:

  ```python
  class TerritoryMutator(Mutator):
      def apply(self, state: GameState, move: Move) -> None:
          from app.game_engine import GameEngine
          if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
              GameEngine._apply_forced_elimination(state, move)
          else:
              GameEngine._apply_territory_claim(state, move)

          state.last_move_at = move.timestamp
  ```

  Critically, it does **not** call [`GameEngine._update_phase()`](ai-service/app/game_engine.py:471) or [`GameEngine._end_turn()`](ai-service/app/game_engine.py:624). It only applies the **explicit** elimination step.

- [`DefaultRulesEngine.apply_move()`](ai-service/app/rules/default_engine.py:285) maintains per-move **shadow contracts** for mutators:
  - For territory moves:

    ```python
    elif move.type in (
        MoveType.PROCESS_TERRITORY_REGION,
        MoveType.ELIMINATE_RINGS_FROM_STACK,
        MoveType.TERRITORY_CLAIM,
        MoveType.CHOOSE_TERRITORY_OPTION,
    ):
        mutator_state = state.model_copy(deep=True)
        territory_mutator = TerritoryMutator()
        territory_mutator.apply(mutator_state, move)

        # ... compare mutator_state vs next_via_engine on board.stacks, markers,
        # collapsed_spaces, eliminated_rings, and players ...
    ```

  - `next_via_engine` is the canonical result from [`GameEngine.apply_move(state, move)`](ai-service/app/game_engine.py:117), which includes:
    - The explicit elimination, and
    - Any host-level forced elimination performed during `_end_turn`.

This mismatch in responsibilities leads directly to:

- `mutator_state.total_rings_eliminated` < `next_via_engine.total_rings_eliminated`.
- `mutator_state.board.stacks` still containing the `P2` stack at `0,0` while `next_via_engine.board.stacks` does not.
- The strict equality checks inside `DefaultRulesEngine.apply_move` detecting a `board.stacks` mismatch and raising:

  ```text
  TerritoryMutator diverged from GameEngine.apply_move
  for territory-processing move: board.stacks mismatch (...)
  ```

### 2.3 Relationship to TS shared helpers

This pattern is consistent with the intended semantics captured in the TS shared engine:

- Region detection and Q23 gating live in [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:1) and [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1).
- Explicit territory decisions (region selection and elimination from a stack) are modelled as `process_territory_region` and `eliminate_rings_from_stack` moves via [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1).
- Forced elimination for a **blocked** next player is a host-level concern handled in TS turn logic (backend `GameEngine` / `TurnEngine` equivalents), not part of the explicit territory decision mutators.

The Python mutator contracts were unintentionally holding `TerritoryMutator.apply` responsible for **both** the explicit decision and the subsequent host-level forced elimination, which is beyond its design scope.

---

## 3. Fix

### 3.1 Escape hatch for host-level forced elimination

The fix adds a targeted **escape hatch** in the territory branch of [`DefaultRulesEngine.apply_move`](ai-service/app/rules/default_engine.py:285) that mirrors the existing movement escape hatch.

After applying `TerritoryMutator` on a deep copy (`mutator_state`) and computing the canonical `next_via_engine` via `GameEngine.apply_move`, the engine now checks:

```python
forced_elimination_occurred = (
    move.type == MoveType.ELIMINATE_RINGS_FROM_STACK
    and next_via_engine.total_rings_eliminated
    > mutator_state.total_rings_eliminated
)
```

- If `forced_elimination_occurred` is **true**, `DefaultRulesEngine`:
  - Treats the extra eliminations as host-level behaviour owned by `GameEngine._end_turn` / `_perform_forced_elimination_for_player`.
  - **Skips** the strict equality checks on `board.stacks`, `board.markers`, `board.collapsed_spaces`, `board.eliminated_rings`, and `players` for this move.
  - Still returns the canonical `next_via_engine` state to callers.

- If `forced_elimination_occurred` is **false** (including all `PROCESS_TERRITORY_REGION` moves and `ELIMINATE_RINGS_FROM_STACK` moves where no extra host-level elimination occurs):
  - The strict per-move contract remains enforced: any divergence between `mutator_state` and `next_via_engine` on board or players still raises a `RuntimeError`.

This behaviour is directly analogous to the **movement escape hatch** earlier in the file:

- Movement mutators also skip strict comparison when `GameEngine.apply_move` performs additional phase/turn side effects (such as forced elimination) that an atomic movement mutator does not reproduce.

### 3.2 Mutator-first mode remains strict

The mutator-first orchestration path in [`_apply_move_with_mutators`](ai-service/app/rules/default_engine.py:147):

- Uses the same [`GameEngine._apply_forced_elimination`](ai-service/app/game_engine.py:2988) helper as the canonical engine when processing `ELIMINATE_RINGS_FROM_STACK` and `FORCED_ELIMINATION` moves.
- Explicitly calls [`GameEngine._update_phase()`](ai-service/app/game_engine.py:471) and `GameEngine._check_victory()` on the mutator-driven state.
- Compares the **full resulting state** against `next_via_engine` (board, players, `current_player`, `current_phase`, `game_status`, `chain_capture_state`, `must_move_from_stack_key`).

Because mutator-first orchestration already mirrors the turn/phase and forced-elimination logic, it **does not need** the same escape hatch. Any divergence there remains a hard error.

---

## 4. Tests guarding the fix

### 4.1 Unit tests for the divergence pattern

All of the following live in [`test_territory_forced_elimination_divergence.py`](ai-service/tests/test_territory_forced_elimination_divergence.py:1):

1. [`test_territory_forced_elimination_divergence_pattern`](ai-service/tests/test_territory_forced_elimination_divergence.py:188)
   - Constructs a synthetic territory-processing state where:
     - `P1` has a stack at `5,5` and no rings in hand.
     - `P2` has a stack at `0,0` completely surrounded by collapsed spaces and no rings in hand.
   - Applies `ELIMINATE_RINGS_FROM_STACK` to `P1`’s stack via both:
     - `GameEngine.apply_move(state, move)`
     - `TerritoryMutator().apply(mut_state, move)`
   - Asserts:
     - Both paths agree on the explicit elimination at `5,5`.
     - `next_via_engine.total_rings_eliminated` is strictly greater than `mut_state.total_rings_eliminated`.
     - `P2`’s stack at `0,0` exists only in the mutator path (canonical path has already eliminated it via host-level forced elimination).

2. [`test_default_engine_no_forced_elim_strict_contract`](ai-service/tests/test_territory_forced_elimination_divergence.py:219)
   - Constructs a non-forced-elimination state where the next player has rings in hand (and therefore **no** forced elimination should occur).
   - Asserts:
     - `canonical = GameEngine.apply_move(state, move)`
     - `next_state = DefaultRulesEngine().apply_move(state, move)` matches `canonical` exactly.
     - No escape hatch is triggered; strict per-move equality remains in force.

3. [`test_default_engine_forced_elim_escape_hatch`](ai-service/tests/test_territory_forced_elimination_divergence.py:236)
   - Reuses the forced-elimination divergence state from test #1.
   - Asserts:
     - `canonical = GameEngine.apply_move(state, move)`
     - `next_state = DefaultRulesEngine().apply_move(state, move)` equals `canonical`.
     - No `RuntimeError` is raised despite the known `board.stacks` / `total_rings_eliminated` divergence between `TerritoryMutator`’s atomic application and the full canonical engine.

Together, these tests:

- Lock in the **expected difference** between mutator-only and canonical paths for `ELIMINATE_RINGS_FROM_STACK` when host-level forced elimination occurs.
- Ensure the escape hatch is **only** applied in that case; strict contracts remain for all other territory moves.

### 4.2 CLI smoke test for the territory generator

The CLI smoke test [`test_generate_territory_dataset_mixed_smoke`](ai-service/tests/test_generate_territory_dataset_smoke.py:15) drives the full territory dataset generator in mixed mode:

- Runs:

  ```bash
  python -m app.training.generate_territory_dataset \
    --num-games 10 \
    --output rr-territory-debug.jsonl \
    --board-type square8 \
    --engine-mode mixed \
    --num-players 2 \
    --max-moves 200 \
    --seed 42
  ```

- Asserts:
  - Exit code is `0`.
  - Stderr **does not contain** `"TerritoryMutator diverged from GameEngine.apply_move"`.
  - The output file is created and non-empty.

This test reproduces the historical failure mode (a crash in mixed mode) and now serves as a **regression guard** to ensure future changes do not reintroduce TerritoryMutator/GameEngine divergence errors into the training CLI.

### 4.3 Existing parity and mutator-equivalence suites

The fix is also validated indirectly by the broader parity and mutator equivalence tests documented in [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:180), including:

- Python vs TS trace fixtures:
  - [`ai-service/tests/parity/test_rules_parity_fixtures.py`](ai-service/tests/parity/test_rules_parity_fixtures.py:1) ensures [`GameEngine.apply_move`](ai-service/app/game_engine.py:117) matches TS trace fixtures (hash + S-invariant parity).
  - Additional tests ensure [`DefaultRulesEngine.apply_move`](ai-service/app/rules/default_engine.py:285) remains in full-state lockstep with `GameEngine.apply_move` for all moves in the traces.
- Mutator-first scenario tests:
  - [`ai-service/tests/rules/test_default_engine_mutator_first_scenarios.py`](ai-service/tests/rules/test_default_engine_mutator_first_scenarios.py:1) exercises `mutator_first=True` scenarios, including synthetic territory-processing setups, and asserts equality with the canonical engine.

All of these suites remained green after the escape hatch was introduced, confirming that:

- The change is **localized** to the per-move shadow contract for `ELIMINATE_RINGS_FROM_STACK`.
- Global parity between Python and TS engines is preserved.

---

## 5. Known unrelated issues (out of scope)

During the investigation, we also observed separate, **unrelated** issues:

- A failing hex-board DescentAI test involving `MagicMock` usage against internal methods, which interferes with property access and causes brittle failures in `ai-service/tests/*` when feature flags or environment configuration change.
- These failures:
  - Do **not** involve `TerritoryMutator`, `DefaultRulesEngine`, or `GameEngine._apply_forced_elimination`.
  - Do **not** affect the territory dataset generator or territory-processing semantics.
  - Remain tracked in the general AI/engine test backlog (see [`docs/supplementary/AI_IMPROVEMENT_BACKLOG.md`](supplementary/AI_IMPROVEMENT_BACKLOG.md:1) and [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md:1)).

They are explicitly **out of scope** for this incident and should be treated as separate engineering tasks.

---

## 6. Follow-up and roadmap hooks

The escape hatch fixes the immediate crash and correctly models host-level forced elimination as **outside** the per-move `TerritoryMutator` contract. Medium-term hardening work should focus on:

- Strengthening TS↔Python parity for territory decision helpers and forced-elimination sequences:
  - TS: [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1).
  - Python: [`BoardManager.find_disconnected_regions`](ai-service/app/board_manager.py:171), [`GameEngine._apply_territory_claim`](ai-service/app/game_engine.py:2718), [`GameEngine._perform_forced_elimination_for_player`](ai-service/app/game_engine.py:1817).
- Adding property-based tests around territory invariants and forced-elimination behaviour in both TS and Python.
- Introducing dataset-level validation for the combined-margin territory dataset to catch impossible or inconsistent `(territory_margin, elim_margin)` combinations early.

These items are captured and prioritized in the AI/rules/training section of the strategic roadmap; see [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:130) for the consolidated checklist.
