# RingRift Rules Engine Architecture & Rollout Strategy

**Last Updated:** November 21, 2025
**Scope:** Python Rules Engine, TypeScript Parity, and Rollout Plan

This document defines the architecture of the Python rules engine within the AI service, its relationship to the canonical TypeScript engine, and the strategy for rolling it out as an authoritative validation source.

---

## 1. Architecture Overview

### Core Concept

RingRift maintains two implementations of the game rules:

1.  **TypeScript Engine (Canonical):** Located in `src/shared/engine/`. Used by the Node.js backend (`GameEngine.ts`) and the client sandbox (`ClientSandboxEngine.ts`). This is the current source of truth.
2.  **Python Engine (AI/Shadow):** Located in `ai-service/app/game_engine.py`. Used for AI search/evaluation and currently being rolled out as a shadow validator for the backend.

### Python Engine Structure

- `ai-service/app/game_engine.py`: Core orchestration, move generation, state transitions, and turn logic.
- `ai-service/app/board_manager.py`: Board-level utilities (hashing, S-invariant, line detection, territory regions).
- `ai-service/app/rules/default_engine.py`: Adapter that delegates to `GameEngine` while routing key move types through dedicated mutators under _shadow contracts_.
- `ai-service/app/models/`: Pydantic models mirroring the TypeScript shared engine types.

### TypeScript Integration

- `src/server/game/RulesBackendFacade.ts`: The primary entry point for the backend. It abstracts the choice between the local TS engine and the Python service based on `RINGRIFT_RULES_MODE`.
- `src/server/services/PythonRulesClient.ts`: Handles HTTP communication with the Python AI service (`/rules/evaluate_move`).

### Parity Mechanisms

To ensure the Python engine behaves exactly like the TypeScript engine:

- **Shared Types:** Domain models (`GameState`, `Move`, etc.) are structurally identical.
- **Unified Move Model:** Both engines use the same extended Move types (`continue_capture_segment`, `process_line`, `process_territory_region`) for complex phases.
- **Hashing:** `hash_game_state` produces identical output (modulo known extensions like `:must_move=`) for state verification.
- **S-Invariant:** Both engines compute the same progress metric (`S = markers + collapsed + eliminated`) to guarantee termination.

---

## 2. Python ↔ TypeScript Parity Mapping

| Feature       | TypeScript Implementation                   | Python Implementation               | Status    |
| :------------ | :------------------------------------------ | :---------------------------------- | :-------- |
| **Movement**  | Ray-based geometry, stack-height distance   | Mirrors TS geometry and constraints | ✅ Parity |
| **Captures**  | Overtaking with cap-height check            | Mirrors TS validation logic         | ✅ Parity |
| **Chains**    | `ChainCaptureState`, mandatory continuation | `ChainCaptureState`, same logic     | ✅ Parity |
| **Placement** | Multi-ring, no-dead-placement check         | Mirrors TS placement rules          | ✅ Parity |
| **Lines**     | Exact vs Overlength rewards                 | Mirrors TS line processing          | ✅ Parity |
| **Territory** | Disconnection, self-elimination prereq      | Mirrors TS territory logic          | ✅ Parity |
| **Victory**   | Ring elimination, territory control         | Mirrors TS victory checks           | ✅ Parity |

**Verification:**

- **Trace Parity:** `tests/unit/Python_vs_TS.traceParity.test.ts` consumes Python-generated vectors.
- **Fixture Parity:** `ai-service/tests/parity/test_rules_parity_fixtures.py` validates TS-generated fixtures against the Python engine.

### DefaultRulesEngine ↔ GameEngine Equivalence Coverage

To safely evolve `DefaultRulesEngine` toward a mutator-driven architecture, we
maintain explicit equivalence tests that assert its behaviour stays in
lockstep with `GameEngine.apply_move` for key move families.

| Move Family                 | Move Types                   | Test File                                                   | Scenario Source                                                            |
| :-------------------------- | :--------------------------- | :---------------------------------------------------------- | :------------------------------------------------------------------------- |
| **Placement**               | `place_ring`                 | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven via `RingRiftEnv` on SQUARE8 + SQUARE19                         |
| **Movement**                | `move_stack`                 | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven via `RingRiftEnv` until first movement                          |
| **Capture – Initial**       | `overtaking_capture`         | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic overtaking segment + env-driven capture search                   |
| **Capture – Continuation**  | `continue_capture_segment`   | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven: apply first capture, then continue chain                       |
| **Line Processing**         | `process_line`               | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic line via `BoardManager.find_all_lines` monkeypatch               |
| **Territory – Region**      | `process_territory_region`   | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic disconnected region via `BoardManager.find_disconnected_regions` |
| **Territory – Elimination** | `eliminate_rings_from_stack` | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic single capped stack via `_get_territory_processing_moves`        |

Additionally, TS-generated trace fixtures are replayed through both engines:

- `ai-service/tests/parity/test_rules_parity_fixtures.py`:
  - `test_replay_ts_trace_fixtures_and_assert_python_state_parity` asserts hash + S-invariant parity for `GameEngine.apply_move` against TS.
  - `test_default_engine_matches_game_engine_when_replaying_ts_traces` asserts
    full-state lockstep between `DefaultRulesEngine.apply_move` and
    `GameEngine.apply_move` for every move in the TS traces (captures,
    line-processing, territory, etc.).
  - `test_default_engine_mutator_first_matches_game_engine_on_ts_traces` runs
    the same traces with `DefaultRulesEngine(mutator_first=True)`, exercising
    the full mutator-first orchestration path while still comparing the
    resulting state against `GameEngine.apply_move`.

- `ai-service/tests/rules/test_default_engine_mutator_first_scenarios.py`:
  - `test_mutator_first_env_smoke_for_place_ring_and_move_stack` uses
    `RingRiftEnv` to find realistic `PLACE_RING` and `MOVE_STACK` moves and
    asserts that mutator-first mode stays aligned with `GameEngine.apply_move`.
  - `test_mutator_first_process_territory_region_synthetic` mirrors the
    synthetic disconnected-region scenario from the equivalence tests but with
    `mutator_first=True`, ensuring territory processing plus downstream
    forced-elimination are consistent.

- `ai-service/tests/rules/test_default_engine_flags.py` verifies configuration
  of the mutator-first mode itself (see below).

These tests form the safety net for any future refactors that change
`DefaultRulesEngine` from a pure adapter into a mutator-first orchestrator.

### Mutator-First Mode and Configuration

`DefaultRulesEngine` exposes an optional _mutator-first_ execution path that
mirrors `GameEngine.apply_move` while delegating board/player mutations to the
Python mutators. This is currently used **only as a shadow contract**; the
canonical state returned from `apply_move` still comes from `GameEngine`.

#### Configuration surfaces

Mutator-first behaviour is controlled by three inputs, evaluated in a fixed
order:

1. **Server-level gate (ops-owned):**
   - `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST`
   - If this is not truthy, mutator-first is **hard-disabled**, regardless of
     any per-service flags or constructor arguments.
2. **Per-service default (AI service):**
   - `RINGRIFT_RULES_MUTATOR_FIRST`
   - Only consulted when the server gate is truthy and the constructor
     argument is omitted.
3. **Constructor override (code-level):**
   - `DefaultRulesEngine(mutator_first=...)`
   - Wins over the per-service env flag whenever the server gate allows
     mutator-first.

#### Constructor argument

```python
engine = DefaultRulesEngine(mutator_first=True)   # request enable (if gate allows)
engine = DefaultRulesEngine(mutator_first=False)  # explicitly disable
engine = DefaultRulesEngine()                     # defer to env flag
```

#### Environment variables

- **Server-level gate:**
  - `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST` is read in `__init__`.
  - Truthy values (case-insensitive) _permit_ mutator-first mode:
    - `"1"`, `"true"`, `"yes"`, `"on"`
  - Any other value (or unset) means mutator-first is **disabled**, even if
    the per-service flag or constructor requests it.

- **Per-service default:**
  - `RINGRIFT_RULES_MUTATOR_FIRST` is read once in `__init__` when
    `mutator_first` is omitted **and** the server gate is truthy.
  - Truthy values (case-insensitive) enable mutator-first mode by default:
    - `"1"`, `"true"`, `"yes"`, `"on"`
  - Any other value (or unset) leaves mutator-first disabled.

Constructor arguments always override the per-service env flag, but are still
subject to the server-level gate.

The behaviour of this configuration model is covered by
`ai-service/tests/rules/test_default_engine_flags.py`.

#### Effective behaviour matrix

The following examples illustrate how the three inputs combine (all env values
are shown after lowercasing):

- **Server gate falsey / unset:** mutator-first is always disabled.
  - `SERVER=""`, `ENV="true"`, `ctor=True` → disabled
  - `SERVER="0"`, `ENV="on"`, `ctor=None` → disabled
- **Server gate truthy, env truthy:**
  - `SERVER="1"`, `ENV="true"`, `ctor=None` → enabled (per-service default)
  - `SERVER="1"`, `ENV="true"`, `ctor=False` → disabled (constructor wins)
  - `SERVER="1"`, `ENV="true"`, `ctor=True` → enabled (constructor wins)
- **Server gate truthy, env falsey / unset:**
  - `SERVER="1"`, `ENV=""`, `ctor=None` → disabled (no default)
  - `SERVER="1"`, `ENV=""`, `ctor=True` → enabled (constructor opt-in)

This matches the semantics in `DefaultRulesEngine.__init__`:

1. If the server gate is not truthy → `_mutator_first_enabled = False`.
2. Else if the constructor argument is provided → use that value.
3. Else → use the per-service env flag.

#### Runtime behaviour when enabled

- `DefaultRulesEngine.apply_move` always computes a canonical result via
  `GameEngine.apply_move(state, move)`.
- It then runs `_apply_move_with_mutators(state, move)`, which:
  - Performs copy-on-write of the board/state.
  - Manages zobrist hashing (player/phase contributions) in the same pattern
    as `GameEngine.apply_move`.
  - Delegates to the specialised mutators for each move family
    (`PlacementMutator`, `MovementMutator`, `CaptureMutator`, `LineMutator`,
    `TerritoryMutator`) and uses `GameEngine._apply_forced_elimination` for
    forced/cap elimination.
  - Updates `move_history`, `last_move_at`, `must_move_from_stack_key`, calls
    `GameEngine._update_phase`, reapplies zobrist contributions, and finally
    invokes `GameEngine._check_victory`.
- The resulting mutator-first state is then compared against the canonical
  `GameEngine.apply_move` result on:
  - `board.stacks`, `board.markers`, `board.collapsed_spaces`,
    `board.eliminated_rings`
  - `players`
  - `current_player`, `current_phase`, `game_status`
  - `chain_capture_state`, `must_move_from_stack_key`
- Any divergence raises a `RuntimeError` with a message that includes a
  compact move description and summary statistics (e.g. counts of stacks or
  markers) to aid debugging.

Because `apply_move` still **returns** the canonical `GameEngine` state, this
mode is safe to enable in shadow for diagnostics without changing external
behaviour.

---

## 3. Rollout Strategy

The rollout is controlled by the `RINGRIFT_RULES_MODE` environment variable.

### Phase 1: Shadow Mode (`RINGRIFT_RULES_MODE=shadow`)

- **Behavior:** TS engine is authoritative. For every move, the backend asynchronously calls the Python service (`/rules/evaluate_move`) to compare results.
- **Metrics:** Mismatches in validity, state hash, or S-invariant are logged and increment Prometheus counters (`rules_parity_*_mismatch_total`).
- **Goal:** Verify parity at scale in staging/production without affecting gameplay.

### Phase 2: Python Authoritative (`RINGRIFT_RULES_MODE=python`)

- **Behavior:** The backend consults the Python service _first_.
  - If Python rejects the move, it is rejected.
  - If Python accepts, the move is applied (currently via TS engine in "reverse shadow" for safety, eventually by trusting Python state).
- **Fallback:** On Python service failure, fall back to TS engine and log `backend_fallback`.
- **Goal:** Make the Python engine the single source of truth for validation, enabling advanced AI features that rely on precise rule simulation.

### Acceptance Criteria for Phase 2

1.  **Shadow Stability:** Zero parity mismatches over a significant period in Phase 1.
2.  **Performance:** Python service latency (P99) is within acceptable bounds (< 50ms).
3.  **Operational Readiness:** AI service is horizontally scalable and monitored.

---

## 4. Future Refactoring

The Python engine is evolving towards a purely functional, mutator-driven architecture:

1.  **Mutator Extraction:** Logic is being moved from `GameEngine` methods into dedicated `Mutator` classes (e.g., `PlacementMutator`, `CaptureMutator`).
2.  **Shadow Contracts:** `DefaultRulesEngine` currently enforces that Mutators produce the same result as `GameEngine`.
3.  **Goal:** Eventually replace `GameEngine` orchestration with a composable pipeline of Validators and Mutators, matching the proposed TypeScript refactoring.

---

## 5. Canonical Territory, Q23, Elimination, and S-Invariant Semantics

This section documents the canonical semantics for **territory disconnection**, **Q23 self-elimination**, **elimination bookkeeping**, and the **S-invariant** as implemented by the TypeScript sandbox and shared engine. The backend GameEngine and Python engine are required to match these behaviours.

### 5.1 Territory Regions and Disconnection

**Reference implementations:**

- Client sandbox: `src/client/sandbox/sandboxTerritory.ts`, `sandboxTerritoryEngine.ts`.
- Shared engine: `src/shared/engine/territoryDetection.ts`, `TerritoryMutator.ts`.
- Backend helpers: `src/server/game/BoardManager.ts`, `src/server/game/rules/territoryProcessing.ts`.

**Board geometries:**

- `square8`, `square19`: orthogonal (Von Neumann) adjacency for regions.
- `hex`: hex adjacency based on the coordinate system in `boardMovementGrid.ts`.

**Disconnected region:**

- A **region** is a maximal connected set of empty spaces on the board under the geometry's adjacency.
- A region is **disconnected** for a given player if:
  - It is completely enclosed by that player's border markers and/or board edges, and
  - It does not touch any other reachable empty spaces outside the border (again with geometry-appropriate adjacency).
- `BoardManager.findDisconnectedRegions` computes these regions. The canonical definition is: _"all empty spaces that are no longer connected to the global exterior via empty-space adjacency, assuming the current marker layout"_.

The shared `territoryDetection` implementation is the normative algorithm for detecting such regions; backend and Python code must either call it or faithfully mirror its behaviour.

### 5.2 Q23: Self-Elimination Prerequisite

**Intent (FAQ Q23):**

> A player may only collapse and score a disconnected territory region they control if they have at least one stack or cap **outside** that region from which the mandatory self-elimination cost can be paid.

We distinguish two cases for the controlling (moving) player:

- **Negative Q23 (not eligible):**
  - The player has **no** stacks/caps anywhere outside the region.
  - Result: the region **must not be processed** at all for that player.
    - No interior spaces collapse.
    - No interior stacks are eliminated for that player.
    - No self-elimination is applied.

- **Positive Q23 (eligible):**
  - The player has **at least one** stack or cap **outside** the region.
  - Result: the region **may be processed** for that player:
    - All interior empty spaces collapse into controlled territory.
    - All interior stacks belonging to any player are eliminated according to the elimination rules (below).
    - The controlling player pays a self-elimination cost from outside the region.

**Canonical implementation notes:**

- Eligibility is computed using `BoardManager.getPlayerStacks(board, player)` over the _current_ board state, and then partitioning those stacks into **inside** vs **outside** the region.
- Q23 is applied **per region** during disconnected-region processing:
  - If the moving player has no outside stacks, that region is skipped.
  - If they do, the region is eligible and can be processed.
- The sandbox and shared engine do **not** treat rings in hand as satisfying Q23; only physical stacks/caps on the board count.

### 5.3 Elimination Semantics

Elimination in RingRift falls into three broad categories:

1. **Self-elimination (territory cost):**
   - Occurs when a player processes a territory region they control under positive Q23.
   - The cost is paid by removing one of the player's own stacks or caps **outside** the region.
   - Canonical behaviour (sandbox + `CaptureMutator.mutateEliminateStack`):
     - If a **cap** is chosen, the **entire capped stack** is removed and the player is credited with `capHeight` eliminated rings.
     - If a plain stack (no cap) is chosen, its entire height is removed and contributes that many rings.

2. **Internal eliminations (inside the region):**
   - All stacks inside the collapsing territory region are eliminated when the region is processed.
   - Each eliminated stack contributes its full height (or cap height when relevant) to `eliminatedRings`.
   - Ownership of eliminated stacks determines how scores/metrics are attributed, but for S-invariant purposes only the count of eliminated rings matters.

3. **Opponent eliminations (captures, lines):**
   - Governed by capture and line rules elsewhere (see capture/line sections and tests), but they also contribute to `eliminatedRings`.
   - Self-elimination and opponent elimination share the same underlying elimination mutator (`CaptureMutator` / Python equivalents) to maintain consistent bookkeeping.

Across all categories:

- The **canonical rule** is that when a stack is eliminated, the number of rings added to `eliminatedRings` equals the number of markers removed from the board for that stack (cap height or full stack height).
- Territory processing must use the same elimination pipeline as captures/lines; no bespoke deletion logic is allowed.

### 5.4 S-Invariant

**Definition:**

> `S = markers + collapsedSpaces + eliminatedRings`

Where:

- `markers` = total count of non-empty marker cells on the board (all players).
- `collapsedSpaces` = number of spaces that have been converted into collapsed territory.
- `eliminatedRings` = cumulative count of rings removed from stacks via capture, self-elimination, line processing, or territory collapse.

**Properties:**

- In all canonical engines (sandbox, shared TS, Python), `S` is **non-decreasing** over the course of a game.
- AI simulations rely on non-decreasing `S` and stall detection rather than strict per-move increase.
- The baseline test `tests/unit/SInvariant.seed17FinalBoard.test.ts` asserts that for a reference game (seed 17) the final value of `S` is exactly `74`, and this value is shared across all engines.

**Guard-rail tests:**

- `tests/unit/SharedMutators.invariants.test.ts` verifies that shared mutators respect S-invariant properties.
- `tests/unit/GameEngine.aiSimulation.test.ts` (and debug variants) ensure that automatic consequence processing (lines + territory + forced elimination) never violates non-decreasing `S`.

Any change to territory, capture, or elimination logic must keep these invariants intact.

### 5.5 Worked Examples (Informal)

The following examples are encoded as tests and serve as _living specifications_:

- **Single-region Q23 positive (square19):**
  - Tests: `tests/unit/GameEngine.territoryDisconnection.test.ts`, `tests/unit/territoryProcessing.rules.test.ts`.
  - Scenario: a 3x3 interior region is fully surrounded by Player 1's markers. Player 1 has at least one stack outside the region.
  - Expected behaviour:
    - Region is detected as disconnected and eligible (positive Q23).
    - All 9 interior spaces collapse to territory; `board.collapsedSpaces` has value `1` at those coordinates.
    - All interior stacks are eliminated; `board.stacks` has no entries at those locations.
    - One outside Player 1 stack is self-eliminated according to `mutateEliminateStack` semantics.
    - `S` increases appropriately due to new `collapsedSpaces` and `eliminatedRings`.

- **Q23 negative (sandbox rules):**
  - Tests: `tests/unit/sandboxTerritory.rules.test.ts`, `tests/unit/sandboxTerritoryEngine.rules.test.ts` (Q23-specific cases).
  - Scenario: a region is fully enclosed by Player 1's markers, but Player 1 has no stacks or caps outside that region.
  - Expected behaviour:
    - Region is detected as disconnected but **ineligible** (negative Q23).
    - No collapse occurs; interior spaces remain normal empties.
    - No stacks are eliminated; `eliminatedRings` and `collapsedSpaces` do not change.

- **Seed 17 parity checkpoints:**
  - Tests: `tests/unit/Seed17GeometryParity.GameEngine_vs_Sandbox.test.ts`, `tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`, `tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts`.
  - These parity suites assert that for a difficult seed (17), backend and sandbox agree on:
    - Geometry and movement legality.
    - Territory disconnections and collapses.
    - Elimination counts and S-invariant at key move numbers, including the final board.

### 5.6 Backend Pipeline Ordering (Territory vs Lines)

The backend GameEngine and sandbox must agree on the **effective order** in which territory and line processing are applied after a move:

- Canonical intention:
  - Disconnected territory regions that become eligible as a result of a move should be processed in a way that is consistent with sandbox behaviour and Q23.
  - Line formation and capture consequences must also be applied, but must not create artefacts that violate Q23 semantics (e.g., erasing all outside stacks _before_ territory is evaluated when sandbox would not).

- Current backend implementation (TS):
  - `GameEngine.processAutomaticConsequences` calls into territory processing and line processing in a carefully chosen order so that:
    - Territory disconnection tests (`GameEngine.territoryDisconnection.*.test.ts`) match sandbox expectations for square8, square19, and hex.
    - Parity suites (`TerritoryParity.GameEngine_vs_Sandbox.test.ts`, Seed 17 parity tests) remain green.

Any future changes to this ordering **must**:

- Be validated first against the sandbox oracle tests (`ClientSandboxEngine.territoryDisconnection.*`, `sandboxTerritory.*`), and
- Maintain green status for the backend territory suites and parity checks listed above.

These semantics are the authoritative contract for later phases (AI parity, WebSocket/RulesBackendFacade integration, Python parity) and must remain stable unless intentionally revised alongside their tests.

---

## 6. References

- Sandbox territory & elimination:
  - `src/client/sandbox/ClientSandboxEngine.ts`
  - `src/client/sandbox/sandboxTerritory.ts`
  - `src/client/sandbox/sandboxTerritoryEngine.ts`
  - `src/client/sandbox/sandboxElimination.ts`
- Shared engine:
  - `src/shared/engine/territoryDetection.ts`
  - `src/shared/engine/mutators/TerritoryMutator.ts`
  - `src/shared/engine/mutators/CaptureMutator.ts`
- Backend:
  - `src/server/game/BoardManager.ts`
  - `src/server/game/rules/territoryProcessing.ts`
  - `src/server/game/GameEngine.ts`
- Python rules engine:
  - `ai-service/app/board_manager.py`
  - `ai-service/app/rules/mutators/territory.py`
  - `ai-service/app/rules/mutators/capture.py`

These components collectively define the canonical behaviour for territory, Q23, elimination, and S-invariant across all RingRift engines.
