# Active-No-Moves & Forced Elimination Behaviour Catalogue

> **Doc Status (2025-11-30): Active (behavioural catalogue, derived)**
>
> **Role:** Scenario-level description of how active-no-moves (ANM) and forced elimination behave across engines, used as input to the canonical rules spec and invariants.
>
> **SSoT alignment:** This file is a derived catalogue over:
>
> - Canonical rules spec [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1) – especially RR-CANON-R070–R072, R100, R170–R173 and the ANM cluster R2xx.
> - Invariants & parity framework [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1) – `INV-ACTIVE-NO-MOVES`, `INV-PHASE-CONSISTENCY`, `INV-TERMINATION`, `PARITY-TS-PY-ACTIVE-NO-MOVES`.
> - Implementation mapping [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:1) and rules engine architecture [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1).
> - Python strict-invariant and parity tests under [`ai-service/tests/invariants`](ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py:1) and [`ai-service/tests/parity`](ai-service/tests/parity/test_active_no_moves_line_processing_regression.py:1).
>
> Where this catalogue disagrees with RR-CANON or the shared TS engine, RR-CANON + code + tests win and this file must be updated.
>
> **Usage note:** Each scenario should cite the tests that exercise it; when adding or changing scenarios, include the primary Jest/pytest file name so readers can jump straight to coverage.

## 1. Core Concepts

- **Global legal action (for ANM):** For a state with `gameStatus == ACTIVE` and current player P, the union of:
  - Phase-local interactive moves currently exposed to P (placements, movements, captures, line decisions, territory decisions, victory acknowledgements) as defined in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:192).
  - Any legal **ring placements** P could make under [`RR-CANON-R080`–`R082`](RULES_CANONICAL_SPEC.md:220) given the current board and `ringsInHand[P]`, even when `currentPhase` is not `ring_placement`.
  - Any legal **forced-elimination** actions for P under [`RR-CANON-R100`](RULES_CANONICAL_SPEC.md:280) when P controls at least one stack but has no placements, movements, or captures.
  - Any legal **recovery actions** for P under [`RR-CANON-R110`–`R115`](RULES_CANONICAL_SPEC.md:579) when P controls no stacks, has no rings in hand, but has markers on the board and buried rings in other stacks.
- **Real action (for last-player-standing):** As in [`RR-CANON-R172`](RULES_CANONICAL_SPEC.md:438): a ring placement, non-capture movement, or overtaking capture on the player's own turn. **Recovery actions do NOT count as real actions** for LPS purposes - this creates strategic tension where players with rings in hand have a "survival budget." Forced eliminations also do **not** count as real actions.
- **Turn-material:** For ANM and phase/turn invariants, a player has turn-material if they either:
  - Control at least one stack (a top ring of their colour on some stack), or
  - Have `ringsInHand[P] > 0`.
    Captured/buried rings of P's colour inside opponents' stacks do not give P turn-material; they matter only for Territory, last-player-standing colour-representation rules, and **recovery action eligibility**.
- **Temporarily eliminated:** A player with no stacks and no rings in hand, but who has **markers on the board** and **buried rings** in other stacks, is "temporarily eliminated." Such a player may still take **recovery actions** (RR-CANON-R110–R115) and is NOT skipped by turn rotation. This is distinct from "fully eliminated."
- **Fully eliminated for turn rotation:** A player with **no** turn-material (no controlled stacks and `ringsInHand == 0`) **and no recovery action available** is fully eliminated for purposes of choosing `currentPlayer` in ACTIVE states. Such a player may still own buried rings for scoring, but is skipped by turn rotation; see [`INV-PHASE-CONSISTENCY`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:167).
- **Active-no-moves (ANM) state:** A (hypothetical) ACTIVE state where the current player P has turn-material or recovery eligibility but **no global legal actions** at all (no placements, movements, captures, recovery actions, line/territory decisions, or forced elimination). `INV-ACTIVE-NO-MOVES` requires that such states never persist in valid play.

The scenarios below capture concrete shapes that have historically exercised these concepts in TS and Python engines. Each scenario lists:

- A short narrative description.
- The relevant phases, player material, and legal actions.
- Expected canonical behaviour under RR-CANON (including the R2xx ANM cluster).
- Implementations and tests that currently exercise the behaviour.

## 2. Scenario Catalogue

### ANM-SCEN-01 – Movement phase, no moves but forced elimination available

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == movement` (or `capture` / `chain_capture`).
  - Current player P controls at least one stack.
  - `get_valid_moves(state, P)` (movement + capture) returns `[]`.
  - No legal ring placements for P under [`RR-CANON-R080`–`R082`](RULES_CANONICAL_SPEC.md:220) (either `ringsInHand[P] == 0` or all placements are illegal by no-dead-placement).
  - Forced-elimination moves exist for P per [`RR-CANON-R100`](RULES_CANONICAL_SPEC.md:280).
- **Historical issue:** Python strict-invariant `_assert_active_player_has_legal_action` originally looked only at phase-local interactive moves, so this shape was treated as violating “ACTIVE player has a legal move” even though forced elimination was available.
- **Expected canonical behaviour:**
  - The state is **not** ANM, because a global legal action exists (forced elimination).
  - Engines must surface a forced-elimination decision to the acting agent (rules intent: the player or AI chooses which eligible stack to eliminate); host-level auto-selection via a deterministic tie-breaker is **not** considered rules-canonical under [`RR-CANON-R206`](RULES_CANONICAL_SPEC.md:287) and is tracked as a temporary implementation compromise in [`KNOWN_ISSUES.md` P0.1](../KNOWN_ISSUES.md:39).
  - `currentPhase` after the elimination should follow [`RR-CANON-R070`–`R072`](RULES_CANONICAL_SPEC.md:192) exactly; no ACTIVE / MOVEMENT state with P and no global actions may persist.
- **Implementations / tests:**
  - Python invariant regression [`test_active_no_moves_movement_forced_elimination_regression.py`](ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py:1).
  - TS orchestrator soaks and `ACTIVE_NO_MOVES` checks in [`scripts/run-orchestrator-soak.ts`](scripts/run-orchestrator-soak.ts:596).
- **RR-CANON mapping:**
  - Legal-action coverage by `RR-CANON-R200` (global legal actions) and `RR-CANON-R203` (ACTIVE states must expose at least one global action).
  - Forced-elimination semantics by `RR-CANON-R072`, `RR-CANON-R100`, and `RR-CANON-R205`.

### ANM-SCEN-02 – Movement phase, placements-only global actions

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == movement`.
  - Current player P has at least one legal **ring placement** under [`RR-CANON-R080`–`R082`](RULES_CANONICAL_SPEC.md:220) (rings in hand and valid no-dead-placement targets).
  - `get_valid_moves(state, P)` (movement + capture) returns `[]`.
  - No forced-elimination moves are available because P has no controlled stacks (all of P’s rings are still in hand) or because stacks still have legal moves after placement.
- **Historical issue:** The original Python strict invariant only consulted `get_valid_moves` (phase-local interactive moves) plus forced elimination. Because placements are not exposed during MOVEMENT, the invariant incorrectly concluded that P had no legal actions and raised an error, even though P could act by placing rings on their next turn.
- **Expected canonical behaviour:**
  - The state is **not** ANM: global legal actions exist in the form of placements that P can take when their next `ring_placement` phase begins.
  - Engines must ensure that future turn rotation brings P back to `ring_placement` where these placements can be taken; they may not deadlock the game while such global actions exist.
  - Phase-local invariants may still treat the MOVEMENT micro-phase as “no local moves”, but global ANM invariants (`INV-ACTIVE-NO-MOVES`) must account for placement availability.
- **Implementations / tests:**
  - Python invariant regression [`test_active_no_moves_movement_placements_only_regression.py`](ai-service/tests/invariants/test_active_no_moves_movement_placements_only_regression.py:1).
  - TS short soaks and plateau diagnostics described in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:84).
- **RR-CANON mapping:**
  - Global placement availability in `RR-CANON-R200`.
  - ANM definition in `RR-CANON-R201` / `RR-CANON-R203` (global actions rather than phase-local only).

### ANM-SCEN-03 – Movement phase with a fully eliminated current player

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == movement`.
  - `currentPlayer == P` where P has **no stacks** and `ringsInHand[P] == 0` (no turn-material).
  - At least one other player Q still has stacks and/or rings in hand.
  - `get_valid_moves(state, P) == []` and `_get_forced_elimination_moves(state, P) == []`.
- **Historical issue:** Python strict-invariant snapshots captured states where a fully eliminated player remained `currentPlayer` in ACTIVE / MOVEMENT, even though other players still had material. This violated both ANM expectations and phase-consistency semantics compared to the TS TurnEngine.
- **Expected canonical behaviour:**
  - Players with no turn-material (no stacks and no rings in hand) must **not** remain `currentPlayer` in ACTIVE states.
  - Turn rotation (`_end_turn` / `advanceTurnAndPhase`) must skip such players and select the next player with turn-material (or terminate the game if none exist).
  - Any attempt to leave an ACTIVE / MOVEMENT state with a fully eliminated `currentPlayer` and no actions is invalid under `INV-ACTIVE-NO-MOVES` and `INV-PHASE-CONSISTENCY`.
- **Implementations / tests:**
  - Python regression [`test_active_no_moves_movement_fully_eliminated_regression.py`](ai-service/tests/invariants/test_active_no_moves_movement_fully_eliminated_regression.py:1).
  - Phase-consistency discussion in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:167).
- **RR-CANON mapping:**
  - Turn-material and fully eliminated players in `RR-CANON-R201` / `RR-CANON-R202`.
  - Turn rotation requirements in `RR-CANON-R070`–`RR-CANON-R072` and the ANM cluster.

### ANM-SCEN-04 – Territory processing with no remaining decisions

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == territory_processing`.
  - The moving player P has just applied a `process_territory_region` or related decision.
  - There are **no further territory-processing moves** for P (no more eligible regions, no pending self-elimination), and no forced-elimination moves.
  - P may or may not still have turn-material after the region is processed.
- **Historical issue:** Earlier Python versions could remain in ACTIVE / TERRITORY_PROCESSING with no legal moves for P after applying a territory decision, diverging from TS turn logic, which always advances out of `territory_processing` when no decisions remain.
- **Expected canonical behaviour:**
  - After each territory-processing decision, engines must recompute the territory decision surface for the moving player.
  - If no further territory decisions remain for P, the engine must either:
    - Apply forced elimination for P if required by [`RR-CANON-R072`–`R100`](RULES_CANONICAL_SPEC.md:209), or
    - Call end-of-turn rotation to the next player with turn-material, then evaluate victory (`RR-CANON-R170`–`R173`).
  - It is invalid to leave an ACTIVE / TERRITORY_PROCESSING state where the current player has no global legal actions.
- **Implementations / tests:**
  - Python regression [`test_active_no_moves_territory_processing_regression.py`](ai-service/tests/invariants/test_active_no_moves_territory_processing_regression.py:1).
  - Territory mutator incident analysis [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:69).
- **RR-CANON mapping:**
  - Phase-exit semantics captured by `RR-CANON-R204` (territory and ANM) together with existing `RR-CANON-R140`–`R145`.

### ANM-SCEN-05 – Line processing with no remaining decisions

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == line_processing`.
  - All eligible lines have already been processed, or the moving player P has declined to process further lines where optional.
  - `get_valid_moves(state, P) == []` for the line-processing surface (no `process_line` / `choose_line_reward` moves).
- **Historical issue:** A Python invariant snapshot captured a state where the game remained ACTIVE / LINE_PROCESSING with no legal decisions for the current player, whereas TS turn logic always auto-advanced when the line decision surface emptied.
- **Expected canonical behaviour:**
  - When no line decisions remain for P, the engine must immediately advance:
    - Either into `territory_processing` if disconnected regions exist, or
    - Directly to victory evaluation (`RR-CANON-R170`–`R173`) and turn rotation.
  - No engine may leave an ACTIVE state in `line_processing` with no legal line decisions for the current player.
- **Implementations / tests:**
  - Python parity regression [`test_active_no_moves_line_processing_regression.py`](ai-service/tests/parity/test_active_no_moves_line_processing_regression.py:1).
  - TS multi-phase orchestrator tests [`tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`](tests/scenarios/Orchestrator.Backend.multiPhase.test.ts:608) and line scenarios [`tests/unit/GameEngine.lines.scenarios.test.ts`](tests/unit/GameEngine.lines.scenarios.test.ts:1).
- **RR-CANON mapping:**
  - Line-processing exit rules in `RR-CANON-R121`–`R122` plus ANM/phase-exit clarifications in `RR-CANON-R204`.

### ANM-SCEN-06 – Global stalemate (no actions for any player)

- **Shape:**
  - `gameStatus == ACTIVE` just before stalemate resolution.
  - There are **no legal placements, movements, captures, or forced eliminations** for **any** player.
  - This can occur only once no stacks remain on the board; otherwise some forced elimination would be available.
- **Behaviour:**
  - Engines convert all rings in hand to eliminated rings for scoring, then apply the stalemate tiebreak ladder as in [`RR-CANON-R173`](RULES_CANONICAL_SPEC.md:454):
    - Most collapsed spaces (Territory).
    - Then most eliminated rings (including converted rings in hand).
    - Then most markers.
    - Then last player to have taken a real action.
  - This is the **only** situation in which a true “no actions for anyone” ANM shape is permitted; it is immediately followed by game termination.
- **Implementations / tests:**
  - Victory logic in [`TypeScript.victoryLogic.evaluateVictory`](src/shared/engine/victoryLogic.ts:45).
  - Stalemate scenarios in [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1) and Python parity tests.
- **RR-CANON mapping:**
  - `RR-CANON-R173` plus the global ANM property that no non-terminal ACTIVE state may be a global stalemate.

### ANM-SCEN-07 – Last-player-standing with only one player having real actions

- **Shape (rules intent):**
  - At least two players still own rings somewhere on the board (including buried rings), but after some point in the game:
    - Exactly one player P has at least one **real action** (placement, movement, capture) available at the start of each of their turns over a full round, and
    - All other players have no real actions on their turns (they may have only forced eliminations, or no actions at all).
- **Expected canonical behaviour (RR-CANON):**
  - Per [`RR-CANON-R172`](RULES_CANONICAL_SPEC.md:700) and clarifications in [`ringrift_complete_rules.md`](../../ringrift_complete_rules.md:1376) §13.3, P should win by Last Player Standing once the “exclusive real-action” condition has held for **two consecutive full rounds** (P takes at least one real action on each of their turns; all others have none).
  - Forced eliminations for other players keep the game legal (no ANM), but do **not** count as real actions that would block Last-Player-Standing victory.
- **Current implementation status:**
  - TS and Python engines now implement explicit early LPS detection via the shared LPS tracker:
    - Shared TS helpers in [`src/shared/engine/lpsTracking.ts`](../../src/shared/engine/lpsTracking.ts:1) (`updateLpsTracking`, `evaluateLpsVictory`).
    - Victory evaluation in the TS `VictoryAggregate` and mirrored logic in Python `GameEngine._check_victory`.
  - This replaces the earlier CCE-006 compromise where LPS was treated as an implicit “play-to-completion” pattern.
- **RR-CANON mapping:**
  - Formal semantics in `RR-CANON-R172` and ANM definitions in the R2xx cluster.
- **Status:**
  - This scenario is treated as **binding** in the canonical rules and is now explicitly implemented and exercised via TS/ Python parity tests and lpsTracking unit tests.

### ANM-SCEN-08 – Multi-player rotation with eliminated and inactive players

- **Shape:**
  - 3–4 player game.
  - Some players have no turn-material (no stacks and no rings in hand) and are “fully eliminated for turn rotation” per §1.
  - Others may still own buried rings of the eliminated colours in mixed stacks.
  - At least one remaining player still has global legal actions.
- **Expected canonical behaviour:**
  - Turn rotation skips players with no turn-material; they are not chosen as `currentPlayer` in ACTIVE states.
  - Colour-based concepts like Territory representation (`ActiveColors` in [`RR-CANON-R142`](RULES_CANONICAL_SPEC.md:384)) still account for buried rings; fully eliminated-for-turn players may continue to matter for region eligibility even though they never act again.
  - ANM invariants are enforced per active player only; inactive/fully eliminated players are irrelevant to `INV-ACTIVE-NO-MOVES` but still contribute to scoring and some connectivity rules.
- **Implementations / tests:**
  - TS and Python turn-rotation logic as described in [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:665).
  - Python invariants and self-play soaks under `INV-ACTIVE-NO-MOVES` and `INV-PHASE-CONSISTENCY` in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:167).
- **RR-CANON mapping:**
  - Turn-material and fully-eliminated semantics in the R2xx cluster, together with `RR-CANON-R142` (Territory colour-representation) and `RR-CANON-R170`–`R173`.

### ANM-SCEN-09 – Temporarily eliminated player with recovery action available

- **Shape:**
  - `gameStatus == ACTIVE`.
  - `currentPhase == movement`.
  - Current player P controls **zero stacks** (no top rings of P's colour).
  - P has **zero rings in hand** in this scenario (`ringsInHand[P] == 0`; eligibility itself is independent of rings in hand).
  - P has **at least one marker** of their colour on the board.
  - P has **at least one buried ring** (a ring of P's colour beneath another ring in some stack).
  - At least one of P's markers can be slid (Moore adjacency for square, hex-adjacency for hex) to an empty cell such that it completes a line of **at least `lineLength`** consecutive markers.
- **Expected canonical behaviour:**
  - This state is **not** ANM: P has a legal **recovery action** per [`RR-CANON-R110`–`R115`](../../RULES_CANONICAL_SPEC.md:579).
  - Recovery action is **NOT** a real action for Last-Player-Standing purposes (unlike placements, movements, and captures). Like forced elimination, recovery does not reset the LPS countdown.
  - P may slide one marker to complete a line, triggering:
    1. Line collapse (markers → territory).
    2. Buried ring extraction as self-elimination cost.
    3. Potential territory cascade processing.
  - After recovery, P may still control zero stacks and have `ringsInHand[P] == 0`. Recovery is a survival / scoring mechanism; it does not “restore” rings to hand. P remains in turn rotation as long as they still have rings somewhere (typically buried rings) or can take recovery actions; they are skipped only once permanently eliminated.
  - Turn rotation must not skip P while P has recovery options; P is "temporarily eliminated" but still has global legal actions.
- **Key constraints:**
  - Line-forming recovery requires completing a line of **at least** `lineLength` markers; overlength lines qualify and introduce the Option 1 / Option 2 choice.
  - Marker slide uses Moore adjacency (8 directions) for square boards, hex-adjacency for hexagonal.
  - Each self-elimination cost (line + any territory claims) requires extracting a buried ring from a stack **outside** the claimed region.
- **Implementations / tests:**
  - TS unit tests: `tests/unit/RecoveryAggregate.shared.test.ts`.
  - Python unit tests: `ai-service/tests/rules/test_recovery.py`.
  - TS↔Python parity: `ai-service/tests/parity/test_recovery_parity.py`.
- **RR-CANON mapping:**
  - Recovery eligibility: `RR-CANON-R110`.
  - Recovery marker slide: `RR-CANON-R111`.
  - Recovery line requirement: `RR-CANON-R112`.
  - Recovery buried ring extraction: `RR-CANON-R113`.
  - Recovery cascade processing: `RR-CANON-R114`.
  - Recovery recording semantics: `RR-CANON-R115`.
- **Interaction with other ANM scenarios:**
  - ANM-SCEN-03 (fully eliminated player): Recovery differs from full elimination; a player with markers + buried rings but no stacks/rings-in-hand is "temporarily eliminated" with recovery options, not "fully eliminated for turn rotation."
  - ANM-SCEN-07 (LPS with one player having real actions): Recovery action does **NOT** count as a real action, so a player with only recovery options does **not** block LPS victory for others; the LPS countdown can still complete.

## 3. Relationship to Invariants and Future Formalisation

- `INV-ACTIVE-NO-MOVES` in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:119) treats any ACTIVE state where `currentPlayer` has no global legal action (per §1) as an invariant violation, except for the terminal global-stalemate shape in ANM-SCEN-06.
- `INV-PHASE-CONSISTENCY` ensures that composite phases (`line_processing`, `territory_processing`, `chain_capture`) never strand the current player in ANM states; scenarios ANM-SCEN-04 and ANM-SCEN-05 are representative.
- `INV-TERMINATION` and the S-invariant [`RR-CANON-R191`](RULES_CANONICAL_SPEC.md:476) rely on forced elimination and Territory/line processing always increasing eliminated rings or Territory; repeated ANM patterns must always consume some finite resource (e.g. caps) so that play cannot continue forever.
- `PARITY-TS-PY-ACTIVE-NO-MOVES` and related parity IDs require TS and Python to agree on which states are ANM-free; the snapshots in `ai-service/tests/invariants/test_active_no_moves_*.py` and `ai-service/tests/parity/test_active_no_moves_line_processing_regression.py` serve as concrete anchors.
- Future formal work (ANM-03-ARCH) will lift these behavioural scenarios into explicit progress measures and machine-checkable invariants over the shared TS engine and Python mirror, using the R2xx rules as the semantic backbone.
- Host-level decision-phase, move-clock, and reconnect-window timeout behaviour (including canonical millisecond units and advisory HUD countdown semantics) is specified in [`P18.3-1_DECISION_LIFECYCLE_SPEC.md`](../archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md:125) and summarised for the client in [`CURRENT_RULES_STATE.md`](CURRENT_RULES_STATE.md:1).

In addition to the scenario-specific suites above, some rules-level property-based
tests exercise ANM-related behaviour more generically:

- The TS shared-engine property harness in
  `tests/unit/territoryProcessing.property.test.ts` uses `fast-check` to
  randomise the position of Q23-style 2×2 disconnected territory regions on
  `square8` while asserting that:
  - internal eliminations, border-collapse behaviour, and territory credit
    remain consistent with `RR-CANON-R140`–`R145`; and
  - every such territory-processing step strictly increases the eliminated-ring
    component of the S-invariant (`RR-CANON-R191`), supporting
    `INV-ACTIVE-NO-MOVES` / `INV-TERMINATION` by construction.
