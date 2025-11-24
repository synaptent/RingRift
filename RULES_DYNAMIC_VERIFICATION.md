# RingRift Rules Dynamic Verification Report

## 1. Introduction & Method

This document derives dynamic test scenarios from the canonical RingRift rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md) and uses the mapping in [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md) plus the static analysis in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md) and [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) to reason about runtime behaviour.

The focus is **dynamic verification** rather than code inspection:

- For each rule cluster (board/entities, turn/forced elimination, placement, movement, capture, lines, territory, victory) we define representative **scenario classes** (normal, edge, adversarial, cross-cluster).
- For each scenario class we:
  - Describe an initial game-state pattern, actions taken, and the expected canonical result under RR-CANON.
  - Map to existing Jest tests and Python AI invariants, identifying overlaps and mismatches.
  - Identify **gaps** where important behaviours are not directly exercised, and design concrete tests (names, files, assertions) for a Code-mode agent to implement.
- For high-risk areas called out in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md) we simulate step-by-step runtime behaviour, including hypothetical bug states (e.g. board repairs, forced-elimination edge cases, sandbox-only flows).

No tests were executed and no non-markdown files were modified. Behaviour is inferred from RR-CANON, the TypeScript and Python engine structure, and the existing test harnesses.

---

## 2. Scenario Matrix by Rule Cluster

Each subsection uses scenario IDs of the form `SCEN-<CLUSTER>-NNN` and lists:

- **Initial state** – relevant board pattern and per-player resources.
- **Actions** – canonical sequence of moves/decisions.
- **Expected result (RR-CANON)** – state changes, invariants, and victory conditions.
- **Existing coverage** – tests or invariants that already exercise the scenario.
- **Gaps / proposed tests** – where coverage is absent or only indirect.

### 2.1 Board & Entities (R001–R003, R020–R023, R030–R031, R040–R041, R050–R052, R060–R062)

This cluster is mostly structural; dynamic risks arise when illegal states are silently “repaired” rather than surfaced.

#### SCEN-BOARD-001 – Marker–stack overlap created by bug

- **Initial state**
  - Legal position with:
    - Player 1 markers forming a near-complete line on an 8×8 board.
    - A single Player 1 stack elsewhere.
  - A bug (in tests or future code) writes a stack to a cell already containing a marker via [`TypeScript.BoardManager.setStack()`](src/server/game/BoardManager.ts:446).
- **Actions**
  - Call any movement or capture helper that eventually triggers [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94) or relies on the mutated board.
- **Expected result (RR-CANON)**
  - Such a state is **unreachable**; RR-CANON-R030–R031 require exclusive occupancy.
  - If it somehow occurred, canonical semantics are undefined; no silent repair should change gameplay.
- **Behaviour implied by current code**
  - `setStack` and `assertBoardInvariants` will **delete the marker** and keep the stack, potentially destroying an otherwise valid line or border.
  - This can **decrease** the S-invariant (RR-CANON-R191) by reducing marker count without a compensating collapse or elimination.
- **Existing coverage**
  - Invariant checks in [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94) are exercised indirectly by many tests, but **no test asserts that repairs never occur** on legal trajectories.
- **Gaps / proposed tests**
  - **Test name**: `BoardManager_does_not_perform_repairs_on_legal_game_traces`
  - **Location**: new backend invariant suite, e.g. [`tests/unit/BoardManager.invariants.repairCounter.test.ts`](tests/unit/BoardManager.invariants.repairCounter.test.ts:1)
  - **Structure**
    - Instrument `BoardManager` (in test build only) with a `repairCount` counter incremented whenever a marker/stack/collapsed overlap is repaired.
    - Replay several long seeded backend vs sandbox parity traces (e.g. existing `seed5` and `seed17` traces referenced in [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md)) through [`TypeScript.GameEngine`](src/server/game/GameEngine.ts:92).
    - **Assert**: `repairCount === 0` for all steps.
  - **Dynamic intent**
    - Turn any occurrence of SCEN-BOARD-001 into a **hard test failure** instead of a silent correction.

#### SCEN-BOARD-002 – Region and territory counters after mixed line + territory turn

- **Initial state**
  - 19×19 board with:
    - Player 1 markers forming a line that will collapse this turn.
    - A physically and color-disconnected region of Player 2 stacks surrounded by Player 1 markers (as in the examples in [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1)).
  - Player 1 controls at least one stack outside the region.
- **Actions**
  - Player 1 performs a capturing or moving action that simultaneously:
    - Finalises the line.
    - Maintains the disconnection of the territory region.
  - Automatic consequences: line processing followed by territory processing.
- **Expected result (RR-CANON)**
  - RR-CANON-R120–R122 and R140–R145:
    - Line markers collapse to P1 territory (with either a ring elimination or Option 2, depending on length).
    - Then the region collapses; all interior rings are eliminated and credited to P1; border markers collapse to P1 territory; one P1 cap outside is self-eliminated.
    - Territory counters `territorySpaces[P1]` equal the number of P1-owned collapsed spaces.
    - Eliminated ring counters and S-invariant increase accordingly.
- **Existing coverage**
  - Well covered by combined scenarios such as `Q15_Q7_combined_line_and_region_backend` in [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:514), and RulesMatrix line+territory tests in [`tests/scenarios/LineAndTerritory.test.ts`](tests/scenarios/LineAndTerritory.test.ts:1).
- **Gaps / proposed tests**
  - No major gaps; see territory cluster for multi-region extensions.

**Cluster assessment**: core entity semantics are dynamically well covered, but **board repair** remains an important hidden behaviour that should be guarded by explicit tests.

### 2.2 Turn, Phases, and Forced Elimination (R070–R072, R100)

#### SCEN-TURN-001 – Blocked player with forced elimination available

- **Initial state**
  - 3-player 8×8 game.
  - Player A’s turn; A controls a single stack with positive capHeight.
  - No legal placements (ring cap reached or ringsInHand == 0).
  - No legal moves or captures from any A-controlled stack (e.g. all neighbours collapsed or blocked).
- **Actions**
  - Turn engine evaluates `hasAnyPlacement`, `hasAnyMovement`, `hasAnyCapture` for A.
  - Enters forced elimination per RR-CANON-R100, eliminating A’s cap.
- **Expected result (RR-CANON)**
  - A legal **forced elimination action** exists and must be applied.
  - Eliminated rings are credited to A (RR-CANON-R060, R100).
  - If A still has stacks with moves, play continues; otherwise the turn passes.
- **Existing coverage**
  - Backend turn-sequence and stalemate ladders in [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:1).
  - Turn/forced-elimination flows in [`tests/unit/GameEngine.turnSequence.scenarios.test.ts`](tests/unit/GameEngine.turnSequence.scenarios.test.ts:1).
  - Python invariant regression [`test_active_no_moves_movement_forced_elimination_regression.py`](ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py:1) ensures any ACTIVE/MOVEMENT state with no regular moves but available forced eliminations is treated as having legal actions.
- **Gaps / proposed tests**
  - Backend tests already assert forced elimination occurs; **sandbox parity** is mostly indirect.
  - **Proposed sandbox test**
    - **Name**: `sandbox_forced_elimination_when_blocked_matches_backend`
    - **Location**: [`tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts`](tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts:1) or a new sandbox turn-sequence file.
    - **Structure**: construct a blocked-state position matching an existing backend scenario and assert that `startTurnForCurrentPlayerSandbox` calls `forceEliminateCap` exactly once and leaves `currentPhase` and `currentPlayer` aligned with backend traces.

#### SCEN-TURN-002 – Player with material but no actions after territory processing

- **Initial state**
  - ACTIVE state, `currentPhase == 'territory_processing'`, `currentPlayer = P1`.
  - Territory region about to be processed eliminates all of P1’s remaining stacks; other players still have material.
- **Actions**
  - P1 applies a `process_territory_region` decision.
  - After collapse, P1 has **no stacks and no rings in hand**.
- **Expected result (RR-CANON)**
  - P1 should **not** remain the current player in an ACTIVE state with no legal actions.
  - Turn should rotate to the next player with material (RR-CANON-R070–R072, R173).
- **Existing coverage**
  - Python regression [`test_active_no_moves_territory_processing_regression.py`](ai-service/tests/invariants/test_active_no_moves_territory_processing_regression.py:1) replays an historical failure and asserts the strict invariant: any ACTIVE state must offer at least one move or forced elimination to the current player.
  - Backend behaviour is validated indirectly via territory scenarios in [`tests/unit/GameEngine.territory.scenarios.test.ts`](tests/unit/GameEngine.territory.scenarios.test.ts:1).
- **Gaps / proposed tests**
  - Add an explicit backend scenario where `currentPhase` is `territory_processing`, P1 processes a region that exhausts their material, and then:
    - **Name**: `territory_processing_rotates_turn_when_moving_player_loses_all_material`
    - **Location**: [`tests/unit/GameEngine.territory.scenarios.test.ts`](tests/unit/GameEngine.territory.scenarios.test.ts:1).
    - **Assertions**:
      - After `process_territory_region` and automatic consequences, `currentPlayer` is advanced to the next player with stacks/rings.
      - There is no intermediate ACTIVE state where [`TypeScript.RuleEngine.getValidMoves()`](src/server/game/RuleEngine.ts:752) and forced-elimination enumerations are both empty for the active player.

**Cluster assessment**: dynamic coverage is **good**, especially thanks to Python strict invariants, but sandbox-specific forced-elimination flows are only indirectly tested and should gain a direct parity scenario.

### 2.3 Placement & Skip (R080–R082)

#### SCEN-PLACEMENT-001 – Mandatory placement with no stacks

- **Initial state**
  - New 3-player 8×8 game; P1 has all rings in hand and no stacks.
- **Actions**
  - Turn begins for P1.
- **Expected result (RR-CANON)**
  - Placement is **mandatory** (RR-CANON-R080); legal placements are those that satisfy no-dead-placement (RR-CANON-R081–R082).
  - `skip_placement` must be illegal.
- **Existing coverage**
  - Covered indirectly by initial-move scenarios and RuleEngine placement enumeration in movement/FAQ suites; specifically by FAQ Q17 in [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1) and RulesMatrix movement scenarios that start from empty boards.
- **Gaps / proposed tests**
  - Add a precise RuleEngine test that enumerates `getValidMoves` in `ring_placement` with no stacks and asserts:
    - All moves have type `place_ring`.
    - `skip_placement` is **not** present.

#### SCEN-PLACEMENT-002 – Optional placement, skip allowed

- **Initial state**
  - P1 controls at least one stack with a legal move or capture.
  - P1 has `ringsInHand > 0`.
- **Actions**
  - Turn begins in `ring_placement`.
- **Expected result (RR-CANON)**
  - Placement is **optional**; both `place_ring` and an explicit `skip_placement` action are legal (RR-CANON-R080).
- **Existing coverage**
  - RuleEngine placement/skip validation (unit tests) and RulesMatrix placement scenarios referenced in [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:146).
- **Gaps / proposed tests**
  - **Sandbox representation asymmetry**: sandbox does not surface `skip_placement` as a first-class move.
  - **Proposed sandbox test**
    - **Name**: `sandbox_start_turn_with_optional_placement_exposes_movement_path`
    - **Location**: new file [`tests/unit/ClientSandboxEngine.placementPhaseParity.test.ts`](tests/unit/ClientSandboxEngine.placementPhaseParity.test.ts:1).
    - **Structure**:
      - Mirror a backend scenario where `getValidMoves` in `ring_placement` exposes both `place_ring` and `skip_placement`.
      - In sandbox, call `startTurnForCurrentPlayerSandbox` and then enumerate movement choices for the current player.
      - **Assert**: sandbox either:
        - Starts directly in `movement` phase with legal movement moves, or
        - Provides an explicit no-op/skip action; in either case, the set of reachable board states for this turn matches the backend.

#### SCEN-PLACEMENT-003 – No-dead-placement enforcement at capacity edge

- **Initial state**
  - 8×8 board where P1 is near the ring cap.
  - Candidate empty cell where adding 3 rings would create a height-3 stack **with no legal moves or captures** (distance or blocking constraints).
- **Actions**
  - P1 attempts to place 3 rings on the dead cell.
- **Expected result (RR-CANON)**
  - Placement is **illegal** by RR-CANON-R081–R082 (no-dead-placement).
  - Legal placements must either use fewer rings or a different cell so that at least one move or capture exists.
- **Existing coverage**
  - Shared validator tests (not included in the snippet) referenced in [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:236) and movement/FAQ suites.
- **Gaps / proposed tests**
  - Add a dedicated shared-validator test:
    - **Name**: `validatePlacementOnBoard_rejects_dead_high_stack_even_when_capacity_allows`
    - **Location**: [`tests/unit/PlacementValidator.rules.test.ts`](tests/unit/PlacementValidator.rules.test.ts:1).
    - **Assertions**:
      - For the dead cell, `validatePlacementOnBoard` returns `NO_LEGAL_MOVES`.
      - For an adjacent non-dead cell, placement is accepted.

#### SCEN-PLACEMENT-004 – Per-player cap approximation with many captured rings

- **Initial state**
  - 8×8 or 19×19 board where P1 controls several tall mixed-colour stacks:
    - Each stack has many opponent rings buried under a small P1 cap.
    - Total stack heights for P1-controlled stacks **exceed** `ringsPerPlayer`, even though P1 has fewer than `ringsPerPlayer` rings of their own colour on the board.
  - P1 still has some rings in hand.
- **Actions**
  - P1 attempts further placements.
- **Expected result (RR-CANON)**
  - If the cap is interpreted as **“own-colour rings only”**, further placements might still be legal.
  - The current TS implementation approximates the cap as **“total rings in P1-controlled stacks”**, potentially forbidding additional placements.
- **Existing coverage**
  - No direct dynamic tests target this approximation; it is treated as benign in [`RULES_STATIC_VERIFICATION.md`](RULES_STATIC_VERIFICATION.md:755).
- **Gaps / proposed tests**
  - **Test name**: `placement_cap_approximation_with_mixed_colour_stacks`
  - **Location**: new shared rules test [`tests/unit/PlacementCap.mixedColourStack.test.ts`](tests/unit/PlacementCap.mixedColourStack.test.ts:1).
  - **Structure**
    - Construct the mixed-colour scenario described above both in TS and Python engines.
    - In TS, assert that [`TypeScript.RuleEngine.getValidRingPlacements()`](src/server/game/RuleEngine.ts:839) yields **no** placements (cap reached).
    - In Python, implement two variants:
      - One using the same approximation (for parity).
      - One counting only own-colour rings.
    - **Decision**:
      - If the canonical spec is updated to match the approximation, this test locks in the behaviour.
      - If RR-CANON is interpreted strictly, this test will highlight the discrepancy and guide a future refactor.

**Cluster assessment**: mandatory/optional/forbidden placement and no-dead-placement are dynamically robust; the main open question is how strictly RR-CANON should define the per-player cap in the presence of many captured rings.

### 2.4 Non-capture Movement & Markers (R090–R092)

#### SCEN-MOVEMENT-001 – Minimum-distance movement on empty board

- **Initial state**
  - 8×8 board; P1 stack of height 2 at centre.
- **Actions**
  - P1 attempts non-capture moves at various distances and directions.
- **Expected result (RR-CANON)**
  - Legal landings are exactly those at distance `d ≥ H` along movement directions with empty intermediate cells and legal landing cells (no stacks, no opponent markers, no collapsed spaces) per RR-CANON-R091.
- **Existing coverage**
  - Shared helper parity test `square8: shared vs sandbox vs RuleEngine on open board` in [`tests/unit/movement.shared.test.ts`](tests/unit/movement.shared.test.ts:56).
  - RulesMatrix movement scenarios M1 in [`tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts`](tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts:1).
- **Gaps / proposed tests**
  - Well covered; no additional tests required beyond parity extensions.

#### SCEN-MOVEMENT-002 – Marker flipping and collapse along a path

- **Initial state**
  - 8×8 board.
  - P1 stack at A, several markers along a ray A→…→B:
    - Some markers belong to P2; some to P1.
- **Actions**
  - P1 performs a legal non-capture movement from A to B, passing over all markers.
- **Expected result (RR-CANON)**
  - RR-CANON-R092:
    - Departure cell gains a P1 marker.
    - Opponent markers along the path are flipped to P1.
    - P1 markers along the path are collapsed to territory for P1.
    - Landing cell marker (if P1-owned) is removed without collapse; then the top ring of the landing stack is eliminated and credited to P1.
- **Existing coverage**
  - Movement + marker behaviour in RuleEngine movement tests and the shared helper tests in [`tests/unit/movement.shared.test.ts`](tests/unit/movement.shared.test.ts:223).
  - Landing-on-own-marker elimination tests in [`tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts`](tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts:1).
- **Gaps / proposed tests**
  - Add a **full-path S-invariant** check where a movement collapses multiple own markers:
    - **Name**: `movement_path_marker_flip_and_collapse_preserves_S_invariant_shape`
    - **Location**: [`tests/unit/ProgressSnapshot.core.test.ts`](tests/unit/ProgressSnapshot.core.test.ts:1).
    - **Assertions**:
      - Compare the S values from [`TypeScript.computeProgressSnapshot()`](src/shared/engine/core.ts:531) before and after the move.
      - Verify it increases by the net effect (collapsed markers + eliminated ring) as per RR-CANON-R191.

#### SCEN-MOVEMENT-003 – Hex movement adjacency and minimum distance

- **Initial state**
  - Hex board; P1 stack of height 1 at centre cube coordinate (0,0,0).
- **Actions**
  - Enumerate all legal non-capture moves from the centre.
- **Expected result (RR-CANON)**
  - Legal moves correspond exactly to hex adjacency directions with distance ≥ 1 and valid landing cells.
- **Existing coverage**
  - `hexagonal: shared vs sandbox vs RuleEngine on open board` in [`tests/unit/movement.shared.test.ts`](tests/unit/movement.shared.test.ts:173).
- **Gaps / proposed tests**
  - No gaps; parity coverage is already strong.

**Cluster assessment**: non-capture movement and marker interactions are dynamically **well covered**, with only a minor opportunity to tie them more explicitly to the S-invariant.

### 2.5 Overtaking Capture & Chains (R101–R103)

#### SCEN-CAPTURE-001 – Single-segment capture legality against multiple targets

- **Initial state**
  - 8×8 board; P1 stack of height 3 at centre.
  - Two P2 stacks with smaller caps on different rays at equal distance.
- **Actions**
  - Enumerate legal capture segments from the centre.
- **Expected result (RR-CANON)**
  - RR-CANON-R101:
    - Legal segments for each target stack with `capHeight_P1 ≥ capHeight_target`.
    - Landing cells satisfy distance and blocking rules.
- **Existing coverage**
  - Shared capture helper tests in [`tests/unit/captureLogic.shared.test.ts`](tests/unit/captureLogic.shared.test.ts:68), especially `enumerates capture segments along multiple rays from a single attacker`.
  - Additional RuleEngine capture tests and FAQ Q5–Q6 in [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1).
- **Gaps / proposed tests**
  - None for single-segment legality.

#### SCEN-CAPTURE-002 – 180° reversal in a capture chain (FAQ Q15.3.1)

- **Initial state**
  - Square19 board; Blue (P1) stack of height 4 at A, Red (P2) stack of height 3 at B on a straight line.
  - Geometry as in FAQ Q15.3.1 and [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:30).
- **Actions**
  - P1 performs `overtaking_capture` A→B→C, then chain captures back C→B→D (180° reversal).
- **Expected result (RR-CANON)**
  - RR-CANON-R103:
    - Direction changes, including 180° reversals, are legal between segments.
    - Target stack at B can be re-captured if cap legality holds.
    - Stack heights and caps after the chain match the FAQ diagram (Blue ends with height 6; Red at B shrinks to 1).
- **Existing coverage**
  - Backend dynamic chain in `Q15.3.1: 180-Degree Reversal Pattern` in [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:30).
  - Additional scenario coverage in `ComplexChainCaptures` and `RulesMatrix.ChainCapture` suites referenced by [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:92).
- **Gaps / proposed tests**
  - **Shared-helper parity**: capture-chain enumeration is still partly wired through legacy [`TypeScript.captureChainEngine`](src/server/game/rules/captureChainEngine.ts:45) and sandbox-specific logic.
  - **Proposed test**
    - **Name**: `captureChainHelpers_180_degree_reversal_matches_GameEngine`
    - **Location**: new shared-engine test [`tests/unit/captureChainHelpers.shared.test.ts`](tests/unit/captureChainHelpers.shared.test.ts:1).
    - **Structure**:
      - Construct the Q15.3.1 position using shared board fixtures.
      - Use [`TypeScript.captureChainHelpers`](src/shared/engine/captureChainHelpers.ts:134) (once fully implemented) to enumerate all legal next segments after the first capture.
      - Use [`TypeScript.GameEngine`](src/server/game/GameEngine.ts:92) to enumerate `continue_capture_segment` moves.
      - **Assert**: the sets of legal continuation segments match, including the 180° reversal.

#### SCEN-CAPTURE-003 – Cyclic capture loop (FAQ Q15.3.2)

- **Initial state**
  - 8×8 or hex board with a triangle of opponent stacks around an initial P1 stack, as in FAQ Q15.3.2 and [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:166).
- **Actions**
  - P1 executes a capture chain A→B→…→C→A, visiting each target exactly once.
- **Expected result (RR-CANON)**
  - Chain continues while legal segments exist; revisiting cells and stacks is allowed if legality holds.
  - Final stack height and elimination counts match FAQ expectations.
- **Existing coverage**
  - Backend chain in `Q15.3.2: Cyclic Pattern` in [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:166).
  - Hex-specific cyclic capture tests in [`tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`](tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts:1).
- **Gaps / proposed tests**
  - Same as SCEN-CAPTURE-002: new shared-helper tests should ensure [`TypeScript.captureChainHelpers`](src/shared/engine/captureChainHelpers.ts:134) can reproduce the cyclic capture graph and that any future refactor does not regress these edge cases.

#### SCEN-CAPTURE-004 – Mandatory continuation despite disadvantage (FAQ Q15.3.3)

- **Initial state**
  - A square8 position where:
    - P1 initiates an overtaking capture that leads into a chain.
    - At least one continuation is legal but leads to a strategically bad outcome (e.g. self-exposure).
- **Actions**
  - P1 starts the chain.
  - During `chain_capture`, at least one legal continuation remains.
- **Expected result (RR-CANON)**
  - P1 **must** choose some continuation as long as any legal capture exists; chain cannot be voluntarily stopped early (RR-CANON-R103).
- **Existing coverage**
  - Test `Q15.3.3: Mandatory Continuation` in [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:254) verifies the chain continues until no `continue_capture_segment` moves remain.
- **Gaps / proposed tests**
  - None; behaviour is explicitly encoded.

**Cluster assessment**: capture chains are dynamically **very well covered** at the GameEngine level, but the future shared capture-chain helpers lack direct, scenario-driven tests. SCEN-CAPTURE-002/003 highlight the most important patterns to lock in.

### 2.6 Lines & Graduated Rewards (R120–R122)

#### SCEN-LINES-001 – Exact-length line with mandatory elimination

- **Initial state**
  - 8×8 board; P1 markers forming an exact-length horizontal line of length `lineLength` and at least one P1 stack elsewhere.
- **Actions**
  - Line detection finds the P1 line.
  - P1 processes the line.
- **Expected result (RR-CANON)**
  - RR-CANON-R122 Case 1:
    - All markers in the line collapse to P1 territory.
    - P1 must eliminate one standalone ring or a whole cap they control.
- **Existing coverage**
  - Backend scenario `Q7_exact_length_line_collapse_backend` in [`tests/unit/GameEngine.lines.scenarios.test.ts`](tests/unit/GameEngine.lines.scenarios.test.ts:81).
  - Sandbox mirror `Q7_exact_length_line_collapse_sandbox` in [`tests/unit/ClientSandboxEngine.lines.test.ts`](tests/unit/ClientSandboxEngine.lines.test.ts:73).
- **Gaps / proposed tests**
  - Already well covered in both hosts.

#### SCEN-LINES-002 – Overlength line: Option 2 default without elimination

- **Initial state**
  - 8×8 board; P1 markers form a line of length `lineLength + 1`.
  - P1 has at least one eliminable stack.
- **Actions**
  - Backend or sandbox processes lines **without** external PlayerChoice (no PlayerInteractionManager, or AI defaults).
- **Expected result (RR-CANON)**
  - RR-CANON-R122 Case 2:
    - For overlength lines, P1 may choose Option 1 (collapse all + elimination) or Option 2 (collapse exactly `requiredLen` markers, no elimination).
  - The engine’s default choice without interaction is an implementation detail but should be consistent across hosts.
- **Existing coverage**
  - Backend default Option 2 verified by `Q22_graduated_rewards_option2_min_collapse_backend_default` in [`tests/unit/GameEngine.lines.scenarios.test.ts`](tests/unit/GameEngine.lines.scenarios.test.ts:150).
  - Sandbox default Option 2 verified in `longer-than-required line collapses minimum markers without elimination` in [`tests/unit/ClientSandboxEngine.lines.test.ts`](tests/unit/ClientSandboxEngine.lines.test.ts:131).
- **Gaps / proposed tests**
  - One missing aspect is an **explicit regression test** ensuring that, when a PlayerChoice is wired (human or AI), Option 1 and Option 2 both remain available and produce the expected S-invariant changes.
  - Existing AI/WebSocket tests (e.g. [`tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`](tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts:1)) cover choice wiring; no additional scenario is strictly required, but a consolidated S-invariant assertion would be valuable.

#### SCEN-LINES-003 – Canonical `choose_line_reward` move without immediate elimination

- **Initial state**
  - P1 overlength line; P1 has at least one eliminable stack elsewhere.
- **Actions**
  - P1 plays a canonical `choose_line_reward` move collapsing all markers in the line.
- **Expected result (RR-CANON)**
  - Line collapse can be separated from the subsequent elimination decision; S-invariant progresses monotonically across both steps.
- **Existing coverage**
  - Sandbox canonical path tested in `canonical choose_line_reward Move collapses entire overlength line and defers elimination reward` in [`tests/unit/ClientSandboxEngine.lines.test.ts`](tests/unit/ClientSandboxEngine.lines.test.ts:291).
- **Gaps / proposed tests**
  - Add a backend counterpart explicitly using [`TypeScript.GameEngine.applyCanonicalMove()`](src/server/game/GameEngine.ts:1258) with a `choose_line_reward` move and verifying that:
    - No elimination occurs until an `eliminate_rings_from_stack` decision is applied.
    - S-invariant increases only due to collapsed markers at the first step.

**Cluster assessment**: lines and graduated rewards are dynamically **strongly covered** across backend, sandbox, AI service, and WebSocket flows. Remaining value lies in S-invariant-focused assertions rather than new geometry.

### 2.7 Territory Disconnection & Region Processing (R140–R145)

#### SCEN-TERRITORY-001 – Single-region collapse with self-elimination (Q23 archetype)

- **Initial state**
  - 8×8 mini-region as in `Rules_12_2_Q23_mini_region_square8_numeric_invariant` from [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19).
  - P2 victim stacks fully inside the region; P1 has a stack outside that can pay the self-elimination cost.
- **Actions**
  - P1 processes the region via `process_territory_region` and then eliminates rings from the outside stack.
- **Expected result (RR-CANON)**
  - RR-CANON-R143–R145:
    - All interior rings (both colours) are eliminated and credited to P1.
    - Region spaces and border markers collapse to P1 territory.
    - One P1 ring or cap outside is eliminated as self-elimination.
- **Existing coverage**
  - Numeric invariants at the rules layer in `territoryProcessing.rules` and sandbox equivalents referenced from [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:252).
  - Data-only scenario validation in [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:19).
- **Gaps / proposed tests**
  - No significant gaps; behaviour is already exercised in both backend and sandbox.

#### SCEN-TERRITORY-002 – Multi-region chain reaction with limited self-elimination budget

- **Initial state**
  - 3-player 19×19 game with **two** disconnected regions R1 and R2 of victim stacks, each fully bordered by P1 markers.
  - P1 has exactly one eligible outside stack, enough to pay the self-elimination cost for only **one** region.
- **Actions**
  - Territory detection identifies both R1 and R2 as physically and color-disconnected.
  - P1 chooses to process only R1 this turn.
- **Expected result (RR-CANON)**
  - RR-CANON-R143–R144:
    - After processing R1, self-elimination prerequisite fails for R2 (no remaining outside caps), so R2 is **not** processed.
    - P1 retains the option to process R2 on a later turn if new outside material appears.
- **Existing coverage**
  - Multi-region backend integration `processes multiple disconnected regions in sequence` in [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:398), but that scenario gives P1 enough material to process **both** regions.
- **Gaps / proposed tests**
  - **Test name**: `territory_multi_region_single_self_elimination_budget`
  - **Location**: [`tests/unit/GameEngine.territory.scenarios.test.ts`](tests/unit/GameEngine.territory.scenarios.test.ts:1).
  - **Structure**:
    - Construct R1 and R2 as in SCEN-TERRITORY-002, but with only one eligible outside P1 stack.
    - Drive region processing via canonical decision moves (or automatic defaults).
    - **Assert**:
      - Exactly one region is processed this turn.
      - R2 remains unchanged and may be processed later if P1 regains outside material.

#### SCEN-TERRITORY-003 – Combined line + region processing in a single turn

- **Initial state**
  - 19×19 example where a capturing move both:
    - Forms a P1 line, and
    - Maintains a disconnected victim region as in SCEN-TERRITORY-001.
- **Actions**
  - P1 performs a capture; automatic consequences run:
    - Line processing (with elimination).
    - Territory processing (with region collapse and self-elimination).
- **Expected result (RR-CANON)**
  - Line and territory phases execute in order (RR-CANON-R070–R071).
  - All collapse and elimination accounting matches the combination of SCEN-LINES-001 and SCEN-TERRITORY-001.
- **Existing coverage**
  - Fully exercised by `Q15_Q7_combined_line_and_region_backend` in [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:514) and the `LineAndTerritory` scenarios in [`tests/scenarios/LineAndTerritory.test.ts`](tests/scenarios/LineAndTerritory.test.ts:1).
- **Gaps / proposed tests**
  - None; combination behaviour appears robust.

**Cluster assessment**: territory mechanics are **well covered** for single and multi-region cases, but one important variant (multi-region with insufficient self-elimination budget) remains untested and should be added.

### 2.8 Victory, Stalemate, and S-invariant (R170–R173, R190–R191)

#### SCEN-VICTORY-001 – Last-player-standing with buried rings

- **Initial state**
  - 3-player 8×8 game where:
    - P1 has active stacks with legal moves.
    - P2 has rings buried under P1 caps but no legal actions (no placements, moves, or captures).
    - P3 has no stacks and no rings in hand.
- **Actions**
  - Full turn for P1, including any lines and territory; game reaches a state where:
    - P1 still has legal actions on their **next** turn.
    - P2 remains inactive; P3 has no material.
- **Expected result (RR-CANON)**
  - P1 wins by **last-player-standing** (RR-CANON-R172); buried rings for inactive players do not prevent this.
- **Existing coverage**
  - Partially covered in victory scenario suites in [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1) and FAQ victory examples in [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1).
- **Gaps / proposed tests**
  - Add an explicit last-player-standing scenario with buried rings:
    - **Name**: `last_player_standing_with_buried_inactive_rings`
    - **Location**: [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1).
    - **Assertions**:
      - At the terminal state, winner is P1 with reason `last_player_standing`.
      - P2 has no legal actions on their would-be next turn, even though they own buried rings.

#### SCEN-VICTORY-002 – Global stalemate and full tiebreak ladder

- **Initial state**
  - No stacks on the board for any player.
  - Each player has some rings in hand and some collapsed territory and markers.
- **Actions**
  - Victory evaluation runs with **no legal moves for any player** (global stalemate).
- **Expected result (RR-CANON)**
  - RR-CANON-R173:
    - All rings in hand are converted to eliminated rings for their owners.
    - Winner is determined by:
      1. Most collapsed spaces.
      2. If tied, most eliminated rings (including converted rings in hand).
      3. If still tied, most markers.
      4. If still tied, last player to have completed a valid turn action.
- **Existing coverage**
  - Stalemate ladder tests and FAQ Q11 scenarios in [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:1) and [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1).
- **Gaps / proposed tests**
  - Existing tests cover typical 2-player ladders; fewer explicit tests cover **3+ player** stalemates with deep ties on the first two rungs.
  - **Proposed test**:
    - **Name**: `stalemate_ladder_three_player_full_tie_break_sequence`
    - **Location**: [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1).
    - **Structure**:
      - Construct a 3-player stalemate where:
        - P1 and P2 tie on territory and eliminated rings.
        - P1 has more markers than P2.
        - P3 is strictly behind.
      - **Assert**: winner is P1 due to the marker rung, and history shows the last-action rung would only be consulted if markers were also tied.

**Cluster assessment**: ring-elimination and territory victories are well tested; **last-player-standing** and multi-player stalemates deserve additional explicit scenarios.

---

## 3. Deep-Dive Simulations for High-Risk Areas

This section focuses on behaviours that are either outside RR-CANON (board repair), approximations (placement capacity), host-specific (sandbox), or under development (shared capture-chain helpers).

### 3.1 Board Invariant “Repair” (`BoardManager.assertBoardInvariants`)

**Canonical expectation (RR-CANON)**  
RR-CANON-R030–R031 and R050–R052 treat stack/marker/collapsed overlaps as **unreachable** in legal play. No rule allows such states to arise; if they did, canonical semantics are undefined.

**Current implementation behaviour**  
[`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94) and its helpers:

- On each write to `stacks`, `markers`, or `collapsedSpaces`, perform a **repair pass** that:
  - Deletes markers that overlap stacks or collapsed spaces.
  - Deletes stacks that overlap collapsed spaces when collapsing markers or setting collapsed spaces.
- Only after repair does it assert invariants (throwing in strict/test modes).

**Dynamic simulation scenario**

1. **Before bug**
   - P1 has a nearly-complete line of markers; S-invariant `S = M + C + E` is high due to many markers.
2. **Buggy write**
   - A faulty move application writes a stack to one of the marker cells without clearing the marker.
3. **Repair phase**
   - `setStack` or `assertBoardInvariants` deletes the marker and keeps the stack.
   - `M` decreases by 1; `C` and `E` are unchanged; so `S` **decreases**, violating RR-CANON-R191.
4. **Downstream effects**
   - Line detection no longer sees a complete line; P1 loses a territory opportunity.
   - Territory detection may see a different border or fail to recognise a disconnected region.

**Impact and recommended tests**

- Impact is limited to **already-illegal** states, but silent repair can make debugging extremely difficult and can break S-invariant reasoning.
- Dynamic tests should:
  - Instrument BoardManager with a `repairEvents` counter in test builds.
  - Assert `repairEvents === 0` on all seeded backend vs sandbox traces and rules-matrix scenarios.
  - Optionally, add a negative test that constructs an overlapping state and asserts that:
    - `repairEvents` increments.
    - The S metric from [`TypeScript.computeProgressSnapshot()`](src/shared/engine/core.ts:531) decreases, demonstrating why such states must be treated as fatal bugs, not gameplay.

### 3.2 Placement Capacity Approximation

**Canonical rule**  
RR-CANON-R020–R023 and R060–R062 define per-player ring counts by colour; ring caps conceptually apply to **rings of that player’s colour**.

**Implementation approximation**  
[`TypeScript.validatePlacementOnBoard()`](src/shared/engine/validators/PlacementValidator.ts:76) and [`TypeScript.RuleEngine.getValidRingPlacements`](src/server/game/RuleEngine.ts:839) approximate “rings on board for this player” as the **sum of stack heights for stacks where `controllingPlayer === player`**, which may include many captured rings of other colours.

**Dynamic scenario**

- Construct P1 stacks:
  - Stack A: `rings = [2,2,2,1]` (three P2 rings under a P1 cap).
  - Stack B: similar pattern, so that total heights of P1-controlled stacks reach `ringsPerPlayer`.
  - P1 still has rings in hand and fewer than `ringsPerPlayer` rings of colour 1 actually on the board.
- Under RR-CANON’s strict reading, additional placements might still be allowed (since many on-board rings are opponent-coloured).
- Under the current approximation, TS will report **no legal placements** because the height-sum cap is reached.

**Recommended tests**

- Implement SCEN-PLACEMENT-004 as a TS/Python parity test:
  - TS engine uses the current approximation; Python engine can be toggled between strict and approximate interpretations.
  - Compare legal placement sets between the two interpretations.
- Decide canonically whether the approximation is intended semantics; if so, encode that decision in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md) and treat strict-by-colour variants as alternative rules.

### 3.3 Sandbox-Only Behaviours

**3.3.1 `skip_placement` representation**

- Backend surfaces `skip_placement` moves explicitly via [`TypeScript.RuleEngine.validateSkipPlacement`](src/server/game/RuleEngine.ts:116).
- Sandbox does **not** model `skip_placement` as a move; it simply omits placement when no legal placements exist and transitions to movement.
- Dynamic risk: parity tooling that expects a `skip_placement` record may not see one, complicating trace comparisons.
- Recommended test: SCEN-PLACEMENT-002 sandbox parity test ensures reachable board states (after placement + movement) are the same for backend and sandbox even if move logs differ.

**3.3.2 Start-of-turn `currentPhase` heuristic**

- [`TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:164) chooses:
  - `currentPhase = 'ring_placement'` if `ringsInHand > 0`.
  - Else `currentPhase = 'movement'`.
- Edge case: `ringsInHand > 0` but no **legal** placements (due to no-dead-placement or cap), while moves or captures exist.
- Canonical behaviour (RR-CANON-R070–R072):
  - Placement should be **optional** but effectively forbidden; backend encodes this as a `skip_placement` move and proceeds to movement.
- Sandbox behaviour:
  - `currentPhase` remains `'ring_placement'`, but UI/AI may still treat the turn as “ready to move”.
- Recommended tests:
  - New sandbox test that constructs this edge case and asserts:
    - `getValidMovesForCurrentPlayer` from sandbox exposes at least one legal movement or capture even while `currentPhase === 'ring_placement'`.
    - Traces remain aligned with backend after a full turn.

**3.3.3 Auto line/territory processing defaults**

- Sandbox convenience methods (e.g. `processLinesForCurrentPlayer`, `processDisconnectedRegionsForCurrentPlayerEngine`) automatically choose:
  - Option 2 for overlength lines when no explicit PlayerChoice is supplied (SCEN-LINES-002).
  - A default region order and self-elimination stack for territory.
- Canonical model treats these as **explicit decisions** (line/region order, reward choice, elimination choice).
- Existing tests (e.g. [`tests/unit/ClientSandboxEngine.lines.test.ts`](tests/unit/ClientSandboxEngine.lines.test.ts:1), [`tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`](tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts:1)) already lock in the current UX defaults.
- Recommended future tests:
  - When sandbox is refactored toward full canonical parity, keep these tests but change their assertions to expect **decision moves** rather than implicit behaviour, thereby guarding against regressions.

### 3.4 Capture-Chain Stubs vs Active Implementations

**Context**

- Real capture chains are implemented in:
  - Backend: [`TypeScript.GameEngine`](src/server/game/GameEngine.ts:92) plus [`TypeScript.captureChainEngine`](src/server/game/rules/captureChainEngine.ts:45).
  - Sandbox: [`TypeScript.sandboxMovementEngine.performCaptureChainSandbox`](src/client/sandbox/sandboxMovementEngine.ts:400).
- Shared helper [`TypeScript.captureChainHelpers`](src/shared/engine/captureChainHelpers.ts:134) is currently a stub intended to centralise capture-chain logic in the future.

**Deep-dive scenarios**

1. **Multi-branch chain with direction changes**
   - Start from a position similar to SCEN-CAPTURE-002 with several legal continuation segments after the first capture (forward, backward, sideways).
   - Expected: `captureChainHelpers` and GameEngine expose identical sets of continuation segments and enforce mandatory continuation.
2. **Revisiting the same target stack**
   - Use a 180° reversal pattern where the same target stack is captured twice in separate segments, shrinking its cap height each time.
   - Expected: shared helpers correctly recompute legality after each segment and prevent illegal re-capture when caps are exhausted.
3. **Interactions with markers and landing-on-own-marker**
   - Construct a chain where at least one segment lands on a P1 marker, triggering immediate top-ring elimination (RR-CANON-R092, R102).
   - Expected: S-invariant increases appropriately (marker removed, ring eliminated) and later segments take the reduced stack height into account.

**Recommended tests**

- New shared test file [`tests/unit/captureChainHelpers.shared.test.ts`](tests/unit/captureChainHelpers.shared.test.ts:1) implementing the scenarios above and comparing:
  - Segment graphs from `captureChainHelpers`.
  - Actual chain moves from GameEngine and sandbox.
- Once these tests pass, future refactors can route all chain logic through shared helpers with high confidence.

---

## 4. Coverage Assessment and Prioritised Gaps

### 4.1 Cluster-Level Coverage Summary

- **Board & entities (R001–R003, R020–R023, R030–R031, R040–R041, R050–R052, R060–R062)**
  - **Status**: Well covered for legal states; **weak** for illegal-state detection.
  - Priority: Add repair-counter tests to ensure board “repairs” never occur on legal trajectories.

- **Turn / phases / forced elimination (R070–R072, R100)**
  - **Status**: Well covered on backend and Python via strict invariants; **partially covered** on sandbox (start-of-turn heuristics, skip representation).
  - Priority: Add sandbox parity tests for forced elimination and optional placement.

- **Placement & skip (R080–R082)**
  - **Status**: Core legality and no-dead-placement are well covered; **partially covered** for per-player cap semantics with many captured rings.
  - Priority: Implement mixed-colour cap scenario to decide and lock in canonical semantics.

- **Non-capture movement & markers (R090–R092)**
  - **Status**: Well covered across shared helpers, backend, sandbox, and rules-matrix scenarios.
  - Priority: Add S-invariant-focused tests for complex marker paths.

- **Overtaking capture & chains (R101–R103)**
  - **Status**: Very well covered at GameEngine/sandbox level (including FAQ Q15 patterns); **weak** coverage for shared `captureChainHelpers`.
  - Priority: Add shared-helper regression suite for 180° reversals, cycles, and marker interactions.

- **Lines & graduated rewards (R120–R122)**
  - **Status**: Well covered for geometry and reward choices across hosts and AI/WebSocket flows.
  - Priority: Minor S-invariant and canonical-move-path tests only.

- **Territory disconnection & region processing (R140–R145)**
  - **Status**: Well covered for single regions and multi-region chains where self-elimination is sufficient; **partially covered** for multi-region cases with insufficient self-elimination budget.
  - Priority: Add SCEN-TERRITORY-002 to assert correct skipping of unaffordable regions.

- **Victory, last-player-standing, stalemate, S-invariant (R170–R173, R190–R191)**
  - **Status**: Ring-elimination and territory victories are well covered; strict S-invariant and no-move invariants in Python provide strong dynamic guarantees. Last-player-standing and complex multi-player stalemates are **partially covered** only.
  - Priority: Add explicit last-player-standing with buried rings and 3-player stalemate ladder scenarios.

### 4.2 Highest-Priority Gaps

1. **Board repair visibility (SCEN-BOARD-001)**
   - Risk: Silent deletion of markers in illegal states can break S-invariant reasoning and hide bugs.
   - Action: Add repair-counter tests on long traces; treat any repair as a hard failure in CI.

2. **Sandbox turn-phase and skip semantics (SCEN-TURN-001/SCEN-PLACEMENT-002)**
   - Risk: Sandbox may report `currentPhase = 'ring_placement'` with no legal placements while backend uses explicit `skip_placement`.
   - Action: Add sandbox parity tests ensuring that, despite representation differences, the set of reachable board states per turn matches backend semantics.

3. **Per-player placement cap with many captured rings (SCEN-PLACEMENT-004)**
   - Risk: Small but real semantic divergence between a strict per-colour interpretation and the current “rings in controlled stacks” approximation.
   - Action: Implement a dynamic parity test that forces this divergence and use it to either update RR-CANON or adjust the implementation.

4. **Shared capture-chain helpers (SCEN-CAPTURE-002/003/004)**
   - Risk: Future refactors that move chain logic into [`TypeScript.captureChainHelpers`](src/shared/engine/captureChainHelpers.ts:134) could regress complex patterns currently only tested at GameEngine level.
   - Action: Add shared-helper tests that directly encode FAQ Q15 patterns and compare against backend/sandbox chains.

5. **Territory multi-region with limited self-elimination (SCEN-TERRITORY-002)**
   - Risk: Edge cases where not all disconnected regions can be afforded in a single turn may behave incorrectly or inconsistently across hosts.
   - Action: Add explicit scenario tests for “one of many eligible regions” with a single self-elimination budget.

6. **Victory edge cases (SCEN-VICTORY-001/002)**
   - Risk: Last-player-standing with buried rings and deep multi-player stalemate ladders could behave incorrectly in rare tournaments.
   - Action: Extend victory scenario suites to include explicit last-player-standing and 3-player stalemate ladder tests, locking in RR-CANON-R172–R173.

---

## 5. Summary

Dynamic coverage for RingRift’s rules is **strong overall**, especially for movement, capture chains, lines, single-region territory, and basic victory conditions. The combination of Jest scenario suites, shared-engine unit tests, sandbox parity tests, and Python strict invariants provides broad and deep exercise of RR-CANON semantics.

The most important remaining work is **targeted**:

- Make illegal board-state repairs observable and test-failing.
- Clarify and lock in per-player placement cap semantics in the presence of many captured rings.
- Ensure sandbox turn and placement flows remain semantically aligned with backend `skip_placement` and forced elimination behaviour.
- Add shared capture-chain helper tests before centralising chain logic.
- Extend victory and territory scenarios to cover rare but high-impact edge cases (multi-region chains with limited self-elimination, last-player-standing with buried rings, complex stalemate ladders).

These additions will move the system from “well tested” to **dynamically robust** across all rule clusters, providing strong regression protection for future engine and sandbox refactors.
