> **Doc Status (2025-11-28): Active (supplementary, derived)**  
> **Role:** Deep-dive analysis of rules interactions and edge cases across hosts (backend, sandbox, Python), sitting on top of the canonical rules semantics and implementation mapping.
>
> **SSoT alignment:** This report is a derived analytical view over the **Rules/invariants semantics SSoT**:
>
> - Canonical rules docs: `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`.
> - Shared TypeScript rules engine helpers, aggregates, and orchestrator under `src/shared/engine/**` plus v2 contract vectors under `tests/fixtures/contract-vectors/v2/**` and their TS/Python runners.
> - Derived implementation and architecture docs: `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`, `docs/MODULE_RESPONSIBILITIES.md`, and `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`.
> - Historical parity/trace harnesses: selected superseded suites are archived under `archive/tests/**` (TS) and `ai-service/tests/archive/**` (Python) for diagnostic reference only; semantics are always taken from the shared engine, contracts, and the active `*.shared.test.ts` + RulesMatrix/FAQ suites.
>
> **Precedence:** This document is **not** a semantics SSoT. If any description here conflicts with the shared TS engine/orchestrator, contract vectors, or their tests (including Python parity/invariant suites), **code + tests + canonical docs win** and this report must be updated.

# RingRift Rules Consistency & Edge-Case Report

## 1. Overview & Methodology

This report analyses how the implemented rules behave when **different rule clusters interact**, when the game reaches **boundary / rare states**, and when hosts (backend, sandbox, Python) see **strange or invalid inputs**.

The focus is **not** to redo full static or dynamic verification, but to sit on top of:

- Canonical rules `RR‑CANON‑R001–R191` in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:36).
- Implementation mapping in [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:1).
- Static analysis in [`archive/RULES_STATIC_VERIFICATION.md`](../../archive/RULES_STATIC_VERIFICATION.md:1).
- Scenario and soak analysis in [`archive/RULES_DYNAMIC_VERIFICATION.md`](../../archive/RULES_DYNAMIC_VERIFICATION.md:1).

Code behaviour is inferred primarily from the shared engine and orchestration:

- Turn / phase / forced elimination via [`TypeScript.turnLogic.advanceTurnAndPhase()`](src/shared/engine/turnLogic.ts:135) and backend / sandbox delegates.
- Movement / capture / markers via [`TypeScript.core`](src/shared/engine/core.ts:1), [`TypeScript.movementLogic`](src/shared/engine/movementLogic.ts:1), and [`TypeScript.captureLogic`](src/shared/engine/captureLogic.ts:1).
- Lines via [`TypeScript.lineDecisionHelpers`](src/shared/engine/lineDecisionHelpers.ts:1).
- Territory via [`TypeScript.territoryDetection`](src/shared/engine/territoryDetection.ts:36), [`TypeScript.territoryProcessing`](src/shared/engine/territoryProcessing.ts:1), and [`TypeScript.territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:1).
- Victory / stalemate via [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45).
- Board invariants and repairs via [`TypeScript.BoardManager`](src/server/game/BoardManager.ts:94).
- Sandbox turn orchestration via [`TypeScript.ClientSandboxEngine` turn helpers](src/client/sandbox/ClientSandboxEngine.ts:1606) composed with [`TypeScript.turnLogic.advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135).

> **Note on legacy module references:** Throughout this report, some references point to historical modules such as `src/client/sandbox/sandboxTurnEngine.ts`, `sandboxMovementEngine.ts`, `sandboxLinesEngine.ts`, `sandboxTerritoryEngine.ts`, or backend helpers under `src/server/game/rules/*.ts` (e.g. `captureChainEngine.ts`). These files have been removed as part of the shared-engine consolidation and now exist only as historical anchors; their responsibilities live in the shared TS engine/orchestrator plus `ClientSandboxEngine.ts`, `SandboxOrchestratorAdapter.ts`, `GameEngine.ts`, `TurnEngineAdapter.ts`, and the Python rules/AI adapters described in the current architecture docs.

The main interaction domains covered are:

- Turn / phase structure and forced elimination.
- Movement / capture with markers, lines, and territory in sequence.
- Multi‑region territory and self‑elimination budget.
- Victory, last‑player‑standing, and stalemates.
- Board invariants, repair logic, and long‑running / soak behaviour.

Each finding below is classified twice:

- **Interaction fidelity** (per‑theme): `Faithful`, `Benign deviation`, `Design choice`, `Probable defect`.
- **Issue status** (per‑CCE entry, §4): `Design‑intent match`, `Intentional but under‑documented`, `Implementation compromise`, `Probable defect`.

## 2. Cross‑Rule Interaction Findings by Theme

Each subsection lists:

- Involved RR‑CANON rules.
- Primary implementation entry points.
- Intended interaction per RR‑CANON.
- Observed behaviour from code + tests.
- Interaction‑level classification and gameplay impact.

### 2.1 Turn / Phase / Forced Elimination Interactions

**RR‑CANON rules:** `R070–R072`, `R080–R082`, `R100`, `R170–R173` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:181)).  
**Key implementations:** [`TypeScript.turnLogic.advanceTurnAndPhase()`](src/shared/engine/turnLogic.ts:135), [`TypeScript.advanceGameForCurrentPlayer`](src/server/game/turn/TurnEngine.ts:91), [`TypeScript.RuleEngine`](src/server/game/RuleEngine.ts:752), [`TypeScript.ClientSandboxEngine.startTurnForCurrentPlayer`](src/client/sandbox/ClientSandboxEngine.ts:1606), [`TypeScript.ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer`](src/client/sandbox/ClientSandboxEngine.ts:1715), Python strict invariants (`ai-service/tests/invariants/**`).

#### 2.1.1 Intended interaction

- Phases execute in the strict order placement → movement / capture (including chains) → line processing → territory processing → victory ([`RR‑CANON‑R070–R071`](RULES_CANONICAL_SPEC.md:181)).
- At the start of a player’s action, if they control a stack but have **no legal placement, movement, or capture**, they must perform forced elimination ([`RR‑CANON‑R072`](RULES_CANONICAL_SPEC.md:196), [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:269)).
- Placement is:
  - Mandatory when no controlled stack has a legal move/capture.
  - Optional (skippable) when some controlled stack can move/capture.
  - Forbidden when `ringsInHand == 0` ([`RR‑CANON‑R080`](RULES_CANONICAL_SPEC.md:208)).
- Pending line / territory decisions must **not** block entry into forced elimination or victory checks at the end of a full turn.

#### 2.1.2 Observed behaviour

- **Backend / shared engine**
  - [`TypeScript.turnLogic.advanceTurnAndPhase()`](src/shared/engine/turnLogic.ts:135) exactly enforces the phase ladder: `ring_placement → movement/capture/chain_capture → line_processing → territory_processing`, then rotates `currentPlayer` and considers forced elimination for the next player.
  - Forced elimination eligibility uses delegates `hasAnyPlacement`, `hasAnyMovement`, `hasAnyCapture`. Backend implementations in [`TypeScript.TurnEngine.hasValidPlacements`](src/server/game/turn/TurnEngine.ts:208) and siblings query [`TypeScript.RuleEngine.getValidMoves`](src/server/game/RuleEngine.ts:752), so “no actions” is defined in terms of actual move enumeration, including `skip_placement`.
  - If a player has stacks but `hasAnyPlacement == hasAnyMovement == hasAnyCapture == false`, `advanceTurnAndPhase` calls `applyForcedElimination`, which in backend is [`TypeScript.turn.processForcedElimination()`](src/server/game/turn/TurnEngine.ts:286) eliminating an entire cap from a chosen stack and then re‑running victory checks. This is consistent with [`RR‑CANON‑R100`](RULES_CANONICAL_SPEC.md:269).
  - Line and territory processing are always entered **after** movement / capture (including chains) and always completed **before** turn rotation. Forced elimination is only considered after `territory_processing`, so pending lines/regions never block it.
- **Sandbox (historical path; legacy `sandboxTurnEngine`, now removed)**
  - In pre‑consolidation builds, sandbox phase advancement used the shared sequencer via [`TypeScript.sandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:81) but wrapped it with start‑of‑turn helpers.
  - At the beginning of a turn, [`TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:164) called [`TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:228), which:
  - In the current architecture, the same semantics are implemented by [`TypeScript.ClientSandboxEngine` turn helpers](src/client/sandbox/ClientSandboxEngine.ts:1606) composed with the shared [`TypeScript.turnLogic.advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135); sandbox no longer uses a separate consolidated `sandboxTurnEngine.ts` module.
    - Computes controlled stacks and approximated ring cap usage.
    - Uses `hooks.hasAnyLegalMoveOrCaptureFrom` and `hooks.enumerateLegalRingPlacements` (both wired back to shared helpers) to decide whether any move/capture/placement exists.
    - If no placements/movements/captures exist but stacks do, calls `forceEliminateCap` and immediately advances to the next player.
  - After forced‑elimination pre‑pass, sandbox chooses starting phase by `ringsInHand > 0 ? 'ring_placement' : 'movement'`, **without** re‑checking whether any legal placements actually exist. This can produce states where `currentPhase === 'ring_placement'` but the only legal action is effectively to skip and move.
- **Python**
  - Python’s `GameEngine` mirrors backend enumeration for placements/movements/captures, and exposes a FORCED_ELIMINATION move whenever stacks exist but no actions are legal.
  - A strict “no active player without a legal move or forced elimination” invariant (see [`archive/RULES_STATIC_VERIFICATION.md`](../../archive/RULES_STATIC_VERIFICATION.md:918)) ensures dynamically that every ACTIVE state offers at least one action to the `current_player`, including forced elimination.

#### 2.1.3 Classification & impact

- **Interaction classification:** `Faithful` for the backend and shared engine; `Benign deviation` for sandbox phase labelling / `skip_placement` representation.
- **Gameplay impact:**
  - Legal action sets (including forced elimination) are consistent across backend, sandbox, and Python for all known scenarios. No evidence of a player being denied a legal forced elimination when required.
  - Sandbox differences are **representation‑level** (no explicit `skip_placement` move, `currentPhase` sometimes remains `'ring_placement'` when only movement is possible). These primarily affect UX and trace/parity tooling rather than legality.
- **Linked CCE issues:** `CCE‑003`, `CCE‑007`.

### 2.2 Movement, Capture, Lines, and Territory in Sequence

**RR‑CANON rules:** `R090–R092`, `R100–R103`, `R120–R122`, `R140–R145` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:231)).  
**Key implementations:** [`TypeScript.core`](src/shared/engine/core.ts:367), [`TypeScript.movementLogic.enumerateSimpleMoveTargetsFromStack()`](src/shared/engine/movementLogic.ts:55), [`TypeScript.captureLogic.enumerateCaptureMoves()`](src/shared/engine/captureLogic.ts:26), backend and sandbox movement / capture application, [`TypeScript.lineDecisionHelpers`](src/shared/engine/lineDecisionHelpers.ts:1), [`TypeScript.territoryProcessing.applyTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:172), [`TypeScript.territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:123), [`TypeScript.turnLogic.advanceTurnAndPhase()`](src/shared/engine/turnLogic.ts:171).

#### 2.2.1 Intended interaction

- Movement and capture segments:
  - Apply marker path effects immediately (departure marker, intermediate flips/collapses, landing‑cell own‑marker removal and top‑ring elimination) ([`RR‑CANON‑R092`](RULES_CANONICAL_SPEC.md:255), [`RR‑CANON‑R102`](RULES_CANONICAL_SPEC.md:295)).
  - Update stack geometry (captures moving rings to bottom) but **do not** yet process lines or territory.
- After **all** movement and any full capture chain:
  - Detect and process lines sequentially ([`RR‑CANON‑R120–R122`](RULES_CANONICAL_SPEC.md:322)).
  - Then detect and process disconnected regions ([`RR‑CANON‑R140–R145`](RULES_CANONICAL_SPEC.md:359)).
  - Then evaluate victory ([`RR‑CANON‑R170–R173`](RULES_CANONICAL_SPEC.md:415)).
- Line and territory processing can change both elimination totals and territory counts within the **same turn**, and those updated values must be seen by victory checks.

#### 2.2.2 Observed behaviour

- Movement and capture reachability and paths are centralised in [`TypeScript.core.hasAnyLegalMoveOrCaptureFromOnBoard()`](src/shared/engine/core.ts:367) and [`TypeScript.movementLogic.enumerateSimpleMoveTargetsFromStack()`](src/shared/engine/movementLogic.ts:55), which:
  - Use [`TypeScript.getMovementDirectionsForBoardType()`](src/shared/engine/core.ts:61) and [`TypeScript.getPathPositions()`](src/shared/engine/core.ts:96) to ensure straight‑line rays and distance ≥ stackHeight.
  - Treat stacks and collapsed spaces as blocking, with markers allowed along the path.
- Marker path effects for both movement and captures are applied via [`TypeScript.applyMarkerEffectsAlongPathOnBoard()`](src/shared/engine/core.ts:619) using host‑provided marker mutators (`setMarker`, `flipMarker`, `collapseMarker`).
- Landing on own marker and immediate top‑ring elimination are implemented in movement/capture mutators (backend GameEngine and sandbox movement engine) and are **completed before** any line detection.
- Phase transitions in [`TypeScript.turnLogic.advanceTurnAndPhase()`](src/shared/engine/turnLogic.ts:171) always move from `movement` / `capture` / `chain_capture` to `line_processing`, and from `line_processing` to `territory_processing`, before any turn rotation.
- Line decisions:
  - Detection via [`TypeScript.lineDetection.findAllLines`](src/shared/engine/lineDetection.ts:21).
  - Decision surfaces and state updates via [`TypeScript.lineDecisionHelpers.enumerateProcessLineMoves()`](src/shared/engine/lineDecisionHelpers.ts:272), [`TypeScript.lineDecisionHelpers.applyProcessLineDecision()`](src/shared/engine/lineDecisionHelpers.ts:474), and [`TypeScript.lineDecisionHelpers.applyChooseLineRewardDecision()`](src/shared/engine/lineDecisionHelpers.ts:543).
  - Exact‑length lines always set `pendingLineRewardElimination = true`; overlength lines set it only for “collapse all” choices, matching [`RR‑CANON‑R122`](RULES_CANONICAL_SPEC.md:339).
- Territory decisions:
  - Region detection via [`TypeScript.territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:36).
  - Self‑elimination gating plus per‑region processing via [`TypeScript.territoryProcessing`](src/shared/engine/territoryProcessing.ts:99) and [`TypeScript.territoryDecisionHelpers.applyProcessTerritoryRegionDecision()`](src/shared/engine/territoryDecisionHelpers.ts:234).
  - `applyProcessTerritoryRegionDecision` updates `players[*].territorySpaces`, `players[*].eliminatedRings`, and `totalRingsEliminated`, and sets `pendingSelfElimination = true`.
- Both backend and sandbox attach victory checks **after** line and territory decisions in their move / decision pipelines, ultimately funnelling into [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45).

#### 2.2.3 Classification & impact

- **Interaction classification:** `Faithful`.
- **Gameplay impact:** The intended ordering “movement / capture (with markers) → lines → territory → victory” is preserved exactly. All automatic consequences of a move are visible to victory evaluation before the next player’s turn, and dynamic scenarios (FAQ Q7, Q15, Q23, combined line+region examples) confirm that elimination and territory changes within a single turn are applied in the intended sequence.
- **Linked CCE issues:** `CCE‑008` (documenting this as a design‑intent match).

### 2.3 Multi‑Region Territory & Self‑Elimination Budget

**RR‑CANON rules:** `R140–R145` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:359)).  
**Key implementations:** [`TypeScript.territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:36), [`TypeScript.territoryProcessing.canProcessTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:99), [`TypeScript.territoryProcessing.applyTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:172), [`TypeScript.territoryDecisionHelpers.enumerateProcessTerritoryRegionMoves()`](src/shared/engine/territoryDecisionHelpers.ts:123), [`TypeScript.territoryDecisionHelpers.applyProcessTerritoryRegionDecision()`](src/shared/engine/territoryDecisionHelpers.ts:234), [`TypeScript.territoryDecisionHelpers.enumerateTerritoryEliminationMoves()`](src/shared/engine/territoryDecisionHelpers.ts:402).

#### 2.3.1 Intended interaction

- Regions are maximal sets of non‑collapsed cells connected by `territoryAdjacency` ([`RR‑CANON‑R040`](RULES_CANONICAL_SPEC.md:111), [`RR‑CANON‑R140`](RULES_CANONICAL_SPEC.md:359)).
- A region is “disconnected” only if it is both:
  - Physically isolated by collapsed spaces and/or markers of a **single** border colour ([`RR‑CANON‑R141`](RULES_CANONICAL_SPEC.md:363)).
  - Colour‑disconnected: its RegionColors are a strict subset of ActiveColors ([`RR‑CANON‑R142`](RULES_CANONICAL_SPEC.md:371)).
- For each region and moving player `P`:
  - `P` may process the region only if, in the hypothetical state where the region’s rings are removed, `P` still controls at least one ring/cap **outside** the region ([`RR‑CANON‑R143`](RULES_CANONICAL_SPEC.md:378)).
  - Processing a region always requires **exactly one** self‑elimination from outside that region ([`RR‑CANON‑R145`](RULES_CANONICAL_SPEC.md:397)).
  - After processing a region and paying self‑elimination, subsequent regions must re‑check the prerequisite using the updated outside material; each outside ring/cap can pay for at most one region.
- Region processing order is per‑region optional; players may process any subset of eligible regions in any order ([`RR‑CANON‑R144`](RULES_CANONICAL_SPEC.md:386)).

#### 2.3.2 Observed behaviour

- Disconnected regions are computed centrally by [`TypeScript.territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:36) and re‑used by backend and sandbox via [`TypeScript.BoardManager.findDisconnectedRegions`](src/server/game/BoardManager.ts:907) and sandbox adapters.
- Processability is determined by [`TypeScript.territoryProcessing.canProcessTerritoryRegion`](src/shared/engine/territoryProcessing.ts:99), which checks for at least one controlled stack **outside** the region.
- [`TypeScript.territoryDecisionHelpers.enumerateProcessTerritoryRegionMoves`](src/shared/engine/territoryDecisionHelpers.ts:123) filters disconnected regions through `canProcessTerritoryRegion` at enumeration time, and hosts re‑call it after each region/self‑elimination pair, naturally implementing the “one outside cap per region” budget.
- [`TypeScript.territoryDecisionHelpers.applyProcessTerritoryRegionDecision`](src/shared/engine/territoryDecisionHelpers.ts:234) calls [`TypeScript.territoryProcessing.applyTerritoryRegion`](src/shared/engine/territoryProcessing.ts:172) to:
  - Eliminate all stacks inside the region, crediting all eliminations to the moving player.
  - Collapse interior and border marker cells to territory for the moving player.
  - Update `players[*].territorySpaces`, `players[*].eliminatedRings`, and `totalRingsEliminated`.
  - Set `pendingSelfElimination: true` for the processed region.
- Self‑elimination is then expressed as one or more `eliminate_rings_from_stack` decisions; [`TypeScript.territoryDecisionHelpers.enumerateTerritoryEliminationMoves`](src/shared/engine/territoryDecisionHelpers.ts:402) currently:
  - Defers self‑elimination while **any** processable disconnected region remains (region‑first ordering).
  - Enumerates candidate stacks purely by `controllingPlayer == player`, without tracking inside/outside relative to the processed region; however, interior stacks have already been removed by `applyTerritoryRegion`, so in practice only outside stacks remain.
- Multi‑region chains:
  - After each region + self‑elimination, engines re‑detect or re‑filter regions using current board state, so chain‑reaction regions (newly disconnected due to the collapse) are discovered and processed deterministically in whatever order the player chooses, matching [`RR‑CANON‑R144`](RULES_CANONICAL_SPEC.md:386).

#### 2.3.3 Classification & impact

- **Interaction classification:** `Faithful`, with a **minor design gap** around explicit “outside‑only” enforcement for self‑elimination.
- **Gameplay impact:**
  - Existing backend and sandbox flows only ever self‑eliminate from stacks that are outside already‑processed regions (interior stacks are removed first), so current behaviour is aligned with RR‑CANON for all known scenarios.
  - The separation between region processing and self‑elimination in shared helpers means that **future variants** could accidentally allow eliminations from inside a region if host code changes order; this should be guarded by tests and (ideally) by using `processedRegionId` in elimination enumeration.
- **Linked CCE issues:** `CCE‑005`.

### 2.4 Victory, Last‑Player‑Standing, and Stalemates

**RR‑CANON rules:** `R170–R173`, `R190–R191` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:413)).  
**Key implementations:** [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45), [`TypeScript.turnLogic`](src/shared/engine/turnLogic.ts:135), backend `RuleEngine.checkGameEnd`, sandbox victory wrappers, Python `GameEngine` and strict no‑move invariants (`ai-service/tests/invariants/**`).

#### 2.4.1 Intended interaction

- Elimination victory when a player’s credited eliminations reach `victoryThreshold` ([`RR‑CANON‑R170`](RULES_CANONICAL_SPEC.md:415)).
- Territory victory when `territorySpaces` reach `territoryVictoryThreshold` ([`RR‑CANON‑R171`](RULES_CANONICAL_SPEC.md:420)).
- Last‑player‑standing when, after **three consecutive full rounds** of turns:
  - One player (P) has at least one legal real action and takes at least one during each of those rounds, and
  - All others have no legal placements, movements, or captures during those rounds (even if they own buried rings) ([`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:703)).
- Global stalemate only when **no player** has any legal placement, movement, capture, or forced elimination, which can occur only after all stacks are gone; tie‑breaking ladder territory → eliminated rings (including converted rings in hand) → markers → last actor ([`RR‑CANON‑R173`](RULES_CANONICAL_SPEC.md:433)).
- S‑invariant `S = M + C + E` must be monotone and bounded under all **legal** moves; illegal states are outside the rules model ([`RR‑CANON‑R191`](RULES_CANONICAL_SPEC.md:455)).

#### 2.4.2 Observed behaviour

- [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45):
  - Implements ring‑elimination and territory victories exactly as specified, using pre‑computed thresholds.
  - Only considers stalemate and tie‑breaking on a **bare board** (`board.stacks.size === 0`):
    - If any player with rings in hand has a legal placement satisfying no‑dead‑placement, the game is **not** over.
    - Otherwise, treats all rings in hand as eliminated for tie‑breaking (`handCountsAsEliminated`), and walks the ladder territory → eliminated rings (including hand) → markers → last actor → `game_completed`.
  - There is **no explicit implementation** of non‑bare‑board last‑player‑standing as defined in [`RR‑CANON‑R172`](RULES_CANONICAL_SPEC.md:703) (three-round exclusive real-action condition); instead, games continue until ring‑elimination, territory victory, or bare‑board stalemate occurs.
- Turn rotation in [`TypeScript.turnLogic`](src/shared/engine/turnLogic.ts:181) and Python strict invariants ensure that:
  - Players with neither stacks nor rings in hand are skipped when choosing the next active player.
  - Any ACTIVE state must offer at least one move or forced elimination to the current player (Python enforces this; TS assumes it via sequencing).
- So‑called “last active player with material” positions (others having no stacks and no rings) are **functionally equivalent** to a state where:
  - The active player keeps taking turns until they win by elimination or territory, or until the game reaches bare‑board stalemate.
  - No additional early‑termination check for R172 is performed.
- Multi‑player stalemates (3–4 players) rely purely on the stalemate ladder; dynamic tests currently emphasise 2‑player cases, though the ladder is symmetric in player count.
- S‑invariant accounting is centralised in [`TypeScript.computeProgressSnapshot()`](src/shared/engine/core.ts:531) and is used extensively in backend / sandbox history and TS↔Python parity, as well as in Python self‑play soaks.

#### 2.4.3 Classification & impact

- **Interaction classification:**
  - `Faithful` for ring‑elimination, territory victories, bare‑board stalemate, and S‑invariant progress.
  - `Implementation compromise` for **non‑bare‑board last‑player‑standing** (R172 is not explicitly implemented in TS; Python appears to share this simplification).
- **Gameplay impact:**
  - For most games, the absence of an explicit R172 check is benign: the unique remaining active player will typically cross the elimination or territory threshold before S is exhausted.
  - However, RR‑CANON currently promises that such a player **wins immediately** by last‑player‑standing; the engines instead continue play until some other termination condition fires.
  - Edge cases with buried rings and complex resource distributions may therefore differ in **termination timing** and in whether the recorded victory reason is “last_player_standing” vs “ring_elimination” / “territory_control”.
- **Linked CCE issues:** `CCE‑006`.

### 2.5 Board Invariants, Repairs, and Long‑Running States

**RR‑CANON rules:** `R021–R023`, `R030–R031`, `R050–R052`, `R190–R191` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:69)).  
**Key implementations:** [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94), marker / collapsed helpers in [`TypeScript.BoardManager`](src/server/game/BoardManager.ts:325), shared `BoardState` invariants in [`src/shared/types/game.ts`](src/shared/types/game.ts:164), S‑invariant helpers in [`TypeScript.core.computeProgressSnapshot()`](src/shared/engine/core.ts:531), sandbox `ClientSandboxEngine.assertBoardInvariants`.

#### 2.5.1 Intended interaction

- Each cell is in exactly one of: empty, stack, marker, collapsed territory ([`RR‑CANON‑R021`](RULES_CANONICAL_SPEC.md:69), [`RR‑CANON‑R030–R031`](RULES_CANONICAL_SPEC.md:94), [`RR‑CANON‑R052`](RULES_CANONICAL_SPEC.md:146)).
- Legal rules **never** create overlapping occupancy (stack+marker, marker+collapsed, stack+collapsed); such states are considered unreachable.
- S‑invariant `S = M + C + E` must be monotone and bounded under all **legal** moves; illegal states are outside the rules model.

#### 2.5.2 Observed behaviour

- [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94) first performs a **repair pass**:
  - Deletes markers that share a key with a stack.
  - Deletes markers on collapsed spaces.
  - Logs diagnostics for both cases.
  - Only then checks invariants and throws in strict/test modes.
- Marker and collapsed helpers (`setMarker`, `collapseMarker`, `setCollapsedSpace`, `setStack`) all call `assertBoardInvariants` after updates, so any overlapping writes are immediately “fixed” by deleting markers and preserving stacks / collapsed spaces.
- Sandbox mirrors exclusivity semantics in [`TypeScript.ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts:964), but its `assertBoardInvariants` helper in tests chooses to **throw** rather than repair; there is no marker‑deletion repair step there.
- Because repairs are only triggered by **already invalid** states, legal move sequences are not intended to invoke this behaviour; however, nothing in the production backend prevents such repairs from happening silently if a bug or external state mutation occurs.
- When a repair deletes a marker without an accompanying collapse or elimination, the S‑invariant `S = M + C + E` can **decrease**: `M` drops while `C` and `E` stay constant. This is acceptable for illegal states, but violates the monotonicity argument if such a state ever arose from a sequence of ostensibly legal moves.
- Python self‑play soaks and invariant tests have not reported any repairs being necessary; instead, they treat active‑no‑moves and similar anomalies as hard errors.

#### 2.5.3 Classification & impact

- **Interaction classification:** `Benign deviation` (defensive repair) under the assumption that repaired states are unreachable from legal play.
- **Gameplay impact:**
  - From a rules perspective, repairs are **not part of play** and should never be visible to players. If they trigger, they represent engine bugs or corrupted external inputs, not alternative legal states.
  - Silent repairs risk hiding the root cause of such bugs and could, in principle, cause S‑invariant decreases and loss of markers that would otherwise contribute to lines or borders.
  - Static and dynamic verification currently treat this as an **implementation safeguard**, but both recommend adding counters / metrics so any repair on a legal trajectory is a hard failure in CI.
- **Linked CCE issues:** `CCE‑001`.

## 3. Specific High‑Focus Issues

This section answers the five explicitly requested questions, with cross‑references to concrete `CCE‑00X` entries in §4.

### 3.1 Board repair behaviour (`CCE‑001`)

- **RR‑CANON:** Exclusivity of stack / marker / collapsed (`R021`, `R030–R031`, `R052`, `R191`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:69).
- **Code:** [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94) and stack / marker / collapsed setters.
- **Assessment:**
  - Repairs (deleting overlapping markers) are a **purely defensive mechanism**, not part of RR‑CANON semantics.
  - Under legal rules, such overlaps should be unreachable; any repair indicates a bug or externally constructed malformed state.
  - In production, repairs can fire if:
    - A move application path mis‑orders stack / marker writes, or
    - External tools (tests, admin scripts, parity harnesses) write inconsistent board states.
  - When they do fire, they deterministically favour stacks / collapsed spaces over markers and can locally decrease the S‑invariant.
- **Classification:** `Implementation compromise` (coded behaviour deviates from the “no illegal states” ideal but is intentionally defensive).
- **Recommendation:** keep repairs as an emergency defence but:
  - Instrument them with counters / metrics so any repair on a trajectory starting from a legal initial state is a **hard CI failure**.
  - In long‑running production games, treat repeated repairs as a P0 incident and prefer failing fast over auto‑healing silently.

### 3.2 Placement cap approximation with mixed‑colour stacks (`CCE‑002`)

- **RR‑CANON:** Ring counts and caps (`R020–R023`, `R060–R062`, `R080–R082`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:61).
- **Code:** `ringsPerPlayer` cap enforced using **total height of stacks controlled by the player**, not the count of rings of that colour, in:
  - Backend placement validator and enumeration (see [`archive/RULES_STATIC_VERIFICATION.md`](../../archive/RULES_STATIC_VERIFICATION.md:755)).
  - Sandbox `hasAnyPlacement` and forced‑elimination gating via [`TypeScript.ClientSandboxEngine` turn helpers](src/client/sandbox/ClientSandboxEngine.ts:1606).
- **Assessment:**
  - The implementation counts **all rings in stacks whose top ring the player controls**, including buried opponent rings, when enforcing a per‑player cap based on `BOARD_CONFIGS[boardType].ringsPerPlayer`.
  - This is a **conservative approximation**: it can only **forbid** some placements that would be legal under a strict “by‑colour rings on board” reading; it cannot create extra rings or allow over‑cap placements.
  - In extreme mixed‑stack positions near the cap, a player may be prevented from placing additional rings even though they still have unused own‑colour rings in hand and relatively few of their own colour on board.
  - These states are rare and already near resource saturation; dynamic tests do not currently exercise them.
- **Classification:** `Implementation compromise` (behaviour deviates from the most natural textual reading but is conservative and simplifies enforcement).
- **Recommendation:** either:
  - **Canonicalise** this as “placement cap is based on the total height of stacks you control, including buried rings, up to `ringsPerPlayer`”, and update [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:61) accordingly; or
  - Tighten the implementation to count only own‑colour rings, and add explicit regression tests for mixed‑colour, near‑cap configurations to ensure parity across TS and Python.

### 3.3 Sandbox phase / skip semantics vs backend (`CCE‑003`)

- **RR‑CANON:** Placement optionality and turn phases (`R070–R072`, `R080–R082`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:181).
- **Code:** Backend `skip_placement` moves in [`TypeScript.RuleEngine`](src/server/game/RuleEngine.ts:116) and backend turn logic, vs sandbox heuristics in [`TypeScript.ClientSandboxEngine.startTurnForCurrentPlayer`](src/client/sandbox/ClientSandboxEngine.ts:1606) and [`TypeScript.ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer`](src/client/sandbox/ClientSandboxEngine.ts:1715).
- **Assessment:**
  - Backend expresses optional placement explicitly via `skip_placement` moves; move logs and AI policies see a clear distinction between “chose not to place” and “could not place”.
  - Sandbox never surfaces `skip_placement` as a Move; it:
    - Uses shared placement validators for legality and no‑dead‑placement.
    - Starts turns in `ring_placement` whenever `ringsInHand > 0`, even if no legal placements exist, and then relies on movement / capture interaction paths to proceed.
  - For all known scenarios:
    - The **set of reachable board states per turn** is the same in sandbox and backend.
    - Forced elimination is triggered under the same preconditions.
  - The divergence is therefore in **phase labels and trace shape**, not in underlying legality.
- **Classification:** `Intentional but under‑documented`.
- **Recommendation:**
  - Document this asymmetry explicitly and add sandbox parity tests that assert backend and sandbox reach the same successor states even when sandbox does not emit a `skip_placement` move.
  - Longer term, consider introducing an internal sandbox `skip_placement` move type (even if not exposed in the UI) to simplify parity tooling.

### 3.4 Capture‑chain helpers vs active implementations (`CCE‑004`)

- **RR‑CANON:** Chain capture semantics (`R101–R103`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:281).
- **Code:** Shared stub [`TypeScript.captureChainHelpers.enumerateChainCaptureSegments()`](src/shared/engine/captureChainHelpers.ts:134) and [`TypeScript.captureChainHelpers.getChainCaptureContinuationInfo()`](src/shared/engine/captureChainHelpers.ts:163) vs live logic in:
  - Backend [`TypeScript.captureChainEngine`](src/server/game/rules/captureChainEngine.ts:45) and [`TypeScript.GameEngine`](src/server/game/GameEngine.ts:92).
  - Sandbox [`TypeScript.sandboxMovementEngine.performCaptureChainSandbox`](src/client/sandbox/sandboxMovementEngine.ts:400).
- **Assessment:**
  - All **live** capture‑chain behaviour (mandatory continuation, 180° reversals, revisiting stacks under cap‑height constraints, landing‑on‑own‑marker elimination) is implemented and tested at the backend / sandbox level (FAQ Q15 scenarios, cyclic captures, etc.).
  - The shared helper is an explicit **design‑time stub** which currently throws if called; no production path depends on it.
  - Future refactors that centralise chain logic into the shared helper must preserve:
    - Enumeration equivalence with [`TypeScript.captureLogic.enumerateCaptureMoves()`](src/shared/engine/captureLogic.ts:26).
    - Mandatory continuation while any capture exists.
    - Freedom of direction change and revisiting targets where cap legality allows.
    - Deferral of line / territory processing until after the entire chain completes.
- **Classification:** `Intentional but under‑documented` (the stub is clearly labelled, but there is no regression suite yet tying its future implementation to current host behaviour).
- **Recommendation:** before wiring hosts to the shared helper, add a dedicated `captureChainHelpers.shared` test suite that replays the FAQ Q15 patterns and asserts equality between helper‑produced segments and current backend / sandbox options.

### 3.5 Victory edge cases (`CCE‑006`)

- **RR‑CANON:** Victory, last‑player‑standing, stalemate ladder (`R170–R173`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:415).
- **Code:** [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45), backend `RuleEngine.checkGameEnd`, sandbox victory wrappers, Python `GameEngine`.
- **Assessment:**
  - Elimination and territory victories, and bare‑board stalemate with tiebreak ladder, are implemented consistently across hosts and match RR‑CANON.
  - RR‑CANON’s **non‑bare‑board last‑player‑standing** condition (one player with legal actions on their next turn while others have none) is **not currently encoded** in TS victory logic; instead, such games continue until elimination, territory, or stalemate conditions are met.
  - Python appears to mirror this simplification; its strict invariant focuses on “no ACTIVE player without a move or forced elimination”, not on early last‑player‑standing wins.
  - Multi‑player stalemates (3–4 players) are logically covered by the ladder, but existing tests emphasise 2‑player cases; more scenarios would increase confidence but are unlikely to reveal qualitative divergence.
- **Classification:** `Implementation compromise` (a subset of RR‑CANON is not yet implemented, but behaviour is deterministic and terminating).
- **Recommendation:**
  - Decide whether RR‑CANON should **retain** the explicit R172 condition or whether “last‑player‑standing” is purely a tiebreak label in stalemate contexts.
  - If R172 is retained, extend `evaluateVictory` (and Python) to implement it explicitly, and add targeted tests (e.g. “last player with material and buried opponents”) to lock in behaviour.

### 3.6 Territory self‑elimination outside vs inside processed region (`CCE‑005`)

- **RR‑CANON:** Self‑elimination prerequisite and processing order (`R143–R145`) in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:378).
- **Code:** [`TypeScript.territoryProcessing.canProcessTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:99), [`TypeScript.territoryProcessing.applyTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:172), [`TypeScript.territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:123,234,402).
- **Assessment:**
  - Enumeration and application correctly enforce “must have an outside stack before processing a region” and “one outside cap per region”.
  - Shared elimination helper [`TypeScript.territoryDecisionHelpers.enumerateTerritoryEliminationMoves`](src/shared/engine/territoryDecisionHelpers.ts:402) currently **ignores** `processedRegionId` and simply offers eliminations from any controlled stack.
  - In current flows this is safe because all interior stacks have already been removed by `applyTerritoryRegion`, so only outside stacks remain.
  - The rules, however, conceptually distinguish between “outside” and “inside” when paying the self‑elimination cost.
- **Classification:** `Design‑intent match` for current flows, with an **under‑specified hook** for future variants.
- **Recommendation:**
  - Clarify in documentation that, given current sequencing, all candidate stacks for self‑elimination are necessarily outside the processed region.
  - If future variants want to allow deferred or batched self‑elimination, extend the helper to enforce an explicit “outside only” constraint using `processedRegionId`.

## 4. Issue Catalogue (CCE‑00X)

Each entry below lists RR‑CANON references, code touchpoints, observed vs intended behaviour, classification, severity, scope, and recommended follow‑up.

### CCE‑001 – Backend board “repair” deletes overlapping markers

- **RR‑CANON rules:** `R021`, `R030–R031`, `R050–R052`, `R191` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:69)).
- **Code / tests:** [`TypeScript.BoardManager.assertBoardInvariants()`](src/server/game/BoardManager.ts:94), [`TypeScript.BoardManager.setMarker`](src/server/game/BoardManager.ts:325), [`TypeScript.BoardManager.setStack`](src/server/game/BoardManager.ts:446), sandbox invariant checks in `ClientSandboxEngine.assertBoardInvariants` and invariant soaks documented in [`docs/STRICT_INVARIANT_SOAKS.md`](docs/STRICT_INVARIANT_SOAKS.md:1).
- **Interaction / edge case:** Buggy or external writes create cells that simultaneously contain a stack plus marker or marker plus collapsed space.
- **Intended behaviour (RR‑CANON):** Such states are unreachable; if they occur, semantics are undefined and should be treated as hard errors, not silently corrected.
- **Observed behaviour:** Backend logs a diagnostic and **repairs** the state by deleting markers while keeping stacks / collapsed spaces, even in non‑test environments, then enforces invariants on the repaired board.
- **Classification:** `Implementation compromise`.
- **Severity:** `Medium` – does not affect legal trajectories under correct engines, but can hide real defects and violate S‑invariant monotonicity for repaired states.
- **Scope:** Backend only; sandbox treats overlaps as assertion failures in tests.
- **Recommendation:**
  - Add a repair counter (or metric) and assert in CI that it is zero on long seeded traces and parity runs.
  - Consider gating repair behaviour behind an explicit debug flag even in production, so that catastrophic corruption surfaces promptly rather than being silently patched.

### CCE‑002 – Placement cap approximation counts all rings in controlled stacks

- **RR‑CANON rules:** `R020–R023`, `R060–R062`, `R080–R082` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:61)).
- **Code / tests:** Shared placement validator and RuleEngine enumeration described in [`archive/RULES_STATIC_VERIFICATION.md`](../../archive/RULES_STATIC_VERIFICATION.md:755); sandbox `hasAnyPlacement` and forced‑elimination gating in [`TypeScript.sandboxTurnEngine`](src/client/sandbox/sandboxTurnEngine.ts:122,278); AI / Python parity tests around placement capacity in `ai-service/tests/**`.
- **Interaction / edge case:** Near‑cap states with tall mixed‑colour stacks under a player’s control (many captured opponent rings under a small own‑colour cap) where total stack heights exceed `ringsPerPlayer` even though few own‑colour rings are actually on the board.
- **Intended behaviour (RR‑CANON):** Caps conceptually apply to rings of that player’s colour; as long as ring conservation is respected, additional placements of own‑colour rings should be legal up to the physical supply.
- **Observed behaviour:** Implementation forbids further placements once the sum of **heights** of controlled stacks reaches `ringsPerPlayer`, regardless of ring colours within those stacks.
- **Classification:** `Implementation compromise`.
- **Severity:** `Low` – conservative; can only under‑approximate placement options in rare, already resource‑dense positions.
- **Scope:** Backend, sandbox, and Python (for parity).
- **Recommendation:** Decide whether to:
  - Embrace this as the canonical “cap based on rings in stacks you control (all colours)” and update RR‑CANON, or
  - Tighten the implementation to count only own‑colour rings, with a dedicated mixed‑colour cap test to keep TS / Python aligned.

### CCE‑003 – Sandbox phase / skip semantics differ from backend

- **RR‑CANON rules:** `R070–R072`, `R080–R082` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:181)).
- **Code / tests:** Backend `skip_placement` handling in [`TypeScript.RuleEngine`](src/server/game/RuleEngine.ts:116,752), shared turn logic in [`TypeScript.turnLogic`](src/shared/engine/turnLogic.ts:135), sandbox start‑of‑turn helpers in [`TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:164) and [`TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox`](src/client/sandbox/sandboxTurnEngine.ts:228), parity tests in `tests/unit/ClientSandboxEngine.*.test.ts`.
- **Interaction / edge case:** Turns where placement is optional (legal placements **and** legal moves exist) or effectively impossible (rings in hand but no legal placements; only movement is available).
- **Intended behaviour (RR‑CANON):**
  - Optional placement should be representable as either “place N rings” or “skip placement, then move”.
  - When no legal placements exist but moves do, the only legal action in the placement phase is to skip.
- **Observed behaviour:**
  - Backend surfaces explicit `skip_placement` moves in both situations.
  - Sandbox:
    - Never emits a `skip_placement` move.
    - Starts in `ring_placement` whenever `ringsInHand > 0`, even if no placements exist, and expects UI / AI to proceed directly to movement.
  - Despite this, all known scenarios reach the **same successor board states** in sandbox and backend; the divergence is in logs and in the value of `currentPhase`.
- **Classification:** `Intentional but under‑documented`.
- **Severity:** `Low–Medium` – safe for rules, but can confuse trace analysis and sandbox UX in rare “only skip is legal” positions.
- **Scope:** Sandbox only.
- **Recommendation:**
  - Add a dedicated parity test that asserts equality of reachable states between backend (with `skip_placement`) and sandbox (without) for edge scenarios.
  - Optionally add an internal sandbox `skip_placement` representation to reduce divergence for tooling.

### CCE‑004 – Shared `captureChainHelpers` unimplemented while hosts are live

- **RR‑CANON rules:** `R101–R103` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:281)).
- **Code / tests:** [`TypeScript.captureChainHelpers`](src/shared/engine/captureChainHelpers.ts:134), backend [`TypeScript.captureChainEngine`](src/server/game/rules/captureChainEngine.ts:45), sandbox [`TypeScript.ClientSandboxEngine.performCaptureChainInternal`](src/client/sandbox/ClientSandboxEngine.ts:2229), FAQ Q15 and cyclic capture tests in [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1) and `tests/unit/GameEngine.cyclicCapture.*.test.ts`.
- **Interaction / edge case:** Complex capture chains (180° reversal, revisiting stacks, long cycles) where refactoring to shared helpers could subtly change continuation sets or mandatory‑continuation semantics.
- **Intended behaviour (RR‑CANON):** Chains must continue while any legal capture exists; direction changes and revisiting stacks are allowed as long as each segment is legal; lines and territory are deferred until the chain ends.
- **Observed behaviour:** Current engines obey these rules at the GameEngine / sandbox level. The shared `enumerateChainCaptureSegments` helper is a stub that throws, and is **not** used in production flows yet.
- **Classification:** `Intentional but under‑documented`.
- **Severity:** `Low` today (no runtime impact), `Medium` risk during future refactors if the helper is wired in without a strong regression suite.
- **Scope:** Future shared‑engine refactors; no effect on current gameplay.
- **Recommendation:** Before adopting the shared helper:
  - Write tests that compare its output against existing backend / sandbox chains on all FAQ Q15 and cyclic scenarios.
  - Only then switch hosts to call the helper, preserving behaviour.

### CCE‑005 – Territory self‑elimination locality is implicit, not enforced

- **RR‑CANON rules:** `R143–R145` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:378)).
- **Code / tests:** [`TypeScript.territoryProcessing`](src/shared/engine/territoryProcessing.ts:99,172), [`TypeScript.territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:123,234,402), territory tests in [`tests/unit/territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1) and [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1).
- **Interaction / edge case:** Multi‑region turns where multiple disconnected regions exist and self‑elimination budget is tight; requirement that self‑elimination come from **outside** the processed region.
- **Intended behaviour (RR‑CANON):** Each processed region must be paid for by eliminating a ring/cap from **outside** that region; processing R1 may remove the only such cap, preventing immediate processing of R2.
- **Observed behaviour:**
  - Enumeration and `canProcessTerritoryRegion` enforce the outside‑stack prerequisite on the **pre‑region** board.
  - `applyTerritoryRegion` eliminates all interior stacks, so subsequent self‑elimination decisions by hosts necessarily come from outside.
  - Shared enumeration of self‑elimination moves (`enumerateTerritoryEliminationMoves`) currently does not encode inside/outside explicitly but is safe given current sequencing.
- **Classification:** `Design‑intent match` for current flows; potential **under‑specification** if future variants change the timing of self‑elimination.
- **Severity:** `Low`.
- **Scope:** All hosts, but only in the presence of complex multi‑region turns.
- **Recommendation:**
  - Keep current sequencing, and document clearly that all candidate self‑elimination stacks are outside processed regions by construction.
  - If a future rule variant allows deferred or batched self‑elimination, extend helpers to enforce explicit outside‑only constraints using `processedRegionId`.

### CCE‑006 – Non‑bare‑board last‑player‑standing (R172) not explicitly implemented

- **RR‑CANON rules:** `R170–R173` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:415)).
- **Code / tests:** [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45), backend victory scenarios in [`tests/unit/GameEngine.victory.scenarios.test.ts`](tests/unit/GameEngine.victory.scenarios.test.ts:1), stalemate / forced‑elimination scenarios in [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:1), Python invariants in `ai-service/tests/invariants/**`.
- **Interaction / edge case:** Games where exactly one player has any legal actions on their next turn while others are permanently inactive (no placements, moves, or captures) but stacks remain on the board.
- **Intended behaviour (RR‑CANON):** Such a player should win immediately by last‑player‑standing.
- **Observed behaviour:**
  - TS victory logic only checks elimination / territory thresholds and bare‑board stalemate; it does not compute per‑player legal‑action availability on future turns for non‑bare‑board states.
  - Python appears to follow the same pattern; its strict invariant ensures no ACTIVE player lacks moves / forced elimination but does not promote last‑player‑standing as a separate victory condition.
- **Classification:** `Implementation compromise`.
- **Severity:** `Medium` – rare, but behaviour diverges from the written RR‑CANON text in timing and victory reason.
- **Scope:** All hosts (backend, sandbox, Python) for long games where some players become permanently inactive without losing all material.
- **Recommendation:**
  - Decide whether to remove or soften R172 in the canonical spec, or to implement it fully (TS + Python) with explicit tests (including buried‑ring examples).
  - Until then, document that current engines only implement last‑player‑standing as part of the stalemate ladder, not as an early‑termination rule.

### CCE‑007 – Forced‑elimination stack‑selection heuristic

- **RR‑CANON rules:** `R072`, `R100` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:196)).
- **Code / tests:** [`TypeScript.turn.processForcedElimination`](src/server/game/turn/TurnEngine.ts:286), [`TypeScript.ClientSandboxEngine.forceEliminateCap`](src/client/sandbox/ClientSandboxEngine.ts:888), invariant tests in `ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py`.
- **Interaction / edge case:** When multiple stacks are eligible for forced elimination, RR‑CANON does not specify which cap must be chosen.
- **Intended behaviour (RR‑CANON):** Any full cap from a controlled stack may be eliminated; tiebreak is unspecified.
- **Observed behaviour:**
  - Backend heuristic chooses the stack with the smallest positive `capHeight` (falling back to the first stack if caps are degenerate).
  - Sandbox uses similar “smallest cap first” logic via its elimination helper.
  - Python mirrors this heuristic for parity.
- **Classification:** `Design‑intent match` (implementation chooses a deterministic, but rules‑permitted, tiebreak).
- **Severity:** `Low`.
- **Scope:** All hosts, only when forced elimination is triggered and multiple caps are available.
- **Recommendation:** Document the heuristic explicitly (so AI / UX can rely on it when no player choice is surfaced) and add a simple regression test to guarantee future changes remain deterministic.

### CCE‑008 – Movement / capture / lines / territory / victory ordering

- **RR‑CANON rules:** `R090–R092`, `R100–R103`, `R120–R122`, `R140–R145`, `R170–R173` ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:231)).
- **Code / tests:** Movement / capture helpers in [`TypeScript.core`](src/shared/engine/core.ts:367), [`TypeScript.movementLogic`](src/shared/engine/movementLogic.ts:55), [`TypeScript.captureLogic`](src/shared/engine/captureLogic.ts:26); line and territory helpers in [`TypeScript.lineDecisionHelpers`](src/shared/engine/lineDecisionHelpers.ts:1) and [`TypeScript.territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:123); turn sequencing in [`TypeScript.turnLogic`](src/shared/engine/turnLogic.ts:135); dynamic scenarios in [`tests/unit/GameEngine.lines.scenarios.test.ts`](tests/unit/GameEngine.lines.scenarios.test.ts:1), [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1), and [`tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1).
- **Interaction / edge case:** Combined turns where a movement or capture chain:
  - Produces markers and collapsed spaces,
  - Forms one or more lines,
  - Creates one or more disconnected regions,
  - And potentially crosses victory thresholds within a single turn.
- **Intended behaviour (RR‑CANON):** All marker and stack effects from movement / capture (including landing‑on‑own‑marker elimination) complete before any line detection; all line processing completes before any territory processing; all such consequences are visible before victory evaluation.
- **Observed behaviour:** Shared helpers and turn logic enforce exactly this ordering; scenario tests (FAQ Q7, Q15, Q23, combined line+region cases) confirm that elimination and territory changes occur in the correct phase order and that victory evaluation sees the fully updated state.
- **Classification:** `Design‑intent match`.
- **Severity:** `Low`.
- **Scope:** All hosts and board types.
- **Recommendation:** Treat current behaviour as the reference ordering and keep future refactors pinned to the shared helpers, avoiding host‑specific shortcuts that bypass them.

### CCE‑009 – Recovery Action interactions with LPS, ANM, FE, and weird states

- **RR‑CANON rules:** `R110–R115` (Recovery Action), `R172` (Last Player Standing), `R200–R207` (Active No Moves), `R100` (Forced Elimination) ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:301)).
- **Code / tests:** Recovery action detection in [`TypeScript.globalActions.canPerformRecoveryAction()`](src/shared/engine/globalActions.ts:45), [`TypeScript.globalActions.enumerateRecoverySlides()`](src/shared/engine/globalActions.ts:89); line processing via [`TypeScript.lineDecisionHelpers`](src/shared/engine/lineDecisionHelpers.ts:1); victory logic in [`TypeScript.victoryLogic.evaluateVictory()`](src/shared/engine/victoryLogic.ts:45); ANM detection in shared engine helpers; weird state detection in [`TypeScript.gameStateWeirdness`](src/client/utils/gameStateWeirdness.ts:1).
- **Interaction / edge case:** A player has zero stacks on board, zero rings in hand, but still has:
  - At least one marker (from prior movements), AND
  - At least one "buried ring" (opponent's stack sitting on top of one of their rings).

  This player can perform a **recovery slide**: move a marker into a position that completes a line, triggering line collapse and potentially eliminating the opponent's cap on the buried ring. Key interaction questions:
  1. **With LPS (R172):** Does a recovery action count as a "real action" that resets the three-round LPS counter?
  2. **With ANM (R200–R207):** Does recovery availability prevent the ANM state (active player with turn-material but no legal global actions)?
  3. **With Forced Elimination (R100):** Can a player with recovery available still be forced to eliminate? (No—recovery is a legal action.)
  4. **With Weird States:** How should the UX present a "recovery only" turn where marker sliding is the sole legal action?

- **Intended behaviour (RR‑CANON):**
  - Recovery is a **real action** (involves board state change via marker slide + line collapse), so it resets the LPS counter and counts toward the "active player has legal actions" check.
  - A player with at least one legal recovery slide is **not** in an ANM state—they have a legal global action.
  - Forced Elimination (R100) only triggers when a player has NO legal placements, NO legal movements, NO legal captures, AND NO recovery actions. Recovery availability blocks forced elimination.
  - UX should recognize "recovery only" as a distinct but valid game state—potentially surfacing a teaching prompt or weird-state banner explaining the limited action set.

- **Observed behaviour:**
  - `canPerformRecoveryAction()` correctly checks the three preconditions (zero stacks, zero rings in hand, has markers AND buried rings).
  - `enumerateRecoverySlides()` produces valid marker-slide moves that complete lines.
  - Turn orchestration includes recovery in the legal-action enumeration, so forced elimination is correctly blocked when recovery is available.
  - ANM detection considers recovery availability in the "has any legal action" check.
  - LPS tracking treats recovery as a real action (increments the active-player counter, resets the inactivity tracker).
  - Weird state detection (`RWS-005` or similar) can surface "recovery only" scenarios, though teaching coverage for this specific state is minimal.

- **Classification:** `Design‑intent match` for core mechanics; `Intentional but under‑documented` for UX/teaching coverage of recovery-only states.
- **Severity:** `Low` for rules correctness; `Medium` for UX completeness.
- **Scope:** All hosts (backend, sandbox, Python) where recovery action is implemented.
- **Recommendation:**
  - Document the interaction matrix (Recovery × LPS × ANM × FE) explicitly in `RULES_CANONICAL_SPEC.md` for clarity.
  - Add or extend teaching scenarios and weird-state banners for "recovery only" turns (player has no stacks/rings but can still act via marker slides).
  - Ensure Python AI service correctly enumerates and evaluates recovery moves in position evaluation.
  - Add targeted regression tests that confirm: (a) recovery blocks forced elimination, (b) recovery resets LPS counter, (c) recovery prevents ANM classification.

#### CCE‑009a – Recovery Option 1 vs Option 2 cost model edge cases

- **RR‑CANON rules:** `R110–R115` (Recovery Action Option 1/2) ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:301)).
- **Interaction / edge case:** When a recovery slide creates an **overlength line** (longer than minimum collapse length), the recovering player must choose:
  - **Option 1:** Collapse all markers in the line. Costs 1 buried ring (extracted from any opponent stack containing the player's buried ring).
  - **Option 2:** Collapse only the minimum length. Free (no ring extraction).

  Key edge cases:
  1. **Single buried ring remaining:** If player has exactly 1 buried ring and creates an overlength line, Option 1 extracts that ring (costs recovery ability), while Option 2 preserves it.
  2. **Option 1 extraction target selection:** When multiple opponent stacks contain buried rings, which stack is the extraction target?
  3. **Option 2 marker selection:** For overlength lines, which subset of markers collapses?

- **Intended behaviour (RR‑CANON):**
  - Option 1 extracts the **bottommost buried ring** from a player-selected stack among eligible stacks.
  - Option 2 collapses the **minimum length** of the line, preserving extra markers; collapsed subset is deterministic (lowest-indexed markers in line order).
  - The choice between Option 1 and Option 2 is **always player-interactive** when both are available.

- **Observed behaviour:**
  - TS and Python implementations surface Option 1/2 as explicit `choose_line_option` decisions when applicable.
  - Extraction target selection follows "player's bottommost buried ring" from the chosen stack.
  - Option 2 marker selection uses deterministic line-scan order.

- **Classification:** `Design‑intent match`.
- **Severity:** `Low`.
- **Scope:** All hosts where recovery with overlength lines can occur.
- **Recommendation:** Add explicit test coverage for:
  - Recovery with exactly 1 buried ring (Option 1 vs Option 2 strategic choice).
  - Recovery with multiple extraction-eligible stacks (target selection).
  - Recovery with maximum overlength lines (Option 2 marker subset).

#### CCE‑009b – Recovery exhaustion and transition to permanent elimination

- **RR‑CANON rules:** `R110–R115` (Recovery eligibility), `R175` (Permanent elimination) ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:301)).
- **Interaction / edge case:** A player exhausts all recovery capability:
  1. All markers consumed by line collapses (no markers left to slide).
  2. All buried rings extracted via Option 1 choices (no buried rings remain).
  3. Opponent captures eliminate all buried rings.

  What happens when recovery eligibility is lost?

- **Intended behaviour (RR‑CANON):**
  - Player transitions from "recovery-eligible" to either:
    - **Temporarily eliminated** (still has rings somewhere in the game but no turn-material), or
    - **Permanently eliminated** (zero rings anywhere—hand, controlled stacks, buried in opponent stacks).
  - If temporarily eliminated, player may regain turn-material if their buried ring resurfaces (opponent's cap eliminated).
  - If permanently eliminated, player is out of the game but turn rotation still includes them for LPS tracking per RR-CANON-R172.

- **Observed behaviour:**
  - Victory logic correctly distinguishes temporary vs permanent elimination.
  - Turn rotation includes eliminated players for LPS round counting.
  - Recovery eligibility check correctly returns false when markers OR buried rings are exhausted.

- **Classification:** `Design‑intent match`.
- **Severity:** `Low`.
- **Scope:** All hosts for long games where recovery exhaustion can occur.
- **Recommendation:** Add regression tests for:
  - Player loses last marker (recovery unavailable, check if temporarily or permanently eliminated).
  - Player loses last buried ring via Option 1 extraction (recovery unavailable).
  - Player loses last buried ring via opponent capture (recovery unavailable).
  - Transition from recovery-eligible → temporarily eliminated → regains turn-material via resurfacing ring.

#### CCE‑009c – Recovery + territory chain scenarios

- **RR‑CANON rules:** `R110–R115` (Recovery), `R120–R122` (Lines), `R140–R145` (Territory) ([`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:301)).
- **Interaction / edge case:** A recovery slide triggers a line collapse that creates a disconnected territory region. Can the recovering player (who has zero stacks) process territory?

- **Intended behaviour (RR‑CANON):**
  - Territory processing requires at least one controlled stack **outside** the region to pay self-elimination cost.
  - A recovering player (zero stacks by definition) **cannot** process any territory regions.
  - Any territory regions created by recovery line collapse remain on the board until the recovering player regains stacks or another player processes them.

- **Observed behaviour:**
  - `canProcessTerritoryRegion()` correctly requires `controlledStacksOutsideRegion > 0`.
  - Recovering player has zero stacks, so territory processing is skipped.
  - Line processing completes, territory regions are detected but not processed, turn advances normally.

- **Classification:** `Design‑intent match`.
- **Severity:** `Low`.
- **Scope:** All hosts where recovery + territory chains can occur.
- **Recommendation:** Add test coverage for recovery slide → line collapse → territory region created → territory processing skipped due to zero stacks.

### CCE‑010 – Capture and chain capture landing on markers

- **RR‑CANON rules:** `R091–R092` (movement landing), `R101–R102` (capture landing) ([`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md:255)).
- **Code / tests:** [`TypeScript.CaptureAggregate.mutateCapture()`](src/shared/engine/aggregates/CaptureAggregate.ts:1) section 5 (landing marker handling), [`Python.game_engine.py`](ai-service/app/game_engine.py:1) (parallel implementation), [`tests/unit/captureLogic.shared.test.ts`](tests/unit/captureLogic.shared.test.ts:1), [`tests/unit/CaptureAggregate.chainCapture.shared.test.ts`](tests/unit/CaptureAggregate.chainCapture.shared.test.ts:1).
- **Interaction / edge case:** During captures and chain captures, the attacker may land on any marker (own or opponent) as a valid landing position. This interaction triggers a 1-ring elimination cost.
- **Intended behaviour (RR‑CANON):**
  - Per [`RR‑CANON‑R091`](RULES_CANONICAL_SPEC.md:255): "Landing cell may contain any marker (own or opponent); landing on markers is always legal but incurs a cap‑elimination cost."
  - Per [`RR‑CANON‑R092`](RULES_CANONICAL_SPEC.md:257): "At landing cell: If there is any marker (own or opponent), remove that marker (do NOT collapse it), then place the moving stack. If a marker was present (regardless of owner), immediately eliminate the top ring of the moving stack's cap and credit it to P."
  - Per [`RR‑CANON‑R101`](RULES_CANONICAL_SPEC.md:295): "Landing is an empty cell or a cell with any marker (own or opponent)."
  - Per [`RR‑CANON‑R102`](RULES_CANONICAL_SPEC.md:297): "If landing on any marker (own or opponent), remove the marker (do NOT collapse it), then land and immediately eliminate the top ring of the attacking stack's cap, crediting it to P (before line/territory processing)."
- **Observed behaviour:**
  - **TypeScript:** `CaptureAggregate.mutateCapture()` checks for `landingMarker` after capture application. If present: removes marker from board (does NOT collapse), eliminates top ring of attacker's cap, updates `totalRingsEliminated` and per-player `eliminatedRings` counts.
  - **Python:** `GameEngine.apply_move()` contains parallel logic with `_eliminate_top_ring_at()` helper, removing marker and eliminating top ring with identical semantics.
  - Both implementations correctly handle this for:
    - Single capture segments landing on markers.
    - Chain capture intermediate landings on markers (each segment that lands on a marker incurs the 1-ring cost).
    - Chain capture final landings on markers.
- **S‑invariant impact:** Each capture landing on a marker causes:
  - ΔM = 0 (departure marker +1, landing marker removed -1)
  - ΔC = 0
  - ΔE = +1 (top ring of attacker eliminated)
  - **Net ΔS = +1** (progress preserved; documented in [`RULES_TERMINATION_ANALYSIS.md`](RULES_TERMINATION_ANALYSIS.md:1) Section 2.2).
- **Cyclic capture and 180° reversal termination:** Even when marker landings offset height gains (net ΔH = 0 per capture segment), infinite cyclic captures and 180° reversals are **impossible**. Both patterns are bounded by the **same three termination mechanisms**. See [`RULES_TERMINATION_ANALYSIS.md`](RULES_TERMINATION_ANALYSIS.md:1) Section 4.1 for deep analysis showing termination is guaranteed by:
  1. **Finite capturable targets** – each capture consumes one enemy stack; chain length ≤ initial stack count.
  2. **Control change via cap depletion** – marker landings eliminate top rings; eventually attacker loses control.
  3. **Board geometry** – path constraints and marker accumulation bound traversable paths.

  **180° reversals** (back-and-forth captures along a line) deposit markers at each endpoint. These markers cause cap eliminations on subsequent landings, accelerating control change. The same three bounds apply: captures consume targets, marker landings deplete cap rings, and board geometry limits traversable paths.

- **Classification:** `Design‑intent match`.
- **Severity:** `Low` – semantics are correctly implemented and preserve termination guarantees.
- **Scope:** All hosts (backend, sandbox, Python) for captures/chain captures landing on any marker.
- **Recommendation:**
  - Ensure test coverage for multi-segment chain captures where multiple intermediate landings occur on markers (cumulative elimination cost).
  - Document in teaching materials that landing on markers during captures is a strategic trade-off (valid positioning vs ring loss).

### CCE‑011 – Empty region processing (regions with no stacks)

- **RR‑CANON rules:** `R040`, `R142`, `R145` ([`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md:143)).
- **Code / tests:** [`TypeScript.territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:36), [`TypeScript.territoryProcessing.canProcessTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:99), [`TypeScript.territoryProcessing.applyTerritoryRegion()`](src/shared/engine/territoryProcessing.ts:172).
- **Interaction / edge case:** A physically disconnected region contains no ring stacks at all—only empty cells and/or markers. Is such a region eligible for territory processing?
- **Intended behaviour (RR‑CANON):**
  - Per [`RR‑CANON‑R040`](../../RULES_CANONICAL_SPEC.md:143): "Regions may contain empty cells, markers, and stacks."
  - Per [`RR‑CANON‑R142`](../../RULES_CANONICAL_SPEC.md:774): RegionColors is the set of players that control at least one stack in R. If a region contains no stacks, RegionColors = ∅ (empty set).
  - Since the empty set is always a strict subset of any non-empty ActiveColors set, an empty region **automatically satisfies** the color-disconnection criterion.
  - Per [`RR‑CANON‑R145`](../../RULES_CANONICAL_SPEC.md:800): When processing an empty region, the "eliminate internal rings" step eliminates zero rings, but processing remains valid.
  - **Conclusion:** Empty regions are fully eligible for processing, subject to physical disconnection (R141) and the self-elimination prerequisite (R143). Processing an empty region yields territory (collapsed spaces) at the cost of the mandatory self-elimination from outside the region.
- **Observed behaviour:**
  - `RegionColors` computation correctly returns an empty set when no stacks exist in a region.
  - `canProcessTerritoryRegion()` correctly evaluates empty RegionColors as a strict subset of ActiveColors.
  - `applyTerritoryRegion()` handles zero internal stacks gracefully (eliminates zero rings, collapses all interior cells).
  - Self-elimination is still required and correctly enforced.
- **Classification:** `Design‑intent match`.
- **Severity:** `Low` – edge case is rare but semantics are well-defined.
- **Scope:** All hosts (backend, sandbox, Python) where territory processing is implemented.
- **Recommendation:**
  - Ensure test coverage for empty region scenarios (physically disconnected region containing only empty cells and/or markers).
  - Document in teaching materials that empty regions can be claimed as territory at the cost of self-elimination.

## 5. Coverage of High‑Risk Areas from Prior Reports

High‑risk themes identified in [`archive/RULES_STATIC_VERIFICATION.md`](../../archive/RULES_STATIC_VERIFICATION.md:975) and [`archive/RULES_DYNAMIC_VERIFICATION.md`](../../archive/RULES_DYNAMIC_VERIFICATION.md:665) are addressed as follows:

- **Board invariant repair (SCEN‑BOARD‑001):** Covered by `CCE‑001`. Classified as an `Implementation compromise` with Medium severity; recommendation is to instrument and treat repairs as test failures on legal traces.
- **Sandbox turn / placement semantics (SCEN‑TURN‑001/SCEN‑PLACEMENT‑002):** Covered by `CCE‑003`. Classified as `Intentional but under‑documented`, sandbox‑only, with recommendations for explicit parity tests and optional internal `skip_placement`.
- **Per‑player placement cap with many captured rings (SCEN‑PLACEMENT‑004):** Covered by `CCE‑002`. Classified as `Implementation compromise`, Low severity, requiring a canonical decision on whether to adopt or refine the approximation.
- **Shared capture‑chain helpers vs live engines (SCEN‑CAPTURE‑002/003/004):** Covered by `CCE‑004`. Current behaviour is correct at host level; risk is future refactor without regression tests.
- **Multi‑region territory with limited self‑elimination budget (SCEN‑TERRITORY‑002):** Covered by `CCE‑005`. Current sequencing enforces the budget correctly; recommendation is to add explicit tests and, if needed, strengthen helper‑level inside/outside constraints.
- **Victory edge cases (SCEN‑VICTORY‑001/002):** Covered by `CCE‑006`. Ring‑elimination / territory / stalemate ladders are correct; explicit non‑bare‑board last‑player‑standing is currently an implementation compromise requiring a design decision.
- **Forced‑elimination entry and stack‑selection heuristics:** Covered by `CCE‑007`. Eligibility is faithful; stack selection is a deterministic but rules‑permitted heuristic.
- **Cross‑cluster sequencing (movement → lines → territory → victory):** Covered by `CCE‑008`. Current engines match RR‑CANON semantics; this should be preserved as the reference ordering.

Overall, the implementation handles **most cross‑rule interactions and edge cases faithfully**, with the remaining differences falling into three categories:

1. **Defensive measures** (board repair) that should be instrumented rather than relied on.
2. **Conservative approximations** (placement cap) that slightly restrict legal play in rare states.
3. **Representation / documentation gaps** (sandbox skip semantics, unimplemented shared helpers, non‑bare‑board last‑player‑standing) that require either explicit design decisions or better tests and docs.

These CCE entries can now be used by:

- Documentation / UX agents to update public rules and UI explanations.
- Lead Integrator to decide where TS vs Python vs sandbox parity must be tightened.
- Code‑mode implementors to add tests, instrumentation, or refactors with a clear understanding of which behaviours are intentional and which are compromises.
