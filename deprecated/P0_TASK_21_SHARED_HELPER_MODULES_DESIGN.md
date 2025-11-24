# P0 Task 21 – Shared Helper Modules Design

This document summarizes the shared helper modules added under `src/shared/engine/*` for Subtask #6. All helpers are currently **design stubs**: they are fully typed and documented but not yet wired into backend or sandbox hosts, so runtime behaviour is unchanged.

## 1. Goals and non‑goals

**Goals**

- Provide canonical, shared APIs for:
  - Movement & capture application (mutators).
  - Capture‑chain enumeration and continuation checks.
  - Line processing and line‑reward decisions.
  - Territory processing and self‑elimination decisions.
  - Placement application and skip‑placement decisions.
  - Turn progression delegates for [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135).
- Align types and semantics with [`types.ts`](src/shared/engine/types.ts:1) and [`game.ts`](src/shared/types/game.ts:1).
- Make later refactors (P0 tasks #7–#9) mechanical by mapping each helper to concrete backend and sandbox hotspots from [`P0_TASK_20_SHARED_RULE_LOGIC_DUPLICATION_AUDIT.md`](P0_TASK_20_SHARED_RULE_LOGIC_DUPLICATION_AUDIT.md:1).

**Non‑goals**

- Do not change behaviour of existing hosts in this task.
- Do not replace existing geometry, validation, or low‑level mutators already centralised under `src/shared/engine/*`.

## 2. New helper modules (high level)

- [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1)
  - Canonical application of non‑capturing movement and single capture segments on `GameState`.
- [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1)
  - Shared enumeration of legal chain‑capture segments and a small continuation info helper.
- [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1)
  - Enumeration and application of `process_line` / `choose_line_reward` moves.
- [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1)
  - Enumeration and application of `process_territory_region` / `eliminate_rings_from_stack` moves.
- [`placementHelpers.ts`](src/shared/engine/placementHelpers.ts:1)
  - Canonical placement application and skip‑placement eligibility evaluation.
- [`turnDelegateHelpers.ts`](src/shared/engine/turnDelegateHelpers.ts:1)
  - Shared "has any placement/movement/capture" predicates and a factory for `TurnLogicDelegates`.

Existing shared modules they build on:

- Geometry, reachability & markers: [`core.ts`](src/shared/engine/core.ts:1), [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1), [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1).
- Lines & territory geometry / core processing: [`lineDetection.ts`](src/shared/engine/lineDetection.ts:1), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1).
- Validators & mutators: `src/shared/engine/validators/*`, `src/shared/engine/mutators/*`.

## 3. Movement & capture application

Module: [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1)

**Functions**

- [`applySimpleMovement(state, params)`](src/shared/engine/movementApplication.ts:55)
  - Params: [`SimpleMovementParams`](src/shared/engine/movementApplication.ts:17) – `{ from, to, player, leaveDepartureMarker? }`.
  - Returns: [`MovementApplicationOutcome`](src/shared/engine/movementApplication.ts:40) – `{ nextState, eliminatedRingsByPlayer? }`.
- [`applyCaptureSegment(state, params)`](src/shared/engine/movementApplication.ts:86)
  - Params: [`CaptureSegmentParams`](src/shared/engine/movementApplication.ts:31) – `{ from, target, landing, player }`.
  - Returns: `MovementApplicationOutcome`.

**Key semantics / invariants (docstring‑level, not yet implemented)**

- Input moves are already validated; helpers do not re‑validate legality.
- Both helpers treat `state` as immutable and return a shallow‑cloned `nextState` with cloned board maps and player array.
- `applySimpleMovement`:
  - Leaves a departure marker at `from` by default.
  - Uses [`applyMarkerEffectsAlongPathOnBoard`](src/shared/engine/core.ts:619) for intermediate markers.
  - Moves or merges the stack to `to`.
  - If landing on an own marker, removes the marker, eliminates the bottom ring of the resulting stack, and updates elimination counts on board and players.
- `applyCaptureSegment`:
  - Same marker semantics along `from → target → landing`.
  - Implements overtaking: attacker moves to `landing`, pops top ring from `target` and appends it to the attacking stack’s bottom, updates both stacks or deletes empty ones.
  - If landing on an own marker, performs the same bottom‑ring elimination as simple movement.

**Duplication hotspots to be replaced later**

- Backend:
  - Movement and overtaking branches in [`GameEngine.applyMove`](src/server/game/GameEngine.ts:1) and `performOvertakingCapture`.
- Sandbox:
  - Non‑capture movement in [`sandboxMovementEngine.handleMovementClickSandbox`](src/client/sandbox/sandboxMovementEngine.ts:1).
  - Capture application in [`sandboxCaptures.applyCaptureSegmentOnBoard`](src/client/sandbox/sandboxCaptures.ts:1) and `ClientSandboxEngine.applyCaptureSegment`.

## 4. Capture chains orchestration primitives

Module: [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1)

**Functions**

- [`enumerateChainCaptureSegments(state, snapshot, options?)`](src/shared/engine/captureChainHelpers.ts:73)
  - Snapshot: [`ChainCaptureStateSnapshot`](src/shared/engine/captureChainHelpers.ts:25) – `{ player, currentPosition, visitedTargets? }`.
  - Options: [`ChainCaptureEnumerationOptions`](src/shared/engine/captureChainHelpers.ts:49) – `{ disallowRevisitedTargets?, moveNumber?, kind? }`.
  - Returns: `Move[]` with type `'overtaking_capture'` (initial) or `'continue_capture_segment'` (continuations).
- [`getChainCaptureContinuationInfo(state, snapshot, options?)`](src/shared/engine/captureChainHelpers.ts:109)
  - Returns: [`ChainCaptureContinuationInfo`](src/shared/engine/captureChainHelpers.ts:93) – `{ hasFurtherCaptures, segments }`.

**Key semantics / invariants**

- Delegates geometry to shared [`enumerateCaptureMoves`](src/shared/engine/captureLogic.ts:26).
- Optional `disallowRevisitedTargets` enforces no‑repeat‑target semantics based on `visitedTargets` (stringified positions).
- Intended to be called after each applied capture segment to decide whether the chain must continue (`hasFurtherCaptures === true`) or can terminate.

**Duplication hotspots to be replaced later**

- Backend:
  - [`rules/captureChainEngine.getCaptureOptionsFromPosition`](src/server/game/rules/captureChainEngine.ts:1).
  - `GameEngine` chain‑capture state updates after `performOvertakingCapture`.
- Sandbox:
  - [`sandboxMovementEngine.performCaptureChainSandbox`](src/client/sandbox/sandboxMovementEngine.ts:1).
  - `ClientSandboxEngine.performCaptureChain` and associated capture‑choice logic.

## 5. Line processing & line reward decisions

Module: [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1)

**Functions**

- [`enumerateProcessLineMoves(state, player, options?)`](src/shared/engine/lineDecisionHelpers.ts:73)
- [`enumerateChooseLineRewardMoves(state, player, lineIndex)`](src/shared/engine/lineDecisionHelpers.ts:100)
- [`applyProcessLineDecision(state, move)`](src/shared/engine/lineDecisionHelpers.ts:141)
- [`applyChooseLineRewardDecision(state, move)`](src/shared/engine/lineDecisionHelpers.ts:169)

**Key semantics / invariants**

- Enumeration helpers operate over either:
  - `state.board.formedLines`, or
  - freshly detected lines via [`findAllLines`](src/shared/engine/lineDetection.ts:21), depending on `LineEnumerationOptions.detectionMode`.
- `enumerateProcessLineMoves`:
  - One `process_line` move per formed line owned by `player`.
  - Each move’s `formedLines[0]` identifies the line; `to` is a representative cell.
- `enumerateChooseLineRewardMoves`:
  - For exact‑length lines: collapse‑all is effectively forced; helpers may yield a single `CHOOSE_LINE_REWARD` or no extra move depending on host integration.
  - For longer lines: yields a `COLLAPSE_ALL` move plus one or more `MINIMUM_COLLAPSE` moves with contiguous subsets of length `L`.
- Application helpers (`applyProcessLineDecision` and `applyChooseLineRewardDecision`):
  - Collapse markers into territory for the acting player.
  - Return rings from stacks on collapsed spaces to owners’ hands (no elimination).
  - Remove processed lines and any other lines broken by the collapse.
  - Indicate via `pendingLineRewardElimination: boolean` whether a ring‑elimination reward must follow (Option 1 on long lines).

**Duplication hotspots to be replaced later**

- Backend:
  - Line enumeration in `GameEngine.getValidLineProcessingMoves` and [`RuleEngine.getValidLineProcessingDecisionMoves`](src/server/game/RuleEngine.ts:1).
  - Line effects in [`rules/lineProcessing.processLinesForCurrentPlayer`](src/server/game/rules/lineProcessing.ts:1) and line‑related branches of `GameEngine.applyDecisionMove`.
- Sandbox:
  - [`sandboxLinesEngine.getValidLineProcessingMoves`](src/client/sandbox/sandboxLinesEngine.ts:1) and `applyLineDecisionMove`.
  - `ClientSandboxEngine.processLinesForCurrentPlayer` and line‑decision handling in `applyCanonicalMoveInternal`.

## 6. Territory processing & self‑elimination

Module: [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1)

**Functions**

- [`enumerateProcessTerritoryRegionMoves(state, player, options?)`](src/shared/engine/territoryDecisionHelpers.ts:76)
- [`applyProcessTerritoryRegionDecision(state, move)`](src/shared/engine/territoryDecisionHelpers.ts:117)
  - Returns [`TerritoryProcessApplicationOutcome`](src/shared/engine/territoryDecisionHelpers.ts:90) – `{ nextState, processedRegionId, processedRegion, pendingSelfElimination }`.
- [`enumerateTerritoryEliminationMoves(state, player, scope?)`](src/shared/engine/territoryDecisionHelpers.ts:175)
  - Scope: [`TerritoryEliminationScope`](src/shared/engine/territoryDecisionHelpers.ts:157) – `{ processedRegionId? }`.
- [`applyEliminateRingsFromStackDecision(state, move)`](src/shared/engine/territoryDecisionHelpers.ts:206)
  - Returns [`EliminateRingsFromStackOutcome`](src/shared/engine/territoryDecisionHelpers.ts:194) – `{ nextState }`.

**Key semantics / invariants**

- `enumerateProcessTerritoryRegionMoves`:
  - Surfaces one `process_territory_region` move per region that passes the shared self‑elimination prerequisite [`canProcessTerritoryRegion`](src/shared/engine/territoryProcessing.ts:99).
  - Regions can be taken either from `board.territories` (`use_board_cache`) or recomputed via [`getProcessableTerritoryRegions`](src/shared/engine/territoryProcessing.ts:146).
- `applyProcessTerritoryRegionDecision`:
  - Delegates geometry and internal eliminations to [`applyTerritoryRegion`](src/shared/engine/territoryProcessing.ts:172).
  - Projects board‑level deltas back into players’ `territorySpaces` and `eliminatedRings`, and `totalRingsEliminated`.
  - Marks `pendingSelfElimination: true` so hosts can surface/require a follow‑up `eliminate_rings_from_stack` decision.
- `enumerateTerritoryEliminationMoves`:
  - Produces one `eliminate_rings_from_stack` decision per eligible stack controlled by the acting player, with `eliminationFromStack` populated for diagnostics.
  - Eligibility rules (inside vs outside processed region) are parameterised by `scope.processedRegionId` and called out as an open question.
- `applyEliminateRingsFromStackDecision`:
  - Encodes the cap‑removal semantics used by forced/self‑elimination in backend and sandbox:
    - removes the consecutive top rings of the controlling colour,
    - updates `board.eliminatedRings`, `players[n].eliminatedRings`, and `totalRingsEliminated`,
    - removes the stack when emptied.

**Duplication hotspots to be replaced later**

- Backend:
  - Territory pipeline in [`rules/territoryProcessing.processDisconnectedRegionsForCurrentPlayer`](src/server/game/rules/territoryProcessing.ts:1).
  - `GameEngine.applyDecisionMove` branches for `process_territory_region` and `eliminate_rings_from_stack`.
- Sandbox:
  - [`sandboxTerritoryEngine.processDisconnectedRegionsForCurrentPlayerEngine`](src/client/sandbox/sandboxTerritoryEngine.ts:1) and `applyTerritoryDecisionMove`.
  - Territory‑decision handling in `ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer` and `applyCanonicalMoveInternal`.

## 7. Placement application & skip‑placement decisions

Module: [`placementHelpers.ts`](src/shared/engine/placementHelpers.ts:1)

**Functions**

- [`applyPlacementMove(state, move)`](src/shared/engine/placementHelpers.ts:70)
  - For `move.type === 'place_ring'` only.
  - Returns [`PlacementApplicationOutcome`](src/shared/engine/placementHelpers.ts:61) – `{ nextState, placementCount, placedOnStack }`.
- [`evaluateSkipPlacementEligibility(state, player)`](src/shared/engine/placementHelpers.ts:132)
  - Returns [`SkipPlacementEligibilityResult`](src/shared/engine/placementHelpers.ts:121) – `{ canSkip, reason?, code? }`.

**Key semantics / invariants**

- `applyPlacementMove`:
  - Interprets `move.to` and `move.placementCount ?? 1`.
  - Reconstructs a `PlacementContext` from `state` and delegates legality/no‑dead‑placement checks to [`validatePlacementOnBoard`](src/shared/engine/validators/PlacementValidator.ts:76).
  - Delegates board mutation to [`applyPlacementOnBoard`](src/shared/engine/mutators/PlacementMutator.ts:16).
  - Decrements `ringsInHand` for the acting player; leaves `totalRingsInPlay` semantics aligned with the existing shared mutator.
- `evaluateSkipPlacementEligibility` (design assumption):
  - Legal only in `ring_placement` for the active player.
  - Requires `ringsInHand > 0` and at least one controlled stack with a legal move or capture (via [`hasAnyLegalMoveOrCaptureFromOnBoard`](src/shared/engine/core.ts:367)).
  - Additionally assumes **no legal placements remain** that satisfy `validatePlacementOnBoard`; this ties skip‑eligibility to dead‑placement semantics and is called out as an open question.

**Duplication hotspots to be replaced later**

- Backend:
  - Placement branch in [`GameEngine.applyMove`](src/server/game/GameEngine.ts:1).
  - `RuleEngine.validateSkipPlacement` and placement enumeration logic that duplicates shared `PlacementValidator`.
- Sandbox:
  - [`sandboxPlacement.enumerateLegalRingPlacements`](src/client/sandbox/sandboxPlacement.ts:1) legacy path and `ClientSandboxEngine.tryPlaceRings`.
  - Skip‑placement gating in `sandboxTurnEngine` / `ClientSandboxEngine`.

## 8. Turn progression delegates

Module: [`turnDelegateHelpers.ts`](src/shared/engine/turnDelegateHelpers.ts:1)

**Functions**

- [`hasAnyPlacementForPlayer(state, player)`](src/shared/engine/turnDelegateHelpers.ts:54)
- [`hasAnyMovementForPlayer(state, player, turn)`](src/shared/engine/turnDelegateHelpers.ts:84)
- [`hasAnyCaptureForPlayer(state, player, turn)`](src/shared/engine/turnDelegateHelpers.ts:112)
- [`createDefaultTurnLogicDelegates(config)`](src/shared/engine/turnDelegateHelpers.ts:172)
  - Config: [`DefaultTurnDelegatesConfig`](src/shared/engine/turnDelegateHelpers.ts:134) – `{ getNextPlayerNumber, applyForcedElimination }`.

**Key semantics / invariants**

- Predicates are intended to be the single source of truth for the questions used by [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135):
  - any legal placement?
  - any legal non‑capturing movement?
  - any legal overtaking capture?
- They are expected to:
  - Use `MovementBoardView` + [`hasAnyLegalMoveOrCaptureFromOnBoard`](src/shared/engine/core.ts:367) and `enumerateSimpleMoveTargetsFromStack` / `enumerateCaptureMoves` internally.
  - Respect per‑turn constraints from [`PerTurnState`](src/shared/engine/turnLogic.ts:30) (e.g. must‑move‑from‑stack).
- [`createDefaultTurnLogicDelegates`](src/shared/engine/turnDelegateHelpers.ts:172):
  - Wires these predicates into a `TurnLogicDelegates` instance while delegating:
    - `getNextPlayerNumber` and
    - `applyForcedElimination`
      to host‑supplied implementations.

**Duplication hotspots to be replaced later**

- Backend:
  - `GameEngine.hasValidPlacements`, `hasValidMovements`, `hasValidCaptures` and the forced‑elimination helpers inside `TurnEngine.advanceGameForCurrentPlayer`.
- Sandbox:
  - `sandboxTurnEngine.hasAnyPlacementForCurrentPlayer`, `hasAnyMovementForCurrentPlayer`, `hasAnyCaptureForCurrentPlayer`, and `maybeProcessForcedEliminationForCurrentPlayer`.

## 9. Adoption plan (P0 tasks #7–#9)

High‑level ordering to minimise risk and duplication while migrating hosts:

1. **Movement & capture application + capture chains**
   - Backend:
     - Refactor `GameEngine.applyMove` non‑capture branches to call [`applySimpleMovement`](src/shared/engine/movementApplication.ts:55).
     - Refactor capture branches and `performOvertakingCapture` to call [`applyCaptureSegment`](src/shared/engine/movementApplication.ts:86).
     - Update [`rules/captureChainEngine`](src/server/game/rules/captureChainEngine.ts:1) to call [`enumerateChainCaptureSegments`](src/shared/engine/captureChainHelpers.ts:73) and [`getChainCaptureContinuationInfo`](src/shared/engine/captureChainHelpers.ts:109).
   - Sandbox:
     - Port `sandboxMovementEngine.handleMovementClickSandbox` and `performCaptureChainSandbox` to the same helpers.

2. **Line decisions**
   - Backend:
     - Replace `GameEngine.getValidLineProcessingMoves` and `RuleEngine.getValidLineProcessingDecisionMoves` with [`enumerateProcessLineMoves`](src/shared/engine/lineDecisionHelpers.ts:73) and [`enumerateChooseLineRewardMoves`](src/shared/engine/lineDecisionHelpers.ts:100).
     - Replace [`rules/lineProcessing.processLinesForCurrentPlayer`](src/server/game/rules/lineProcessing.ts:1) and line branches of `GameEngine.applyDecisionMove` with [`applyProcessLineDecision`](src/shared/engine/lineDecisionHelpers.ts:141) and [`applyChooseLineRewardDecision`](src/shared/engine/lineDecisionHelpers.ts:169).
   - Sandbox:
     - Port `sandboxLinesEngine.getValidLineProcessingMoves` / `applyLineDecisionMove` and `ClientSandboxEngine.processLinesForCurrentPlayer` to the same helpers.

3. **Territory pipeline & self‑elimination**
   - Backend:
     - Replace [`rules/territoryProcessing.processDisconnectedRegionsForCurrentPlayer`](src/server/game/rules/territoryProcessing.ts:1) and related helpers with [`enumerateProcessTerritoryRegionMoves`](src/shared/engine/territoryDecisionHelpers.ts:76) and [`applyProcessTerritoryRegionDecision`](src/shared/engine/territoryDecisionHelpers.ts:117).
     - Replace elimination decisions in `GameEngine.applyDecisionMove` with [`enumerateTerritoryEliminationMoves`](src/shared/engine/territoryDecisionHelpers.ts:175) and [`applyEliminateRingsFromStackDecision`](src/shared/engine/territoryDecisionHelpers.ts:206).
   - Sandbox:
     - Port `sandboxTerritoryEngine.processDisconnectedRegionsForCurrentPlayerEngine` / `applyTerritoryDecisionMove` and related `ClientSandboxEngine` branches to the shared helpers.

4. **Placement & skip‑placement**
   - Backend:
     - Replace placement mutation in `GameEngine.applyMove` with [`applyPlacementMove`](src/shared/engine/placementHelpers.ts:192).
     - Align `RuleEngine.validateSkipPlacement` with [`evaluateSkipPlacementEligibility`](src/shared/engine/placementHelpers.ts:132) semantics.
   - Sandbox:
     - Replace legacy path in [`sandboxPlacement.enumerateLegalRingPlacements`](src/client/sandbox/sandboxPlacement.ts:1) and `ClientSandboxEngine.tryPlaceRings` with calls into shared placement validator + [`applyPlacementMove`](src/shared/engine/placementHelpers.ts:192).

5. **Turn progression delegates**
   - Backend:
     - Refactor `TurnEngine.advanceGameForCurrentPlayer` to use [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135) with delegates created by [`createDefaultTurnLogicDelegates`](src/shared/engine/turnDelegateHelpers.ts:172).
   - Sandbox:
     - Refactor `sandboxTurnEngine` similarly, sharing the same `hasAnyPlacement` / `hasAnyMovement` / `hasAnyCapture` semantics.

Each step should keep behaviour parity validated via existing `RulesMatrix.*` and parity tests.

## 10. Open questions and assumptions

**NOTE 1 – Skip‑placement semantics**

- Assumption in [`evaluateSkipPlacementEligibility`](src/shared/engine/placementHelpers.ts:132): skip is only legal when **no** legal placements remain that satisfy `validatePlacementOnBoard`.
- Existing backend and sandbox logic occasionally allow skip even when placements exist, as long as movement/capture is available.
- Maintainers should confirm desired semantics before wiring hosts to this helper; see:
  - [`RuleEngine.validateSkipPlacement`](src/server/game/RuleEngine.ts:1)
  - Sandbox skip logic in `sandboxTurnEngine` / `ClientSandboxEngine`.

**NOTE 2 – Territory self‑elimination scope (inside vs outside processed region)**

- [`TerritoryEliminationScope.processedRegionId`](src/shared/engine/territoryDecisionHelpers.ts:157) is designed to support "must eliminate from outside the processed region" semantics.
- Current backend/sandbox code is not fully consistent on whether eliminating from within the processed region is allowed.
- Maintainers should clarify intended rule (FAQ Q23 / §12.2) and update implementations of:
  - [`enumerateTerritoryEliminationMoves`](src/shared/engine/territoryDecisionHelpers.ts:175), and
  - host‑level callers in `GameEngine.applyDecisionMove` and `sandboxTerritoryEngine`.

**NOTE 3 – Chain‑capture visited‑target rules**

- [`ChainCaptureStateSnapshot.visitedTargets`](src/shared/engine/captureChainHelpers.ts:25) and `disallowRevisitedTargets` encode a conservative rule: do not capture the same target position twice in a chain.
- Existing backend `captureChainEngine` and sandbox `performCaptureChainSandbox` appear to disallow immediate backtracking but details on longer cycles are implicit.
- When wiring hosts to [`enumerateChainCaptureSegments`](src/shared/engine/captureChainHelpers.ts:73), confirm whether any historical fixtures rely on more permissive behaviour.

**NOTE 4 – Line‑reward elimination application surface**

- This design assumes ring‑elimination rewards from long lines will be expressed as explicit `eliminate_rings_from_stack` moves using the territory‑elimination helpers.
- Some existing code paths (especially older backend line‑processing flows) apply the elimination directly inside line logic without a separate move.
- When migrating to [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1) + [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1), clarify whether all such eliminations should become explicit moves for parity and AI training, or remain implicit in some modes.
