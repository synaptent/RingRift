# Client Decomposition Plan

Date: 2026-04-10

This document records Phase 6 decomposition targets for the largest client files. It complements `CLIENT_SANDBOX_ENGINE_DECOMPOSITION_PLAN.md`, which focuses only on `ClientSandboxEngine.ts`.

## Current File Sizes

| File                                        | Lines | Role                                                                          |
| ------------------------------------------- | ----: | ----------------------------------------------------------------------------- |
| `src/client/sandbox/ClientSandboxEngine.ts` | 4,821 | Sandbox host over the shared engine; adapter, not canonical rules.            |
| `src/client/components/BoardView.tsx`       | 2,858 | Presentational board renderer, geometry overlays, keyboard/touch interaction. |
| `src/client/sandbox/sandboxAI.ts`           | 2,564 | Sandbox AI turn orchestration and backend AI service bridge.                  |

## ClientSandboxEngine.ts Targets

The five largest remaining clusters should be extracted as adapter modules with explicit hook interfaces. Do not move canonical rules logic here; all extracted modules must continue delegating to `src/shared/engine/**`.

| Target                           | Current methods                                                                                                                                                                                                              | Estimated LOC | Dependencies                                                                                                                      | Proposed module                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------: | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Replay auto-resolution           | `applyCanonicalMoveForReplay`, `autoResolvePendingDecisionPhasesForReplay`, `autoResolveOneTerritoryRegionForReplay`, `autoResolveOneLineForReplay`                                                                          |           874 | `gameState`, `traceMode`, history hooks, turn advancement, line/territory helpers, capture enumeration, victory checks            | `src/client/sandbox/replay/SandboxReplayAdapter.ts`     |
| Territory processing             | `processDisconnectedRegionsForCurrentPlayer`, `canProcessDisconnectedRegion`, `getValidTerritoryProcessingMovesForCurrentPlayer`, `getValidEliminationDecisionMovesForCurrentPlayer`, `applyCanonicalProcessTerritoryRegion` |           485 | `interactionHandler`, `_pendingTerritorySelfElimination`, board territories, decision mapping, history snapshots, victory checks  | `src/client/sandbox/territory/SandboxTerritoryFlow.ts`  |
| Movement and capture interaction | `handleMovementClick`, `performCaptureChainInternal`, `advanceAfterMovement`, `promptForCaptureDirection`, `handleChainCaptureClick`, `applyCaptureSegment`, `performCaptureChain`                                           |           425 | selected stack key, must-move key, capture segment enumeration, marker effects, interaction prompts, turn advancement             | `src/client/sandbox/interaction/SandboxMovementFlow.ts` |
| Line processing and ring rewards | `processLinesForCurrentPlayer`, `getValidLineProcessingMovesForCurrentPlayer`, `collapseLineMarkers`, `eliminateRingForLineReward`, `forceEliminateCap`, `forceEliminateCapSync`                                             |           402 | formed lines, line reward choice prompts, marker collapse, cap elimination, history, victory checks                               | `src/client/sandbox/lines/SandboxLineFlow.ts`           |
| Human setup and turn progression | `handleHumanCellClick`, `tryPlaceRings`, `maybeAutoAdvanceHumanWithNoRings`, `handleStartOfInteractiveTurn`, `startTurnForCurrentPlayer`, forced-elimination helpers, LPS helpers                                            |           689 | placement state, `_placementPositionThisTurn`, LPS state, `_hasPlacedThisTurn`, forced-elimination options, `advanceTurnAndPhase` | `src/client/sandbox/turn/SandboxInteractiveTurnFlow.ts` |

Recommended extraction order: replay first, then territory, then movement/capture. Replay is large but mostly isolated behind `applyCanonicalMoveForReplay`, while territory and movement have more interactive UI coupling and should be extracted only with focused tests.

## BoardView.tsx Targets

`BoardView.tsx` mixes pure presentation, DOM geometry measurement, touch/keyboard event handling, animation, and two large board renderers. The safest decomposition is to extract pure render helpers before moving hooks.

| Target               | Current code                                                                                           | Estimated LOC | Dependencies                                                                                            | Proposed module                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------ | ------------: | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Stack rendering      | `StackWidget`, `StackFromViewModel`, `getPlayerColors`, ring sizing logic                              |           290 | `RingStack`, `StackViewModel`, `BoardType`, player colors, animation class strings                      | `src/client/components/board/StackRenderer.tsx`      |
| Animation layer      | `MoveAnimationLayer` and path interpolation helpers                                                    |           160 | `MoveAnimationData`, `Position`, DOM refs, `requestAnimationFrame`                                      | `src/client/components/board/MoveAnimationLayer.tsx` |
| Board geometry hooks | scale calculation, movement-grid DOM measurement, keyboard neighbor calculation                        |           650 | DOM refs, `computeBoardMovementGrid`, board type/size, viewport dimensions                              | `src/client/components/board/useBoardGeometry.ts`    |
| Overlay rendering    | `renderMovementOverlay`, `renderChainCapturePathOverlay`, coordinate label rendering                   |           330 | movement-grid data, chain-capture path, coordinate notation, DOM cell centers                           | `src/client/components/board/BoardOverlays.tsx`      |
| Cell rendering       | duplicated square and hex cell classification, ARIA labels, touch handlers, decision highlight classes |         1,050 | `CellViewModel`, `BoardDecisionHighlightsViewModel`, stacks, markers, collapsed spaces, touch callbacks | `src/client/components/board/BoardCells.tsx`         |

The biggest duplication is square-cell and hex-cell class/ARIA construction. Extract a pure `buildCellPresentation()` helper before moving JSX. That reduces risk because the host can snapshot-test class names and ARIA labels before splitting render loops.

## sandboxAI.ts Targets

`sandboxAI.ts` has a single dominant function, `maybeRunAITurnSandbox`, at roughly 1,779 lines. It should not be split by AI difficulty first; the more stable seam is by turn phase and service-vs-local strategy.

| Target                             | Current code                                                                                                            | Estimated LOC | Dependencies                                                                             | Proposed module                                       |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------: | ---------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Backend AI service bridge          | `tryRequestSandboxAIMove`, service telemetry parsing, exact move matching                                               |           220 | `fetch`, `Move`, `SandboxAIHooks`, diagnostics tracker, move key helpers                 | `src/client/sandbox/ai/sandboxAIServiceClient.ts`     |
| Movement candidate strategy        | `buildSandboxMovementCandidates`, `selectSandboxMovementMove`, movement branch inside `maybeRunAITurnSandbox`           |           600 | movement candidates, capture segments, must-move state, forced elimination fallback, RNG | `src/client/sandbox/ai/sandboxAIMovementStrategy.ts`  |
| Placement strategy                 | placement branch in `maybeRunAITurnSandbox`, skip-placement checks, hypothetical placement action checks                |           460 | placement enumeration, ring supply, stack action availability, swap-sides policy         | `src/client/sandbox/ai/sandboxAIPlacementStrategy.ts` |
| Decision-phase strategy            | `getLineDecisionMovesForSandboxAI`, `getTerritoryDecisionMovesForSandboxAI`, line/territory/forced-elimination branches |           500 | valid move enumeration, decision move types, elimination fallback, no-action moves       | `src/client/sandbox/ai/sandboxAIDecisionStrategy.ts`  |
| Stall diagnostics and trace buffer | stall counters, trace buffer, no-op protection, ANM fallback                                                            |           250 | hash comparisons, `isANMState`, `evaluateVictory`, browser debug window                  | `src/client/sandbox/ai/sandboxAIDiagnostics.ts`       |

The first implementation step should be `sandboxAIServiceClient.ts` because it is already a contained helper and now carries production telemetry fields. After that, split phase strategies behind a `SandboxAITurnContext` object rather than passing the full hooks object through every helper.

## TypeScript Check Notes

Phase 6 ran `npx tsc --noEmit` on 2026-04-10 after the client audit and after the easy `as any` cleanup. The check passed with no TypeScript errors.

## Easy as-any Cleanup Policy

Search target: `src/client/**` and `src/server/**`.

Preferred fixes:

- Replace `as any` on imported libraries with a local structural interface when only a few properties are read.
- Replace `as any` on JSON-like data with `unknown` plus runtime guards.
- Leave test-only or vendor-interop casts if removing them would require changing public API types.

The Phase 6 cleanup removed low-risk casts in `statePersistence.ts`, `SandboxContext.tsx`, `useSandboxDiagnostics.ts`, `SandboxGameHost.tsx`, `ClientSandboxEngine.ts`, and `metricsMiddleware.ts`. Remaining casts are concentrated in parity CLI private-method access, a replay bridge call in `SandboxGameHost.tsx`, one opaque crash-recovery snapshot field in `GameSession.ts`, and one generic option callback in `ChoiceDialog.tsx`.
