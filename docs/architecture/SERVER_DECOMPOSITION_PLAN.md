# Server Decomposition Plan

Updated: April 10, 2026

## Current State

- `src/server/routes/game.ts`: 2,355 LOC after extracting the three largest handlers.
- `src/server/game/GameEngine.ts`: 2,657 LOC.
- `src/server/game/GameSession.ts`: 2,473 LOC.
- `src/server/services/AIServiceClient.ts`: AI move telemetry fields are already wired through the server client:
  - `model_id`
  - `eval_mode`
  - `simulation_budget`
  - `device`
  - `search_stats_summary`

## Route Layer

### Completed in this pass

- Extracted `POST /games` to `src/server/routes/game/createGameRoute.ts`
- Extracted `POST /games/:gameId/leave` to `src/server/routes/game/leaveGameRoute.ts`
- Extracted `GET /games/user/:userId` to `src/server/routes/game/userGamesRoute.ts`
- Moved participant/spectator authorization helpers to `src/server/routes/game/accessControl.ts`
- Kept `src/server/routes/game.ts` as the registration surface and re-export point

### Remaining route clusters in `src/server/routes/game.ts`

1. Sandbox AI helpers
   - Endpoints:
     - `POST /games/sandbox/evaluate`
     - `POST /games/sandbox/ai/move`
     - `GET /games/sandbox/ai/ladder/health`
   - Shared dependencies:
     - `deserializeGameState`
     - `getAIServiceClient()`
     - sandbox feature-flag gating
   - Recommended extraction:
     - `src/server/routes/game/sandboxRoutes.ts`

2. Invite and lobby join flows
   - Endpoints:
     - `GET /games/invite/:inviteCode`
     - `POST /games/invite/:inviteCode/join`
     - `POST /games/:gameId/join`
   - Shared dependencies:
     - Prisma game lookup/update
     - lobby broadcasts
     - waiting-to-active transition logic
   - Recommended extraction:
     - `src/server/routes/game/joinRoutes.ts`

3. Read-only game inspection APIs
   - Endpoints:
     - `GET /games/:gameId`
     - `GET /games/:gameId/moves`
     - `GET /games/:gameId/history`
     - `GET /games/:gameId/diagnostics/session`
   - Shared dependencies:
     - participant/spectator access control
     - Prisma history queries
     - final-state/result projection
   - Recommended extraction:
     - `src/server/routes/game/readRoutes.ts`

4. HTTP move harness
   - Endpoint:
     - `POST /games/:gameId/moves`
   - Shared dependencies:
     - `MoveSchema`
     - `wsServerInstance.handlePlayerMoveFromHttp`
     - error-code normalization and timeout protection
   - Recommended extraction:
     - `src/server/routes/game/httpMoveHarnessRoute.ts`

5. Lobby and matchmaking queries
   - Endpoints:
     - `GET /games/lobby/available`
     - `GET /games/matchmaking/stats`
   - Shared dependencies:
     - Prisma lobby/matchmaking reads
   - Recommended extraction:
     - `src/server/routes/game/lobbyRoutes.ts`

## `GameEngine.ts`

### Public surface that should remain stable

- `swapSidesApplied`
- `getLpsTrackingSummary()`
- `setDebugCheckpointHook()`
- `enableMoveDrivenDecisionPhases()`
- `enableOrchestratorAdapter()`
- `disableOrchestratorAdapter()`
- `isOrchestratorAdapterEnabled()`
- `getInternalStateForPersistence()`
- `restoreInternalStateFromSnapshot()`
- `resignPlayer()`
- `abandonPlayer()`
- `abandonGameAsDraw()`
- `makeMoveById()`
- `resolveBlockedStateForCurrentPlayerForTesting()`
- `stepAutomaticPhasesForTesting()`

### Largest method clusters

1. Adapter and orchestration bridge
   - Methods:
     - `applySwapSidesMove`
     - `createAdapterForCurrentGame`
     - `processMoveViaAdapter`
     - `getInternalStateForPersistence`
     - `restoreInternalStateFromSnapshot`
   - Approx. footprint: lines 311-1134
   - Shared state:
     - `gameState`
     - `interactionManager`
     - replay/orchestrator flags
     - internal per-turn state flags
   - Recommended extraction:
     - `server/game/engine/adapterBridge.ts`

2. Phase-specific move generation and elimination bookkeeping
   - Methods:
     - `getCaptureOptionsFromPosition`
     - `getValidLineProcessingMoves`
     - `getValidTerritoryProcessingMoves`
     - `eliminatePlayerRingOrCap`
     - `eliminateFromStack`
     - `updatePlayerEliminatedRings`
   - Approx. footprint: lines 1479-1666
   - Shared state:
     - `gameState.board`
     - `boardManager`
     - `ruleEngine`
   - Recommended extraction:
     - `server/game/engine/phaseResolvers.ts`

3. Turn advancement, timer handoff, and game completion
   - Methods:
     - `advanceGame`
     - `startPlayerTimer`
     - `transitionPlayerTimer`
     - `endGame`
     - `updatePlayerRatings`
     - `resignPlayer`
     - `abandonPlayer`
     - `abandonGameAsDraw`
   - Approx. footprint: lines 1667-1945
   - Shared state:
     - `clockManager`
     - `gameState.players`
     - rating/endgame helpers
   - Recommended extraction:
     - `server/game/engine/gameLifecycle.ts`

4. Move application and legality host
   - Methods:
     - `makeMoveById`
     - `hasAnyRealActionForPlayer`
   - Approx. footprint: lines 2313-2414
   - Shared state:
     - `gameState`
     - move history
     - orchestrator/aggregate helpers
   - Recommended extraction:
     - `server/game/engine/moveExecution.ts`

5. Test-only progression helpers
   - Methods:
     - `resolveBlockedStateForCurrentPlayerForTesting`
     - `stepAutomaticPhasesForTesting`
   - Approx. footprint: lines 2415-2634
   - Shared state:
     - `gameState`
     - interaction manager
     - auto-phase resolution
   - Recommended extraction:
     - `server/game/engine/testingHarness.ts`

## `GameSession.ts`

### Largest method clusters

1. Session bootstrap and replay reconstruction
   - Methods:
     - `initialize`
     - `createPlayer`
     - `configureEngineSelection`
     - `replayMove`
   - Approx. footprint: lines 221-709
   - Shared state:
     - `gameEngine`
     - `rulesFacade`
     - `interactionManager`
     - persisted fixture/internal-state metadata
   - Recommended extraction:
     - `server/game/session/sessionBootstrap.ts`

2. Player move intake and persistence
   - Methods:
     - `handlePlayerMove`
     - `handlePlayerMoveFromHttp`
     - `handlePlayerMoveForUser`
     - `handlePlayerMoveById`
     - `handlePlayerResignationByUserId`
     - `handleAbandonmentForDisconnectedPlayer`
     - `persistMove`
     - `finishGameWithResult`
     - `handleTimeoutResult`
     - `broadcastUpdate`
   - Approx. footprint: lines 860-1351
   - Shared state:
     - `gameEngine`
     - `rulesFacade`
     - socket IO room broadcast
     - persistence services
   - Recommended extraction:
     - `server/game/session/playerActions.ts`

3. Position evaluation and AI-quality diagnostics
   - Methods:
     - `computeAIQualityMode`
     - `mapGameResultToOutcome`
     - `computeFinalScore`
     - `isAnalysisModeEnabled`
     - `evaluateAndBroadcastPosition`
     - `createLocalAIRng`
   - Approx. footprint: lines 797-859 and 1352-1405
   - Shared state:
     - diagnostics snapshot
     - evaluation broadcast plumbing
   - Recommended extraction:
     - `server/game/session/diagnostics.ts`

4. AI turn execution and fallback handling
   - Methods:
     - `maybePerformAITurn`
     - `getAIMoveWithTimeout`
     - `handleServiceMoveRejected`
     - `handleNoMoveFromService`
     - `handleAIFatalFailure`
     - `persistAIMove`
     - `startAIWatchdog`
     - `checkAIWatchdog`
   - Approx. footprint: lines 1406-1955 and 2404-2421
   - Shared state:
     - `aiRequestState`
     - `diagnosticsSnapshot`
     - AI cancellation/watchdog timers
   - Recommended extraction:
     - `server/game/session/aiTurnManager.ts`

5. Decision timeout subsystem
   - Methods:
     - `mapPhaseToTimeoutPhase`
     - `scheduleDecisionPhaseTimeout`
     - `classifyDecisionSurface`
     - `emitDecisionPhaseTimeoutWarning`
     - `handleDecisionPhaseTimedOut`
     - `handleDecisionPhaseTimedOutLocked`
   - Approx. footprint: lines 1956-2403
   - Shared state:
     - decision timeout deadline/handles
     - session cancellation
     - active decision metadata
   - Recommended extraction:
     - `server/game/session/decisionTimeouts.ts`

## Recommended Next Order

1. Extract `readRoutes.ts` from `src/server/routes/game.ts`
2. Extract `httpMoveHarnessRoute.ts`
3. Split `GameSession.ts` at the AI turn manager boundary first
4. Split `GameEngine.ts` at the lifecycle and phase-resolver boundaries
