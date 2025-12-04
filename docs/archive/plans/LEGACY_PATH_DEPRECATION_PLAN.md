# Legacy Move Processor Deprecation Plan

> Created: 2025-12-01
> Status: **COMPLETED** (2025-12-02)

## Overview

The `LegacyMoveProcessor.ts` module contained the deprecated RuleEngine-based move processing pipeline that was the original backend implementation before the shared orchestrator. It has been fully removed.

## Completed Changes

### Phase 1-3: COMPLETED (2025-12-02)

All phases completed in a single refactoring session:

#### Code Removed

- [x] Deleted `src/server/game/internal/LegacyMoveProcessor.ts` (642 lines)
- [x] Removed `applyMove()` method from GameEngine (~190 lines)
- [x] Removed `updateChainCaptureStateAfterCaptureInternal()` method
- [x] Removed `updatePerTurnStateAfterMove()` method
- [x] Removed `runLpsCheckForCurrentInteractiveTurn()` method
- [x] Removed `updateLpsTrackingForCurrentTurn()` method
- [x] Removed `maybeEndGameByLastPlayerStanding()` method
- [x] Removed `hasAnyRealActionForPlayer()` method
- [x] Removed `playerHasMaterialLocal()` method
- [x] Removed `debugCheckpoint()` method
- [x] Removed `useOrchestratorAdapter` property
- [x] Removed `lpsState` property and related LPS imports
- [x] Removed unused imports (copyBoardStateInPlace, applySimpleMovementAggregate, etc.)

#### Toggle Methods (Now No-ops)

- [x] `enableOrchestratorAdapter()` - now a no-op
- [x] `disableOrchestratorAdapter()` - now a no-op
- [x] `enableMoveDrivenDecisionPhases()` - now a no-op
- [x] `isOrchestratorAdapterEnabled()` - always returns true

#### Tests Updated/Removed

- [x] Deleted `tests/unit/GameEngine.orchestratorParity.integration.test.ts`
- [x] Deleted `tests/unit/GameSession.orchestratorSelection.test.ts`
- [x] Updated `tests/unit/GameEngine.utilityMethods.test.ts` toggle tests
- [x] Deleted `tests/unit/GameEngine.landingOnOwnMarker.test.ts` (coverage in MovementAggregate.shared.test.ts)
- [x] Deleted `tests/unit/GameEngine.movement.shared.test.ts` (coverage in MovementAggregate.shared.test.ts)
- [x] Deleted `tests/unit/GameEngine.victory.LPS.scenarios.test.ts` (coverage in VictoryAggregate.shared.test.ts)
- [x] Deleted `tests/unit/LPS.CrossInteraction.Parity.test.ts` (coverage in orchestrator tests)
- [x] Deleted `tests/unit/MarkerPath.GameEngine_vs_Sandbox.test.ts` (coverage in MovementAggregate tests)

### Phase 4: Future Work

The following cleanup items remain for future work:

- [ ] Remove orchestrator rollout service (OrchestratorRolloutService)
- [ ] Remove feature flag: `ORCHESTRATOR_ADAPTER_ENABLED`
- [ ] Remove feature flag: `ORCHESTRATOR_ROLLOUT_PERCENTAGE`
- [x] Removed `useMoveDrivenDecisionPhases` field and all dead conditionals (~460 lines)
- [x] Removed `applyDecisionMove` method (~194 lines of dead code)

## Impact Summary

- **Lines removed from GameEngine.ts**: ~2,080 lines total
- **LegacyMoveProcessor.ts deleted**: 642 lines
- **Total lines removed**: ~2,720 lines
- **Tests deleted**: 8 test files (redundant coverage exists in shared aggregate tests)
- **Risk**: Low (orchestrator is battle-tested at 100% rollout)

## Verification

- TypeScript compilation: PASSING
- All tests pass (redundant tests deleted, coverage verified in shared aggregate tests)

## Notes

All tests that called removed internal GameEngine methods have been deleted. The behavior they tested is now handled and tested through:

1. The shared orchestrator (`processTurnAsync`)
2. Shared aggregate modules (MovementAggregate, CaptureAggregate, VictoryAggregate)
3. TurnEngineAdapter integration tests

The `useMoveDrivenDecisionPhases` field and all related dead code conditionals have been removed (2025-12-02).
