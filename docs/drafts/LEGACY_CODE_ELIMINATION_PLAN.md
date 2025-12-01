# Legacy Code Elimination Plan

> **⚠️ HISTORICAL / COMPLETED** – This plan was completed in November 2025. The orchestrator is now at 100% rollout.
> For current status, see:
> - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – Production rollout status
> - `CURRENT_STATE_ASSESSMENT.md` – Implementation status
> - `TODO.md` – Wave 5.4 (legacy cleanup) for remaining items

> **Phase 5.4 Deliverable** - Architecture Remediation Plan
> **Status:** ✅ Phase 6.1 Complete - Adapters Enabled by Default
> **Last Updated:** 2025-11-30
> **Phase 6.1 Complete:** 2025-11-26

## Phase 6.1 Summary

The parity bugs blocking orchestrator adapter enablement have been fixed:

1. ✅ **Race condition in MovementAggregate.ts** - Fixed marker read before deletion
2. ✅ **Top ring elimination** - Corrected from bottom ring to top ring

### Changes Made in Phase 6.1

**Client (`ClientSandboxEngine.ts`):**

- `useOrchestratorAdapter` now defaults to `true`

**Backend (`env.ts`):**

- `ORCHESTRATOR_ADAPTER_ENABLED` environment variable now defaults to `true`
- Transform logic inverted: `val !== 'false' && val !== '0'` (opt-out instead of opt-in)

### Rollback Procedure

To disable orchestrator adapters if issues arise:

**Client:** Call `engine.disableOrchestratorAdapter()` or modify the default in `ClientSandboxEngine.ts`

**Backend:** Set environment variable `ORCHESTRATOR_ADAPTER_ENABLED=false`

## Historical Context (Phase 6 Attempt)

Phase 6 attempted to eliminate Tier 1 legacy code (~1,713 lines). The following blockers were discovered and subsequently fixed:

### Parity Issues (Now Resolved)

#### Issue 1: "Landing on Own Marker" Rule

When the orchestrator adapter was enabled, the "landing on own marker eliminates top ring" rule did not trigger correctly.

**Root Cause:** Race condition in `MovementAggregate.ts` where the marker was read after being deleted.

**Fix:** Capture marker owner before removal in `applyMovementStep()`.

#### Issue 2: Ring Elimination Direction

The elimination was incorrectly removing the bottom ring instead of the top ring.

**Fix:** Corrected the slice operation in `MovementAggregate.ts`.

---

## Overview

This document identifies sandbox code that can be eliminated after full migration to the canonical orchestrator (`src/shared/engine/orchestration/`). The orchestrator adapters (`TurnEngineAdapter.ts` and `SandboxOrchestratorAdapter.ts`) now provide platform-specific wrappers around the shared engine, making the legacy per-platform rule implementations redundant.

## Migration Prerequisites

Before eliminating legacy code:

1. **Feature Flag Rollout Complete**: `ORCHESTRATOR_ADAPTER_ENABLED=true` in production
2. **Parity Tests Passing**: All cross-language contract tests pass with adapters enabled
3. **Production Validation**: At least 2 weeks of stable operation with adapters in production
4. **Rollback Capability**: Ability to revert to legacy path documented and tested

## Redundant Sandbox Modules

### Tier 1: Core Engine Logic (High Priority for Elimination)

These modules duplicate rules logic that is now handled by `src/shared/engine/orchestration/turnOrchestrator.ts`:

| Module                                                                                                   | Lines | Replacement                                                                                                                                                                                                    | Dependencies                                           |
| -------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| ~~[`src/client/sandbox/sandboxTurnEngine.ts`](../../src/client/sandbox/sandboxTurnEngine.ts)~~           | 377   | **Removed in Phase 6.2** – responsibilities now live in shared `turnLogic` and `ClientSandboxEngine` turn helpers                                                                                              | (was: `ClientSandboxEngine.ts`, tests)                 |
| ~~[`src/client/sandbox/sandboxMovementEngine.ts`](../../src/client/sandbox/sandboxMovementEngine.ts)~~   | 678   | **Removed in Phase 6.2** – movement/capture orchestration now lives in MovementAggregate/CaptureAggregate plus `ClientSandboxEngine` helpers                                                                   | (was: `ClientSandboxEngine.ts`, tests)                 |
| ~~[`src/client/sandbox/sandboxLinesEngine.ts`](../../src/client/sandbox/sandboxLinesEngine.ts)~~         | 285   | **Removed in Phase 6.2** – responsibilities now live in shared `lineDecisionHelpers` and `ClientSandboxEngine.processLinesForCurrentPlayer()`                                                                  | (was: `ClientSandboxEngine.ts`, `sandboxLines.ts`)     |
| ~~[`src/client/sandbox/sandboxTerritoryEngine.ts`](../../src/client/sandbox/sandboxTerritoryEngine.ts)~~ | 373   | **Removed in Phase 6.2** – responsibilities now live in shared territory helpers and `ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer()` / `getValidTerritoryProcessingMovesForCurrentPlayer()` | (was: `ClientSandboxEngine.ts`, `sandboxTerritory.ts`) |

**Subtotal Tier 1:** ~1,713 lines

### Tier 2: Support Modules (Medium Priority)

These modules provide helper functions used by Tier 1. After Tier 1 elimination, these become largely unused:

| Module                                                                                           | Lines | Status                | Notes                                                         |
| ------------------------------------------------------------------------------------------------ | ----- | --------------------- | ------------------------------------------------------------- |
| [`src/client/sandbox/sandboxMovement.ts`](../../src/client/sandbox/sandboxMovement.ts)           | ~200  | Partially replaceable | `applyMarkerEffectsAlongPath` may be useful for visualization |
| [`src/client/sandbox/sandboxCaptures.ts`](../../src/client/sandbox/sandboxCaptures.ts)           | ~400  | Replaceable           | `CaptureValidator` + `CaptureMutator` in orchestrator         |
| [`src/client/sandbox/sandboxCaptureSearch.ts`](../../src/client/sandbox/sandboxCaptureSearch.ts) | ~150  | Replaceable           | `captureLogic.ts` in shared engine                            |
| [`src/client/sandbox/sandboxElimination.ts`](../../src/client/sandbox/sandboxElimination.ts)     | ~100  | Replaceable           | `TerritoryMutator.eliminateRingsFromStack()`                  |
| [`src/client/sandbox/sandboxLines.ts`](../../src/client/sandbox/sandboxLines.ts)                 | ~150  | Replaceable           | `lineDetection.ts` in shared engine                           |
| [`src/client/sandbox/sandboxTerritory.ts`](../../src/client/sandbox/sandboxTerritory.ts)         | ~250  | Replaceable           | `territoryDetection.ts` + `TerritoryMutator`                  |
| [`src/client/sandbox/sandboxPlacement.ts`](../../src/client/sandbox/sandboxPlacement.ts)         | ~100  | Replaceable           | `PlacementMutator` in orchestrator                            |
| [`src/client/sandbox/sandboxVictory.ts`](../../src/client/sandbox/sandboxVictory.ts)             | ~50   | Replaceable           | `victoryLogic.ts` in shared engine                            |
| [`src/client/sandbox/sandboxGameEnd.ts`](../../src/client/sandbox/sandboxGameEnd.ts)             | ~80   | Replaceable           | Victory processing in orchestrator                            |

**Subtotal Tier 2:** ~1,480 lines (estimated)

### Tier 3: Backend Parallel Implementations

Legacy backend rules code that parallels shared engine after adapter migration:

| Module                                                                                               | Lines | Status      | Notes                                                    |
| ---------------------------------------------------------------------------------------------------- | ----- | ----------- | -------------------------------------------------------- |
| [`src/server/game/rules/lineProcessing.ts`](../../src/server/game/rules/lineProcessing.ts)           | ~200  | Replaceable | Uses shared engine but can delegate entirely via adapter |
| [`src/server/game/rules/territoryProcessing.ts`](../../src/server/game/rules/territoryProcessing.ts) | ~250  | Replaceable | Uses shared engine but can delegate entirely via adapter |
| [`src/server/game/rules/captureChainEngine.ts`](../../src/server/game/rules/captureChainEngine.ts)   | ~150  | Replaceable | Thin wrapper around shared `captureChainHelpers.ts`      |

**Subtotal Tier 3:** ~600 lines

## Total Lines for Elimination

| Tier                         | Lines      | Priority |
| ---------------------------- | ---------- | -------- |
| Tier 1: Core Sandbox Engines | ~1,713     | High     |
| Tier 2: Support Modules      | ~1,480     | Medium   |
| Tier 3: Backend Parallel     | ~600       | Medium   |
| **Total**                    | **~3,793** | -        |

## Elimination Phases

### Phase A: Feature Toggle Bake-in (Current State)

- [x] `TurnEngineAdapter.ts` implemented and tested
- [x] `SandboxOrchestratorAdapter.ts` implemented and tested
- [x] `ORCHESTRATOR_ADAPTER_ENABLED` environment variable added
- [x] `GameEngine.ts` reads feature flag from config

**Duration:** 2-4 weeks of production validation

### Phase B: Tier 1 Elimination

1. **Remove `sandboxTurnEngine.ts`** ✅
   - `ClientSandboxEngine.ts` now calls shared `advanceTurnAndPhase` directly with sandbox-specific delegates for placements, movement/capture, and forced elimination.
   - Legacy `sandboxTurnEngine.ts` has been deleted; phase management remains in `ClientSandboxEngine` as a host adapter over shared turn logic.
2. **Remove `sandboxMovementEngine.ts`** ✅
   - Movement click handling now lives entirely in `ClientSandboxEngine.handleMovementClick` / `handleLegacyMovementClick`, which delegate to MovementAggregate/CaptureAggregate and orchestrator where enabled.
   - Legacy `sandboxMovementEngine.ts` (handleMovementClickSandbox/performCaptureChainSandbox) has been deleted.
3. **Remove `sandboxLinesEngine.ts`** ✅
   - Line processing in the sandbox now routes directly through shared `lineDecisionHelpers` via `ClientSandboxEngine.getValidLineProcessingMovesForCurrentPlayer()` and `processLinesForCurrentPlayer()`.
   - Module `sandboxLinesEngine.ts` has been deleted; remaining sandbox line helpers are thin adapters over the shared engine.
4. **Remove `sandboxTerritoryEngine.ts`** ✅
   - Territory processing in the sandbox now routes directly through shared territory helpers via `ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer()` and `getValidTerritoryProcessingMovesForCurrentPlayer()`.
   - Module `sandboxTerritoryEngine.ts` has been deleted; remaining sandbox territory helpers are thin adapters over the shared engine.

**Dependencies to Update:**

- `ClientSandboxEngine.ts` - Major refactor to use only adapter
- `tests/unit/ClientSandboxEngine.*.test.ts` - Update test imports
- `tests/scenarios/*.test.ts` - May need adapter enabling

### Phase C: Tier 2 Cleanup

After Phase B validation, systematically eliminate helper modules:

1. Remove imports from `ClientSandboxEngine.ts`
2. Delete modules with zero imports
3. Update any remaining visualization-only helpers

### Phase D: Backend Consolidation

1. Remove `lineProcessing.ts` if GameEngine uses only adapter
2. Remove `territoryProcessing.ts` if GameEngine uses only adapter
3. Keep `captureChainEngine.ts` only if still needed for chain state tracking

## Rollback Strategy

If issues arise during elimination:

1. **Revert Commits**: Each phase is a single commit for easy revert
2. **Feature Flag**: Set `ORCHESTRATOR_ADAPTER_ENABLED=false` to use legacy path
3. **Dual Path Testing**: Legacy path remains functional until Phase D

## Code Quality Impact

### Positive Impacts

- **Single Source of Truth**: All rules logic in `src/shared/engine/`
- **Reduced Duplication**: ~3,800 fewer lines of duplicated logic
- **Easier Maintenance**: One place to fix bugs or add features
- **Better Testing**: Contract tests cover all platforms

### Risk Mitigation

- **Incremental Deletion**: One module at a time with full test suite runs
- **Contract Tests**: Cross-language parity verified before each deletion
- **Production Monitoring**: Adapter metrics track divergence

## Test Coverage After Elimination

Tests that will need updates:

| Test Category        | Files                           | Action                     |
| -------------------- | ------------------------------- | -------------------------- |
| Unit tests (sandbox) | `ClientSandboxEngine.*.test.ts` | Update to use adapter mode |
| Scenario tests       | `tests/scenarios/*.test.ts`     | Enable adapter globally    |
| Parity tests         | `ai-service/tests/parity/*.py`  | Already adapter-aware      |

## Timeline Estimate

| Phase      | Duration      | Dependencies          |
| ---------- | ------------- | --------------------- |
| A: Bake-in | 2-4 weeks     | Production monitoring |
| B: Tier 1  | 1-2 weeks     | Phase A complete      |
| C: Tier 2  | 1 week        | Phase B complete      |
| D: Backend | 1 week        | Phase C complete      |
| **Total**  | **5-8 weeks** | -                     |

## Success Metrics

1. **Zero Regression**: All tests pass after each elimination
2. **Reduced Build Size**: Measurable decrease in client bundle
3. **Simplified Imports**: Fewer cross-module dependencies
4. **Faster CI**: Fewer test files to run

## References

- [Phase 3 Adapter Migration Report](PHASE3_ADAPTER_MIGRATION_REPORT.md)
- [Rules Engine Consolidation Design](RULES_ENGINE_CONSOLIDATION_DESIGN.md)
- [Test Layering Strategy](../../tests/TEST_LAYERS.md)
- [Environment Variables](../ENVIRONMENT_VARIABLES.md)
