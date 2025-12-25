# Skipped Tests Triage Report

> **Date:** 2025-12-25 (updated)
> **Status:** Analysis Complete - **New Category F identified: Legacy Orchestrator Tests (32 files, ~100+ tests)**
> **Priority:** Critical (per WAVE2_ASSESSMENT_REPORT.md)

---

## 1. Executive Summary

### Total Skipped Tests Found

| Language   | Count   | Notes                                                         |
| ---------- | ------- | ------------------------------------------------------------- |
| TypeScript | **102** | Includes describe.skip, it.skip, test.skip, conditional skips |
| Python     | 16      | Includes @pytest.mark.skip and @pytest.mark.skipif            |
| **Total**  | **118** | Direct skip markers identified                                |

> **Note:** The Wave 2 assessment mentioned "160+ skipped tests". This higher number includes:
>
> - Tests within skipped describe blocks (each counted as 1 marker but N individual tests)
> - The ~100+ tests in 32 files skipped via `skipWithOrchestrator` pattern
> - Tests that were unskipped since the assessment

### Count by Category

| Category                      | Files/Tests  | Impact                                                              |
| ----------------------------- | ------------ | ------------------------------------------------------------------- |
| **A. UNSKIP-NOW**             | 0            | No tests ready to unskip without code changes                       |
| **B. UNSKIP-PENDING**         | 0            | All resolved or reclassified                                        |
| **C. DELETE**                 | 0            | None identified for deletion                                        |
| **D. KEEP-SKIPPED**           | ~42          | Valid reasons to remain skipped                                     |
| **E. REWRITE**                | 0            | All fixed/deleted (see Fix History)                                 |
| **F. LEGACY-ORCHESTRATOR** 🆕 | **32 files** | Tests using legacy internal state manipulation; needs major rewrite |

### Effort Estimate for Remediation

| Priority                             | Effort      | Scope                            |
| ------------------------------------ | ----------- | -------------------------------- |
| Batch 1 (A - Quick wins)             | 0 hours     | None available                   |
| Batch 2 (C - Deletions)              | 0 hours     | None identified                  |
| Batch 3 (D - Keep as-is)             | 0 hours     | Document and monitor             |
| Batch 4 (F - Legacy Orchestrator) 🆕 | 40-80 hours | Rewrite to use orchestrator APIs |

---

## 2. NEW: Legacy Orchestrator Tests (Category F)

**Date added:** 2025-12-25

### Overview

**32 test files** and approximately **100+ individual tests** are skipped via the `skipWithOrchestrator` pattern. These tests directly manipulate internal `GameEngine` or `ClientSandboxEngine` state (e.g., `gameState.currentPhase = 'capture'`), which bypasses the orchestrator.

### Root Cause

As of **2025-12-01**, the orchestrator adapter is **permanently enabled**:

```typescript
// src/server/config/env.ts lines 401-404
ORCHESTRATOR_ADAPTER_ENABLED: z
  .any()
  .transform((): true => true)
  .default(true),
```

The test setup file [`tests/setup.ts`](../../tests/setup.ts:10) sets `process.env.ORCHESTRATOR_ADAPTER_ENABLED = 'true'`, which causes all tests using the following pattern to be skipped:

```typescript
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
(skipWithOrchestrator ? describe.skip : describe)('Test Suite', () => { ... });
```

### Files Using `skipWithOrchestrator` Pattern

| Directory            | Files | Description                              |
| -------------------- | ----- | ---------------------------------------- |
| `tests/unit/`        | 22    | GameEngine_vs_Sandbox parity, seed tests |
| `tests/scenarios/`   | 5     | FAQ Q1-Q23 rules scenarios               |
| `tests/integration/` | 3     | GameSession, Reconnection tests          |
| `tests/parity/`      | 2     | Backend_vs_Sandbox checkpoints           |

**Full list of affected files:**

```
tests/unit/TerritoryDecisions.HexRegionThenElim.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDecisions.SquareTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDecisions.SquareRegionThenElim.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDecisions.HexTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDecisions.Square19TwoRegionThenElim.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts
tests/unit/TerritoryDetection.seed5Move45.parity.test.ts
tests/unit/TerritoryDecision.seed5Move45.parity.test.ts
tests/unit/ClientSandboxEngine.invariants.test.ts
tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts
tests/unit/ClientSandboxEngine.traceStructure.test.ts
tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts
tests/unit/GameEngine.aiSimulation.debug.test.ts
tests/unit/GameEngine.aiSimulation.seed10.debug.test.ts
tests/unit/GameEngine.chainCapture.test.ts (3 tests)
tests/unit/ClientSandboxEngine.chainCapture.scenarios.test.ts (1 test)
tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts (1 test)
tests/unit/Seed5Move63.lineDetectionSnapshot.test.ts
tests/unit/Seed14Move35LineParity.test.ts
tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts
tests/unit/Seed17GeometryParity.GameEngine_vs_Sandbox.test.ts
tests/unit/TraceParity.seed5.firstDivergence.test.ts
tests/unit/RefactoredEngineParity.test.ts
tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts
tests/scenarios/FAQ_Q01_Q06.test.ts (21 tests)
tests/scenarios/FAQ_Q07_Q08.test.ts (11 tests)
tests/scenarios/FAQ_Q09_Q14.test.ts (23 tests)
tests/scenarios/FAQ_Q16_Q18.test.ts (10 tests)
tests/scenarios/FAQ_Q22_Q23.test.ts (16 tests)
tests/integration/GameSession.aiDeterminism.test.ts
tests/integration/GameReconnection.test.ts
tests/integration/GameSession.aiFatalFailure.test.ts
tests/parity/Backend_vs_Sandbox.seed5.bisectParity.test.ts
tests/parity/Backend_vs_Sandbox.seed5.checkpoints.test.ts
```

### Why These Tests Are Skipped

These tests manipulate internal engine state directly:

```typescript
// Example from FAQ_Q01_Q06.test.ts
const engineAny: any = engine;
const gameState = engineAny.gameState;
gameState.currentPhase = 'capture'; // Direct manipulation!
gameState.currentPlayer = 1;
```

The orchestrator now wraps all phase transitions and move validation. When it's enabled:

- Direct state manipulation is ignored by the orchestrator's phase machine
- Tests that rely on setting `currentPhase` manually no longer work
- The internal `chainCaptureState` field is not used by the orchestrator

### Recommended Actions

| Option               | Description                                                                    | Effort      | Recommendation                |
| -------------------- | ------------------------------------------------------------------------------ | ----------- | ----------------------------- |
| **1. Rewrite tests** | Update tests to use orchestrator-compatible APIs (TurnEngineAdapter, makeMove) | 40-80 hours | Recommended for FAQ/scenarios |
| **2. Keep skipped**  | Leave as documentation of legacy behavior                                      | 0 hours     | Acceptable for seed parity    |
| **3. Delete tests**  | Remove if coverage exists elsewhere                                            | 4-8 hours   | Consider for redundant files  |

### Priority Recommendations

1. **FAQ Q1-Q23 tests (5 files, 81 tests)** - HIGH PRIORITY rewrite candidates
   - Test fundamental game rules (stack mechanics, capture, movement)
   - Rules coverage is critical and should use canonical orchestrator
2. **TerritoryDecisions.\* tests (9 files)** - MEDIUM PRIORITY
   - Test territory processing decision flow
   - May be redundant with other territory tests in the suite
3. **Seed\* parity tests (6 files)** - LOW PRIORITY
   - Historical parity regression tests for specific seeds
   - Useful for debugging specific issues, not critical for CI

4. **Integration tests (3 files)** - MEDIUM PRIORITY
   - GameSession tests may need orchestrator-aware rewrites
   - Reconnection testing remains valuable

---

## 3. Detailed Triage Table

### TypeScript Tests (Category D - KEEP-SKIPPED)

| File                                                                                                                         | Test Name                                                        | Skip Reason                                                                     | Category | Action       | Notes                                                   |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------- | ------------ | ------------------------------------------------------- |
| [`envFlags.test.ts`](../../tests/unit/envFlags.test.ts:117)                                                                  | allows placeholder JWT secrets in development                    | TODO-ENV-ISOLATION: Jest runs with NODE_ENV=test, can't test other environments | D        | Keep skipped | Requires process spawn harness                          |
| [`envFlags.test.ts`](../../tests/unit/envFlags.test.ts:135)                                                                  | accepts strong non-placeholder JWT secrets in production         | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                          |
| [`envFlags.test.ts`](../../tests/unit/envFlags.test.ts:153)                                                                  | rejects missing JWT secrets in production                        | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                          |
| [`envFlags.test.ts`](../../tests/unit/envFlags.test.ts:170)                                                                  | rejects placeholder JWT secrets in production                    | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                          |
| [`statePersistence.branchCoverage.test.ts`](../../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:211)            | creates download link (browser-only)                             | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                    |
| [`statePersistence.branchCoverage.test.ts`](../../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:217)            | exports game state with given name (browser-only)                | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                    |
| [`statePersistence.branchCoverage.test.ts`](../../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:311-323)        | importScenarioFromFile variants (13 tests)                       | Requires File.text() browser API                                                | D        | Keep skipped | Can't run in Node.js                                    |
| [`statePersistence.branchCoverage.test.ts`](../../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:327)            | importAndSaveScenarioFromFile (browser-only)                     | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                    |
| [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](../../tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts:1095)     | DIAGNOSTIC ONLY: backend movement moves                          | Intentional diagnostic test                                                     | D        | Keep skipped | For local debugging only                                |
| [`FullGameFlow.test.ts`](../../tests/integration/FullGameFlow.test.ts:43)                                                    | Full Game Flow (entire describe)                                 | Heavy AI soak test, 30+ seconds, times out                                      | D        | Keep skipped | Run via nightly orchestrator soak                       |
| [`Python_vs_TS.selfplayReplayFixtureParity.test.ts`](../../tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts)    | Python vs TS self-play replay parity (entire describe)           | Skipped pending parity infrastructure completion                                | D        | Keep skipped | 24 tests pass; describe.skip preserved for CI stability |
| [`contractVectorRunner.test.ts`](../../tests/contracts/contractVectorRunner.test.ts:404)                                     | sequence with skip field (conditional)                           | Vectors may have skip field for pending implementation                          | D        | Keep skipped | Valid conditional skip                                  |
| [`goldenReplay.test.ts`](../../tests/golden/goldenReplay.test.ts:28)                                                         | No golden game fixtures found (conditional)                      | Fixture-dependent skip                                                          | D        | Keep skipped | Valid conditional skip                                  |
| [`Python_vs_TS.traceParity.test.ts`](../../tests/unit/Python_vs_TS.traceParity.test.ts:69)                                   | No test vectors found (conditional)                              | Vector-dependent skip                                                           | D        | Keep skipped | Valid conditional skip                                  |
| [`turnOrchestrator.core.branchCoverage.test.ts`](../../tests/unit/turnOrchestrator.core.branchCoverage.test.ts:934)          | handles chain capture decision by returning without auto-resolve | Multi-phase model transition                                                    | D        | Keep skipped | Phase transitions to line_processing                    |
| [`turnOrchestrator.advanced.branchCoverage.test.ts`](../../tests/unit/turnOrchestrator.advanced.branchCoverage.test.ts:296)  | handles end_chain_capture move                                   | Invalid move type                                                               | D        | Keep skipped | end_chain_capture not in allowed moves                  |
| [`turnOrchestrator.advanced.branchCoverage.test.ts`](../../tests/unit/turnOrchestrator.advanced.branchCoverage.test.ts:994)  | returns early on chain capture decision without auto-resolving   | Undefined move passed                                                           | D        | Keep skipped | Test design incompatible with strict validation         |
| [`turnOrchestrator.advanced.branchCoverage.test.ts`](../../tests/unit/turnOrchestrator.advanced.branchCoverage.test.ts:1318) | ends chain capture when no more continuations available          | Invalid move type                                                               | D        | Keep skipped | end_chain_capture not in allowed moves                  |

### Python Tests

| File                                                                                                   | Test Name                                    | Skip Reason                                 | Category | Action       | Notes                                  |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------- | -------- | ------------ | -------------------------------------- |
| [`test_multi_board_evaluation.py`](../../ai-service/tests/test_multi_board_evaluation.py:50)           | test_zero_profile_is_worse...                | Slow integration test (plays full AI games) | D        | Keep skipped | Run locally with --timeout=300         |
| [`test_benchmark_make_unmake.py`](../../ai-service/tests/test_benchmark_make_unmake.py:86)             | test_incremental_faster_than_legacy          | Performance varies by environment           | D        | Keep skipped | Environment-dependent performance test |
| [`test_mps_architecture.py:199`](../../ai-service/tests/test_mps_architecture.py:199)                  | MPS backend tests (2 tests)                  | Requires Apple Silicon with MPS backend     | D        | Keep skipped | Hardware-dependent                     |
| [`test_mcts_ai.py:71`](../../ai-service/tests/test_mcts_ai.py:71)                                      | MCTS tests                                   | MCTS_TESTS_ENABLED env variable must be set | D        | Keep skipped | Controlled by env variable             |
| [`test_rules_parity_fixtures.py:122`](../../ai-service/tests/parity/test_rules_parity_fixtures.py:122) | Rule parity fixture tests (4 entries)        | Fixture-dependent skipif                    | D        | Keep skipped | Valid conditional skip                 |
| [`test_golden_replay.py:303`](../../ai-service/tests/golden/test_golden_replay.py:303)                 | Golden replay tests                          | Fixture-dependent skipif                    | D        | Keep skipped | Valid conditional skip                 |
| [`invariants/test_*.py`](../../ai-service/tests/invariants)                                            | Various invariant regression tests (5 files) | Snapshot-file-dependent skipif              | D        | Keep skipped | Valid conditional skip                 |

---

## 4. Skip Pattern Summary

| Pattern                                        | Count     | Description                                               | Action                     |
| ---------------------------------------------- | --------- | --------------------------------------------------------- | -------------------------- |
| `skipWithOrchestrator` / `orchestratorEnabled` | 32 files  | Tests manipulating internal state, bypassing orchestrator | Category F - needs rewrite |
| Browser-only (`browser-only` comment)          | 16 tests  | Tests requiring DOM APIs                                  | Keep skipped               |
| Environment-dependent (`NODE_ENV`)             | 4 tests   | Tests requiring non-test environment                      | Keep skipped               |
| Fixture-dependent (conditional skip)           | ~15 tests | Tests skipping when fixtures don't exist                  | Keep skipped               |
| Heavy/slow tests                               | ~5 tests  | Performance or long-running tests                         | Keep skipped (run nightly) |
| Hardware-dependent (MPS)                       | 2 tests   | Tests requiring Apple Silicon                             | Keep skipped               |
| Obsolete (TH-5)                                | 4 tests   | Tests for removed functionality                           | Already deleted            |

---

## 5. Patterns Identified

| Pattern                       | Count    | Root Cause                              | Solution                                         |
| ----------------------------- | -------- | --------------------------------------- | ------------------------------------------------ |
| **Legacy orchestrator tests** | 32 files | Direct internal state manipulation      | Rewrite to use TurnEngineAdapter / makeMove APIs |
| Fixture-dependent skips       | ~17      | Running tests before fixtures generated | Generate fixtures in CI pre-step                 |
| Browser-only tests            | 16       | Jest/Node can't run DOM APIs            | Move to E2E browser suite (Playwright)           |
| Environment isolation         | 4        | Jest's NODE_ENV=test limitation         | Use process spawning or accept skip              |
| Performance tests             | ~5       | Too slow for regular CI                 | Run in nightly/weekly schedule                   |

---

## 6. Related Documents

- [`FSM_MIGRATION_STATUS_2025_12.md`](../architecture/FSM_MIGRATION_STATUS_2025_12.md) - Orchestrator migration status
- [`ORCHESTRATOR_ROLLOUT_PLAN.md`](../architecture/ORCHESTRATOR_ROLLOUT_PLAN.md) - Original rollout plan
- [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) - TH-NEW-1 tracks skipped tests
- [`ai-service/TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md) - Parity gate status

---

## 7. Fix History

| Date       | Test                                                               | Fix Applied                                                                              | Verified              |
| ---------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | --------------------- |
| 2025-12-06 | `test_evaluate_fitness_zero_profile`                               | Rewrote as wiring test (`test_evaluate_fitness_zero_profile_wiring_and_stats`)           | ✅ Passes             |
| 2025-12-06 | `test_overlength_line_option2_segments_exhaustive`                 | Already fixed with monkeypatch isolation                                                 | ✅ 3/3 params pass    |
| 2025-12-06 | `handles chain capture decision by returning without auto-resolve` | DELETED - tested obsolete phase behavior                                                 | ✅ Removed (TH-5)     |
| 2025-12-06 | `handles end_chain_capture move`                                   | DELETED - end_chain_capture move type doesn't exist                                      | ✅ Removed (TH-5)     |
| 2025-12-06 | `returns early on chain capture decision without auto-resolving`   | DELETED - tested passing undefined moves                                                 | ✅ Removed (TH-5)     |
| 2025-12-06 | `ends chain capture when no more continuations available`          | DELETED - end_chain_capture move type doesn't exist                                      | ✅ Removed (TH-5)     |
| 2025-12-06 | `should pass all movement vectors`                                 | UNSKIPPED - runVector now auto-completes multi-phase turns                               | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all capture vectors`                                  | UNSKIPPED - runVector now auto-completes multi-phase turns                               | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all line detection vectors`                           | UNSKIPPED - runVector now auto-completes multi-phase turns                               | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all smoke vectors`                                    | UNSKIPPED - runVector now auto-completes multi-phase turns                               | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth2.square19`                           | UNSKIPPED - multi-step sequences now use autoCompleteTurn                                | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.square8`                     | UNSKIPPED - multi-step sequences now use autoCompleteTurn                                | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.square19`                    | UNSKIPPED - multi-step sequences now use autoCompleteTurn                                | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence hex_edge_case.edge_chain.hexagonal`                      | UNSKIPPED - multi-step sequences now use autoCompleteTurn                                | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.hexagonal`                   | SKIPPED - phase mismatch: continue_capture_segment in ring_placement                     | ⏸️ Vector needs regen |
| 2025-12-23 | `GameEngine.gameEndExplanation.shared.test.ts` Q23-style test      | FIXED - Updated assertions for correct mini-region threshold (≤4 cells)                  | ✅ Passes             |
| 2025-12-23 | `ClientSandboxEngine.victory.LPS.sandboxFixtureRegression.test.ts` | DELETED - Fixture incomplete (missing initial moves); LPS covered by 106+ tests          | ✅ Removed            |
| 2025-12-23 | `TestDistributedTrainer` class in `test_distributed_training.py`   | DELETED - Tested obsolete DistributedTrainer class replaced by IntegratedTrainingManager | ✅ Removed            |
| 2025-12-23 | `TestDescentAIHex` class in `test_descent_ai.py`                   | DELETED - Tested obsolete `hex_model` attribute removed in v3 architecture               | ✅ Removed            |
| 2025-12-23 | `test_monitor_alerting.py` (entire file)                           | DELETED - Tests non-existent `scripts.monitor.alerting` module                           | ✅ Removed            |
| 2025-12-25 | Category F (Legacy Orchestrator) tests                             | DOCUMENTED - 32 files, ~100+ tests identified and categorized                            | 📝 Analysis complete  |

---

_Last updated: 2025-12-25 (Category F Legacy Orchestrator tests documented)_
