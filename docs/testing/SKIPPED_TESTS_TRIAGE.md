# Skipped Tests Triage Report

> **Date:** 2025-12-06 (updated)
> **Status:** Analysis Complete, **2 Category E tests fixed**, **4 TH-5 obsolete tests deleted**, **22 contract vector tests skipped with documentation**
> **Priority:** Critical (per WAVE2_ASSESSMENT_REPORT.md)

---

## 1. Executive Summary

### Total Skipped Tests Found

| Language   | Count  | Notes                                                             |
| ---------- | ------ | ----------------------------------------------------------------- |
| TypeScript | 28     | Includes describe.skip, it.skip, test.skip (4 deleted 2025-12-06) |
| Python     | 19     | Includes @pytest.mark.skip and @pytest.mark.skipif                |
| **Total**  | **47** | Direct skip markers identified                                    |

> **Note:** The Wave 2 assessment mentioned "160+ skipped tests". This higher number likely includes:
>
> - Tests dynamically skipped at runtime via conditional logic
> - Nested tests within skipped describe blocks (counted as 1 here but may execute as N tests)
> - Tests that were unskipped since the assessment

### Count by Category

| Category              | Count       | Impact                                                        |
| --------------------- | ----------- | ------------------------------------------------------------- |
| **A. UNSKIP-NOW**     | 0           | No tests ready to unskip without code changes                 |
| **B. UNSKIP-PENDING** | ~~3~~ **0** | **2025-12-11**: All Category B items resolved or reclassified |
| **C. DELETE**         | 1           | Test file no longer exists                                    |
| **D. KEEP-SKIPPED**   | 44          | Valid reasons to remain skipped                               |
| **E. REWRITE**        | ~~5~~ **3** | ~~Test concept valid but needs rework~~ 2 fixed (see below)   |

### Effort Estimate for Remediation

| Priority                        | Effort     | Scope                |
| ------------------------------- | ---------- | -------------------- |
| Batch 1 (A - Quick wins)        | 0 hours    | None available       |
| Batch 2 (C - Deletions)         | 0 hours    | None identified      |
| Batch 3 (B+E - Pending/Rewrite) | 8-16 hours | 8 tests              |
| Batch 4 (D - Keep as-is)        | 0 hours    | Document and monitor |

---

## 2. Detailed Triage Table

### TypeScript Tests

| File                                                                                                                       | Test Name                                                        | Skip Reason                                                                     | Category | Action       | Notes                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| [`envFlags.test.ts`](../tests/unit/envFlags.test.ts:117)                                                                   | allows placeholder JWT secrets in development                    | TODO-ENV-ISOLATION: Jest runs with NODE_ENV=test, can't test other environments | D        | Keep skipped | Requires process spawn harness                                                                                                    |
| [`envFlags.test.ts`](../tests/unit/envFlags.test.ts:135)                                                                   | accepts strong non-placeholder JWT secrets in production         | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                                                                                                    |
| [`envFlags.test.ts`](../tests/unit/envFlags.test.ts:153)                                                                   | rejects missing JWT secrets in production                        | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                                                                                                    |
| [`envFlags.test.ts`](../tests/unit/envFlags.test.ts:170)                                                                   | rejects placeholder JWT secrets in production                    | Same as above                                                                   | D        | Keep skipped | Requires process spawn harness                                                                                                    |
| [`statePersistence.branchCoverage.test.ts`](../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:211)             | creates download link (browser-only)                             | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                                                                                              |
| [`statePersistence.branchCoverage.test.ts`](../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:217)             | exports game state with given name (browser-only)                | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                                                                                              |
| [`statePersistence.branchCoverage.test.ts`](../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:311-323)         | importScenarioFromFile variants (13 tests)                       | Requires File.text() browser API                                                | D        | Keep skipped | Can't run in Node.js                                                                                                              |
| [`statePersistence.branchCoverage.test.ts`](../tests/unit/sandbox/statePersistence.branchCoverage.test.ts:327)             | importAndSaveScenarioFromFile (browser-only)                     | Requires browser DOM APIs                                                       | D        | Keep skipped | Can't run in Node.js                                                                                                              |
| [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](../tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts:1095)      | DIAGNOSTIC ONLY: backend movement moves                          | Intentional diagnostic test                                                     | D        | Keep skipped | For local debugging only                                                                                                          |
| [`FullGameFlow.test.ts`](../tests/integration/FullGameFlow.test.ts:43)                                                     | Full Game Flow (entire describe)                                 | Heavy AI soak test, 30+ seconds, times out                                      | D        | Keep skipped | Run via nightly orchestrator soak                                                                                                 |
| [`Python_vs_TS.selfplayReplayFixtureParity.test.ts`](../tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts:135) | Python vs TS self-play replay parity (entire describe)           | Skipped pending parity infrastructure completion                                | D        | Keep skipped | **Updated 2025-12-11**: 24 tests pass; describe.skip preserved for CI stability; some square19 fixtures have expected divergences |
| [`Python_vs_TS.selfplayReplayFixtureParity.test.ts`](../tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts:146) | No parity fixtures found (conditional)                           | Fixture-dependent skip                                                          | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`Python_vs_TS.selfplayReplayFixtureParity.test.ts`](../tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts:341) | No state bundles found (conditional)                             | Fixture-dependent skip                                                          | D        | Keep skipped | Valid conditional skip                                                                                                            |
| ~~`Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts`~~                                                                 | ~~combined line + territory parity~~                             | ~~traceMode breaks on line_order after choose_line_option~~                     | ~~B~~    | **DELETED**  | **2025-12-11**: Test file no longer exists; functionality covered by other parity tests                                           |
| [`contractVectorRunner.test.ts`](../tests/contracts/contractVectorRunner.test.ts:404)                                      | sequence with skip field (conditional)                           | Vectors may have skip field for pending implementation                          | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`goldenReplay.test.ts`](../tests/golden/goldenReplay.test.ts:28)                                                          | No golden game fixtures found (conditional)                      | Fixture-dependent skip                                                          | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`Python_vs_TS.traceParity.test.ts`](../tests/unit/Python_vs_TS.traceParity.test.ts:69)                                    | No test vectors found (conditional)                              | Vector-dependent skip                                                           | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`visual-regression.e2e.spec.ts`](../tests/e2e/visual-regression.e2e.spec.ts:304)                                          | Skip if not in local sandbox mode                                | Environment-dependent skip                                                      | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`visual-regression.e2e.spec.ts`](../tests/e2e/visual-regression.e2e.spec.ts:401)                                          | Victory modal did not appear                                     | Flow-dependent skip                                                             | D        | Keep skipped | Valid conditional skip                                                                                                            |
| [`turnOrchestrator.branchCoverage.test.ts`](../tests/unit/turnOrchestrator.branchCoverage.test.ts:921)                     | handles chain capture decision by returning without auto-resolve | Multi-phase model transition                                                    | D        | Keep skipped | Phase transitions to line_processing                                                                                              |
| [`turnOrchestrator.branchCoverage.test.ts`](../tests/unit/turnOrchestrator.branchCoverage.test.ts:1417)                    | handles end_chain_capture move                                   | Invalid move type                                                               | D        | Keep skipped | end_chain_capture not in allowed moves                                                                                            |
| [`turnOrchestrator.branchCoverage.test.ts`](../tests/unit/turnOrchestrator.branchCoverage.test.ts:2067)                    | returns early on chain capture decision without auto-resolving   | Undefined move passed                                                           | D        | Keep skipped | Test design incompatible with strict validation                                                                                   |
| [`turnOrchestrator.branchCoverage.test.ts`](../tests/unit/turnOrchestrator.branchCoverage.test.ts:2435)                    | ends chain capture when no more continuations available          | Invalid move type                                                               | D        | Keep skipped | end_chain_capture not in allowed moves                                                                                            |

### Python Tests

| File                                                                                                                          | Test Name                                        | Skip Reason                                 | Category | Action                | Notes                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------- | -------- | --------------------- | -------------------------------------------------------------------------------- |
| [`test_heuristic_training_evaluation.py`](../ai-service/tests/test_heuristic_training_evaluation.py:100)                      | test_evaluate_fitness_zero_profile...            | ~~TODO-HEURISTIC-ZERO-WEIGHTS~~             | ~~E~~ ✅ | ~~Rewrite~~ **FIXED** | **Rewrote as wiring test `test_evaluate_fitness_zero_profile_wiring_and_stats`** |
| [`test_multi_board_evaluation.py`](../ai-service/tests/test_multi_board_evaluation.py:50)                                     | test_zero_profile_is_worse...                    | Slow integration test (plays full AI games) | D        | Keep skipped          | Run locally with --timeout=300                                                   |
| [`test_benchmark_make_unmake.py`](../ai-service/tests/test_benchmark_make_unmake.py:86)                                       | test_incremental_faster_than_legacy              | Performance varies by environment           | D        | Keep skipped          | Environment-dependent performance test                                           |
| [`test_mps_architecture.py:199`](../ai-service/tests/test_mps_architecture.py:199)                                            | MPS backend tests (2 tests)                      | Requires Apple Silicon with MPS backend     | D        | Keep skipped          | Hardware-dependent                                                               |
| [`test_mcts_ai.py:71`](../ai-service/tests/test_mcts_ai.py:71)                                                                | MCTS tests                                       | MCTS_TESTS_ENABLED env variable must be set | D        | Keep skipped          | Controlled by env variable                                                       |
| [`test_line_and_territory_scenario_parity.py:640`](../ai-service/tests/parity/test_line_and_territory_scenario_parity.py:640) | test_overlength_line_option2_segments_exhaustive | ~~Test isolation issue~~                    | ~~E~~ ✅ | ~~Rewrite~~ **FIXED** | **Uses `monkeypatch` for proper isolation; 3 parametrized tests pass**           |
| [`test_rules_parity_fixtures.py:122`](../ai-service/tests/parity/test_rules_parity_fixtures.py:122)                           | Rule parity fixture tests (4 entries)            | Fixture-dependent skipif                    | D        | Keep skipped          | Valid conditional skip                                                           |
| [`test_golden_replay.py:303`](../ai-service/tests/golden/test_golden_replay.py:303)                                           | Golden replay tests                              | Fixture-dependent skipif                    | D        | Keep skipped          | Valid conditional skip                                                           |
| [`invariants/test_*.py`](../ai-service/tests/invariants/)                                                                     | Various invariant regression tests (5 files)     | Snapshot-file-dependent skipif              | D        | Keep skipped          | Valid conditional skip                                                           |

---

## 3. Priority Batches

### Batch 1: Quick Wins (Category A - UNSKIP-NOW)

**No tests in this category.** All skipped tests have valid reasons or require code changes.

---

### Batch 2: Deletions (Category C - DELETE)

**No tests identified for deletion.** All skipped tests test valid functionality.

---

### Batch 3: Requires Work (Categories B and E)

**Estimated Effort: 8-16 hours**

#### B. UNSKIP-PENDING (3 tests)

| Priority | Test                                                   | Blocker                           | Estimated Effort                 |
| -------- | ------------------------------------------------------ | --------------------------------- | -------------------------------- |
| 1        | Python_vs_TS.selfplayReplayFixtureParity describe.skip | Parity infrastructure (PA-1/PA-3) | 4 hours (after blocker resolved) |
| 2        | Backend_vs_Sandbox combined line+territory             | Sandbox traceMode fix needed      | 2-4 hours                        |

#### E. REWRITE (~~5~~ 3 tests remaining, 2 FIXED)

| Priority | Test                                                 | Issue                               | Estimated Effort | Status                  |
| -------- | ---------------------------------------------------- | ----------------------------------- | ---------------- | ----------------------- |
| ~~1~~    | ~~test_evaluate_fitness_zero_profile~~               | ~~HeuristicAI hardcoded penalties~~ | ~~2-4 hours~~    | ✅ **FIXED 2025-12-06** |
| ~~2~~    | ~~test_overlength_line_option2_segments_exhaustive~~ | ~~Mock cleanup isolation~~          | ~~1-2 hours~~    | ✅ **FIXED 2025-12-06** |

**Fixes applied:**

1. **test_evaluate_fitness_zero_profile** → Rewrote as `test_evaluate_fitness_zero_profile_wiring_and_stats`. The original test had a flawed premise (assuming zero-weight profile = worse play), but zero weights actually lead to deterministic first-move selection which can paradoxically win. The new test is a **wiring test** that verifies the harness tracks stats correctly and applies weights, without asserting which profile wins.

2. **test_overlength_line_option2_segments_exhaustive** → Already fixed in prior work. Uses pytest `monkeypatch` fixture for proper test isolation. All 3 parametrized tests (square8, square19, hexagonal) pass when run in suite or isolation.

---

## 4. Contract Vector Tests (PA-1)

**Date added:** 2025-12-06

### Summary

Contract vector tests in `tests/contracts/contractVectorRunner.test.ts` have been partially skipped due to a fundamental architectural change in the turn orchestrator.

| Status      | Count | Notes                                                                                |
| ----------- | ----- | ------------------------------------------------------------------------------------ |
| **Passing** | 18    | Loading/structure, placement, movement, capture, line, smoke, 4 multi-step sequences |
| **Skipped** | 14    | Territory vectors (data issues), 13 multi-step sequences (territory-related)         |
| **Total**   | 32    |                                                                                      |

**2025-12-06 Update:** Added `autoCompleteTurn()` function to handle multi-phase turns. This fixed 8 tests:

- Single-vector: movement, capture, line detection, smoke
- Multi-step: chain_capture.depth2.square19, chain_capture.depth3.linear.square8/19, hex_edge_case.edge_chain.hexagonal

### Root Cause

The contract vectors were created with **single-step turn completion** expectations:

- Vector expects: `status: 'complete'` after one move
- Orchestrator returns: `status: 'awaiting_decision'` for multi-phase turns

The turn orchestrator now **correctly implements multi-phase turns** per the game rules:

1. Movement creates a marker → triggers `line_processing` phase
2. Player must process lines (or NO_LINE_ACTION) → triggers `territory_processing` phase
3. Player may process any subset of territories (PROCESS_REGION + any ELIMINATE_FROM_STACK), or explicitly skip (SKIP_TERRITORY_PROCESSING); if no regions exist, they record NO_TERRITORY_ACTION → turn completes

### Skipped Test Categories

| Test                                       | Reason                                   | File                             |
| ------------------------------------------ | ---------------------------------------- | -------------------------------- |
| `should pass all movement vectors`         | Expects complete, gets awaiting_decision | contractVectorRunner.test.ts:269 |
| `should pass all capture vectors`          | Expects complete, gets awaiting_decision | contractVectorRunner.test.ts:296 |
| `should pass all line detection vectors`   | Expects complete, gets awaiting_decision | contractVectorRunner.test.ts:323 |
| `should pass all territory vectors`        | Expects complete, gets awaiting_decision | contractVectorRunner.test.ts:350 |
| `should pass all smoke vectors`            | Includes movement/capture tests          | contractVectorRunner.test.ts:372 |
| `Multi-step contract sequences` (describe) | Phase invariant violations               | contractVectorRunner.test.ts:406 |

### Remediation

**Option 1: Regenerate vectors with decision paths** (Recommended)

- Update vector generator scripts to include explicit decision moves
- Each movement/capture vector needs: move → NO_LINE_ACTION → (NO_TERRITORY_ACTION if no regions exist; otherwise SKIP_TERRITORY_PROCESSING or explicit PROCESS_REGION decision chain)
- Effort: 8-16 hours

**Option 2: Create simple vectors that don't trigger line/territory**

- Design scenarios with isolated stacks that can't form lines
- Limited coverage but faster to implement
- Effort: 4-8 hours

### Related Issues

- **PA-1**: Contract vectors need multi-phase decision paths
- **PA-4**: MCTS AI phase model mismatch (uses simplified phases, documented separately)

---

## 5. TurnOrchestrator Chain Capture Tests (TH-5)

**Date added:** 2025-12-06

### Summary

Four tests in `tests/unit/turnOrchestrator.branchCoverage.test.ts` have been skipped due to the multi-phase turn model and strict phase/move validation.

| Status      | Count | Notes                           |
| ----------- | ----- | ------------------------------- |
| **Passing** | 104   | Core orchestrator functionality |
| **Skipped** | 4     | Chain capture phase tests       |
| **Total**   | 108   |                                 |

### Root Cause

The tests were written for an earlier API that expected:

1. `end_chain_capture` as a valid move type - but it doesn't exist in the current phase model
2. Chain capture to remain in `chain_capture` phase after processing - but it transitions to `line_processing`
3. Ability to pass undefined moves after capture - but strict validation now rejects this

The `chain_capture` phase only allows: `overtaking_capture`, `continue_capture_segment`. Chain captures end via decision resolution, not explicit `end_chain_capture` moves.

### Skipped Tests

| Test                                                               | Reason                                                                   |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `handles chain capture decision by returning without auto-resolve` | After `continue_capture_segment`, phase transitions to `line_processing` |
| `handles end_chain_capture move`                                   | `end_chain_capture` not in allowed moves for `chain_capture` phase       |
| `returns early on chain capture decision without auto-resolving`   | Test passes undefined move to orchestrator                               |
| `ends chain capture when no more continuations available`          | `end_chain_capture` not in allowed moves                                 |

### Remediation

**Option 1: Delete tests** (Recommended for 3 of 4)

- The `end_chain_capture` tests test functionality that no longer exists
- Chain captures now end automatically via decision resolution

**Option 2: Rewrite to match current model**

- Create new tests that verify chain capture → line_processing transition
- Effort: 2-4 hours

---

### Batch 4: Keep As-Is (Category D - KEEP-SKIPPED)

**43 tests** across the following categories should remain skipped:

| Reason                         | Count | Examples                               |
| ------------------------------ | ----- | -------------------------------------- |
| **Browser-only tests**         | 15    | statePersistence DOM tests             |
| **Environment limitations**    | 4     | envFlags NODE_ENV tests                |
| **Fixture/snapshot-dependent** | 12    | Parity, golden replay, invariant tests |
| **Performance/slow tests**     | 3     | FullGameFlow, multi-board evaluation   |
| **Hardware-dependent**         | 2     | MPS (Apple Silicon) tests              |
| **Intentional diagnostics**    | 1     | DIAGNOSTIC ONLY tests                  |
| **Env-variable controlled**    | 2     | MCTS tests                             |

---

## 4. Recommendations

### Immediate Actions (This Sprint)

1. **Add skip comments to all skipped tests** - Many tests lack explanatory comments. Add `// SKIP-REASON:` comments per CI guard proposal in TH-6.

2. **Document browser-only tests** - Add a `@browser-only` tag or similar to the 15 statePersistence tests and consider moving them to a separate E2E browser test suite.

3. **Enable conditional skip validation** - Verify that fixture-dependent skipif conditions are checking for the correct paths and that the fixture generation scripts are documented.

### Short-term Actions (Next 2 Weeks)

1. **Fix test_overlength_line_option2_segments_exhaustive** - The mock cleanup isolation issue is straightforward to fix by using proper test fixtures instead of monkeypatching module-level functions.

2. **Address HeuristicAI hardcoded penalties** - Refactor `HeuristicAI` to make all evaluation components weight-parameterized, then unskip `test_evaluate_fitness_zero_profile`.

3. **Fix sandbox traceMode** - After fixing the traceMode issue with line_order decisions, unskip the combined line+territory parity test.

### Medium-term Actions (4+ Weeks)

1. **Complete parity infrastructure (PA-1, PA-3)** - Once TypeScript/Python parity vectors reach 100+, the describe.skip on `Python_vs_TS.selfplayReplayFixtureParity` can be removed.

2. **Consider browser test framework** - Set up Playwright or similar for the 15 browser-only tests instead of keeping them permanently skipped in Jest.

### Patterns Identified

| Pattern                 | Count | Root Cause                              | Solution                            |
| ----------------------- | ----- | --------------------------------------- | ----------------------------------- |
| Fixture-dependent skips | 17    | Running tests before fixtures generated | Generate fixtures in CI pre-step    |
| Browser-only tests      | 15    | Jest/Node can't run DOM APIs            | Move to E2E browser suite           |
| Environment isolation   | 4     | Jest's NODE_ENV=test limitation         | Use process spawning or accept skip |
| Performance tests       | 3     | Too slow for regular CI                 | Run in nightly/weekly schedule      |

---

## 5. Comparison to Wave 2 Assessment

The Wave 2 assessment identified "160+ skipped tests". Our scan found 47 explicit skip markers. The difference is likely due to:

1. **Nested test counting** - The `describe.skip` blocks (e.g., FullGameFlow, selfplayReplayFixtureParity) each contain multiple `it()` blocks counted as one skip marker but representing many tests.

2. **Dynamic skips** - Tests using `test.skip()` inside a loop or conditional may generate multiple skipped tests at runtime.

3. **Recent unskips** - Some tests may have been unskipped since the Wave 2 assessment.

**Recommendation:** Run `npm test -- --listTests --no-coverage` and filter for skipped tests to get the exact runtime count.

---

## 6. Related Documents

- [`WAVE2_ASSESSMENT_REPORT.md`](./WAVE2_ASSESSMENT_REPORT.md) - Source of the 160+ skipped tests finding
- [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) - TH-NEW-1 tracks skipped tests
- [`ai-service/TRAINING_DATA_REGISTRY.md`](../ai-service/TRAINING_DATA_REGISTRY.md) - Parity gate status

---

## 7. Fix History

| Date       | Test                                                               | Fix Applied                                                                    | Verified              |
| ---------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------------------- |
| 2025-12-06 | `test_evaluate_fitness_zero_profile`                               | Rewrote as wiring test (`test_evaluate_fitness_zero_profile_wiring_and_stats`) | ✅ Passes             |
| 2025-12-06 | `test_overlength_line_option2_segments_exhaustive`                 | Already fixed with monkeypatch isolation                                       | ✅ 3/3 params pass    |
| 2025-12-06 | `handles chain capture decision by returning without auto-resolve` | DELETED - tested obsolete phase behavior                                       | ✅ Removed (TH-5)     |
| 2025-12-06 | `handles end_chain_capture move`                                   | DELETED - end_chain_capture move type doesn't exist                            | ✅ Removed (TH-5)     |
| 2025-12-06 | `returns early on chain capture decision without auto-resolving`   | DELETED - tested passing undefined moves                                       | ✅ Removed (TH-5)     |
| 2025-12-06 | `ends chain capture when no more continuations available`          | DELETED - end_chain_capture move type doesn't exist                            | ✅ Removed (TH-5)     |
| 2025-12-06 | `should pass all movement vectors`                                 | UNSKIPPED - runVector now auto-completes multi-phase turns                     | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all capture vectors`                                  | UNSKIPPED - runVector now auto-completes multi-phase turns                     | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all line detection vectors`                           | UNSKIPPED - runVector now auto-completes multi-phase turns                     | ✅ Passes (PA-1)      |
| 2025-12-06 | `should pass all smoke vectors`                                    | UNSKIPPED - runVector now auto-completes multi-phase turns                     | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth2.square19`                           | UNSKIPPED - multi-step sequences now use autoCompleteTurn                      | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.square8`                     | UNSKIPPED - multi-step sequences now use autoCompleteTurn                      | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.square19`                    | UNSKIPPED - multi-step sequences now use autoCompleteTurn                      | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence hex_edge_case.edge_chain.hexagonal`                      | UNSKIPPED - multi-step sequences now use autoCompleteTurn                      | ✅ Passes (PA-1)      |
| 2025-12-06 | `sequence chain_capture.depth3.linear.hexagonal`                   | SKIPPED - phase mismatch: continue_capture_segment in ring_placement           | ⏸️ Vector needs regen |

---

_Last updated: 2025-12-06 (Multi-step sequence tests unskipped via autoCompleteTurn)_
