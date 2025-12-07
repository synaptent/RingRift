# Skipped Test Triage

This document categorizes all skipped tests in the RingRift codebase and provides remediation recommendations.

**Total skipped tests: 41 across 17 files**
**Date: 2025-12-06**

## Summary by Category

| Category                                | Count | Action                       |
| --------------------------------------- | ----- | ---------------------------- |
| Browser-only (DOM required)             | 16    | Document as known limitation |
| Missing fixtures (expected)             | 4     | Keep - correct behavior      |
| Environment isolation (Jest limitation) | 4     | Document as known limitation |
| Archived/redirect stubs                 | 3     | Delete stubs                 |
| Behavior changed (stale)                | 5     | Delete or update             |
| Debug/diagnostic (manual only)          | 5     | Move to archive              |
| Complex setup (infrastructure)          | 4     | Document requirements        |

## Detailed Categorization

### Category 1: Browser-Only Tests (Keep - Documented Limitation)

These tests require browser DOM APIs (`createElement`, `URL.createObjectURL`, `File.text()`) that don't exist in Node.js.

**File: `tests/unit/sandbox/statePersistence.branchCoverage.test.ts`** (16 tests)

- Line 211: `creates download link with sanitized filename (browser-only)`
- Line 217: `exports game state with given name (browser-only)`
- Lines 311-323: Import validation tests (12 tests)
- Line 327: `imports and saves valid scenario (browser-only)`

**Recommendation:** Keep skipped with clear documentation. Consider adding Playwright/browser-based test runner for these in the future.

---

### Category 2: Missing Fixtures Tests (Keep - Expected Behavior)

These tests correctly skip when fixture data hasn't been generated.

**File: `tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts`**

- Line 146: `No parity fixtures found – run ai-service/scripts/check_ts_python_replay_parity.py with --emit-fixtures-dir first`
- Line 341: `No state bundles found – run check_ts_python_replay_parity.py with --emit-state-bundles first`

**File: `tests/golden/goldenReplay.test.ts`**

- Line 28: `No golden game fixtures found - run curation script to generate`

**File: `tests/unit/Python_vs_TS.traceParity.test.ts`**

- Line 69: `No test vectors found`

**Recommendation:** Keep as-is. These are guarding against test failures when fixtures don't exist.

---

### Category 3: Environment Isolation Tests (Keep - Documented Jest Limitation)

These tests cannot run in Jest because Jest sets `NODE_ENV=test` at process start and config modules cache values.

**File: `tests/unit/envFlags.test.ts`** (4 tests)

- Line 117: `allows placeholder JWT secrets in development`
- Line 135: `accepts strong non-placeholder JWT secrets in production`
- Line 153: `rejects missing JWT secrets in production`
- Line 170: `rejects placeholder JWT secrets in production`

**Recommendation:** Keep skipped. Would require separate process spawner to test NODE_ENV variations properly. Consider moving to integration tests with process isolation.

---

### Category 4: Archived/Redirect Stubs (Delete)

These are placeholder tests that just point to archived versions.

**File: `tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`**

- Line 10: Redirect stub to `archive/tests/parity/`

**File: `tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`**

- Line 10: Redirect stub to `archive/tests/parity/`

**File: `tests/contracts/contractVectorRunner.test.ts`**

- Line 404: `sequence ${sequenceId} is internally consistent across steps` (dynamic skip)

**Recommendation:** Delete the redirect stub files entirely. The contract vector test appears to be a dynamic skip that may be valid.

---

### Category 5: Behavior Changed Tests (Delete or Update)

These tests expect behavior that has since changed.

**File: `tests/unit/GameSession.branchCoverage.test.ts`** (5 tests)

- Line 4067: `throws when move destination is missing` - TODO comment says behavior changed
- Line 4168: `throws when user is a spectator`
- Line 4321: `executes AI turn when current player is AI`
- Line 4366: `handles decision phase moves for AI player`
- Line 4419: `uses local fallback when getAIMove returns null`

**Recommendation:** Delete these tests. They test obsolete behavior and provide no value.

---

### Category 6: Debug/Diagnostic Tests (Move to Archive)

These are manual debugging aids, not CI-worthy tests.

**File: `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`**

- Line 47: `Hex cyclic capture (height 3 targets, r=4 triangle) sandbox diagnostics`

**File: `tests/unit/TraceParity.seed14.firstDivergence.test.ts`**

- Line 24: `Trace parity first-divergence helper: square8 / 2p / seed=14`

**File: `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`**

- Line 1095: `DIAGNOSTIC ONLY: backend movement moves at first movement-phase turn`

**File: `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`**

- Line 34: Already in archive

**File: `archive/tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`**

- Line 46: Already in archive

**Recommendation:** Move non-archived files to `archive/tests/` directory.

---

### Category 7: Complex Setup Tests (Document Requirements)

These tests require specific infrastructure that may not be available in all environments.

**File: `tests/integration/FullGameFlow.test.ts`**

- Line 43: `Full Game Flow Integration (AI Fallback)` - Requires AI service

**File: `tests/parity/Python_vs_TS.selfplayReplayFixtureParity.test.ts`**

- Line 135: `Python vs TS self-play replay parity (DB fixtures)` - Requires DB fixtures

**File: `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts`**

- Line 1014: `combined line + territory parity` - Complex scenario

**File: `archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`**

- Line 36: Already in archive

**Recommendation:** Document infrastructure requirements. Consider enabling in CI with proper setup.

---

## Action Plan

### Phase 1: Immediate Cleanup (Delete)

1. Delete `tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`
2. Delete `tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`
3. Delete 5 stale tests from `tests/unit/GameSession.branchCoverage.test.ts`

### Phase 2: Move to Archive

1. Move `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts` to archive
2. Move `tests/unit/TraceParity.seed14.firstDivergence.test.ts` to archive
3. Move diagnostic test from `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` to archive

### Phase 3: Documentation

1. Add skip reason comments to all remaining skipped tests
2. Update this document with any new findings

---

## Metrics After Remediation

| Metric                 | Before | After | Change     |
| ---------------------- | ------ | ----- | ---------- |
| Total skipped (tests/) | 41     | 29    | -12 (-29%) |
| Documented limitations | 0      | 29    | +29        |
| Stale/invalid tests    | 8      | 0     | -8         |

## Actions Completed (2025-12-06)

1. **Deleted redirect stub files:**
   - `tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`
   - `tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`

2. **Removed stale tests from `tests/unit/GameSession.branchCoverage.test.ts`:**
   - `throws when move destination is missing`
   - `throws when user is a spectator`
   - `executes AI turn when current player is AI`
   - `handles decision phase moves for AI player`
   - `uses local fallback when getAIMove returns null`

3. **Moved diagnostic tests to archive:**
   - `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts` -> `archive/tests/unit/`
   - `tests/unit/TraceParity.seed14.firstDivergence.test.ts` -> `archive/tests/unit/`

---

## Known Issue: Monolithic Test Files

The following test files exceed 1500 lines and should be decomposed in a future refactor:

| File                                        | Lines | Recommended Split                                 |
| ------------------------------------------- | ----- | ------------------------------------------------- |
| `GameSession.branchCoverage.test.ts`        | 4,460 | By method category (move handling, AI, broadcast) |
| `TerritoryAggregate.branchCoverage.test.ts` | 2,571 | By territory calculation type                     |
| `turnOrchestrator.branchCoverage.test.ts`   | 2,517 | By phase (placement, movement, capture, decision) |
| `LineAggregate.branchCoverage.test.ts`      | 1,917 | By line detection scenario                        |
| `CaptureAndTerritoryParity.test.ts`         | 1,619 | By parity test category                           |

**Recommendation:** Defer decomposition until a dedicated refactoring sprint to avoid test regressions.

---

## Known Issue: Weak Assertions

Found 66 uses of `toBeDefined()` across 20 test files. Many of these are legitimate guards (checking object exists before testing properties), but some could be strengthened.

**Priority:** Low - most are valid guards, not test weakness.

---

## Known Issue: Type Safety Gaps

Found 76 `as any` usages across 20 files in `src/`. High-count files:

| File                     | Count | Nature                        |
| ------------------------ | ----- | ----------------------------- |
| `ReplayService.ts`       | 20    | Dynamic replay state handling |
| `test-parity-cli.ts`     | 7     | CLI tool, acceptable          |
| `SelfPlayGameService.ts` | 6     | AI integration                |
| `useGameActions.ts`      | 6     | React hook state management   |
| `statePersistence.ts`    | 6     | JSON serialization            |
| `game.ts` (routes)       | 5     | Express request handling      |

**Recommendation:** Address incrementally. Focus on `ReplayService.ts` first as it has the highest count. Many usages are intentional for dynamic data or external APIs.

---

## Known Issue: Prisma Upgrade Pending

**Current Version:** 6.19.0
**Latest Version:** 7.1.0

Prisma 7.0 is a major release with significant breaking changes:

1. **ESM module format** - Requires migration from CommonJS
2. **New prisma.config.ts** - Required for CLI configuration
3. **Datasource URL changes** - No longer in schema files, must use adapters
4. **Environment variable loading** - Not loaded by default
5. **Node.js 20.19+** - Minimum version requirement
6. **Metrics feature removed** - Must use driver adapters instead

**Recommendation:** Defer to a dedicated upgrade task. The current 6.x version is stable and functional. Upgrade when there's a specific need for 7.x features.

**Resources:**

- [Upgrade Guide](https://www.prisma.io/docs/orm/more/upgrade-guides/upgrading-versions/upgrading-to-prisma-7)
- [Breaking Changes](https://github.com/prisma/prisma/issues/28573)
