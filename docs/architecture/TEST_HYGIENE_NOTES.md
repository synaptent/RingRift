# Test Hygiene Notes

> **Doc Status (2025-12-11): Active**
>
> This document tracks test suite hygiene issues and cleanup opportunities.

**Created:** 2025-12-11
**Last Updated:** 2025-12-11

---

## 1. Completed Cleanup (Dec 11, 2025)

### 1.1 Victory Test Consolidation

- **Renamed:** `victoryLogic.branchCoverage.test.ts` → `victory.evaluateVictory.branchCoverage.test.ts`
- **Reason:** `victoryLogic.ts` was removed; all victory logic is now in `VictoryAggregate.ts`
- **Updated references in:**
  - `tests/unit/VictoryAnmChains.scenarios.test.ts`
  - `tests/fixtures/anmFixtures.ts`
  - `tests/README.md`
  - `tests/parity/Backend_vs_Sandbox.eliminationTrace.test.ts`

### 1.2 Middleware Test Clarification

Renamed root-level middleware tests to clarify they are integration-level (not duplicates):

| Original Name                | New Name                                 | Reason                       |
| ---------------------------- | ---------------------------------------- | ---------------------------- |
| `rateLimiter.test.ts`        | `rateLimiter.integration.test.ts`        | Integration tests (39 tests) |
| `degradationHeaders.test.ts` | `degradationHeaders.integration.test.ts` | Integration tests            |
| `metricsMiddleware.test.ts`  | `metricsMiddleware.integration.test.ts`  | Integration tests            |

The `tests/unit/middleware/` directory contains unit tests for middleware internals.
Both test sets are complementary and provide different coverage.

**Total middleware tests:** 193 tests across all files

---

## 2. Known Test Organization Issues

### 2.1 Skipped Browser-Only Tests

**Location:** `tests/unit/sandbox/statePersistence.branchCoverage.test.ts`

18 tests are skipped due to browser-specific APIs:

- `File.text()` API
- DOM APIs for file download
- Browser-specific import/export functionality

**Recommendation:** Consider browser integration tests or platform-agnostic mocks.

### 2.2 Feature-Gated Test Suites

Several test suites use conditional skipping based on feature flags:

| Flag                    | Affected Tests                                |
| ----------------------- | --------------------------------------------- |
| `SEED17_PARITY_ENABLED` | TraceParity.seed17.firstDivergence.test.ts    |
| `skipWithOrchestrator`  | Multiple GameEngine/ClientSandboxEngine tests |
| `orchestratorEnabled`   | FSM and orchestrator integration tests        |

**Status:** These are intentional - tests enable/disable based on migration progress.

### 2.3 Orchestrator-Skipped Tests (Analysis Dec 11, 2025)

**Finding:** 36 test files contain `skipWithOrchestrator` or `orchestratorEnabled` conditionals.

**Root Cause:** These tests manipulate internal engine state directly (e.g., `engineAny.gameState`), which is incompatible with the orchestrator adapter's unified processing flow.

**Examples of incompatible patterns:**

- Setting `state.currentPhase` directly
- Manipulating `board.stacks` without going through move application
- Testing specific phase transition sequences that differ under orchestrator

**Decision:** Keep these tests skipped. They are:

1. **Legacy diagnostic tests** - Not missing coverage, but tests of old implementation details
2. **Intentionally incompatible** - Explicitly documented as such
3. **Not production code paths** - Orchestrator is canonical (`ORCHESTRATOR_ADAPTER_ENABLED=true`)

**Coverage assurance:** Equivalent functionality is tested via:

- Contract vector tests (234+ vectors)
- Orchestrator-aware scenario tests
- Parity tests between TS and Python

**Files affected (36 total):**

- `tests/unit/Seed*.test.ts` - Seed-specific parity debugging (7 files)
- `tests/unit/TerritoryDecisions.*.test.ts` - Territory parity tests (8 files)
- `tests/unit/ClientSandboxEngine.*.test.ts` - Sandbox internal tests (5 files)
- `tests/unit/GameEngine.*.test.ts` - GameEngine internal tests (4 files)
- `tests/scenarios/FAQ_*.test.ts` - FAQ scenario tests (5 files)
- `tests/integration/*.test.ts` - Integration tests (3 files)
- `tests/parity/*.test.ts` - Parity tests (2 files)
- Other diagnostic tests (2 files)

---

## 3. Test Naming Conventions

### Current Patterns

Victory-related tests use multiple naming patterns:

- `{Engine}.{feature}.test.ts` - Engine-specific tests
- `{feature}.shared.test.ts` - Shared logic tests
- `{Module}.branchCoverage.test.ts` - Branch coverage tests
- `{Feature}.scenarios.test.ts` - Scenario-based tests

### Recommended Standard

```
{Module|Feature}.{type}.test.ts

Types:
- .shared.test.ts - Tests for shared logic (engine-agnostic)
- .branchCoverage.test.ts - Targeted branch coverage tests
- .scenarios.test.ts - Scenario/integration tests
- .test.ts - General unit tests
```

---

## 4. Test File Inventory

As of Dec 11, 2025:

| Directory          | File Count | Notes                                    |
| ------------------ | ---------- | ---------------------------------------- |
| tests/unit/        | 374        | Main test directory (needs organization) |
| tests/scenarios/   | 29         | Scenario-based tests                     |
| tests/parity/      | 8          | Cross-engine parity tests                |
| tests/fixtures/    | ~40        | Test fixtures and helpers                |
| tests/integration/ | 12         | Integration tests                        |
| tests/e2e/         | ~10        | Visual regression tests                  |

---

## Change Log

| Date       | Change                                                            |
| ---------- | ----------------------------------------------------------------- |
| 2025-12-11 | Initial document after victory consolidation cleanup              |
| 2025-12-11 | Added Section 2.3: Analysis of 36 orchestrator-skipped test files |
