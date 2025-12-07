# Orchestrator Migration Completion Plan

> **Created:** 2025-12-01
> **Completed:** 2025-12-01
> **Task:** P20.4-1 – Orchestrator Migration Completion Plan
> **Status:** ✅ COMPLETE (Phase 3)
> **Priority:** P0 – Unblocking for P20.3-5 and P20.3-6

## Executive Summary

The orchestrator migration is **COMPLETE through Phase 3**. All critical legacy code paths have been removed:

- **Phase 1 (Verification):** ✅ Completed 2025-12-01 – 269 orchestrator tests passing
- **Phase 2 (Test Migration):** ✅ Completed 2025-12-01 – 4 tests migrated, 4 retained for debugging
- **Phase 3 (Code Cleanup):** ✅ Completed 2025-12-01 – ~1,118 lines of legacy code removed

**Legacy Code Removed:**

- Feature flag infrastructure (~19 lines)
- ClientSandboxEngine legacy methods (786 lines)
- RuleEngine deprecated methods (~120 lines)
- Obsolete test file (193 lines)

**What Remains (deferred to Phase 4):**

- Tier 2 sandbox support modules (~1,200 lines) – needed for sandbox UX
- 4 parity tests retained for debugging (GameEngine.orchestratorParity.integration.test.ts)
- SSOT banner additions for support modules

---

## 1. Current State Analysis

### 1.1 Orchestrator Rollout Status

| Metric                             | Value / Notes                                   |
| ---------------------------------- | ----------------------------------------------- |
| `ORCHESTRATOR_ADAPTER_ENABLED`     | `true` (hardcoded in `EnvSchema`)               |
| `ORCHESTRATOR_ROLLOUT_PERCENTAGE`  | **Removed** in Phase 3 cleanup (forced to 100%) |
| `ORCHESTRATOR_SHADOW_MODE_ENABLED` | `false` (default)                               |
| `RINGRIFT_RULES_MODE`              | `ts`                                            |
| Environment Phase                  | Phase 4 – Orchestrator Authoritative            |

**Reference:** [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md) lines 1-8

### 1.2 Feature Flag Configuration

From [`src/server/config/env.ts`](../src/server/config/env.ts):

```typescript
ORCHESTRATOR_ADAPTER_ENABLED: z
  .any()
  .transform((): true => true)
  .default(true),

ORCHESTRATOR_SHADOW_MODE_ENABLED: z
  .string()
  .default('false')
  .transform((val) => val === 'true' || val === '1'),
```

### 1.3 Code Elimination Status

| Tier                            | Status     | Lines Removed |
| ------------------------------- | ---------- | ------------- |
| Tier 1: Core Sandbox Engines    | ✅ Deleted | ~1,713        |
| Tier 2: Support Modules         | 🔲 Pending | ~1,200        |
| Tier 3: Backend Rules Directory | ✅ Deleted | ~600          |

**Deleted modules:**

- `sandboxTurnEngine.ts`
- `sandboxMovementEngine.ts`
- `sandboxLinesEngine.ts`
- `sandboxTerritoryEngine.ts`
- `src/server/game/rules/*`

**Remaining Tier 2 sandbox modules:**

- `sandboxMovement.ts` (~70 lines)
- `sandboxCaptures.ts` (~175 lines)
- `sandboxCaptureSearch.ts` (~200 lines)
- `sandboxElimination.ts` (~150 lines)
- `sandboxLines.ts` (~145 lines)
- `sandboxTerritory.ts` (~165 lines)
- `sandboxPlacement.ts` (~220 lines)
- `sandboxVictory.ts` (~120 lines)
- `sandboxGameEnd.ts` (~110 lines)
- `sandboxAI.ts` (~1,345 lines – AI harness, not duplicate rules)
- `localSandboxController.ts` (~265 lines – DIAGNOSTICS-ONLY)

---

## 2. Blocking Items Inventory

### 2.1 Tests Explicitly Disabling Orchestrator

**8 test files** call `disableOrchestratorAdapter()`:

| File                                                                                                                                                | Purpose                               | Migration Strategy                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------- |
| [`GameEngine.orchestratorParity.integration.test.ts`](../tests/unit/GameEngine.orchestratorParity.integration.test.ts:65)                           | Parity testing legacy vs orchestrator | **Keep** – Critical parity verification |
| [`ClientSandboxEngine.chainCapture.getValidMoves.test.ts`](../tests/unit/ClientSandboxEngine.chainCapture.getValidMoves.test.ts:56)                 | Capture enumeration                   | **Migrate** – Use orchestrator adapter  |
| [`ClientSandboxEngine.movementParity.shared.test.ts`](../tests/unit/ClientSandboxEngine.movementParity.shared.test.ts:66)                           | Movement parity                       | **Keep** – Dual-path verification       |
| [`ClientSandboxEngine.placement.shared.test.ts`](../tests/unit/ClientSandboxEngine.placement.shared.test.ts:44)                                     | Placement testing                     | **Migrate** – Use orchestrator          |
| [`GameEngine.utilityMethods.test.ts`](../tests/unit/GameEngine.utilityMethods.test.ts:232)                                                          | Utility method testing                | **Keep** – Tests the toggle itself      |
| [`Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts`](../tests/unit/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts:182)                    | Critical parity                       | **Keep** – Dual-path verification       |
| [`ClientSandboxEngine.orchestratorParity.test.ts`](../tests/unit/ClientSandboxEngine.orchestratorParity.test.ts:64)                                 | Parity testing                        | **Keep** – Critical parity verification |
| [`ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts`](../tests/unit/ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts:75) | Territory decisions                   | **Migrate** – Use orchestrator          |

### 2.2 Tests Importing RuleEngine Directly

**~15 test files** import RuleEngine for validation:

| Category                 | Files   | Migration Strategy                          |
| ------------------------ | ------- | ------------------------------------------- |
| **Parity Tests** (Keep)  | 5 files | RuleEngine validates moves; keep for parity |
| **Unit Tests** (Migrate) | 6 files | Migrate to shared engine helpers            |
| **Historical Scripts**   | 4 files | Archive or update to orchestrator           |

**Parity tests to keep:**

- `PlacementParity.RuleEngine_vs_Sandbox.test.ts`
- `VictoryParity.RuleEngine_vs_Sandbox.test.ts`
- `MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`
- `reachabilityParity.RuleEngine_vs_Sandbox.test.ts`
- `Python_vs_TS.traceParity.test.ts`

**Unit tests to migrate:**

- `movement.shared.test.ts`
- `RuleEngine.movement.scenarios.test.ts`
- `RuleEngine.skipPlacement.shared.test.ts`
- `RuleEngine.placement.shared.test.ts`
- `RuleEngine.placementMultiRing.test.ts`
- `ClientSandboxEngine.moveParity.test.ts`

### 2.3 Deprecated Methods in Production Code

**GameEngine deprecated methods (9):**

| Method                                            | Line | Purpose                |
| ------------------------------------------------- | ---- | ---------------------- |
| `enableMoveDrivenDecisionPhases()`                | 449  | No-op in Phase 4       |
| `disableOrchestratorAdapter()`                    | 487  | Test/diagnostic only   |
| `processLineFormations()`                         | 2351 | Legacy line processing |
| `processOneLine()`                                | 2732 | Legacy line processing |
| `processDisconnectedRegionCore()`                 | 3053 | Legacy territory       |
| `processOneDisconnectedRegion()`                  | 3098 | Legacy territory       |
| `processDisconnectedRegions()`                    | 3135 | Legacy territory       |
| `resolveBlockedStateForCurrentPlayerForTesting()` | 4191 | Test helper            |
| `stepAutomaticPhasesForTesting()`                 | 4416 | Test helper            |

**RuleEngine deprecated methods (4):**

| Method                            | Purpose                 |
| --------------------------------- | ----------------------- |
| `processMove()`                   | Legacy move application |
| `processChainReactions()`         | Legacy capture chains   |
| `processLineFormation()`          | Legacy line processing  |
| `processTerritoryDisconnection()` | Legacy territory        |

### 2.4 ClientSandboxEngine Feature Flag Path

`ClientSandboxEngine` contains:

- `useOrchestratorAdapter` private field (default: `true`)
- `enableOrchestratorAdapter()` / `disableOrchestratorAdapter()` methods
- Legacy `handleLegacyMovementClick()` method marked deprecated
- Fallback paths in movement, capture, and territory processing

---

## 3. Phased Migration Plan

### Phase 1: Verification (1-2 days)

**Objective:** Confirm orchestrator is stable and parity is verified

1. **Confirm orchestrator status:**
   - Verify `ORCHESTRATOR_ADAPTER_ENABLED=true` (hardcoded in env schema)
   - Confirm `ORCHESTRATOR_ROLLOUT_PERCENTAGE` is no longer referenced in env/config (flag removed)
   - Check soak test results for zero invariant violations

2. **Run extended parity verification:**

   ```bash
   npm run test:orchestrator-parity:ts
   ./scripts/run-python-contract-tests.sh --verbose
   npm run soak:orchestrator:short
   ```

3. **Validate no regressions:**
   - All CI gates green
   - No P0/P1 incidents related to orchestrator

**Exit Criteria:**

- All parity and contract tests pass
- Soak tests show `totalInvariantViolations == 0`
- No active orchestrator-related incidents

### Phase 2: Test Migration (3-5 days)

**Objective:** Migrate unit tests to orchestrator-only, retain parity tests

#### 2.1 Migrate Unit Tests (3 days)

| Test File                                                        | Action                                   | Effort |
| ---------------------------------------------------------------- | ---------------------------------------- | ------ |
| `ClientSandboxEngine.chainCapture.getValidMoves.test.ts`         | Remove `disableOrchestratorAdapter()`    | 0.5h   |
| `ClientSandboxEngine.placement.shared.test.ts`                   | Remove `disableOrchestratorAdapter()`    | 0.5h   |
| `ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts` | Remove `disableOrchestratorAdapter()`    | 1h     |
| `movement.shared.test.ts`                                        | Use shared helpers instead of RuleEngine | 2h     |
| `RuleEngine.movement.scenarios.test.ts`                          | Migrate to MovementAggregate             | 2h     |
| `RuleEngine.skipPlacement.shared.test.ts`                        | Migrate to PlacementAggregate            | 1h     |
| `RuleEngine.placement.shared.test.ts`                            | Migrate to PlacementAggregate            | 2h     |
| `RuleEngine.placementMultiRing.test.ts`                          | Migrate to PlacementAggregate            | 1h     |
| `ClientSandboxEngine.moveParity.test.ts`                         | Use orchestrator for both paths          | 2h     |

**Subtotal:** ~12 hours

#### 2.2 Retain Critical Parity Tests (1 day)

Mark these tests with explicit comments explaining dual-path necessity:

- `GameEngine.orchestratorParity.integration.test.ts` – Verifies legacy vs orchestrator
- `ClientSandboxEngine.orchestratorParity.test.ts` – Verifies sandbox paths
- `Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` – Cross-host parity
- `ClientSandboxEngine.movementParity.shared.test.ts` – Movement parity

#### 2.3 Archive Legacy Stubs (0.5 day)

Move to `archive/tests/` or update headers:

- `RulesMatrix.Comprehensive.test.ts` (already trimmed)
- `LineAndTerritory.test.ts` (already archived stub)
- `ExportLineAndTerritorySnapshot.test.ts` (already trimmed)
- `GameEngine.lineRewardChoiceWebSocketIntegration.test.ts` (already archived stub)
- `GameEngine.lineRewardChoiceAIService.integration.test.ts` (already archived stub)

### Phase 3: Code Cleanup (2-3 days)

**Objective:** Remove deprecated code paths and hardcode orchestrator

#### 3.1 Hardcode Orchestrator in ClientSandboxEngine (1 day)

1. Remove `disableOrchestratorAdapter()` method
2. Remove `useOrchestratorAdapter` field
3. Remove `enableOrchestratorAdapter()` method
4. Delete `handleLegacyMovementClick()` method
5. Simplify `getValidMoves()` to orchestrator-only path
6. Update constructor to always use orchestrator

**Impact:** ~200 lines removed from ClientSandboxEngine

#### 3.2 Hardcode Orchestrator in GameEngine (0.5 day)

1. Remove `disableOrchestratorAdapter()` method
2. Remove `enableOrchestratorAdapter()` method (keep for compatibility)
3. Remove `useOrchestratorAdapter` conditional in `makeMove()`
4. Simplify legacy path to throw error if accessed

**Impact:** ~100 lines simplified

#### 3.3 Remove Feature Flag Infrastructure (0.5 day)

Update `src/server/config/env.ts`:

- Keep `ORCHESTRATOR_ADAPTER_ENABLED` but document as always true
- Remove from rollout percentage logic
- Update `OrchestratorRolloutService` to hardcode 100%

#### 3.4 Remove Deprecated Methods (1 day)

**GameEngine methods to remove:**

- `processLineFormations()`
- `processOneLine()`
- `processDisconnectedRegionCore()`
- `processOneDisconnectedRegion()`
- `processDisconnectedRegions()`
- `resolveBlockedStateForCurrentPlayerForTesting()`
- `stepAutomaticPhasesForTesting()`

**RuleEngine methods to remove:**

- `processMove()`
- `processChainReactions()`
- `processLineFormation()`
- `processTerritoryDisconnection()`

### Phase 4: Tier 2 Sandbox Cleanup (1-2 days)

**Objective:** Remove redundant sandbox helper modules

#### 4.1 Analysis: Module Dependencies

| Module                      | Used By                        | Removable?                    |
| --------------------------- | ------------------------------ | ----------------------------- |
| `sandboxMovement.ts`        | ClientSandboxEngine, sandboxAI | No – still needed             |
| `sandboxCaptures.ts`        | ClientSandboxEngine, tests     | Partial – diagnostics only    |
| `sandboxCaptureSearch.ts`   | Tests only                     | Yes – DIAGNOSTICS-ONLY        |
| `sandboxElimination.ts`     | ClientSandboxEngine            | No – elimination logic        |
| `sandboxLines.ts`           | ClientSandboxEngine            | No – line detection           |
| `sandboxTerritory.ts`       | ClientSandboxEngine            | No – territory detection      |
| `sandboxPlacement.ts`       | ClientSandboxEngine, sandboxAI | No – placement logic          |
| `sandboxVictory.ts`         | ClientSandboxEngine            | No – victory checking         |
| `sandboxGameEnd.ts`         | ClientSandboxEngine            | No – game end handling        |
| `sandboxAI.ts`              | ClientSandboxEngine            | No – sandbox AI harness       |
| `localSandboxController.ts` | Tests only                     | Yes – DIAGNOSTICS-ONLY legacy |

#### 4.2 Modules to Remove/Archive

1. **`sandboxCaptureSearch.ts`** – Move to diagnostics namespace
2. **`localSandboxController.ts`** – Archive (already DIAGNOSTICS-ONLY)

#### 4.3 Modules to Keep (with SSOT banners)

Add explicit SSOT banners declaring these as:

- UX/diagnostics adapters over shared engine
- Not alternative rule implementations

---

## 4. Task Breakdown with Effort Estimates

### Summary Table

| Phase       | Task                                 | Effort        | Dependencies |
| ----------- | ------------------------------------ | ------------- | ------------ |
| **Phase 1** | Verification                         | 1-2 days      | None         |
| 1.1         | Confirm orchestrator status          | 0.5 day       |              |
| 1.2         | Run extended parity verification     | 0.5 day       |              |
| 1.3         | Validate no regressions              | 0.5 day       |              |
| **Phase 2** | Test Migration                       | 3-5 days      | Phase 1      |
| 2.1         | Migrate unit tests (~9 files)        | 3 days        |              |
| 2.2         | Document retained parity tests       | 0.5 day       |              |
| 2.3         | Archive legacy stubs                 | 0.5 day       |              |
| **Phase 3** | Code Cleanup                         | 2-3 days      | Phase 2      |
| 3.1         | Hardcode ClientSandboxEngine         | 1 day         |              |
| 3.2         | Hardcode GameEngine                  | 0.5 day       |              |
| 3.3         | Remove feature flag infrastructure   | 0.5 day       |              |
| 3.4         | Remove deprecated methods            | 1 day         |              |
| **Phase 4** | Tier 2 Cleanup                       | 1-2 days      | Phase 3      |
| 4.1         | Remove/archive sandboxCaptureSearch  | 0.5 day       |              |
| 4.2         | Archive localSandboxController       | 0.5 day       |              |
| 4.3         | Add SSOT banners to retained modules | 0.5 day       |              |
| **Total**   |                                      | **7-12 days** |              |

### Detailed Task List

#### Phase 1: Verification ✅ COMPLETE (2025-12-01)

- [x] P20.5-1: Verify orchestrator env flags in all environments
- [x] P20.5-1: Run `npm run test:orchestrator-parity:ts` (269 tests passed)
- [x] P20.5-1: Confirm no active orchestrator incidents

#### Phase 2: Test Migration ✅ COMPLETE (2025-12-01)

- [x] P20.6-1: Assessment of 8 test files complete
- [x] P20.6-1: Migrated 4 tests to orchestrator-only paths
- [x] P20.6-1: Retained 4 parity tests for debugging purposes
- [x] P20.6-1: Documented retention rationale in test files

#### Phase 3: Code Cleanup ✅ COMPLETE (2025-12-01)

- [x] P20.7-1: Hardcoded `ORCHESTRATOR_ADAPTER_ENABLED=true` (4 files)
- [x] P20.7-2: Removed feature flag infrastructure (~19 lines)
- [x] P20.7-3: Assessment of deprecated methods complete
- [x] P20.7-4: Removed ClientSandboxEngine legacy methods (786 lines)
- [x] P20.7-5: Deleted obsolete test file (193 lines)

#### Phase 4: Tier 2 Cleanup (DEFERRED to post-MVP)

- [ ] P20.4-4.1: Archive `sandboxCaptureSearch.ts` to diagnostics namespace
- [ ] P20.4-4.2: Archive `localSandboxController.ts`
- [ ] P20.4-4.3: Add SSOT banners to `sandboxMovement.ts`
- [ ] P20.4-4.4: Add SSOT banners to `sandboxCaptures.ts`
- [ ] P20.4-4.5: Add SSOT banners to `sandboxElimination.ts`
- [ ] P20.4-4.6: Add SSOT banners to `sandboxLines.ts`
- [ ] P20.4-4.7: Add SSOT banners to `sandboxTerritory.ts`
- [ ] P20.4-4.8: Add SSOT banners to `sandboxPlacement.ts`
- [ ] P20.4-4.9: Add SSOT banners to `sandboxVictory.ts`
- [ ] P20.4-4.10: Add SSOT banners to `sandboxGameEnd.ts`
- [ ] P20.4-4.11: Update `scripts/ssot/rules-ssot-check.ts`

**Note:** Phase 4 tasks are deferred to post-MVP. The Tier 2 sandbox modules are still needed for sandbox UX functionality and will be addressed in a future cleanup pass.

---

## 5. Risk Assessment

### 5.1 High Risks

| Risk                           | Impact              | Likelihood | Mitigation                                          |
| ------------------------------ | ------------------- | ---------- | --------------------------------------------------- |
| **Parity regression**          | Breaking game logic | Medium     | Keep 4 critical parity tests until Phase 4 complete |
| **Test failures from removal** | CI breakage         | High       | Migrate tests before removing code                  |
| **Hidden legacy path usage**   | Runtime errors      | Medium     | Search codebase for all callers before removal      |

### 5.2 Medium Risks

| Risk                         | Impact             | Likelihood | Mitigation                     |
| ---------------------------- | ------------------ | ---------- | ------------------------------ |
| **Python parity divergence** | AI training issues | Low        | Contract tests verify parity   |
| **Sandbox AI regression**    | Broken local games | Medium     | Comprehensive AI harness tests |
| **Documentation drift**      | Confusion          | Medium     | Update docs in same PR as code |

### 5.3 Low Risks

| Risk                        | Impact            | Likelihood | Mitigation                     |
| --------------------------- | ----------------- | ---------- | ------------------------------ |
| **Third-party integration** | External breakage | Low        | No known external consumers    |
| **Performance regression**  | Slower games      | Very Low   | Orchestrator already optimized |

### 5.4 Rollback Strategy

**During Phase 2-4:**

- Each phase is a separate PR for easy revert
- Feature flags remain available until Phase 3.3
- Legacy code retained until Phase 3.4

**If issues arise after Phase 4:**

- Revert to previous release via deployment rollback
- Re-enable feature flags if needed
- Git revert specific Phase 3-4 commits

---

## 6. Success Criteria

### Phase Completion Criteria

| Phase   | Criteria                                                                   |
| ------- | -------------------------------------------------------------------------- |
| Phase 1 | All parity tests pass, soak shows zero violations                          |
| Phase 2 | All migrated tests pass, no new test failures                              |
| Phase 3 | No `disableOrchestratorAdapter()` calls remain, deprecated methods removed |
| Phase 4 | SSOT check passes, diagnostics modules properly fenced                     |

### Overall Migration Complete When

1. ✅ No test files call `disableOrchestratorAdapter()` (except 4 retained parity tests)
2. ✅ No production code contains `useOrchestratorAdapter` conditional – **DONE (P20.7-4)**
3. ✅ Deprecated GameEngine/RuleEngine methods removed – **DONE (P20.7-4, P20.3-\*)**
4. ✅ Feature flag infrastructure simplified – **DONE (P20.7-1, P20.7-2)**
5. ⬜ Tier 2 sandbox modules have SSOT banners or are archived – **DEFERRED to Phase 4**
6. ✅ All CI gates pass – **2,987 tests passing**
7. ✅ No regressions in Python parity – **49/49 contract vectors passing**
8. ✅ Documentation updated – **PASS20 completion summary created**

---

## 8. PASS20 Completion Summary

### Phase 3 Completion (2025-12-01)

| Task    | Description                                 | Lines Removed   |
| ------- | ------------------------------------------- | --------------- |
| P20.7-1 | Hardcoded ORCHESTRATOR_ADAPTER_ENABLED=true | 4 files updated |
| P20.7-2 | Removed feature flag infrastructure         | ~19 lines       |
| P20.7-3 | Deprecated methods assessment               | Complete        |
| P20.7-4 | ClientSandboxEngine legacy methods          | 786 lines       |
| P20.7-5 | Deleted obsolete test file                  | 193 lines       |

### Total Legacy Code Removed in PASS20

| Category                           | Lines      |
| ---------------------------------- | ---------- |
| RuleEngine deprecated methods      | ~120       |
| Feature flag infrastructure        | ~19        |
| ClientSandboxEngine legacy methods | 786        |
| Obsolete test file                 | 193        |
| **Total**                          | **~1,118** |

### What Was Kept (and Why)

1. **GameEngine.orchestratorParity.integration.test.ts** – Critical for debugging parity issues between hosts
2. **4 parity test files** – Dual-path verification capability retained for edge case debugging
3. **Tier 2 sandbox modules** – Still needed for sandbox UX functionality (~1,200 lines)

---

## 7. References

- [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md) – Phase 4 status and SLOs
- [`docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md`](./drafts/LEGACY_CODE_ELIMINATION_PLAN.md) – Historical context
- [`LEGACY_CODE_DEPRECATION_REPORT.md`](../LEGACY_CODE_DEPRECATION_REPORT.md) – Deprecation audit
- [`src/server/config/env.ts`](../src/server/config/env.ts) – Feature flag configuration
- [`src/client/sandbox/ClientSandboxEngine.ts`](../src/client/sandbox/ClientSandboxEngine.ts) – Sandbox host
- [`src/server/game/GameEngine.ts`](../src/server/game/GameEngine.ts) – Backend host
