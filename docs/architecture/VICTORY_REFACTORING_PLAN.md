# Victory Detection Refactoring Plan

> **Doc Status (2025-12-11): Complete - All Phases Done**
>
> This document outlines a refactoring plan to consolidate victory detection logic and reduce code duplication.

**Created:** 2025-12-11
**Related Documents:**

- [AI_PIPELINE_PARITY_FIXES.md](../ai/AI_PIPELINE_PARITY_FIXES.md)
- [PYTHON_PARITY_REQUIREMENTS.md](../rules/PYTHON_PARITY_REQUIREMENTS.md)
- [MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md)

---

## Executive Summary

The codebase currently has **duplicate victory detection implementations** that increase maintenance burden:

| File                  | Lines | Status                             |
| --------------------- | ----- | ---------------------------------- |
| `victoryLogic.ts`     | 418   | **DUPLICATE** - Should be retired  |
| `VictoryAggregate.ts` | 770   | **PRIMARY** - Keep and consolidate |

**Recommendation:** Retire `victoryLogic.ts` and consolidate all victory logic into `VictoryAggregate.ts`.

---

## 1. Current Architecture

### 1.1 Duplicate Functions

| Function                            | victoryLogic.ts | VictoryAggregate.ts |
| ----------------------------------- | --------------- | ------------------- |
| `countTotalRingsForPlayer()`        | Lines 11-25     | Lines 43-57         |
| `evaluateVictory()`                 | Lines 68-257    | Lines 301-489       |
| `getLastActor()`                    | Lines 267-294   | Lines 637-664       |
| `hasAnyLegalPlacementOnBareBoard()` | Lines 302-382   | Lines 130-210       |
| `forEachBoardPosition()`            | Lines 384-404   | Lines 215-235       |
| `isValidBoardPosition()`            | Lines 406-418   | Lines 240-252       |

### 1.2 Export Structure

Current exports in `src/shared/engine/index.ts`:

- Line 138: Legacy exports from `victoryLogic.ts`
- Lines 273, 276: Re-exports of victory functions
- Lines 510-532: Aggregate exports with `*Aggregate` suffix

---

## 2. Proposed Changes

### 2.1 Phase 1: Consolidate Exports (Low Risk)

**Goal:** Route all imports through VictoryAggregate while maintaining backward compatibility.

**Changes to `src/shared/engine/index.ts`:**

```typescript
// Remove these lines:
// export { evaluateVictory, getLastActor } from './victoryLogic';

// Add these lines (maintain same export names):
export {
  evaluateVictory,
  getLastActor,
  countTotalRingsForPlayer,
} from './aggregates/VictoryAggregate';
```

**Verification:**

- Run `npm test` to ensure all imports still work
- Run contract vector tests
- Run parity tests

### 2.2 Phase 2: Update Direct Imports (Medium Risk)

**Goal:** Update any files that directly import from `victoryLogic.ts`.

**Files to check:**

```bash
grep -r "from.*victoryLogic" src/
```

**Expected files:**

- `scripts/selfplay-db-ts-replay.ts` - Update import
- Any test files importing directly

### 2.3 Phase 3: Deprecate victoryLogic.ts (Low Risk)

**Goal:** Add deprecation notice to `victoryLogic.ts`.

**Add to top of file:**

```typescript
/**
 * @deprecated This file is deprecated. Import from VictoryAggregate instead.
 * All functions in this file are duplicates of VictoryAggregate.ts.
 * This file is retained for reference only and will be removed in a future release.
 */
```

### 2.4 Phase 4: Remove victoryLogic.ts (After Verification)

**Goal:** Delete the deprecated file once all consumers are migrated.

**Prerequisites:**

- All tests pass with consolidated imports
- No direct imports of victoryLogic.ts remain
- Parity tests confirm no behavioral changes

---

## 3. LPS Tracking Integration

### 3.1 Current Separation

LPS (Last-Player-Standing) tracking is currently separate from victory evaluation:

- `lpsTracking.ts` - Round-based LPS tracking
- `evaluateLpsVictory()` - Called separately from `evaluateVictory()`

### 3.2 Recommendation

**Keep separate** - The round-based LPS tracking requires state that persists across turns, which is different from the stateless `evaluateVictory()` function. The current separation is intentional.

**Document the separation** by adding comments explaining why LPS tracking is separate.

---

## 4. Replay Infrastructure Improvements

### 4.1 Extract Common Victory Evaluation

**Current:** Victory evaluation is duplicated in `selfplay-db-ts-replay.ts`:

- Lines 1024-1046: First check
- Lines 1200-1230: Redundant check

**Proposed:** Extract to shared helper:

```typescript
// New function in turnOrchestrator.ts or victoryHelpers.ts
function evaluateGameVictoryWithLps(
  state: GameState,
  lpsState: LpsTrackingState
): VictoryEvaluation {
  const basicVictory = evaluateVictory(state);
  if (basicVictory.isGameOver) return basicVictory;

  const lpsVictory = evaluateLpsVictory(state, lpsState);
  if (lpsVictory.isGameOver) return lpsVictory;

  return { isGameOver: false };
}
```

### 4.2 Consolidate State Summary Creation

**Status:** Already consolidated. The replay script has a single `summarizeState()` function
(line 621) that is used consistently throughout the file for all state summary needs.
No further consolidation needed.

---

## 5. Priority and Timeline

### High Priority (Immediate)

- [x] Early LPS victory fix (Done Dec 11, 2025)
- [x] Territory victory threshold fix (Done Dec 11, 2025)
- [x] Phase 1: Consolidate exports (Done Dec 11, 2025)

### Medium Priority (Next Sprint)

- [x] Phase 2: Update direct imports (Done Dec 11, 2025)
- [x] Phase 3: Deprecate victoryLogic.ts (Done Dec 11, 2025)
- [x] Extract common victory evaluation in replay (Done Dec 11, 2025)

### Low Priority (Future)

- [x] Phase 4: Remove victoryLogic.ts (Done Dec 11, 2025)
- [x] Consolidate state summary creation (Already done - single summarizeState() function exists)
- [x] Document LPS tracking separation (Done Dec 11, 2025)

---

## 6. Validation Checklist

Before completing each phase:

- [ ] All Jest tests pass (`npm test`)
- [ ] Contract vector tests pass (48 vectors)
- [ ] Parity tests pass (`check_ts_python_replay_parity.py`)
- [ ] Replay scenarios pass (golden games)
- [ ] LPS tracking tests pass
- [ ] No TypeScript compilation errors
- [ ] No direct imports of deprecated files remain

---

## 7. Risk Assessment

| Phase                        | Risk Level | Mitigation                            |
| ---------------------------- | ---------- | ------------------------------------- |
| Phase 1: Consolidate exports | Low        | Backward-compatible re-exports        |
| Phase 2: Update imports      | Medium     | Search-and-replace, test verification |
| Phase 3: Deprecate           | Low        | Documentation only                    |
| Phase 4: Remove              | Medium     | Full test suite verification          |

---

## 8. Change Log

| Date       | Change                                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| 2025-12-11 | Initial plan created after parity fixes                                                                                 |
| 2025-12-11 | **Phase 1 Complete**: Updated `src/shared/engine/index.ts` to export from VictoryAggregate                              |
| 2025-12-11 | **Phase 2 Complete**: Updated 4 test files to use consolidated imports                                                  |
| 2025-12-11 | **Phase 3 Complete**: Added deprecation notice to `victoryLogic.ts`                                                     |
| 2025-12-11 | All victory-related tests passing (15 test suites)                                                                      |
| 2025-12-11 | **Phase 4 Complete**: Removed `victoryLogic.ts` entirely                                                                |
| 2025-12-11 | Updated documentation references across active docs                                                                     |
| 2025-12-11 | All 389 victory tests passing after removal                                                                             |
| 2025-12-11 | **Test Cleanup**: Renamed `victoryLogic.branchCoverage.test.ts` → `victory.evaluateVictory.branchCoverage.test.ts`      |
| 2025-12-11 | **Replay Helper**: Added `evaluateVictoryWithLps()` to consolidate victory+LPS evaluation in `selfplay-db-ts-replay.ts` |
| 2025-12-11 | **LPS Documentation**: Added architectural notes to VictoryAggregate.ts and lpsTracking.ts explaining separation        |
| 2025-12-11 | **State Summary**: Verified single `summarizeState()` function already exists - no consolidation needed                 |
| 2025-12-11 | **All tasks complete**: Victory refactoring plan fully implemented                                                      |
