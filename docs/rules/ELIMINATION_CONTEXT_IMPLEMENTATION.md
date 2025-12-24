# Elimination Context Implementation

> **Status:** ✅ Implementation Complete, Validation Complete
> **Last Updated:** 2025-12-11
> **Canonical Rules:** RR-CANON-R022, RR-CANON-R122, RR-CANON-R145, RR-CANON-R100

This document tracks the implementation and validation of the elimination context distinction between line processing and territory processing, as defined in the canonical rules.

---

## Overview

The elimination cost for processing lines vs territories differs:

| Context                  | Cost            | Eligible Stacks                              | Canonical Reference |
| ------------------------ | --------------- | -------------------------------------------- | ------------------- |
| **Line Processing**      | 1 ring from top | Any controlled stack (including height-1)    | RR-CANON-R122       |
| **Territory Processing** | Entire cap      | Multicolor stacks OR single-color height > 1 | RR-CANON-R145       |
| **Forced Elimination**   | Entire cap      | Any controlled stack (including height-1)    | RR-CANON-R100       |
| **Recovery Move**        | 1 buried ring   | Stacks with buried rings                     | RR-CANON-R082       |

### Key Definitions (RR-CANON-R022)

- **Cap**: All consecutive top rings of the controlling player's color
- **Cap Height**: Number of rings in the cap
- **Controlling Player**: Owner of the top ring

---

## Implementation Status

### TypeScript Implementation

| Component                | File                                                     | Status      | Notes                                                    |
| ------------------------ | -------------------------------------------------------- | ----------- | -------------------------------------------------------- |
| Type Definition          | `src/shared/engine/types.ts:114`                         | ✅ Complete | `eliminationContext?: 'line' \| 'territory' \| 'forced'` |
| TerritoryMutator         | `src/shared/engine/mutators/TerritoryMutator.ts:116-120` | ✅ Complete | Checks context, eliminates 1 for line                    |
| TerritoryAggregate       | `src/shared/engine/aggregates/TerritoryAggregate.ts`     | ✅ Complete | Same logic as mutator                                    |
| sandboxElimination       | `src/client/sandbox/sandboxElimination.ts:79-104`        | ✅ Complete | Accepts `eliminationContext` parameter                   |
| ClientSandboxEngine      | `src/client/sandbox/ClientSandboxEngine.ts`              | ✅ Complete | Renamed to `eliminateRingForLineReward`, passes `'line'` |
| territoryDecisionHelpers | `src/shared/engine/territoryDecisionHelpers.ts:603-607`  | ✅ Complete | Already had correct logic                                |

### Python Implementation

| Component  | File                                               | Status      | Notes                     |
| ---------- | -------------------------------------------------- | ----------- | ------------------------- |
| GameEngine | `ai-service/app/game_engine/__init__.py:4003-4008` | ✅ Complete | Already had correct logic |

---

## Test Coverage

### Existing Tests (All Passing)

| Test File                                                       | Coverage                      | Tests      |
| --------------------------------------------------------------- | ----------------------------- | ---------- |
| `tests/unit/territoryDecisionHelpers.shared.test.ts`            | Line vs Territory distinction | 12 tests   |
| `tests/unit/sandboxElimination.test.ts`                         | forceEliminateCapOnBoard      | 10 tests   |
| `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` | Combined line + territory     | 4 tests    |
| `tests/unit/LineAggregate.shared.test.ts`                       | Line processing               | 50+ tests  |
| `tests/unit/TerritoryAggregate.*.test.ts`                       | Territory processing          | 100+ tests |

### Specific Edge Case Tests

From `territoryDecisionHelpers.shared.test.ts`:

- `line elimination context includes height-1 standalone rings as eligible targets`
- `line elimination removes exactly ONE ring, not entire cap`
- `territory elimination removes entire cap`
- `line elimination on multicolor stack removes exactly 1 ring from cap`
- `territory elimination on multicolor stack removes entire cap exposing buried rings`
- `forced elimination context allows any stack including height-1`
- `forced elimination removes entire cap like territory elimination`
- `default elimination context (undefined) behaves like territory elimination`

---

## Validation Checklist

### All Completed

- [x] TypeScript unit tests pass (592+ tests, 44 suites)
- [x] Python unit tests pass (108+ tests)
- [x] TS/Python parity validation on selfplay replays (0 FSM failures)
- [x] Fresh selfplay games complete without errors
- [x] Test expectations updated for new elimination counts
- [x] UI/UX audit - elimination selection shows `ringsToEliminate` with context-aware prompts
- [x] Extended parity test with line+territory in same turn
- [x] Performance profile of move enumeration (0.05ms avg per call)
- [x] EliminationAggregate operations < 0.002ms per call
- [x] Training data generation validates (2p, 3p games)
- [x] Branch coverage tests pass (2572 tests)
- [x] Victory threshold test fixed (RR-CANON-R061 formula correction)

---

## Architectural Notes

### Design Decision: `eliminationContext` Field

The `eliminationContext` field was added to `EliminateStackAction` to explicitly communicate the elimination type through the move pipeline. This approach:

**Pros:**

- Explicit over implicit - the context is preserved through serialization
- Debuggable - can trace exactly which elimination type was requested
- Extensible - can add new contexts (e.g., 'recovery') if needed

**Cons:**

- Requires all callers to set the context correctly
- Default behavior (undefined → 'territory') may mask bugs

**Alternative Considered:**

- Inferring context from game phase - rejected because phase alone doesn't always determine context (e.g., recovery moves during various phases)

### Code Locations

The elimination logic is duplicated in several places for historical reasons:

1. `TerritoryMutator.mutateEliminateStack` - used by server-side engine
2. `TerritoryAggregate.mutateEliminateStack` - used by aggregate pattern
3. `sandboxElimination.forceEliminateCapOnBoard` - used by client sandbox
4. `territoryDecisionHelpers.applyEliminateRingsFromStackDecision` - used by decision enumeration

**Refactoring Opportunity:** Consider consolidating these into a single source of truth. The `sandboxElimination` module could be promoted to shared code and used by all paths.

### Refactoring Recommendations

Based on analysis of the codebase, the following refactoring opportunities would improve long-term maintainability:

#### 1. Consolidate Elimination Logic (Priority: High)

**Current State:** Elimination logic exists in 5 locations:

- `EliminationAggregate.eliminateFromStack` (canonical source)
- `TerritoryMutator.mutateEliminateStack`
- `TerritoryAggregate.mutateEliminateStack`
- `sandboxElimination.forceEliminateCapOnBoard`
- `territoryDecisionHelpers.applyEliminateRingsFromStackDecision`

**Recommendation:** Refactor all callers to use `EliminationAggregate.eliminateFromStack` as the single source of truth. The aggregate pattern is already the canonical implementation with proper context handling.

#### 2. Simplify EliminationContext Flow (Priority: Medium)

**Current State:** The `eliminationContext` is passed through multiple layers and sometimes needs to be inferred from phase.

**Recommendation:** Consider creating an `EliminationRequest` interface that bundles all elimination parameters:

```typescript
interface EliminationRequest {
  position: Position;
  player: number;
  context: EliminationContext;
  claimedRegion?: Position[]; // For territory context
}
```

#### 3. Remove Duplicate Cap Calculation (Priority: Low)

**Current State:** Cap height is calculated in multiple places using slightly different approaches.

**Recommendation:** Always use `calculateCapHeight` from `EliminationAggregate` and ensure it's exported and used consistently.

#### 4. Victory Threshold Formula Consistency (Priority: Medium)

**Current State:** Victory threshold formula (RR-CANON-R061) is implemented in `initialState.ts` but some tests had outdated expectations.

**Recommendation:**

- Add a helper function `calculateVictoryThreshold(ringsPerPlayer, numPlayers)` to a shared location
- Add unit tests specifically for the threshold formula
- Document the formula change prominently

---

## Related Documentation

- [RULES_CANONICAL_SPEC.md](../../RULES_CANONICAL_SPEC.md) - Authoritative rules
- [COMPLETE_RULES.md]COMPLETE_RULES.md) - Full rulebook
- [INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md) - Parity invariants
- [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md) - Rules to code mapping

---

## Changelog

### 2025-12-11 (Validation Complete)

- ✅ Full test suite passes (592 tests, 44 suites)
- ✅ Branch coverage tests pass (2572 tests, 66 suites)
- ✅ Performance validated:
  - `getValidMoves`: 0.05ms average
  - `isStackEligibleForElimination`: 0.00006ms average
  - `getRingsToEliminate`: 0.00002ms average
  - `eliminateFromStack`: 0.002ms average
- ✅ Parity validation: 7 selfplay games replayed with 0 FSM failures
- ✅ Training data generation: 2p and 3p games validated
- ✅ UI/UX: Added `ringsToEliminate` to `RingEliminationChoice` with context-aware prompts
- ✅ Fixed victory threshold test (RR-CANON-R061: 18 for 2p, not 19)

### 2025-12-11 (Initial)

- Initial implementation of `eliminationContext` field
- Updated TerritoryMutator, TerritoryAggregate, sandboxElimination
- Renamed `eliminateCapForLineReward` → `eliminateRingForLineReward`
- Updated test expectations in territoryDisconnection test
- Verified TS/Python parity on selfplay replays
- Created this tracking document
