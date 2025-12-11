# Architectural Debt Assessment

**Created**: 2025-12-11
**Last Updated**: 2025-12-11
**Status**: Active - Tracking Progress

## Overview

This document tracks architectural debt identified in the RingRift codebase and progress toward resolution. The goal is to establish a sound long-term architecture that is simple, debuggable, and maintainable.

## Guiding Principles

1. **Single Source of Truth**: Each concept should be implemented in exactly one place
2. **Explicit Over Implicit**: Contexts, errors, and state transitions should be explicit
3. **Parity**: TypeScript and Python implementations should have identical semantics
4. **Layered Responsibility**: Clear separation between validation, mutation, and orchestration

## Priority Matrix

| Priority | Area                           | Status        | Impact   | Effort |
| -------- | ------------------------------ | ------------- | -------- | ------ |
| P1       | Deprecated Phase Orchestrators | Deferred      | Critical | High   |
| P2       | Action Availability Predicates | Complete ✅   | High     | Medium |
| P3       | Cap Height Consolidation       | Complete ✅   | Medium   | Low    |
| P4       | Validation Result Unification  | Documented    | Medium   | High   |
| P5       | Sandbox Aggregate Delegation   | Complete ✅   | Medium   | Low    |
| P6       | Dead Code Cleanup              | Blocked by P1 | Low      | Low    |

---

## P1: Deprecated Phase Orchestrators

### Problem

Two competing phase transition systems exist:

1. **Legacy**: `phaseStateMachine.ts` + `phase_machine.py` (deprecated but still used)
2. **Canonical**: `TurnStateMachine.ts` + FSM (the intended source of truth)

Both marked @deprecated but still in production paths.

### Files Involved

| File                                                   | Lines | Status              |
| ------------------------------------------------------ | ----- | ------------------- |
| `src/shared/engine/orchestration/phaseStateMachine.ts` | 447   | DEPRECATED - Remove |
| `ai-service/app/rules/phase_machine.py`                | 380   | DEPRECATED - Remove |
| `src/shared/engine/fsm/TurnStateMachine.ts`            | 500+  | CANONICAL           |
| `src/shared/engine/fsm/FSMAdapter.ts`                  | ~300  | CANONICAL           |
| `ai-service/app/rules/fsm.py`                          | 865   | EXPERIMENTAL        |

### Resolution Plan

1. ~~Verify FSM handles all phase transitions currently in phaseStateMachine~~
2. Update turnOrchestrator to route ALL phase logic through FSMAdapter
3. Remove phaseStateMachine.ts exports from engine/index.ts
4. Delete phaseStateMachine.ts
5. Mark phase_machine.py for removal (Python should use FSM equivalent)

### Progress

- [x] FSM exists and is marked canonical
- [ ] turnOrchestrator fully migrated to FSM
- [ ] phaseStateMachine.ts deleted
- [ ] phase_machine.py migration started

---

## P2: Action Availability Predicates

### Problem

Three different implementations of "has any action?" predicates:

- Backend TurnEngine
- Client ClientSandboxEngine
- Python GameEngine

### Files Involved

| File                                        | Function                   | Status                   |
| ------------------------------------------- | -------------------------- | ------------------------ |
| `src/shared/engine/turnDelegateHelpers.ts`  | `hasAnyMovementForPlayer`  | ✅ CANONICAL (TS)        |
| `src/shared/engine/turnDelegateHelpers.ts`  | `hasAnyCaptureForPlayer`   | ✅ CANONICAL (TS)        |
| `src/shared/engine/turnDelegateHelpers.ts`  | `hasAnyPlacementForPlayer` | ✅ CANONICAL (TS)        |
| `src/server/game/turn/TurnEngine.ts`        | `hasValidMovements`        | ✅ REMOVED - uses shared |
| `src/client/sandbox/ClientSandboxEngine.ts` | Various                    | ✅ WIRED - uses shared   |
| `ai-service/app/game_engine.py`             | `_has_valid_placements`    | ✅ PARITY (Python)       |
| `ai-service/app/game_engine.py`             | `_has_valid_movements`     | ✅ PARITY (Python)       |
| `ai-service/app/game_engine.py`             | `_has_valid_captures`      | ✅ PARITY (Python)       |

### Resolution Plan

1. ~~Implement `hasAnyMovementForPlayer` in turnDelegateHelpers.ts~~
2. ~~Implement `hasAnyCaptureForPlayer` in turnDelegateHelpers.ts~~
3. ~~Implement `hasAnyPlacementForPlayer` in turnDelegateHelpers.ts~~
4. Wire TurnEngine to use shared predicates
5. Wire ClientSandboxEngine to use shared predicates
6. Wire Python GameEngine to use equivalent logic

### Progress

- [x] turnDelegateHelpers.ts predicates implemented (2025-12-11)
  - `hasAnyPlacementForPlayer` - uses validatePlacementOnBoard
  - `hasAnyMovementForPlayer` - delegates to enumerateSimpleMovesForPlayer
  - `hasAnyCaptureForPlayer` - delegates to enumerateAllCaptureMoves
  - All three respect mustMoveFromStackKey constraints from PerTurnState
  - 35 unit tests passing
- [x] TurnEngine migrated (2025-12-11)
  - Removed local `hasValidPlacements`, `hasValidMovements`, `hasValidCaptures`
  - `advanceGameForCurrentPlayer` delegates now use shared predicates
  - `hasValidActions` uses shared predicates (recovery checked separately)
  - 254 GameEngine/TurnEngine tests passing
- [x] ClientSandboxEngine migrated (2025-12-11)
  - `createTurnLogicDelegates()` now uses shared predicates
  - All ClientSandboxEngine unit tests passing
- [x] Python parity verified (2025-12-11)
  - `GameEngine._has_valid_placements` - mirrors hasAnyPlacementForPlayer
  - `GameEngine._has_valid_movements` - mirrors hasAnyMovementForPlayer (with ignore_must_move_key flag)
  - `GameEngine._has_valid_captures` - mirrors hasAnyCaptureForPlayer
  - All respect mustMoveFromStackKey constraint (Python: `game_state.must_move_from_stack_key`)
  - Parity validated via selfplay and integration tests

### Python Parity Notes

The Python predicates are implemented as static methods on `GameEngine` rather than standalone functions. This is an acceptable pattern difference that doesn't affect parity.

**Semantic Equivalence:**

| TypeScript                         | Python                                   | Notes                             |
| ---------------------------------- | ---------------------------------------- | --------------------------------- |
| `hasAnyPlacementForPlayer(s, p)`   | `GameEngine._has_valid_placements(s, p)` | Both use no-dead-placement rule   |
| `hasAnyMovementForPlayer(s, p, t)` | `GameEngine._has_valid_movements(s, p)`  | Both respect mustMoveFromStackKey |
| `hasAnyCaptureForPlayer(s, p, t)`  | `GameEngine._has_valid_captures(s, p)`   | Both respect mustMoveFromStackKey |

**Key Implementation Details:**

- TS passes `PerTurnState` to access `mustMoveFromStackKey`
- Python reads `game_state.must_move_from_stack_key` directly
- Python `_has_valid_movements` has `ignore_must_move_key` param for FE eligibility checks
- Both delegate to shared movement/capture enumeration for actual reachability

---

## P3: Cap Height Consolidation

### Problem

`calculateCapHeight` implemented in multiple locations:

- `src/shared/engine/core.ts` (canonical)
- `src/shared/engine/aggregates/EliminationAggregate.ts` (re-export)
- `ai-service/app/rules/core.py`
- `ai-service/app/rules/elimination.py`

Risk of subtle divergence in ring indexing semantics.

### Files Involved

| File                                                   | Status                |
| ------------------------------------------------------ | --------------------- |
| `src/shared/engine/core.ts`                            | CANONICAL             |
| `src/shared/engine/aggregates/EliminationAggregate.ts` | Uses canonical        |
| `ai-service/app/rules/elimination.py`                  | Parity implementation |

### Resolution Plan

1. Verify all TS code imports from core.ts or EliminationAggregate
2. Verify Python elimination.py matches TS semantics exactly
3. Add parity tests for cap height edge cases
4. Document ring indexing convention (top vs bottom)

### Progress

- [x] EliminationAggregate created with canonical logic
- [x] Python elimination.py created with parity tests
- [x] 32 Python tests + 33 TS tests passing
- [x] Ring indexing documented (2025-12-11)

### Ring Indexing Convention

**TypeScript** (`rings[0]` = TOP):

```typescript
// EliminationAggregate.ts
const topPlayer = rings[0]; // First element is top of stack
for (const ring of rings) {
  // Iterate from top to bottom
  if (ring === topPlayer) capHeight++;
  else break;
}
```

**Python** (`rings[-1]` = TOP):

```python
# elimination.py
top_player = rings[-1]  # Last element is top of stack
for ring in reversed(rings):  # Iterate from top to bottom
    if ring == top_player: cap_height += 1
    else: break
```

**Rationale**: The convention difference exists because:

1. TypeScript arrays naturally grow by appending to the end, but RingRift uses index-0-as-top for direct access to controlling player without `.at(-1)`.
2. Python uses Pythonic convention where `rings.append()` adds to top and `rings[-1]` accesses top.

**Parity Guarantee**: Both implementations produce identical cap heights for semantically equivalent stacks. The controlling player is always the ring at the "top" of the physical stack:

- TS: `rings[0]`
- Python: `rings[-1]`

---

## P4: Validation Result Unification

### Problem

Three different validation result structures exist:

1. `ValidationResult` (types.ts:135):

   ```typescript
   { valid: true } | { valid: false; reason: string; code: string }
   ```

2. `ValidationResult<T>` (contracts/validators.ts:570):

   ```typescript
   { success: true; data: T } | { success: false; error: string }
   ```

3. `PlacementValidationResult` (PlacementValidator.ts, PlacementAggregate.ts):
   ```typescript
   { valid: boolean; maxPlacementCount?: number; reason?: string; code?: string }
   ```

- Python validators: `bool` only (no structured results)

### Analysis (2025-12-11)

**Impact Assessment:**

- Low impact on correctness (all patterns work)
- Medium impact on debuggability (inconsistent error handling)
- High effort to unify (touches 19+ files, ~500 usages)

**Recommended Approach:**

1. Define `ValidationOutcome<T>` as the unified type
2. Add error code enum `ValidationErrorCode`
3. Migrate validators incrementally (placement first, then movement, etc.)
4. Python parity can use similar structure

**Deferred Rationale:**
Current inconsistency doesn't cause bugs - it's a DX/maintainability issue.
Higher priority work (P1 FSM migration, P2 predicates) completed first.

### Progress

- [x] Current state documented (2025-12-11)
- [ ] Unified ValidationOutcome<T> type defined
- [ ] Error code enum created
- [ ] TS validators migrated
- [ ] Python validators migrated

---

## P5: Sandbox Aggregate Delegation

### Problem

Sandbox modules (sandboxPlacement.ts, sandboxElimination.ts, etc.) reimplement logic instead of delegating to shared aggregates.

### Files Involved

| File                    | Lines | Should Delegate To                  |
| ----------------------- | ----- | ----------------------------------- |
| `sandboxPlacement.ts`   | 220   | PlacementAggregate/PlacementMutator |
| `sandboxElimination.ts` | 180   | EliminationAggregate                |
| `sandboxCaptures.ts`    | 174   | CaptureAggregate                    |
| `sandboxMovement.ts`    | 114   | MovementAggregate                   |
| `sandboxLines.ts`       | 144   | LineAggregate                       |
| `sandboxTerritory.ts`   | 164   | TerritoryAggregate                  |

### Resolution Plan

1. Audit each sandbox module for duplicate logic
2. Refactor to delegate to aggregates
3. Keep sandbox modules as thin wrappers for UI-specific concerns
4. Ensure parity tests cover sandbox behavior

### Progress

- [x] sandboxElimination.ts delegates to EliminationAggregate
- [x] sandboxPlacement.ts - ALREADY OPTIMAL (delegates to validatePlacementOnBoard)
- [x] sandboxMovement.ts - ALREADY OPTIMAL (thin adapter only)
- [x] sandboxTerritory.ts - ALREADY OPTIMAL (proper delegation)
- [x] sandboxLines.ts - dead code removed (2025-12-11)
  - Removed `getLineDirectionsForBoard`, `findLineInDirectionOnBoard` (~100 lines)
  - Exported canonical `getLineDirections`, `findLineInDirection` from LineAggregate
  - Updated tests to use canonical functions
  - File reduced from 145 to 49 lines
- [x] sandboxCaptures.ts - assessed (2025-12-11)
  - `enumerateCaptureSegmentsFromBoard` delegates to shared `enumerateCaptureMovesShared` ✅
  - `applyCaptureSegmentOnBoard` marked as DIAGNOSTICS-ONLY, used by:
    - `sandboxCaptureSearch.ts` for chain capture enumeration
    - Test files for parity verification
    - Scripts for exhaustive testing
  - Intentionally kept for diagnostic/test tooling - not production code path

---

## P6: Dead Code Cleanup

### Problem

Deprecated functions still exported, design-time stubs that throw, unused helper modules.

### Files to Review

- `phaseStateMachine.ts` - Entire module deprecated
- `turnDelegateHelpers.ts` - Stubs that throw
- `rulesConfig.ts` - Possibly unused
- Various `*Helpers.ts` files with incomplete consolidation

### Resolution Plan

1. Identify all @deprecated exports
2. Verify no production code depends on deprecated exports
3. Remove deprecated code or add migration warnings
4. Clean up design-time stubs

### Progress

- [x] TurnEngine dead code removed (2025-12-11)
  - Removed `hasValidActions`, `hasValidRecovery`, `nextPlayer` (never called)
  - Removed `_debugUseInternalTurnEngineHelpers` hook (no longer needed)
  - ~50 lines removed
- [x] ClientSandboxEngine dead code removed (2025-12-11)
  - Removed `hasAnyMovementOrCaptureForPlayer` (replaced by shared predicates)
  - ~25 lines removed
- [x] RecoverySlideTarget.cost deprecated field removed (2025-12-11)
  - Replaced by `option1Cost` and `option2Cost` fields
  - Test updated to use `option1Cost`
- [x] Deprecated exports inventoried (2025-12-11)
- [x] Dependencies verified (2025-12-11)
- [ ] Remaining cleanup pending (phaseStateMachine.ts depends on P1)

### Deprecated Exports Inventory (2025-12-11)

| Location                   | Export                    | Status  | Blocker                                       |
| -------------------------- | ------------------------- | ------- | --------------------------------------------- |
| `phaseStateMachine.ts`     | Entire module (9 exports) | BLOCKED | P1 - turnOrchestrator uses it                 |
| `turnOrchestrator.ts:2280` | `computeNextPhase`        | Keep    | Internal orchestrator use                     |
| `turnOrchestrator.ts:2785` | `validateMove` (legacy)   | Keep    | Used by processTurn, FSM validation preferred |
| `core.ts:781`              | `hashGameState`           | Keep    | Alias to fingerprintGameState, heavy usage    |
| `types/game.ts:138-146`    | MoveType aliases          | Keep    | Historical game recordings                    |

**MoveType Deprecations (keep for backwards compatibility):**

- `line_formation` → Use `process_line` + `choose_line_reward`
- `territory_claim` → Use `process_territory_region` + `skip_territory_processing`
- `skip_placement` → Use `no_placement_action`

**Assessment:**

- All deprecated exports are either blocked by P1 or intentionally kept for backwards compatibility
- No low-risk removals available without completing P1 (FSM migration)
- `hashGameState` has 40+ usages - safe as forwarding alias

---

## Completed Work

### Elimination Logic Consolidation (2025-12-11)

**Problem**: Elimination logic scattered across multiple files with context-specific rules.

**Solution**: Created `EliminationAggregate.ts` and `elimination.py` as single source of truth.

**Files Created/Modified**:

- `src/shared/engine/aggregates/EliminationAggregate.ts` (new)
- `ai-service/app/rules/elimination.py` (new)
- `tests/unit/engine/EliminationAggregate.test.ts` (new - 33 tests)
- `ai-service/tests/rules/test_elimination.py` (new - 32 tests)

**Consumers Updated**:

- `sandboxElimination.ts` - delegates to EliminationAggregate
- `territoryDecisionHelpers.ts` - uses EliminationAggregate
- `TerritoryAggregate.ts` - uses isStackEligibleForElimination
- `TerritoryMutator.ts` - delegates to EliminationAggregate
- `globalActions.ts` - adds eliminationContext to moves
- `ai-service/app/rules/validators/territory.py` - uses elimination module

**Key Design Decisions**:

- Four elimination contexts: `line`, `territory`, `forced`, `recovery`
- Height-1 stacks NOT eligible for territory (RR-CANON-R145)
- Context determines cost (1 ring for line, entire cap for territory/forced)

### Phase Validation Matrix (2025-12-11)

**Problem**: No declarative mapping of which moves are valid in which phases.

**Solution**: Created `phaseValidation.ts` with `VALID_MOVES_BY_PHASE` matrix.

**Files Created**:

- `src/shared/engine/phaseValidation.ts` (new)
- `tests/unit/engine/phaseValidation.test.ts` (new - 44 tests)

---

## References

- `ELIMINATION_REFACTORING_PLAN.md` - Detailed elimination consolidation tracking
- `RULES_CANONICAL_SPEC.md` - Canonical rules reference
- `CANONICAL_ENGINE_API.md` - Public API documentation
