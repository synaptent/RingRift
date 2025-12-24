# Elimination Logic Refactoring Plan

**Status**: In Progress
**Created**: 2025-12-11
**Last Updated**: 2025-12-11
**SSoT**: RULES_CANONICAL_SPEC.md

## Overview

This document tracks the architectural refactoring to consolidate elimination logic and improve long-term maintainability, debuggability, and correctness.

## Problem Statement

Elimination logic is currently spread across multiple files with subtle differences:

| File                                   | Responsibility                  | Issues                                                   |
| -------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| `sandboxElimination.ts`                | Client-side elimination         | Duplicates backend logic                                 |
| `territoryDecisionHelpers.ts`          | Territory elimination moves     | Has eligibility filtering                                |
| `TerritoryAggregate.ts`                | Territory processing            | Missing eligibility check in `canProcessTerritoryRegion` |
| `LineAggregate.ts`                     | Line elimination                | Separate elimination context                             |
| `TerritoryMutator.ts`                  | Applies territory mutations     | Has own elimination logic                                |
| `recovery.py` / `RecoveryAggregate.ts` | Recovery buried ring extraction | Exception to normal elimination                          |

### Canonical Rules (from RULES_CANONICAL_SPEC.md)

| Context                  | Elimination Cost         | Eligible Stacks                                      | Reference     |
| ------------------------ | ------------------------ | ---------------------------------------------------- | ------------- |
| **Line Processing**      | 1 ring from top          | Any controlled stack (including height-1)            | RR-CANON-R122 |
| **Territory Processing** | Entire cap               | Multicolor OR single-color height > 1 (NOT height-1) | RR-CANON-R145 |
| **Forced Elimination**   | Entire cap               | Any controlled stack (including height-1)            | RR-CANON-R100 |
| **Recovery Action**      | 1 buried ring extraction | Any stack with player's buried ring                  | RR-CANON-R113 |

## Refactoring Goals

1. **Single Source of Truth**: Each elimination concept in exactly one place
2. **Explicit Context**: Elimination context (`line`/`territory`/`forced`/`recovery`) explicit in all code paths
3. **Debuggability**: Clear audit trail for all elimination operations
4. **Parity**: Identical logic in Python and TypeScript

## Phase 1: Consolidate TypeScript Elimination Logic

### 1.1 Create EliminationAggregate.ts

New file: `src/shared/engine/aggregates/EliminationAggregate.ts`

```typescript
/**
 * Canonical elimination logic for all elimination contexts.
 *
 * This is the ONLY place elimination semantics should be implemented.
 * All other code should delegate to this module.
 */

export type EliminationContext = 'line' | 'territory' | 'forced' | 'recovery';

export interface EliminationParams {
  context: EliminationContext;
  player: number;
  stackPosition: Position;
  board: BoardState;
}

export interface EliminationResult {
  success: boolean;
  ringsEliminated: number;
  updatedBoard: BoardState;
  error?: string;
}

// Core function - all elimination goes through here
export function eliminateFromStack(params: EliminationParams): EliminationResult;

// Eligibility check - context-aware
export function isStackEligibleForElimination(
  stack: RingStack,
  context: EliminationContext,
  player: number
): boolean;

// Enumerate eligible stacks for a given context
export function enumerateEligibleStacks(
  board: BoardState,
  player: number,
  context: EliminationContext,
  excludePositions?: Set<string> // For territory: exclude region positions
): RingStack[];
```

### 1.2 Update Consumers

Files to update:

- [ ] `LineAggregate.ts` - delegate to EliminationAggregate
- [ ] `TerritoryAggregate.ts` - delegate to EliminationAggregate
- [ ] `TerritoryMutator.ts` - delegate to EliminationAggregate
- [ ] `territoryDecisionHelpers.ts` - delegate to EliminationAggregate
- [ ] `sandboxElimination.ts` - delegate to EliminationAggregate
- [ ] `turnOrchestrator.ts` - use EliminationAggregate for forced elimination

### 1.3 Remove Duplicated Logic

After consolidation, remove:

- Eligibility checks scattered in multiple files
- Redundant `calculateCapHeight` calls
- Duplicated ring removal logic

## Phase 2: Unify Territory Processing

### 2.1 Consolidate `canProcessTerritoryRegion`

Current duplication:

- `territoryProcessing.ts:109-135` - HAS eligibility check
- `TerritoryAggregate.ts:752-771` - MISSING eligibility check

Resolution:

- Keep `territoryProcessing.ts` as canonical
- Have `TerritoryAggregate.ts` delegate to it
- Or merge into single location

### 2.2 Add Recovery Context Flag

```typescript
interface TerritoryProcessingContext {
  player: number;
  boardType: BoardType;
  isRecoveryTriggered?: boolean; // NEW: explicit flag for recovery exception
}
```

When `isRecoveryTriggered === true`:

- Self-elimination cost = 1 buried ring extraction
- Different eligibility rules apply

## Phase 3: Create Declarative Phase-Move Matrix

### 3.1 Define Matrix

```typescript
// src/shared/engine/phaseValidation.ts

export const VALID_MOVES_BY_PHASE: Record<GamePhase, readonly MoveType[]> = {
  ring_placement: [
    'place_ring',
    'skip_placement',
    'forced_elimination',
    'recovery_slide',
    'skip_recovery',
  ],
  movement: ['move_stack', 'no_movement_action'],
  line_processing: ['process_line', 'no_line_action', 'eliminate_rings_from_stack'],
  territory_processing: [
    'choose_territory_option', // legacy alias: 'process_territory_region'
    'no_territory_action',
    'eliminate_rings_from_stack',
  ],
  forced_elimination: ['forced_elimination', 'eliminate_rings_from_stack'],
  game_over: [],
} as const;

export function isMoveValidInPhase(moveType: MoveType, phase: GamePhase): boolean {
  return VALID_MOVES_BY_PHASE[phase]?.includes(moveType) ?? false;
}
```

### 3.2 Update Turn Orchestrator

Replace scattered phase checks with:

```typescript
if (!isMoveValidInPhase(move.type, state.currentPhase)) {
  return {
    success: false,
    error: `Move type '${move.type}' not valid in phase '${state.currentPhase}'`,
  };
}
```

## Phase 4: Add Elimination Audit Trail

### 4.1 Define Event Structure

```typescript
// src/shared/engine/eliminationAudit.ts

export interface EliminationEvent {
  timestamp: Date;
  moveNumber: number;
  context: EliminationContext;
  player: number;
  stackPosition: Position;
  ringsEliminated: number;
  stackHeightBefore: number;
  stackHeightAfter: number;
  capHeightBefore: number;
  controllingPlayerBefore: number;
  controllingPlayerAfter: number | null;
  reason: EliminationReason;
}

export type EliminationReason =
  | 'line_reward_option1'
  | 'territory_self_elimination'
  | 'forced_elimination_anm'
  | 'recovery_buried_extraction'
  | 'capture_overtake';
```

### 4.2 Emit Events

All elimination operations emit events via:

```typescript
export function emitEliminationEvent(event: EliminationEvent): void {
  if (process.env.RINGRIFT_ELIMINATION_AUDIT) {
    console.log('[ELIMINATION_AUDIT]', JSON.stringify(event));
  }
}
```

## Phase 5: Python Parity

### 5.1 Create Python EliminationAggregate

Mirror TypeScript structure in:
`ai-service/app/rules/elimination.py`

### 5.2 Update Python Consumers

- [ ] `territory.py`
- [ ] `recovery.py`
- [ ] `game_engine/__init__.py`
- [ ] `mutable_state.py`

## Phase 6: Fix Hex Parity

### 6.1 Add Phase Transition Audit

```typescript
function auditPhaseTransition(
  before: { phase: GamePhase; player: number; stateHash: string },
  after: { phase: GamePhase; player: number; stateHash: string },
  trigger: string
): void {
  if (process.env.RINGRIFT_PHASE_AUDIT) {
    console.log('[PHASE_TRANSITION]', { before, after, trigger });
  }
}
```

### 6.2 Run Hex Game with Audit

Identify exact divergence point by running:

```bash
RINGRIFT_PHASE_AUDIT=1 npx ts-node scripts/selfplay-db-ts-replay.ts --db canonical_hexagonal.db --game <id>
```

## Progress Tracking

| Phase                                     | Status      | Completion Date | Notes                                                                                                            |
| ----------------------------------------- | ----------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1.1 Create EliminationAggregate           | ✅ Complete | 2025-12-11      | 33 unit tests passing                                                                                            |
| 1.2 Update Consumers                      | ✅ Complete | 2025-12-11      | sandboxElimination ✅, territoryDecisionHelpers ✅, globalActions ✅, TerritoryAggregate ✅, TerritoryMutator ✅ |
| 1.3 Remove Duplicates                     | Deferred    |                 | Duplicate code removed via delegation, functions retained for API compat                                         |
| 2.1 Consolidate canProcessTerritoryRegion | ✅ Complete | 2025-12-11      | TerritoryAggregate now delegates to EliminationAggregate, fixtures fixed for height-1 rule                       |
| 2.2 Add Recovery Context Flag             | Not Started |                 |                                                                                                                  |
| 3.1 Define Phase-Move Matrix              | ✅ Complete | 2025-12-11      | phaseValidation.ts created with 44 unit tests passing                                                            |
| 3.2 Update Turn Orchestrator              | Not Started |                 |                                                                                                                  |
| 4.1 Define Audit Structure                | Not Started |                 |                                                                                                                  |
| 4.2 Emit Events                           | Not Started |                 |                                                                                                                  |
| 5.1 Create Python EliminationAggregate    | ✅ Complete | 2025-12-11      | 32 unit tests passing                                                                                            |
| 5.2 Update Python Consumers               | ✅ Complete | 2025-12-11      | TerritoryValidator now delegates to EliminationAggregate for eligibility                                         |
| 6.1 Add Phase Transition Audit            | Not Started |                 |                                                                                                                  |
| 6.2 Fix Hex Parity                        | Not Started |                 |                                                                                                                  |

## Testing Strategy

1. **Unit Tests**: Each new function in EliminationAggregate
2. **Integration Tests**: End-to-end elimination scenarios
3. **Parity Tests**: Python/TS produce identical results
4. **Regression Tests**: Existing tests continue to pass

## Rollback Plan

All changes are additive initially:

1. New EliminationAggregate created alongside existing code
2. Consumers updated one at a time
3. Old code removed only after all tests pass
4. Each phase is a separate commit for easy revert

## References

- RULES_CANONICAL_SPEC.md (elimination rules: RR-CANON-R022, R100, R113, R122, R145)
- src/shared/engine/aggregates/\*.ts (current implementations)
- ai-service/app/rules/\*.py (Python implementations)
