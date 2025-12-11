# Phase Orchestration Architecture

**Created**: 2025-12-11
**Status**: Reference Documentation

## Overview

RingRift uses a dual-system architecture for phase orchestration. This document explains why both systems exist, how they interact, and when to use each.

## The Two Systems

### 1. TurnStateMachine (FSM) - Canonical for Validation

**Location**: `src/shared/engine/fsm/TurnStateMachine.ts`

The FSM is a type-safe finite state machine that:

- Defines all valid (state, event) → nextState transitions
- Provides compile-time guarantees against invalid transitions
- Is the **canonical validator** for all moves via `validateMoveWithFSM()`

```typescript
// FSM validation is always used
const fsmValidationResult = validateMoveWithFSM(state, move);
if (!fsmValidationResult.valid) {
  throw new Error(`Invalid move: ${fsmValidationResult.reason}`);
}
```

### 2. PhaseStateMachine (Legacy) - Used for State Tracking

**Location**: `src/shared/engine/orchestration/phaseStateMachine.ts`

The legacy state machine:

- Tracks processing state during turn execution (pending lines, chain captures)
- Manages `gameState` mutations during `processTurn()`
- Provides the `TurnProcessingState` structure

```typescript
// Legacy machine used for state tracking during processTurn
const stateMachine = new PhaseStateMachine(createTurnProcessingState(state, move));
stateMachine.updateGameState(applyResult.nextState);
```

## How They Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                        processTurn()                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. VALIDATION (FSM)                                             │
│     ┌──────────────────────────────────────────┐                 │
│     │ validateMoveWithFSM(state, move)         │                 │
│     │ - Phase-aware validation                 │                 │
│     │ - Type-safe transition guards            │                 │
│     │ - Returns FSMValidationResult            │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  2. STATE TRACKING (Legacy PhaseStateMachine)                    │
│     ┌──────────────────────────────────────────┐                 │
│     │ new PhaseStateMachine(...)               │                 │
│     │ - Tracks pending lines/regions           │                 │
│     │ - Manages chain capture state            │                 │
│     │ - Holds mutable gameState reference      │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  3. MOVE APPLICATION (Domain Aggregates)                         │
│     ┌──────────────────────────────────────────┐                 │
│     │ applyMoveWithChainInfo()                 │                 │
│     │ - Delegates to PlacementAggregate,       │                 │
│     │   MovementAggregate, CaptureAggregate    │                 │
│     │ - Returns updated state + chain info     │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  4. POST-MOVE PROCESSING (Mixed)                                 │
│     ┌──────────────────────────────────────────┐                 │
│     │ processPostMovePhases()                  │                 │
│     │ - Uses legacy stateMachine for tracking  │                 │
│     │ - Line detection, territory processing   │                 │
│     │ - Victory evaluation                     │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  5. PHASE RESOLUTION (FSM)                                       │
│     ┌──────────────────────────────────────────┐                 │
│     │ computeFSMOrchestration()                │                 │
│     │ - Determines next phase                  │                 │
│     │ - Computes pending decisions             │                 │
│     │ - Handles player rotation                │                 │
│     └──────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Both Systems Exist

### Historical Context

1. **PhaseStateMachine** was the original implementation
2. **TurnStateMachine (FSM)** was created later with better type safety
3. FSM was integrated for validation first (lower risk)
4. Full migration of state tracking was deferred due to complexity

### Current Reality

- **FSM validation works correctly** - all moves are validated through FSM
- **Legacy state tracking works correctly** - no bugs in current behavior
- **Migration risk is high** - 69 method calls to replace in turnOrchestrator
- **Python parity would be required** - `phase_machine.py` would need equivalent changes

### Decision: Keep Both (For Now)

The dual system is:

- ✅ Working correctly
- ✅ Well-tested (254+ GameEngine tests)
- ✅ Type-safe where it matters (validation)
- ⚠️ Architecturally redundant
- ⚠️ Harder to understand for new developers

## Guidelines for Developers

### When Adding New Validation Logic

**Use FSM** (`TurnStateMachine.ts` / `FSMAdapter.ts`):

```typescript
// Add new phase guards in TurnStateMachine
// Add validation helpers in FSMAdapter
```

### When Adding New State Tracking

**Use Legacy** (`phaseStateMachine.ts`) until full migration:

```typescript
// Add to TurnProcessingState interface
// Update PhaseStateMachine methods
```

### When to Consider Full Migration

Consider migrating to FSM-only if:

1. A bug is found that requires deep phase handling changes
2. A feature requires FSM-only capabilities
3. Python FSM reaches feature parity
4. Significant refactoring is already planned for turnOrchestrator

## File Reference

| File                                 | Purpose                          | Status                |
| ------------------------------------ | -------------------------------- | --------------------- |
| `fsm/TurnStateMachine.ts`            | Type-safe FSM states/transitions | Canonical             |
| `fsm/FSMAdapter.ts`                  | Bridges FSM with game types      | Canonical             |
| `fsm/index.ts`                       | FSM public API                   | Canonical             |
| `orchestration/phaseStateMachine.ts` | Legacy state tracking            | Deprecated (but used) |
| `orchestration/turnOrchestrator.ts`  | Main orchestrator                | Uses both systems     |

## Python Parity

| TypeScript             | Python             | Status            |
| ---------------------- | ------------------ | ----------------- |
| `TurnStateMachine.ts`  | `fsm.py`           | Experimental      |
| `FSMAdapter.ts`        | (none)             | Not implemented   |
| `phaseStateMachine.ts` | `phase_machine.py` | Parity maintained |

Python currently uses `phase_machine.py` for all phase logic. Full FSM parity would require implementing `FSMAdapter` equivalent in Python.

## Future Work

If full FSM migration is undertaken:

1. **Phase 1**: Replace `stateMachine.gameState` with FSM context
2. **Phase 2**: Replace `processPostMovePhases()` with FSM-driven logic
3. **Phase 3**: Remove PhaseStateMachine class
4. **Phase 4**: Update Python to use FSM

Estimated effort: High (2-3 weeks of focused work + Python parity)
