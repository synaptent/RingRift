> **Doc Status (2025-12-07): Proposal / Partially Superseded**
> **Role:** Architectural proposal for reducing ClientSandboxEngine.ts file size through state machine extraction.
>
> **Status (2025-12-07):** This is a **deferred proposal**. The core parity testing concern has been addressed by `CanonicalReplayEngine` (426 lines), which provides coercion-free replay without any ClientSandboxEngine dependencies.
>
> **Architecture Update (Dec 2025):**
>
> - `CanonicalReplayEngine` now powers FSM validation and DB replay scripts
> - ClientSandboxEngine coercions (~250 lines) remain only for interactive play and legacy recording replay
> - 20 parity test files still use ClientSandboxEngine with `traceMode: true`; migration to CanonicalReplayEngine is tracked in `TODO.md`
>
> **Related Docs:**
>
> - `docs/architecture/SHARED_ENGINE_CONSOLIDATION_PLAN.md` (completed consolidation)
> - `src/shared/replay/CanonicalReplayEngine.ts` (new clean replay engine)
> - `src/client/sandbox/ClientSandboxEngine.ts` (target file)
> - `src/client/sandbox/sandboxDecisionMapping.ts` (already extracted)
> - `src/client/sandbox/boardViewFactory.ts` (already extracted)
> - `docs/runbooks/FSM_VALIDATION_ROLLOUT.md` (FSM validation using CanonicalReplayEngine)

# ClientSandboxEngine State Machine Refactor Proposal

## Executive Summary

This document proposes extracting ~940 lines of replay and territory processing logic from `ClientSandboxEngine.ts` (currently 4,162 lines) into a separate `SandboxReplayStateMachine` class. This would reduce the file to approximately 3,200 lines and improve testability by making state mutations explicit.

**Recommendation:** Do not proceed at this time. The code is working with validated Python parity, and the refactor risk exceeds the benefit.

---

## Current State Analysis

### File Metrics (as of 2025-12-07)

| Metric                    | Value        |
| ------------------------- | ------------ |
| Total Lines               | 4,162        |
| Target                    | <3,000       |
| Gap                       | ~1,162 lines |
| `this.` References        | 574          |
| Private Methods           | 63           |
| Avg State Refs per Method | ~9           |

### Extraction Candidates

The following large methods are tightly coupled to class state and cannot be extracted as pure functions without architectural changes:

| Method                                       | Lines | Primary Dependencies                                                                                                                                                  |
| -------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `applyCanonicalMoveForReplay`                | ~363  | gameState (r/w), traceMode, appendHistoryEntry, advanceTurnAndPhaseForCurrentPlayer, checkAndApplyVictory, enumerateCaptureSegmentsFrom, canProcessDisconnectedRegion |
| `autoResolvePendingDecisionPhasesForReplay`  | ~290  | gameState (r/w), advanceTurnAndPhaseForCurrentPlayer, autoResolveOneTerritoryRegionForReplay, autoResolveOneLineForReplay                                             |
| `processDisconnectedRegionsForCurrentPlayer` | ~285  | gameState (r/w), traceMode, interactionHandler, appendHistoryEntry, canProcessDisconnectedRegion, \_pendingTerritorySelfElimination                                   |

### Already Completed Extractions

| File                        | Lines    | Description                                        |
| --------------------------- | -------- | -------------------------------------------------- |
| `sandboxDecisionMapping.ts` | 373      | PlayerChoice builders, decision/response mapping   |
| `boardViewFactory.ts`       | 275      | Board view adapters for placement/movement/capture |
| `sandboxPlacement.ts`       | existing | Placement validation and hypothetical boards       |
| `sandboxMovement.ts`        | existing | Movement landing enumeration                       |
| `sandboxCaptures.ts`        | existing | Capture segment enumeration                        |
| `sandboxTerritory.ts`       | existing | Disconnected region detection                      |
| `sandboxLines.ts`           | existing | Line detection                                     |
| `sandboxElimination.ts`     | existing | Cap elimination helpers                            |
| `sandboxGameEnd.ts`         | existing | Victory detection and game end hooks               |
| `sandboxAI.ts`              | existing | AI turn execution                                  |

---

## Proposed Architecture: State Machine Pattern

### Design

Extract replay and territory processing logic into a `SandboxReplayStateMachine` class that receives explicit callbacks for all mutations.

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// New File: src/client/sandbox/SandboxReplayStateMachine.ts
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Read-only context passed to state machine methods.
 * No mutations allowed through this interface.
 */
export interface ReplayContext {
  readonly state: GameState;
  readonly traceMode: boolean;
  readonly boardType: BoardType;
}

/**
 * Mutation callbacks that the state machine invokes.
 * Each callback is explicit and auditable.
 */
export interface ReplayActions {
  // State mutations
  setState(newState: GameState): void;
  setPhase(phase: GamePhase): void;
  setCurrentPlayer(player: number): void;

  // History tracking
  appendHistoryEntry(before: GameState, move: Move): void;

  // Victory checks
  checkAndApplyVictory(): void;

  // Decision phase helpers (delegate back to engine)
  advanceTurnAndPhase(): void;
  autoResolveOneTerritoryRegion(): Promise<boolean>;
  autoResolveOneLine(): Promise<boolean>;

  // Query helpers (read-only, but may need engine context)
  enumerateCaptureSegmentsFrom(pos: Position, player: number): CaptureSegment[];
  canProcessDisconnectedRegion(spaces: Position[], player: number, board: BoardState): boolean;
}

/**
 * State machine for replay move application.
 * All state access is through ReplayContext.
 * All mutations are through ReplayActions.
 */
export class SandboxReplayStateMachine {
  /**
   * Apply a canonical move during replay.
   * Returns true if state changed.
   */
  async applyMoveForReplay(
    ctx: ReplayContext,
    actions: ReplayActions,
    move: Move,
    nextMove?: Move | null
  ): Promise<boolean> {
    // ... ~363 lines of logic moved from ClientSandboxEngine
  }

  /**
   * Auto-resolve pending decision phases to align with next move.
   */
  async autoResolvePendingDecisionPhases(
    ctx: ReplayContext,
    actions: ReplayActions,
    nextMove: Move
  ): Promise<void> {
    // ... ~290 lines of logic moved from ClientSandboxEngine
  }
}
```

### Integration in ClientSandboxEngine

```typescript
// In ClientSandboxEngine constructor
private replayStateMachine = new SandboxReplayStateMachine();

// Refactored applyCanonicalMoveForReplay becomes a thin wrapper
public async applyCanonicalMoveForReplay(move: Move, nextMove?: Move | null): Promise<void> {
  const ctx: ReplayContext = {
    state: this.gameState,
    traceMode: this.traceMode,
    boardType: this.gameState.boardType,
  };

  const actions: ReplayActions = {
    setState: (s) => { this.gameState = s; },
    setPhase: (p) => { this.gameState.currentPhase = p; },
    setCurrentPlayer: (p) => { this.gameState.currentPlayer = p; },
    appendHistoryEntry: (b, m) => this.appendHistoryEntry(b, m),
    checkAndApplyVictory: () => this.checkAndApplyVictory(),
    advanceTurnAndPhase: () => this.advanceTurnAndPhaseForCurrentPlayer(),
    autoResolveOneTerritoryRegion: () => this.autoResolveOneTerritoryRegionForReplay(),
    autoResolveOneLine: () => this.autoResolveOneLineForReplay(),
    enumerateCaptureSegmentsFrom: (p, pl) => this.enumerateCaptureSegmentsFrom(p, pl),
    canProcessDisconnectedRegion: (s, p, b) => this.canProcessDisconnectedRegion(s, p, b),
  };

  await this.replayStateMachine.applyMoveForReplay(ctx, actions, move, nextMove);
}
```

---

## Risk Assessment

### High Risks

| Risk                        | Description                                                                                                                                                    | Mitigation                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Parity Regression**       | The replay methods contain critical Python parity logic with detailed comments explaining alignment. Refactoring could break subtle phase transition behavior. | Comprehensive snapshot tests before refactoring. Run full parity suite after each change.       |
| **Async Callback Ordering** | State machine actions are async. Incorrect ordering in callbacks could cause race conditions or inconsistent state.                                            | Careful code review. Consider making some actions synchronous where possible.                   |
| **Callback Hell**           | 10+ callbacks in ReplayActions interface creates complexity and makes debugging harder.                                                                        | Group related callbacks. Consider using an object with methods instead of individual callbacks. |
| **Hidden Dependencies**     | Methods may have implicit ordering dependencies (e.g., `appendHistoryEntry` must be called after state mutation).                                              | Document all ordering requirements. Add assertions in debug mode.                               |

### Medium Risks

| Risk                     | Description                                                                                                      | Mitigation                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Context Threading**    | ReplayContext must be rebuilt for each call, adding overhead and potential for stale data.                       | Ensure context is always fresh. Consider lazy accessors. |
| **Test Rewrite**         | Existing tests mock `ClientSandboxEngine` methods directly. These would need updating for state machine pattern. | Plan test migration as part of refactor.                 |
| **Debugging Complexity** | Stack traces will span state machine and callback implementations.                                               | Add tracing/logging at state machine entry points.       |

### Low Risks

| Risk                     | Description                                          | Mitigation                                      |
| ------------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| **Performance Overhead** | Additional function call overhead for each callback. | Minimal impact; not on hot path.                |
| **Type Complexity**      | Complex generic types for context and actions.       | Keep interfaces simple. Avoid over-abstraction. |

---

## Benefits Assessment

### Definite Benefits

| Benefit                      | Value                                           |
| ---------------------------- | ----------------------------------------------- |
| **File Size Reduction**      | ~600-700 lines moved to separate file           |
| **Testability**              | State machine can be tested with mock actions   |
| **Explicit State Mutations** | All changes go through actions; easier to audit |
| **Separation of Concerns**   | Replay logic separated from interactive logic   |

### Potential Benefits

| Benefit                  | Conditions                                                          |
| ------------------------ | ------------------------------------------------------------------- |
| **Third Execution Mode** | Only valuable if we need a third mode (not replay, not interactive) |
| **Parallel Development** | Only valuable if multiple developers frequently modify this code    |
| **Coverage Improvement** | Only valuable if we need targeted coverage for replay paths         |

### Unlikely Benefits

| Claimed Benefit     | Why Unlikely                                                          |
| ------------------- | --------------------------------------------------------------------- |
| **Maintainability** | Complexity shifts to callback management; net complexity may increase |
| **Debuggability**   | Callback indirection may make debugging harder, not easier            |
| **Code Reuse**      | State machine is specific to sandbox replay; limited reuse potential  |

---

## Effort Estimation

| Phase                                | Effort        | Risk     |
| ------------------------------------ | ------------- | -------- |
| Create comprehensive snapshot tests  | 1 day         | Low      |
| Extract state machine with callbacks | 2-3 days      | High     |
| Update all callsites (~40-60)        | 1 day         | Medium   |
| Migrate existing tests               | 1-2 days      | Medium   |
| Full parity validation               | 1 day         | High     |
| Bug fixes and adjustments            | 1-2 days      | High     |
| **Total**                            | **7-10 days** | **High** |

---

## Decision Criteria

Proceed with this refactor **only if**:

1. **Third Execution Mode Required**: A new execution context (not replay, not interactive) needs to reuse replay logic without the full `ClientSandboxEngine` class.

2. **Frequent Merge Conflicts**: Multiple developers are regularly conflicting on `ClientSandboxEngine.ts` changes.

3. **Coverage Requirements**: Specific coverage metrics are mandated for replay paths that cannot be achieved with current architecture.

4. **Performance Profiling**: Profiling shows that state machine isolation would enable meaningful optimizations.

**Do not proceed if**:

- The motivation is purely file size reduction
- Parity testing shows any regressions
- No clear business need beyond code aesthetics

---

## Alternative Approaches Considered

### Option B: Context Object Pattern

Pass a context object instead of using `this`, without full state machine extraction.

**Pros:** Lower risk, gradual migration possible
**Cons:** Still requires ~40+ callsite changes, context interface becomes complex
**Verdict:** If we must do something, this is safer than full state machine

### Option C: Mixins/Subclasses

Split into `SandboxReplayMixin` and `SandboxInteractiveMixin`.

**Pros:** Logical separation, each can be tested independently
**Cons:** TypeScript mixin complexity, shared state still accessed via `this`, debugging harder
**Verdict:** Not recommended; adds complexity without solving the core coupling issue

### Option D: Do Nothing

Leave the code as-is.

**Pros:** Zero risk, zero effort, code is working and tested
**Cons:** File remains at 4,162 lines
**Verdict:** **Recommended for now**

---

## Appendix: Parity Test Coverage

The following tests validate TS↔Python parity for the code in question:

| Test Suite                                 | Status                 | Coverage                             |
| ------------------------------------------ | ---------------------- | ------------------------------------ |
| `contractVectorRunner.test.ts`             | 18 passing             | All move types, phase transitions    |
| `SelfPlayGameService.test.ts`              | 24 passing             | DB replay, state serialization       |
| `Seed5*.parity.test.ts`                    | Passing                | Territory detection, terminal states |
| `TraceFixtures.sharedEngineParity.test.ts` | 7 passing, 2 failing\* | Trace mode replay                    |

\*Failures are pre-existing issues unrelated to file structure (forced_elimination move type naming).

---

## Revision History

| Date       | Author                 | Changes                  |
| ---------- | ---------------------- | ------------------------ |
| 2025-12-07 | Claude (via assistant) | Initial proposal created |
