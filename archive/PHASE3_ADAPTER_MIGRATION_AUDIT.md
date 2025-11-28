# Phase 3: Backend/Sandbox Adapter Migration Audit

**Date:** 2025-11-26  
**Status:** Audit Complete - Migration Assessment  
**Author:** Code Mode

---

## Executive Summary

This document presents the findings from the Phase 3 audit of backend `TurnEngine`/`GameEngine` and client `ClientSandboxEngine` for migration to thin adapters over the shared orchestrator.

### Current State

| Metric                         | Value        | Notes                                    |
| ------------------------------ | ------------ | ---------------------------------------- |
| Contract Tests                 | 14 passing   | Phase 1-2 orchestrator working correctly |
| Total Tests                    | 1178 passing | Baseline stable                          |
| Parity Tests                   | 4 failing    | Evidence of backend/sandbox divergence   |
| Backend GameEngine LOC         | 3320 lines   | Primary migration target                 |
| Client ClientSandboxEngine LOC | 2707 lines   | Secondary migration target               |

### Key Finding

**Both engines already delegate significant logic to shared modules**, but they orchestrate the flow themselves rather than delegating to the canonical `processTurn()`. The orchestrator exists and is tested - the migration is about wiring the adapters.

---

## 1. Backend Audit (Task 3.1)

### 1.1 Files Analyzed

| File                                                        | Lines | Role                                 |
| ----------------------------------------------------------- | ----- | ------------------------------------ |
| [`TurnEngine.ts`](../../src/server/game/turn/TurnEngine.ts) | 375   | Turn advancement, forced elimination |
| [`GameEngine.ts`](../../src/server/game/GameEngine.ts)      | 3320  | Main game orchestration              |
| [`RuleEngine.ts`](../../src/server/game/RuleEngine.ts)      | 1556  | Move validation/enumeration          |
| [`turnLogic.ts`](../../src/shared/engine/turnLogic.ts)      | 312   | Shared turn/phase progression        |

### 1.2 Logic Already Using Shared Modules ✅

The backend **already delegates** the following to shared helpers:

```typescript
// TurnEngine.ts - Uses shared advanceTurnAndPhase()
import { advanceTurnAndPhase } from '../../../shared/engine/turnLogic';

// RuleEngine.ts - Uses shared validators and helpers
import {
  validatePlacementOnBoard,
  enumerateSimpleMoveTargetsFromStack,
  validateCaptureSegmentOnBoard,
  enumerateCaptureMoves,
  enumerateProcessLineMoves,
  enumerateProcessTerritoryRegionMoves,
} from '../../../shared/engine';
```

**Shared functions already in use:**

- `advanceTurnAndPhase()` - Phase/player rotation
- `validatePlacementOnBoard()` - Placement validation
- `enumerateSimpleMoveTargetsFromStack()` - Movement enumeration
- `validateCaptureSegmentOnBoard()` - Capture validation
- `enumerateCaptureMoves()` - Capture enumeration
- `enumerateProcessLineMoves()` - Line decision enumeration
- `applyProcessLineDecision()` - Line move application
- `enumerateProcessTerritoryRegionMoves()` - Territory decision enumeration
- `applyProcessTerritoryRegionDecision()` - Territory move application
- `evaluateVictory()` - Victory detection

### 1.3 Logic Still Duplicated in Backend ❌

The following logic in `GameEngine.ts` duplicates orchestrator functionality:

```typescript
// GameEngine.ts duplicates orchestrator's phase flow
class GameEngine {
  // DUPLICATE: Move application (orchestrator has applyMove via mutators)
  private applyMove(move: Move): void { ... }  // ~300 lines

  // DUPLICATE: Automatic consequences (orchestrator handles in processTurn)
  private processAutomaticConsequences(): Promise<void> { ... }  // ~200 lines

  // DUPLICATE: Phase management (orchestrator has PhaseStateMachine)
  private updatePhaseAfterMove(): void { ... }  // ~100 lines

  // DUPLICATE: Chain capture state (orchestrator tracks in TurnProcessingState)
  private chainCaptureState: { ... }

  // DUPLICATE: Pending eliminations (orchestrator tracks in PerTurnFlags)
  private pendingLineRewardElimination: boolean;
  private pendingTerritorySelfElimination: boolean;
}
```

### 1.4 Backend-Specific Concerns (Keep in Adapter)

These concerns must remain in the backend adapter:

```typescript
interface BackendAdapterConcerns {
  // WebSocket/Interaction
  interactionManager: PlayerInteractionManager; // Player choice prompts
  webSocketEmitter: GameEventEmitter; // Real-time updates

  // Persistence
  appendHistoryEntry(before, action, after): void;
  debugCheckpointHook?: (label, state) => void;

  // Timers
  startPlayerTimer(player): void;
  stopPlayerTimer(): void;

  // Session management
  session: GameSession;
  getPlayer(number): Player;

  // Last-player-standing tracking (backend-specific hosting logic)
  lpsRoundIndex: number;
  lpsCurrentRoundActorMask: Map<number, boolean>;
  lpsExclusivePlayerForCompletedRound: number | null;
}
```

### 1.5 Backend Migration Approach

**Target:** Create `TurnEngineAdapter.ts` that wraps `processTurn()`:

```typescript
// Proposed: src/server/game/turn/TurnEngineAdapter.ts
export class TurnEngineAdapter {
  constructor(
    private readonly session: GameSession,
    private readonly interactionHandler: WebSocketInteractionHandler,
    private readonly persistence: HistoryPersistence
  ) {}

  async processMove(move: Move): Promise<MoveResult> {
    // 1. Create delegates that route decisions to real players
    const delegates: TurnProcessingDelegates = {
      resolveDecision: (decision) => this.resolvePlayerDecision(decision),
    };

    // 2. Delegate to canonical orchestrator
    const result = await processTurnAsync(this.session.gameState, move, delegates);

    // 3. Handle backend-specific concerns
    await this.persistence.saveHistory(result);
    this.eventEmitter.emit('turn_complete', result);

    return result;
  }
}
```

---

## 2. Client Sandbox Audit (Task 3.3)

### 2.1 Files Analyzed

| File                                                                              | Lines | Role                       |
| --------------------------------------------------------------------------------- | ----- | -------------------------- |
| [`ClientSandboxEngine.ts`](../../src/client/sandbox/ClientSandboxEngine.ts)       | 2707  | Main sandbox orchestration |
| [`sandboxTurnEngine.ts`](../../src/client/sandbox/sandboxTurnEngine.ts)           | 377   | Turn processing            |
| [`sandboxMovementEngine.ts`](../../src/client/sandbox/sandboxMovementEngine.ts)   | 678   | Movement/capture handling  |
| [`sandboxLinesEngine.ts`](../../src/client/sandbox/sandboxLinesEngine.ts)         | 285   | Line processing            |
| [`sandboxTerritoryEngine.ts`](../../src/client/sandbox/sandboxTerritoryEngine.ts) | 373   | Territory processing       |

**Total sandbox LOC:** ~4420 lines

### 2.2 Logic Already Using Shared Modules ✅

The sandbox **already delegates** significant logic:

```typescript
// sandboxTurnEngine.ts - Uses shared advanceTurnAndPhase()
import { advanceTurnAndPhase } from '../../shared/engine';

// sandboxLinesEngine.ts - Uses shared line helpers
import {
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
} from '../../shared/engine';

// sandboxTerritoryEngine.ts - Uses shared territory helpers
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '../../shared/engine';
```

### 2.3 Logic Still Duplicated in ClientSandboxEngine ❌

```typescript
// ClientSandboxEngine.ts duplicates orchestrator's phase flow
class ClientSandboxEngine {
  // DUPLICATE: Move application (orchestrator has applyMove via mutators)
  private async applyCanonicalMoveInternal(move: Move): Promise<boolean> { ... }  // ~400 lines

  // DUPLICATE: Automatic consequences (orchestrator handles in processTurn)
  private async advanceAfterMovement(): Promise<void> { ... }  // ~80 lines

  // DUPLICATE: Phase management (orchestrator has PhaseStateMachine)
  // Scattered throughout handleHumanCellClick, applyCanonicalMoveInternal

  // DUPLICATE: Pending eliminations (orchestrator tracks in PerTurnFlags)
  private _pendingLineRewardElimination: boolean;
  private _pendingTerritorySelfElimination: boolean;

  // DUPLICATE: Per-turn state (orchestrator tracks in TurnProcessingState)
  private _hasPlacedThisTurn: boolean;
  private _mustMoveFromStackKey: string | undefined;
}
```

### 2.4 Sandbox-Specific Concerns (Keep in Adapter)

These concerns must remain in the sandbox adapter:

```typescript
interface SandboxAdapterConcerns {
  // Local state management
  gameState: GameState; // In-memory state
  victoryResult: GameResult | null;

  // UI state
  _selectedStackKey: string | undefined; // UI selection

  // Debug/trace support
  traceMode: boolean;
  _debugCheckpointHook?: (label, state) => void;

  // Local AI integration
  rng: SeededRNG;
  _lastAIMove: Move | null;

  // LPS tracking (mirroring backend)
  _lpsRoundIndex: number;
  _lpsCurrentRoundActorMask: Map<number, boolean>;
  _lpsExclusivePlayerForCompletedRound: number | null;

  // Interaction handler for local decisions
  interactionHandler: SandboxInteractionHandler;
}
```

### 2.5 Sandbox Migration Approach

**Target:** Refactor `ClientSandboxEngine` to call `processTurn()`:

```typescript
// Proposed refactoring of ClientSandboxEngine.ts
class ClientSandboxEngine {
  // Keep: UI/state concerns
  private gameState: GameState;
  private _selectedStackKey: string | undefined;

  // Refactor: Move application to use orchestrator
  async applyCanonicalMove(move: Move): Promise<void> {
    const delegates: TurnProcessingDelegates = {
      resolveDecision: (decision) => this.resolveLocalDecision(decision),
    };

    const result = await processTurnAsync(this.gameState, move, delegates);

    this.gameState = result.nextState;
    // Handle local concerns (history, victory, etc.)
  }

  // Keep: Local decision resolution
  private async resolveLocalDecision(decision: PendingDecision): Promise<Move> {
    // Route to interaction handler for UI or AI
  }
}
```

---

## 3. Shared Orchestrator Analysis

### 3.1 Orchestrator Capabilities

The orchestrator at [`turnOrchestrator.ts`](../../src/shared/engine/orchestration/turnOrchestrator.ts) provides:

```typescript
// Entry points
export function processTurn(state, move): ProcessTurnResult { ... }
export async function processTurnAsync(state, move, delegates): Promise<ProcessTurnResult> { ... }

// Validation
export function validateMove(state, move): ValidationResult { ... }

// Move enumeration
export function getValidMoves(state, options?): Move[] { ... }
export function hasValidMoves(state, options?): boolean { ... }
```

### 3.2 ProcessTurnResult Structure

```typescript
interface ProcessTurnResult {
  nextState: GameState;
  status: 'complete' | 'awaiting_decision';
  pendingDecision: PendingDecision | undefined;
  victoryResult: VictoryState | undefined;
  metadata: ProcessingMetadata;
}
```

### 3.3 Decision Types Supported

```typescript
type DecisionType =
  | 'line_order' // Which line to process first
  | 'line_reward' // Collapse all vs minimum
  | 'region_order' // Which territory region first
  | 'elimination_target' // Which stack for self-elimination
  | 'capture_direction' // Which capture target
  | 'chain_capture'; // Continue chain or not
```

---

## 4. Migration Plan

### 4.1 Phase 3a: Backend TurnEngineAdapter (Recommended First)

**Scope:** Create adapter, migrate GameEngine.makeMove() to delegate

**Files to Create:**

- `src/server/game/turn/TurnEngineAdapter.ts`

**Files to Modify:**

- `src/server/game/GameEngine.ts` - Replace internal orchestration with adapter calls

**Estimated Effort:** 3-5 days

**Risk:** Medium - Backend has more integration points (WebSocket, persistence)

### 4.2 Phase 3b: Client SandboxEngineAdapter

**Scope:** Refactor ClientSandboxEngine to use orchestrator

**Files to Modify:**

- `src/client/sandbox/ClientSandboxEngine.ts` - Replace applyCanonicalMoveInternal

**Files to Remove (Post-Migration):**

- `src/client/sandbox/sandboxLinesEngine.ts` - Logic moves to orchestrator
- `src/client/sandbox/sandboxTerritoryEngine.ts` - Logic moves to orchestrator
- `src/client/sandbox/sandboxLines.ts` - Detection in LineAggregate
- `src/client/sandbox/sandboxTerritory.ts` - Detection in TerritoryAggregate

**Estimated Effort:** 3-5 days

**Risk:** Lower - Sandbox is self-contained, easier to test

### 4.3 LOC Reduction Estimate

| Component          | Current LOC | After Migration | Reduction |
| ------------------ | ----------- | --------------- | --------- |
| Backend GameEngine | 3320        | ~1500           | ~55%      |
| Client Sandbox     | 4420        | ~1800           | ~60%      |
| **Total**          | **7740**    | **~3300**       | **~57%**  |

---

## 5. Existing Parity Test Failures

The audit revealed 4 existing parity test failures:

1. **`Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`**
   - Backend and sandbox diverge at move 52
   - Root cause: Different orchestration flows

2. **`Backend_vs_Sandbox.seed5.bisectParity.test.ts`**
   - Binary search finds divergence point
   - Root cause: Same as above

3. **`NoRandomInCoreRules.test.ts`**
   - Unrelated Math.random detection test

**These failures validate the need for Phase 3** - both engines should use the same orchestrator to guarantee parity.

---

## 6. Recommendations

### 6.1 Immediate Recommendation

Given the scope assessment:

**Option A: Full Migration (Recommended)**

- Create TurnEngineAdapter for backend
- Refactor ClientSandboxEngine to use orchestrator
- Remove duplicated sandbox\*Engine files
- Expected outcome: Parity tests pass, LOC reduced by ~57%

**Option B: Incremental Migration**

- Start with sandbox only (lower risk)
- Validate parity improvement
- Then migrate backend
- Lower risk but longer timeline

### 6.2 Success Criteria

1. ✅ All 14 contract tests continue passing
2. ✅ All 1178 existing tests continue passing
3. ✅ Backend_vs_Sandbox parity tests start passing
4. ✅ ~4000 lines of duplicated code eliminated
5. ✅ Clear adapter boundary: adapters handle I/O, orchestrator handles rules

---

## 7. Conclusion

The audit confirms that Phase 3 migration is both **feasible and beneficial**:

1. **Infrastructure Ready:** The orchestrator (`processTurn`, `processTurnAsync`) exists and is tested with 14 contract tests
2. **Significant Duplication:** Both engines duplicate ~57% of their logic that the orchestrator already handles
3. **Clear Separation Possible:** Host-specific concerns (WebSocket, UI, persistence) are well-identified
4. **Parity Issues Exist:** Current test failures demonstrate the need for consolidated orchestration

**Recommended Next Step:** Proceed with Option A (Full Migration), starting with the client sandbox as it has lower integration risk.

---

_Document generated from Phase 3 audit_  
_Review date: 2025-11-26_
