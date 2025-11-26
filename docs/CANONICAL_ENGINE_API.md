# Canonical Engine Public API

**Task:** T1-W1-B  
**Date:** 2025-11-26  
**Status:** Complete

## 1. Overview

This document defines the narrow, stable public API boundary for the canonical RingRift rules engine located in [`src/shared/engine/`](../src/shared/engine/). This API is designed to be:

1. **Narrow** - Only essential functions are exported
2. **Stable** - Changes inside the engine don't break adapters
3. **Domain-Driven** - Organized by game domain (Placement, Movement, Capture, Line, Territory, Victory)
4. **Type-Safe** - All inputs/outputs have explicit TypeScript types
5. **Pure** - No side effects; state passed in and returned out

### Design Philosophy

The canonical engine implements RingRift rules as **pure functions** that transform [`GameState`](../src/shared/types/game.ts:471) immutably. Adapters (Server GameEngine, Client Sandbox, Python AI Service) are responsible for:

- Orchestrating turn/phase flow
- Managing player interaction (choices, timeouts)
- Persistence and networking
- AI integration

The engine itself knows nothing about WebSockets, databases, or user interfaces.

---

## 2. Type Exports

### 2.1 Core Types (from [`src/shared/types/game.ts`](../src/shared/types/game.ts))

```typescript
// Board & Position
export type { Position, BoardType, BoardState, BoardConfig };
export type { RingStack, MarkerInfo, MarkerType };

// Game State
export type { GameState, GameStatus, GamePhase };
export type { Player, PlayerType, AIProfile };
export type { TimeControl };

// Moves & Actions
export type { Move, MoveType, MovePayload };
export type { LineInfo, Territory };

// Configuration
export { BOARD_CONFIGS };

// Utilities
export { positionToString, stringToPosition, positionsEqual };
```

### 2.2 Engine Types (from [`src/shared/engine/types.ts`](../src/shared/engine/types.ts))

```typescript
// Validation
export type { ValidationResult };

// Actions (for host-level action dispatch)
export type { GameAction, ActionType };
export type { PlaceRingAction, MoveStackAction, OvertakingCaptureAction };
export type { ProcessLineAction, ChooseLineRewardAction };
export type { ProcessTerritoryAction, EliminateStackAction };
export type { SkipPlacementAction };

// Generics
export type { Validator, Mutator };
```

### 2.3 Domain Result Types

```typescript
// Turn/Phase
export type { TurnAdvanceResult, PerTurnState } from './turnLogic';
export type { TurnLogicDelegates } from './turnLogic';

// Victory
export type { VictoryResult, VictoryReason } from './victoryLogic';

// Placement
export type { PlacementContext, PlacementValidationResult } from './validators/PlacementValidator';
export type { PlacementApplicationOutcome } from './placementHelpers';

// Movement
export type { SimpleMoveTarget, MovementBoardAdapters } from './movementLogic';

// Capture
export type { CaptureBoardAdapters } from './captureLogic';

// Line Processing
export type { LineEnumerationOptions, LineDecisionApplicationOutcome } from './lineDecisionHelpers';

// Territory Processing
export type { TerritoryProcessingContext, TerritoryProcessingOutcome } from './territoryProcessing';
export type {
  TerritoryEnumerationOptions,
  TerritoryProcessApplicationOutcome,
} from './territoryDecisionHelpers';
export type {
  TerritoryEliminationScope,
  EliminateRingsFromStackOutcome,
} from './territoryDecisionHelpers';

// Board Views (for adapters implementing board interfaces)
export type { MovementBoardView, CaptureSegmentBoardView } from './core';
export type { MarkerPathHelpers } from './core';

// Progress Tracking
export type { ProgressSnapshot, BoardSummary, GameHistoryEntry, GameTrace } from '../types/game';
```

---

## 3. Function Exports by Domain

### 3.1 Placement Domain

**Location:** [`validators/PlacementValidator.ts`](../src/shared/engine/validators/PlacementValidator.ts), [`placementHelpers.ts`](../src/shared/engine/placementHelpers.ts)

```typescript
// Validation (Board-Level)
validatePlacementOnBoard(
  board: BoardState,
  to: Position,
  requestedCount: number,
  ctx: PlacementContext
): PlacementValidationResult

// Validation (GameState-Level)
validatePlacement(state: GameState, action: PlaceRingAction): ValidationResult
validateSkipPlacement(state: GameState, action: SkipPlacementAction): ValidationResult

// Application (TODO: stub in placementHelpers.ts)
applyPlacementMove(state: GameState, move: Move): PlacementApplicationOutcome
evaluateSkipPlacementEligibility(state: GameState, player: number): SkipPlacementEligibilityResult
```

**Usage Pattern:**

```typescript
import {
  validatePlacementOnBoard,
  PlacementContext,
} from '@shared/engine/validators/PlacementValidator';

const ctx: PlacementContext = {
  boardType: state.board.type,
  player: currentPlayer,
  ringsInHand: player.ringsInHand,
  ringsPerPlayerCap: BOARD_CONFIGS[state.board.type].ringsPerPlayer,
};

const result = validatePlacementOnBoard(state.board, targetPos, 1, ctx);
if (result.valid) {
  // Apply placement via mutator or custom logic
}
```

---

### 3.2 Movement Domain

**Location:** [`movementLogic.ts`](../src/shared/engine/movementLogic.ts), [`validators/MovementValidator.ts`](../src/shared/engine/validators/MovementValidator.ts)

```typescript
// Enumeration
enumerateSimpleMoveTargetsFromStack(
  boardType: BoardType,
  from: Position,
  player: number,
  board: MovementBoardAdapters
): SimpleMoveTarget[]

// Validation
validateMovement(state: GameState, action: MoveStackAction): ValidationResult

// Reachability Check (used for no-dead-placement and forced elimination)
hasAnyLegalMoveOrCaptureFromOnBoard(
  boardType: BoardType,
  from: Position,
  player: number,
  board: MovementBoardView,
  options?: { maxNonCaptureDistance?: number; maxCaptureLandingDistance?: number }
): boolean
```

**Usage Pattern:**

```typescript
import { enumerateSimpleMoveTargetsFromStack, MovementBoardAdapters } from '@shared/engine/movementLogic';

const adapters: MovementBoardAdapters = {
  isValidPosition: (pos) => isValidPosition(pos, boardType, board.size),
  isCollapsedSpace: (pos) => board.collapsedSpaces.has(positionToString(pos)),
  getStackAt: (pos) => /* stack lookup */,
  getMarkerOwner: (pos) => /* marker lookup */,
};

const moves = enumerateSimpleMoveTargetsFromStack(boardType, fromPos, player, adapters);
```

---

### 3.3 Capture Domain

**Location:** [`captureLogic.ts`](../src/shared/engine/captureLogic.ts), [`core.ts`](../src/shared/engine/core.ts), [`validators/CaptureValidator.ts`](../src/shared/engine/validators/CaptureValidator.ts)

```typescript
// Enumeration
enumerateCaptureMoves(
  boardType: BoardType,
  from: Position,
  playerNumber: number,
  adapters: CaptureBoardAdapters,
  moveNumber: number
): Move[]

// Validation (single segment)
validateCaptureSegmentOnBoard(
  boardType: BoardType,
  from: Position,
  target: Position,
  landing: Position,
  player: number,
  board: CaptureSegmentBoardView
): boolean

// Validation (GameState-level)
validateCapture(state: GameState, action: OvertakingCaptureAction): ValidationResult
```

**Usage Pattern:**

```typescript
import { enumerateCaptureMoves, CaptureBoardAdapters } from '@shared/engine/captureLogic';

const adapters: CaptureBoardAdapters = {
  isValidPosition: (pos) => /* bounds check */,
  isCollapsedSpace: (pos) => /* collapsed lookup */,
  getStackAt: (pos) => /* stack lookup with capHeight */,
  getMarkerOwner: (pos) => /* marker lookup */,
};

const captures = enumerateCaptureMoves(boardType, fromPos, player, adapters, moveNumber);
```

---

### 3.4 Line Domain

**Location:** [`lineDetection.ts`](../src/shared/engine/lineDetection.ts), [`lineDecisionHelpers.ts`](../src/shared/engine/lineDecisionHelpers.ts)

```typescript
// Detection
findAllLines(board: BoardState): LineInfo[]
findLinesForPlayer(board: BoardState, playerNumber: number): LineInfo[]

// Decision Enumeration
enumerateProcessLineMoves(
  state: GameState,
  player: number,
  options?: LineEnumerationOptions
): Move[]

enumerateChooseLineRewardMoves(
  state: GameState,
  player: number,
  lineIndex: number
): Move[]

// Application
applyProcessLineDecision(state: GameState, move: Move): LineDecisionApplicationOutcome
applyChooseLineRewardDecision(state: GameState, move: Move): LineDecisionApplicationOutcome
```

**Result Type:**

```typescript
interface LineDecisionApplicationOutcome {
  nextState: GameState;
  pendingLineRewardElimination: boolean;
}
```

**Usage Pattern:**

```typescript
import { findLinesForPlayer } from '@shared/engine/lineDetection';
import {
  enumerateProcessLineMoves,
  applyProcessLineDecision,
} from '@shared/engine/lineDecisionHelpers';

// Detect lines for current player
const lines = findLinesForPlayer(state.board, currentPlayer);

if (lines.length > 0) {
  // Enumerate decision moves
  const lineMoves = enumerateProcessLineMoves(state, currentPlayer);

  // Let player/AI choose a move
  const chosenMove = await getPlayerChoice(lineMoves);

  // Apply the decision
  const { nextState, pendingLineRewardElimination } = applyProcessLineDecision(state, chosenMove);
}
```

---

### 3.5 Territory Domain

**Location:** [`territoryDetection.ts`](../src/shared/engine/territoryDetection.ts), [`territoryProcessing.ts`](../src/shared/engine/territoryProcessing.ts), [`territoryDecisionHelpers.ts`](../src/shared/engine/territoryDecisionHelpers.ts)

```typescript
// Detection
findDisconnectedRegions(board: BoardState): Territory[]

// Processability Check
canProcessTerritoryRegion(
  board: BoardState,
  region: Territory,
  ctx: TerritoryProcessingContext
): boolean

filterProcessableTerritoryRegions(
  board: BoardState,
  regions: Territory[],
  ctx: TerritoryProcessingContext
): Territory[]

getProcessableTerritoryRegions(
  board: BoardState,
  ctx: TerritoryProcessingContext
): Territory[]

// Board-Level Application
applyTerritoryRegion(
  board: BoardState,
  region: Territory,
  ctx: TerritoryProcessingContext
): TerritoryProcessingOutcome

// Decision Enumeration
enumerateProcessTerritoryRegionMoves(
  state: GameState,
  player: number,
  options?: TerritoryEnumerationOptions
): Move[]

enumerateTerritoryEliminationMoves(
  state: GameState,
  player: number,
  scope?: TerritoryEliminationScope
): Move[]

// Application
applyProcessTerritoryRegionDecision(
  state: GameState,
  move: Move
): TerritoryProcessApplicationOutcome

applyEliminateRingsFromStackDecision(
  state: GameState,
  move: Move
): EliminateRingsFromStackOutcome
```

**Result Types:**

```typescript
interface TerritoryProcessApplicationOutcome {
  nextState: GameState;
  processedRegionId: string;
  processedRegion: Territory;
  pendingSelfElimination: boolean;
}

interface EliminateRingsFromStackOutcome {
  nextState: GameState;
}
```

**Usage Pattern:**

```typescript
import { findDisconnectedRegions } from '@shared/engine/territoryDetection';
import {
  getProcessableTerritoryRegions,
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  enumerateTerritoryEliminationMoves,
  applyEliminateRingsFromStackDecision,
} from '@shared/engine/territoryDecisionHelpers';

// Find processable regions for current player
const regions = getProcessableTerritoryRegions(state.board, { player: currentPlayer });

if (regions.length > 0) {
  const regionMoves = enumerateProcessTerritoryRegionMoves(state, currentPlayer);
  const chosenMove = await getPlayerChoice(regionMoves);

  const { nextState, pendingSelfElimination } = applyProcessTerritoryRegionDecision(
    state,
    chosenMove
  );

  if (pendingSelfElimination) {
    const elimMoves = enumerateTerritoryEliminationMoves(nextState, currentPlayer);
    const elimChoice = await getPlayerChoice(elimMoves);
    const { nextState: finalState } = applyEliminateRingsFromStackDecision(nextState, elimChoice);
  }
}
```

---

### 3.6 Victory Domain

**Location:** [`victoryLogic.ts`](../src/shared/engine/victoryLogic.ts)

```typescript
// Evaluation
evaluateVictory(state: GameState): VictoryResult

// Tie-breaker Helper
getLastActor(state: GameState): number | undefined
```

**Result Type:**

```typescript
interface VictoryResult {
  isGameOver: boolean;
  winner?: number;
  reason?: VictoryReason;
  handCountsAsEliminated?: boolean;
}

type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'game_completed';
```

**Usage Pattern:**

```typescript
import { evaluateVictory } from '@shared/engine/victoryLogic';

const result = evaluateVictory(state);
if (result.isGameOver) {
  // Handle game end
  console.log(`Game over: ${result.reason}, winner: ${result.winner}`);
}
```

---

### 3.7 Turn Management

**Location:** [`turnLogic.ts`](../src/shared/engine/turnLogic.ts)

```typescript
// Turn/Phase Progression
advanceTurnAndPhase(
  state: GameState,
  turn: PerTurnState,
  delegates: TurnLogicDelegates
): TurnAdvanceResult
```

**Types:**

```typescript
interface PerTurnState {
  hasPlacedThisTurn: boolean;
  mustMoveFromStackKey?: string;
}

interface TurnAdvanceResult {
  nextState: GameState;
  nextTurn: PerTurnState;
}

interface TurnLogicDelegates {
  getPlayerStacks(
    state: GameState,
    player: number
  ): Array<{ position: Position; stackHeight: number }>;
  hasAnyPlacement(state: GameState, player: number): boolean;
  hasAnyMovement(state: GameState, player: number, turn: PerTurnState): boolean;
  hasAnyCapture(state: GameState, player: number, turn: PerTurnState): boolean;
  applyForcedElimination(state: GameState, player: number): GameState;
  getNextPlayerNumber(state: GameState, current: number): number;
}
```

**Usage Pattern:**

```typescript
import { advanceTurnAndPhase, PerTurnState, TurnLogicDelegates } from '@shared/engine/turnLogic';

const turnState: PerTurnState = { hasPlacedThisTurn: false };

const delegates: TurnLogicDelegates = {
  getPlayerStacks: (s, p) => /* enumerate player stacks */,
  hasAnyPlacement: (s, p) => /* check placement availability */,
  hasAnyMovement: (s, p, t) => /* check movement availability */,
  hasAnyCapture: (s, p, t) => /* check capture availability */,
  applyForcedElimination: (s, p) => /* handle forced elimination */,
  getNextPlayerNumber: (s, c) => /* seat order logic */,
};

const { nextState, nextTurn } = advanceTurnAndPhase(state, turnState, delegates);
```

---

### 3.8 Core Utilities

**Location:** [`core.ts`](../src/shared/engine/core.ts)

```typescript
// Geometry
getMovementDirectionsForBoardType(boardType: BoardType): Direction[]
getPathPositions(from: Position, to: Position): Position[]
calculateDistance(boardType: BoardType, from: Position, to: Position): number

// Stack Calculations
calculateCapHeight(rings: number[]): number
countRingsOnBoardForPlayer(board: BoardState, playerNumber: number): number
countRingsInPlayForPlayer(state: GameState, playerNumber: number): number

// Marker Effects
applyMarkerEffectsAlongPathOnBoard(
  board: BoardState,
  from: Position,
  to: Position,
  playerNumber: number,
  helpers: MarkerPathHelpers,
  options?: { leaveDepartureMarker?: boolean }
): void

// State Hashing & Debugging
hashGameState(state: GameState): string
summarizeBoard(board: BoardState): BoardSummary
computeProgressSnapshot(state: GameState): ProgressSnapshot
```

---

## 4. Internal vs External Boundary

### 4.1 Internal (Not Exported)

The following remain internal implementation details:

| Category                    | Examples                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| **Geometry Internals**      | `getNeighbors()`, `isValidPosition()`, `generateValidPositions()` (in territoryDetection) |
| **Detection Internals**     | `exploreRegionWithBorderColor()`, `getRepresentedPlayers()`, `findLineInDirection()`      |
| **Caching**                 | `adjacencyCache` in territoryDetection                                                    |
| **Board Cloning**           | `cloneBoard()` helpers in processing modules                                              |
| **Move Number Computation** | `computeNextMoveNumber()` in decision helpers                                             |
| **Line Matching**           | `canonicalLineKey()`, `resolveLineForMove()` in lineDecisionHelpers                       |
| **Collapse Internals**      | `collapseLinePositions()` in lineDecisionHelpers                                          |

### 4.2 Future Extensions (Reserved)

| Module                            | Purpose                              |
| --------------------------------- | ------------------------------------ |
| `lpsTracker.ts` (proposed)        | Last Player Standing state machine   |
| `gameStateMachine.ts` (proposed)  | Unified game lifecycle state machine |
| `chainCaptureLogic.ts` (proposed) | Chain capture state management       |

---

## 5. Adapter Requirements

Adapters (Server GameEngine, Client Sandbox, Python AI Service) must handle:

### 5.1 Player Interaction

| Requirement             | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Choice Presentation** | Surface `PlayerChoice` objects for line order, region order, elimination choices |
| **Timeout Handling**    | Enforce decision timeouts with auto-selection fallback                           |
| **AI Integration**      | Route AI turn decisions through appropriate engines                              |

### 5.2 State Management

| Requirement           | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| **Board Adapters**    | Implement `MovementBoardView`, `CaptureSegmentBoardView` interfaces |
| **Turn Delegates**    | Implement `TurnLogicDelegates` for `advanceTurnAndPhase()`          |
| **History Recording** | Build `GameHistoryEntry` objects from applied moves                 |

### 5.3 Infrastructure

| Requirement           | Description                                             |
| --------------------- | ------------------------------------------------------- |
| **Persistence**       | Save/restore GameState to database or local storage     |
| **WebSocket Events**  | Emit state changes, move notifications, choice requests |
| **Parity Validation** | (Python) Shadow contract validation against TS engine   |

---

## 6. API Stability Guarantees

### 6.1 Semantic Versioning

| Change Type                       | Compatibility                                |
| --------------------------------- | -------------------------------------------- |
| **Parameter Addition** (optional) | Backward compatible                          |
| **New Function Export**           | Backward compatible                          |
| **Return Type Extension**         | Backward compatible (adding fields)          |
| **Parameter Removal**             | Breaking change                              |
| **Type Signature Change**         | Breaking change                              |
| **Behavioral Change**             | Breaking change (requires parity validation) |

### 6.2 Deprecation Process

1. Mark deprecated functions with `@deprecated` JSDoc
2. Add console warnings in development builds
3. Maintain deprecated functions for at least one release cycle
4. Remove in subsequent major version

---

## 7. Example: Complete Turn Flow

```typescript
import { GameState } from '@shared/types/game';
import { validatePlacementOnBoard } from '@shared/engine/validators/PlacementValidator';
import { enumerateSimpleMoveTargetsFromStack } from '@shared/engine/movementLogic';
import { enumerateCaptureMoves } from '@shared/engine/captureLogic';
import {
  findLinesForPlayer,
  enumerateProcessLineMoves,
  applyProcessLineDecision,
} from '@shared/engine/lineDecisionHelpers';
import {
  getProcessableTerritoryRegions,
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '@shared/engine/territoryDecisionHelpers';
import { evaluateVictory } from '@shared/engine/victoryLogic';
import { advanceTurnAndPhase } from '@shared/engine/turnLogic';

async function executeTurn(state: GameState, player: number): Promise<GameState> {
  let current = state;

  // 1. Ring Placement Phase
  if (current.currentPhase === 'ring_placement') {
    const placementMove = await getPlacementFromPlayer(current, player);
    if (placementMove) {
      current = applyPlacement(current, placementMove);
    }
    current = advancePhase(current);
  }

  // 2. Movement/Capture Phase
  if (current.currentPhase === 'movement') {
    const moveOrCapture = await getMovementFromPlayer(current, player);
    current = applyMovementOrCapture(current, moveOrCapture);
    current = advancePhase(current);
  }

  // 3. Line Processing Phase
  while (findLinesForPlayer(current.board, player).length > 0) {
    const lineMoves = enumerateProcessLineMoves(current, player);
    const lineChoice = await getPlayerChoice(lineMoves);
    const { nextState } = applyProcessLineDecision(current, lineChoice);
    current = nextState;
  }
  current = advancePhase(current);

  // 4. Territory Processing Phase
  while (getProcessableTerritoryRegions(current.board, { player }).length > 0) {
    const regionMoves = enumerateProcessTerritoryRegionMoves(current, player);
    const regionChoice = await getPlayerChoice(regionMoves);
    const { nextState, pendingSelfElimination } = applyProcessTerritoryRegionDecision(
      current,
      regionChoice
    );
    current = nextState;

    if (pendingSelfElimination) {
      current = await handleSelfElimination(current, player);
    }
  }

  // 5. Victory Check
  const victory = evaluateVictory(current);
  if (victory.isGameOver) {
    return markGameComplete(current, victory);
  }

  // 6. Advance to Next Player
  return advanceToNextPlayer(current);
}
```

---

## 8. Migration Notes

### 8.1 For Server GameEngine

The Server GameEngine already delegates extensively to shared helpers. Remaining work:

1. Replace internal turn-state tracking with `advanceTurnAndPhase()`
2. Adopt `enumerateProcessLineMoves()` / `applyProcessLineDecision()` for line phase
3. Adopt `enumerateProcessTerritoryRegionMoves()` / `applyProcessTerritoryRegionDecision()` for territory phase

### 8.2 For Client Sandbox

The Client Sandbox similarly delegates to shared helpers. Remaining work:

1. Replace `sandboxLinesEngine` internals with shared line helpers
2. Replace `sandboxTerritoryEngine` internals with shared territory helpers
3. Consolidate turn progression logic via `advanceTurnAndPhase()`

### 8.3 For Python AI Service

The Python AI Service currently duplicates rules logic. Long-term options:

1. **Shadow Contracts** (current): Validate against TS engine at runtime
2. **Code Generation**: Generate Python from TS source
3. **WASM Compilation**: Compile TS engine to WASM for Python consumption

---

## Conclusion

This API specification defines a stable, narrow boundary for the canonical RingRift rules engine. By organizing exports by game domain and maintaining pure function semantics, adapters can rely on predictable behavior while the internal implementation evolves.

**Next Steps:**

1. T1-W1-C: Implement stubbed functions (`applyPlacementMove`, `evaluateSkipPlacementEligibility`)
2. T1-W1-D: Create adapter layer specifications for each host
3. T2: Address Python parity via code generation or WASM
