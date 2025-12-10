# Turn Orchestration Layer

This module provides the canonical turn processing orchestrator for the RingRift game engine. It serves as the single entry point for all turn processing, delegating to domain aggregates for actual game logic.

## Architecture Overview

```
processTurn(state, move)
        │
        ▼
┌───────────────────────────────────────────────────┐
│              Turn Orchestrator                     │
│  ┌─────────────────────────────────────────────┐  │
│  │         PhaseStateMachine                   │  │
│  │  ring_placement → movement → capture →      │  │
│  │  line_processing → territory_processing     │  │
│  └─────────────────────────────────────────────┘  │
│                        │                          │
│                        ▼                          │
│  ┌─────────────────────────────────────────────┐  │
│  │          Domain Aggregates                  │  │
│  │  • PlacementAggregate                       │  │
│  │  • MovementAggregate                        │  │
│  │  • CaptureAggregate                         │  │
│  │  • LineAggregate                            │  │
│  │  • TerritoryAggregate                       │  │
│  │  • VictoryAggregate                         │  │
│  └─────────────────────────────────────────────┘  │
│                        │                          │
│                        ▼                          │
│           ProcessTurnResult                       │
└───────────────────────────────────────────────────┘
```

## Key Files

- **turnOrchestrator.ts** - Main `processTurn` and `processTurnAsync` entry points
- **phaseStateMachine.ts** - Manages game phase transitions (deprecated, being replaced by FSM)
- **types.ts** - Type definitions for orchestration layer
- **../fsm/FSMAdapter.ts** - FSM-based validation and orchestration (canonical)

## FSM Integration

The turn orchestrator uses FSM (Finite State Machine) as the **canonical** source for:

1. **Move validation** - `validateMoveWithFSM()` validates moves per RR-CANON-R070/R075
2. **Phase transitions** - `computeFSMOrchestration()` determines next phase and player
3. **Decision timing** - FSM's `pendingDecisionType` drives when decisions are needed

### Validation

FSM validation is enforced in `processTurn`. The legacy `validateMove()` function is deprecated:

```typescript
// Preferred: Use FSM validation directly
import { validateMoveWithFSM } from '../fsm';

const result = validateMoveWithFSM(gameState, move);
if (!result.valid) {
  console.error('Invalid move:', result.reason, 'errorCode:', result.errorCode);
}

// Deprecated: Legacy validation (maintained for backward compatibility)
import { validateMove } from './orchestration/turnOrchestrator';
const validation = validateMove(gameState, proposedMove); // @deprecated
```

### FSMDecisionSurface

`ProcessTurnResult` includes an optional `fsmDecisionSurface` field exposing raw FSM orchestration data:

```typescript
interface FSMDecisionSurface {
  pendingDecisionType?: 'chain_capture' | 'line_order_required' | ...;
  pendingLines?: Array<{ positions: Position[]; player?: number }>;
  pendingRegions?: Array<{ positions: Position[]; eliminationsRequired?: number }>;
  chainContinuations?: Array<{ target: Position }>;
  forcedEliminationCount?: number;
}
```

This is useful for advanced host implementations that need direct access to FSM data.

## Usage

### Basic Turn Processing (Synchronous)

```typescript
import { processTurn } from './orchestration/turnOrchestrator';

// Process a ring placement move
const result = processTurn(gameState, {
  id: 'move-1',
  type: 'place_ring',
  player: 1,
  to: { x: 3, y: 3 },
  timestamp: new Date(),
  thinkTime: 0,
  moveNumber: 1,
});

if (result.status === 'complete') {
  // Turn finished, proceed to next player
  const nextState = result.nextState;
} else if (result.status === 'awaiting_decision') {
  // Player needs to make a decision (e.g., choose line order)
  const decision = result.pendingDecision;
  // Present options to player and call processTurn with chosen option
}
```

### Async Turn Processing (With Decision Resolution)

For human players where decisions need async resolution:

```typescript
import { processTurnAsync } from './orchestration/turnOrchestrator';
import type { TurnProcessingDelegates } from './orchestration/types';

const delegates: TurnProcessingDelegates = {
  // Called when a decision is needed
  resolveDecision: async (decision) => {
    // Present decision.options to the player via UI
    // Return the chosen Move
    return await promptPlayerForDecision(decision);
  },

  // Optional: called for processing events
  onProcessingEvent: (event) => {
    console.log('Processing event:', event.type);
  },
};

const result = await processTurnAsync(gameState, move, delegates);
// Result will be complete (all decisions resolved)
```

### Getting Valid Moves

```typescript
import { getValidMoves, validateMove } from './orchestration/turnOrchestrator';

// Get legal moves for the current player/phase.
// - In decision phases (line/territory), this is **interactive moves only**;
//   when a phase has no interactive actions, the orchestrator instead
//   returns a PendingDecision of type `no_*_action_required` and hosts
//   must construct/apply the corresponding `no_*_action` Move.
// - In ring_placement/movement, the current implementation still surfaces
//   canonical `no_placement_action` / `no_movement_action` bookkeeping
//   moves when no interactive actions exist; these are being migrated
//   toward the same PendingDecision pattern (RR-CANON-R075/R076).
const validMoves = getValidMoves(gameState);

// Validate a specific move
const validation = validateMove(gameState, proposedMove);
if (!validation.valid) {
  console.error('Invalid move:', validation.reason);
}
```

## ProcessTurnResult

The result of `processTurn` contains:

```typescript
interface ProcessTurnResult {
  // The updated game state after processing
  nextState: GameState;

  // 'complete' = turn finished, 'awaiting_decision' = needs player input
  status: 'complete' | 'awaiting_decision';

  // Present when status === 'awaiting_decision'
  pendingDecision?: PendingDecision;

  // Present when game ends
  victoryResult?: VictoryState;

  // Processing metadata (timings, phases traversed, etc.)
  metadata: ProcessingMetadata;

  // FSM-derived decision surface (for advanced host implementations)
  // Contains raw FSM orchestration data for debugging or custom handling
  fsmDecisionSurface?: FSMDecisionSurface;
}
```

## Pending Decisions

When the orchestrator encounters a situation requiring player choice, it returns a `PendingDecision`:

```typescript
interface PendingDecision {
  // Type of decision needed
  type:
    | 'line_order'
    | 'line_reward'
    | 'region_order'
    | 'elimination_target'
    | 'capture_direction'
    | 'chain_capture'
    // RR-CANON-R075/R076: required no-action bookkeeping when a phase has
    // no interactive moves; hosts must synthesize the corresponding
    // `no_*_action` Move and apply it via the public API.
    | 'no_line_action_required'
    | 'no_territory_action_required'
    | 'no_movement_action_required'
    | 'no_placement_action_required';

  // Which player needs to decide
  player: number;

  // Available options (as Move objects). For `no_*_action_required`
  // decision types this array is intentionally empty; hosts must build
  // the appropriate `no_*_action` Move themselves.
  options: Move[];

  // Context for UI presentation
  context?: {
    description?: string;
    relevantPositions?: Position[];
  };
}
```

### Decision Types

1. **line_order** – Multiple lines detected, player chooses which to process first.
2. **line_reward** – A processed line grants a reward; player chooses which reward option to take.
3. **region_order** – Multiple disconnected Territory regions, player chooses processing order.
4. **elimination_target** – Player earned elimination and must choose target stack.
5. **capture_direction** – (When surfaced) player chooses among multiple capture directions.
6. **chain_capture** – Chain capture in progress, player chooses next capture segment.
7. **no_line_action_required** – No line decisions exist; host must emit an explicit `no_line_action` move to record the phase.
8. **no_territory_action_required** – No Territory decisions exist; host must emit an explicit `no_territory_action` move.
9. **no_movement_action_required** – No legal movement/capture exists; host must emit `no_movement_action`.
10. **no_placement_action_required** – No legal placements exist; host must emit `no_placement_action`.

## Creating TurnProcessingDelegates from Aggregates

The orchestrator already imports and uses the aggregates internally. For custom integrations:

```typescript
import {
  validatePlacement,
  mutatePlacement,
  enumeratePlacementPositions,
} from '../aggregates/PlacementAggregate';

import {
  validateMovement,
  enumerateSimpleMovesForPlayer,
  applySimpleMovement,
} from '../aggregates/MovementAggregate';

// Similar imports for Capture, Line, Territory, Victory aggregates
```

Each aggregate provides:

- `validate*` - Validate actions/moves
- `enumerate*` - List available moves
- `apply*` / `mutate*` - Execute mutations

## Test Vector Validation

The orchestrator behavior is validated against contract test vectors:

```bash
# Location of test vectors
tests/fixtures/contract-vectors/v2/
  ├── placement.vectors.json
  ├── movement.vectors.json
  ├── capture.vectors.json
  ├── line_detection.vectors.json
  └── territory.vectors.json

# Run contract tests
npm test -- tests/contracts/contractVectorRunner.test.ts
```

Test vectors define input states, moves, and expected outputs in a canonical JSON format that can also be used for cross-engine parity testing with the Python implementation.

## S-Invariant

The orchestrator tracks the S-invariant for debugging:

```
S = markers + collapsedSpaces + eliminatedRings
```

This value should be non-decreasing across turns (monotonic increase with game progress).

Implementation details:

- S is computed via the shared helper `computeProgressSnapshot(state).S` from `src/shared/engine/core.ts`.
- Each call to `processTurn` records `sInvariantBefore` and `sInvariantAfter` in `ProcessingMetadata`,
  which is exposed both to hosts (e.g. `GameEngine` via `TurnEngineAdapter`) and to contract
  schemas (`ProcessingMetadata` in `src/shared/engine/contracts/validators.ts`).
- These fields are used by:
  - the orchestrator invariant soak harness (`scripts/run-orchestrator-soak.ts`);
  - S-invariant regression tests (`tests/unit/OrchestratorSInvariant.regression.test.ts`);
  - contract vectors (`sInvariantDelta` assertions in `tests/fixtures/contract-vectors/v2/**`).

## Phase State Machine

The turn processing phases follow this flow:

```
ring_placement
      │
      ▼ (place/skip)
  movement ◄──────────┐
      │               │
      ▼ (move/capture)│
   capture ──────────►│ (chain continues)
      │               │
      ▼ (chain ends)  │
line_processing ◄─────┘
      │
      ▼ (lines processed)
territory_processing
      │
      ▼ (regions processed)
  [end turn or victory check]
```

## Error Handling

Invalid moves throw errors with descriptive messages:

```typescript
try {
  const result = processTurn(state, invalidMove);
} catch (error) {
  // error.message contains validation failure reason
}
```

For validation before processing, use `validateMove`:

```typescript
const validation = validateMove(state, move);
if (!validation.valid) {
  // Handle invalid move without throwing
  console.error(validation.reason);
}
```
