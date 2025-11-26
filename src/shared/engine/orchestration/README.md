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
- **phaseStateMachine.ts** - Manages game phase transitions
- **types.ts** - Type definitions for orchestration layer

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

// Get all valid moves for current player and phase
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
}
```

## Pending Decisions

When the orchestrator encounters a situation requiring player choice, it returns a `PendingDecision`:

```typescript
interface PendingDecision {
  // Type of decision needed
  type: 'line_order' | 'region_order' | 'elimination_target' | 'chain_capture';

  // Which player needs to decide
  player: number;

  // Available options (as Move objects)
  options: Move[];

  // Context for UI presentation
  context?: {
    description?: string;
    relevantPositions?: Position[];
  };
}
```

### Decision Types

1. **line_order** - Multiple lines detected, player chooses which to process first
2. **region_order** - Multiple disconnected territory regions, player chooses order
3. **elimination_target** - Player earned elimination and must choose target stack
4. **chain_capture** - Chain capture in progress, player chooses next capture

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
