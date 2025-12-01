# Test Infrastructure

## Overview

This document describes the test infrastructure components used in the RingRift project. The test infrastructure provides specialized utilities for testing complex multiplayer game scenarios, including multi-client WebSocket coordination, network resilience testing, time-controlled timeout testing, and deterministic game state fixtures.

## Test Helpers

### MultiClientCoordinator

**Location:** [`tests/helpers/MultiClientCoordinator.ts`](../tests/helpers/MultiClientCoordinator.ts)

**Purpose:** Coordinates multiple WebSocket client connections for E2E testing of multiplayer game scenarios. Enables synchronized actions between multiple players with Promise-based event waiting.

**Key Features:**

- Manage multiple Socket.IO client connections
- Coordinate actions between clients (e.g., player 1 moves, player 2 responds)
- Promise-based waiting for specific game events
- Message queuing per client for inspection
- Robust cleanup of all connections after tests

**Usage:**

```typescript
import { MultiClientCoordinator } from '../helpers/MultiClientCoordinator';

const coordinator = new MultiClientCoordinator('http://localhost:3000');

// Connect two players
await coordinator.connect('player1', { playerId: 'p1', token: 'jwt-token-1' });
await coordinator.connect('player2', { playerId: 'p2', token: 'jwt-token-2' });

// Join a game
await coordinator.joinGame('player1', 'game-123');
await coordinator.joinGame('player2', 'game-123');

// Wait for both to receive game state
await coordinator.waitForAll(['player1', 'player2'], {
  type: 'event',
  eventName: 'game_state',
  predicate: (data) => data.data?.gameId === 'game-123',
});

// Send moves and wait for responses
await coordinator.sendMoveById('player1', 'game-123', 'move-id');
await coordinator.waitForGameState('player2', (state) => state.currentPlayer === 2);

// Cleanup
await coordinator.cleanup();
```

**Core Methods:**
| Method | Description |
|--------|-------------|
| [`connect(clientId, config)`](../tests/helpers/MultiClientCoordinator.ts:157) | Connect a client to the WebSocket server |
| [`disconnect(clientId)`](../tests/helpers/MultiClientCoordinator.ts:204) | Disconnect a specific client |
| [`cleanup()`](../tests/helpers/MultiClientCoordinator.ts:243) | Clean up all connections (use in `afterEach`) |
| [`send(clientId, event, payload)`](../tests/helpers/MultiClientCoordinator.ts:376) | Send a message from a client |
| [`waitFor(clientId, condition)`](../tests/helpers/MultiClientCoordinator.ts:429) | Wait for a condition on a client |
| [`waitForAll(clientIds, condition)`](../tests/helpers/MultiClientCoordinator.ts:467) | Wait for condition on multiple clients |
| [`waitForGameState(clientId, predicate)`](../tests/helpers/MultiClientCoordinator.ts:502) | Wait for specific game state |
| [`waitForPhase(clientId, phase)`](../tests/helpers/MultiClientCoordinator.ts:520) | Wait for game to reach a phase |
| [`waitForGameOver(clientId)`](../tests/helpers/MultiClientCoordinator.ts:546) | Wait for game to end |
| [`getMessages(clientId)`](../tests/helpers/MultiClientCoordinator.ts:609) | Get all captured messages |
| [`getLastGameState(clientId)`](../tests/helpers/MultiClientCoordinator.ts:650) | Get most recent game state |

---

### NetworkSimulator

**Location:** [`tests/helpers/NetworkSimulator.ts`](../tests/helpers/NetworkSimulator.ts)

**Purpose:** Simulates network conditions for resilience testing. Enables testing of reconnection scenarios, network partitions, latency, and packet loss.

**Key Features:**

- Force disconnect clients (simulate network loss)
- Delay/drop messages (simulate poor network)
- Simulate client-side disconnect followed by reconnect
- Integrates with MultiClientCoordinator
- Network condition configuration (latency, packet loss, auto-disconnect)

**Usage:**

```typescript
import { MultiClientCoordinator } from '../helpers/MultiClientCoordinator';
import { NetworkSimulator, NetworkAwareCoordinator } from '../helpers/NetworkSimulator';

// Option 1: Use with existing coordinator
const coordinator = new MultiClientCoordinator('http://localhost:3000');
const networkSimulator = new NetworkSimulator(coordinator);

await coordinator.connect('player1', config1);
await coordinator.connect('player2', config2);

// Simulate network partition for player 1
await networkSimulator.forceDisconnect('player1');

// Wait some time, then reconnect
await networkSimulator.simulateReconnect('player1', 1000);

// Option 2: Use NetworkAwareCoordinator (all-in-one)
const coordinator = new NetworkAwareCoordinator('http://localhost:3000');
await coordinator.connect('player1', config1);

// Access network simulator via .network property
await coordinator.network.forceDisconnect('player1');
await coordinator.network.simulateReconnect('player1', 1000);
```

**Network Conditions:**

```typescript
// Apply network conditions
networkSimulator.setCondition('player1', {
  latencyMs: 100, // Add 100ms latency
  packetLoss: 0.1, // 10% packet loss
  disconnectAfter: 30000, // Auto-disconnect after 30s
});

// Message interception
networkSimulator.interceptMessages('player1', (message, direction) => {
  if (shouldDrop(message)) {
    return { action: 'drop' };
  }
  return { action: 'pass' };
});

// One-shot operations
networkSimulator.dropNextMessage('player1');
networkSimulator.delayNextMessage('player1', 500);
```

**Core Methods:**
| Method | Description |
|--------|-------------|
| [`setCondition(clientId, condition)`](../tests/helpers/NetworkSimulator.ts:134) | Apply network conditions |
| [`clearCondition(clientId)`](../tests/helpers/NetworkSimulator.ts:160) | Clear network conditions |
| [`forceDisconnect(clientId)`](../tests/helpers/NetworkSimulator.ts:192) | Force disconnect a client |
| [`simulateReconnect(clientId, delay)`](../tests/helpers/NetworkSimulator.ts:232) | Reconnect after partition |
| [`interceptMessages(clientId, interceptor)`](../tests/helpers/NetworkSimulator.ts:296) | Set message interceptor |
| [`dropNextMessage(clientId)`](../tests/helpers/NetworkSimulator.ts:325) | Drop next outgoing message |
| [`delayNextMessage(clientId, delayMs)`](../tests/helpers/NetworkSimulator.ts:344) | Delay next outgoing message |
| [`cleanup()`](../tests/helpers/NetworkSimulator.ts:382) | Clean up all state |

---

### TimeController

**Location:** [`tests/helpers/TimeController.ts`](../tests/helpers/TimeController.ts)

**Purpose:** Controls time in tests for timeout testing. Uses Jest's built-in fake timers to enable rapid testing of timeout scenarios that would otherwise take 30+ seconds in real time.

**Key Features:**

- Install/uninstall time mocking
- Advance time programmatically
- Set acceleration factor for real-time tests
- Pause/resume time flow
- Works with Jest's modern fake timers

**Usage:**

```typescript
import { TimeController, createTimeController, withTimeControl } from '../helpers/TimeController';

describe('Timeout tests', () => {
  let timeController: TimeController;

  beforeEach(() => {
    timeController = createTimeController();
    timeController.install();
  });

  afterEach(() => {
    timeController.uninstall();
  });

  it('should timeout decision after 30 seconds', async () => {
    // Setup game with pending decision
    await setupGame();

    // Advance time by 30 seconds (instant in test time)
    await timeController.advanceTime(30000);

    // Verify timeout occurred
    expect(game.state.phase).toBe('timeout_processing');
  });

  // Alternative: Use withTimeControl helper
  it('should timeout using helper', async () => {
    await withTimeControl(async (tc) => {
      const game = createGame();
      await tc.advanceTime(30_000);
      expect(game.isTimedOut).toBe(true);
    });
  });
});
```

**Convenience Methods:**

```typescript
// Advance by seconds or minutes
await timeController.advanceTimeBySeconds(30);
await timeController.advanceTimeByMinutes(5);

// Advance to specific timestamp
await timeController.advanceTimeTo(Date.now() + 60000);

// Run all pending timers
await timeController.runAllTimers();

// Get pending timer count
const pending = timeController.getPendingTimerCount();

// Pause/resume time
timeController.pause();
timeController.resume();
```

**Factory Functions:**
| Function | Description |
|----------|-------------|
| [`createTimeController(options)`](../tests/helpers/TimeController.ts:468) | Create with default options |
| [`createFastTimeoutController()`](../tests/helpers/TimeController.ts:481) | Optimized for 30s timeout tests |
| [`createHybridTimeController()`](../tests/helpers/TimeController.ts:498) | For mixed async/timer testing |
| [`withTimeControl(testFn)`](../tests/helpers/TimeController.ts:535) | Auto setup/teardown wrapper |
| [`waitForConditionWithTimeAdvance()`](../tests/helpers/TimeController.ts:569) | Wait for condition while advancing time |

---

### OrchestratorTestUtils

**Location:** [`tests/helpers/orchestratorTestUtils.ts`](../tests/helpers/orchestratorTestUtils.ts)

**Purpose:** Shared helpers for orchestrator-centric backend and sandbox tests. Provides utilities for creating game engines with orchestrator-first turn processing and seeding complex game states.

**Key Features:**

- Create orchestrator-enabled GameEngine instances
- Seed overlength marker lines for line detection tests
- Seed territory regions with outside stacks
- Filter action moves from decision moves

**Usage:**

```typescript
import {
  createOrchestratorBackendEngine,
  createBackendOrchestratorHarness,
  seedOverlengthLineForPlayer,
  seedTerritoryRegionWithOutsideStack,
  toEngineMove,
  filterRealActionMoves,
} from '../helpers/orchestratorTestUtils';

// Create orchestrator-enabled engine
const engine = createOrchestratorBackendEngine('game-123', 'square8');

// Create harness for orchestrator testing
const harness = createBackendOrchestratorHarness(engine);
const validMoves = harness.adapter.getValidMoves();

// Seed an overlength line for player 1
const linePositions = seedOverlengthLineForPlayer(engine, 1, 0, 1);

// Seed territory region for elimination tests
seedTerritoryRegionWithOutsideStack(engine, {
  regionSpaces: [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
  ],
  controllingPlayer: 1,
  victimPlayer: 2,
  outsideStackPosition: { x: 5, y: 5 },
  outsideStackHeight: 3,
});

// Filter to only real action moves (excludes decisions)
const actionMoves = filterRealActionMoves(validMoves);
```

---

## Test Fixtures

### TypeScript Fixtures

#### nearVictoryTerritoryFixture

**Location:** [`tests/fixtures/nearVictoryTerritoryFixture.ts`](../tests/fixtures/nearVictoryTerritoryFixture.ts)

**Purpose:** Provides game states where Player 1 is one territory region resolution away from winning by territory control.

```typescript
import { createNearVictoryTerritoryFixture } from '../fixtures/nearVictoryTerritoryFixture';

const fixture = createNearVictoryTerritoryFixture({
  boardType: 'square8',
  spacesbelowThreshold: 1,
  pendingRegionSize: 1,
});

// Use fixture.gameState, fixture.winningMove
// Victory threshold: > 50% of board spaces (33 for square8)
```

#### chainCaptureExtendedFixture

**Location:** [`tests/fixtures/chainCaptureExtendedFixture.ts`](../tests/fixtures/chainCaptureExtendedFixture.ts)

**Purpose:** Provides game states for extended chain captures with 4+ targets, testing sequential decision-making during chain captures.

```typescript
import {
  createChainCapture3Fixture,
  createChainCapture4Fixture,
  createChainCaptureZigzagFixture,
  createChainCaptureEdgeTerminationFixture,
} from '../fixtures/chainCaptureExtendedFixture';

// 4-target chain capture (SE → SE → SE → W)
const fixture = createChainCapture4Fixture();

// Zigzag pattern with direction changes (E → S → E)
const zigzag = createChainCaptureZigzagFixture();

// Edge termination (chain ends at board edge)
const edge = createChainCaptureEdgeTerminationFixture();
```

#### multiPhaseTurnFixture

**Location:** [`tests/fixtures/multiPhaseTurnFixture.ts`](../tests/fixtures/multiPhaseTurnFixture.ts)

**Purpose:** Provides game states where a single action triggers multiple turn phases in sequence: placement → movement/capture → chain_capture → line_processing → territory_processing.

```typescript
import {
  createMultiPhaseTurnFixture,
  createFullSequenceTurnFixture,
  createPlacementToMovementFixture,
} from '../fixtures/multiPhaseTurnFixture';

// Movement → chain_capture → line_processing → territory_processing
const fixture = createMultiPhaseTurnFixture();

// Capture triggers chain → line → disconnected region
const fullSeq = createFullSequenceTurnFixture();

// Placement → movement phase transition
const placement = createPlacementToMovementFixture();
```

---

### Contract Vectors

**Location:** [`tests/fixtures/contract-vectors/`](../tests/fixtures/contract-vectors/)

**Purpose:** JSON-based test vectors for contract-based parity testing between TypeScript canonical engine and Python AI rules engine.

#### v2 Vectors (Current)

| File                                                                                                               | Category             | Description                       |
| ------------------------------------------------------------------------------------------------------------------ | -------------------- | --------------------------------- |
| [`placement.vectors.json`](../tests/fixtures/contract-vectors/v2/placement.vectors.json)                           | placement            | Ring placement and skip scenarios |
| [`movement.vectors.json`](../tests/fixtures/contract-vectors/v2/movement.vectors.json)                             | movement             | Stack movement scenarios          |
| [`capture.vectors.json`](../tests/fixtures/contract-vectors/v2/capture.vectors.json)                               | capture              | Overtaking capture scenarios      |
| [`chain_capture.vectors.json`](../tests/fixtures/contract-vectors/v2/chain_capture.vectors.json)                   | chain_capture        | Chain-capture continuations       |
| [`chain_capture_extended.vectors.json`](../tests/fixtures/contract-vectors/v2/chain_capture_extended.vectors.json) | chain_capture        | Extended chains (3+ segments)     |
| [`line_detection.vectors.json`](../tests/fixtures/contract-vectors/v2/line_detection.vectors.json)                 | line_detection       | Line-detection entry scenarios    |
| [`territory.vectors.json`](../tests/fixtures/contract-vectors/v2/territory.vectors.json)                           | territory            | Territory detection/credit        |
| [`territory_processing.vectors.json`](../tests/fixtures/contract-vectors/v2/territory_processing.vectors.json)     | territory_processing | Region processing + elimination   |
| [`forced_elimination.vectors.json`](../tests/fixtures/contract-vectors/v2/forced_elimination.vectors.json)         | edge_case            | Forced-elimination and ANM        |
| [`hex_edge_cases.vectors.json`](../tests/fixtures/contract-vectors/v2/hex_edge_cases.vectors.json)                 | edge_case            | Hex-specific edge cases           |
| [`multi_phase_turn.vectors.json`](../tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json)             | multi_phase          | Multi-phase turn sequences        |
| [`near_victory_territory.vectors.json`](../tests/fixtures/contract-vectors/v2/near_victory_territory.vectors.json) | territory            | Near-victory scenarios            |

**Vector Format:**

```json
{
  "id": "category.scenario.variant",
  "version": "v2",
  "category": "placement|movement|capture|chain_capture|...",
  "description": "Human-readable description",
  "tags": ["smoke", "regression", "edge-case"],
  "source": "manual|recorded|generated|regression",
  "input": {
    "state": {
      /* SerializedGameState */
    },
    "move": {
      /* Move */
    }
  },
  "expectedOutput": {
    "status": "complete|awaiting_decision",
    "assertions": {
      "currentPlayer": 1,
      "currentPhase": "movement",
      "gameStatus": "active",
      "stackCount": 1,
      "markerCount": 0,
      "sInvariantDelta": 0
    }
  }
}
```

**Usage (TypeScript):**

```typescript
import { importVectorBundle, validateAgainstAssertions } from '@/shared/engine/contracts';
import { readFileSync } from 'fs';

const json = readFileSync('tests/fixtures/contract-vectors/v2/placement.vectors.json', 'utf-8');
const vectors = importVectorBundle(json);

for (const vector of vectors) {
  const result = processTurn(deserializeGameState(vector.input.state), vector.input.move);
  const validation = validateAgainstAssertions(result.nextState, vector.expectedOutput.assertions);

  if (!validation.valid) {
    console.error(`Vector ${vector.id} failed:`, validation.failures);
  }
}
```

---

### Rules Parity Fixtures

**Location:** [`tests/fixtures/rules-parity/`](../tests/fixtures/rules-parity/)

**Purpose:** State snapshots and traces for TS↔Python parity testing. Contains both v1 (simpler) and v2 (complex multi-board) fixtures.

| Category         | Description                          |
| ---------------- | ------------------------------------ |
| `state_action.*` | Single state + action pairs          |
| `trace.*`        | Multi-step game traces               |
| `state_only.*`   | State-only fixtures (initial states) |

---

### Heuristic Fixtures

**Location:** [`tests/fixtures/heuristic/`](../tests/fixtures/heuristic/)

**Purpose:** Fixtures for AI heuristic evaluation testing.

---

## Best Practices

### General Guidelines

1. **Use TimeController for timeout tests** - Never use real `setTimeout` for testing timeouts. Use TimeController to advance time instantly.

2. **Use NetworkSimulator for resilience tests** - Test reconnection, partition recovery, and degraded network conditions.

3. **Use contract vectors for deterministic rules testing** - Ensures TS↔Python parity with verifiable assertions.

4. **Clean up resources** - Always call `cleanup()` in `afterEach` for coordinators and `uninstall()` for TimeController.

### Test Structure

```typescript
describe('Feature', () => {
  let coordinator: MultiClientCoordinator;
  let timeController: TimeController;

  beforeEach(() => {
    coordinator = new MultiClientCoordinator('http://localhost:3000');
    timeController = createTimeController();
    timeController.install();
  });

  afterEach(async () => {
    await coordinator.cleanup();
    timeController.uninstall();
  });

  it('should handle scenario', async () => {
    // Test implementation
  });
});
```

### Multi-Client Testing Pattern

```typescript
it('should synchronize two players', async () => {
  // 1. Connect clients
  await coordinator.connect('p1', { playerId: 'p1', token: token1 });
  await coordinator.connect('p2', { playerId: 'p2', token: token2 });

  // 2. Join game
  await coordinator.joinGame('p1', gameId);
  await coordinator.joinGame('p2', gameId);

  // 3. Wait for both to receive state
  await coordinator.waitForAll(['p1', 'p2'], {
    type: 'gameState',
    predicate: (data) => data.data?.gameId === gameId,
  });

  // 4. Execute and verify actions
  await coordinator.sendMoveById('p1', gameId, moveId);
  const result = await coordinator.waitForGameState('p2', (state) => state.currentPlayer === 2);

  expect(result.data.gameState.currentPlayer).toBe(2);
});
```

### Network Resilience Testing Pattern

```typescript
it('should recover from network partition', async () => {
  const coordinator = new NetworkAwareCoordinator('http://localhost:3000');

  await coordinator.connect('p1', config1);
  await coordinator.connect('p2', config2);
  await coordinator.joinGame('p1', gameId);
  await coordinator.joinGame('p2', gameId);

  // Simulate network partition
  await coordinator.network.forceDisconnect('p1');

  // Verify p2 sees disconnect
  await coordinator.waitForEvent('p2', 'player_disconnected');

  // Reconnect after delay
  await coordinator.network.simulateReconnect('p1', 2000);

  // Verify reconnection
  await coordinator.waitForEvent('p2', 'player_reconnected');
});
```

### Timeout Testing Pattern

```typescript
it('should timeout decision after 30 seconds', async () => {
  await withTimeControl(async (tc) => {
    const game = setupGameWithPendingDecision();

    // Advance to just before timeout
    await tc.advanceTime(29_000);
    expect(game.isTimedOut).toBe(false);

    // Advance past timeout
    await tc.advanceTime(2_000);
    expect(game.isTimedOut).toBe(true);
  });
});
```

## See Also

- [`docs/CONTRACT_VECTORS_DESIGN.md`](CONTRACT_VECTORS_DESIGN.md) - Contract vector design and implementation
- [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](INVARIANTS_AND_PARITY_FRAMEWORK.md) - Parity testing framework
- [`docs/TEST_CATEGORIES.md`](TEST_CATEGORIES.md) - Test categorization guide
