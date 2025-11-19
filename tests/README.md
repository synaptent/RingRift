# RingRift Testing Guide

## Overview

This directory contains the comprehensive testing framework for RingRift. The testing infrastructure uses **Jest** with **TypeScript** support via ts-jest.

## Directory Structure

```
tests/
├── README.md                 # This file - testing documentation
├── setup.ts                  # Jest setup - runs AFTER test framework
├── setup-env.ts              # Jest env setup (dotenv, timers, etc.)
├── test-environment.js       # Custom Jest environment (fixes localStorage)
├── utils/
│   └── fixtures.ts          # Test utilities and fixture creators
└── unit/
    ├── BoardManager.*.test.ts                 # Board geometry, lines, territory
    ├── GameEngine.*.test.ts                   # Core rules, chain capture, choices
    ├── ClientSandboxEngine.*.test.ts          # Client-local sandbox engine: movement, captures, lines, territory, victory
    ├── AIEngine.*.test.ts                     # AI service client + heuristics
    ├── WebSocket*.test.ts                     # WebSocket & PlayerInteractionManager flows
    └── ...                                    # Additional focused rule/interaction suites
```

## Running Tests

### Basic Commands

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run tests with coverage in watch mode
npm run test:coverage:watch

# Run tests for CI/CD (optimized)
npm run test:ci

# Run only unit tests
npm run test:unit

# Run only integration tests
npm run test:integration

# Run tests with verbose output
npm run test:verbose

# Run only the client-local sandbox engine suites
npm test -- ClientSandboxEngine

# Run only GameEngine territory/region tests
npm test -- GameEngine.territoryDisconnection
```

## Test Configuration

### Jest Configuration (`jest.config.js`)

- **Test Environment**: Custom Node environment with localStorage mock
- **Coverage Target**: 80% for branches, functions, lines, statements
- **Test Match Patterns**: `**/*.test.ts`, `**/*.spec.ts`
- **Coverage Directory**: `coverage/` (gitignored)
- **Timeout**: 10 seconds per test

### TypeScript Support

Tests are written in TypeScript and compiled via ts-jest. No separate compilation step needed.

### Path Aliases

The following path aliases are configured for imports:

- `@/` → `src/`
- `@shared/` → `src/shared/`
- `@server/` → `src/server/`
- `@client/` → `src/client/`

## Test Utilities (`tests/utils/fixtures.ts`)

### Board Creation

```typescript
import { createTestBoard } from '../utils/fixtures';

// Create square 8x8 board
const board = createTestBoard('square8');

// Create square 19x19 board
const board = createTestBoard('square19');

// Create hexagonal board
const board = createTestBoard('hexagonal');
```

### Player Creation

```typescript
import { createTestPlayer } from '../utils/fixtures';

// Create default player
const player = createTestPlayer(1);

// Create player with overrides
const player = createTestPlayer(2, {
  ringsInHand: 10,
  eliminatedRings: 5,
});
```

### Game State Creation

```typescript
import { createTestGameState } from '../utils/fixtures';

// Create default game state
const gameState = createTestGameState();

// Create with custom board type
const gameState = createTestGameState({ boardType: 'hexagonal' });
```

### Board Manipulation

```typescript
import { addStack, addMarker, addCollapsedSpace, pos } from '../utils/fixtures';

// Add a stack
addStack(board, pos(3, 3), playerNumber, height);

// Add a marker
addMarker(board, pos(2, 2), playerNumber);

// Add collapsed space
addCollapsedSpace(board, pos(5, 5), playerNumber);

// Create a line of markers
createMarkerLine(board, pos(0, 0), { dx: 1, dy: 0 }, length, player);
```

### Position Helpers

```typescript
import { pos, posStr } from '../utils/fixtures';

// Create square board position
const position = pos(3, 3);

// Create hexagonal position
const position = pos(0, 0, 0);

// Convert to string
const key = posStr(3, 3); // "3,3"
const hexKey = posStr(0, 0, 0); // "0,0,0"
```

### Assertions

```typescript
import {
  assertPositionHasStack,
  assertPositionHasMarker,
  assertPositionCollapsed,
} from '../utils/fixtures';

// Assert stack exists with optional player check
assertPositionHasStack(board, pos(3, 3), expectedPlayer);

// Assert marker exists
assertPositionHasMarker(board, pos(2, 2), expectedPlayer);

// Assert space is collapsed
assertPositionCollapsed(board, pos(5, 5), expectedPlayer);
```

### Constants

```typescript
import { BOARD_CONFIGS, SQUARE_POSITIONS, HEX_POSITIONS, GAME_PHASES } from '../utils/fixtures';

// Board configurations
const config = BOARD_CONFIGS.square8;
// { type: 'square8', size: 8, ringsPerPlayer: 18, minLineLength: 4, ... }

// Common positions
const center = SQUARE_POSITIONS.center8;
const hexCenter = HEX_POSITIONS.center;

// All game phases
GAME_PHASES.forEach((phase) => {
  /* ... */
});
```

## Backend route tests & Prisma stub harness

For backend HTTP route tests (auth, users, games, etc.) that exercise real Express routers
against an in-memory database, use the shared Prisma stub harness in
`tests/utils/prismaTestUtils.ts`.

This helper provides:

- `mockDb` – simple in-memory collections backing the stub:
  - `mockDb.users: any[]`
  - `mockDb.refreshTokens: any[]`
- `prismaStub` – a minimal Prisma-like client object implementing the subset of
  methods used by the current routes:
  - `user.findFirst`, `user.findUnique`, `user.create`, `user.update`
  - `refreshToken.create`, `refreshToken.findFirst`, `refreshToken.delete`, `refreshToken.deleteMany`
  - `$transaction([...])` – sequentially awaits each operation in order
- `resetPrismaMockDb()` – clears `mockDb.users` and `mockDb.refreshTokens` between tests

Typical usage in a route test (example: `tests/unit/auth.routes.test.ts`):

```ts
import express from 'express';
import request from 'supertest';
import authRoutes from '../../src/server/routes/auth';
import { errorHandler } from '../../src/server/middleware/errorHandler';
import { mockDb, prismaStub, resetPrismaMockDb } from '../utils/prismaTestUtils';

// Wire the in-memory Prisma stub into the real database connection helper.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => prismaStub,
}));

describe('Auth HTTP routes', () => {
  beforeEach(() => {
    resetPrismaMockDb();
  });

  it('registers a new user', async () => {
    const app = express();
    app.use(express.json());
    app.use('/api/auth', authRoutes);
    app.use(errorHandler);

    const res = await request(app)
      .post('/api/auth/register')
      .send({
        /* ... */
      })
      .expect(201);

    expect(prismaStub.user.create).toHaveBeenCalled();
  });

  it('returns 409 when email already exists', async () => {
    mockDb.users.push({
      id: 'user-1',
      email: 'user1@example.com',
      username: 'other',
      password: 'hashed:Secret123',
      role: 'USER',
      isActive: true,
      emailVerified: false,
      createdAt: new Date(),
    });

    // ... call route and assert on 409 + EMAIL_EXISTS
  });
});
```

When adding new route tests (for example `user`/`game` routes):

1. Import `mockDb`, `prismaStub`, and `resetPrismaMockDb` from `tests/utils/prismaTestUtils`.
2. `jest.mock('../../src/server/database/connection', () => ({ getDatabaseClient: () => prismaStub }))`.
3. Call `resetPrismaMockDb()` in your `beforeEach` to clear in-memory state between tests.
4. Seed `mockDb.*` collections directly in each test to set up scenarios.
5. Extend `prismaTestUtils.ts` with additional models/methods as new routes require them,
   keeping all Prisma stubbing logic centralized.

This pattern keeps route tests close to the real Express + middleware stack while avoiding
network and real database dependencies.

## Writing Tests

### Basic Test Structure

```typescript
import { createTestBoard, addStack, pos } from '../utils/fixtures';

describe('Feature Name', () => {
  let board: ReturnType<typeof createTestBoard>;

  beforeEach(() => {
    board = createTestBoard('square8');
  });

  describe('Specific Functionality', () => {
    it('should do something specific', () => {
      // Arrange
      addStack(board, pos(3, 3), 1);

      // Act
      const result = someFunction(board);

      // Assert
      expect(result).toBe(expectedValue);
    });
  });
});
```

### Testing All Board Types

```typescript
import { BOARD_CONFIGS } from '../utils/fixtures';

describe.each([
  ['square8', BOARD_CONFIGS.square8],
  ['square19', BOARD_CONFIGS.square19],
  ['hexagonal', BOARD_CONFIGS.hexagonal],
])('Feature on %s board', (boardType, config) => {
  it('should work correctly', () => {
    const board = createTestBoard(config.type);
    // Test logic...
  });
});
```

## Coverage Reports

After running `npm run test:coverage`, coverage reports are generated in:

- **Terminal**: Text summary
- **HTML**: `coverage/lcov-report/index.html` (open in browser)
- **LCOV**: `coverage/lcov.info` (for CI tools)
- **JSON**: `coverage/coverage-final.json`

## CI/CD Integration

The `npm run test:ci` command is optimized for CI/CD pipelines:

- Runs in CI mode (no watch)
- Generates coverage reports
- Limits workers for resource efficiency
- Fails if coverage thresholds not met

## Best Practices

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Follow the AAA pattern for test structure
3. **One Assertion**: Focus each test on one specific behavior
4. **Use Fixtures**: Leverage test utilities for consistent test data
5. **Mock Carefully**: Only mock what's necessary for isolation
6. **Clean Up**: Tests clean up automatically via `afterEach` hooks
7. **Coverage**: Aim for 80%+ coverage on all metrics

## Debugging Tests

```bash
# Run single test file
npm test -- tests/unit/board.test.ts

# Run tests matching pattern
npm test -- --testNamePattern="createTestBoard"

# Run in debug mode
node --inspect-brk node_modules/.bin/jest --runInBand
```

## Common Issues

### localStorage SecurityError

Fixed by using custom test environment (`tests/test-environment.js`). If you encounter issues, ensure `testEnvironment` in `jest.config.js` points to the custom environment.

### TypeScript Errors

Ensure your test files are included in `tsconfig.json` or create a separate `tsconfig.test.json` if needed.

### Coverage Not Collecting

Check `collectCoverageFrom` patterns in `jest.config.js` to ensure your source files are included.

## Next Steps

See `TODO.md` Phase 2 for comprehensive test coverage tasks:

- Unit tests for all BoardManager, GameEngine, RuleEngine methods
- Integration tests for complete game flows
- Scenario tests from rules document
- Edge case coverage

## Trace & parity utilities (GameTrace)

Several AI-heavy suites use a shared **GameTrace** abstraction and trace replay helpers to compare backend and sandbox behaviour step-by-step.

**Key types (in `src/shared/types/game.ts`):**

- `GameHistoryEntry` – a single canonical action with before/after:
  - `moveNumber`, `action: Move`, `actor`
  - `phaseBefore/After`, `statusBefore/After`
  - `progressBefore/After: ProgressSnapshot` (S = markers + collapsed + eliminated)
  - Optional `stateHashBefore/After` and `boardBefore/AfterSummary` for diagnostics
- `GameTrace` – `{ initialState: GameState; entries: GameHistoryEntry[] }`

**Trace helpers (in `tests/utils/traces.ts`):**

- `runSandboxAITrace(boardType, numPlayers, seed, maxSteps): Promise<GameTrace>`
  - Runs a seeded AI-vs-AI game in the **client-local sandbox engine** (`ClientSandboxEngine`).
  - Returns the initial sandbox `GameState` plus the sandbox-emitted `GameHistoryEntry[]`.
  - Uses a deterministic `SandboxInteractionHandler` so `capture_direction` choices are stable.
- `replayTraceOnBackend(trace: GameTrace): Promise<GameTrace>`
  - Builds a fresh backend `GameEngine` from `trace.initialState`.
  - For each sandbox `entry.action`, calls `getValidMoves`, finds a semantically matching backend move via `findMatchingBackendMove`, and feeds it to `GameEngine.makeMove`.
  - Returns the backend engine’s own `GameTrace` (initial state + backend history) for parity comparison.
- `replayTraceOnSandbox(trace: GameTrace): Promise<GameTrace>`
  - Builds a fresh `ClientSandboxEngine` from `trace.initialState`.
  - Replays each `entry.action` via `applyCanonicalMove`, returning a second sandbox `GameTrace`.

These helpers are used by suites like:

- `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`
- `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`
- `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`

**Debug/diagnostic environment variables:**

- `RINGRIFT_TRACE_DEBUG`
  - When set to `1`/`true`, `runSandboxAITrace` and `replayTraceOnBackend` emit structured JSON diagnostics via `logAiDiagnostic` to `logs/ai/trace-parity.log`.
  - Currently logs the sandbox trace opening sequence (initial S/hash + first few history entries) and backend move-mismatch snapshots (sandbox move, backend valid moves, S/hash, per-player counters).
- `RINGRIFT_AI_DEBUG`
  - When set to `1`/`true`, AI-heavy suites mirror detailed diagnostics to the console in addition to writing `logs/ai/*.log`.
  - Also enables extra sandbox AI debug logs inside [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts) for “no landingCandidates” and “no-op movement” situations.

**Sandbox AI determinism (capture selection):**

- In the sandbox movement phase, when multiple overtaking capture segments are available from the same stack, [`ClientSandboxEngine.maybeRunAITurn()`](src/client/sandbox/ClientSandboxEngine.ts:277) chooses the segment whose `landing` position is **lexicographically smallest** by `(x, y, z)`.
- This matches the deterministic `capture_direction` test handler in [`ClientSandboxEngine.aiMovementCaptures.test.ts`](tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts), which also selects the lexicographically smallest `landingPosition`. This keeps sandbox AI traces reproducible across runs and aligned with backend/sandbox parity tooling.

To debug a tricky AI/parity failure locally:

1. Set `RINGRIFT_TRACE_DEBUG=1 RINGRIFT_AI_DEBUG=1` in your test environment.
2. Re-run the relevant trace/parity test (for example `Backend_vs_Sandbox.traceParity.test.ts`).
3. Inspect `logs/ai/trace-parity.log` for the structured JSON entries referenced in the failing test output.

## Sandbox AI simulation diagnostics

A separate set of AI-vs-AI sandbox diagnostics lives in `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`. These tests run seeded games entirely in `ClientSandboxEngine` and are intended as **diagnostic tools**, not as part of the default CI signal.

```bash
RINGRIFT_ENABLE_SANDBOX_AI_SIM=1 npm test -- ClientSandboxEngine.aiSimulation
```

The harness:

- Uses a deterministic PRNG seed to make runs reproducible.
- Monitors the shared progress snapshot `S = markers + collapsed + eliminated` and asserts that S is **non-decreasing** over canonical AI actions.
- Enforces a cap on the number of AI actions per run (`MAX_AI_ACTIONS`) and flags seeds that fail to reach a terminal state within that budget.

Some seeded configurations (including `square8` with 2 AI players and seed `1`) are currently expected to exceed `MAX_AI_ACTIONS`; they are tracked as **diagnostic failures** under P1.4 in `KNOWN_ISSUES.md` rather than as hard CI blockers.

For a targeted regression test of a previously observed sandbox stall on `square8` with 2 AI players and seed `1`, use:

```bash
RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1 \
RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1 \
npm test -- ClientSandboxEngine.aiStall.seed1
```

This test asserts that the engine **does not** get stuck in a long run of consecutive AI turns with no state change for that seed and emits `[Sandbox AI Stall Diagnostic]` warnings when it encounters “no captures/moves and no forced elimination” situations while debugging.

## Scenario Matrix (Rules/FAQ → Jest suites)

This matrix links key sections of `ringrift_complete_rules.md` and FAQ entries to concrete Jest suites. Existing suites are marked **(existing)**; scenario-focused suites under `tests/scenarios/` are marked **(scenario)**; proposed suites are marked **(planned)**.

> Naming convention for scenario-style tests:
>
> > Use rule/FAQ IDs in the `describe`/`it` names, e.g. `Q15_3_1_180_degree_reversal`.
> > Prefer square8 examples first, then mirror high‑value cases on square19/hex where relevant.

### Turn sequence & forced elimination

- **Section 4 (Turn Sequence)**, **FAQ 15.2 (Flowchart of a Turn)**, **FAQ 24 (Forced elimination when blocked)**
  - (scenario) `tests/unit/GameEngine.turnSequence.scenarios.test.ts` — backend turn-sequence and forced-elimination orchestration tests covering blocked-with-stacks and skip-over-no-material players.

#### Mini rules→tests matrix: Turn sequence, movement & progress (cluster example)

| Rule / FAQ cluster                             | Description                                                                  | Primary tests                                                                                                                                                                                                                         |
| ---------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Section 4.x Turn Sequence & Forced Elimination | Turn start/end, blocked-with-stacks behaviour, skipping dead players         | [`tests/unit/GameEngine.turnSequence.scenarios.test.ts`](tests/unit/GameEngine.turnSequence.scenarios.test.ts:1), [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:12) |
| Sections 8.2–8.3 Non‑capture Movement          | Minimum distance ≥ stack height, marker landing, blocked paths               | [`tests/unit/RuleEngine.movementCapture.test.ts`](tests/unit/RuleEngine.movementCapture.test.ts:1)                                                                                                                                    |
| Section 13.5 Progress & Termination Invariant  | S-invariant (markers + collapsed + eliminated) under forced elim & stalemate | [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:12)                                                                                                                   |
| FAQ 15.2 / 24 (Turn flow & forced elimination) | Flowchart of a turn and forced elimination when blocked                      | [`tests/unit/GameEngine.turnSequence.scenarios.test.ts`](tests/unit/GameEngine.turnSequence.scenarios.test.ts:1), [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:12) |

### Movement, minimum distance, and markers

- **Sections 8.2–8.3 (Minimum Distance, Marker Interaction)**, **FAQ 2–3**
  - (existing) `tests/unit/RuleEngine.movementCapture.test.ts`
  - (planned) `tests/unit/RuleEngine.movement.scenarios.test.ts` — explicit distance + landing cases for square8, square19, hex

### Chain captures & capture patterns

- **Sections 9–10 (Overtaking & Chain Overtaking)**, **FAQ 5–6, 9, 12, 14**, **FAQ 15.3.1–15.3.2 (180° reversal, cyclic)**
  - (existing) `tests/unit/GameEngine.chainCapture.test.ts` — core chain engine behaviour, 180° reversal, marker interactions, termination rules
  - (existing) `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts` — backend chain-capture geometry enumeration + CaptureDirectionChoice integration for the orthogonal multi-branch scenario (Rust player-choice parity)
  - (existing) `tests/unit/ClientSandboxEngine.chainCapture.test.ts` — sandbox parity for the core two-step chain and the orthogonal multi-branch capture_direction PlayerChoice scenario
  - (scenario) `tests/scenarios/ComplexChainCaptures.test.ts` — end-to-end backend chain-capture examples for FAQ 15.3.1 (180° reversal) and 15.3.2 (cyclic pattern), plus multi-step chains with direction changes on `square8`.

### Line formation & graduated rewards

- **Section 11 (Line Formation & Collapse)**, **FAQ 7, 22**, **Sections 16.5/16.9.4.3**
  - (existing) `tests/unit/ClientSandboxEngine.lines.test.ts` — sandbox line processing
  - (existing) `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts` — backend + AI choice for line rewards
  - (existing) `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts` — backend + WebSocket choice flow
  - (existing) `tests/unit/GameEngine.lines.scenarios.test.ts` — backend line semantics aligned with Section 11 and FAQ Q7/Q22 on `square8`.

### Territory disconnection & chain reactions

- **Section 12 (Area Disconnection & Collapse)**, **FAQ 10, 15, 20, 23**, **Sections 16.9.4.4, 16.9.6–16.9.8**
  - (existing) `tests/unit/BoardManager.territoryDisconnection.test.ts` / `.hex.test.ts` — region detection & adjacency
  - (existing) `tests/unit/GameEngine.territoryDisconnection.test.ts` / `.hex.test.ts` — engine‑level region processing
  - (existing) `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` / `.hex.test.ts` — sandbox parity
  - (existing) `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts` — region-order PlayerChoice in sandbox
  - (existing) `tests/unit/GameEngine.territory.scenarios.test.ts` — explicit self‑elimination prerequisite and multi‑region chain reactions mapped to Q15, Q20, Q23

### Victory conditions & stalemate

- **Section 13 (Victory Conditions)**, **FAQ 11, 18, 21, 24**, **Sections 16.6, 16.9.4.5**
  - (existing) `tests/unit/ClientSandboxEngine.victory.test.ts` — sandbox ring‑elimination & territory victories
  - (planned) `tests/unit/GameEngine.victory.scenarios.test.ts` — ring‑elimination, territory‑majority, last‑player‑standing, stalemate tiebreaker examples

### PlayerChoice flows (engine + transport + sandbox)

- **Sections 4.5, 10.3, 11–12 (Lines, Territory, Chain choices)**, **FAQ 7, 15, 22–23**
  - (existing) `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`
  - (existing) `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`
  - (existing) `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`
  - (existing) `tests/unit/GameEngine.captureDirectionChoice.test.ts` / `.captureDirectionChoiceWebSocketIntegration.test.ts` — helper-level capture_direction logic plus a full orthogonal chain capture scenario driven end-to-end over WebSockets
  - (existing) `tests/unit/PlayerInteractionManager.test.ts`
  - (existing) `tests/unit/WebSocketInteractionHandler.test.ts`
  - (existing) `tests/unit/AIInteractionHandler.test.ts`
  - (planned) Additional focused scenarios in the above suites to ensure each choice type has at least one rule/FAQ‑tagged test name.

---

**Last Updated**: November 18, 2025  
**Framework**: Jest 29.7.0 + ts-jest 29.1.1
