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

### High-Level Testing Overview (by purpose)

This section groups the Jest suites by **what** they validate. For detailed
classification (rules-level vs trace-level vs integration-level) see
[`tests/TEST_SUITE_PARITY_PLAN.md`](tests/TEST_SUITE_PARITY_PLAN.md:28).

#### 1. Shared-helper rules tests (canonical semantics)

These suites exercise the shared rules helpers under
[`src/shared/engine`](../src/shared/engine/types.ts:1) and should be treated as
the primary specification for game semantics:

- **Movement & captures**
  - [`movement.shared.test.ts`](tests/unit/movement.shared.test.ts:1) – non‑capturing movement reachability over `movementLogic.ts`.
  - [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts:1) – overtaking segment enumeration over `captureLogic.ts`.
  - [`RuleEngine.movementCapture.test.ts`](tests/unit/RuleEngine.movementCapture.test.ts:1) – backend adapter alignment with shared movement/capture helpers.
- **Lines**
  - [`lineDetection.shared.test.ts`](tests/unit/lineDetection.shared.test.ts:1) – shared marker-line geometry.
  - [`LineDetectionParity.rules.test.ts`](tests/unit/LineDetectionParity.rules.test.ts:1) – line semantics across shared engine, backend, and sandbox.
  - [`Seed14Move35LineParity.test.ts`](tests/unit/Seed14Move35LineParity.test.ts:1) – seed‑14 guardrail for “no valid line” at a historically ambiguous state.
- **Territory (detection, borders, processing)**
  - [`territoryBorders.shared.test.ts`](tests/unit/territoryBorders.shared.test.ts:1) – shared border-marker expansion.
  - [`territoryProcessing.shared.test.ts`](tests/unit/territoryProcessing.shared.test.ts:1) – shared region-processing pipeline.
  - [`territoryProcessing.rules.test.ts`](tests/unit/territoryProcessing.rules.test.ts:1),
    [`sandboxTerritory.rules.test.ts`](tests/unit/sandboxTerritory.rules.test.ts:1),
    [`sandboxTerritoryEngine.rules.test.ts`](tests/unit/sandboxTerritoryEngine.rules.test.ts:1) – rules-level suites for Q23, region collapse, and internal eliminations.
- **Placement / no-dead-placement**
  - [`placement.shared.test.ts`](tests/unit/placement.shared.test.ts:1) – shared placement validation rules.
  - [`RuleEngine.placementMultiRing.test.ts`](tests/unit/RuleEngine.placementMultiRing.test.ts:1) – backend multi-ring placement behaviour over the shared helpers.
- **Victory & invariants**
  - [`victory.shared.test.ts`](tests/unit/victory.shared.test.ts:1) – shared victory/stalemate ladder over `victoryLogic.ts`.
  - [`SInvariant.seed17FinalBoard.test.ts`](tests/unit/SInvariant.seed17FinalBoard.test.ts:1),
    [`SharedMutators.invariants.test.ts`](tests/unit/SharedMutators.invariants.test.ts:1),
    [`ProgressSnapshot.core.test.ts`](tests/unit/ProgressSnapshot.core.test.ts:1) – S-invariant and progress-snapshot guarantees.
- **Turn sequencing & termination**
  - [`GameEngine.turnSequence.scenarios.test.ts`](tests/unit/GameEngine.turnSequence.scenarios.test.ts:1) – backend turn ladder over the shared `turnLogic.ts`.
  - [`SandboxAI.ringPlacementNoopRegression.test.ts`](tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts:1),
    [`ClientSandboxEngine.aiSimulation.test.ts`](tests/unit/ClientSandboxEngine.aiSimulation.test.ts:1) – shared termination ladder as exercised by sandbox AI.

When changing rules, update or extend these suites first wherever possible.

#### 2. Host parity tests (backend ↔ sandbox ↔ shared/Python)

These suites ensure that backend `GameEngine`/`RuleEngine`, `ClientSandboxEngine`,
and Python rules behave identically for a given ruleset:

- **Backend vs sandbox (movement, captures, placement, territory, victory)**
  - [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1)
  - [`PlacementParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts:1)
  - [`movementReachabilityParity.test.ts`](tests/unit/movementReachabilityParity.test.ts:1)
  - [`reachabilityParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/reachabilityParity.RuleEngine_vs_Sandbox.test.ts:1)
  - [`TerritoryParity.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1)
  - [`TerritoryBorders.Backend_vs_Sandbox.test.ts`](tests/unit/TerritoryBorders.Backend_vs_Sandbox.test.ts:1)
  - [`TerritoryCore.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts:1)
  - [`VictoryParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts:1)
- **Trace/fixture parity (shared engine vs hosts, TS vs Python)**
  - [`TraceFixtures.sharedEngineParity.test.ts`](tests/unit/TraceFixtures.sharedEngineParity.test.ts:1) – shared `GameEngine` vs backend.
  - [`Backend_vs_Sandbox.traceParity.test.ts`](tests/unit/Backend_vs_Sandbox.traceParity.test.ts:1) – curated seeded trace parity between backend and sandbox.
  - [`Sandbox_vs_Backend.aiRngParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1),
    [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1) – RNG-driven AI parity harnesses.
  - [`Python_vs_TS.traceParity.test.ts`](tests/unit/Python_vs_TS.traceParity.test.ts:1) plus the Python-side parity suites under
    `ai-service/tests/parity/*`, driven by shared-engine fixtures.

These are **smoke tests and regression nets**; when they disagree with rules-level
suites, treat the traces/fixtures as derived artifacts (see below and
[`tests/TEST_SUITE_PARITY_PLAN.md`](tests/TEST_SUITE_PARITY_PLAN.md:56)).

#### 3. Scenario / RulesMatrix / FAQ tests

Scenario-style suites encode concrete board positions and expected behaviour:

- **RulesMatrix scenarios**
  - Files under `tests/scenarios/RulesMatrix.*.test.ts`, for example
    [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1) and
    the comprehensive movement/territory/victory suites referenced from
    [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md:1).
- **FAQ scenarios (Q1–Q24)**
  - [`FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1),
    [`FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1),
    [`FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1),
    [`FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1),
    [`FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1),
    [`FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1),
    [`FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1).
- **Compound/termination scenarios**
  - [`ForcedEliminationAndStalemate.test.ts`](tests/scenarios/ForcedEliminationAndStalemate.test.ts:12),
    `LineAndTerritory.test.ts`, and other compound examples linked from
    [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md:1).

Scenario suites are the best starting point when you want to **see** how a rule
behaves in a complete position or full turn sequence.

### Quiet / logged runs (recommended in Cline/VSCode)

Some suites (especially AI simulations and parity tests) can emit a lot of
output. To avoid overwhelming the terminal or tooling, prefer running them in a
"logged" mode and then viewing the result through the size‑limited
`safe-view` helper.

From the project root:

```bash
# 1. Run the full Jest suite quietly, logging to logs/jest/latest.log
npm run test:all:quiet:log

# 2. Create a safe, truncated view of the latest Jest log
npm run logs:view:jest

# 3. Run the Python AI-service pytest suite quietly, logging to logs/pytest
npm run test:python:quiet:log

# 4. Create a safe, truncated view of the latest pytest log
npm run logs:view:pytest
```

Notes:

- `scripts/safe-view.js` wraps long lines and caps the total number of lines
  written to the `*.view.txt` files, so opening them in VSCode or Cline will
  not explode the context.
- Leave diagnostic flags like `RINGRIFT_AI_DEBUG` and `RINGRIFT_TRACE_DEBUG`
  **unset** during normal runs; only enable them when you actively need
  low-level AI/trace diagnostics.

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
8. **Determinism**: Use explicit seeds for reproducible tests with AI or random behavior

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

Tests are compiled using `tsconfig.jest.json` via **ts-jest**. If you add new
TypeScript/TSX test files and see unexpected TypeScript errors:

- Confirm they live under `tests/**/*` so they are included by `tsconfig.jest.json`.
- If you introduce new path aliases, keep `tsconfig.jest.json` and the main
  `tsconfig*.json` files in sync so Jest and your build agree on module resolution.

### Coverage Not Collecting

Check `collectCoverageFrom` patterns in `jest.config.js` to ensure your source files are included.

## Next Steps

See `TODO.md` Phase 2 for comprehensive test coverage tasks:

- Unit tests for all BoardManager, GameEngine, RuleEngine methods
- Integration tests for complete game flows
- Scenario tests from rules document
- Edge case coverage

## Test taxonomy: rules-level, trace-level, integration-level

To keep a rapidly growing suite coherent (TS backend, client sandbox, and Python AI‑service), all tests should be classified along two main axes:

### 1. Semantic level

- **Rules-level tests**  
  Directly exercise **canonical rules semantics** as implemented in `src/shared/engine/` and documented in `ringrift_complete_rules.md`.
  - Prefer to go through `src/shared/engine/GameEngine` and its validators/mutators where possible.
  - Examples:
    - `tests/unit/RefactoredEngine.test.ts`
    - `tests/unit/LineDetectionParity.rules.test.ts`
    - `tests/unit/sandboxTerritory.rules.test.ts`
    - `tests/unit/territoryProcessing.rules.test.ts`
    - `tests/unit/sandboxTerritoryEngine.rules.test.ts`
    - `tests/unit/Seed14Move35LineParity.test.ts`
  - **Naming convention**: include `.rules.` (e.g. `*.rules.test.ts`) or otherwise mention "rules" explicitly in the filename when the suite is intended to be authoritative for rules semantics.

- **Trace-level tests**  
  Exercise **particular move sequences** (traces) through one or more engines. These are **smoke tests and regression nets**, not the primary source of truth for rules.
  - Typically use `GameTrace` helpers from `tests/utils/traces.ts` to:
    - Generate sandbox AI traces (`runSandboxAITrace`).
    - Replay them on backend or sandbox (`replayTraceOnBackend`, `replayTraceOnSandbox`).
  - Examples:
    - `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`
    - `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`
    - `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`
    - `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`
    - `tests/unit/TraceParity.seed5.firstDivergence.test.ts`
  - **Naming convention**: include `traceParity` / `*Parity.*` / `seed*.trace` in the filename when the suite is fundamentally trace‑driven.

- **Integration-level tests**  
  Validate cross‑component flows (HTTP routes, WebSockets, AI service, UI wiring). Rules semantics show up only indirectly.
  - Examples:
    - `tests/integration/FullGameFlow.test.ts`
    - `tests/unit/WebSocketServer.*.integration.test.ts`
    - `tests/unit/GameEngine.*WebSocketIntegration.test.ts`
    - `tests/unit/AIEngine.serviceClient.test.ts`
    - `tests/unit/auth.routes.test.ts`, `tests/unit/server.health-and-routes.test.ts`
  - **Naming convention**: include `.integration.` in the filename when the suite covers multi‑component flows.

When adding new tests, pick the semantic level first:

- If you are fixing/clarifying rules → **rules-level**.
- If you want parity smoke coverage across engines/hosts → **trace-level**.
- If you are testing transport, orchestration, or end‑to‑end flows → **integration-level**.

### 2. Host / language domain

Each test also lives in one or more domains:

- **TS shared engine** – `src/shared/engine/*` (canonical semantics)
- **TS backend adapter** – `src/server/game/*` (BoardManager, RuleEngine, GameEngine wrappers)
- **TS client sandbox adapter** – `src/client/sandbox/*` (ClientSandboxEngine and helpers)
- **Python AI‑service** – `ai-service/app/*`
- **Cross-host / cross-language parity** – suites that explicitly compare behaviour across multiple engines (e.g. sandbox vs backend, TS vs Python).

For cross‑host parity suites:

- Prefer **rules-level fixtures** generated by the shared engine when asserting deep semantic equivalence (see `ai-service/tests/parity/*`).
- Use **trace-level parity harnesses** as smoke tests only; they must yield to rules‑level tests when semantics change.

### Traces are derived artifacts (seed‑14 precedent)

Recorded traces are **derived artifacts**, not ground truth. The canonical rules are:

- The implementation in `src/shared/engine/` (types, validators, mutators, `GameEngine`).
- The written rules in `ringrift_complete_rules.md`.

When a trace-based parity test fails:

1. **Check rules-level tests first.**
   If suites like `RefactoredEngine.test.ts`, `LineDetectionParity.rules.test.ts`, `Seed14Move35LineParity.test.ts`, and related rules‑level tests are green, treat the shared engine semantics as authoritative.

2. **Determine whether the trace is stale.**
   Many historic traces were recorded before rules fixes (e.g. line detection, territory, elimination). If the trace expects behaviour that now violates the canonical rules, the trace itself is outdated.

3. **Apply the seed‑14 pattern:**
   - Example: historical **seed 14** sandbox trace in `Backend_vs_Sandbox.traceParity.test.ts` contained a `process_line` move at step 35.
   - After fixing line detection and reconciling backend/sandbox detectors with `ringrift_complete_rules.md` Section 11.1, the shared engine and detectors both agree that **no valid lines exist** in that state.
   - `tests/unit/Seed14Move35LineParity.test.ts` codifies this as a rules‑level assertion, and seed 14 was removed from the generic trace parity harness (now only exercising seed 5).

4. **Update tests accordingly:**
   - Do **not** bend canonical rules to preserve historic traces.
   - Instead:
     - Regenerate traces under current semantics **or**
     - Remove/replace the offending seed from generic trace harnesses and add a **focused rules-level test** that expresses the intended behaviour (as done for seed 14).

This policy applies equally to TS↔Python parity:

- Python rules and `ai-service/tests/parity/*` must align with the shared TS engine and rules‑level fixtures.
- If a TS↔Python trace parity test diverges but rules-level suites and fixtures agree, treat the trace as stale and update/replace it.

### Workflow for rule changes (shared engine first)

When you change or extend **game rules**, use the following workflow (see also
[`tests/TEST_SUITE_PARITY_PLAN.md`](tests/TEST_SUITE_PARITY_PLAN.md:257)):

1. **Update shared helpers under `src/shared/engine/*`.**
   - Identify the relevant module(s) for the rule you are changing (movement/capture, lines, territory, placement, victory, or turn sequencing).
   - Make the change in the shared helpers so backend and sandbox can both reuse it.

2. **Extend shared-helper rules tests.**
   - Add or update tests in the `*.shared.test.ts` and `*.rules.test.ts` suites listed in the
     “Shared-helper rules tests” section above (for example
     [`movement.shared.test.ts`](tests/unit/movement.shared.test.ts:1),
     [`territoryProcessing.shared.test.ts`](tests/unit/territoryProcessing.shared.test.ts:1),
     [`victory.shared.test.ts`](tests/unit/victory.shared.test.ts:1)).
   - Keep these suites green; they are the semantic authority.

3. **Run and, if needed, extend parity suites.**
   - Ensure backend vs sandbox vs shared parity suites still pass (for example
     [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1),
     [`PlacementParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts:1),
     [`TerritoryParity.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1),
     [`VictoryParity.RuleEngine_vs_Sandbox.test.ts`](tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts:1),
     and [`TraceFixtures.sharedEngineParity.test.ts`](tests/unit/TraceFixtures.sharedEngineParity.test.ts:1)).
   - If a parity suite fails but rules-level tests are green, treat the failing trace/fixture as stale and update or regenerate it.

4. **Add or extend scenario tests.**
   - Encode representative positions and turns in RulesMatrix or FAQ suites to make the new/changed behaviour concrete (see
     [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md:1) and the FAQ suites under
     [`tests/scenarios/FAQ_*.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)).

5. **Run the parity plan.**
   - For larger changes, follow the plan in
     [`tests/TEST_SUITE_PARITY_PLAN.md`](tests/TEST_SUITE_PARITY_PLAN.md:56):
     - Run shared-helper rules tests.
     - Run targeted parity suites.
     - Optionally run heavier AI/trace harnesses (e.g. `Sandbox_vs_Backend.aiRngFullParity`, sandbox AI simulations) in diagnostic mode.

---

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
- `RINGRIFT_SANDBOX_AI_TRACE_MODE`
  - When set to `1`/`true`, parity-focused tests construct `ClientSandboxEngine` instances with a `traceMode` option enabled.
  - In trace mode, sandbox AI still uses the full proportional policy (`chooseLocalMoveFromCandidates`), but the sandbox engine:
    - Applies moves exclusively via canonical `Move` shapes (`place_ring`, `skip_placement`, `move_stack`/`move_ring`, `overtaking_capture`, `continue_capture_segment`).
    - Records history entries so that backend `GameEngine` can replay the same canonical move list in lockstep for trace parity.
    - Aligns chain-capture phase transitions and continuation semantics with the backend (`chain_capture` phase, explicit `continue_capture_segment` moves).

Trace mode is wired through the trace utilities in `tests/utils/traces.ts` (see `runSandboxAITrace`, `replayTraceOnBackend`, and `replayTraceOnSandbox`), and is only used by tests – normal `/sandbox` gameplay does **not** enable it by default.

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

### AI fuzz harness mid-game plateau regressions

To lock in behaviour around a historically problematic square8/2p plateau (seed `1` around action ~58), there are two additional tests and a small helper harness:

- `tests/utils/aiSeedSnapshots.ts` – seed reproduction utilities, notably `reproduceSquare8TwoAiSeed1AtAction(targetActionIndex)` which:
  - Recreates the sandbox AI configuration used by the heavy fuzz harness (square8 / 2 AI players, deterministic seed).
  - Advances the game via `ClientSandboxEngine.maybeRunAITurn` until a requested action index or until an early stall/termination, enforcing the S-invariant along the way.
  - Returns a full `GameState`, an order-stable `ComparableSnapshot`, the number of actions taken, and a live `ClientSandboxEngine` bound to that checkpoint.
- `tests/unit/ClientSandboxEngine.aiStallRegression.test.ts` – unit-level regression that:
  - Uses the seed harness to checkpoint a mid-game plateau near action ≈58 for `square8` / 2 AI / seed 1.
  - Asserts that from this checkpoint the sandbox AI does **not** enter a long active stall (no 8+ consecutive no-op AI turns while `gameStatus === 'active'`).
- `tests/scenarios/AI_TerminationFromSeed1Plateau.test.ts` – scenario-level termination test that:
  - Reuses the same plateau and focuses on global S-invariant + eventual termination behaviour.
  - Asserts that S remains non-decreasing from the plateau and that the game either completes or continues to evolve under additional AI play within a generous bound.

These tests are safe to run in normal CI and serve as targeted, reproducible guards around the fuzz harness findings, without re-running the full heavy aiSimulation suite.

## Sandbox AI stall diagnostics (engine parity and repro)

In addition to the general aiSimulation harness above, there is a focused regression test for a previously observed sandbox AI stall:

- File: `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts`
- Scenario: `square8`, 2 AI players, deterministic seed `1`
- Behaviour: Asserts that the sandbox engine does **not** get stuck in a long run of consecutive AI turns with no state change for this seed.

This suite is intentionally **opt-in** and gated by environment flags so that it does not run in normal CI:

```bash
RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1 \
RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1 \
npm test -- ClientSandboxEngine.aiStall.seed1
```

- `RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1`
  Enables the stall-repro suite itself (the test will be skipped when this flag is unset).
- `RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1`
  Turns on additional sandbox AI stall diagnostics inside `sandboxAI.maybeRunAITurnSandbox`, including:
  - Hash-based detection of repeated no-op AI turns (unchanged `GameState` hash with the same AI player to move).
  - Emission of `[Sandbox AI Stall Diagnostic]` and `[Sandbox AI Stall Detector]` warnings to the console.
  - Structured per-turn/stall entries appended to `window.__RINGRIFT_SANDBOX_TRACE__` for later analysis or replay.

These diagnostics are especially useful when combined with the browser-based `/sandbox` UI, which can surface potential stalls and export `__RINGRIFT_SANDBOX_TRACE__` snapshots via the local stall watchdog.

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
  - (existing) `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` — backend↔sandbox Q23 parity on `square19` (shared initial state covering positive, negative, and two-region multi-region territory-disconnection cases)
  - (existing) `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` — direct backend↔sandbox parity for a canonical Q23‑positive disconnected region scenario (3×3 block + border, shared initial state)

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

## Rules/FAQ → Scenario Matrix

For a rule-centric view of test coverage, see:

- `RULES_SCENARIO_MATRIX.md` – a living matrix mapping sections of `ringrift_complete_rules.md` and the FAQ to concrete Jest suites (backend engine, sandbox engine, WebSocket/choice flows, and AI boundary tests).

When you add or modify scenario-style tests, update that matrix so it remains the single source of truth for how rules map to executable tests.

## FAQ Scenario Tests

Each FAQ question from [`ringrift_complete_rules.md`](../ringrift_complete_rules.md:1) has dedicated test coverage in scenario-style test files under `tests/scenarios/FAQ_*.test.ts`.

### Running FAQ Tests

```bash
# Run all FAQ scenario tests
npm test -- FAQ_

# Run specific FAQ question groups
npm test -- FAQ_Q01_Q06     # Basic mechanics (Q1-Q6)
npm test -- FAQ_Q07_Q08     # Line formation (Q7-Q8)
npm test -- FAQ_Q09_Q14     # Edge cases & special mechanics (Q9-Q14)
npm test -- FAQ_Q15         # Chain capture patterns (Q15)
npm test -- FAQ_Q16_Q18     # Victory conditions & control (Q16-Q18)
npm test -- FAQ_Q19_Q21_Q24 # Player counts & thresholds (Q19-Q21, Q24)
npm test -- FAQ_Q22_Q23     # Graduated rewards & territory (Q22-Q23)

# Run with verbose output
npm test -- FAQ_Q15 --verbose
```

### FAQ Test Coverage Map

| FAQ Questions | Test File                                                                              | Topics Covered                                                            |
| ------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Q1-Q6         | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | Stack order, minimum distance, capture landing, overtaking vs elimination |
| Q7-Q8         | [`tests/scenarios/FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1)         | Line formation, exact vs overlength lines, no rings to eliminate          |
| Q9-Q14        | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | Chain blocking, multicolored stacks, Moore vs Von Neumann adjacency       |
| Q15           | [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1)                 | 180° reversal, cyclic patterns, mandatory chain continuation              |
| Q16-Q18       | [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1)         | Control transfer, first placement, multiple victory conditions            |
| Q19-Q21, Q24  | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | Player count variations, thresholds, forced elimination, stalemate        |
| Q22-Q23       | [`tests/scenarios/FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1)         | Graduated line rewards, territory self-elimination prerequisite           |

### FAQ Test Design Principles

1. **Direct FAQ Mapping**: Each test explicitly references its FAQ question number in the describe/it names
2. **Multiple Board Types**: Tests cover square8, square19, and hexagonal where applicable
3. **Both Engines**: Critical FAQs tested on both backend GameEngine and sandbox ClientSandboxEngine
4. **Complete Examples**: Each FAQ example from the rulebook is encoded as a test case
5. **Edge Cases**: FAQ scenarios include both positive and negative test cases

### Coverage Statistics

- **Total FAQ Questions**: 24
- **FAQ Questions with Dedicated Tests**: 24 (100%)
- **Test Files Created**: 7
- **Approximate Test Cases**: 50+
- **Board Types Covered**: square8, square19, hexagonal
- **Engines Validated**: GameEngine (backend), ClientSandboxEngine (sandbox)

### Notes for Game Designers

Game designers who want to see concrete examples of how a rule plays out on the
board can treat the RulesMatrix and FAQ suites as a **living, executable
rulebook**:

- Each scenario test encodes an explicit board setup, the sequence of canonical
  `Move` objects, and the expected final state.
- Both backend `GameEngine` and `ClientSandboxEngine` run these scenarios
  through the same shared helpers under
  [`src/shared/engine`](../src/shared/engine/types.ts:1), so behaviour in tests
  matches behaviour in production.
- When in doubt about an interpretation in
  [`ringrift_complete_rules.md`](../ringrift_complete_rules.md:1), look for a
  matching scenario in `tests/scenarios/RulesMatrix.*.test.ts` or
  `tests/scenarios/FAQ_*.test.ts` and treat that as the authoritative example.

For the complete FAQ → test mapping, see [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md:1) Section 9.

## Writing Deterministic Tests

### Using SeededRNG in Tests

When writing tests that involve random behavior (AI moves, tie-breaking, shuffling), always use explicit seeds for reproducibility:

```typescript
import { SeededRNG } from '../../src/shared/utils/rng';

describe('Deterministic AI Test', () => {
  it('should produce same result with same seed', () => {
    const seed = 42;
    const rng1 = new SeededRNG(seed);
    const rng2 = new SeededRNG(seed);

    // Both should produce identical sequences
    expect(rng1.next()).toBe(rng2.next());
  });
});
```

### Creating Games with Explicit Seeds

```typescript
import { createInitialGameState } from '../../src/shared/engine/initialState';

const gameState = createInitialGameState(
  gameId,
  boardType,
  players,
  timeControl,
  isRated,
  42 // Explicit seed for determinism
);
```

### Testing Sandbox AI with Seeds

```typescript
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { SeededRNG } from '../../src/shared/utils/rng';

const seed = 12345;
const rng = new SeededRNG(seed);
const engine = new ClientSandboxEngine({
  config: { boardType: 'square8', numPlayers: 2, playerKinds: ['ai', 'ai'] },
  interactionHandler: mockHandler,
});

// Use explicit RNG for deterministic AI behavior
await engine.maybeRunAITurn(() => rng.next());
```

### Testing Backend AI with Seeds

The backend `AIEngine` already accepts an optional RNG parameter:

```typescript
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';
import { SeededRNG } from '../../src/shared/utils/rng';

const seed = 999;
const rng = new SeededRNG(seed);

const move = globalAIEngine.chooseLocalMoveFromCandidates(playerNumber, gameState, candidates, () =>
  rng.next()
);
```

### Cross-Engine Parity with Seeds

When testing that backend and sandbox produce identical results:

```typescript
const seed = 42;
const backendRng = new SeededRNG(seed);
const sandboxRng = new SeededRNG(seed);

// Backend move selection
const backendMove = await backendAI.getMove(gameState, () => backendRng.next());

// Sandbox move selection
const sandboxMove = await sandboxEngine.maybeRunAITurn(() => sandboxRng.next());

// Should select equivalent moves
expect(backendMove.type).toBe(sandboxMove.type);
expect(backendMove.to).toEqual(sandboxMove.to);
```

### Python AI Service Tests

Python uses `random.Random(seed)` for deterministic sequences:

```python
import random
from app.ai.random_ai import RandomAI
from app.models import AIConfig

config = AIConfig(difficulty=3, randomness=0.2, rngSeed=42)
ai = RandomAI(player_number=2, config=config)

# All random operations use ai.rng (seeded Random instance)
move = ai.select_move(game_state)
```

### Guidelines for Deterministic Testing

1. **Always use explicit seeds** in tests involving randomness
2. **Document seed values** used in test descriptions
3. **Avoid `Math.random()`** - use `SeededRNG` instances instead
4. **Test both determinism and variation**:
   - Same seed → same output
   - Different seeds → different outputs (where applicable)
5. **Use seeds for debugging** - when a test fails, the seed can reproduce the exact scenario

---

## RNG hooks and AI parity tests

To support trace-mode debugging and backend↔sandbox AI comparisons under a **shared RNG policy**, the local AI selector and AI entrypoints accept an injectable RNG:

- `src/shared/engine/localAIMoveSelection.ts`:
  - `export type LocalAIRng = () => number;`
  - `chooseLocalMoveFromCandidates(player, gameState, candidates, rng = Math.random)`
  - All random draws inside this helper (placement vs skip, capture vs move, within-bucket choice) now call `rng()` instead of `Math.random()`.

- Sandbox AI:
  - `maybeRunAITurnSandbox(hooks, rng = Math.random)` in `src/client/sandbox/sandboxAI.ts`.
  - `ClientSandboxEngine.maybeRunAITurn(rng?: LocalAIRng)` in `src/client/sandbox/ClientSandboxEngine.ts` forwards `rng` into `maybeRunAITurnSandbox`.

- Backend local AI:
  - `AIEngine.chooseLocalMoveFromCandidates(player, gameState, candidates, rng = Math.random)` in `src/server/game/ai/AIEngine.ts` delegates to the shared selector with the provided RNG.
  - `AIEngine.getLocalAIMove(player, gameState, rng = Math.random)` uses the same RNG when falling back to local heuristics.

**Trace harness RNG wiring (`tests/utils/traces.ts`):**

- `runSandboxAITrace(boardType, numPlayers, seed, maxSteps)`:
  - Builds a deterministic LCG via `makePrng(seed)`.
  - Temporarily overrides `Math.random` for compatibility.
  - **Also** passes the same RNG instance into the sandbox engine: `await engine.maybeRunAITurn(rng);`.
  - This guarantees that sandbox AI decisions in trace-mode are driven by an explicit seeded RNG, not implicit global randomness.

**RNG-focused Jest suites:**

- `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts` (new)
  - Verifies that, when a RNG is provided:
    - `ClientSandboxEngine.maybeRunAITurn(rng)` uses the injected RNG and never calls `Math.random`.
    - `AIEngine.chooseLocalMoveFromCandidates(..., rng)` uses the injected RNG and never calls `Math.random`.

- `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts` (new, **diagnostic**, `describe.skip`)
  - Builds backend and sandbox engines from the same initial state on `square8` with 2 AI players.
  - For seeds like `1`, `5`, and `14`, and a small number of early steps, drives:
    - Backend via `GameEngine.getValidMoves` + `AIEngine.chooseLocalMoveFromCandidates(..., rngBackend)`.
    - Sandbox via `ClientSandboxEngine.maybeRunAITurn(rngSandbox)`.
  - Uses identically seeded RNG instances (`rngBackend`, `rngSandbox`) and the same loose move‑matching semantics as the heuristic coverage harness to assert that sandbox vs backend AI choose equivalent canonical moves while states remain structurally aligned.
  - Intended for **manual debugging** and deeper parity investigations; enable locally by removing `describe.skip` if needed.

## Canonical sandbox chain‑capture history

The client‑local sandbox engine (`ClientSandboxEngine`) now emits canonical capture‑chain history for **human** flows that aligns with backend semantics:

- Human clicks go through `ClientSandboxEngine.handleHumanCellClick`, which delegates capture/movement to `sandboxMovementEngine` with history‑aware hooks.
- Each capture chain is recorded as:
  - One `overtaking_capture` `Move` for the first segment.
  - One or more `continue_capture_segment` `Move`s for follow‑up segments while `currentPhase === 'chain_capture'`.
- `GameHistoryEntry.phaseBefore/phaseAfter` track entry into and exit from the `chain_capture` phase so history can be replayed or compared directly with backend traces.

Key scenario test:

- `tests/unit/ClientSandboxEngine.chainCapture.scenarios.test.ts`
  - Mirrors FAQ 15.3.1 (180° reversal pattern) on `square19` in the sandbox.
  - Drives the capture chain entirely via `handleHumanCellClick` (selecting the attacking stack, then the landing cell).
  - Asserts **board outcome** and **canonical history**:
    - Exactly two capture history entries: one `overtaking_capture` followed by one `continue_capture_segment`.
    - First entry transitions `phase: movement → chain_capture`.
    - Second entry transitions `phase: chain_capture → <non-chain phase>`.

These guarantees, together with the trace/parity helpers above, ensure that both AI‑driven and human‑driven capture chains in the sandbox can be compared directly to backend `GameEngine` behaviour via canonical `Move` history.

**Last Updated**: November 20, 2025  
**Framework**: Jest 29.7.0 + ts-jest 29.1.1
