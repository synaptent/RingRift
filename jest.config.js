/** @type {import('jest').Config} */

// =============================================================================
// TEST PROFILE SEPARATION (P0-TEST-001)
// =============================================================================
// Heavy Jest test suites that cause CI failures (OOM, RangeError: Invalid string
// length, Jest worker crashes) are separated into a "diagnostics" profile.
//
// CORE PROFILE (`npm run test:core`):
//   - Fast, reliable tests for PR gates
//   - Excludes heavy combinatorial/enumeration suites
//   - Should complete in under 5 minutes without OOM
//
// DIAGNOSTICS PROFILE (`npm run test:diagnostics`):
//   - Heavy suites that exhaustively enumerate state spaces
//   - Run nightly or manually, not on every PR
//
// HEAVY SUITES (excluded from core, included in diagnostics):
//   - GameEngine.decisionPhases.MoveDriven.test.ts
//       Enumerates/simulates large state spaces for decision phase coverage.
//       Causes: Node heap OOM (Allocation failed - JavaScript heap out of memory)
//
//   - RuleEngine.movementCapture.test.ts
//       Uses FakeBoardManager stub with direct RuleEngine instantiation.
//       Causes: Infinite hang during Jest module loading (possibly import graph issue)
//       Note: The underlying capture logic is fully covered by other passing tests.
//
// REFACTORED (now CI-safe, included in core):
//   - captureSequenceEnumeration.test.ts
//       Was: Exhaustively enumerates capture sequences across 150 random boards.
//       Now: Uses 10 deterministic test cases + 4 bounded random tests (max 2 targets).
//       Refactored to use bounded complexity with strict limits (MAX_CHAIN_DEPTH=4,
//       MAX_SEQUENCES_PER_CASE=50). Total runtime: ~1.4 seconds.
//
// To run core tests (CI default):  npm run test:core
// To run heavy diagnostics:        npm run test:diagnostics
// To run everything (local dev):   npm run test
//
// =============================================================================
// DIAGNOSING STUCK JEST WORKERS
// =============================================================================
// Common causes of Jest worker processes hanging indefinitely:
//
// 1. UNRESOLVED PROMISES / OPEN HANDLES
//    - setTimeout/setInterval not cleared in afterEach/afterAll
//    - WebSocket/HTTP connections not closed
//    - Database connections left open
//    - Event listeners not removed
//    Diagnose: npm run test:debug-hangs (uses --detectOpenHandles)
//
// 2. INFINITE LOOPS IN TEST SETUP/TEARDOWN
//    - beforeAll/afterAll that never resolve
//    - Circular imports causing module resolution hangs
//    - Mock implementations with infinite recursion
//
// 3. MEMORY PRESSURE / OOM
//    - Large state space enumeration tests
//    - Memory leaks in test fixtures
//    - Too many parallel workers on constrained systems
//    Mitigate: --workerIdleMemoryLimit=512MB kills bloated workers
//
// 4. EXTERNAL SERVICE DEPENDENCIES
//    - Tests waiting for unreachable services
//    - Missing mock for network calls
//    - DNS resolution hangs
//
// SAFEGUARDS IN PLACE:
//   - testTimeout: 30s per test (30min for coverage runs)
//   - --forceExit: Kill Jest when tests complete even if handles remain
//   - --workerIdleMemoryLimit=512MB: Kill workers exceeding memory
//   - timeout 30m wrapper: System-level kill for entire test run
//
// =============================================================================

 // Heavy diagnostic suite patterns (excluded from test:core, run by test:diagnostics)
const HEAVY_DIAGNOSTIC_SUITES = [
  'GameEngine\\.decisionPhases\\.MoveDriven\\.test\\.ts$',
  'RuleEngine\\.movementCapture\\.test\\.ts$',
];

  // Detect when Jest is invoked with coverage so we can safely exclude
 // heavy diagnostic suites that are known to be unstable under instrumentation.
 const isCoverageRun = process.argv.includes('--coverage');
 
 const COVERAGE_MAX_WORKERS = 2;
 const COVERAGE_TEST_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes per-test timeout for coverage runs
 
 // Base ignore patterns that always apply (regardless of coverage).
const BASE_TEST_PATH_IGNORE_PATTERNS = [
  '/node_modules/',
  '/dist/',
  '/build/',
  '<rootDir>/tests/e2e/',
  '<rootDir>/tests/unit/archive/',
];

// When collecting coverage, also ignore the heavy diagnostics suites to avoid
// extremely long runtimes and Node heap OOMs.
const COVERAGE_TEST_PATH_IGNORE_PATTERNS = isCoverageRun
  ? [...BASE_TEST_PATH_IGNORE_PATTERNS, ...HEAVY_DIAGNOSTIC_SUITES]
  : BASE_TEST_PATH_IGNORE_PATTERNS;

module.exports = {
  // Use ts-jest preset for TypeScript support
  preset: 'ts-jest',
  
  // Use custom jsdom environment with import.meta support
  testEnvironment: '<rootDir>/tests/jest-environment-jsdom.js',
  
  // Roots for test discovery
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  
  // Test match patterns
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/?(*.)+(spec|test).+(ts|tsx|js)'
  ],
  
  // Module paths
  modulePaths: ['<rootDir>/src'],
  
  // Transform files with ts-jest and handle import.meta
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: '<rootDir>/tsconfig.jest.json',
      diagnostics: {
        ignoreCodes: ['TS1343'], // Ignore "import.meta" errors during transformation
      },
      babelConfig: {
        plugins: ['babel-plugin-transform-import-meta'],
      },
    }],
  },
  
  // Coverage configuration
  collectCoverage: false, // Enable with --coverage flag
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/client/index.tsx',
    '!src/server/index.ts',
    '!src/**/__tests__/**',
    '!src/**/*.test.{ts,tsx}',
    '!src/**/*.spec.{ts,tsx}',
  ],
  
  // Coverage thresholds (80% target)
  // Global thresholds remain at 80%, and we also enforce the same target on
  // core rules modules so regressions in the engine/sandbox surface clearly.
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    '<rootDir>/src/server/game/BoardManager.ts': {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    '<rootDir>/src/server/game/RuleEngine.ts': {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    '<rootDir>/src/server/game/GameEngine.ts': {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    '<rootDir>/src/client/sandbox/ClientSandboxEngine.ts': {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  
  // Coverage reporters
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json',
  ],
  
  // Setup files - runs BEFORE test framework is installed
  setupFiles: ['<rootDir>/tests/setup-jsdom.ts'],
  
  // Setup files - runs AFTER test framework is installed
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  
  
  // Module name mapper for path aliases
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@shared/(.*)$': '<rootDir>/src/shared/$1',
    '^@server/(.*)$': '<rootDir>/src/server/$1',
    '^@client/(.*)$': '<rootDir>/src/client/$1',
    // Mock CSS and asset imports for React component tests
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/tests/__mocks__/fileMock.js',
  },
  
    // Ignore patterns
    // NOTE: E2E Playwright specs live under tests/e2e and are executed via
    // `npm run test:e2e` / Playwright directly, not via Jest. We exclude them
    // here so that "npm test" only runs Jest-based unit/contract/integration
    // suites (see tests/TEST_LAYERS.md for layering).
    testPathIgnorePatterns: COVERAGE_TEST_PATH_IGNORE_PATTERNS,
    
      // ===========================================================================
      // GLOBAL TEST TIMEOUT (CI Safety Net)
      // ===========================================================================
      // Default timeout for all tests. Individual tests may override with
      // jest.setTimeout() when needed. This acts as a safety net to prevent
      // tests from hanging indefinitely in CI environments.
      //
      // In non-coverage runs, 30 seconds is generous enough for complex game
      // simulations while still catching infinite loops and hung promises within
      // a reasonable time.
      //
      // Coverage runs get a much higher cap (30 minutes) to account for
      // instrumentation overhead while still enforcing a finite upper bound on
      // hung tests.
      // ===========================================================================
      testTimeout: isCoverageRun ? COVERAGE_TEST_TIMEOUT_MS : 30000,
      maxWorkers: isCoverageRun ? COVERAGE_MAX_WORKERS : '50%',
  
  // Verbose output
  // Default to non-verbose so large suites (especially AI simulations)
  // don't spam the console. You can override this per-run with
  // `jest --verbose` or `npm run test:verbose` when you actually
  // want per-test output.
  verbose: false,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks between tests
  restoreMocks: true,
  
  // Reset mocks between tests
  // NOTE: We keep resetMocks disabled so that manual jest.fn-based stub
  // implementations (e.g. the in-memory Prisma/bcrypt auth harness) remain
  // attached across tests. We still clear/restore mocks between tests via
  // clearMocks/restoreMocks and tests/setup.ts.
  resetMocks: false,
};
