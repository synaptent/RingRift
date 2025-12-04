# Test Categories

> **Doc Status (2025-12-01): Active**
>
> **Purpose:** Define the different categories of tests in the RingRift project, explaining which tests fall into which category and what "all tests passing" means for CI vs development.
>
> **Related docs:** `tests/README.md`, `tests/TEST_LAYERS.md`, `tests/TEST_SUITE_PARITY_PLAN.md`, `KNOWN_ISSUES.md`, `CURRENT_STATE_ASSESSMENT.md`, `docs/PASS19B_ASSESSMENT_REPORT.md`, `docs/PASS20_ASSESSMENT.md`, `jest-results.json`

## Overview

The RingRift test suite is organized into categories based on runtime requirements, purpose, and when they should run. Understanding these categories is essential for:

- Knowing what tests must pass before merging a PR
- Understanding what infrastructure is required for different test suites
- Debugging test failures appropriately

## Categories

### 1. CI-Gated Tests (Default)

**Purpose:** Core functionality tests that must always pass for merge.

**Run:** `npm run test:core` or `npm run test:ci`

**Requirements:** Node.js only (no external services)

**Characteristics:**

- Fast execution (~5 minutes)
- 100% deterministic
- No external service dependencies
- Excludes heavy/flaky suites

**Includes:**

- Unit tests (`tests/unit/*.test.ts`) - excluding heavy patterns
- Contract vector tests (`tests/contracts/`)
- Scenario tests (`tests/scenarios/RulesMatrix.*.test.ts`, `tests/scenarios/FAQ_*.test.ts`)
- Shared engine tests (`tests/unit/*.shared.test.ts`)
- Rules/parity-focused suites when run via the CI lanes below (see **1.1 Orchestrator & Rules-Parity Lanes**)

**Excludes from core:** (see `test:core` script in `package.json`)

- Heavy diagnostic suites (`GameEngine.decisionPhases.MoveDriven`, `RuleEngine.movementCapture`)
- AI simulation tests
- Parity/bisect diagnostic tests
- E2E tests (run via Playwright separately)

---

#### 1.1 Orchestrator & Rules-Parity Lanes (CI-gated)

These lanes are CI‑gated and focus on rules correctness and orchestrator/rules‑parity. They are part of the **CI profile** summarised in `CURRENT_STATE_ASSESSMENT.md` and `docs/PASS19B_ASSESSMENT_REPORT.md`.

**Commands:**

```bash
# Rules-level suites (shared engine + RulesMatrix/FAQ)
npm run test:ts-rules-engine

# Curated orchestrator-ON parity bundle (shared tests, contracts, RulesMatrix,
# FAQ, key backend/sandbox territory/capture tests)
npm run test:orchestrator-parity

# Orchestrator gating bundle (contract vectors + extended-vector soak + short soaks)
npm run orchestrator:gating
```

**Representative files/patterns:**

- Shared-engine & contracts:
  - `tests/unit/RefactoredEngine.test.ts`
  - `tests/unit/*shared.test.ts`
  - `tests/contracts/contractVectorRunner.test.ts`
  - `tests/fixtures/contract-vectors/v2/*.vectors.json`
- RulesMatrix & FAQ:
  - `tests/scenarios/RulesMatrix.*.test.ts`
  - `tests/scenarios/FAQ_Q*.test.ts`
  - `tests/scenarios/nearVictoryTerritory.test.ts`
  - `tests/scenarios/chainCaptureExtended.test.ts`
- Orchestrator-backed multi-phase suites:
  - `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`
  - `tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts`
- Decision/timeout invariants:
  - `tests/unit/GameSession.decisionPhaseTimeout.test.ts`
  - `tests/unit/GameSession.reconnectFlow.test.ts`
  - `tests/unit/GameSession.resignAndAbandon.test.ts`
  - `tests/unit/OrchestratorSInvariant.regression.test.ts`

**Notes:**

- These suites anchor the CI view of “rules are correct and hosts are in parity”.
- Failures here are **always** treated as regressions (see PASS19B/PASS20).
- Additional diagnostic/parity suites live under Category 3.

### 2. Environment-Gated Tests

**Purpose:** Integration tests requiring external services or specific configuration.

**Requirements:** Various infrastructure components

#### Python AI Service Tests

```bash
AI_SERVICE_URL=http://localhost:8001 \
RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION=1 \
npm test -- PythonRulesClient.live.integration.test.ts
```

**Files:**

- `tests/integration/PythonRulesClient.live.integration.test.ts`
- `tests/integration/PythonRulesClient.swapRule.live.integration.test.ts`

**Requires:** Python AI service running at `AI_SERVICE_URL`

#### Live Server Tests

```bash
ENABLE_LIVE_TESTS=true npm test -- LobbyRealtime.test.ts
```

**Files:**

- `tests/integration/LobbyRealtime.test.ts`

**Requires:** Backend server running locally

#### Seed Parity Tests (Optional Diagnostics)

```bash
RINGRIFT_ENABLE_SEED14_PARITY=1 npm test -- Seed14Move35LineParity
RINGRIFT_ENABLE_SEED17_PARITY=1 npm test -- Seed17
```

**Files:**

- `tests/unit/Seed14Move35LineParity.test.ts`
- `tests/unit/Seed17*.test.ts`
- `tests/unit/TraceParity.seed*.test.ts`

**Purpose:** Deep parity investigation for specific game seeds

---

### 3. Diagnostic Tests

**Purpose:** Heavy suites for debugging, analysis, or parity verification.

**Characteristics:**

- May be slow (~10+ minutes)
- May require increased heap size
- May have expected failures (tracked in `KNOWN_ISSUES.md`)
- Not required for PR merge

#### Heavy Suites

```bash
NODE_OPTIONS="--max-old-space-size=8192" npm run test:diagnostics
```

**Files:**

- `tests/unit/GameEngine.decisionPhases.MoveDriven.test.ts`
- `tests/unit/RuleEngine.movementCapture.test.ts`

**Issue:** Can cause OOM or Jest worker crashes

#### AI Simulation Tests

```bash
RINGRIFT_ENABLE_BACKEND_AI_SIM=1 npm run test:ai-backend:quiet
RINGRIFT_ENABLE_SANDBOX_AI_SIM=1 npm run test:ai-sandbox:quiet
```

**Files:**

- `tests/unit/GameEngine.aiSimulation.test.ts`
- `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`

**Purpose:** Run seeded AI-vs-AI games to stress test termination and invariants

#### AI Stall Diagnostics

```bash
RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1 npm test -- archive/tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts
```

**Files:**

- `archive/tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts` (archived, diagnostic-only stub at `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts`)

#### Legacy RuleEngine / GameEngine Diagnostics

These suites exercise legacy helpers and pre‑orchestrator code paths. They are **diagnostic only** and are not part of the CI‑gated rules surface.

**Representative files/patterns:**

- Legacy GameEngine flows:
  - `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`
  - `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`
  - `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`
  - `tests/unit/GameEngine.lines.scenarios.test.ts`
- Legacy RuleEngine / RuleEngine‑vs‑Sandbox parity:
  - `tests/unit/RuleEngine.movement.scenarios.test.ts`
  - `tests/unit/RuleEngine.placement*.test.ts`
  - `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`
  - `tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts`
  - `tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts`
- Archived or snapshot‑style GameEngine exports:
  - `tests/unit/ExportLineAndTerritorySnapshot.test.ts`
  - `tests/unit/ExportSeed*.test.ts`

**Notes:**

- New behaviour should **not** be added to these suites; use orchestrator/shared‑engine tests instead (Categories 1 and 2).
- Failures here are useful diagnostics during refactors but are **not** blockers for orchestrator‑first correctness.

#### Parity & Trace Diagnostics (TS)

```bash
# Host/trace parity suites (backend vs sandbox, TS vs Python)
npm run test:ts-parity

# Heavier WebSocket / integration flows (includes some reconnection/abandonment paths)
npm run test:ts-integration

# Full Jest run (CI-gated + diagnostic); used to generate jest-results.json
npm run test:all:quiet:log
```

**Files/patterns:**

- Host/trace parity:
  - `tests/unit/Backend_vs_Sandbox.*.test.ts` (plus historical suites under `archive/tests/unit/Backend_vs_Sandbox.*.test.ts`)
  - `tests/unit/Seed*.parity.test.ts`
  - `tests/unit/TraceParity.seed*.test.ts`
  - `tests/unit/Python_vs_TS.traceParity.test.ts`
  - `tests/unit/Territory*.GameEngine_vs_Sandbox.test.ts`
- Deep sandbox AI diagnostics:
  - `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`
  - `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`
  - `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`
- Misc. invariants/seed guards:
  - `tests/unit/SInvariant.seed17FinalBoard.test.ts`
  - `tests/unit/SharedMutators.invariants.test.ts`

**Notes:**

- These suites are the main contributors to the **72 failing tests in `jest-results.json`** analysed in `docs/PASS20_ASSESSMENT.md`.
- Many failures here are **expected diagnostics** or known gaps, documented in `KNOWN_ISSUES.md` and `docs/PARITY_SEED_TRIAGE.md`.
- They are not PR‑blocking but should be consulted when investigating full Jest runs.

---

### 4. E2E Tests (Playwright)

**Purpose:** Full browser-based user journeys.

**Run:** `npm run test:e2e`

**Requirements:**

- Backend server running
- Browser environment

**Files:** Playwright specs under `tests/e2e/`:

- `*.e2e.spec.ts` (UI‑centric flows)
- `*.test.ts` (WebSocket/multi‑client/network simulation helpers)

**Key suites:**

- `auth.e2e.spec.ts` - Authentication flows
- `game-flow.e2e.spec.ts` - Game play flows
- `sandbox.e2e.spec.ts` - Sandbox mode flows
- `multiplayer.e2e.spec.ts` - Multiplayer coordination
- `victory-conditions.e2e.spec.ts` - Near‑victory fixtures and victory modal UX
- `ratings.e2e.spec.ts` - Ratings display and rated vs unrated resign behaviour
- `decision-phase-timeout.e2e.spec.ts` - Decision‑phase timeout events for line/territory/chain_capture (shortTimeoutMs fixtures, deterministic `autoSelectedMoveId`)
- `multiPlayer.coordination.test.ts` - Multi‑client WebSocket coordination (near‑victory elimination/territory fixtures, swap‑rule meta‑moves, deep chain‑capture choices, back‑to‑back fixture runs with a spectator)
- `reconnection.simulation.test.ts` - Network partition/reconnection window abandonment flows (rated vs unrated, including rating behaviour)

**Key helpers:**

- `tests/helpers/MultiClientCoordinator.ts` - Multi‑client/WebSocket orchestration
- `tests/helpers/NetworkSimulator.ts` - Network partition / reconnect simulation
- `tests/helpers/TimeController.ts` - Time‑acceleration for decision and reconnection windows

**Smoke test:** `npm run test:e2e:smoke` (fast subset)

---

### 5. Python Tests & TS↔Python Parity

Python tests live under `ai-service/tests/` and are run with `pytest`. They are split into core behaviour/AI tests and contract/parity tests.

#### 5.1 Core Python Rules & AI Tests

**Run:**

```bash
cd ai-service
pytest

# Or from the repo root with logging:
npm run test:python:quiet:log
```

**Representative files:**

- `ai-service/tests/test_engine_correctness.py`
- `ai-service/tests/test_heuristic_ai.py`
- `ai-service/tests/test_minimax_ai.py`
- `ai-service/tests/test_mcts_ai.py`
- `ai-service/tests/test_progress_reporter.py`
- `ai-service/tests/test_cmaes_optimization.py`

**Purpose:**

- Validate Python rules/engine behaviour and AI algorithms.
- Support training/optimization flows (CMA‑ES, plateau diagnostics, etc.).

#### 5.2 Python Contract & Parity Tests

**Run:**

```bash
cd ai-service
pytest tests/contracts/test_contract_vectors.py -q
pytest tests/parity -q
```

**Files:**

- `ai-service/tests/contracts/test_contract_vectors.py`
- `ai-service/tests/parity/*.py`

**Purpose:**

- Ensure **49/49** v2 contract vectors pass identically in TS and Python (see `CURRENT_STATE_ASSESSMENT.md`, `docs/PASS18_REMEDIATION_PLAN.md`).
- Backstop TS↔Python rules parity for movement, capture/chain_capture, forced_elimination, territory/territory_line endgames (including `near_victory_territory`), hex edge cases, and meta moves (`swap_sides`, multi‑phase turns).

**Notes:**

- These suites, together with the TS contract runner, are the **authoritative cross‑language parity check** described in `docs/PYTHON_PARITY_REQUIREMENTS.md`.
- Failures here are treated as hard regressions; diagnostic TS parity suites (Category 3) are smoke/supporting signals.

## Test Organization

| Directory                     | Category               | Description                              |
| ----------------------------- | ---------------------- | ---------------------------------------- |
| `tests/unit/`                 | CI-Gated (mostly)      | Unit tests for modules/components        |
| `tests/unit/*.shared.test.ts` | CI-Gated               | Canonical shared engine tests            |
| `tests/contracts/`            | CI-Gated               | Contract vector runner                   |
| `tests/scenarios/`            | CI-Gated               | Rules/FAQ scenario tests                 |
| `tests/integration/`          | Environment-Gated      | Service integration tests                |
| `tests/e2e/`                  | E2E (Playwright)       | Browser automation tests                 |
| `tests/fixtures/`             | N/A                    | Test data and vectors                    |
| `tests/helpers/`              | N/A                    | Test utilities                           |
| `ai-service/tests/`           | Python (core + parity) | Python rules/AI + TS↔Python parity tests |

---

## Skipped Tests

Tests can be skipped for several reasons:

### Orchestrator Mode Skips

Many tests skip when `ORCHESTRATOR_ADAPTER_ENABLED=true` because they test legacy code paths:

```typescript
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
const describeOrSkip = skipWithOrchestrator ? describe.skip : describe;
```

**Affected suites (~25 files):**

- FAQ scenario tests (`FAQ_Q*.test.ts`)
- Parity tests (`*Parity*.test.ts`)
- Trace tests (`Trace*.test.ts`)
- Various backend↔sandbox comparison tests

### Feature Flag Skips

Tests that require opt-in via environment variable:

- `RINGRIFT_ENABLE_BACKEND_AI_SIM` / `RINGRIFT_ENABLE_SANDBOX_AI_SIM`
- `RINGRIFT_ENABLE_SEED14_PARITY` / `RINGRIFT_ENABLE_SEED17_PARITY`
- `RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO`
- `ENABLE_LIVE_TESTS`

### Legacy/Deprecated Skips

Tests marked `.skip` with explanatory comments:

- `tests/unit/archive/*.test.ts` - Archived tests
- Various `describe.skip` with migration notes

---

## Running Tests

### All CI Tests (Recommended for PRs)

```bash
npm run test:ci
```

### Specific Categories

```bash
# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# Contract tests only
npm test -- --testPathPattern="contracts"

# Scenario tests only
npm test -- --testPathPattern="scenarios"

# E2E tests
npm run test:e2e
```

### Orchestrator-Focused

```bash
# Rules engine with orchestrator enabled
npm run test:orchestrator-parity

# Orchestrator S-invariant regression
npm run test:orchestrator:s-invariant
```

### Diagnostic / Parity Profiles

```bash
# Heavy diagnostics (legacy GameEngine/RuleEngine helpers, large suites)
npm run test:diagnostics

# TS host/trace parity and WebSocket-heavy flows
npm run test:ts-parity
npm run test:ts-integration

# Python core + parity (from repo root)
npm run test:python:quiet:log
```

---

## Adding New Tests

When adding a new test file, determine its category:

1. **CI-Gated** → `tests/unit/` or `tests/scenarios/`
   - Keep fast (<1s per test)
   - No external dependencies
   - Must be deterministic

2. **Environment-Gated** → `tests/integration/`
   - Use feature flag for skip logic
   - Document required services
   - Add to README.md

3. **Diagnostic** → `tests/unit/` with skip logic
   - Use `RINGRIFT_ENABLE_*` pattern
   - Document in `KNOWN_ISSUES.md` if expected failures
   - Consider separate npm script

4. **E2E** → `tests/e2e/`
   - Use Playwright conventions
   - Add to smoke suite if fast

---

## What "All Tests Passing" Means

| Context           | Command                                                                             | Meaning                                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| PR Gate           | `npm run test:ci`                                                                   | CI‑gated Jest suites (core + rules/parity lanes) pass                                                                    |
| Full Jest (local) | `npm test` or `npm run test:all:quiet:log`                                          | Extended Jest profile (CI + diagnostic suites); see `jest-results.json` + `docs/PASS20_ASSESSMENT.md` for current status |
| Release Gate      | `npm run test:ci && npm run test:e2e:smoke` (plus targeted parity/soak scripts)     | Core + key E2E/soak gates pass (see `CURRENT_STATE_ASSESSMENT.md`, `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`)                  |
| Diagnostics       | `npm run test:diagnostics`, `npm run test:ts-parity`, `npm run test:ts-integration` | Heavy/parity suites pass (may have expected failures; see `KNOWN_ISSUES.md`)                                             |

---

## Related Documentation

- [`tests/README.md`](../tests/README.md) - Comprehensive testing guide
- [`tests/TEST_LAYERS.md`](../tests/TEST_LAYERS.md) - Test layer strategy
- [`tests/TEST_SUITE_PARITY_PLAN.md`](../tests/TEST_SUITE_PARITY_PLAN.md) - Parity test planning
- [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) - Expected failures and diagnostics
- [`docs/PARITY_SEED_TRIAGE.md`](./PARITY_SEED_TRIAGE.md) - Seed-specific debugging
- [`CURRENT_STATE_ASSESSMENT.md`](../CURRENT_STATE_ASSESSMENT.md) - Canonical test counts and CI profile
- [`docs/PASS19B_ASSESSMENT_REPORT.md`](./PASS19B_ASSESSMENT_REPORT.md) - CI‑gated test health summary
- [`docs/PASS20_ASSESSMENT.md`](./PASS20_ASSESSMENT.md) - Extended/diagnostic Jest profile analysis (including `jest-results.json`)
