# RingRift Testing Guide

> **Doc Status (2025-12-22): Active (test meta-doc, non-semantics)**
>
> **Coverage Status (2025-12-22):** 10,259+ TypeScript tests passing (75.67% statement coverage), 130 React component test files (80.55% component coverage), ~88% rules scenario coverage via 170 contract vectors and 30 scenario test files.
>
> **Role:** High-level guide to the RingRift TS+Python test suites: how they are structured, how to run them in different profiles (core/diagnostics/CI), and where to start when adding or debugging tests. This doc is a **test/meta reference only** – it explains how tests are organised and which commands/environments to use; it does **not** define game rules or lifecycle semantics.
>
> **Not a semantics SSoT:** Canonical rules and lifecycle semantics are owned by the rules specification (`RULES_CANONICAL_SPEC.md` plus `../docs/rules/COMPLETE_RULES.md` / `../docs/rules/COMPACT_RULES.md`) and its shared TypeScript engine implementation and contracts/vectors (`src/shared/engine/**`, `src/shared/engine/contracts/**`, `tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`), together with the lifecycle docs (`../docs/architecture/CANONICAL_ENGINE_API.md`). When this guide refers to “canonical semantics” or “authoritative tests”, it is pointing back to those SSoTs and to the rules-level suites that exercise them.
>
> **Related docs:** `./TEST_LAYERS.md`, `./TEST_SUITE_PARITY_PLAN.md`, `../docs/rules/PARITY_SEED_TRIAGE.md`, `RULES_SCENARIO_MATRIX.md`, `../docs/architecture/RULES_ENGINE_ARCHITECTURE.md`, `../docs/architecture/AI_ARCHITECTURE.md`, and `../DOCUMENTATION_INDEX.md`.

## Overview

This directory contains the comprehensive testing framework for RingRift. The testing infrastructure uses **Jest** with **TypeScript** support via ts-jest.

## Directory Structure

```
tests/
├── README.md                 # This file - testing documentation
├── setup.ts                  # Jest setup - runs AFTER test framework
├── setup-env.ts              # Jest env setup (dotenv, timers, etc.)
├── setup-jsdom.ts            # JSDOM-specific setup for React tests
├── test-environment.js       # Custom Jest environment (fixes localStorage)
├── jest-environment-jsdom.js # Custom JSDOM environment
├── TEST_LAYERS.md            # Test layer documentation
├── TEST_SUITE_PARITY_PLAN.md # Parity test planning
├── __mocks__/                # Jest module mocks
├── contracts/                # Contract test runners (TS side)
├── e2e/                      # End-to-end Playwright tests (helpers, POMs, flows)
├── fixtures/                 # Test fixtures and contract vectors
│   └── contract-vectors/v2/  # Contract test vectors (placement, movement, capture, etc.)
├── integration/              # Integration tests (full game flows, WebSocket)
├── scenarios/                # Rules/FAQ scenario tests
│   ├── FAQ_*.test.ts         # FAQ question coverage (Q1-Q24)
│   └── RulesMatrix.*.test.ts # Rules section coverage
├── scripts/                  # Test helper scripts
├── utils/
│   ├── fixtures.ts           # Test utilities and fixture creators
│   ├── traces.ts             # GameTrace utilities for parity tests
│   ├── prismaTestUtils.ts    # Prisma stub for route tests
│   └── ...                   # Additional test utilities
└── unit/
    ├── BoardManager.*.test.ts                 # Board geometry, lines, territory
    ├── GameEngine.*.test.ts                   # Core rules, chain capture, choices
    ├── ClientSandboxEngine.*.test.ts          # Client-local sandbox engine: movement, captures, lines, territory, victory
    ├── AIEngine.*.test.ts                     # AI service client + heuristics
    ├── WebSocket*.test.ts                     # WebSocket & PlayerInteractionManager flows
    ├── *Parity*.test.ts                       # Backend ↔ Sandbox parity tests
    ├── *.shared.test.ts                       # Shared engine helper tests
    └── ...                                    # Additional focused rule/interaction suites
```

## Running Tests

### Quick “How do I run X?” guide

This table mirrors the Jest layer/profile mapping in `../docs/testing/TEST_INFRASTRUCTURE.md` and `./TEST_LAYERS.md` and gives the shortest answer to “How do I run \<layer\>?“.

| Goal / Layer                             | What it runs (high level)                                                                                              | Command                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **Run all unit tests**                   | All `tests/unit/**` Jest suites (shared-engine, backend, sandbox, middleware, UI unit tests)                           | `npm run test:unit`                        |
| **Core PR/CI gate (fast core profile)**  | Unit + most integration tests, excluding heavy diagnostics and large parity/scenario suites                            | `npm run test:core`                        |
| **Check coverage (core focus)**          | All Jest tests except e2e/archive/heavy diagnostics, with coverage thresholds from `jest.config.js`                    | `npm run test:coverage`                    |
| **Run only heavy diagnostics**           | Heavy suites (`ClientSandboxEngine.territoryDecisionPhases.MoveDriven`, `MovementCaptureParity.RuleEngine_vs_Sandbox`) | `npm run test:diagnostics`                 |
| **TS rules engine / RulesMatrix focus**  | Shared-engine rules suites: `RefactoredEngine`, `*.rules.*`, `RulesMatrix.*`, FAQ/RulesMatrix scenarios                | `npm run test:ts-rules-engine`             |
| **TS parity / trace diagnostics**        | TS↔Python and host parity traces: `*Parity.*`, `TraceParity.*`, `Python_vs_TS.traceParity.test.ts`                     | `npm run test:ts-parity`                   |
| **Backend/WebSocket/HTTP integration**   | Integration suites in `tests/integration/**`, `WebSocketServer.*`, `FullGameFlow.test.ts`                              | `npm run test:ts-integration`              |
| **Orchestrator parity (TS, adapter ON)** | `.shared` tests, contract vectors, RulesMatrix + FAQ suites, key territory/line/chain-capture decisions                | `npm run test:orchestrator-parity`         |
| **AI simulation diagnostics (backend)**  | `tests/unit/GameEngine.aiSimulation.test.ts`                                                                           | `npm run test:ai-backend:quiet`            |
| **AI simulation diagnostics (sandbox)**  | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`                                                                  | `npm run test:ai-sandbox:quiet`            |
| **AI movement/capture diagnostics**      | `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`                                                            | `npm run test:ai-movement:quiet`           |
| **All Jest tests (TS, no Playwright)**   | Everything under Jest except `tests/e2e/**` and archived suites filtered by `jest.config.js`                           | `npm test`                                 |
| **All Playwright E2E tests**             | Browser-based journeys in `tests/e2e/**`                                                                               | `npm run test:e2e`                         |
| **Python contract vectors (AI/rules)**   | `ai-service/tests/contracts/test_contract_vectors.py` and related parity/contract tests                                | `cd ai-service && pytest tests/contracts/` |

For a more detailed explanation of layers and which suites are semantic gates vs diagnostics, see:

- `../docs/testing/TEST_INFRASTRUCTURE.md` – infra helpers, Jest profiles, and the full layer→command table.
- `./TEST_LAYERS.md` – layer definitions, examples, and CI usage.
- `./TEST_SUITE_PARITY_PLAN.md` – classification of parity/trace suites and their canonical anchors.

For a more detailed explanation of layers and which suites are semantic gates vs diagnostics, see:

- `../docs/testing/TEST_INFRASTRUCTURE.md` – infra helpers, Jest profiles, and the full layer→command table.
- `./TEST_LAYERS.md` – layer definitions, examples, and CI usage.
- `./TEST_SUITE_PARITY_PLAN.md` – classification of parity/trace suites and their canonical anchors.

### Test Profiles (P0-TEST-001)

The test suite is split into two core profiles to ensure CI reliability:

#### Core Profile (`npm run test:core`)

- **Purpose**: Fast, reliable tests for PR gates.
- **Runs**: All unit and integration tests EXCEPT heavy suites (see `package.json` and `jest.config.js` for ignore patterns).
- **Duration**: Should complete in under 5 minutes.
- **Used by**: `npm run test:ci`.

#### Diagnostics Profile (`npm run test:diagnostics`)

- **Purpose**: Heavy combinatorial/enumeration suites.
- **Runs**: Only the heavy diagnostic suites.
- **Duration**: May take 10+ minutes and require increased heap size.
- **Used by**: Nightly CI runs, manual debugging.

> **Orchestrator defaults:** Unless explicitly overridden, both profiles are
> expected to run with `ORCHESTRATOR_ADAPTER_ENABLED=true` and
> `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`, treating the shared turn orchestrator
> as the default rules path. Legacy/SHADOW modes and any tests that require
> `ORCHESTRATOR_ADAPTER_ENABLED=false` should be considered **diagnostic
> only** and are documented in `./TEST_LAYERS.md` / `./TEST_SUITE_PARITY_PLAN.md`.

#### P0 Robustness Profile (`npm run test:p0-robustness`)

- **Purpose**: One-stop “P0 robustness” lane that combines rules-engine, integration, and key parity/cancellation tests for real-world safety around Q7/Q20 territory and decision-phase termination.
- **Runs (in order)**:
  - `npm run test:ts-rules-engine` – shared-engine + orchestrator rules suites (RulesMatrix, FAQ, advanced turn/territory helpers).
  - `npm run test:ts-integration` – backend/WebSocket/full game-flow integration tests.
  - `jest --runInBand` on:
    - `tests/contracts/contractVectorRunner.test.ts` (v2 contract vectors, including mixed line+territory sequences),
    - `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` (advanced capture + single-/multi-region line+territory backend↔sandbox parity),
    - `tests/unit/WebSocketServer.sessionTermination.test.ts` (session termination + decision-phase cancellation, including AI-backed choices).
- **Recommended use**: Local P0 gate for any change that touches rules, advanced territory behaviour, WebSocket lifecycle, or AI/decision-phase flows. CI can either call this script directly or treat its constituent lanes as required jobs.

> **PR workflow tip:** For any PR that touches `src/shared/engine/**`, `src/server/game/**`, `src/client/sandbox/**`, or WebSocket/AI adapters, run:
>
> ```bash
> npm run test:p0-robustness
> ```
>
> locally before pushing. This is the canonical “rules + AI + WebSocket robustness” bundle and includes the fixed `Sandbox_vs_Backend.aiHeuristicCoverage` harness and `WebSocketServer.sessionTermination` tests as first-class signals (not flaky diagnostics).

#### Heavy Suites (excluded from core)

The following suites are excluded from `test:core` due to runtime and memory pressure:

1. **`ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts`**
   - Enumerates large decision-phase state spaces to stress territory processing
   - Can trigger heap pressure and longer runtimes

2. **`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`**
   - Compares backend vs sandbox movement/capture enumeration parity
   - Slow on larger boards; best run on-demand

To run diagnostics with increased heap:

```bash
NODE_OPTIONS="--max-old-space-size=8192" npm run test:diagnostics
```

### Basic Commands

```bash
# Run all tests (local development)
npm test

# Run core tests only (excludes heavy suites) - RECOMMENDED FOR CI
npm run test:core

# Run heavy diagnostic suites only
npm run test:diagnostics

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run tests with coverage in watch mode
npm run test:coverage:watch

# Run tests for CI/CD (uses core profile, optimized)
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

#### Orchestrator-focused lanes

In addition to the core scripts above, a set of orchestrator- and rules-focused
lanes are wired into CI as primary gates:

```bash
# TS rules-level suites (shared engine + orchestrator)
npm run test:ts-rules-engine

# Curated orchestrator-ON parity bundle (shared tests, contracts, RulesMatrix,
# FAQ, key backend/sandbox territory disconnection tests)
npm run test:orchestrator-parity

# Additional CI lanes
npm run test:ts-parity        # Trace/host parity and RNG-oriented suites
npm run test:ts-integration   # Routes/WebSocket/full game-flow integration

# Orchestrator invariant smoke + S-invariant regression harness
npm run soak:orchestrator:smoke    # Single short backend game; fails on invariant violations
npm run test:orchestrator:s-invariant  # Seeded S-invariant regression tests promoted from soak

# Single-source-of-truth guardrails (docs/env/CI/rules + fences)
npm run ssot-check
```

For a lightweight HTTP load/scale smoke against a running backend under the
orchestrator‑ON profile, you can also run:

```bash
TS_NODE_PROJECT=tsconfig.server.json npm run load:orchestrator:smoke
#
# Or run the full orchestrator preflight bundle (invariant soak smoke +
# HTTP/load smoke + metrics/observability smoke) via:
#
npm run smoke:orchestrator:preflight
```

This script:

- Registers a small number of throwaway users via `/api/auth/register`.
- Creates short games via `/api/games` and fetches game lists/details.
- Optionally inspects `/metrics` for orchestrator rollout metrics.

It is intended as a developer‑driven smoke harness rather than a hard CI gate,
and assumes the server is already running with orchestrator enabled as in
`ORCHESTRATOR_ROLLOUT_PLAN.md` Table 4.

For a **metrics & observability smoke** that validates the `/metrics` endpoint
and the presence of key orchestrator gauges, there is a dedicated Playwright
E2E spec:

```bash
npm run test:e2e -- tests/e2e/metrics.e2e.spec.ts
```

This suite:

- Waits for the backend API to become ready (`/ready`).
- Scrapes `/metrics` via `E2E_API_BASE_URL` (or `http://localhost:3000`).
- Asserts Prometheus-format output and the presence of:
  - `ringrift_orchestrator_error_rate`
  - `ringrift_orchestrator_rollout_percentage`
  - `ringrift_orchestrator_circuit_breaker_state`

It is a fast, non-exhaustive guardrail to catch regressions where metrics are
disabled or orchestrator gauges are removed/renamed, and complements the
alert definitions and dashboard guidance in `docs/operations/ALERTING_THRESHOLDS.md`.

CI runs these lanes with the orchestrator adapter forced ON:

- `RINGRIFT_RULES_MODE=ts`
- `ORCHESTRATOR_ADAPTER_ENABLED=true`
- `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`
- `ORCHESTRATOR_SHADOW_MODE_ENABLED=false`

and the **TS Orchestrator Parity (adapter‑ON)** job is intended to be a required
status check for `main` alongside the core `TS Rules Engine (rules-level)` lane.

### CI‑gated vs diagnostic commands (P0-TEST-002)

For quick reference, these are the primary commands used in CI vs the broader
diagnostic/extended profiles:

**CI‑gated lanes (must stay green):**

- `npm run test:ci` – core Jest CI profile (unit + integration, E2E excluded; heavy/diagnostic suites filtered via `testPathIgnorePatterns`).
- `npm run test:ts-rules-engine` – rules‑level + RulesMatrix/FAQ coverage (shared engine, orchestrator parity).
- `npm run orchestrator:gating` – orchestrator gating bundle (contract vectors and extended‑vector soak + short orchestrator soaks).

**Extended / diagnostic lanes (opt‑in, may include expected failures):**

- `npm run test:ai-backend:quiet` / `npm run test:ai-sandbox:quiet` – backend and sandbox AI S‑invariant simulations (`GameEngine.aiSimulation.test.ts`, `ClientSandboxEngine.aiSimulation.test.ts`).
- `npm run test:ai-movement:quiet` – sandbox AI movement/capture diagnostics.
- `npm run test:ts-parity` – trace and host parity suites (Backend_vs_Sandbox trace parity, Python_vs_TS trace parity, deep seed parity diagnostics).
- `npm run test:ts-integration` – heavier WebSocket/integration flows (multi‑player, reconnect, rules‑backend integration).
- `npm run test:all:quiet:log` – full Jest run with output captured to `logs/jest/latest.log` (used to generate `jest-results.json`; includes CI‑gated + diagnostic suites).

In documentation:

- **PASS19B_ASSESSMENT_REPORT.md** and **docs/archive/historical/CURRENT_STATE_ASSESSMENT.md** report counts for the **CI‑gated** profile.
- **PASS20_ASSESSMENT.md** and `jest-results.json` analyse the broader **extended/diagnostic** profile. When interpreting failures there, cross‑check `KNOWN_ISSUES.md` and `../docs/rules/PARITY_SEED_TRIAGE.md` to distinguish expected diagnostics from regressions.

### Mapping jest-results.json failures to TEST_CATEGORIES (P0-TEST-003)

When inspecting `jest-results.json` (or `logs/jest/latest.view.txt` from `npm run test:all:quiet:log`), use the file path to locate the test **category**:

- `tests/unit/Backend_vs_Sandbox.*.test.ts`, `tests/unit/TraceParity.seed*.test.ts`, `tests/unit/Sandbox_vs_Backend.*.test.ts`  
  → **Diagnostic parity/trace** category (see “Parity & Trace Diagnostics (TS)” in `docs/TEST_CATEGORIES.md`).
- `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`  
  → **AI Simulation (Diagnostic)** category.
- `tests/scenarios/RulesMatrix.*.test.ts`, `tests/scenarios/FAQ_Q*.test.ts`, `tests/contracts/contractVectorRunner.test.ts`  
  → **Rules/Contracts (CI‑gated)** category.
- `tests/e2e/*.e2e.spec.ts`, `tests/e2e/*reconnection*.test.ts`, `tests/e2e/*multiPlayer*.test.ts`  
  → **E2E (Playwright)** category.
- `ai-service/tests/contracts/test_contract_vectors.py`, `ai-service/tests/parity/*.py`  
  → **Python Contract/Parity** category.

For each category, `docs/TEST_CATEGORIES.md` documents:

- Whether failures are **CI‑gated** vs **diagnostic/optional**.
- Which commands drive them (e.g. `npm run test:ci`, `npm run test:ts-parity`, `npm run test:ai-backend:quiet`, `pytest ai-service/tests/...`).

**Legacy RuleEngine/GameEngine diagnostics examples:**

- `tests/unit/GameEngine.lines.scenarios.test.ts`, `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`  
  → **Legacy RuleEngine / GameEngine Diagnostics** (see “Diagnostic Tests → Legacy RuleEngine / GameEngine Diagnostics” in `docs/TEST_CATEGORIES.md`). These suites exist for historical coverage and migration safety and are not CI‑gated; new rules work should target orchestrator/shared‑engine suites instead.

#### Python‑mode TS↔Python parity profile (`RINGRIFT_RULES_MODE=python`)

For targeted runtime parity debugging, run the backend in Python‑authoritative validation mode. The backend validates moves via Python, applies them through TS, and emits parity mismatch metrics.

```bash
RINGRIFT_RULES_MODE=python npm run dev:server
```

Then, in a separate terminal, drive a small amount of traffic (for example via the orchestrator HTTP/load smoke):

```bash
TS_NODE_PROJECT=tsconfig.server.json npm run load:orchestrator:smoke
```

and inspect `/metrics` for:

- `ringrift_rules_parity_mismatches_total{suite="runtime_python_mode",mismatch_type=...}`
- `ringrift_orchestrator_shadow_mismatch_rate`

This profile is **not** used by default in CI (which runs with `RINGRIFT_RULES_MODE=ts`); it is intended for ad‑hoc parity investigations and to exercise the shadow‑mode counters and alerts described in `docs/operations/ALERTING_THRESHOLDS.md` and `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`.

### High-Level Testing Overview (by purpose)

This section groups the Jest suites by **what** they validate. For detailed
classification (rules-level vs trace-level vs integration-level) see
[`./TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md:28).

#### 1. Shared-helper rules tests (canonical semantics)

These suites exercise the shared rules helpers under
[`src/shared/engine`](../src/shared/engine/types.ts:1) and should be treated as
the primary specification for game semantics. In particular, the **decision‑helper
suites for lines and territory** are the authoritative spec for line/territory
decision phases; per‑host integration/parity tests are consumers of these
semantics, not independent sources of truth.

- **Movement & captures**
  - [`movement.shared.test.ts`](./unit/movement.shared.test.ts:1) – canonical non‑capturing movement reachability over [`movementLogic.ts`](../src/shared/engine/movementLogic.ts:1), including integration with the shared no‑dead‑placement helper `hasAnyLegalMoveOrCaptureFromOnBoard`.
  - [`captureLogic.shared.test.ts`](./unit/captureLogic.shared.test.ts:1) – canonical overtaking capture enumeration and reachability over [`captureLogic.ts`](../src/shared/engine/captureLogic.ts:1).
  - [`captureSequenceEnumeration.test.ts`](./unit/captureSequenceEnumeration.test.ts:1) – legacy capture‑sequence enumeration harness kept as a regression/diagnostic suite over the same helpers.
  - [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](./unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1) – backend adapter alignment with shared movement/capture helpers.
- **Lines**
  - [`lineDetection.shared.test.ts`](./unit/lineDetection.shared.test.ts:1) – shared marker‑line geometry.
  - [`LineDetectionParity.rules.test.ts`](./unit/LineDetectionParity.rules.test.ts:1) – line semantics across shared engine, backend, and sandbox.
  - [`Seed14Move35LineParity.test.ts`](./unit/Seed14Move35LineParity.test.ts:1) – seed‑14 guardrail for “no valid line” at a historically ambiguous state.
  - [`lineDecisionHelpers.shared.test.ts`](./unit/lineDecisionHelpers.shared.test.ts:1) – **canonical line‑decision enumeration and application**, covering `process_line` and `choose_line_option` `Move`s and when line collapses grant `eliminate_rings_from_stack` opportunities via `pendingLineRewardElimination`.
- **Territory (detection, borders, processing)**
  - [`territoryBorders.shared.test.ts`](./unit/territoryBorders.shared.test.ts:1) – shared border‑marker expansion.
  - [`territoryProcessing.shared.test.ts`](./unit/territoryProcessing.shared.test.ts:1) – shared region‑processing pipeline (collapse + internal eliminations).
  - [`TerritoryAggregate.hex.feParity.test.ts`](./unit/TerritoryAggregate.hex.feParity.test.ts:1),
    [`sandboxTerritory.rules.test.ts`](./unit/sandboxTerritory.rules.test.ts:1),
    [`sandboxTerritoryEngine.rules.test.ts`](./unit/sandboxTerritoryEngine.rules.test.ts:1) – rules‑level suites for Q23, region collapse, and internal eliminations.
  - [`territoryDecisionHelpers.shared.test.ts`](./unit/territoryDecisionHelpers.shared.test.ts:1) – **canonical territory decision semantics**, covering `choose_territory_option` / `eliminate_rings_from_stack` `Move`s, Q23 gating, and per‑player elimination bookkeeping.
- **Placement / no-dead-placement**
  - [`placement.shared.test.ts`](./unit/placement.shared.test.ts:1) – shared placement validation rules.
  - [`RuleEngine.placementMultiRing.test.ts`](./unit/RuleEngine.placementMultiRing.test.ts:1) – backend multi‑ring placement behaviour over the shared helpers.
- **Victory & invariants**
  - [`victory.shared.test.ts`](./unit/victory.shared.test.ts:1) – shared victory/stalemate ladder over `VictoryAggregate.ts`.
  - [`SInvariant.seed17FinalBoard.test.ts`](./unit/SInvariant.seed17FinalBoard.test.ts:1),
    [`SharedMutators.invariants.test.ts`](./unit/SharedMutators.invariants.test.ts:1),
    [`ProgressSnapshot.core.test.ts`](./unit/ProgressSnapshot.core.test.ts:1) – S‑invariant and progress‑snapshot guarantees.
- **Turn sequencing & termination**
  - [`GameEngine.turnSequence.scenarios.test.ts`](./unit/GameEngine.turnSequence.scenarios.test.ts:1) – backend turn ladder over the shared `turnLogic.ts`.
  - [`MovementNoAction.anmParity.test.ts`](./unit/MovementNoAction.anmParity.test.ts:1),
    [`ClientSandboxEngine.aiSimulation.test.ts`](./unit/ClientSandboxEngine.aiSimulation.test.ts:1) – shared termination ladder as exercised by sandbox AI.

When changing rules, update or extend these suites first wherever possible.

#### 2. Host parity tests (backend ↔ sandbox ↔ shared/Python)

These suites ensure that backend `GameEngine`/`RuleEngine`, `ClientSandboxEngine`,
and Python rules behave identically for a given ruleset:

- **Movement / capture / placement**
  - [`MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](./unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1) – backend adapter alignment with shared movement/capture helpers.
  - [`archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`](../archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts:1) – **archived** seeded trace parity harness for movement/capture/placement between backend and sandbox (**diagnostic only; non‑canonical and not a CI gate**).
  - [`Backend_vs_Sandbox.eliminationTrace.test.ts`](./parity/Backend_vs_Sandbox.eliminationTrace.test.ts:1) – elimination and capture/placement consequence traces.
  - [`Backend_vs_Sandbox.seed5.internalStateParity.test.ts`](./parity/Backend_vs_Sandbox.seed5.internalStateParity.test.ts:1), [`Backend_vs_Sandbox.seed5.checkpoints.test.ts`](./parity/Backend_vs_Sandbox.seed5.checkpoints.test.ts:1) – deep internal-state parity for a canonical seed.
  - [`archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`](../archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts:1), [`archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`](../archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts:1) – snapshot‑level regression nets (both suites fully archived; kept for historical debugging only and excluded from CI parity gates).
  - [`movementReachabilityParity.test.ts`](./unit/movementReachabilityParity.test.ts:1), [`CaptureMarker.GameEngine_vs_Sandbox.test.ts`](./unit/CaptureMarker.GameEngine_vs_Sandbox.test.ts:1) – marker‑path and capture/elimination parity harnesses.
- **Territory & borders**
  - Board‑level region‑detection geometry:
    - [`BoardManager.territoryDisconnection.square8.test.ts`](./unit/BoardManager.territoryDisconnection.square8.test.ts:1)
    - [`BoardManager.territoryDisconnection.test.ts`](./unit/BoardManager.territoryDisconnection.test.ts:1)
    - [`BoardManager.territoryDisconnection.hex.test.ts`](./unit/BoardManager.territoryDisconnection.hex.test.ts:1)
  - Engine‑level region processing:
    - [`ClientSandboxEngine.territoryDisconnection.test.ts`](./unit/ClientSandboxEngine.territoryDisconnection.test.ts:1)
    - [`ClientSandboxEngine.territoryDisconnection.hex.test.ts`](./unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts:1)
  - Backend↔sandbox territory parity:
    - [`TerritoryBorders.Backend_vs_Sandbox.test.ts`](./unit/TerritoryBorders.Backend_vs_Sandbox.test.ts:1)
    - [`TerritoryCore.GameEngine_vs_Sandbox.test.ts`](./unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts:1)
    - [`TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts`](./unit/TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts:1)
    - [`archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`](../archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1) – **archived diagnostic 19×19 parity harness**; canonical territory decision semantics now live in [`territoryDecisionHelpers.shared.test.ts`](./unit/territoryDecisionHelpers.shared.test.ts:1) and the RulesMatrix/FAQ Q23 suites.
    - [`TerritoryDecision.seed5Move45.parity.test.ts`](./unit/TerritoryDecision.seed5Move45.parity.test.ts:1),
      [`TerritoryDetection.seed5Move45.parity.test.ts`](./unit/TerritoryDetection.seed5Move45.parity.test.ts:1),
      [`TerritoryDecisions.GameEngine_vs_Sandbox.test.ts`](./unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts:1) – seed‑based diagnostics around specific territory scenarios (**diagnostic, may be skipped in CI**).
- **Victory**
  - [`VictoryParity.RuleEngine_vs_Sandbox.test.ts`](./unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts:1)
- **Trace/fixture parity (shared engine vs hosts, TS vs Python)**
  - [`TraceFixtures.sharedEngineParity.test.ts`](./unit/TraceFixtures.sharedEngineParity.test.ts:1) – shared `GameEngine` vs backend.
  - [`archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`](../archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts:1) – **archived** seeded trace parity harness between backend and sandbox (**diagnostic only; semantics anchored in shared‑engine rules tests and contract vectors**).
  - [`Sandbox_vs_Backend.aiRngParity.test.ts`](./unit/Sandbox_vs_Backend.aiRngParity.test.ts:1),
    [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](./unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1) – RNG‑driven AI parity harnesses (**diagnostic; `Sandbox_vs_Backend.aiRngFullParity` is `describe.skip` by default and not a CI gate**).
  - [`Python_vs_TS.traceParity.test.ts`](./unit/Python_vs_TS.traceParity.test.ts:1) plus the Python‑side parity suites under
    `ai-service/tests/parity/*`, driven by shared‑engine fixtures.

These are **smoke tests and regression nets**; when they disagree with rules‑level
suites (including the shared decision‑helper tests for lines and territory),
treat the traces/fixtures as derived artifacts (see below and
[`./TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md:56)).

**NOTE:** Maintainers should periodically confirm which of the above parity
suites run in CI vs are `describe.skip`/env‑gated, and keep any heavy 19×19 or
seed‑based territory/AI RNG harnesses clearly marked as **diagnostic**.

#### 3. Scenario / RulesMatrix / FAQ tests

Scenario-style suites encode concrete board positions and expected behaviour:

- **RulesMatrix scenarios**
  - Files under `tests/scenarios/RulesMatrix.*.test.ts`, for example
    [`RulesMatrix.Territory.MiniRegion.test.ts`](./scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1) and
    the comprehensive movement/territory/victory suites referenced from
    [`RULES_SCENARIO_MATRIX.md`](../docs/rules/RULES_SCENARIO_MATRIX.md:1).
- **FAQ scenarios (Q1–Q24)**
  - [`FAQ_Q01_Q06.test.ts`](./scenarios/FAQ_Q01_Q06.test.ts:1),
    [`FAQ_Q07_Q08.test.ts`](./scenarios/FAQ_Q07_Q08.test.ts:1),
    [`FAQ_Q09_Q14.test.ts`](./scenarios/FAQ_Q09_Q14.test.ts:1),
    [`FAQ_Q15.test.ts`](./scenarios/FAQ_Q15.test.ts:1),
    [`FAQ_Q16_Q18.test.ts`](./scenarios/FAQ_Q16_Q18.test.ts:1),
    [`FAQ_Q19_Q21_Q24.test.ts`](./scenarios/FAQ_Q19_Q21_Q24.test.ts:1),
    [`FAQ_Q22_Q23.test.ts`](./scenarios/FAQ_Q22_Q23.test.ts:1).
- **Compound/termination scenarios**
  - [`ForcedEliminationAndStalemate.test.ts`](./scenarios/ForcedEliminationAndStalemate.test.ts:12),
    `LineAndTerritory.test.ts`, and other compound examples linked from
    [`RULES_SCENARIO_MATRIX.md`](../docs/rules/RULES_SCENARIO_MATRIX.md:1).

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

````typescript
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

## Playwright E2E tests and helpers (`tests/e2e`)

The `tests/e2e` directory contains Playwright tests, shared helpers, and
Page Object Models (POMs) for end‑to‑end coverage of auth, lobby, game
flows, and the sandbox host:

- `tests/e2e/helpers/test-utils.ts` – canonical E2E helper surface:
  - Auth helpers: `generateTestUser`, `registerUser`, `loginUser`,
    `registerAndLogin`, `logout`.
  - Backend readiness: `waitForApiReady(page)` polls the backend `/ready`
    endpoint (using `E2E_API_BASE_URL` or `http://localhost:3000`) before
    any auth/game calls to avoid Vite proxy 500s during startup.
  - Game helpers: `createGame`, `createBackendGameFromLobby`,
    `waitForGameReady`, `waitForWebSocketConnection`, `joinGame`,
    `clickValidPlacementTarget`, `makeMove`, `placePiece`.
  - Navigation helpers: `goToHome`, `goToLobby`, `goToGame`, `goToSandbox`
    (navigates to `/sandbox` and asserts the sandbox pre‑game heading).
  - Board/log assertions: `assertBoardState`, `assertPlayerTurn`,
    `assertGamePhase`, `assertMoveLogged`, `waitForMoveLog`.
  - Multi‑player helpers: `setupMultiplayerGame`, `coordinateTurn`,
    `waitForTurn`, `isPlayerTurn`, `makeRingPlacement`, `cleanupMultiplayerGame`.

- `tests/e2e/helpers/index.ts` – re‑exports the helpers and POMs so tests
  can import from a single module:

  ```ts
  import {
    registerAndLogin,
    createGame,
    waitForGameReady,
    goToLobby,
    goToSandbox,
    GamePage,
  } from './helpers';
````

- `tests/e2e/pages/*.ts` – Page Object Models:
  - `LoginPage`, `RegisterPage`, `HomePage` – auth + landing flows.
  - `GamePage` – `/game/:gameId` host; encapsulates board selectors, game
    log, and HUD connection state:
    - `waitForReady()` waits for board + `"Connection: Connected"` +
      turn indicator, mirroring `waitForGameReady`.
    - `assertConnected()` asserts the HUD reports a stable `"Connection: Connected"`
      state rather than a transient reconnect message.

### E2E helper smoke tests

The `tests/e2e/helpers.smoke.e2e.spec.ts` suite provides a fast sanity
check that the helpers and POMs are wired correctly:

- Verifies `generateTestUser` structure/uniqueness.
- Exercises `LoginPage`/`RegisterPage` navigation.
- Confirms `registerAndLogin` yields an authenticated session and that
  `HomePage.assertUsernameDisplayed` handles responsive layouts.
- Creates a backend AI game via `createGame` and checks `GamePage.waitForReady`.
- Asserts `waitForGameReady` waits for core UI + HUD connection state.
- Exercises navigation helpers:
  - `goToLobby` (after auth).
  - `goToHome` (from login).
  - `goToSandbox` (ensures the sandbox pre‑game view and “Launch Game”
    control are present).

For a quick E2E sanity run in local dev or CI, use the Playwright smoke
script:

```bash
npm run test:e2e:smoke
```

which runs:

- `auth.e2e.spec.ts` – core auth flows.
- `helpers.smoke.e2e.spec.ts` – helper and POM wiring.
- `sandbox.e2e.spec.ts` – `/sandbox` host “Launch Game” path into `/game/:gameId`.

When adding new E2E helpers or POM capabilities, prefer to:

- Export them via `tests/e2e/helpers/index.ts` for discoverability.
- Add a small assertion to `helpers.smoke.e2e.spec.ts` so that changes to
  routes or headings are caught early without depending on heavier game
  flow suites.

````

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
````

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
  Directly exercise **canonical rules semantics** as implemented in `src/shared/engine/` and documented in `../docs/rules/COMPLETE_RULES.md`.
  - Prefer to go through `src/shared/engine/GameEngine` and its validators/mutators where possible.
  - Examples:
    - `tests/unit/RefactoredEngine.test.ts`
    - `tests/unit/LineDetectionParity.rules.test.ts`
    - `tests/unit/sandboxTerritory.rules.test.ts`
    - `tests/unit/territoryProcessing.shared.test.ts`
    - `tests/unit/sandboxTerritoryEngine.rules.test.ts`
    - `tests/unit/Seed14Move35LineParity.test.ts`
  - **Naming convention**: include `.rules.` (e.g. `*.rules.test.ts`) or otherwise mention "rules" explicitly in the filename when the suite is intended to be authoritative for rules semantics.

- **Trace-level tests**  
  Exercise **particular move sequences** (traces) through one or more engines. These are **smoke tests and regression nets**, not the primary source of truth for rules.
  - Typically use `GameTrace` helpers from `tests/utils/traces.ts` to:
    - Generate sandbox AI traces (`runSandboxAITrace`).
    - Replay them on backend or sandbox (`replayTraceOnBackend`, `replayTraceOnSandbox`).
  - Examples:
    - `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts` (archived trace‑parity harness; diagnostic only)
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

Recorded traces are **derived artifacts**, not ground truth. The canonical rules SSoT is:

- The written rules in `RULES_CANONICAL_SPEC.md` (together with `../docs/rules/COMPLETE_RULES.md` / `../docs/rules/COMPACT_RULES.md`).

The shared TS engine implementation in `src/shared/engine/` (types, validators, mutators, `GameEngine`) is the
**primary executable derivation** of that spec. When there is a disagreement between the written rules and the
shared engine, the rules spec wins and the engine must be updated to match.

When a trace-based parity test fails:

1. **Check rules-level tests first.**
   If suites like `RefactoredEngine.test.ts`, `LineDetectionParity.rules.test.ts`, `Seed14Move35LineParity.test.ts`, and related rules‑level tests are green, treat the shared engine semantics as authoritative.

2. **Determine whether the trace is stale.**
   Many historic traces were recorded before rules fixes (e.g. line detection, territory, elimination). If the trace expects behaviour that now violates the canonical rules, the trace itself is outdated.

3. **Apply the seed‑14 pattern:**
   - Example: historical **seed 14** sandbox trace in the (now archived) `Backend_vs_Sandbox.traceParity.test.ts` harness contained a `process_line` move at step 35.
   - After fixing line detection and reconciling backend/sandbox detectors with `../docs/rules/COMPLETE_RULES.md` Section 11.1, the shared engine and detectors both agree that **no valid lines exist** in that state.
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
[`./TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md:257)):

1. **Update shared helpers under `src/shared/engine/*`.**
   - Identify the relevant module(s) for the rule you are changing (movement/capture, lines, territory, placement, victory, or turn sequencing).
   - For **line decision phases** (rewards, collapse‑all vs minimal‑collapse, when eliminations are granted), change [`lineDecisionHelpers.ts`](../src/shared/engine/lineDecisionHelpers.ts:1) and, where necessary, its supporting geometry/validator/mutator modules.
   - For **territory decision phases** (Q23 eligibility, region selection, self‑elimination), change [`territoryDecisionHelpers.ts`](../src/shared/engine/territoryDecisionHelpers.ts:1) and, where necessary, its supporting detection/borders/processing modules.
   - For other behaviour (movement, placement, victory, turn sequencing), change the corresponding shared helpers so backend and sandbox can both reuse them.

2. **Extend shared-helper rules tests.**
   - Add or update tests in the `*.shared.test.ts` and `*.rules.test.ts` suites listed in the
     “Shared-helper rules tests” section above, with special attention to:
     - [`movement.shared.test.ts`](./unit/movement.shared.test.ts:1) and [`captureLogic.shared.test.ts`](./unit/captureLogic.shared.test.ts:1) for movement/capture semantics and no‑dead‑placement.
     - [`lineDecisionHelpers.shared.test.ts`](./unit/lineDecisionHelpers.shared.test.ts:1) for line decision semantics.
     - [`territoryDecisionHelpers.shared.test.ts`](./unit/territoryDecisionHelpers.shared.test.ts:1) and [`territoryProcessing.shared.test.ts`](./unit/territoryProcessing.shared.test.ts:1) for territory decision/processing semantics.
     - [`victory.shared.test.ts`](./unit/victory.shared.test.ts:1) for victory/termination semantics.
   - Keep these suites green; they are the semantic authority.

3. **Run and, if needed, extend parity suites.**
   - Ensure backend vs sandbox vs shared parity suites still pass (for example
     [`archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`](../archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts:1) (archived diagnostic harness),
     [`../parity/Backend_vs_Sandbox.eliminationTrace.test.ts`](./parity/Backend_vs_Sandbox.eliminationTrace.test.ts:1),
     [`../parity/Backend_vs_Sandbox.seed5.internalStateParity.test.ts`](./parity/Backend_vs_Sandbox.seed5.internalStateParity.test.ts:1),
     [`../parity/Backend_vs_Sandbox.seed5.checkpoints.test.ts`](./parity/Backend_vs_Sandbox.seed5.checkpoints.test.ts:1),
     [`../archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`](../archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts:1),
     [`../archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`](../archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts:1),
     [`archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`](../archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1),
     [`TerritoryCore.GameEngine_vs_Sandbox.test.ts`](./unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts:1),
     [`TerritoryBorders.Backend_vs_Sandbox.test.ts`](./unit/TerritoryBorders.Backend_vs_Sandbox.test.ts:1),
     [`movementReachabilityParity.test.ts`](./unit/movementReachabilityParity.test.ts:1),
     and [`TraceFixtures.sharedEngineParity.test.ts`](./unit/TraceFixtures.sharedEngineParity.test.ts:1)).
   - If a parity suite fails but rules‑level tests (including the decision‑helper suites) are green, treat the failing trace/fixture as **stale** and update or regenerate it rather than changing the shared helpers to match old traces.

4. **Add or extend scenario tests.**
   - Encode representative positions and turns in RulesMatrix or FAQ suites to make the new/changed behaviour concrete (see
     [`RULES_SCENARIO_MATRIX.md`](../docs/rules/RULES_SCENARIO_MATRIX.md:1) and the FAQ suites under
     [`tests/scenarios/FAQ_*.test.ts`](./scenarios/FAQ_Q09_Q14.test.ts:1)).
   - For line and territory changes, make sure there is at least one RulesMatrix and/or FAQ scenario that exercises the updated decision behaviour.

5. **Run the parity plan.**
   - For larger changes, follow the plan in
     [`./TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md:56):
     - Run shared-helper rules tests (including the decision‑helper suites).
     - Run targeted parity suites.
     - Optionally run heavier AI/trace harnesses (e.g. `Sandbox_vs_Backend.aiRngFullParity`, sandbox AI simulations) in diagnostic mode; keep these explicitly documented as diagnostic rather than canonical.

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

- `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts` (archived diagnostic harness)
- `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`
- `archive/tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts` (archived diagnostic harness)

**Debug/diagnostic environment variables:**

- `RINGRIFT_TRACE_DEBUG`
  - When set to `1`/`true`, `runSandboxAITrace` and `replayTraceOnBackend` emit structured JSON diagnostics via `logAiDiagnostic` to `logs/ai/trace-parity.log`.
  - Currently logs the sandbox trace opening sequence (initial S/hash + first few history entries) and backend move-mismatch snapshots (sandbox move, backend valid moves, S/hash, per-player counters).
- `RINGRIFT_AI_DEBUG`
  - When set to `1`/`true`, AI-heavy suites mirror detailed diagnostics to the console in addition to writing `logs/ai/*.log`.
  - Also enables extra sandbox AI debug logs inside [`ClientSandboxEngine`](../src/client/sandbox/ClientSandboxEngine.ts) for “no landingCandidates” and “no-op movement” situations.
- `RINGRIFT_SANDBOX_AI_TRACE_MODE`
  - When set to `1`/`true`, parity-focused tests construct `ClientSandboxEngine` instances with a `traceMode` option enabled.
  - In trace mode, sandbox AI still uses the full proportional policy (`chooseLocalMoveFromCandidates`), but the sandbox engine:
    - Applies moves exclusively via canonical `Move` shapes (`place_ring`, `skip_placement`, `move_stack`/`move_stack`, `overtaking_capture`, `continue_capture_segment`).
    - Records history entries so that backend `GameEngine` can replay the same canonical move list in lockstep for trace parity.
    - Aligns chain-capture phase transitions and continuation semantics with the backend (`chain_capture` phase, explicit `continue_capture_segment` moves).

Trace mode is wired through the trace utilities in `tests/utils/traces.ts` (see `runSandboxAITrace`, `replayTraceOnBackend`, and `replayTraceOnSandbox`), and is only used by tests – normal `/sandbox` gameplay does **not** enable it by default.

**Sandbox AI determinism (capture selection):**

- In the sandbox movement phase, when multiple overtaking capture segments are available from the same stack, [`ClientSandboxEngine.maybeRunAITurn()`](../src/client/sandbox/ClientSandboxEngine.ts:277) chooses the segment whose `landing` position is **lexicographically smallest** by `(x, y, z)`.
- This matches the deterministic `capture_direction` test handler in [`ClientSandboxEngine.aiMovementCaptures.test.ts`](./unit/ClientSandboxEngine.aiMovementCaptures.test.ts), which also selects the lexicographically smallest `landingPosition`. This keeps sandbox AI traces reproducible across runs and aligned with backend/sandbox parity tooling.

To debug a tricky AI/parity failure locally:

1. Set `RINGRIFT_TRACE_DEBUG=1 RINGRIFT_AI_DEBUG=1` in your test environment.
2. Re-run the relevant trace/parity test (for example `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`, which is an archived diagnostic harness).
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

## Backend AI-style simulations (diagnostic)

For backend-focused termination and S-invariant diagnostics, there is a
companion AI-vs-AI harness under `tests/unit/GameEngine.aiSimulation.test.ts`.
Like the sandbox fuzz harness, it is intended as a **diagnostic** tool rather
than a default CI gate and is disabled unless explicitly opted in:

```bash
RINGRIFT_ENABLE_BACKEND_AI_SIM=1 npm run test:ai-backend:quiet
```

The backend harness:

- Drives `GameEngine` (using the orchestrator adapter when enabled) via
  `getValidMoves` and `makeMove`, across multiple board types and player
  counts with a deterministic PRNG.
- Enforces global non-decreasing S (`S = markers + collapsed + eliminated`)
  over the lifetime of each game, mirroring the rules-level invariant from
  `computeProgressSnapshot`.
- Detects stalls and non-terminating games via a combination of:
  - per-game move caps,
  - a stagnant-state detector (no state change for multiple consecutive moves),
  - and time-based guards to prevent runaway simulations.

When chasing tricky termination or S-invariant regressions in backend hosts,
run this harness alongside the sandbox AI simulation diagnostics and the
orchestrator S-invariant regression suite.

## Sandbox AI stall diagnostics (engine parity and repro)

> **Historical harness (diagnostic, superseded):**
>
> Older documentation and debug reports referred to a dedicated stall‑repro suite
> `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts` and a browser‑driven
> `/sandbox` stall watchdog. That harness is now treated as **historical
> debugging infrastructure**:
>
> - The modern, reproducible plateau/stall diagnostics are the suites listed in
>   the previous section:
>   - `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`
>   - `tests/utils/aiSeedSnapshots.ts`
>   - `tests/unit/ClientSandboxEngine.aiStallRegression.test.ts`
>   - `tests/scenarios/AI_TerminationFromSeed1Plateau.test.ts`
>   - `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts`
> - For a narrative of the original stall investigation and the legacy
>   `aiStall.seed1` harness, see `archive/AI_STALL_DEBUG_SUMMARY.md`.
>
> When these historical traces or harnesses disagree with rules‑level suites or
> the shared invariants, treat the **rules SSoTs and modern plateau/stall
> regressions as authoritative**, and update or retire the legacy harnesses as
> needed.

## Scenario Matrix (Rules/FAQ → Jest suites)

This matrix links key sections of `../docs/rules/COMPLETE_RULES.md` and FAQ entries to concrete Jest suites. Existing suites are marked **(existing)**; scenario-focused suites under `tests/scenarios/` are marked **(scenario)**; proposed suites are marked **(planned)**.

> Naming convention for scenario-style tests:
>
> > Use rule/FAQ IDs in the `describe`/`it` names, e.g. `Q15_3_1_180_degree_reversal`.
> > Prefer square8 examples first, then mirror high‑value cases on square19/hex where relevant.

### Turn sequence & forced elimination

- **Section 4 (Turn Sequence)**, **FAQ 15.2 (Flowchart of a Turn)**, **FAQ 24 (Forced elimination when blocked)**
  - (scenario) `tests/unit/GameEngine.turnSequence.scenarios.test.ts` — backend turn-sequence and forced-elimination orchestration tests covering blocked-with-stacks and skip-over-no-material players.

#### Mini rules→tests matrix: Turn sequence, movement & progress (cluster example)

| Rule / FAQ cluster                             | Description                                                                  | Primary tests                                                                                                                                                                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Section 4.x Turn Sequence & Forced Elimination | Turn start/end, blocked-with-stacks behaviour, skipping dead players         | [`tests/unit/GameEngine.turnSequence.scenarios.test.ts`](./unit/GameEngine.turnSequence.scenarios.test.ts:1), [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](./scenarios/ForcedEliminationAndStalemate.test.ts:12) |
| Sections 8.2–8.3 Non‑capture Movement          | Minimum distance ≥ stack height, marker landing, blocked paths               | [`tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`](./unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts:1)                                                                                              |
| Section 13.5 Progress & Termination Invariant  | S-invariant (markers + collapsed + eliminated) under forced elim & stalemate | [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](./scenarios/ForcedEliminationAndStalemate.test.ts:12)                                                                                                               |
| FAQ 15.2 / 24 (Turn flow & forced elimination) | Flowchart of a turn and forced elimination when blocked                      | [`tests/unit/GameEngine.turnSequence.scenarios.test.ts`](./unit/GameEngine.turnSequence.scenarios.test.ts:1), [`tests/scenarios/ForcedEliminationAndStalemate.test.ts`](./scenarios/ForcedEliminationAndStalemate.test.ts:12) |

### Movement, minimum distance, and markers

- **Sections 8.2–8.3 (Minimum Distance, Marker Interaction)**, **FAQ 2–3**
  - (existing) `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`
  - (planned) `tests/unit/RuleEngine.movement.scenarios.test.ts` — explicit distance + landing cases for square8, square19, hex

### Chain captures & capture patterns

- **Sections 9–10 (Overtaking & Chain Overtaking)**, **FAQ 5–6, 9, 12, 14**, **FAQ 15.3.1–15.3.2 (180° reversal, cyclic)**
  - (existing) `tests/unit/GameEngine.chainCapture.test.ts` — core chain engine behaviour, 180° reversal, marker interactions, termination rules
  - (existing) `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts` — backend chain-capture geometry enumeration + CaptureDirectionChoice integration for the orthogonal multi-branch scenario (Rust player-choice parity)
  - (existing) `tests/unit/ClientSandboxEngine.chainCapture.test.ts` — sandbox parity for the core two-step chain and the orthogonal multi-branch capture_direction PlayerChoice scenario
  - (scenario) `tests/scenarios/ComplexChainCaptures.test.ts` — end-to-end backend chain-capture examples for FAQ 15.3.1 (180° reversal) and 15.3.2 (cyclic pattern), plus multi-step chains with direction changes on `square8`.

### Line formation & graduated rewards

- **Section 11 (Line Formation & Collapse)**, **FAQ 7, 22**, **Sections 16.5/16.9.4.3**
  - (existing) `tests/unit/lineDecisionHelpers.shared.test.ts` — canonical line‑decision enumeration and application over the shared helpers (`process_line`, `choose_line_option`, and when eliminations are granted).
  - (existing) `tests/unit/ClientSandboxEngine.lines.test.ts` — sandbox line processing driven by the shared helpers.
  - (existing) `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts` — backend + AI choice for line rewards.
  - (existing) `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts` — backend + WebSocket choice flow.
  - (existing) `tests/unit/GameEngine.lines.scenarios.test.ts` — backend line semantics aligned with Section 11 and FAQ Q7/Q22 on `square8`, using the same shared decision helpers.

### Territory disconnection & chain reactions

- **Section 12 (Area Disconnection & Collapse)**, **FAQ 10, 15, 20, 23**, **Sections 16.9.4.4, 16.9.6–16.9.8**
  - (existing) `tests/unit/territoryDecisionHelpers.shared.test.ts` — canonical `choose_territory_option` / `eliminate_rings_from_stack` decision semantics (Q23, self‑elimination bookkeeping) over the shared helpers.
  - (existing) `tests/unit/BoardManager.territoryDisconnection.test.ts` / `.square8.test.ts` / `.hex.test.ts` — region detection & adjacency for square19, square8, and hex.
  - (existing) `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` / `.hex.test.ts` — engine‑level region processing order and interaction with the shared helpers.
  - (existing) `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` / `.hex.test.ts` — sandbox parity.
  - (existing) `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts` — region‑order PlayerChoice in sandbox.
  - (existing) `tests/unit/GameEngine.territory.scenarios.test.ts` — explicit self‑elimination prerequisite and multi‑region chain reactions mapped to Q15, Q20, Q23.
  - (existing, diagnostic; `TerritoryParity` archived) `archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`, `tests/unit/TerritoryDecisions.GameEngine_vs_Sandbox.test.ts`, and related seed‑based parity suites — backend↔sandbox Q23 parity on larger boards and specific seeds (**diagnostic; canonical semantics live in the shared decision‑helper tests and RulesMatrix/FAQ territory suites such as `RulesMatrix.Territory.MiniRegion.test.ts` and `FAQ_Q22_Q23.test.ts`**).
  - (existing) `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` — advanced-phase backend↔sandbox parity harness for capture chains, single‑region line→territory, and mixed line+multi‑region territory flows (including Q20/Q23‑style two‑region scenarios on square8/19/hex).

### Victory conditions & stalemate

- **Section 13 (Victory Conditions)**, **FAQ 11, 18, 21, 24**, **Sections 16.6, 16.9.4.5**
  - (existing) `tests/unit/ClientSandboxEngine.victory.test.ts` — sandbox ring‑elimination & territory victories
  - (planned) `tests/unit/GameEngine.victory.scenarios.test.ts` — ring‑elimination, territory‑majority, last‑player‑standing, stalemate tiebreaker examples

### Connection lifecycle & reconnection

- **Lifecycle & reconnection window semantics**, **../docs/architecture/CANONICAL_ENGINE_API.md §3.9.4**, **RULES_SCENARIO_MATRIX LF1**
  - (existing) `tests/unit/GameSession.reconnectFlow.test.ts` — server-side reconnection windows, GameSession preservation, and abandonment hooks.
  - (existing) `tests/unit/GameSession.reconnectDuringDecision.test.ts` — reconnect during an in‑flight PlayerChoice / decision phase.
  - (existing) `tests/unit/GameConnection.reconnection.test.ts` — client `SocketGameConnection` status transitions and reconnect attempts.
  - (existing) `tests/integration/GameReconnection.test.ts` — WebSocket `player_disconnected` / `player_reconnected` and reconnection-window semantics at the API edge.
  - (existing) `tests/integration/LobbyRealtime.test.ts` — lobby `lobby:subscribe`/`lobby:game_*` updates under join/leave/reconnect flows.
  - (existing) `tests/e2e/reconnection.simulation.test.ts` — network partition and reconnection-window expiry (rated vs unrated abandonment) plus HUD‑level reconnect UX.

### PlayerChoice flows (engine + transport + sandbox)

- **Sections 4.5, 10.3, 11–12 (Lines, Territory, Chain choices)**, **FAQ 7, 15, 22–23**
  - (existing) `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`
  - (existing) `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`
  - (existing) `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`
  - (existing) `tests/unit/GameEngine.captureDirectionChoice.test.ts` / `.captureDirectionChoiceWebSocketIntegration.test.ts` — helper-level capture_direction logic plus a full orthogonal chain capture scenario driven end-to-end over WebSockets
  - (existing) `tests/unit/PlayerInteractionManager.test.ts`
  - (existing) `tests/unit/WebSocketInteractionHandler.test.ts`
  - (existing) `tests/unit/AIInteractionHandler.test.ts`
  - (existing) `tests/unit/WebSocketServer.sessionTermination.test.ts` — WebSocket session termination and decision‑phase cancellation semantics, including AI‑backed and human‑backed PlayerChoice flows (line rewards, ring elimination, region order) under `terminateUserSessions`.
  - (planned) Additional focused scenarios in the above suites to ensure each choice type has at least one rule/FAQ‑tagged test name.

---

## Rules/FAQ → Scenario Matrix

For a rule-centric view of test coverage, see:

- `RULES_SCENARIO_MATRIX.md` – a living matrix mapping sections of `../docs/rules/COMPLETE_RULES.md` and the FAQ to concrete Jest suites (backend engine, sandbox engine, WebSocket/choice flows, and AI boundary tests).

When you add or modify scenario-style tests, update that matrix so it remains the single source of truth for how rules map to executable tests.

## FAQ Scenario Tests

Each FAQ question from [`../docs/rules/COMPLETE_RULES.md`](../docs/rules/COMPLETE_RULES.md:1) has dedicated test coverage in scenario-style test files under `tests/scenarios/FAQ*\*.test.ts`.

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

| FAQ Questions | Test File                                                                          | Topics Covered                                                            |
| ------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Q1-Q6         | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](./scenarios/FAQ_Q01_Q06.test.ts:1)         | Stack order, minimum distance, capture landing, overtaking vs elimination |
| Q7-Q8         | [`tests/scenarios/FAQ_Q07_Q08.test.ts`](./scenarios/FAQ_Q07_Q08.test.ts:1)         | Line formation, exact vs overlength lines, no rings to eliminate          |
| Q9-Q14        | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](./scenarios/FAQ_Q09_Q14.test.ts:1)         | Chain blocking, multicolored stacks, Moore vs Von Neumann adjacency       |
| Q15           | [`tests/scenarios/FAQ_Q15.test.ts`](./scenarios/FAQ_Q15.test.ts:1)                 | 180° reversal, cyclic patterns, mandatory chain continuation              |
| Q16-Q18       | [`tests/scenarios/FAQ_Q16_Q18.test.ts`](./scenarios/FAQ_Q16_Q18.test.ts:1)         | Control transfer, first placement, multiple victory conditions            |
| Q19-Q21, Q24  | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](./scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | Player count variations, thresholds, forced elimination, stalemate        |
| Q22-Q23       | [`tests/scenarios/FAQ_Q22_Q23.test.ts`](./scenarios/FAQ_Q22_Q23.test.ts:1)         | Graduated line rewards, territory self-elimination prerequisite           |

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
  [`../docs/rules/COMPLETE_RULES.md`](../docs/rules/COMPLETE_RULES.md:1), look for a
  matching scenario in `tests/scenarios/RulesMatrix.*.test.ts` or
  `tests/scenarios/FAQ*\*.test.ts` and treat that as the authoritative example.

For the complete FAQ → test mapping, see [`RULES_SCENARIO_MATRIX.md`](../docs/rules/RULES_SCENARIO_MATRIX.md:1) Section 9.

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

> **Historical Python diagnostics (archived):**
>
> - `ai-service/tests/archive/archived_test_determinism.py` – legacy determinism helper, now archived and superseded by `test_engine_determinism.py` and `test_no_random_in_rules_core.py`; kept only as a non‑canonical diagnostic reference.
> - `ai-service/tests/archive/archived_test_rules_parity.py` – historical TS↔Python rules‑parity harness, archived in favour of the fixture‑driven suites under `ai-service/tests/parity/*`; useful for archaeology, but not a CI gate or semantic authority.

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

- Human clicks go through `ClientSandboxEngine.handleHumanCellClick`, whose movement path now calls `ClientSandboxEngine.handleMovementClick` / `handleLegacyMovementClick`, layering sandbox‑specific history over shared Movement/Capture aggregates and the orchestrator adapter.
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

**Last Updated**: November 24, 2025
**Framework**: Jest 29.7.0 + ts-jest 29.1.1
