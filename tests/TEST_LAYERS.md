# RingRift Test Layering Strategy

> **Doc Status (2025-11-27): Active (test meta-doc, non-semantics)**
>
> **Role:** Define a clear test layering strategy (unit → contract/scenario → integration → E2E) so that suites are classified consistently, redundancy is minimized, and CI profiles remain predictable. This doc is a **test/meta reference only** – it describes how tests are organised and which layers should run where; it does **not** define game rules or lifecycle semantics.
>
> **Not a semantics SSoT:** Canonical rules and lifecycle semantics are owned by the shared TypeScript rules engine and contracts/vectors (`src/shared/engine/**`, `src/shared/engine/contracts/**`, `tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`) together with the written rules and lifecycle docs (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `docs/CANONICAL_ENGINE_API.md`). This file should always defer to those SSoTs when describing “what is correct”; it only describes **how we test** that correctness.
>
> **Related docs:** `tests/README.md`, `tests/TEST_SUITE_PARITY_PLAN.md`, `docs/PARITY_SEED_TRIAGE.md`, `RULES_SCENARIO_MATRIX.md`, `RULES_ENGINE_ARCHITECTURE.md`, and `DOCUMENTATION_INDEX.md`.

> **Purpose:** Define a clear test layering strategy to minimize redundancy, improve iteration speed, and ensure each layer has a specific purpose.

## Test Layer Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: E2E Tests (Playwright)                                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Full browser-based user journeys                                      │
│ Location: tests/e2e/                                                         │
│ Run: Before release, after major changes                                     │
│ Speed: Slow (~minutes)                                                       │
│ Count: 2 test files                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Integration Tests                                                   │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Service interactions, AI service, WebSocket flows                     │
│ Location: tests/integration/, tests/unit/*Integration*.test.ts               │
│ Run: CI pipeline, major feature changes                                      │
│ Speed: Medium (~30-60 seconds)                                               │
│ Count: ~10 test files                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Contract/Scenario Tests                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Cross-language parity, canonical engine behavior                      │
│ Location: tests/contracts/, tests/scenarios/, fixtures/contract-vectors/     │
│ Run: Every commit, after rules changes                                       │
│ Speed: Fast-Medium (~10-30 seconds)                                          │
│ Count: ~20 test files + vector bundles                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Unit Tests                                                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Individual module behavior, shared engine functions                   │
│ Location: tests/unit/*.shared.test.ts, tests/unit/*Engine*.test.ts           │
│ Run: Every commit, during development                                        │
│ Speed: Fast (~5-15 seconds)                                                  │
│ Count: ~60 test files                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Unit Tests (Fast, Per-Module)

### Purpose

Test individual functions, modules, and components in isolation.

### Characteristics

- **Speed:** < 100ms per test
- **Dependencies:** Mocked or minimal
- **Determinism:** 100% deterministic
- **Granularity:** Single function or module

### Key Test Categories

#### Shared Engine Core (`*.shared.test.ts`)

| File                                      | Purpose                       | Keep/Review |
| ----------------------------------------- | ----------------------------- | ----------- |
| `movement.shared.test.ts`                 | Movement validation logic     | ✅ Keep     |
| `captureLogic.shared.test.ts`             | Capture chain logic           | ✅ Keep     |
| `lineDetection.shared.test.ts`            | Line detection algorithms     | ✅ Keep     |
| `territoryBorders.shared.test.ts`         | Territory border calculations | ✅ Keep     |
| `territoryDecisionHelpers.shared.test.ts` | Territory decision helpers    | ✅ Keep     |
| `lineDecisionHelpers.shared.test.ts`      | Line decision helpers         | ✅ Keep     |
| `victory.shared.test.ts`                  | Victory condition logic       | ✅ Keep     |
| `heuristicParity.shared.test.ts`          | Heuristic evaluation parity   | ✅ Keep     |

#### Server Components

| File                         | Purpose               | Keep/Review |
| ---------------------------- | --------------------- | ----------- |
| `auth.routes.test.ts`        | Authentication routes | ✅ Keep     |
| `rateLimiter.test.ts`        | Rate limiting         | ✅ Keep     |
| `securityHeaders.test.ts`    | Security middleware   | ✅ Keep     |
| `logger.test.ts`             | Logging utilities     | ✅ Keep     |
| `validation.schemas.test.ts` | Input validation      | ✅ Keep     |
| `MetricsService.test.ts`     | Prometheus metrics    | ✅ Keep     |

#### Infrastructure

| File                          | Purpose                  | Keep/Review                                                                                                 |
| ----------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `envFlags.test.ts`            | Environment flag parsing | ✅ Keep                                                                                                     |
| `notation.test.ts`            | Move notation parsing    | ✅ Keep                                                                                                     |
| `NoRandomInCoreRules.test.ts` | Determinism guard        | ✅ Keep                                                                                                     |
| `RNGDeterminism.test.ts`      | RNG consistency          | 🕒 Historical (removed; superseded by `EngineDeterminism.shared.test.ts` and `NoRandomInCoreRules.test.ts`) |

---

## Layer 2: Contract/Scenario Tests (Canonical Engine)

### Purpose

Validate that the **canonical shared TS engine** (helpers + aggregates + orchestrator under `src/shared/engine/`) behaves as specified by the rules docs, and that other engines (backend hosts, sandbox adapters, Python rules engine) match it via **contract vectors and shared fixtures**.

### Characteristics

- **Speed:** < 500ms per vector
- **Dependencies:** Shared TS engine, contract schemas, vector fixtures, Python parity runners
- **Determinism:** 100% deterministic
- **Granularity:** State transitions, move validation, domain‑aggregate behaviour

### Contract Test Vectors (`tests/fixtures/contract-vectors/v2/`)

| File                          | Category             | Vector Count |
| ----------------------------- | -------------------- | ------------ |
| `placement.vectors.json`      | Placement moves      | 10+          |
| `movement.vectors.json`       | Movement/capture     | 15+          |
| `capture.vectors.json`        | Capture chains       | 10+          |
| `line_detection.vectors.json` | Line detection       | 10+          |
| `territory.vectors.json`      | Territory processing | 10+          |

### Contract Runner

- **File:** `tests/contracts/contractVectorRunner.test.ts`
- **Role:** Single source of truth for cross-language parity
- **Python counterpart:** `ai-service/tests/contracts/test_contract_vectors.py`

### Scenario Tests (`tests/scenarios/`)

| Category     | Files                                                                   | Purpose                         |
| ------------ | ----------------------------------------------------------------------- | ------------------------------- |
| Rules Matrix | `RulesMatrix.*.test.ts`                                                 | Comprehensive rules validation  |
| FAQ Tests    | `FAQ_Q*.test.ts`                                                        | User-facing rule clarifications |
| Edge Cases   | `ComplexChainCaptures.test.ts`, `ForcedEliminationAndStalemate.test.ts` | Complex scenarios               |

---

## Layer 3: Parity/Integration Tests

### Purpose

Verify that hosts (backend, sandbox, Python) behave identically and that external service integrations work correctly.

### Parity Tests (Review for Consolidation)

#### Backend vs Sandbox Parity

| File                                     | Purpose               | Recommendation                      |
| ---------------------------------------- | --------------------- | ----------------------------------- |
| `Backend_vs_Sandbox.traceParity.test.ts` | Trace-level parity    | ⚠️ Consolidate with contract tests  |
| `Backend_vs_Sandbox.seed*.test.ts`       | Seed-specific parity  | ⚠️ Move to historical failures only |
| `*Parity*.test.ts` (10+ files)           | Various parity checks | ⚠️ Promote to contract vectors      |

#### Python vs TS Parity

| File                               | Purpose               | Recommendation                      |
| ---------------------------------- | --------------------- | ----------------------------------- |
| `Python_vs_TS.traceParity.test.ts` | Cross-language parity | ✅ Keep, supplement with contracts  |
| `ai-service/tests/parity/*.py`     | Python parity tests   | ✅ Keep, use contracts as authority |

### Integration Tests (`tests/integration/`)

| File                                         | Purpose                     | Keep/Review                 |
| -------------------------------------------- | --------------------------- | --------------------------- |
| `AIGameCreation.test.ts`                     | AI game creation flow       | ✅ Keep                     |
| `AIResilience.test.ts`                       | AI service failure handling | ✅ Keep                     |
| `FullGameFlow.test.ts`                       | Complete game lifecycle     | ✅ Keep                     |
| `GameReconnection.test.ts`                   | WebSocket reconnection      | ✅ Keep                     |
| `LobbyRealtime.test.ts`                      | Lobby real-time updates     | ✅ Keep                     |
| `PythonRulesClient.live.integration.test.ts` | Live Python service         | ⚠️ Requires running service |

---

## Layer 4: E2E Tests (Playwright)

### Purpose

Validate complete user journeys in a real browser environment.

### Test Files (`tests/e2e/`)

| File                    | Purpose             |
| ----------------------- | ------------------- |
| `auth.e2e.spec.ts`      | Authentication flow |
| `game-flow.e2e.spec.ts` | Game play flow      |

### When to Run

- Before releases
- After UI changes
- After major backend changes

---

## Test Categories by Host

### Shared Engine Tests (Authoritative)

```
tests/unit/*.shared.test.ts          # Core logic tests
tests/unit/GameEngine.movement.shared.test.ts             # Backend movement wired to MovementAggregate
tests/unit/ClientSandboxEngine.movementParity.shared.test.ts  # Sandbox movement parity vs MovementAggregate
tests/unit/ClientSandboxEngine.placement.shared.test.ts   # Sandbox placement/skip parity vs PlacementAggregate
tests/unit/RuleEngine.skipPlacement.shared.test.ts        # RuleEngine skip_placement parity vs PlacementAggregate
tests/contracts/                      # Contract vectors
tests/fixtures/contract-vectors/      # Test vector data
```

### Backend Tests

```
tests/unit/GameEngine.*.test.ts       # Backend game engine
tests/unit/RuleEngine.*.test.ts       # Legacy rule engine
tests/unit/BoardManager.*.test.ts     # Board management
tests/integration/                    # Service integration
```

### Sandbox Tests

```
tests/unit/ClientSandboxEngine.*.test.ts   # Client sandbox
tests/unit/sandboxTerritory*.test.ts       # Territory logic
```

### Python Tests

```
ai-service/tests/contracts/           # Contract validation
ai-service/tests/parity/              # TS parity tests
ai-service/tests/rules/               # Python rules tests
ai-service/tests/invariants/          # Invariant tests
```

---

## Redundancy Analysis & Consolidation Candidates

### High Priority for Consolidation

#### Parity Tests → Contract Vectors

Many parity and snapshot parity tests can be converted to or backed by contract vectors:

| Parity Test Suite                                 | Contract Vector / Anchor                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `Backend_vs_Sandbox.traceParity.test.ts`          | Backed by `movement.vectors.json`, `capture.vectors.json`, `line_detection.vectors.json`, `territory.vectors.json` |
| `Backend_vs_Sandbox.eliminationTrace.test.ts`     | Backed by elimination/territory vectors and victory semantics (add dedicated victory vectors over time)            |
| `Backend_vs_Sandbox.seed*.snapshotParity.test.ts` | Should be derivable from and consistent with vector-backed traces                                                  |
| `TerritoryParity.GameEngine_vs_Sandbox.test.ts`   | → `territory.vectors.json`                                                                                         |
| `TerritoryCore.GameEngine_vs_Sandbox.test.ts`     | → `territory.vectors.json`                                                                                         |
| `TraceFixtures.sharedEngineParity.test.ts`        | → `movement.vectors.json`, `capture.vectors.json`, `line_detection.vectors.json`, `territory.vectors.json`         |

#### Seed-Specific Tests → Historical Failures

Keep seed-specific tests only for documented historical failures:

| File                                  | Status    | Action                       |
| ------------------------------------- | --------- | ---------------------------- |
| `Seed14Move35LineParity.test.ts`      | ✅ Keep   | Documents seed-14 resolution |
| `Backend_vs_Sandbox.seed5.*.test.ts`  | ⚠️ Review | Consolidate or document      |
| `Backend_vs_Sandbox.seed17.*.test.ts` | ⚠️ Review | Consolidate or document      |
| `Backend_vs_Sandbox.seed18.*.test.ts` | ⚠️ Review | Consolidate or document      |

### Medium Priority

#### Engine-Specific Tests

Some tests duplicate behavior across engines:

| GameEngine Test                             | Sandbox / Cross-Host Test                                                                                                               | Recommendation                                                                                                                                                                    |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GameEngine.lines.scenarios.test.ts`        | `ClientSandboxEngine.lines.test.ts`                                                                                                     | Promote shared cases to contracts                                                                                                                                                 |
| `GameEngine.territoryDisconnection.test.ts` | `ClientSandboxEngine.territoryDisconnection.test.ts`                                                                                    | Same                                                                                                                                                                              |
| `GameEngine.victory.*.test.ts`              | `GameEngine.victory.LPS.scenarios.test.ts`, `ClientSandboxEngine.victory.LPS.crossInteraction.test.ts`, `RulesMatrix.Victory.*.test.ts` | Same high-level semantics; prefer shared `victory.shared.test.ts` + RulesMatrix victory suites as the semantic anchor, and treat engine-/sandbox-specific suites as adapter views |

---

## CI Pipeline Configuration

> **Orchestrator profiles:** Unless explicitly noted otherwise, these commands are assumed to run with `ORCHESTRATOR_ADAPTER_ENABLED=true` and `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`, treating the orchestrator adapter as the default rules path. Legacy/SHADOW profiles (with the adapter disabled or `EngineSelection.LEGACY`/`SHADOW` forced by configuration) should be reserved for targeted parity/regression jobs that focus on legacy behaviour and `ShadowModeComparator` metrics.
>
> **Semantic gates vs diagnostics:**
>
> - **Semantic gates in CI:** `*.shared.test.ts` suites, contract‑vector tests (`tests/contracts/**` + `tests/fixtures/contract-vectors/v2/**`), and RulesMatrix/FAQ scenario suites (`tests/scenarios/RulesMatrix.*.test.ts`, `tests/scenarios/FAQ_*.test.ts`) are the primary rules **authorities** and should be treated as hard gates in CI.
> - **Diagnostic / legacy suites:** Seeded trace parity, backend↔sandbox parity suites (`Backend_vs_Sandbox.*`, `TerritoryParity.*`, `Sandbox_vs_Backend.*`), and historical/seed‑specific tests are **diagnostic nets**. They may run in separate jobs or be skipped/env‑gated; when they disagree with the semantic gates, the `.shared` + contracts + RulesMatrix suites win.

### Fast Feedback (< 2 min)

```bash
# Layer 1: Unit tests
npm run test:unit -- --testPathPattern="\.shared\.test\."

# Layer 2: Contract tests
npm run test -- --testPathPattern="contracts"
```

### Standard CI (< 5 min)

```bash
# Layers 1-2 + key scenarios
npm run test -- --testPathIgnorePatterns="e2e|integration|\.debug\.|\.seed\d+\."
```

### Full CI (< 15 min)

```bash
# All layers except E2E
npm run test

# Python contracts
cd ai-service && pytest tests/contracts/
```

### Release Gate

```bash
# All tests including E2E
npm run test
npm run test:e2e

# Python full suite
cd ai-service && pytest
```

---

## Migration Path: Adapters & Legacy Elimination

With the orchestrator adapters (`TurnEngineAdapter.ts`, `SandboxOrchestratorAdapter.ts`) now in place:

### Phase A: Enable Adapters (Current)

1. Wire feature flag `ORCHESTRATOR_ADAPTER_ENABLED`
2. Run all tests with flag disabled (baseline)
3. Run all tests with flag enabled (parity check)
4. Enable in staging/canary

### Phase B: Trim Redundant Parity Tests

Once adapters are stable:

1. Remove seed-specific parity tests (keep only historical failure docs)
2. Consolidate engine-specific tests to contract vectors
3. Remove legacy trace parity tests

### Phase C: Remove Legacy Code

After full adapter migration:

1. Remove legacy sandbox turn engines
2. Update tests to use adapters directly
3. Simplify test suite by ~30%

---

## Related Documents

- [`tests/README.md`](./README.md) - How to run tests
- [`tests/TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md) - Detailed test classification
- [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md) - Rules-to-test mapping
- [`docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md`](../docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md) - Legacy removal plan
