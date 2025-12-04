# RingRift Architecture Assessment & Roadmap

**Assessment Date:** November 21, 2025
**Last Updated:** December 1, 2025 (Phases 1-4 Complete, PASS20-21)
**Status:** Architecture overview, evaluation, and roadmap
**Scope:** Server, client, shared engine, Python AI service, tests, and infrastructure

> **SSoT alignment:** This document is a derived architectural view over the following canonical sources:
>
> - **Rules semantics SSoT:** Shared TypeScript rules engine under `src/shared/engine/**` (helpers → domain aggregates → turn orchestrator → contracts) plus v2 contract vectors and runners (`tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`) and rules docs (`RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`).
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas under `src/shared/types/**`, `src/shared/engine/orchestration/types.ts`, and `src/shared/validation/websocketSchemas.ts` for the executable Move/orchestrator/WebSocket lifecycle.
> - **Operational SSoT:** CI workflows (`.github/workflows/*.yml`), Dockerfiles, docker-compose stacks, monitoring configs under `monitoring/**`, and runtime config/env validation code under `src/server/config/**`, `src/shared/utils/envFlags.ts`, and `scripts/validate-deployment-config.ts`.
> - **Precedence:** If this document ever conflicts with those specs, engines, types, or configs, **code + tests win**, and this doc must be updated to match them.
>
> **Doc Status (2025-11-27): Active (with some historical/aspirational content)**  
> Architectural assessment and roadmap for the overall system topology. This document is **not** a rules or lifecycle SSoT; it summarizes and evaluates the architecture built around the canonical rules and lifecycle sources of truth.
>
> Backend (`GameEngine`, `RuleEngine`, `BoardManager`, `TurnEngineAdapter`, `WebSocketInteractionHandler`, `RulesBackendFacade`) and sandbox (`ClientSandboxEngine`, `SandboxOrchestratorAdapter`, `sandbox*` modules) act as **hosts/adapters** over the shared engine. Historical descriptions that treat them as standalone engines should now be read in that light.
> Some sections below still describe future/aspirational work (e.g., observability, client UX separation); these are intentionally kept as roadmap items.
>
> This document consolidates previous assessments (`CODEBASE_EVALUATION.md`, `TECHNICAL_ARCHITECTURE_ANALYSIS.md`) and earlier designs (`REFACTORING_ARCHITECTURE_DESIGN.md`, `ringrift_architecture_plan.md`), updated to reflect the implemented orchestrator + aggregates + contracts stack and the SSoTs indexed in `DOCUMENTATION_INDEX.md`.
>
> **🎉 Remediation Complete (2025-11-26)**: The rules engine consolidation (Phases 1-4) is now complete.
> **🎉 Observability Implemented (2025-12-01, PASS21)**: Grafana dashboards and k6 load testing framework added.
>
> - Canonical turn orchestrator in [`src/shared/engine/orchestration/`](src/shared/engine/orchestration/)
> - Backend host/adapter ([`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts)) and client host/adapter ([`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts))
> - Cross-language contract tests achieving 100% parity (49 test vectors) via shared contracts under `src/shared/engine/contracts/*` and fixtures under `tests/fixtures/contract-vectors/v2/`
> - 3 Grafana dashboards (game-performance, rules-correctness, system-health) with 22 panels
> - k6 load testing framework with 4 production-scale scenarios

---

## 1. Executive Summary

RingRift follows a **TypeScript-first architecture** with a Node.js backend and React frontend, supplemented by a Python AI microservice.

**Overall Architecture Grade: B+**

- ✅ **Strengths:** Strong type system, a **shared canonical rules engine** (helpers + aggregates + orchestrator + contracts) reused across hosts, modern stack (React/Node/Prisma/Redis), and unusually strong rules/architecture documentation.
- ⚠️ **Weaknesses:** Some architecture docs still mix historical designs with current reality (now being normalized), AI integration and observability are still maturing, and frontend UX lacks polish.
- ❌ **Gaps:** Advanced AI tactics (MCTS/NeuralNet) are not yet production-ready, observability is minimal, and some legacy rules hosts (older sandbox paths) remain to be fully retired.

**Verdict:** The current **shared-engine + orchestrator + host/adapters** architecture is appropriate for the current stage. Premature microservice extraction (beyond the existing AI service) is still strongly discouraged; investment should focus on completing the orchestrator rollout, deprecating legacy paths, and improving UX/observability.

---

## 2. Current Architecture Overview

### Backend (Node.js + TypeScript)

- **Rules Hosts & Core Engine:**
  - **Canonical rules:** Shared engine helpers + domain aggregates + orchestrator under `src/shared/engine/` (TS single source of truth for rules semantics).
  - **Backend host/adapter:** `TurnEngineAdapter` wraps the orchestrator for server use, handling AI turns, timeouts, and player interaction while delegating rules semantics to `processTurn` / `processTurnAsync`.
  - **Legacy orchestration plumbing:** `GameEngine` (server-local orchestration wrapper), `RuleEngine` (historical validation surface), and `BoardManager` (server-owned board utilities). These now act primarily as hosts/adapters and plumbing atop the shared engine rather than a separate rules implementation.
- **Session Management:** `GameSessionManager` handles lifecycle and distributed locking; `GameSession` wraps the engine host, interaction handlers, and persistence.
- **Rules Abstraction:** `RulesBackendFacade` mediates between the TS orchestrator and Python service, supporting TS-only, shadow, and Python-authoritative modes.
- **API:** Express.js for Auth/Game/User routes.
- **Real-time:** Socket.IO for game state updates and moves.
- **Persistence:** PostgreSQL (via Prisma) and Redis (caching/locking).

### Frontend (React + TypeScript)

- **Framework:** Vite-based SPA.
- **State:** React Context (`GameContext`) + React Query.
- **Components:** `BoardView` (rendering), `ChoiceDialog` (interaction), `GameHUD`.
- **Sandbox:** `ClientSandboxEngine` runs a full local copy of the game rules for testing/analysis.

### AI Service (Python)

- **Framework:** FastAPI.
- **Role:** Stateless move generation and position evaluation.
- **Integration:** Called via `AIServiceClient` in Node.js.
- **Parity:** Maintains a Python rules engine (`game_engine.py` + `ai-service/app/rules/*`) that mirrors the **shared TS orchestrator + aggregates**. Parity is enforced via contract vectors (`tests/fixtures/contract-vectors/v2/*.json`) and TS/Python contract test runners (`tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`), rather than by duplicating the historical TS validator/mutator tree one-for-one.

### Shared Core (`src/shared/`)

- **Types:** Canonical `GameState`, `Move`, `MoveType`, `GamePhase`, `GameStatus`, `PlayerChoice*`, and related domain types under `src/shared/types/`.
- **Rules Engine:** Shared helpers + domain aggregates + orchestrator under `src/shared/engine/` provide the single canonical TS rules surface.
- **Contracts:** Cross-language contract schemas and serialization under `src/shared/engine/contracts/`.
- **Logic:** Pure functions for geometry, hashing, invariant checking, line/territory/victory decisions, and local AI evaluation used by both client and server and by the Python parity layer.

---

## 3. Codebase Evaluation

### Strengths

1.  **Rules Implementation:** Core mechanics implemented and verified (49 contract vectors passing, comprehensive tests)
2.  **Type Safety:** Shared types prevent drift between frontend, backend, and AI service
3.  **Documentation:** Rules and architecture well-documented and kept in sync
4.  **Infrastructure:** Docker/Compose setup production-ready
5.  **Observability:** 3 Grafana dashboards + k6 load testing framework (PASS21)

### Risks & Technical Debt

1.  ~~**Scenario Coverage:** While unit tests are strong, a systematic matrix of rule scenarios (from the rulebook/FAQ) is missing.~~ ✅ **Resolved:** Scenario matrix and FAQ tests implemented.
2.  ~~**Parity Gaps:** Subtle semantic differences exist between the backend engine and the client sandbox (e.g., in AI trace replay).~~ ✅ **Resolved:** Canonical orchestrator with adapters ensures identical behavior.
3.  **UX Polish:** The UI is functional but developer-centric; missing robust spectator/reconnect flows. _(Still pending)_
4.  ~~**AI Boundary:** The `RulesBackendFacade` is robust, but the Python service itself needs hardening against timeouts and model mismatches.~~ ✅ **Resolved:** Contract test validation replaces runtime shadow contracts.

---

## 4. Future Architecture: Refactoring Design

**Goal:** Transition from a class-based monolithic engine to a modular, functional, and type-safe system.

### Core Principles

1.  **Immutability:** State transitions produce new `GameState` objects.
2.  **Pure Functions:** Logic is separated into `Validators` (check legality) and `Mutators` (apply changes).
3.  **Shared Code:** Core logic resides in `src/shared/engine` and is used by Server, Client, and AI.

### Proposed Structure (Partially Historical / Semantic Boundaries Diagram)

> **Note:** This diagram reflects the full validators/mutators tree as an idealized semantic boundary map. The canonical TS rules surface is the helpers + aggregates + orchestrator + contracts stack under `src/shared/engine/`. See `docs/MODULE_RESPONSIBILITIES.md` and `docs/DOMAIN_AGGREGATE_DESIGN.md` for the current responsibilities layout. Not all of the `validators/*.ts` and `mutators/*.ts` modules listed here exist in the **current** tree; some (for example `MovementValidator.ts`, `CaptureValidator.ts`, `LineValidator.ts`, `TerritoryValidator.ts`, `LineMutator.ts`, `TerritoryMutator.ts`, `TurnMutator.ts`) are historical or conceptual and must **not** be treated as active SSoT modules.

```
src/shared/engine/
├── types.ts                # Core definitions
├── core.ts                 # Pure geometry, hashing, invariants
├── aggregates/             # Domain aggregates (6 total)
│   ├── PlacementAggregate.ts
│   ├── MovementAggregate.ts
│   ├── CaptureAggregate.ts
│   ├── LineAggregate.ts
│   ├── TerritoryAggregate.ts
│   └── VictoryAggregate.ts
├── validators/             # Pure validation logic
│   ├── PlacementValidator.ts
│   ├── MovementValidator.ts
│   ├── CaptureValidator.ts
│   ├── LineValidator.ts
│   └── TerritoryValidator.ts
├── mutators/               # Pure state mutation logic
│   ├── PlacementMutator.ts
│   ├── MovementMutator.ts
│   ├── CaptureMutator.ts
│   ├── LineMutator.ts
│   ├── TerritoryMutator.ts
│   └── TurnMutator.ts
├── orchestration/          # Phase management ✅ NEW
│   ├── turnOrchestrator.ts # processTurn(), processTurnAsync()
│   ├── phaseStateMachine.ts
│   ├── types.ts
│   └── README.md
└── contracts/              # Contract test infrastructure ✅ NEW
    ├── schemas.ts
    ├── serialization.ts
    └── testVectorGenerator.ts
```

### The "Move" Lifecycle (Canonical Orchestrator + Hosts)

Canonical rules semantics are expressed in terms of **`Move`** (from `src/shared/types/game.ts`) and the shared turn orchestrator:

1.  **Input (canonical):**
    - Hosts obtain a `Move` either directly (AI, sandbox tooling) or via a `PendingDecision` → `PlayerChoice` → `Move.id` flow (see `docs/CANONICAL_ENGINE_API.md` and `AI_ARCHITECTURE.md`).
2.  **Orchestration:**
    - `processTurn` / `processTurnAsync` in `turnOrchestrator.ts` drives the game through all relevant phases, delegating to domain aggregates (`PlacementAggregate`, `MovementAggregate`, `CaptureAggregate`, `LineAggregate`, `TerritoryAggregate`, `VictoryAggregate`).
3.  **Validation + Mutation:**
    - Each aggregate uses shared helpers (`core.ts`, domain helpers) to validate the move and apply the resulting state transition, returning a new `GameState`.
4.  **Consequences:**
    - Automatic consequences (lines, Territory, forced elimination, victory) are applied via the same aggregates and helpers, ensuring a single shared semantics surface across backend and sandbox.
5.  **Transition:**
    - The phase state machine (`orchestration/phaseStateMachine.ts`) determines the next phase/player and whether further decisions are required, emitting updated `ProcessTurnResult` and `PendingDecision` values.
6.  **Emit:**
    - Backend and sandbox hosts/adapters translate `ProcessTurnResult` + `PendingDecision` into transport-layer messages (WebSocket payloads, `PlayerChoice` structures) and broadcast updated state to clients.

> **Historical note:** The original lifecycle was described in terms of `GameAction` → Validator → Mutator → `GameEngine`. That path now exists primarily as a compatibility layer; new flows should be expressed in terms of `Move` → orchestrator → aggregates.

### Migration Strategy (Strangler Fig) ✅ COMPLETE

1.  ✅ **Phase 1:** Created canonical turn orchestrator in `src/shared/engine/orchestration/`
2.  ✅ **Phase 2:** Wired orchestrator to all 6 aggregates with contract test vectors
3.  ✅ **Phase 3:** Created adapters: `TurnEngineAdapter.ts` (backend), `SandboxOrchestratorAdapter.ts` (client)
4.  ✅ **Phase 4:** Python contract test runner with 100% parity (12 vectors, 15 tests)

**Remaining:**

- Enable adapters by default (currently behind feature flags)
- Remove legacy duplicated code (~2,200 lines in client sandbox)

---

## 5. Strategic Recommendations

### Immediate Focus (P0/P1) - ✅ LARGELY COMPLETE

1.  ✅ **Scenario Matrix:** Comprehensive test suite with FAQ tests implemented (1195+ tests).
2.  ✅ **Parity Hardening:** Canonical orchestrator with adapters ensures identical behavior.
3.  **Frontend Polish:** Implement a complete HUD, spectator mode, and robust reconnection handling. _(Still pending)_

### Medium Term (P2) - 🔄 IN PROGRESS

1.  ✅ **AI Hardening:** Python contract tests replace runtime shadow validation.
2.  **Observability:** Add Prometheus metrics for AI latency and rule parity mismatches. _(Pending)_
3.  ✅ **Refactoring:** Full modular architecture implemented with aggregates, validators, mutators, and orchestration.

### Next Steps

- ✅ Adapters enabled by default (PASS20 complete - December 2025)
- ✅ Legacy code removed (PASS20: ~1,176 lines; Phase 4 Tier 2 deferred to post-MVP)
- ✅ Observability infrastructure (PASS21: 3 dashboards, k6 load testing)
- 🔄 Production validation (Execute load tests at scale, establish baselines)
- 🔄 Operational drills (Execute secrets rotation, backup/restore procedures)
- 🔄 Phase‑2 robustness focus (Dec 2025):
  - Engine/host lifecycle clarity (backend, sandbox, Python) for advanced phases via shared orchestrator/aggregates.
  - WebSocket lifecycle + reconnection windows documented in `docs/CANONICAL_ENGINE_API.md` and backed by reconnection/lobby/rematch tests.
  - TS↔Python territory & forced‑elimination parity finish‑up using contract vectors plus targeted Jest/Pytest suites.

---

**Document Version:** 3.0
**Last Updated:** November 26, 2025
**Maintained By:** Architecture Team
