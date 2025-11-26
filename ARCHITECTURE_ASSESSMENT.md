# RingRift Architecture Assessment & Roadmap

**Assessment Date:** November 21, 2025
**Last Updated:** November 26, 2025 (Phases 1-4 Complete)
**Status:** Comprehensive architecture review, evaluation, and future design
**Scope:** Server, client, shared engine, Python AI service, tests, and infrastructure

> This document is the **Single Source of Truth** for the project's architecture.
> It consolidates previous assessments (`CODEBASE_EVALUATION.md`) and future designs (`REFACTORING_ARCHITECTURE_DESIGN.md`).

> **🎉 Remediation Complete (2025-11-26)**: The rules engine consolidation (Phases 1-4) is now complete with:
>
> - Canonical turn orchestrator in [`src/shared/engine/orchestration/`](src/shared/engine/orchestration/)
> - Backend adapter ([`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts)) and client adapter ([`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts))
> - Cross-language contract tests achieving 100% parity (12 test vectors, 15 Python tests)

---

## 1. Executive Summary

RingRift follows a **TypeScript-first architecture** with a Node.js backend and React frontend, supplemented by a Python AI microservice.

**Overall Architecture Grade: B+**

- ✅ **Strengths:** Strong type system, shared engine logic, modern stack (React/Node/Prisma/Redis), and excellent documentation.
- ⚠️ **Weaknesses:** Testing coverage is uneven (strong unit tests but missing scenario matrix), AI integration is still maturing, and frontend UX lacks polish.
- ❌ **Gaps:** Advanced AI tactics (MCTS/NeuralNet) are not yet production-ready, and observability is minimal.

**Verdict:** The current monolithic architecture is **optimal** for the current stage. Premature microservice extraction (beyond the existing AI service) is strongly discouraged.

---

## 2. Current Architecture Overview

### Backend (Node.js + TypeScript)

- **Core Engine:** `GameEngine` (orchestration), `RuleEngine` (validation), `BoardManager` (state).
- **Session Management:** `GameSessionManager` handles lifecycle and distributed locking; `GameSession` wraps the engine, interaction handlers, and persistence.
- **Rules Abstraction:** `RulesBackendFacade` mediates between the TS engine and Python service, supporting shadow/parity modes.
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
- **Parity:** Maintains a shadow rules engine (`game_engine.py`) that mirrors TypeScript logic for validation.

### Shared Core (`src/shared/`)

- **Types:** Canonical `GameState`, `Move`, `Player` definitions.
- **Logic:** Pure functions for geometry, hashing, and invariant checking used by both client and server.

---

## 3. Codebase Evaluation

### Strengths

1.  **Rules Implementation:** Core mechanics (movement, capture chains, lines, territory) are implemented and verified across backend and sandbox.
2.  **Type Safety:** Shared types prevent drift between frontend, backend, and AI service.
3.  **Documentation:** Rules and architecture are well-documented and kept in sync.
4.  **Infrastructure:** Docker/Compose setup is production-ready.

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

### Proposed Structure ✅ IMPLEMENTED

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

### The "Move" Lifecycle

1.  **Input:** User submits `GameAction`.
2.  **Validation:** `GameEngine` calls specific `Validator`.
3.  **Mutation:** If valid, `Mutator` produces `nextState`.
4.  **Consequences:** Engine checks for lines/territory (derived actions).
5.  **Transition:** `PhaseManager` determines next phase/player.
6.  **Emit:** State broadcast to clients.

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

1.  Enable orchestrator adapters by default
2.  Remove legacy duplicated code in client sandbox (~2,200 lines)
3.  Complete production hardening (see `docs/drafts/PHASE1_REMEDIATION_PLAN.md`)

---

**Document Version:** 3.0
**Last Updated:** November 26, 2025
**Maintained By:** Architecture Team
