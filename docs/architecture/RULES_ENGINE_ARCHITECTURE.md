# RingRift Rules Engine Architecture & Rollout Strategy

> **SSoT alignment:** This document is a derived architectural view over the following canonical sources:
>
> - **Single Source of Truth (SSoT):** The canonical rules defined in `RULES_CANONICAL_SPEC.md` (together with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`) are the **ultimate authority** for RingRift game semantics. All implementations must derive from and faithfully implement these canonical rules.
> - **Implementation hierarchy:**
>   - **TS shared engine** (`src/shared/engine/**`) is the _primary executable derivation_ of the canonical rules spec. If the TS engine and the canonical rules document disagree, that is a bug in the TS engine.
>   - **Python AI service** (`ai-service/app/**`) is a _host adapter_ that must mirror the canonical rules. If Python disagrees with the canonical rules or the validated TS engine behaviour, Python must be updated—never the other way around.
>   - **Backend, sandbox, replay** and other hosts similarly derive from the canonical rules via the shared engine; they must not introduce independent rules semantics.
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` together with the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`) define the executable Move + orchestrator + WebSocket lifecycle that this doc describes.
> - **Precedence:** If this document ever conflicts with the canonical rules spec or with the shared TS engine + orchestrator/contracts that faithfully implement it, **the canonical rules + tests win** and this document must be updated to match them.
>
> In other words, this file explains how hosts are wired around the canonical rules SSoT and its shared TS implementation; it does not introduce a separate semantics SSoT.

**Last Updated:** December 06, 2025
**Scope:** Python Rules Engine, TypeScript Parity, Canonical Orchestrator, and Rollout Plan

**Doc Status (2025-11-26): Active (with historical/aspirational content)**

- Canonical rules semantics SSoT is the written spec in `RULES_CANONICAL_SPEC.md` together with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`. The **shared TypeScript engine** under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts: `schemas.ts`, `serialization.ts`, `testVectorGenerator.ts` + v2 vectors under `tests/fixtures/contract-vectors/v2/`) is the primary executable derivation of that spec and must be kept in lockstep with the canonical rules.
- Move/decision/WebSocket lifecycle semantics are documented in `docs/CANONICAL_ENGINE_API.md` and the shared TS/WebSocket types (`src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, `src/shared/validation/websocketSchemas.ts`).
- Backend (`GameEngine`, `TurnEngineAdapter`), client sandbox (`ClientSandboxEngine`, `SandboxOrchestratorAdapter`), and Python rules engine (`ai-service/app/game_engine.py`, `ai-service/app/rules/*`) are **hosts/adapters** over this rules SSoT; they must remain parity-validated against the canonical rules spec + shared engine but are not independent sources of rules semantics.
- Sections describing Python mutator-first refactors and future rollout phases should be read as **aspirational design** layered on top of the canonical rules SSoT (canonical rules spec plus shared TS engine), not as a redefinition of rules semantics.

This document defines the architecture of the Python rules engine within the AI service, its relationship to the canonical TypeScript engine, and the strategy for rolling it out as a parity-validated host over the canonical TypeScript engine in online validation flows.

---

## 1. Architecture Overview

### Core Concept

RingRift maintains two implementations of the game rules with a new canonical orchestration layer:

1.  **TypeScript Engine (Canonical):** Located in `src/shared/engine/`. Used by the Node.js backend (`GameEngine.ts`) and the client sandbox (`ClientSandboxEngine.ts`). This is the current source of truth.
2.  **Canonical Turn Orchestrator (NEW):** Located in `src/shared/engine/orchestration/`. Provides a single entry point (`processTurn()`) that orchestrates all domain aggregates in a deterministic sequence. Backend and sandbox adapters delegate to this layer.
3.  **Python Engine (AI/Shadow):** Located in `ai-service/app/game_engine.py`. Used for AI search/evaluation and currently being rolled out as a shadow validator for the backend. Contract tests ensure cross-language parity.

### Shared Rules Implementation

The canonical RingRift rules are implemented in the shared TypeScript engine under
[`src/shared/engine`](src/shared/engine/types.ts:1) as the primary executable
derivation of the canonical rules specification (`RULES_CANONICAL_SPEC.md` together
with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`). These modules are
pure, deterministic helpers that operate on shared `GameState` / `BoardState` types
and are reused by every host (backend, sandbox, tests, Python parity). The most
important groups are:

- **Movement & captures**
  - Geometry & non‑capturing reachability: [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1)
  - Overtaking capture enumeration: [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1)
  - Shared helpers for applying movement/capture segments and marker path effects: [`core.ts`](src/shared/engine/core.ts:1), [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1), [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1)
  - Aggregates and mutators: [`MovementAggregate.ts`](src/shared/engine/aggregates/MovementAggregate.ts:1), [`CaptureAggregate.ts`](src/shared/engine/aggregates/CaptureAggregate.ts:1), [`MovementMutator.ts`](src/shared/engine/mutators/MovementMutator.ts:1), [`CaptureMutator.ts`](src/shared/engine/mutators/CaptureMutator.ts:1)

- **Lines**
  - Marker‑line geometry: [`lineDetection.ts`](src/shared/engine/lineDetection.ts:21)
  - Canonical decision helpers: [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1) – enumerate and apply `process_line` and `choose_line_reward` decision `Move`s, and standardise when a line collapse grants a ring‑elimination opportunity (via `pendingLineRewardElimination`) for all hosts.
  - Aggregate: [`LineAggregate.ts`](src/shared/engine/aggregates/LineAggregate.ts:1)

- **Territory (detection, borders, processing)**
  - Region detection: [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36)
  - Border‑marker expansion: [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35)
  - Region processing pipeline (Q23 / self‑elimination compatible): [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1)
  - Canonical decision helpers: [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) – enumerates and applies `process_territory_region` and `eliminate_rings_from_stack` decisions, enforcing Q23, S‑invariant, and per‑player elimination bookkeeping for backend and sandbox hosts.
  - Aggregate: [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1)

- **Placement**
  - Placement validation (including no‑dead‑placement wiring): [`PlacementValidator.ts`](src/shared/engine/validators/PlacementValidator.ts:1)
  - Board‑level placement mutators: [`PlacementMutator.ts`](src/shared/engine/mutators/PlacementMutator.ts:1)

- **Victory**
  - Ring‑elimination, Territory‑majority, and stalemate ladder: [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts:51)

- **Turn sequencing & orchestration**
  - Shared phase/turn state machine: [`turnLogic.ts`](src/shared/engine/turnLogic.ts:132)
  - Canonical phase state machine for orchestrator: [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts:1)
  - **Canonical Turn Orchestrator:** [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – single entry point (`processTurn` / `processTurnAsync`) for all turn processing, delegating to domain aggregates ([`PlacementAggregate.ts`](src/shared/engine/aggregates/PlacementAggregate.ts:1), [`MovementAggregate.ts`](src/shared/engine/aggregates/MovementAggregate.ts:1), [`CaptureAggregate.ts`](src/shared/engine/aggregates/CaptureAggregate.ts:1), [`LineAggregate.ts`](src/shared/engine/aggregates/LineAggregate.ts:1), [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1), [`VictoryAggregate.ts`](src/shared/engine/aggregates/VictoryAggregate.ts:1)) in deterministic order

- **Orchestration (NEW - Phases 1-4 Complete)**
  - Canonical orchestrator: [`src/shared/engine/orchestration/`](src/shared/engine/orchestration/README.md:1)
  - Phase state machine: [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts:1)
  - Orchestration types: [`types.ts`](src/shared/engine/orchestration/types.ts:1)
  - Backend adapter: [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts:1)
  - Client adapter: [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1)
  - Host orchestration modes:
    - Backend `GameEngine.makeMove()` can operate in **adapter mode** (delegating to `TurnEngineAdapter` / `processTurnAsync`) or in **legacy mode** (validating via `RuleEngine` and applying moves directly via `applyMove`). Movement/capture mutation in the legacy path now uses Movement/Capture aggregates, but phase progression is still host-managed. In production, adapter mode is the **default and canonical** path; legacy mode is confined to tests, diagnostics, and kill‑switch/circuit‑breaker scenarios (see `EngineSelection.LEGACY` below).
    - Client `ClientSandboxEngine` now treats orchestrator‑driven flows (via `SandboxOrchestratorAdapter` and shared `turnLogic`) as the **canonical** rules path for sandbox games. Legacy click‑driven flows remain only as thin UI/test shims over shared aggregates and turn helpers, and are treated as **tests/tools‑only** surfaces rather than independent rules engines.
    - Python has no embedded TS orchestrator; instead, its `GameEngine` is parity‑validated against the TS orchestrator + contracts via v2 contract vectors and parity suites. It remains a host/adapter, not a separate rules SSoT.
  - Engine selection modes (backend):
    - `EngineSelection.ORCHESTRATOR` – canonical production path: `GameSession.configureEngineSelection()` enables the `TurnEngineAdapter` (`useOrchestratorAdapter=true`), and all turn processing goes through `processTurnAsync` with Movement/Capture/Placement/Line/Territory/Victory aggregates as the single source of rules.
    - `EngineSelection.SHADOW` – legacy remains authoritative, but the orchestrator runs in parallel on a cloned state via `ShadowModeComparator` and reports mismatch/error/latency metrics to Prometheus. Used for safe rollout and regression detection.
    - `EngineSelection.LEGACY` – test/tools‑only path: bypasses the adapter and uses the legacy `GameEngine`/`RuleEngine` turn pipeline. In production this mode is reachable only via the orchestrator kill switch or an open circuit breaker (see `OrchestratorRolloutService` and the rollout docs listed below). Normal gameplay is expected to use `ORCHESTRATOR` or `SHADOW`.

      **Legacy mode (REMOVED in PASS20):** Previously available as circuit-breaker fallback; ~1,147 lines of legacy code removed in December 2025. Orchestrator is now the only production path.

> **Rollout & runbooks:** The canonical description of environment phases, feature flags, SLOs, and rollback levers for orchestrator rollout now lives in:
>
> - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – environment phases 0–4, CI/SLO gates, and flag matrices.
> - `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md` – operator runbook for enabling/disabling orchestrator in staging/production.
> - `docs/STRICT_INVARIANT_SOAKS.md` – invariant soak posture (TS orchestrator + Python strict no‑move) used as rollout SLO inputs.

- **Contract Testing (NEW)**
  - Contract schemas: [`src/shared/engine/contracts/schemas.ts`](src/shared/engine/contracts/schemas.ts:1)
  - Contract serialization: [`src/shared/engine/contracts/serialization.ts`](src/shared/engine/contracts/serialization.ts:1)
  - Test vector generator: [`src/shared/engine/contracts/testVectorGenerator.ts`](src/shared/engine/contracts/testVectorGenerator.ts:1)
  - Contract test runner (TS): [`tests/contracts/contractVectorRunner.test.ts`](tests/contracts/contractVectorRunner.test.ts:1)
  - Contract test runner (Python): [`ai-service/tests/contracts/test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py:1)
  - Test vectors: [`tests/fixtures/contract-vectors/v2/`](tests/fixtures/contract-vectors/v2/) (12 vectors across 5 categories)

The **canonical description of the Move/decision/WebSocket lifecycle and engine decision surfaces** lives in `docs/CANONICAL_ENGINE_API.md`. This architecture document focuses on how the shared TS rules engine (helpers → aggregates → orchestrator → contracts) is hosted by the backend, sandbox, and Python engines, and how the Python engine is rolled out as a parity-validated validation host.

### Backend, Shared Engine, Orchestrator, and Sandbox Roles

At runtime there are four TypeScript layers:

- **Shared engine (pure helpers)** – modules under
  [`src/shared/engine`](src/shared/engine/types.ts:1) implement geometry,
  validation, and mutation operating on `GameState` / `BoardState`. They are
  side‑effect free except for returning updated state/board structures and are
  used as the single source of truth for rules semantics.

- **Canonical Turn Orchestrator (NEW)** – the orchestration layer in
  [`src/shared/engine/orchestration/`](src/shared/engine/orchestration/README.md:1) that:
  - Provides `processTurn()` and `processTurnAsync()` as single entry points for turn processing.
  - Calls domain aggregates (Placement, Movement, Capture, Line, Territory, Victory) in deterministic order.
  - Implements a phase state machine that advances through turn phases automatically.
  - Returns `TurnResult` with pending decisions, state changes, and victory conditions.
  - Is used directly by both backend and sandbox adapters.

- **Backend host (`TurnEngineAdapter`)** – server‑side adapter in
  [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts:1) that:
  - Wraps the canonical orchestrator with WebSocket and session concerns.
  - Handles player interaction delegation via `PlayerInteractionManager`.
  - Manages AI turn integration and timeout handling.
  - Calls through to the underlying `RuleEngine` / `GameEngine` for legacy flows.
  - Controlled by `useOrchestratorAdapter` feature flag.

- **Client sandbox host (`SandboxOrchestratorAdapter`)** – browser‑side adapter in
  [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1) that:
  - Wraps the canonical orchestrator for local simulation.
  - Provides the same interface as `ClientSandboxEngine` for seamless switching.
  - Uses the same shared helpers for movement, captures, lines, Territory,
    victory, and turn progression.
  - Controlled by `useOrchestratorAdapter` feature flag.
  - Provides local experimentation, AI‑vs‑AI simulations, and RulesMatrix / FAQ
    scenario playback. Legacy sandbox engines (`sandboxLinesEngine.ts`, `sandboxTerritoryEngine.ts`,
    `sandboxTurnEngine.ts`, `sandboxMovementEngine.ts`) have now been removed in favour of
    shared helpers and `ClientSandboxEngine` orchestration for movement, captures, lines,
    territory, and turn progression.

### How to Change or Extend Rules Safely

When changing rules, treat the shared engine as the only editable source of
truth:

1. Identify the relevant shared module(s):
   - Movement & captures (enumeration, application, chain state, aggregates): [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1),
     [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1),
     [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1),
     [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1),
     [`MovementAggregate.ts`](src/shared/engine/aggregates/MovementAggregate.ts:1),
     [`CaptureAggregate.ts`](src/shared/engine/aggregates/CaptureAggregate.ts:1),
     [`MovementMutator.ts`](src/shared/engine/mutators/MovementMutator.ts:1),
     [`CaptureMutator.ts`](src/shared/engine/mutators/CaptureMutator.ts:1),
     [`core.validateCaptureSegmentOnBoard`](src/shared/engine/core.ts:1).
   - Lines (geometry, aggregation, decision phases): [`lineDetection.ts`](src/shared/engine/lineDetection.ts:21),
     [`LineAggregate.ts`](src/shared/engine/aggregates/LineAggregate.ts:1),
     [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1) for `process_line` / `choose_line_reward` decisions and when line collapses grant `eliminate_rings_from_stack` opportunities.

- Territory (detection, borders, processing, decisions, aggregates): [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36),
  [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35),
  [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1),
  [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) for `process_territory_region` / `eliminate_rings_from_stack` decisions, Q23 gating, and elimination bookkeeping,
  [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1).
- Placement: [`PlacementValidator.ts`](src/shared/engine/validators/PlacementValidator.ts:1),
  [`PlacementMutator.ts`](src/shared/engine/mutators/PlacementMutator.ts:1),
  [`PlacementAggregate.ts`](src/shared/engine/aggregates/PlacementAggregate.ts:1) (canonical `place_ring` / `skip_placement` validation and mutation, including no-dead-placement and skip-eligibility helpers now consumed directly by backend `RuleEngine`, `GameEngine`, and `ClientSandboxEngine`),
  [`placementHelpers.ts`](src/shared/engine/placementHelpers.ts:1).
- Victory: [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts:51),
  [`VictoryAggregate.ts`](src/shared/engine/aggregates/VictoryAggregate.ts:1).
- Turn orchestration: [`turnLogic.ts`](src/shared/engine/turnLogic.ts:132),
  [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts:1),
  [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts:1),
  plus the domain aggregates listed above.

2. Update the shared helper(s) under
   [`src/shared/engine`](src/shared/engine/types.ts:1). Avoid editing
   backend‑only or sandbox‑only geometry unless absolutely necessary.

3. Extend or adjust the **shared unit tests** that cover these modules:
   - Movement & captures:
     [`movement.shared.test.ts`](tests/unit/movement.shared.test.ts:1),
     [`captureLogic.shared.test.ts`](tests/unit/captureLogic.shared.test.ts:1),
     [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts:1).
   - Lines:
     [`lineDetection.shared.test.ts`](tests/unit/lineDetection.shared.test.ts:1),
     [`LineDetectionParity.rules.test.ts`](tests/unit/LineDetectionParity.rules.test.ts:1),
     [`lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts:1) – **canonical** line‑decision enumeration and application (`process_line`, `choose_line_reward`) and when line collapses grant eliminations.
   - Territory:
     [`territoryBorders.shared.test.ts`](tests/unit/territoryBorders.shared.test.ts:1),
     [`territoryProcessing.shared.test.ts`](tests/unit/territoryProcessing.shared.test.ts:1),
     [`territoryProcessing.rules.test.ts`](tests/unit/territoryProcessing.rules.test.ts:1),
     [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1) – **canonical** `process_territory_region` / `eliminate_rings_from_stack` decision semantics, including Q23 and S‑invariant bookkeeping.
   - Placement:
     [`placement.shared.test.ts`](tests/unit/placement.shared.test.ts:1).
   - Victory:
     [`victory.shared.test.ts`](tests/unit/victory.shared.test.ts:1).
   - Turn sequencing & termination:
     [`GameEngine.decisionPhases.MoveDriven.test.ts`](tests/unit/GameEngine.decisionPhases.MoveDriven.test.ts:1),
     [`SandboxAI.ringPlacementNoopRegression.test.ts`](tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts:1),
     [`GameEngine.aiSimulation.test.ts`](tests/unit/GameEngine.aiSimulation.test.ts:1).

4. Re‑run parity suites that ensure backend and sandbox still match the shared
   helpers and orchestrator:
   - Backend vs sandbox & host engines (TypeScript): representative suites under `tests/unit/` such as
     [`Backend_vs_Sandbox.traceParity.test.ts`](tests/unit/Backend_vs_Sandbox.traceParity.test.ts:1),
     [`Backend_vs_Sandbox.eliminationTrace.test.ts`](tests/unit/Backend_vs_Sandbox.eliminationTrace.test.ts:1),
     [`Backend_vs_Sandbox.seed5.internalStateParity.test.ts`](tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts:1),
     [`Backend_vs_Sandbox.seed5.checkpoints.test.ts`](tests/unit/Backend_vs_Sandbox.seed5.checkpoints.test.ts:1),
     [`Backend_vs_Sandbox.seed1.snapshotParity.test.ts`](tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts:1) / [`archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts`](archive/tests/parity/Backend_vs_Sandbox.seed1.snapshotParity.test.ts:1),
     [`Backend_vs_Sandbox.seed18.snapshotParity.test.ts`](tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts:1) / [`archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts`](archive/tests/parity/Backend_vs_Sandbox.seed18.snapshotParity.test.ts:1),
     [`Sandbox_vs_Backend.aiRngParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1),
     [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1),
     [`TerritoryParity.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts:1),
     [`TerritoryBorders.Backend_vs_Sandbox.test.ts`](tests/unit/TerritoryBorders.Backend_vs_Sandbox.test.ts:1),
     [`TerritoryCore.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts:1),
     [`MarkerPath.GameEngine_vs_Sandbox.test.ts`](tests/unit/MarkerPath.GameEngine_vs_Sandbox.test.ts:1),
     [`TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts`](tests/unit/TerritoryPendingFlag.GameEngine_vs_Sandbox.test.ts:1),
     [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts:1) (heuristic AI parity, including pie-rule/swap_sides and deep seeded runs).
   - Shared engine vs hosts (TypeScript):
     [`TraceFixtures.sharedEngineParity.test.ts`](tests/unit/TraceFixtures.sharedEngineParity.test.ts:1),
     [`EngineDeterminism.shared.test.ts`](tests/unit/EngineDeterminism.shared.test.ts:1),
     [`NoRandomInCoreRules.test.ts`](tests/unit/NoRandomInCoreRules.test.ts:1).
   - Python vs TypeScript parity (AI service):
     [`test_rules_parity.py`](ai-service/tests/parity/test_rules_parity.py:1),
     [`test_rules_parity_fixtures.py`](ai-service/tests/parity/test_rules_parity_fixtures.py:1),
     [`test_ts_seed_plateau_snapshot_parity.py`](ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py:1),
     [`test_line_and_territory_scenario_parity.py`](ai-service/tests/parity/test_line_and_territory_scenario_parity.py:1),
     [`test_ai_plateau_progress.py`](ai-service/tests/parity/test_ai_plateau_progress.py:1).

5. Validate **RulesMatrix** and **FAQ** scenarios under
   [`tests/scenarios`](tests/scenarios/FAQ_Q09_Q14.test.ts:1) remain green; add
   new scenario cases where appropriate (see
   [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:1) and
   [`tests/README.md`](tests/README.md:1)).

### Python Engine Structure

- [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py:1): Core host adapter exposing:
  - `get_valid_moves(state, player)` – **interactive-only** legal moves for the current phase (no auto `NO_*_ACTION` or forced-elimination moves).
  - `get_phase_requirement(state, player)` – phase-level requirements when no interactive moves exist (e.g., `NO_*_ACTION_REQUIRED`, `FORCED_ELIMINATION_REQUIRED`).
  - `synthesize_bookkeeping_move(requirement, state)` – host-level helper to construct canonical `NO_*_ACTION` / `FORCED_ELIMINATION` moves from a `PhaseRequirement`.
  - `apply_move(state, move, trace_mode=False)` – move application + phase transitions, delegating phase logic to the dedicated phase machine.
- [`ai-service/app/rules/phase_machine.py`](ai-service/app/rules/phase_machine.py:1): Phase/turn state machine:
  - Pure phase + turn transitions (no board/player mutation, no move fabrication).
  - Mirrors TS `phaseStateMachine.ts` / `turnOrchestrator.processPostMovePhases` for:
    - `ring_placement → movement → capture/chain_capture → line_processing → territory_processing → forced_elimination → rotation`.
    - `NO_*_ACTION` transitions between decision phases.
- [`ai-service/app/board_manager.py`](ai-service/app/board_manager.py:1): Board-level utilities (hashing, S-invariant, line detection, Territory regions).
- [`ai-service/app/rules/default_engine.py`](ai-service/app/rules/default_engine.py:1): Adapter that delegates to `GameEngine` while routing key move types through dedicated mutators under _shadow contracts_.
- [`ai-service/app/models/`](ai-service/app/models/__init__.py:1): Pydantic models mirroring the TypeScript shared engine types.

### TypeScript Integration

- [`src/server/game/RulesBackendFacade.ts`](src/server/game/RulesBackendFacade.ts:1): The primary entry point for the backend. It abstracts the choice between the local TS engine and the Python Service based on `RINGRIFT_RULES_MODE` while always applying moves via the canonical shared-engine–backed `GameEngine`.
- [`src/server/services/PythonRulesClient.ts`](src/server/services/PythonRulesClient.ts:1): Handles HTTP communication with the Python AI Service (`/rules/evaluate_move`).

### Parity Mechanisms

To ensure the Python engine behaves exactly like the TypeScript engine:

- **Shared Types:** Domain models (`GameState`, `Move`, etc.) are structurally identical.
- **Unified Move Model:** Both engines use the same extended Move types (`continue_capture_segment`, `process_line`, `process_territory_region`) for complex phases.
- **Hashing:** `hash_game_state` produces identical output (modulo known extensions like `:must_move=`) for state verification.
- **S-Invariant:** Both engines compute the same progress metric (`S = markers + collapsed + eliminated`) to guarantee termination.

---

## 2. Python ↔ TypeScript Parity Mapping

| Feature       | TypeScript Implementation                   | Python Implementation               | Status    |
| :------------ | :------------------------------------------ | :---------------------------------- | :-------- |
| **Movement**  | Ray-based geometry, stack-height distance   | Mirrors TS geometry and constraints | ✅ Parity |
| **Captures**  | Overtaking with cap-height check            | Mirrors TS validation logic         | ✅ Parity |
| **Chains**    | `ChainCaptureState`, mandatory continuation | `ChainCaptureState`, same logic     | ✅ Parity |
| **Placement** | Multi-ring, no-dead-placement check         | Mirrors TS placement rules          | ✅ Parity |
| **Lines**     | Exact vs Overlength rewards                 | Mirrors TS line processing          | ✅ Parity |
| **Territory** | Disconnection, self-elimination prereq      | Mirrors TS territory logic          | ✅ Parity |
| **Victory**   | Ring elimination, Territory control         | Mirrors TS victory checks           | ✅ Parity |

**Verification:**

- **Trace Parity:** TS-generated parity traces under `tests/fixtures/rules-parity/v1/**` together with the integration harness in [`src/server/game/test-python-rules-integration.ts`](src/server/game/test-python-rules-integration.ts:1) feed the Python parity suites.
- **Fixture Parity:** `ai-service/tests/parity/test_rules_parity_fixtures.py` validates TS-generated fixtures against the Python engine.

### 2.1 Multi-phase turn vectors and phase sequencing

The canonical **multi-phase turn sequence** (ring_placement → movement / capture → chain_capture → line_processing → territory_processing → forced_elimination) is encoded and exercised in three layers:

- **Contract vectors (TS SSoT for phase sequences).**
  - `tests/fixtures/contract-vectors/v2/multi_phase_turn.vectors.json` contains v2 test vectors such as:
    - `multi_phase.placement_capture_line`
    - `multi_phase.full_sequence_with_territory`
  - Each vector specifies:
    - An initial TS `GameState` snapshot.
    - An initial `overtaking_capture` move.
    - An `expectedPhaseSequence`, typically:
      - `["movement", "chain_capture", "line_processing", "territory_processing"]`
    - A set of `phaseTransitions` describing when chain capture becomes available, when a line is completed, and when Territory processing should begin.
  - The `sequence:turn.line_then_territory.*` tags on these vectors
    (for `square8`, `square19`, and `hexagonal`) are the canonical
    identifiers for “line-then-Territory” turns referenced by RR‑CANON‑R208/R209.

- **Snapshot parity tests (TS ↔ Python structure).**
  - `ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py`:
    - Validates plateau snapshots (seeds 1 and 18) against Python models using TS-generated `ComparableSnapshot` JSON.
  - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`:
    - Encodes the Q7/Q20/Q22-style line+Territory scenario for all three board types.
    - Contains:
      - Per-board synthetic line+Territory fixtures:
        - `LINE_TERRITORY_SNAPSHOT_BY_BOARD[...]` → `line_territory_scenario_*.snapshot.json`
        - `MULTI_REGION_SNAPSHOT` → `line_territory_multi_region_square8.snapshot.json`
      - Tests that:
        - Rebuild a Python `GameState` equivalent to the TS snapshot.
        - Exercise the combined `line_processing` → `territory_processing` flow via `GameEngine._get_line_processing_moves` and `GameEngine._get_territory_processing_moves`.
        - Assert deep equality between TS and Python `ComparableSnapshot` shapes for line+Territory scenarios.

- **As implemented (Dec 2025) – host adapters for multi-phase turns.**
  - **Backend TS GameEngine (`src/server/game/GameEngine.ts`).**
    - Multi-phase boundaries and decision surfaces are wired through:
      - `makeMove` / `processMoveViaAdapter`:
        - Detect chain-capture continuation via `getChainCaptureContinuationInfo` and set `currentPhase = 'chain_capture'`, maintaining an internal `chainCaptureState`.
        - On chain exhaustion, clear `chainCaptureState` and transition into `line_processing` when lines exist (or apply RR‑CANON‑R204 when not).
      - `getValidLineProcessingMoves(playerNumber)`:
        - Thin adapter over `enumerateProcessLineMoves` and `enumerateChooseLineRewardMoves` to surface canonical `process_line` / `choose_line_reward` Moves for the current player.
      - `getValidTerritoryProcessingMoves(playerNumber)`:
        - Thin adapter over `enumerateProcessTerritoryRegionMoves` to surface canonical `process_territory_region` Moves, with elimination decisions delegated to shared helpers.
      - `getValidMoves(playerNumber)`:
        - Wraps the shared `RuleEngine.getValidMoves` and layers in:
          - `chain_capture`-specific behaviour (restricted `continue_capture_segment` options).
          - The swap-sides meta-move via `shouldOfferSwapSidesMetaMove`.
          - The ACTIVE_NO_MOVES safeguard and resolver (`resolveBlockedStateForCurrentPlayerForTesting`) for rare blocked states.
    - These methods must remain aligned with the shared TS orchestrator and the RR‑CANON‑R208/R209 phase ordering; the backend is not allowed to introduce divergent multi-phase semantics.

  - **Python AI-service GameEngine (`ai-service/app/game_engine.py`).**
    - Mirrors the same multi-phase sequence as a host adapter over Python models:
      - `GameEngine.get_valid_moves(state, player_number)`:
        - Dispatches to `_get_ring_placement_moves`, `_get_movement_moves`, `_get_capture_moves`, `_get_line_processing_moves`, `_get_territory_processing_moves`, and `_get_forced_elimination_moves` based on `current_phase`.
        - Mirrors TS behaviour for:
          - Optional placement with `skip_placement`.
          - Movement / capture + chain-capture enumeration (`_apply_chain_capture` and `enumerate_capture_moves_py`).
          - Line-processing and Territory-processing decision phases.
      - `_get_line_processing_moves`:
        - Adapts BoardManager line detection + TS-style line decision semantics into Python `Move` instances (`PROCESS_LINE`, `CHOOSE_LINE_REWARD`), matching `getValidLineProcessingMoves` in the TS backend.
      - `_get_territory_processing_moves`:
        - Mirrors TS Territory decision enumeration for `PROCESS_TERRITORY_REGION` and `ELIMINATE_RINGS_FROM_STACK`, using the same Q23 gating as the shared engine.
      - `_update_phase` / `_advance_to_line_processing` / `_end_turn`:
        - Implement the phase transitions described by RR‑CANON‑R208/R209:
          - `movement` / `capture` → `chain_capture` (when continuation exists).
          - `chain_capture` → `line_processing` → `territory_processing` (when applicable) → `forced_elimination` (if blocked) → next `ring_placement` / `movement` or victory.
    - Python parity tests treat these helpers as **host wiring** and assert that they reproduce the TS contract vectors and snapshots exactly; they are not independent semantics SSoTs.

### DefaultRulesEngine ↔ GameEngine Equivalence Coverage

To safely evolve `DefaultRulesEngine` toward a mutator-driven architecture, we
maintain explicit equivalence tests that assert its behaviour stays in
lockstep with `GameEngine.apply_move` for key move families.

| Move Family                 | Move Types                   | Test File                                                   | Scenario Source                                                                                                                                                                                                                                                                                                   |
| :-------------------------- | :--------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Placement**               | `place_ring`                 | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven via `RingRiftEnv` on SQUARE8 + SQUARE19                                                                                                                                                                                                                                                                |
| **Movement**                | `move_stack`                 | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven via `RingRiftEnv` until first movement                                                                                                                                                                                                                                                                 |
| **Capture – Initial**       | `overtaking_capture`         | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic overtaking segment + env-driven capture search                                                                                                                                                                                                                                                          |
| **Capture – Continuation**  | `continue_capture_segment`   | `ai-service/tests/rules/test_default_engine_equivalence.py` | Env-driven: apply first capture, then continue chain                                                                                                                                                                                                                                                              |
| **Line Processing**         | `process_line`               | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic line via `BoardManager.find_all_lines` monkeypatch; canonical TS semantics live in [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1) and are enforced by [`lineDecisionHelpers.shared.test.ts`](tests/unit/lineDecisionHelpers.shared.test.ts:1)                                   |
| **Territory – Region**      | `process_territory_region`   | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic disconnected region via `BoardManager.find_disconnected_regions`; canonical TS semantics live in [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) and are enforced by [`territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1) |
| **Territory – Elimination** | `eliminate_rings_from_stack` | `ai-service/tests/rules/test_default_engine_equivalence.py` | Synthetic single capped stack via `_get_territory_processing_moves`; elimination bookkeeping must match the shared helpers in [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) and `TerritoryMutator`                                                                             |

Additionally, TS-generated trace fixtures are replayed through both engines:

- `ai-service/tests/parity/test_rules_parity_fixtures.py`:
  - `test_replay_ts_trace_fixtures_and_assert_python_state_parity` asserts hash + S-invariant parity for `GameEngine.apply_move` against TS.
  - `test_default_engine_matches_game_engine_when_replaying_ts_traces` asserts
    full-state lockstep between `DefaultRulesEngine.apply_move` and
    `GameEngine.apply_move` for every move in the TS traces (captures,
    line-processing, Territory, etc.).
  - `test_default_engine_mutator_first_matches_game_engine_on_ts_traces` runs
    the same traces with `DefaultRulesEngine(mutator_first=True)`, exercising
    the full mutator-first orchestration path while still comparing the
    resulting state against `GameEngine.apply_move`.

- `ai-service/tests/rules/test_default_engine_mutator_first_scenarios.py`:
  - `test_mutator_first_env_smoke_for_place_ring_and_move_stack` uses
    `RingRiftEnv` to find realistic `PLACE_RING` and `MOVE_STACK` moves and
    asserts that mutator-first mode stays aligned with `GameEngine.apply_move`.
  - `test_mutator_first_process_territory_region_synthetic` mirrors the
    synthetic disconnected-region scenario from the equivalence tests but with
    `mutator_first=True`, ensuring Territory processing plus downstream
    forced-elimination are consistent.

- `ai-service/tests/rules/test_default_engine_flags.py` verifies configuration
  of the mutator-first mode itself (see below).

These tests form the safety net for any future refactors that change
`DefaultRulesEngine` from a pure adapter into a mutator-first orchestrator.

### Mutator-First Mode and Configuration

`DefaultRulesEngine` exposes an optional _mutator-first_ execution path that
mirrors `GameEngine.apply_move` while delegating board/player mutations to the
Python mutators. This is currently used **only as a shadow contract**; the
canonical state returned from `apply_move` still comes from `GameEngine`.

#### Configuration surfaces

Mutator-first behaviour is controlled by three inputs, evaluated in a fixed
order:

1. **Server-level gate (ops-owned):**
   - `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST`
   - If this is not truthy, mutator-first is **hard-disabled**, regardless of
     any per-service flags or constructor arguments.
2. **Per-service default (AI service):**
   - `RINGRIFT_RULES_MUTATOR_FIRST`
   - Only consulted when the server gate is truthy and the constructor
     argument is omitted.
3. **Constructor override (code-level):**
   - `DefaultRulesEngine(mutator_first=...)`
   - Wins over the per-service env flag whenever the server gate allows
     mutator-first.

#### Constructor argument

```python
engine = DefaultRulesEngine(mutator_first=True)   # request enable (if gate allows)
engine = DefaultRulesEngine(mutator_first=False)  # explicitly disable
engine = DefaultRulesEngine()                     # defer to env flag
```

#### Environment variables

- **Server-level gate:**
  - `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST` is read in `__init__`.
  - Truthy values (case-insensitive) _permit_ mutator-first mode:
    - `"1"`, `"true"`, `"yes"`, `"on"`
  - Any other value (or unset) means mutator-first is **disabled**, even if
    the per-service flag or constructor requests it.

- **Per-service default:**
  - `RINGRIFT_RULES_MUTATOR_FIRST` is read once in `__init__` when
    `mutator_first` is omitted **and** the server gate is truthy.
  - Truthy values (case-insensitive) enable mutator-first mode by default:
    - `"1"`, `"true"`, `"yes"`, `"on"`
  - Any other value (or unset) leaves mutator-first disabled.

Constructor arguments always override the per-service env flag, but are still
subject to the server-level gate.

The behaviour of this configuration model is covered by
`ai-service/tests/rules/test_default_engine_flags.py`.

#### Effective behaviour matrix

The following examples illustrate how the three inputs combine (all env values
are shown after lowercasing):

- **Server gate falsey / unset:** mutator-first is always disabled.
  - `SERVER=""`, `ENV="true"`, `ctor=True` → disabled
  - `SERVER="0"`, `ENV="on"`, `ctor=None` → disabled
- **Server gate truthy, env truthy:**
  - `SERVER="1"`, `ENV="true"`, `ctor=None` → enabled (per-service default)
  - `SERVER="1"`, `ENV="true"`, `ctor=False` → disabled (constructor wins)
  - `SERVER="1"`, `ENV="true"`, `ctor=True` → enabled (constructor wins)
- **Server gate truthy, env falsey / unset:**
  - `SERVER="1"`, `ENV=""`, `ctor=None` → disabled (no default)
  - `SERVER="1"`, `ENV=""`, `ctor=True` → enabled (constructor opt-in)

This matches the semantics in `DefaultRulesEngine.__init__`:

1. If the server gate is not truthy → `_mutator_first_enabled = False`.
2. Else if the constructor argument is provided → use that value.
3. Else → use the per-service env flag.

#### Runtime behaviour when enabled

- `DefaultRulesEngine.apply_move` always computes a canonical result via
  `GameEngine.apply_move(state, move)`.
- It then runs `_apply_move_with_mutators(state, move)`, which:
  - Performs copy-on-write of the board/state.
  - Manages zobrist hashing (player/phase contributions) in the same pattern
    as `GameEngine.apply_move`.
  - Delegates to the specialised mutators for each move family
    (`PlacementMutator`, `MovementMutator`, `CaptureMutator`, `LineMutator`,
    `TerritoryMutator`) and uses `GameEngine._apply_forced_elimination` for
    forced/cap elimination.
  - Updates `move_history`, `last_move_at`, `must_move_from_stack_key`, calls
    `GameEngine._update_phase`, reapplies zobrist contributions, and finally
    invokes `GameEngine._check_victory`.
- The resulting mutator-first state is then compared against the canonical
  `GameEngine.apply_move` result on:
  - `board.stacks`, `board.markers`, `board.collapsed_spaces`,
    `board.eliminated_rings`
  - `players`
  - `current_player`, `current_phase`, `game_status`
  - `chain_capture_state`, `must_move_from_stack_key`
- Any divergence raises a `RuntimeError` with a message that includes a
  compact move description and summary statistics (e.g. counts of stacks or
  markers) to aid debugging.

Because `apply_move` still **returns** the canonical `GameEngine` state, this
mode is safe to enable in shadow for diagnostics without changing external
behaviour.

---

## 3. Rollout Strategy

The rollout is controlled by the `RINGRIFT_RULES_MODE` environment variable.

### Phase 1: Shadow Mode (`RINGRIFT_RULES_MODE=shadow`)

- **Behavior:** TS engine is authoritative. For every move, the backend asynchronously calls the Python service (`/rules/evaluate_move`) to compare results.
- **Metrics:** Mismatches in validity, state hash, or S-invariant are logged and increment Prometheus counters (`rules_parity_*_mismatch_total`).
- **Goal:** Verify parity at scale in staging/production without affecting gameplay.

### Phase 2: Python Authoritative (`RINGRIFT_RULES_MODE=python`)

- **Behavior:** The backend consults the Python service _first_.
  - If Python rejects the move, it is rejected.
  - If Python accepts, the move is applied (currently via TS engine in "reverse shadow" for safety, eventually by trusting Python state).
- **Fallback:** On Python service failure, fall back to TS engine and log `backend_fallback`.
- **Goal:** Make the Python engine the **primary online validation host over the canonical TS orchestrator + contracts**, enabling advanced AI features that rely on precise rule simulation while keeping the written canonical rules spec as the rules SSoT and the TS shared engine as its primary executable implementation.

### Acceptance Criteria for Phase 2

1.  **Shadow Stability:** Zero parity mismatches over a significant period in Phase 1.
2.  **Performance:** Python service latency (P99) is within acceptable bounds (< 50ms).
3.  **Operational Readiness:** AI service is horizontally scalable and monitored.

---

## 4. Future Refactoring

The Python engine is evolving towards a purely functional, mutator-driven architecture:

1.  **Mutator Extraction:** Logic is being moved from `GameEngine` methods into dedicated `Mutator` classes (e.g., `PlacementMutator`, `CaptureMutator`).
2.  **Shadow Contracts:** `DefaultRulesEngine` currently enforces that Mutators produce the same result as `GameEngine`.
3.  **Goal:** Eventually replace `GameEngine` orchestration with a composable pipeline of Validators and Mutators, matching the proposed TypeScript refactoring.

---

## 5. Canonical Territory, Q23, Elimination, and S-Invariant Semantics

This section documents the canonical semantics for **Territory disconnection**, **Q23 self-elimination**, **elimination bookkeeping**, and the **S-invariant** as implemented by the TypeScript sandbox and shared engine. The backend GameEngine and Python engine are required to match these behaviours.

### 5.0 Board Cell Exclusivity Invariants

Across all engines (backend TS, sandbox TS, Python), **each board cell is exclusive** in terms of what it may contain at steady state:

- A cell may be **empty**, or
- It may contain **exactly one stack** (one `RingStack`), or
- It may contain **exactly one marker** (a single `Marker` entry), or
- It may be marked as **collapsed Territory** (`collapsedSpaces` entry for one player),

but **never any combination** of these at the same time.

Concretely:

- **No stack on collapsed Territory:** if a position is in `board.collapsedSpaces`, `board.stacks` must not have an entry for the same key.
- **No stack + marker coexistence:** if `board.stacks` has a stack at a key, `board.markers` must not have a marker at that key.
- **No marker on collapsed Territory:** if `board.collapsedSpaces` contains a key, `board.markers` must not contain that key.

The following implementation points enforce these invariants:

- Backend `BoardManager.setCollapsedSpace` removes any stack and marker before inserting a `collapsedSpaces` entry.
- Backend `BoardManager.setStack` now logs and then **removes any existing marker** before setting a stack at that position.
- Sandbox `ClientSandboxEngine.collapseMarker` and `collapseLineMarkers` clear both stacks and markers when creating collapsed Territory.
- Sandbox placement (`tryPlaceRings`) rejects any attempt to place a stack on top of a marker, keeping stack/marker maps disjoint.
- The sandbox test helper `ClientSandboxEngine.assertBoardInvariants` asserts these conditions under Jest when `NODE_ENV=test`.

These exclusivity rules are considered **part of the rules contract** and must be maintained by any future engine or mutator implementations.

### 5.1 Territory Regions and Disconnection

**Reference implementations:**

- Client sandbox:
  [`sandboxTerritory.ts`](src/client/sandbox/sandboxTerritory.ts:1),
  [`ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer`](src/client/sandbox/ClientSandboxEngine.ts:2057).
- Shared engine (canonical semantics SSoT):
  [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36),
  [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35),
  [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1),
  [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1).
- Python adapter (semantic boundary):
  [`territory.py` (TerritoryMutator)](ai-service/app/rules/mutators/territory.py:1) – Python mutator layered over the canonical TS Territory helpers/aggregate above; it is a semantic boundary, **not** an independent rules SSoT.
- Backend helpers:
  [`BoardManager.ts`](src/server/game/BoardManager.ts:1),
  [`GameEngine.ts`](src/server/game/GameEngine.ts:1) – both delegate to the
  shared Territory helpers above rather than maintaining a separate
  server‑local implementation.

**Board geometries:**

- `square8`, `square19`: territory regions use Von Neumann (orthogonal) adjacency, via `BOARD_CONFIGS[boardType].territoryAdjacency`.
- `hexagonal`: territory regions use hex adjacency on cube coordinates, also driven by `BOARD_CONFIGS[boardType].territoryAdjacency`.

**Region definition and disconnection (canonical):**

- A **region** is a maximal set of **non‑collapsed** cells (empty, marker, or stack) that are connected via the board’s `territoryAdjacency`. This matches RR‑CANON‑R040 and the implementation in [`territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:36).
- A region is **physically disconnected** when every adjacency path from any cell in the region to any non‑collapsed cell outside the region must cross only:
  - collapsed spaces, and/or
  - board edge, and/or
  - markers belonging to exactly one border color, as formalised in [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35).
- Color representation (RR‑CANON‑R142) and the Q23 self‑elimination prerequisite (RR‑CANON‑R143) are handled by [`territoryProcessing.canProcessTerritoryRegion`](src/shared/engine/territoryProcessing.ts:99) and [`territoryDecisionHelpers`](src/shared/engine/territoryDecisionHelpers.ts:1); `BoardManager.findDisconnectedRegions` and the sandbox adapters now delegate to these shared helpers rather than maintaining a separate server‑local implementation.

The shared `territoryDetection` implementation is the normative algorithm for detecting such regions; backend and Python code must either call it or faithfully mirror its behaviour.

### 5.2 Q23: Self-Elimination Prerequisite

**Intent (FAQ Q23):**

> A player may only collapse and score a disconnected Territory region they control if they have at least one stack or cap **outside** that region from which the mandatory self-elimination cost can be paid.

We distinguish two cases for the controlling (moving) player:

- **Negative Q23 (not eligible):**
  - The player has **no** stacks/caps anywhere outside the region.
  - Result: the region **must not be processed** at all for that player.
    - No interior spaces collapse.
    - No interior stacks are eliminated for that player.
    - No self-elimination is applied.

- **Positive Q23 (eligible):**
  - The player has **at least one** stack or cap **outside** the region.
  - Result: the region **may be processed** for that player:
    - All interior empty spaces collapse into controlled Territory.
    - All interior stacks belonging to any player are eliminated according to the elimination rules (below).
    - The controlling player pays a self-elimination cost from outside the region.

**Canonical implementation notes:**

- Eligibility is computed using `BoardManager.getPlayerStacks(board, player)` over the _current_ board state, and then partitioning those stacks into **inside** vs **outside** the region.
- Q23 is applied **per region** during disconnected-region processing:
  - If the moving player has no outside stacks, that region is skipped.
  - If they do, the region is eligible and can be processed.
- The sandbox and shared engine do **not** treat rings in hand as satisfying Q23; only physical stacks/caps on the board count.

### 5.3 Elimination Semantics

Elimination in RingRift falls into three broad categories:

1. **Self-elimination (Territory cost):**
   - Occurs when a player processes a Territory region they control under positive Q23.
   - The cost is paid by removing one of the player's own stacks or caps **outside** the region.
   - Canonical behaviour (sandbox + `CaptureMutator.mutateEliminateStack`):
     - If a **cap** is chosen, the **entire capped stack** is removed and the player is credited with `capHeight` eliminated rings.
     - If a plain stack (no cap) is chosen, its entire height is removed and contributes that many rings.

2. **Internal eliminations (inside the region):**
   - All stacks inside the collapsing Territory region are eliminated when the region is processed.
   - Each eliminated stack contributes its full height (or cap height when relevant) to `eliminatedRings`.
   - Ownership of eliminated stacks determines how scores/metrics are attributed, but for S-invariant purposes only the count of eliminated rings matters.

3. **Opponent eliminations (captures, lines):**
   - Governed by capture and line rules elsewhere (see capture/line sections and tests), but they also contribute to `eliminatedRings`.
   - Self-elimination and opponent elimination share the same underlying elimination mutator (`CaptureMutator` / Python equivalents) to maintain consistent bookkeeping.

Across all categories:

- The **canonical rule** is that when a stack is eliminated, the number of rings added to `eliminatedRings` equals the number of markers removed from the board for that stack (cap height or full stack height).
- Territory processing must use the same elimination pipeline as captures/lines; no bespoke deletion logic is allowed.

### 5.4 S-Invariant

**Definition:**

> `S = markers + collapsedSpaces + eliminatedRings`

Where:

- `markers` = total count of non-empty marker cells on the board (all players).
- `collapsedSpaces` = number of spaces that have been converted into collapsed Territory.
- `eliminatedRings` = cumulative count of rings removed from stacks via capture, self-elimination, line processing, or Territory collapse.

**Properties:**

- In all canonical engines (sandbox, shared TS, Python), `S` is **non-decreasing** over the course of a game.
- AI simulations rely on non-decreasing `S` and stall detection rather than strict per-move increase.
- The baseline test `tests/unit/SInvariant.seed17FinalBoard.test.ts` asserts that for a reference game (seed 17) the final value of `S` is exactly `74`, and this value is shared across all engines.

**Guard-rail tests:**

- `tests/unit/SharedMutators.invariants.test.ts` verifies that shared mutators respect S-invariant properties.
- `tests/unit/GameEngine.aiSimulation.test.ts` (and debug variants) ensure that automatic consequence processing (lines + Territory + forced elimination) never violates non-decreasing `S`.

Any change to Territory, capture, or elimination logic must keep these invariants intact.

### 5.5 Worked Examples (Informal)

The following examples are encoded as tests and serve as _living specifications_:

- **Single-region Q23 positive (square19):**
  - Tests: `tests/unit/GameEngine.territoryDisconnection.test.ts`, `tests/unit/territoryProcessing.rules.test.ts`.
  - Scenario: a 3x3 interior region is fully surrounded by Player 1's markers. Player 1 has at least one stack outside the region.
  - Expected behaviour:
    - Region is detected as disconnected and eligible (positive Q23).
    - All 9 interior spaces collapse to Territory; `board.collapsedSpaces` has value `1` at those coordinates.
    - All interior stacks are eliminated; `board.stacks` has no entries at those locations.
    - One outside Player 1 stack is self-eliminated according to `mutateEliminateStack` semantics.
    - `S` increases appropriately due to new `collapsedSpaces` and `eliminatedRings`.

- **Q23 negative (sandbox rules):**
  - Tests: `tests/unit/sandboxTerritoryEngine.rules.test.ts` (legacy‑named diagnostic harness that now exercises `ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer`), `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` (Q23-specific cases).
  - Scenario: a region is fully enclosed by Player 1's markers, but Player 1 has no stacks or caps outside that region.
  - Expected behaviour:
    - Region is detected as disconnected but **ineligible** (negative Q23).
    - No collapse occurs; interior spaces remain normal empties.
    - No stacks are eliminated; `eliminatedRings` and `collapsedSpaces` do not change.

- **Seed 17 parity checkpoints:**
  - Tests: `tests/unit/Seed17GeometryParity.GameEngine_vs_Sandbox.test.ts`, `tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`, `tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts`.
  - These parity suites assert that for a difficult seed (17), backend and sandbox agree on:
    - Geometry and movement legality.
    - Territory disconnections and collapses.
    - Elimination counts and S-invariant at key move numbers, including the final board.

### 5.6 Backend Pipeline Ordering (Territory vs Lines)

The backend GameEngine and sandbox must agree on the **effective order** in which Territory and line processing are applied after a move:

- Canonical intention:
  - Disconnected Territory regions that become eligible as a result of a move should be processed in a way that is consistent with sandbox behaviour and Q23.
  - Line formation and capture consequences must also be applied, but must not create artefacts that violate Q23 semantics (e.g., erasing all outside stacks _before_ Territory is evaluated when sandbox would not).

- Current backend implementation (TS):
  - `GameEngine.processAutomaticConsequences` calls into Territory processing and line processing in a carefully chosen order so that:
    - Territory disconnection tests (`GameEngine.territoryDisconnection.*.test.ts`) match sandbox expectations for square8, square19, and hex.
    - Parity suites (`TerritoryParity.GameEngine_vs_Sandbox.test.ts`, Seed 17 parity tests) remain green.

Any future changes to this ordering **must**:

- Be validated first against the sandbox oracle tests (`ClientSandboxEngine.territoryDisconnection.*`, `sandboxTerritory.*`), and
- Maintain green status for the backend Territory suites and parity checks listed above.

These semantics are the authoritative contract for later phases (AI parity, WebSocket/RulesBackendFacade integration, Python parity) and must remain stable unless intentionally revised alongside their tests.

---

## 6. Backend vs Sandbox Invariant Enforcement & Termination Ladder

This section documents where **board invariants** and the **termination ladder** are enforced across the backend GameEngine, client sandbox, and shared core. It also clarifies how the no-dead-placement and forced-elimination policies interact so that long AI sequences cannot stall while valid progress exists.

### 6.1 Board Invariant Enforcement (Backend vs Sandbox)

**Backend (server-side TS):**

- Core ownership: `src/server/game/BoardManager.ts`.
- Key responsibilities:
  - Maintain board geometry, stacks, markers, and collapsed Territories.
  - Provide mutators such as `setStack`, `removeStack`, `setMarker`, `removeMarker`, `collapseMarker`, `setCollapsedSpace`.
  - Discover disconnected regions and border markers.
- Invariant helper:

  ```ts
  // Pseudocode – see BoardManager for the exact implementation
  private assertBoardInvariants(board: BoardState, context: string): void {
    const errors: string[] = [];

    for (const key of board.stacks.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`stack present on collapsed space at ${key}`);
      }
      if (board.markers.has(key)) {
        errors.push(`stack and marker coexist at ${key}`);
      }
    }

    for (const key of board.markers.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`marker present on collapsed space at ${key}`);
      }
    }

    if (errors.length === 0) return;

    const message =
      `[BoardManager] invariant violation (${context}):` + '\n' + errors.join('\n');
    console.error(message);

    if (BOARD_INVARIANTS_STRICT) {
      throw new Error(message);
    }
  }
  ```

- Strictness:
  - Under `NODE_ENV === 'test'` or when `RINGRIFT_ENABLE_BACKEND_BOARD_INVARIANTS` is set, invariant violations **throw**.
  - In production, violations are logged but do not currently throw by default.
- Repair behaviour in hot mutators:
  - `setStack` removes any marker on the destination cell **before** placing the stack, then calls `assertBoardInvariants`.
  - `setCollapsedSpace` removes any stack or marker before marking the cell as collapsed, then calls `assertBoardInvariants`.
  - `collapseMarker` removes the marker and any stack on that cell, sets collapsed Territory, and then calls `assertBoardInvariants`.

**Sandbox (client-side TS):**

- Core ownership: `src/client/sandbox/ClientSandboxEngine.ts`.
- Invariant helper (test-only):

  ```ts
  (ClientSandboxEngine.prototype as any).assertBoardInvariants = function (
    this: ClientSandboxEngine,
    context: string
  ): void {
    const board: BoardState = (this as any).gameState.board as BoardState;
    const errors: string[] = [];

    for (const key of board.stacks.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`stack present on collapsed space at ${key}`);
      }
      if (board.markers.has(key)) {
        errors.push(`stack and marker coexist at ${key}`);
      }
    }

    for (const key of board.markers.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`marker present on collapsed space at ${key}`);
      }
    }

    if (errors.length === 0) return;

    const message =
      `ClientSandboxEngine invariant violation (${context}):` + '\n' + errors.join('\n');

    console.error(message);

    if (isTestEnv) {
      throw new Error(message);
    }
  };
  ```

- Usage:
  - Called from AI-simulation and debug harnesses after each AI turn.
  - Ensures sandbox-side logic never produces illegal stack/marker/Territory overlaps.
- Mutators that preserve invariants by construction:
  - Placement (`tryPlaceRings`) rejects placement on markers or collapsed spaces.
  - Line collapse (`collapseLineMarkers`) removes stacks and markers before collapsing to Territory.
  - `collapseMarker` in `ClientSandboxEngine` removes both marker and stack, then sets collapsed Territory.
  - Territory processing in `sandboxTerritory.ts` (historically via `sandboxTerritoryEngine.ts`, now removed) removes stacks/markers before collapsing region spaces and border markers.

Together, these guarantees mean that **any** invariant violation that does slip through is surfaced immediately in tests on both backend and sandbox paths, and repair logic in hot mutators ensures new overlaps are corrected before they become latent bugs.

### 6.2 Shared No-Dead-Placement Core

The no-dead-placement rule is implemented once in the shared core and then
wrapped by backend, sandbox, and Python engines.

- Core helper: `hasAnyLegalMoveOrCaptureFromOnBoard` in
  `src/shared/engine/core.ts`.
  - Inputs: `boardType`, `from` position, `player`, and a minimal
    `MovementBoardView` (valid-position, collapsed-space, stack + marker
    lookup).
  - Behaviour:
    - Checks for at least one legal **non-capture move** from the stack at
      `from`.
    - If none exist, checks for at least one legal **overtaking capture**
      from `from`.
    - Returns `true` iff some legal movement or capture exists.
- Backend usage (TypeScript):
  - `BoardManager` adapters expose a `MovementBoardView` over the server-side
    board.
  - `RuleEngine.validateRingPlacement` and related helpers call
    `hasAnyLegalMoveOrCaptureFromOnBoard` (directly or via shared placement
    aggregates) to enforce that placements do not create dead stacks.
- Sandbox usage (TypeScript):
  - `ClientSandboxEngine.hasAnyLegalMoveOrCaptureFrom` builds a
    `MovementBoardView` from the sandbox board and delegates to the shared core.
  - `ClientSandboxEngine.tryPlaceRings` uses this to gate placements.
  - `sandboxPlacement.enumerateLegalRingPlacements` uses the same check when
    generating candidate positions for sandbox and AI, either directly in its
    legacy path or indirectly via `validatePlacementOnBoard` +
    `PlacementContext`.
- Python usage:
  - `_create_hypothetical_board_with_placement` in
    `ai-service/app/game_engine.py` mirrors
    `createHypotheticalBoardWithPlacement` from `sandboxPlacement.ts`, taking
    a `BoardState`, position, player, and placement count and returning a
    hypothetical post-placement board.
  - `_has_any_movement_or_capture_after_hypothetical_placement` in
    `ai-service/app/game_engine.py` constructs a temporary `GameState` in the
    `MOVEMENT` phase, fixes `must_move_from_stack_key` to the placed stack,
    seeds a synthetic `place_ring` move into `move_history`, and then reuses
    `_get_movement_moves` and `_get_capture_moves` to answer the same
    question as `hasAnyLegalMoveOrCaptureFromOnBoard`: “does this placement
    leave at least one legal move or capture from the placed stack?”
  - `_get_ring_placement_moves` calls both helpers for each candidate
    placement position and count, only emitting `place_ring` moves where the
    hypothetical board passes the no-dead-placement check. This mirrors the
    TS-side `enumerateLegalRingPlacements` +
    `hasAnyLegalMoveOrCaptureFromOnBoard` pipeline and ensures both engines
    reject dead-stack placements near edges, collapsed spaces, and tall
    stacks.

These helpers together are the **single source of truth** for no-dead-
placement across backend, sandbox, and Python engines.

### 6.3 Termination Ladder & Sandbox AI Control Flow

The **rules-level termination ladder**, as captured in
`docs/supplementary/RULES_TERMINATION_ANALYSIS.md`, is:

1. If a player can **move**, they must move.
2. If they cannot move and cannot place, they must undergo **forced
   elimination**.
3. If they can place but cannot move, they must place and then move; that
   process must eventually exhaust rings in hand and on board.

The sandbox AI enforces this ladder using three main components:

- Shared core: `hasAnyLegalMoveOrCaptureFromOnBoard` (no-dead-placement).
- Sandbox AI policy: `src/client/sandbox/sandboxAI.ts`.
- Turn engine / forced elimination: `ClientSandboxEngine` turn helpers layered over shared `turnLogic.advanceTurnAndPhase`.

#### 6.3.1 Ring Placement Phase (Sandbox AI)

Key implementation: `maybeRunAITurnSandbox` in `src/client/sandbox/sandboxAI.ts`.

- When `currentPhase === 'ring_placement'`, the AI:
  1. Reads `ringsInHand` for the current player.
  2. If `ringsInHand <= 0`:
     - Computes `hasAnyActionFromStacks` using `hooks.getPlayerStacks` and
       `hooks.hasAnyLegalMoveOrCaptureFrom`.
     - If `hasAnyActionFromStacks` is true, applies an explicit
       `skip_placement` move via `hooks.applyCanonicalMove`, advancing into
       movement – mirroring backend `RuleEngine.validateSkipPlacement`.
     - Otherwise, calls `hooks.maybeProcessForcedEliminationForCurrentPlayer()`
       and logs diagnostics if elimination does not change state.
  3. If `ringsInHand > 0`:
     - Computes `placementCandidates` via
       `hooks.enumerateLegalRingPlacements(current.playerNumber)`.
     - For each candidate position, uses
       `hooks.createHypotheticalBoardWithPlacement` +
       `hooks.hasAnyLegalMoveOrCaptureFrom` to filter out placements that
       would create dead stacks.
     - Builds `place_ring` moves for each surviving candidate, respecting
       per-placement and per-player caps.
     - Optionally appends a `skip_placement` candidate when the player has
       legal moves from existing stacks.
     - Uses `chooseLocalMoveFromCandidates` (shared selector) to choose
       between `place_ring` and `skip_placement` in a deterministic,
       proportional way.
     - Applies the chosen move via `hooks.applyCanonicalMove`.

**Stall fixes introduced for the historical seed-18 bug:**

- `ringsInHand <= 0` in ring_placement is no longer treated as a silent
  no-op; the AI now **always** either:
  - emits a `skip_placement` move when moves from stacks exist, or
  - invokes forced elimination via `hooks.maybeProcessForcedEliminationForCurrentPlayer()`.
- Defensive logging and diagnostics ensure that if forced elimination fails
  to change state while the game remains active, the situation is surfaced in
  test logs rather than silently looping.

These changes guarantee that the sandbox AI cannot remain in ring_placement
with `ringsInHand <= 0` for many consecutive no-op turns.

#### 6.3.2 Forced Elimination & Turn Engine (Sandbox)

Key implementation: `ClientSandboxEngine.startTurnForCurrentPlayer` and
`ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer`, backed by
the shared `turnLogic.advanceTurnAndPhase` helper.

- Inputs:
  - `state: GameState` and per-turn `SandboxTurnState` (hasPlacedThisTurn, mustMoveFromStackKey).
  - Local helpers that expose `enumerateLegalRingPlacements`,
    `hasAnyLegalMoveOrCaptureFrom`, `getPlayerStacks`, `forceEliminateCap`,
    and `checkAndApplyVictory`.
- Behaviour (summarised from `ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayerInternal`):
  1. Compute stacks for the current player.
  2. If **no stacks**:
     - If `ringsInHand <= 0`: advance `currentPlayer` to the next player,
       reset per-turn state, and treat this as an eliminated turn.
     - If `ringsInHand > 0`: do **not** advance; the player may act again on
       a future `ring_placement` turn.
  3. If stacks **do exist**:
     - Compute whether any movement/capture is available:
       - If a must-move stack is tracked for movement, check that stack
         first; otherwise, scan all stacks.
     - Compute whether any legal ring placements exist that satisfy the
       no-dead-placement rule using `enumerateLegalRingPlacements` and cap checks.
  4. If the player has **any** legal action (move, capture, or
     no-dead-placement placement), return with `eliminated: false`.
  5. If they have **no** such actions, call `forceEliminateCap` to perform a
     cap elimination, rotate `currentPlayer` to the next seat, and reset per-turn
     state before the next turn starts.

These helpers are used both when starting a new turn and during movement
phases where a player might become completely blocked. Combined with the
shared turn sequencer, they enforce the rules-level condition
"cannot move and cannot place → must eliminate" consistently in the sandbox.

### 6.4 Termination-Oriented Tests & Diagnostics

Several test suites and harnesses work together to enforce the termination
ladder and detect stalls:

- **Single-seed debug harness:**
  - `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts` drives a single
    sandbox AI-vs-AI game for `square8` / `2p` / `seed=18`.
  - After each AI action, it:
    - Asserts S-invariant non-decrease (`computeProgressSnapshot`).
    - Tracks consecutive stagnant steps (unchanged state hash while active)
      and breaks early when a stall threshold is reached.
    - Calls `engineAny.assertBoardInvariants(...)` to enforce board
      exclusivity.
    - At failure time (pre-fix), logged rich diagnostics including
      `legalPlacements`, movement options on hypothetical boards, and a
      rolling `recentHistory` of S and resource snapshots.
  - Post-fix, this test now **terminates within the action budget** and the
    final `gameStatus` is not `active`.

- **Fuzz AI simulation harness:**
  - `tests/unit/ClientSandboxEngine.aiSimulation.test.ts` fuzzes across
    multiple board types and player counts.
  - For each scenario and seed, it:
    - Enforces global S-invariant non-decrease.
    - Detects stalls via a `MAX_STAGNANT` window (unchanged hashes while
      active) and logs diagnostics with `logAiDiagnostic`.
    - Requires each game to terminate within `MAX_AI_ACTIONS` AI actions.
  - This harness is gated behind `RINGRIFT_ENABLE_SANDBOX_AI_SIM=1` due to
    its cost but is the primary long-run stall detector.

- **Targeted regression test:**
  - `tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts` encodes the
    historical seed-18 stall as a fast, CI-friendly regression:
    - Replays the same `square8` / `2p` / `seed=18` setup with the shared LCG.
    - Calls `maybeRunAITurn` up to a modest `MAX_AI_ACTIONS`.
    - Tracks consecutive no-op AI turns (unchanged hash while active) and
      fails if a long no-op run is observed.
    - Asserts that the final `gameStatus` is **not** `active`.
  - This test ensures that any future change to sandbox AI ring_placement or
    forced-elimination control flow that would reintroduce a stall is caught
    quickly.

Together, these tests and diagnostics ensure that the **practical** behaviour
of the sandbox AI matches the theoretical termination guarantees in
`docs/supplementary/RULES_TERMINATION_ANALYSIS.md`.

---

## 7. References

- Sandbox Territory & elimination:
  - `src/client/sandbox/ClientSandboxEngine.ts`
  - `src/client/sandbox/sandboxTerritory.ts`
  - ~~`src/client/sandbox/sandboxTerritoryEngine.ts`~~ (historical; module removed after consolidation into shared Territory helpers and `ClientSandboxEngine` turn helpers)
  - `src/client/sandbox/sandboxElimination.ts`
- Shared engine:
  - [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36)
  - [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1)
  - [`CaptureMutator.ts`](src/shared/engine/mutators/CaptureMutator.ts:1)
  - [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts:35)
  - [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1)
  - [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts:51)
  - [`turnLogic.ts`](src/shared/engine/turnLogic.ts:132)
- Backend:
  - [`BoardManager.ts`](src/server/game/BoardManager.ts:1)
  - [`GameEngine.ts`](src/server/game/GameEngine.ts:1)
- Python Rules Engine:
  - [`board_manager.py`](ai-service/app/board_manager.py:1)
  - [`territory.py`](ai-service/app/rules/mutators/territory.py:1)
  - [`capture.py`](ai-service/app/rules/mutators/capture.py:1)
- Topology & geometry overview:
  - `docs/TOPOLOGY_MODES.md`
- Termination & sandbox AI:
  - `docs/supplementary/RULES_TERMINATION_ANALYSIS.md`
  - `src/client/sandbox/sandboxAI.ts`
  - `src/client/sandbox/ClientSandboxEngine.ts` (turn helpers)
  - `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts`
  - `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`
  - `tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts`
