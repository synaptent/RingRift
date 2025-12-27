> **Doc Status (2025-12-09): Active (derived, post-rollout)**
> **Role:** Orchestrator-first rollout and legacy rules shutdown plan for Track A.
>
> **Rollout Status (2025-12-09):** **Phase 4 – FSM Canonical (Orchestrator Authoritative)**.
> `ORCHESTRATOR_ADAPTER_ENABLED` is now hardcoded to `true` in `EnvSchema`, and the
> former `ORCHESTRATOR_ROLLOUT_PERCENTAGE` flag was removed during the Phase 3 cleanup
> (adapter is always 100%). **Shadow mode has been fully removed** – FSM is now the
> canonical game state orchestrator (RR-CANON compliance). The `RINGRIFT_RULES_MODE`
> schema now only accepts `ts` or `python` values; `shadow` is no longer valid.
> Soak tests show zero invariant violations across all board types.
>
> **SSoT alignment:** This document is a derived architectural and rollout plan over:
>
> - **Rules/invariants semantics SSoT:** `RULES_CANONICAL_SPEC.md`, `../rules/COMPLETE_RULES.md`, `../rules/COMPACT_RULES.md`, and the shared TypeScript rules engine under `src/shared/engine/**` plus v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`.
> - **Lifecycle/API SSoT:** `docs/architecture/CANONICAL_ENGINE_API.md` and shared TS/WebSocket types under `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
> - **TS↔Python parity & determinism SSoT:** `docs/rules/PYTHON_PARITY_REQUIREMENTS.md` and the TS and Python parity/determinism test suites.
>
> **Precedence:** This plan is never the source of truth for rules behaviour or lifecycle semantics. On any conflict with executable code, tests, or canonical rules/lifecycle docs, **code + tests win** and this document must be updated.

# Orchestrator Rollout and Legacy Rules Shutdown Plan

## 1. Purpose and Scope

This document defines the orchestrator-first rollout strategy and legacy rules shutdown blueprint for Track A (tasks P16.6.\*, P16.7, P16.8). It assumes:

- Rules semantics are single-sourced in the shared TS engine under `src/shared/engine/**` (helpers → aggregates → turn orchestrator → contracts) as documented in `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` and `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`.
- The Python rules engine under `ai-service/app/rules/**` is a parity and contract mirror, not an independent SSOT.
- Orchestrator adapters exist for both backend and sandbox hosts and are wired into CI and runbooks as described in [`../archive/historical/CURRENT_STATE_ASSESSMENT.md`](../archive/historical/CURRENT_STATE_ASSESSMENT.md), `WEAKNESS_ASSESSMENT_REPORT.md`, and `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`.

The goals of this plan are to:

1. Declare explicit SSOT and ownership boundaries for turn processing.
2. Inventory remaining legacy rules-related modules and paths.
3. Define a small number of concrete shutdown phases.
4. Specify gating tests, metrics, and rollback levers per phase.
5. Provide a concise overview consumable by Track A implementation tasks (P16.6.\*, P16.7, P16.8).

> **Historical reference:** The rollout tables and percentage-based steps below are preserved for context but are no longer executable as written. `ORCHESTRATOR_ROLLOUT_PERCENTAGE` was removed in Phase 3 and the adapter is always on; treat the percentage columns as historical log, not an operational lever.

### 1.1 Orchestrator flags and incident posture

At runtime, orchestrator routing is fixed (always on); remaining flags are
diagnostic/telemetry controls:

- `ORCHESTRATOR_ADAPTER_ENABLED` – master switch to enable/disable orchestrator adapters for new sessions. **Hardcoded to `true`** since Phase 3.
- `ORCHESTRATOR_ROLLOUT_PERCENTAGE` – **removed in Phase 3**; historically controlled gradual rollout (adapter is now always 100%).
- `ORCHESTRATOR_SHADOW_MODE_ENABLED` – **removed in Phase 4**; historically toggled shadow runs for comparison. FSM is now canonical.
- `RINGRIFT_RULES_MODE` – high‑level rules mode selector (allowed values: `ts`, `python` as per `src/server/config/env.ts`). **Note:** `shadow` value was removed in Phase 4 – FSM is now canonical.

> **Post-Phase 4 note:** Both `ORCHESTRATOR_ROLLOUT_PERCENTAGE` and `ORCHESTRATOR_SHADOW_MODE_ENABLED` are no longer honoured by the codebase. Shadow mode has been completely removed – FSM is the canonical game state orchestrator. The phased percentage tables and rollback steps below are preserved for historical context only.

**During incidents:**

- Treat these flags as **rules‑engine levers**, not general‑purpose mitigations:
  - If symptoms are clearly AI‑only (remote AI service down/slow/erroring) or infra‑related (timeouts, WebSocket saturation, host overload), leave orchestrator flags in the **orchestrator‑ON** posture and follow the AI and infra runbooks:
    - `docs/runbooks/AI_ERRORS.md`
    - `docs/runbooks/AI_PERFORMANCE.md`
    - `docs/runbooks/AI_FALLBACK.md`
    - `docs/runbooks/AI_SERVICE_DOWN.md`
    - `docs/runbooks/HIGH_LATENCY.md`, `docs/runbooks/SERVICE_DEGRADATION.md`
  - Only adjust `RINGRIFT_RULES_MODE` when there is strong evidence of a **rules‑engine or orchestrator defect** (e.g. canonical contract tests failing, `.shared` suites red, or explicit violation of `RULES_CANONICAL_SPEC.md`), and then follow the Safe rollback flow in this document. Note: `ORCHESTRATOR_ADAPTER_ENABLED` is hardcoded to `true`, and `ORCHESTRATOR_SHADOW_MODE_ENABLED` has been removed – FSM is now canonical.
- See `AI_ARCHITECTURE.md` §0 (AI Incident Overview) for a quick “rules vs AI vs infra” classification, and use that to choose between **this plan** (rules/orchestrator rollback) and the AI/infra runbooks above.

## 2. Canonical SSOT and Ownership Boundaries

### 2.1 Turn-processing entrypoints

For host-driven turn processing and rules-surface queries, the canonical orchestrator APIs into the shared TS rules engine are:

- `processTurnAsync(state, move, delegates)` in [`turnOrchestrator.ts`](../../src/shared/engine/orchestration/turnOrchestrator.ts:1) – **canonical host-facing entrypoint** for applying moves.
- `processTurn(state, move)` in [`turnOrchestrator.ts`](../../src/shared/engine/orchestration/turnOrchestrator.ts:1) – synchronous helper used where decisions can be resolved inline.
- `validateMove(state, move)`, `getValidMoves(state)`, and `hasValidMoves(state)` in [`turnOrchestrator.ts`](../../src/shared/engine/orchestration/turnOrchestrator.ts:1) – canonical validation and enumeration helpers for hosts and diagnostics harnesses.

All host stacks (backend `GameEngine`, client sandbox `ClientSandboxEngine`, diagnostics harnesses) **must** treat `processTurnAsync` and these helpers as the lifecycle and rules-surface SSOT for turn processing. Legacy turn loops in `GameEngine` and `ClientSandboxEngine` are treated as migration scaffolding to be removed or demoted by this plan.

### 2.2 Domain aggregates as rules semantics SSOT

The following domain aggregates under `src/shared/engine/aggregates/**` are the **single source of truth for rules semantics** in their respective domains:

- [`MovementAggregate`](../../src/shared/engine/aggregates/MovementAggregate.ts:1) – non-capturing movement validation, enumeration, and mutation.
- [`CaptureAggregate`](../../src/shared/engine/aggregates/CaptureAggregate.ts:1) – capture and chain-capture validation, enumeration, mutation, and continuation logic.
- [`PlacementAggregate`](../../src/shared/engine/aggregates/PlacementAggregate.ts:1) – placement and no-dead-placement validation, enumeration, and mutation.
- [`RecoveryAggregate`](../../src/shared/engine/aggregates/RecoveryAggregate.ts:1) – recovery eligibility, enumeration, and mutation for temporarily eliminated players.
- [`LineAggregate`](../../src/shared/engine/aggregates/LineAggregate.ts:1) – line detection and decision moves via `enumerateProcessLineMoves` and `applyProcessLineDecision`.
- [`TerritoryAggregate`](../../src/shared/engine/aggregates/TerritoryAggregate.ts:1) – disconnected-region detection, Q23 gating, territory collapse, and elimination decisions.
- [`EliminationAggregate`](../../src/shared/engine/aggregates/EliminationAggregate.ts:1) – ring elimination semantics for line, territory, and forced elimination contexts.
- [`VictoryAggregate`](../../src/shared/engine/aggregates/VictoryAggregate.ts:1) – victory evaluation and tie-breaking, surfaced via `evaluateVictory`.

All hosts and helpers (backend, sandbox, Python mirror, diagnostics scripts) **must** treat these aggregates and their helper modules (`movementLogic.ts`, `captureLogic.ts`, `lineDecisionHelpers.ts`, `territoryDecisionHelpers.ts`, `VictoryAggregate.ts`, etc.) as the **only authoritative implementation of placement, movement, recovery, capture, line, territory, elimination, and victory semantics**.

### 2.3 Host adapters and hosts

The orchestrator integration layers are:

- Backend adapter: [`TurnEngineAdapter`](../../src/server/game/turn/TurnEngineAdapter.ts:1).
- Sandbox adapter: [`SandboxOrchestratorAdapter`](../../src/client/sandbox/SandboxOrchestratorAdapter.ts:1).

These adapters:

- Own all calls from hosts into `processTurnAsync` / `processTurn`.
- Bridge host-specific concerns (state mutability, timers, WebSocket notifications, AI interaction, diagnostics) with the pure shared engine.
- Provide validation/enumeration shims (`validateMove`, `getValidMoves`) that directly delegate to the orchestrator.

**Exclusive integration rule**

For orchestrator-driven turn processing, **all** production and sandbox hosts **must** integrate with the shared rules engine **only via**:

- `TurnEngineAdapter` on the backend.
- `SandboxOrchestratorAdapter` in the sandbox.

Direct calls from hosts into aggregate helpers are allowed for:

- Pure read-side diagnostics and tooling (e.g. board visualisations, offline analysis).
- Test harnesses that deliberately exercise the core engine.

They are **not** allowed as alternative production turn-processing pipelines once this plan is complete.

### 2.4 Host stacks and Python mirror

- Backend host stack:
  - `GameEngine` in [`GameEngine.ts`](../../src/server/game/GameEngine.ts:1) – stateful backend host responsible for timers, WebSocket integration, rating updates, and structured history. It currently has:
    - An orchestrator path via `processMoveViaAdapter` + `TurnEngineAdapter`.
    - A legacy path via `makeMove` and internal phase loops that apply moves via shared aggregates and helpers.
  - `RuleEngine` in [`RuleEngine.ts`](../../src/server/game/RuleEngine.ts:1) – rules-facing validation and enumeration surface used by `GameEngine` and `TurnEngine`. It delegates to shared helpers and aggregates but still exposes an older `processMove` pipeline.
  - `TurnEngine` in [`turn/TurnEngine.ts`](../../src/server/game/turn/TurnEngine.ts:1) – shared backend turn/phase lifecycle for the backend path, already aligned with shared `turnLogic`.

- Sandbox host stack:
  - `ClientSandboxEngine` in [`ClientSandboxEngine.ts`](../../src/client/sandbox/ClientSandboxEngine.ts:1) – client-local host for `/sandbox`, with:
    - An orchestrator path via `processMoveViaAdapter` + `SandboxOrchestratorAdapter`.
    - A legacy sandbox pipeline composed out of shared aggregates and sandbox helpers (movement, capture, lines, territory, forced elimination, LPS, and victory).
  - Sandbox helpers under `src/client/sandbox/**` (movement, captures, territory, lines, victory, AI, game-end) – now predominantly UX/diagnostics wrappers over shared aggregates.

- Python rules/AI mirror:
  - `GameEngine` in [`ai-service/app/game_engine/__init__.py`](../../ai-service/app/game_engine/__init__.py:1) and rule modules under `ai-service/app/rules/**` implement a parity-checked Python port of the TS rules engine.
  - Parity and contract tests under `ai-service/tests/**` and `tests/parity/**` validate that Python behaviour matches the TS SSOT.
  - Python code is **not** a semantics SSOT; any divergence must be fixed by updating Python to match the TS shared engine and contracts.

## 3. Architecture Overview

```mermaid
flowchart TD
  subgraph SharedEngine
    Orchestrator[Turn orchestrator processTurnAsync]
    Aggregates[Domain aggregates placement movement recovery capture line territory elimination victory]
  end

  subgraph BackendHost
    RulesFacade[RulesBackendFacade]
    GameEngineNode[GameEngine backend host]
    TurnAdapter[TurnEngineAdapter]
  end

  subgraph SandboxHost
    SandboxEngine[ClientSandboxEngine sandbox host]
    SandboxAdapter[SandboxOrchestratorAdapter]
  end

  subgraph PythonRules
    PyEngine[Python rules engine and game engine]
    PyParity[Python parity and contract tests]
  end

  Client[React client and WebSocket server]

  RulesFacade --> GameEngineNode
  GameEngineNode --> TurnAdapter
  TurnAdapter --> Orchestrator
  SandboxEngine --> SandboxAdapter
  SandboxAdapter --> Orchestrator
  Orchestrator --> Aggregates
  Aggregates --> Orchestrator

  RulesFacade --> PyEngine
  PyEngine --> PyParity

  GameEngineNode --> Client
  SandboxEngine --> Client
```

Key properties:

- The **SharedEngine** subgraph (orchestrator + aggregates + helpers) is the **rules semantics SSOT**.
- `TurnEngineAdapter` and `SandboxOrchestratorAdapter` are the **only sanctioned host integration layers** into `processTurnAsync`.
- Backend and sandbox hosts are responsible only for state ownership, player interaction, transport, and diagnostics.
- Python rules/AI remain a validated mirror and do not own semantics.

## 4. Legacy Rules Surfaces Inventory

This section enumerates remaining TS backend and sandbox modules that either:

- Implement or previously implemented movement, capture, placement, line, territory, or victory semantics **outside** the orchestrator+aggregates stack; or
- Provide post-processing over older `RuleEngine` / `GameEngine` / sandbox semantics that are now redundant.

### 4.1 Backend host and rules modules

**Table 1 – Backend rules-related modules and paths**

| File                                                | Current role                                                                                                                                                                                                                                                       | Rules Semantics? (Y/N)                                                                                                  | Remove?                                                                                                                                                                   | Diagnostics-only?                                                              | Target phase                                                                            |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `src/server/game/GameEngine.ts`                     | Stateful backend host; currently has dual path: legacy `makeMove` turn pipeline and orchestrator-based `processMoveViaAdapter` using `TurnEngineAdapter`. Contains chain-capture state, line and territory decision helpers, forced elimination, and LPS tracking. | Y – host-level orchestration semantics layered over aggregates and `TurnEngine`.                                        | Partial – keep GameEngine as backend host but **remove the legacy non-adapter pipeline** and require `TurnEngineAdapter` for all production moves.                        | No – remains primary backend host.                                             | Phase A (enforce adapter) and Phase C (delete dead helpers once adapter-only)           |
| `src/server/game/RuleEngine.ts`                     | Stateless rules facade for validation and enumeration. Delegates to shared aggregates and helpers but still exposes legacy `processMove`, `processLineFormation`, and `processTerritoryDisconnection` flows.                                                       | Y – contains historical orchestration and some pre-aggregate helpers, though main semantics now delegate to aggregates. | Partial – retain validation/enumeration entrypoints, but **delete or quarantine legacy `processMove` and post-processing helpers** once orchestrator-only path is stable. | Yes for legacy helpers – legacy helpers become diagnostics-only until removal. | Phase A (no production calls to `processMove`), Phase C (delete or move to diagnostics) |
| `src/server/game/turn/TurnEngine.ts`                | Shared backend turn lifecycle built on `turnLogic` and shared aggregates. Used by `GameEngine.advanceGame`.                                                                                                                                                        | N – lifecycle orchestration only (semantics live in shared engine).                                                     | No.                                                                                                                                                                       | No.                                                                            | N/A (already canonical host lifecycle)                                                  |
| `src/server/game/BoardManager.ts`                   | Backend board container and geometry bridge. Used by `GameEngine`, `RuleEngine`, and `TurnEngine`.                                                                                                                                                                 | N – uses shared core geometry and does not define independent rules.                                                    | No.                                                                                                                                                                       | No.                                                                            | N/A                                                                                     |
<<<<<<< Updated upstream
| `src/server/game/rules/lineProcessing.ts`           | Historical backend line-processing module referenced in earlier passes. Marked as removed in [`docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`](../rules/RULES_ENGINE_SURFACE_AUDIT.md:105).                                                                                   | Y (historical)                                                                                                          | Already removed (no current code path).                                                                                                                                   | N/A                                                                            | Historical only                                                                         |
| `src/server/game/rules/territoryProcessing.ts`      | Historical backend territory-processing module. Marked as removed in [`docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`](../rules/RULES_ENGINE_SURFACE_AUDIT.md:106).                                                                                                           | Y (historical)                                                                                                          | Already removed.                                                                                                                                                          | N/A                                                                            | Historical only                                                                         |
=======
| `src/server/game/rules/lineProcessing.ts`           | Historical backend line-processing module referenced in earlier passes. Marked as removed in [`docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`](../rules/RULES_ENGINE_SURFACE_AUDIT.md:105).                                                                             | Y (historical)                                                                                                          | Already removed (no current code path).                                                                                                                                   | N/A                                                                            | Historical only                                                                         |
| `src/server/game/rules/territoryProcessing.ts`      | Historical backend territory-processing module. Marked as removed in [`docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`](../rules/RULES_ENGINE_SURFACE_AUDIT.md:106).                                                                                                     | Y (historical)                                                                                                          | Already removed.                                                                                                                                                          | N/A                                                                            | Historical only                                                                         |
>>>>>>> Stashed changes
| `src/server/game/rules/captureChainEngine.ts`       | Historical backend capture-chain state helper. Replaced by `CaptureAggregate` and `GameEngine` wiring.                                                                                                                                                             | Y (historical)                                                                                                          | Already removed.                                                                                                                                                          | N/A                                                                            | Historical only                                                                         |
| `src/server/game/RulesBackendFacade.ts`             | Backend rules/AI boundary; selects TS vs Python authority and coordinates runtime parity checks via `rulesParityMetrics` when `RINGRIFT_RULES_MODE=python`.                                                                                                        | N – engine selection and parity only.                                                                                   | No.                                                                                                                                                                       | No.                                                                            | N/A                                                                                     |
| `src/server/services/OrchestratorRolloutService.ts` | Backend service tracking circuit-breaker state and selection reasons (allow/deny lists) for metrics and `/api/admin/orchestrator/status`. Routing is fixed; rollout/shadow flags are historical.                                                                   | N – diagnostics/metrics only.                                                                                           | No.                                                                                                                                                                       | No.                                                                            | N/A                                                                                     |

**Backend summary**

- The only substantive _legacy semantics_ remaining live in:
  - The non-adapter branch of `GameEngine.makeMove` and associated helpers (`applyMove`, `processLineFormations`, `processOneLine`, `processOneDisconnectedRegion`, legacy forced-elimination and LPS wiring).
  - Legacy orchestration helpers in `RuleEngine` that predate `CaptureAggregate`, `LineAggregate`, and `TerritoryAggregate`.

- All direct geometry and mutator logic now delegates to shared helpers and aggregates as recorded in `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` and `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`.

### 4.2 Sandbox host and helper modules

**Table 2 – Sandbox rules-related modules and paths**

| File                                               | Current role                                                                                                                                                                                                                                               | Rules Semantics? (Y/N)                                                                                       | Remove?                                                                                                                                                         | Diagnostics-only?                                                 | Target phase                                                                                  |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `src/client/sandbox/ClientSandboxEngine.ts`        | Client-local sandbox host. Supports two paths: orchestrator-based via `SandboxOrchestratorAdapter` and legacy sandbox pipeline using shared aggregates plus sandbox helpers for movement, capture, lines, territory, forced elimination, LPS, and victory. | Y – host-level orchestration semantics layered over aggregates.                                              | Partial – retain as sandbox host but **remove or fence legacy pipeline** once orchestrator path is validated; orchestrator becomes the only sandbox rules path. | Legacy pipeline becomes diagnostics-only until removed.           | Phase B (orchestrator-only) and Phase C (remove dead helpers)                                 |
| `src/client/sandbox/SandboxOrchestratorAdapter.ts` | Sandbox adapter over `processTurn` / `processTurnAsync`; exposes `processMove`, `processMoveSync`, `previewMove`, `validateMove`, and `getValidMoves`.                                                                                                     | N – pure adapter over orchestrator.                                                                          | No.                                                                                                                                                             | No.                                                               | N/A (canonical adapter)                                                                       |
| `src/client/sandbox/sandboxMovement.ts`            | Sandbox movement helpers: simple movement landing enumeration and marker-path effects, now delegating to `MovementAggregate` and shared marker helpers. Used by `ClientSandboxEngine` and AI helpers.                                                      | N (semantics delegated to shared engine; acts as adapter and UX helper).                                     | No immediate removal; refactor to explicit UX/diagnostics namespace.                                                                                            | Yes – treated as UX/diagnostics-only helper, not a rules surface. | Phase B (ensure no direct semantics) and Phase C (namespace / banner update)                  |
| `src/client/sandbox/sandboxCaptures.ts`            | Sandbox capture helpers and board-level chain enumeration. Delegates to `CaptureAggregate` and shared helpers, with a retained mutable-board simulator for diagnostics (`applyCaptureSegmentOnBoard`).                                                     | N for live engine semantics (delegates to aggregate); Y for the mutable-board simulator used in diagnostics. | Keep as diagnostics-only; ensure live engine paths use `CaptureAggregate` exclusively.                                                                          | Yes – diagnostics-only (chain search, parity tooling).            | Phase B (engine paths fully adapter-based) and Phase C (diagnostics fencing and SSOT banners) |
| `src/client/sandbox/sandboxCaptureSearch.ts`       | Offline DFS search for maximal capture chains using shared capture enumeration plus sandbox-local mutation for analysis. Not used by live sandbox engine or backend.                                                                                       | Y (analysis semantics only; not a production rules path).                                                    | Keep; move under diagnostics namespace if desired.                                                                                                              | Yes – diagnostics-only.                                           | Phase C                                                                                       |
| `src/client/sandbox/sandboxTerritory.ts`           | Sandbox helpers for disconnected-region discovery and eligibility. Already delegates to shared `territoryDetection`, `territoryProcessing`, and `territoryDecisionHelpers` and is explicitly documented as thin adapter.                                   | N – adapter/visualisation on top of shared semantics.                                                        | Keep; ensure clearly marked as adapter-only.                                                                                                                    | Yes – UX/diagnostics-only.                                        | Phase B and C                                                                                 |
| `src/client/sandbox/sandboxLines.ts`               | Sandbox line-detection helpers; delegates to shared `lineDetection` and is documented as adapter-only.                                                                                                                                                     | N – adapter/visualisation only.                                                                              | Keep; ensure clearly marked as adapter-only.                                                                                                                    | Yes – UX/diagnostics-only.                                        | Phase B and C                                                                                 |
| `src/client/sandbox/sandboxVictory.ts`             | Sandbox wrapper around shared victory evaluation for local games.                                                                                                                                                                                          | N – adapter only.                                                                                            | Keep.                                                                                                                                                           | No.                                                               | N/A                                                                                           |
| `src/client/sandbox/sandboxElimination.ts`         | Sandbox forced-elimination helpers for local games, built on shared helpers.                                                                                                                                                                               | N – host-level orchestration only.                                                                           | Keep; ensure semantics remain aligned with shared aggregates.                                                                                                   | No.                                                               | N/A                                                                                           |
| `src/client/sandbox/sandboxGameEnd.ts`             | Sandbox game-end utilities and stalemate resolution, delegating to shared helpers.                                                                                                                                                                         | N – host-level orchestration only.                                                                           | Keep.                                                                                                                                                           | No.                                                               | N/A                                                                                           |
| `src/client/sandbox/sandboxAI.ts`                  | Local sandbox AI harness for human vs AI sandbox games. Uses shared engine helpers and sandbox movement/capture adapters.                                                                                                                                  | N – AI/UX only.                                                                                              | Keep; may be refactored but not part of rules semantics SSOT.                                                                                                   | No.                                                               | N/A                                                                                           |
| `src/client/sandbox/localSandboxController.ts`     | Minimal, browser-safe local sandbox harness with a very small, experimental rule subset (ring placement + simple movement) independent of `ClientSandboxEngine`.                                                                                           | Y – limited semantics outside orchestrator+aggregates.                                                       | Yes – once `ClientSandboxEngine` orchestrator path is stable and `/sandbox` no longer depends on this harness.                                                  | Yes – treat as legacy/diagnostics-only until removal.             | Phase C                                                                                       |
| `src/client/sandbox/test-sandbox-parity-cli.ts`    | CLI parity harness used for diagnostics, built on `ClientSandboxEngine` and shared engine.                                                                                                                                                                 | N – diagnostics tooling only.                                                                                | Keep; ensure clearly marked as diagnostics-only.                                                                                                                | Yes.                                                              | Phase C                                                                                       |

**Sandbox summary**

- All sandbox helpers that once contained independent rules logic have been refactored to delegate to shared helpers and aggregates.
- Remaining risk lies in the **legacy sandbox orchestration pipeline** in `ClientSandboxEngine` and the implicit assumption that some helpers are still “engine-like” rather than pure UX/diagnostics.
- This plan formalises the orchestrator adapter as the only production rules path for sandbox, demoting the legacy pipeline and analysis helpers to diagnostics-only status.

### 4.3 Diagnostics scripts and tooling

Some scripts under `scripts/` use shared engine helpers to explore or debug rules behaviour (for example, [`scripts/findCyclicCaptures.js`](../../scripts/findCyclicCaptures.js:1) and [`scripts/findCyclicCapturesHex.js`](../../scripts/findCyclicCapturesHex.js:1)). These scripts:

- Do **not** implement independent semantics.
- Are explicitly documented as analysis tools.
- Should remain available but clearly fenced as diagnostics-only.

They are not considered legacy rules _paths_ for the purposes of rollback; they are covered by **Phase C – Legacy helper shutdown & diagnostics fencing** purely for namespace and SSOT-banner hygiene.

## 5. Shutdown Phases

The remaining orchestrator rollout and legacy shutdown work is organised into four coarse-grained phases that can be implemented in 1–2 passes each by Code, QA, and DevOps agents.

> **Historical note:** These phases have been completed. Rollout-percentage and
> shadow-mode levers were removed; the content below is retained for context and
> should not be executed as a live operational playbook.

### 5.1 Phase overview table

**Table 3 – Rollout phases**

| Phase                                                      | Description                                                                                                                                                                        | Modules in scope (examples)                                                                                                                                                                                                                                                     | Required tests & metrics (gates)                                                                                                                                                                                                                                                                                                                                   | Rollback levers                                                                                                                                                                                                             |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase A – Backend orchestrator-only path**               | Make `TurnEngineAdapter` + `processTurnAsync` the only production backend turn path. Legacy `GameEngine` and `RuleEngine` pipelines remain only as test harnesses and diagnostics. | `GameEngine.ts` (non-adapter branch of `makeMove` and decision loops), `RuleEngine.ts` legacy `processMove` and post-processing helpers, `TurnEngineAdapter.ts`, `RulesBackendFacade.ts`, `OrchestratorRolloutService.ts`.                                                      | TS: full `test:core`, rules scenario suites (`tests/scenarios/**`), backend parity suites (`tests/unit/Backend_vs_Sandbox.*`), adapter tests. Python: all `ai-service/tests/**` including parity/contract tests. Metrics: orchestrator error rate < 2%, runtime python parity mismatches near zero when diagnostics run, `game_move_latency_ms` meeting v1.0 SLOs. | Rollback: deployment rollback only; production stays `RINGRIFT_RULES_MODE=ts` with `python` reserved for diagnostics.                                                                                                       |
| **Phase B – Sandbox orchestrator-only path**               | Make `SandboxOrchestratorAdapter` + `processTurnAsync` the only sandbox rules path. Legacy sandbox pipeline remains only in trace/parity harnesses.                                | `ClientSandboxEngine.ts` legacy pipeline and post-movement helpers, `SandboxOrchestratorAdapter.ts`, sandbox helpers (`sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxLines.ts`, `sandboxTerritory.ts`, `sandboxGameEnd.ts`, `sandboxElimination.ts`, `sandboxVictory.ts`). | TS: sandbox unit tests (`tests/unit/ClientSandboxEngine.*.test.ts`), RulesMatrix sandbox scenarios, backend vs sandbox parity tests, orchestrator adapter tests. Metrics: local sandbox parity vs backend on trace seeds, no regressions in `tests/scenarios/RulesMatrix.*`, no change in CLI parity harness behaviour.                                            | Flag: `ClientSandboxEngine.useOrchestratorAdapter` default `true`, with explicit opt-out preserved only for diagnostics. Git: Phase B changes split from Phase A so sandbox-only regressions can be reverted independently. |
| **Phase C – Legacy helper shutdown & diagnostics fencing** | Remove truly redundant modules and move any remaining tools into a diagnostics namespace with explicit SSOT banners.                                                               | Backend legacy helpers in `RuleEngine.ts` and `GameEngine.ts` that are no longer referenced; sandbox helpers that are pure analysis (`sandboxCaptureSearch.ts`, mutable-board simulators); diagnostics scripts under `scripts/`.                                                | TS: any tests that directly reference legacy helpers either updated to call shared aggregates or moved under an `archive/` diagnostics suite. Contract and parity tests remain green. No new references to deprecated helpers appear in `eslint-report.json` or `scripts/ssot/rules-ssot-check.ts`.                                                                | Git: removal commits grouped per host (backend vs sandbox) so reverts are straightforward. No feature flags required; by this point production paths are already orchestrator-only.                                         |
<<<<<<< Updated upstream
| **Phase D – Final clean-up and documentation alignment**   | Align documentation and runbooks with orchestrator-only architecture; ensure SSOT checks cover this plan.                                                                          | `docs/architecture/CANONICAL_ENGINE_API.md`, `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`, `docs/PASS16_ASSESSMENT_REPORT.md`, `docs/INDEX.md`, `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`, `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`.                                                         | Docs: updated SSOT banners and diagrams; cross-links from PASS16 and the docs index to this plan. Scripts: `scripts/ssot/rules-ssot-check.ts` extended to assert that backend and sandbox hosts reference the orchestrator and aggregates as SSOT. All tests and metrics remain green as in previous phases.                                                       | Git: doc-only commits, easily reversible but low risk. No feature flags.                                                                                                                                                    |
=======
| **Phase D – Final clean-up and documentation alignment**   | Align documentation and runbooks with orchestrator-only architecture; ensure SSOT checks cover this plan.                                                                          | `docs/architecture/CANONICAL_ENGINE_API.md`, `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`, `docs/PASS16_ASSESSMENT_REPORT.md`, `docs/INDEX.md`, `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`, `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`.                                      | Docs: updated SSOT banners and diagrams; cross-links from PASS16 and the docs index to this plan. Scripts: `scripts/ssot/rules-ssot-check.ts` extended to assert that backend and sandbox hosts reference the orchestrator and aggregates as SSOT. All tests and metrics remain green as in previous phases.                                                       | Git: doc-only commits, easily reversible but low risk. No feature flags.                                                                                                                                                    |
>>>>>>> Stashed changes

### 5.2 Phase A – Backend orchestrator-only path

**Concrete code-level actions**

- Enforce `TurnEngineAdapter` for all production backend moves:
  - Update `GameEngine.makeMove` to:
    - Consult `OrchestratorRolloutService` for circuit-breaker state and allow/deny list tagging (diagnostics only).
    - Route all non-test moves through `processMoveViaAdapter`, with legacy path reserved only for test harnesses (guarded by `NODE_ENV === 'test'` or explicit override).

- Quarantine and then deprecate legacy `RuleEngine` pipelines:
  - Identify all remaining call sites of `RuleEngine.processMove`, `processLineFormation`, `processTerritoryDisconnection`, and similar legacy helpers.
  - Replace those call sites with orchestrator adapter calls or direct aggregate usage where appropriate.
  - Mark legacy helpers as deprecated with SSOT banners pointing to shared aggregates and orchestrator, and add TODOs assigning them to **Phase C** removal.

- Tighten backend testing around the adapter:
  - Ensure that existing backend tests for lines, territory, and LPS (`tests/unit/GameEngine.lines.scenarios.test.ts`, `tests/unit/BoardManager.territoryDisconnection.test.ts`, etc.) run with `ORCHESTRATOR_ADAPTER_ENABLED=true` and `RINGRIFT_RULES_MODE=ts`.
  - Add or extend adapter-focused tests if needed to cover:
    - Capture chains including continuation and termination.
    - Line processing including overlength line rewards.
    - Territory disconnection including Q23 self-elimination prerequisite.
    - Last-player-standing (R172) scenarios.

**Gating tests and metrics**

- All TS tests passing:
  - Unit and scenario tests under `tests/unit/**` and `tests/scenarios/**`.
  - Backend vs sandbox parity tests.
  - Contract tests (`tests/contracts/**` and TS-side contract runners).

- All Python tests passing:
  - `ai-service/tests/**`, including parity suites such as `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`.

- Operational gates:
  - No regressions in `game_move_latency_ms` and WebSocket error rates as described in `docs/runbooks/GAME_HEALTH.md`.
  - Orchestrator rollout metrics from `OrchestratorRolloutService`:
    - `ringrift_orchestrator_error_rate < 0.02`.
    - Runtime python parity mismatches near zero when running `RINGRIFT_RULES_MODE=python`.

**Rollback**

- Immediate rollback:
  - Use a deployment rollback to a known-good build; runtime flags no longer change routing.

- Phase rollback:
  - Keep all adapter-enforcement changes in clearly labelled commits (e.g. `P16.6.1 backend orchestrator-first`) so that `git revert` restores the previous behaviour if necessary.

### 5.3 Phase B – Sandbox orchestrator-only path

**Concrete code-level actions**

- Make `SandboxOrchestratorAdapter` the default and only rules path for sandbox:
  - Ensure `ClientSandboxEngine.useOrchestratorAdapter` defaults to `true` across all environments (already true in current code).
  - Tighten sandbox entrypoints so that:
    - Canonical move application (`applyCanonicalMove`, AI turns, parity harnesses) route exclusively through `processMoveViaAdapter`.
    - Legacy sandbox movement/capture/territory pipelines are used only in explicitly marked trace or diagnostics modes.

- Demote sandbox helpers to UX and diagnostics:
  - Review `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxCaptureSearch.ts`, `sandboxLines.ts`, `sandboxTerritory.ts`, `sandboxGameEnd.ts`, `sandboxElimination.ts`, and `sandboxVictory.ts`:
    - Confirm all live semantics delegate to shared aggregates and helpers.
    - Add SSOT banners marking these modules as UX/diagnostics-only, deferring to shared TS engine for rules semantics.
    - Ensure no new callers treat these helpers as an alternative engine (SSOT tooling like `rules-ssot-check.ts` can be extended to enforce this).

**Gating tests and metrics**

- All sandbox-focused TS tests passing:
  - `tests/unit/ClientSandboxEngine.*.test.ts` suites (lines, territory, LPS).
  - RulesMatrix sandbox scenarios (territory, chain capture, late-game flows).
  - Backend vs sandbox parity tests and orchestrator adapter tests.

- Operational checks:
  - No regressions in local sandbox behaviour observed via `test-sandbox-parity-cli.ts`.
  - CI jobs that exercise sandbox-only flows (if present) run with `useOrchestratorAdapter=true`.

**Rollback**

- Immediate rollback:
  - Expose a constructor or configuration option to disable orchestrator adapter in `ClientSandboxEngine` for local diagnostics or emergency rollback.
  - For production-style sandbox builds, toggle the same flag via environment variable (mirroring backend `ORCHESTRATOR_ADAPTER_ENABLED`).

- Phase rollback:
  - Keep sandbox orchestrator-enforcement refactors in dedicated commits (e.g. `P16.6.2 sandbox orchestrator-first`) so that `git revert` can restore the previous mixed-path behaviour without touching backend.

### 5.4 Phase C – Legacy helper shutdown and diagnostics fencing

**Concrete code-level actions**

- Backend:
  - Once Phase A is stable and all production call sites use `TurnEngineAdapter`:
    - Delete or archive legacy helpers in `RuleEngine.ts` that are no longer referenced (e.g. `processMove`, `processLineFormation`, `processTerritoryDisconnection`, internal ray-walk helpers).
    - Remove or archive any remaining `src/server/game/rules/**` modules that duplicate aggregate behaviour (most are already removed, as documented in `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`).

- Sandbox:
  - Move pure analysis helpers (not used by `ClientSandboxEngine` in orchestrator mode) into a dedicated diagnostics namespace (for example, `src/client/sandbox/diagnostics/**`):
    - Chain search and maximal-chain exploration in `sandboxCaptureSearch.ts`.
    - Any residual mutable-board simulators.

  - Add SSOT banners to diagnostics modules making clear that they are **not** rules semantics SSOT; they interpret the shared engine for analysis only.

- Tooling:
  - Update `scripts/ssot/rules-ssot-check.ts` to:
    - Assert that no production code imports removed legacy modules.
    - Enforce that diagnostics modules carry the correct SSOT banners and are not imported from server or client runtime entrypoints.

**Gating tests and metrics**

- All TS and Python tests stay green with no changes to semantics.
- CI reports no imports from deleted legacy modules.
- `rulesParityMetrics` and orchestrator rollout metrics are stable, indicating that helper removal did not alter behaviour.

**Rollback**

- Phase C is inherently low-risk if executed after Phases A and B:
  - All changes are deletions or namespace moves guarded by tests and SSOT scripts.
  - If necessary, individual helper deletions can be reverted using `git revert` without reintroducing them into production code paths.

### 5.5 Phase D – Final clean-up and documentation alignment

**Concrete code-level and documentation actions**

- Update documentation to reflect orchestrator-only architecture:
  - `docs/architecture/CANONICAL_ENGINE_API.md` – ensure turn lifecycle sections describe `processTurnAsync` as the only host entrypoint and update placement helper language to reflect that helpers and `PlacementAggregate` are now canonical and production-backed.
  - `docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` – mark backend and sandbox consolidation phases as complete; update duplicate-line estimates; point to this plan for legacy shutdown status.
  - `docs/PASS16_ASSESSMENT_REPORT.md` – in Section 8 (Remediation Roadmap), reference this plan as the authoritative blueprint for P16.6.\*, P16.7, P16.8.
  - `docs/INDEX.md` – add a link to `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` under the Canonical Orchestrator or architecture sections.

- Tighten SSOT checks and runbooks:
  - Extend `scripts/ssot/rules-ssot-check.ts` and related checks to cover:
    - Orchestrator ownership boundaries (adapters and aggregates as SSOT).
    - Document banners on this plan and on key rules docs.

  - Update `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md` to:
    - Reference the final steady-state where legacy paths are removed.
    - Document that CI and staging always run with `ORCHESTRATOR_ADAPTER_ENABLED=true` and `RINGRIFT_RULES_MODE=ts` (rollout percentage fixed at 100).

**Gating tests and metrics**

- No new behavioural gates beyond those for Phases A–C.
- Documentation checks (`scripts/ssot/docs-banner-ssot-check.ts`) and rules SSOT checks pass in CI.

**Rollback**

- Documentation-only changes; rollback is straightforward via `git revert` but unlikely to be needed.

## 6. SLOs, tests, metrics, and rollback summary

This section defines orchestrator‑specific SLOs and error budgets and then restates
the supporting tests, metrics, and rollback levers that enforce them. Alert
thresholds in [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1) and
[`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml:1) are
intentionally a little looser than these SLO targets so that on‑call receives
early warning before the error budget is exhausted.

### 6.1 Orchestrator SLO overview

The SLOs below are intentionally small in number and map directly to concrete
metrics or CI jobs:

- CI gating SLOs (per‑release):
  - `SLO-CI-ORCH-PARITY` – orchestrator parity CI job always green for releases.
  - `SLO-CI-ORCH-SHORT-SOAK` – short orchestrator soak has **zero** invariant
    violations before a release can be promoted.

- Staging SLOs (orchestrator‑only staging, used to gate promotion to production):
  - `SLO-STAGE-ORCH-ERROR` – orchestrator error fraction in staging stays well
    below overall HTTP error SLOs.
  - `SLO-STAGE-ORCH-PARITY` – runtime TS↔Python parity mismatches in staging
    are effectively zero when python-authoritative diagnostics are enabled.
  - `SLO-STAGE-ORCH-INVARIANTS` – no new orchestrator invariant violations in
    staging or short soaks.

- Production SLOs (orchestrator‑specific):
  - `SLO-PROD-ORCH-ERROR` – orchestrator‑related errors on game requests stay
    below global availability SLOs.
  - `SLO-PROD-ORCH-INVARIANTS` – no new invariant violations attributable to the
    orchestrator in production or in nightly soaks.
  - `SLO-PROD-RULES-PARITY` – runtime TS↔Python rules‑parity incidents are
    extremely rare and never affect game outcomes.

Each SLO is defined more precisely below with a name, metric, target, window,
and a rough error budget.

#### 6.1.1 Phase 2 – Production Preview success criteria (P18.4-4)

> **Historical only:** The production preview phase described a limited-scope incremental rollout (<=10% traffic). Rollout percentage is no longer configurable; this section is retained for context.

Phase 2 – Production Preview (P18.4-4) mapped to **environment Phase 3 – Incremental production rollout** (see `docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md` §8.5) and intentionally capped traffic until preview SLOs were met.

A Production Preview window is considered successful when all of the following hold over the observation period (typically 1–3 hours per step and ≥24h overall at the final preview percentage):

- **Stability**
  - No unhandled exceptions or crashes in orchestrator adapters or `processTurnAsync` paths visible in application logs.
  - No `OrchestratorCircuitBreakerOpen` alerts in production during the preview window.

- **Performance**
  - `game_move_latency_ms` p95 for orchestrator-handled moves remains within **+10%** of the pre-preview baseline for the same endpoints.
  - No sustained `HighLatency` / game-performance alerts attributable to orchestrator (see `docs/runbooks/GAME_PERFORMANCE.md` for thresholds).

- **Parity**
  - When python-authoritative diagnostics are enabled, runtime parity mismatches remain near zero and **zero** `RulesParityGameStatusMismatch` alerts fire.
  - No new orchestrator-specific invariant violations in production logs (`ringrift_orchestrator_invariant_violations_total`), and short soaks against the production image continue to report `totalInvariantViolations == 0`.

- **User impact**
  - No P0/P1 incidents or tickets referencing "stuck games", "impossible moves", or obviously incorrect winners that are attributable to orchestrator.
  - Support/telemetry show no spike in client-visible errors for move submission or game state streaming during the preview window.

These conditions defined "Success" for Phase 2 in the context of P18.4-4; rollout percentages are historical and no longer configurable.

### 6.2 CI SLOs (pre‑merge and pre‑release)

**SLO-CI-ORCH-PARITY – Orchestrator parity CI gate**

- **Metric:** Status of the `orchestrator-parity` GitHub Actions job defined in
  [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml:1), which runs
  `npm run test:orchestrator-parity:ts` and
  `./scripts/run-python-contract-tests.sh --verbose`.
- **Target:** For any commit that will be:
  - merged to `main`, and
  - promoted to staging or production,
    the `orchestrator-parity` job must be green.
- **Window:** Per‑commit / per‑release candidate.
- **Error budget:** 0 red jobs for release candidates. Any red run on `main`
  blocks promotion until fixed. If the job is flaky (multiple red runs in a
  week for unrelated commits), treat that as SLO debt and prioritise
  hardening the tests rather than weakening the SLO.

**SLO-CI-ORCH-SHORT-SOAK – Short orchestrator soak**

- **Metric:** Exit status and invariant summary from the short orchestrator soak:
  - Canonical CI command:
    `npm run soak:orchestrator:short`
    (as used by the `orchestrator-short-soak` job in
    [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml:1)).
  - Smoke variant (single-game fast check):
    `npm run soak:orchestrator:smoke` producing
    [`results/orchestrator_soak_smoke.json`](../../results/orchestrator_soak_smoke.json:1).
  - For deeper offline or scheduled runs (not required for this SLO) see
    `npm run soak:orchestrator:nightly`, which produces
    [`results/orchestrator_soak_summary.json`](../../results/orchestrator_soak_summary.json:1).
- **Target:**
  - `totalInvariantViolations == 0` and no S-invariant or host-consistency
    violations recorded in the summary.
  - Process exit code `0` (no `--failOnViolation` failure).
- **Window:** Must be run on the exact commit being promoted:
  - At least once before tagging a release that will go to staging or production.
  - Optionally as part of a nightly job against `main`.
- **Error budget:** 0. Any invariant violation in the short soak is an SLO
  breach and **must** block promotion until the underlying rules bug is
  understood and fixed or explicitly waived.

These CI SLOs correspond to **Phase 0 – Pre‑requisites** in the historical
environment rollout plan (see `docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md` §8).

**Auxiliary CI signal – Python AI self‑play invariants**

Although not a hard gate for orchestrator promotion, Python strict‑invariant healthchecks provide an important **P1/P2‑level early warning** for rules/AI regressions:

- **Metric / alerts:**
  - `ringrift_python_invariant_violations_total{invariant_id, type}` exported by the AI self‑play soak harness (`run_self_play_soak.py` under `RINGRIFT_STRICT_NO_MOVE_INVARIANT=1`).
  - `PythonInvariantViolations` alert in [`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml:580) and described in [`docs/operations/ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:640).
- **CI / workflows:**
  - `python-ai-healthcheck` job in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml:1) – runs `python scripts/run_self_play_soak.py --profile ai-healthcheck --max-moves 200 --fail-on-anomaly` across `square8`, `square19`, and `hexagonal`, emitting invariant violations keyed by `INV-*` IDs (see [`docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md`](../rules/INVARIANTS_AND_PARITY_FRAMEWORK.md:1)).
  - `AI Self-Play Healthcheck (Nightly)` workflow in [`.github/workflows/ai-healthcheck-nightly.yml`](../../.github/workflows/ai-healthcheck-nightly.yml:1) – deeper variant with increased `RINGRIFT_AI_HEALTHCHECK_GAMES` and a higher `--max-moves` cap.
- **Rollout posture:**
  - For any release candidate going to staging or production, Python AI healthchecks **should be green** and `PythonInvariantViolations` should be quiet over the recent window, or any violations must be understood and explicitly triaged as AI‑only or training‑only issues.
  - When Python invariant alerts indicate potential cross‑stack rules regressions (for example `INV-ACTIVE-NO-MOVES` or `INV-S-MONOTONIC` anomalies), treat them as inputs to the same investigation loop as orchestrator invariant and rules‑parity signals before advancing phases in `docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md` §8.

### 6.3 Staging SLOs (orchestrator‑only staging)

Staging runs with the orchestrator as the only rules path:

- `ORCHESTRATOR_ADAPTER_ENABLED=true` (hardcoded)
- `RINGRIFT_RULES_MODE=ts`

The SLOs below are primarily used to **gate promotion** from staging to
production.

**SLO-STAGE-ORCH-ERROR – Staging orchestrator error rate**

- **Metric:** `ringrift_orchestrator_error_rate{environment="staging"}` – the
  fraction of orchestrator‑handled requests that failed in the most recent
  error window, as exposed by [`MetricsService`](../../src/server/services/MetricsService.ts:1).
- **Target:**
  `ringrift_orchestrator_error_rate <= 0.001` (0.1%) over a trailing 24‑hour window.
- **Window:** 24h trailing, evaluated via e.g.:
  - `max_over_time(ringrift_orchestrator_error_rate{environment="staging"}[24h])`.
- **Error budget (staging):**
  - At most one short spike above `0.001` lasting < 10 minutes in a 24‑hour
    period.
  - Any sustained period `>= 0.01` (1%) for 10 minutes or more, or any
    `OrchestratorCircuitBreakerOpen` alert in staging, is considered an SLO
    breach and blocks promotion to production.

**SLO-STAGE-ORCH-PARITY – Staging TS↔Python parity mismatches**

- **Metrics:**
  - Rules‑parity counters in the `rules-parity` alert group, for staging
    traffic or scheduled parity jobs:
    - `ringrift_rules_parity_valid_mismatch_total`
    - `ringrift_rules_parity_hash_mismatch_total`
    - `ringrift_rules_parity_game_status_mismatch_total`
- **Target:**
  - Over any 24‑hour period in staging:
    - Validation/hash mismatches:
      `increase(..._valid_mismatch_total[24h]) <= 5` and
      `increase(..._hash_mismatch_total[24h]) <= 5`
    - Game‑status mismatches:
      `increase(..._game_status_mismatch_total[24h]) == 0`
- **Window:** 24h trailing.
- **Error budget (staging):**
  - 0 game‑status mismatches.
  - At most a handful of validation/hash mismatches during early staging,
    all investigated and either fixed or documented as expected legacy
    behaviour before promotion.

**SLO-STAGE-ORCH-INVARIANTS – Staging invariant violations**

- **Metrics:**
  - `ringrift_orchestrator_invariant_violations_total{environment="staging",type,invariant_id}`.
  - In‑cluster logs derived from orchestrator invariant guards and from
    `npm run soak:orchestrator` runs against staging images.
- **Target:**
  - No **new** invariant violation types in staging.
  - Short orchestrator soaks against the staging image behave like CI short
    soaks: `totalInvariantViolations == 0`.
- **Window:** 7d trailing for trend analysis; hard gate is per‑release soak.
- **Error budget (staging):**
  - Discovery of a single new invariant‐violation pattern is acceptable as
    long as it is immediately triaged and turned into a regression test,
    but promotion to production is paused until fixed or explicitly waived.

### 6.4 Production SLOs (orchestrator‑specific)

Production SLOs are phrased in terms of orchestrator‑specific metrics and sit
under the broader availability and latency SLOs documented in
[`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1).

**SLO-PROD-ORCH-ERROR – Production orchestrator error rate**

- **Metrics:**
  - `ringrift_orchestrator_error_rate{environment="production"}`.
  - HTTP error‑rate fraction for game move endpoints, approximated by:
    ```promql
    sum(rate(http_requests_total{route=~"/api/game/.+",status=~"5.."}[5m]))
      /
    sum(rate(http_requests_total{route=~"/api/game/.+"}[5m]))
    ```
- **Target:**
  - `ringrift_orchestrator_error_rate <= 0.005` (0.5%) over a 28‑day window.
  - Game move HTTP 5xx fraction `<= 0.01` (1%) over the same window.
- **Window:** 28 days trailing.
- **Error budget (production):**
  - Roughly 0.5% of orchestrator‑handled moves in a 28‑day period may fail
    before the SLO is considered breached.
  - Practically: more than **one** orchestrator‑specific P1 incident or more
    than **three** `OrchestratorErrorRateWarning` episodes lasting > 10m in a
    28‑day period should trigger a release freeze and potentially a deployment
    rollback to the previous stable build.

**SLO-PROD-ORCH-INVARIANTS – Production invariant violations**

- **Metrics:**
  - `ringrift_orchestrator_invariant_violations_total{environment="production",type,invariant_id}`.
  - Logs and soak summaries derived from the orchestrator soak harness
    running periodically against production images
    (see [`STRICT_INVARIANT_SOAKS.md`](../testing/STRICT_INVARIANT_SOAKS.md:1)).
- **Target:**
  - No invariant violations attributable to the orchestrator in production
    logs.
  - Short soaks run against the production image continue to report
    `totalInvariantViolations == 0`.
- **Window:** 28 days trailing for production logs; per‑run for soaks.
- **Error budget (production):**
  - Any new invariant violation observed in production should be treated as
    an SLO breach for release purposes:
    - Freeze further promotions or deprecations.
    - Escalate to rules maintainers to add a regression test and fix the
      bug before resuming promotion.

**SLO-PROD-RULES-PARITY – Runtime TS↔Python rules parity**

- **Metrics:** Same `rules-parity` metrics as in staging, but restricted to
  the production environment.
- **Target:**
  - No production `RulesParityGameStatusMismatch` alerts.
  - Validation/hash mismatches `<= 5` per 24h and limited to known,
    documented historical cases (see `docs/rules/PARITY_SEED_TRIAGE.md`).
- **Window:** 28 days trailing, with fine‑grained 1h / 24h views for
  debugging.
- **Error budget (production):**
  - 0 game‑status mismatch incidents affecting real games.
  - If more than 2 parity‑related incidents occur in a quarter, consider:
    - Avoiding python-authoritative diagnostics in production traffic.
    - Tightening parity tests and contract vectors before further promotion.

### 6.5 Test suites

For each phase (implementation Phases A–D and environment Phases 0–4), the
following suites must be green. These are explicitly tied to the **PASS18 Weakest Aspect** (Host Integration & Parity) and **Hardest Problem** (Deep Multi-Engine Parity).

- **TypeScript (Host Integration & Parity)**
  - **Capture & Territory Host Parity:**
    - `tests/unit/captureSequenceEnumeration.test.ts`
    - `tests/unit/GameEngine.chainCapture*.test.ts`
    - `tests/unit/BoardManager.territoryDisconnection.test.ts`
    - `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`
  - **AI RNG Parity:**
    - `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`
    - `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`
  - **General Parity & Contracts:**
    - Backend vs sandbox parity (`tests/unit/Backend_vs_Sandbox.*`).
    - Orchestrator adapter tests for backend and sandbox.
    - Contract tests: `tests/contracts/**` and TS‑side contract runners.
  - **Core Scenarios:**
    - `tests/scenarios/**` (including RulesMatrix territory and line tests).

- **Python (Deep Multi-Engine Parity)**
  - **Core Rules:** `ai-service/tests/**`.
  - **Parity & Contracts:**
    - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py` (covers `PARITY-TS-PY-TERRITORY-LINE`)
    - `ai-service/tests/parity/test_chain_capture_parity.py`
    - `ai-service/tests/contracts/test_contract_vectors.py` (covers `PARITY-TS-PY-CONTRACT-VECTORS`)
  - **Invariants:**
    - `ai-service/tests/invariants/**` (covers `INV-ACTIVE-NO-MOVES` regressions).

These suites are part of the **SLO enforcement mechanism**: failing tests
invalidate `SLO-CI-ORCH-PARITY` and halt promotion until fixed.

### 6.6 Metrics and observability

Across implementation Phases A–C and environment Phases 0–4, the following
metrics must be monitored, as described in
[`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`](../runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md:1),
[`docs/runbooks/RULES_PARITY.md`](../runbooks/RULES_PARITY.md:1), and
[`docs/runbooks/GAME_HEALTH.md`](../runbooks/GAME_HEALTH.md:1):

- Orchestrator‑specific metrics:
  - `ringrift_orchestrator_error_rate`.
  - `ringrift_orchestrator_circuit_breaker_state`.
  - `ringrift_orchestrator_rollout_percentage` (fixed at 100).

- Game performance metrics:
  - `game_move_latency_ms` (backend move latency).
  - `ai_move_latency_ms` and AI fallback counts (`ai_fallback_total`).
  - WebSocket connection and error metrics.

- Rules parity metrics:
  - `rulesParityMetrics` counters surfaced by `RulesBackendFacade` in
    python‑authoritative diagnostics and by TS↔Python parity jobs.

Target thresholds should align with v1.0 user‑visible SLOs in
[`PROJECT_GOALS.md`](../../PROJECT_GOALS.md:76) and the thresholds in
[`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml:1),
while remaining consistent with the orchestrator SLOs in §§6.2–6.4.

### 6.7 Rollback levers

Rollbacks should rely on:

- Deployment rollback procedures:
  - Use the existing deployment runbooks to return to a known‑good build when
    orchestrator regressions are confirmed.
- Diagnostic flags (do **not** change routing):
  - `RINGRIFT_RULES_MODE=python` for parity diagnostics only.
  - `ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED` and thresholds for telemetry.
  - Sandbox `useOrchestratorAdapter` flags in tests/diagnostics only.

- Git and deployment practices:
  - Each phase implemented in focused, reversible commits or pull requests.
  - Staging deployments gated on a full green test suite and healthy
    orchestrator metrics.
  - Production deployments using the existing deployment runbooks and
    rollback procedures in
    [`docs/runbooks/DEPLOYMENT_ROUTINE.md`](../runbooks/DEPLOYMENT_ROUTINE.md:1)
    and [`docs/runbooks/DEPLOYMENT_ROLLBACK.md`](../runbooks/DEPLOYMENT_ROLLBACK.md:1).

These practices inform the historical environment phases (`docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md` §8); there is no
runtime rollback lever in current production configurations.

### 6.8 Test / CI profiles (orchestrator vs python diagnostics)

To keep semantics and CI signals consistent, test and CI jobs should be
organised around a small set of standard profiles:

- **Orchestrator‑ON (default gate)**
  - Env:
    - `ORCHESTRATOR_ADAPTER_ENABLED=true` (hardcoded)
    - `RINGRIFT_RULES_MODE=ts`
  - TS:
    - Core/unit/integration suites (`npm run test:core`, `npm run test:ci`),
      including `.shared` helpers, contract vectors, RulesMatrix/FAQ
      scenarios, and adapter‑level tests.
  - Python:
    - `pytest` over `ai-service/tests/**`, including contracts and
      fixture‑driven parity.
  - Policy:
    - These jobs are the **only** ones allowed to block merges; any failure
      here is a hard gate and a direct `SLO-CI-ORCH-PARITY` breach.

- **Python‑authoritative diagnostics**
  - Env (examples; exact wiring may vary by job):
    - `ORCHESTRATOR_ADAPTER_ENABLED=true` (hardcoded)
    - `RINGRIFT_RULES_MODE=python`
  - TS:
    - Selected parity/trace suites and host‑comparison tests that are
      explicitly tagged as **diagnostic** (see `tests/README.md`,
      `tests/TEST_SUITE_PARITY_PLAN.md`).
  - Python:
    - Optional additional parity / soak tests over TS‑generated fixtures.
  - Policy:
    - These jobs are **non‑canonical** and must not redefine semantics; when
      they disagree with the orchestrator‑ON profile and `.shared` tests,
      treat traces as derived and update or archive them.
    - Failures here should open investigation tickets but do not
      automatically block merges unless explicitly promoted.

### 6.9 Orchestrator gating set (tests, soaks, metrics)

For each promotion (staging and production), treat the following as the
**orchestrator gating set**. All items must be green or within SLO before
proceeding:

- **TypeScript tests (rules + hosts)**
  - Coverage: suites listed in §6.5 under **TypeScript (Host Integration & Parity)**.
  - CI job: `orchestrator-parity` (TS side).
  - Local example:
    ```bash
    npm run test:orchestrator-parity:ts
    ```

- **Python tests (parity + invariants)**
  - Coverage: suites listed in §6.5 under **Python (Deep Multi-Engine Parity)**.
  - CI job: Python contract/parity lane (see `SUPPLY_CHAIN_AND_CI_SECURITY.md`).
  - Local example (when `PYTHONPATH` is configured as in CI):
    ```bash
    ./scripts/run-python-contract-tests.sh --verbose
    ```

- **Orchestrator short soak (invariant gate)**
  - Goal: ensure `totalInvariantViolations == 0` on the candidate image.
  - CI job: `orchestrator-short-soak` (SLO `SLO-CI-ORCH-SHORT-SOAK`).
  - Local example:
    ```bash
    npm run soak:orchestrator:short
    # equivalent:
    # TS_NODE_PROJECT=tsconfig.server.json \
    #   ts-node scripts/run-orchestrator-soak.ts \
    #     --profile=ci-short --failOnViolation=true
    ```

- **Metrics and alerts (runtime posture)**
  - Metrics: `ringrift_orchestrator_error_rate`,
    `ringrift_orchestrator_invariant_violations_total`,
    `game_move_latency_ms`, `ai_move_latency_ms`, WebSocket error metrics.
  - Alerts: `OrchestratorCircuitBreakerOpen`,
    `OrchestratorErrorRateWarning`, `RulesParity*`, and the general
    game-health alerts in
    `monitoring/prometheus/alerts.yml`.
  - Requirement: no active P0/P1 alerts in these groups for the candidate
    build or environment; SLOs in §§6.2–6.4 and the runbooks above must be
    satisfied.

### 6.10 Optional AI health overlay (evaluate_ai_models.py)

Separately from rules/orchestrator correctness, operators may layer in an
**AI strength and latency healthcheck** using the existing evaluation
framework in `ai-service/scripts/evaluate_ai_models.py`. This is **not yet a
hard gate** but is recommended as a companion signal when preparing major
releases that touch AI difficulty profiles or models.

- **Purpose**
  - Track win-rates, game lengths, and decision times for higher difficulty
    bands (e.g. 7–10 “Stronger Opponents”) relative to a baseline.
  - Detect regressions in AI strength or latency even when orchestrator
    rules tests and soaks are green.

- **Example invocation (staging image, nightly or pre-release)**

  ```bash
  cd ai-service
  python scripts/evaluate_ai_models.py \
    --player1 neural_network \
    --player2 minimax \
    --games 50 \
    --board square8 \
    --seed 42 \
    --output ../results/ai_eval_neural_vs_minimax.json
  ```

- **Consumption**
  - Treat the JSON output as an input to AI performance dashboards or ad-hoc
    analysis (see `docs/runbooks/AI_PERFORMANCE.md` and
    `ai-service/AI_IMPROVEMENT_PLAN.md` for interpretation).
  - If evaluation reveals clear strength/latency regressions, pause promotion
    and handle under the AI runbooks even if orchestrator gates are green.

When adding new CI jobs or local profiles, align them with one of these two
buckets and keep their environment flags documented so it is always clear
whether a given failure is in the canonical orchestrator‑ON lane or in a
python‑authoritative diagnostics lane.

### 6.9 Orchestrator Parity CI Job (`orchestrator-parity`)

The orchestrator parity CI gate is implemented as the `orchestrator-parity` job
in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml:1). It runs under the
**orchestrator‑ON** profile described above and is required for
orchestrator‑first promotion and TS↔Python parity guarantees.

- **TS orchestrator/host tests:**
  - Command: `npm run test:orchestrator-parity:ts`
  - Scope: backend and sandbox orchestrator multi‑phase scenarios plus core
    lines/territory unit suites:
    - [`tests/scenarios/MultiPhaseTurn.contractVectors.test.ts`](../../tests/scenarios/MultiPhaseTurn.contractVectors.test.ts:1)
    - [`tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts`](../../tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts:1)
    - [`tests/unit/GameEngine.lines.scenarios.test.ts`](../../tests/unit/GameEngine.lines.scenarios.test.ts:1)
    - [`tests/unit/sandboxLines.test.ts`](../../tests/unit/sandboxLines.test.ts:1)
    - [`tests/unit/territoryDecisionHelpers.shared.test.ts`](../../tests/unit/territoryDecisionHelpers.shared.test.ts:1)
    - [`tests/unit/BoardManager.territoryDisconnection.test.ts`](../../tests/unit/BoardManager.territoryDisconnection.test.ts:1)
    - [`tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`](../../tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts:1)

- **Python contract vectors:**
  - Command (from repo root): `./scripts/run-python-contract-tests.sh --verbose`
  - Uses [`ai-service/requirements.txt`](../../ai-service/requirements.txt:1) and
    [`ai-service/tests/contracts/test_contract_vectors.py`](../../ai-service/tests/contracts/test_contract_vectors.py:1)
    to validate Python rules behaviour against shared TS contract vectors.

Failures in this job indicate either an orchestrator/host regression in the TS
engine or a TS↔Python contract vector divergence, and **must block merges**
until resolved. They are treated as direct `SLO-CI-ORCH-PARITY` violations.

## 7. Track A Orchestrator Rollout Overview

This section summarises the plan for use by Track A implementation tasks:

- **P16.6.1-CODE – Backend orchestrator-first**
  - Implement **Phase A** actions, making `TurnEngineAdapter` + `processTurnAsync` the only backend production path.
  - Quarantine and then deprecate legacy `RuleEngine` pipelines.
  - Ensure all backend tests run with orchestrator enabled and parity metrics monitored.

- **P16.6.2-CODE – Sandbox orchestrator-first**
  - Implement **Phase B** actions, making `SandboxOrchestratorAdapter` + `processTurnAsync` the only sandbox rules path.
  - Demote sandbox helpers to UX/diagnostics-only and ensure orchestrator is used for canonical move processing in `/sandbox`.

- **P16.6.3-CODE – Legacy shutdown & diagnostics fencing**
  - Implement **Phase C**, removing dead backend and sandbox helpers and fencing diagnostics tooling into clearly marked namespaces with SSOT banners.

- **P16.7-QA – Parity and soak testing**
  - Expand orchestrator parity and soak testing across representative seeds and board types, exercising deep capture chains, line and territory combinations, and LPS cases.
  - Ensure TS and Python parity suites remain green and that `rulesParityMetrics` do not regress under realistic workloads.

- **P16.8-DEVOPS – Staging rollout and SLO gating**
  - Use orchestrator metrics and parity diagnostics as described in `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`.
  - Require at least one successful staging bake-in period (SLOs met) before production promotion.
  - Maintain clear rollback procedures using deployment runbooks (no runtime rollout flags remain).

### 7.1 Phase A – Backend orchestrator-only status (2025-11-28)

- Backend `GameSession` / `GameEngine` production move processing now routes
  exclusively through `TurnEngineAdapter.processMove(...)` and the shared
  `processTurnAsync` orchestrator whenever `config.isTest === false`.
- Legacy `GameEngine` / `RuleEngine` turn pipelines remain available only for
  test and diagnostics harnesses:
  - `GameEngine.makeMove` falls back to the legacy branch only under
    `config.isTest === true` when tests explicitly call
    `disableOrchestratorAdapter()`.
  - `RuleEngine.processMove`, `processLineFormation`, and
    `processTerritoryDisconnection` are now marked as
    **DIAGNOSTICS-ONLY (legacy)** and have no production call sites.
- WebSocket handlers (`GameSession.handlePlayerMove` /
  `handlePlayerMoveById`) and `RulesBackendFacade.applyMove` /
  `applyMoveById` therefore apply moves via the shared orchestrator-only
  backend path in production.

### 7.2 Phase B – Sandbox orchestrator-only status (2025-11-28)

- `/sandbox` and the React host in [`GamePage`](../../src/client/pages/GamePage.tsx:1) now
  drive all canonical local sandbox moves through:
  - `ClientSandboxEngine.processMoveViaAdapter`
  - [`SandboxOrchestratorAdapter`](../../src/client/sandbox/SandboxOrchestratorAdapter.ts:1)
  - `processTurnAsync` in the shared orchestrator.
- The legacy `LocalSandboxState` + `localSandboxController` harness is fully
  fenced off from production sandbox flows:
  - `GamePage.tsx` no longer imports `LocalSandboxState` or projects it into a
    faux `GameState` for the live `/sandbox` view.
  - [`useSandboxInteractions`](../../src/client/hooks/useSandboxInteractions.ts:1) no longer
    imports or calls `handleLocalSandboxCellClick`; it always uses
    `ClientSandboxEngine` when a sandbox is configured.
  - `SandboxContext` continues to construct `ClientSandboxEngine` via
    `initLocalSandboxEngine`; no code paths re-instantiate the legacy harness
    for production `/sandbox` usage.

### 7.5 Wave 4 – Orchestrator Rollout & Invariant Hardening (summary)

For planning and status-tracking purposes, the Track A work above can be viewed as
**Wave 4 – Orchestrator Rollout & Invariant Hardening**, composed of four focused
sub‑waves:

- **4‑A – Parity & contract expansion**
  - TS orchestrator parity suites (backend + sandbox multi‑phase scenarios, lines/
    territory helpers) as described in §6.8–6.9 and §7.1–7.2.
  - Python contract‑vector tests (`scripts/run-python-contract-tests.sh`,
    `ai-service/tests/contracts/test_contract_vectors.py`) anchored on the v2 contract
    vectors under `tests/fixtures/contract-vectors/v2/**`.
- **4‑B – Invariant soaks & CI gates**
  - Short orchestrator invariant soak (`npm run soak:orchestrator:short` and the
    `orchestrator-short-soak` CI job) backing `SLO-CI-ORCH-SHORT-SOAK` (§6.2 and
    `docs/testing/STRICT_INVARIANT_SOAKS.md`), with `soak:orchestrator:smoke` available
    as a fast single-game smoke profile.
  - Longer/staged soaks against `main` and staging images using
    `scripts/run-orchestrator-soak.ts` with `--failOnViolation` (for example
    `npm run soak:orchestrator:nightly`), producing
    `results/orchestrator_soak_summary.json` / `results/orchestrator_soak_smoke.json`
    for regression mining.
- **4‑C – Rollout flags, topology & fallbacks**
  - Environment/flag wiring via `OrchestratorRolloutService` and
    `src/server/config/env.ts`, with CI/staging/production profiles matching the
    environment phases referenced throughout §6 and §7.
  - Clear rollback guidance using deployment runbooks and circuit‑breaker telemetry
    (no runtime rollout/shadow flags remain), as summarised in §6.7.
- **4‑D – Observability & incident readiness**
  - Orchestrator and rules‑parity metrics (`ringrift_orchestrator_*`,
    `rulesParityMetrics`) exported by `MetricsService` and wired into Prometheus
    alerts (see §6.6 and `monitoring/prometheus/alerts.yml`).
  - Runbooks (`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`,
    `docs/runbooks/RULES_PARITY.md`, `docs/runbooks/GAME_HEALTH.md`) that map alerts
    to concrete responses (deployment rollback, parity diagnostics, or AI/infra
    runbooks).

Wave 4 is considered complete when:

- The `orchestrator-parity` CI job is stable and treated as a hard gate for merges to
  `main`.
- Orchestrator invariant soaks (short CI soak plus longer scheduled soaks) reliably
  block promotion on new invariant violations.
- Environment phases (0–4) have a direct mapping to env flags and have been verified in
  staging.
- Orchestrator metrics and alerts match the SLOs defined in §6 and are backed by
  up‑to‑date runbooks.
- The `localSandboxController.ts` module remains in place as a
  **DIAGNOSTICS-ONLY (legacy)** harness for tests and CLI tooling, and is
  fenced by `scripts/ssot/rules-ssot-check.ts` so it cannot be imported from
  `GamePage`, `SandboxContext`, or `ClientSandboxEngine`.

### 7.3 Phase C – Legacy helper shutdown & diagnostics fencing status (2025-11-28)

- **Backend helpers**
  - Legacy helpers in [`RuleEngine`](../../src/server/game/RuleEngine.ts:1)
    (`processMove`, `processChainReactions`, `processLineFormation`,
    `processTerritoryDisconnection`) are explicitly documented as
    **DIAGNOSTICS-ONLY (legacy backend pipeline)** and are not called from:
    - `GameEngine.makeMove`
    - `GameSession`
    - WebSocket handlers under `src/server/websocket/**`.
  - `GameEngine` uses `RuleEngine` only for validation/enumeration
    (`validateMove`, `getValidMoves`, `checkGameEnd`); all production move
    application is driven via shared aggregates and the orchestrator.
  - No additional backend deletions were required for this pass; legacy helpers
    remain available for deep diagnostics and archived tests but are fully
    fenced from production hosts by call-site search and SSOT checks.
- **Sandbox diagnostics surface**
  - [`sandboxCaptures.applyCaptureSegmentOnBoard`](../../src/client/sandbox/sandboxCaptures.ts:103)
    now carries a top-level **DIAGNOSTICS-ONLY (SANDBOX ANALYSIS TOOL)** banner.
    It is only used by:
    - Capture/chain diagnostics tests (e.g.
      [`captureSequenceEnumeration.test.ts`](../../tests/unit/captureSequenceEnumeration.test.ts:1),
      cyclic-capture suites).
    - Diagnostic helpers built on cloned boards; live sandbox capture mutation
      in `ClientSandboxEngine` delegates exclusively to `CaptureAggregate`.
  - [`sandboxCaptureSearch`](../../src/client/sandbox/sandboxCaptureSearch.ts:1) now has a
    module header declaring it **DIAGNOSTICS-ONLY (SANDBOX CAPTURE CHAIN SEARCH)**:
    - It performs DFS over cloned `BoardState`s using shared
      `enumerateCaptureSegmentsFromBoard` + `applyCaptureSegmentOnBoard`.
    - It is imported only from tests and diagnostics/CLI code, never from
      `ClientSandboxEngine`, `SandboxContext`, or `/sandbox` UI components.
  - `scripts/ssot/rules-ssot-check.ts` includes narrow, string-based guards that
    fail the SSOT check if `localSandboxController` or `sandboxCaptureSearch`
    are imported from production hosts (backend game stack, `GamePage`,
    `SandboxContext`, `ClientSandboxEngine`), preventing accidental reintroduction
    of legacy rules paths.

Once all phases are complete, **all production and sandbox rules paths will route exclusively through**:

- `processTurnAsync` in the shared orchestrator as the lifecycle SSOT.
- The eight domain aggregates as the rules semantics SSOT for placement, movement, recovery, capture, line, territory, elimination, and victory.
- `TurnEngineAdapter` and `SandboxOrchestratorAdapter` as the exclusive host integration layers.

Legacy rules pipelines in backend and sandbox hosts will be fully removed or quarantined as diagnostics-only helpers, and documentation will accurately reflect the orchestrator-first architecture.

## 8. Current posture summary

- The orchestrator adapter is always on; legacy turn pipelines are diagnostics-only.
- Production and staging run with `RINGRIFT_RULES_MODE=ts`.
- `RINGRIFT_RULES_MODE=python` is reserved for explicit parity diagnostics; it does not change production routing.
- Rollout percentage and shadow-mode flags are removed; the circuit breaker reports telemetry only.
- Rollback is deployment-based (use the standard rollback runbooks for regressions).

Historical environment rollout phases and presets live in:
`docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md`.
