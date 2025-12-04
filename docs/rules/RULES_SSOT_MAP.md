# RingRift Rules & Orchestrator SSOT Map

> **Scope:** Design-level map of rules and orchestrator single sources of truth (SSOTs) and legacy/diagnostic surfaces.  
> **Role:** Derived overview for maintainers; does not introduce new rules semantics or APIs.  
> **Upstream semantics SSoT:** [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md), `ringrift_complete_rules.md`, `ringrift_compact_rules.md`, and the shared TS engine + orchestrator under [`src/shared/engine`](src/shared/engine).  
> **Upstream lifecycle/API SSoT:** [`docs/CANONICAL_ENGINE_API.md`](docs/CANONICAL_ENGINE_API.md), shared types in [`src/shared/types/game.ts`](src/shared/types/game.ts), orchestrator types in [`src/shared/engine/orchestration/types.ts`](src/shared/engine/orchestration/types.ts), and WebSocket contracts in [`src/shared/types/websocket.ts`](src/shared/types/websocket.ts) and [`src/shared/validation/websocketSchemas.ts`](src/shared/validation/websocketSchemas.ts).  
> **Upstream implementation-status SSoT:** [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md).  
> **This doc:** Clarifies what is canonical vs historical/diagnostic for rules execution across TS backend, sandbox, and Python, and records known drift items for follow-up tasks.

---

## 1. SSOT Hierarchy Overview

The hierarchy below orders sources by normative authority. Higher bullets win on conflicts.

**1. Rules semantics (normative)**

- Rules-level spec: [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md) (RR-CANON rules; conflicts resolved in favour of Compact Spec as documented there).
- Narrative sources (commentary only): [`ringrift_complete_rules.md`](ringrift_complete_rules.md), [`ringrift_compact_rules.md`](ringrift_compact_rules.md).

**2. Executable rules semantics SSoT (TS engine)**

- Shared engine helpers and core: [`core.ts`](src/shared/engine/core.ts:1), [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1), [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1), [`lineDetection.ts`](src/shared/engine/lineDetection.ts:21), [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:36), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1), [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1), [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts:45), placement helpers and related modules under [`src/shared/engine`](src/shared/engine).
- Domain aggregates (single source of truth per domain):
  - Placement: [`PlacementAggregate.ts`](src/shared/engine/aggregates/PlacementAggregate.ts:1)
  - Movement: [`MovementAggregate.ts`](src/shared/engine/aggregates/MovementAggregate.ts:1)
  - Capture: [`CaptureAggregate.ts`](src/shared/engine/aggregates/CaptureAggregate.ts:56)
  - Line: [`LineAggregate.ts`](src/shared/engine/aggregates/LineAggregate.ts:1)
  - Territory: [`TerritoryAggregate.ts`](src/shared/engine/aggregates/TerritoryAggregate.ts:1)
  - Victory: [`VictoryAggregate.ts`](src/shared/engine/aggregates/VictoryAggregate.ts:1)
- Turn orchestrator (canonical execution surface over aggregates): [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts:1) and [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts:1).
- Cross-language contracts and vectors: [`src/shared/engine/contracts/*.ts`](src/shared/engine/contracts/schemas.ts:1) and v2 vectors under [`tests/fixtures/contract-vectors/v2`](tests/fixtures/contract-vectors/v2).

**3. Lifecycle/API SSoT**

- Engine API and Move/decision lifecycle: [`docs/CANONICAL_ENGINE_API.md`](docs/CANONICAL_ENGINE_API.md).
- Shared types and schemas: [`src/shared/types/game.ts`](src/shared/types/game.ts:1), orchestrator types in [`src/shared/engine/orchestration/types.ts`](src/shared/engine/orchestration/types.ts:1), WebSocket types in [`src/shared/types/websocket.ts`](src/shared/types/websocket.ts:1), schemas in [`src/shared/validation/websocketSchemas.ts`](src/shared/validation/websocketSchemas.ts:1).

**4. Host integration boundaries (TS hosts over the engine)**

- Backend host:
  - Game host and lifecycle: [`GameEngine.ts`](src/server/game/GameEngine.ts:1).
  - Turn orchestrator adapter (canonical): [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts:1).
  - Rules facade and Python bridge: [`RuleEngine.ts`](src/server/game/RuleEngine.ts:46) (validation/enumeration only in steady state), [`RulesBackendFacade.ts`](src/server/game/RulesBackendFacade.ts:54), [`PythonRulesClient.ts`](src/server/services/PythonRulesClient.ts:33).
- Sandbox host:
  - Client-local host: [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:137).
  - Orchestrator adapter (canonical sandbox integration): [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1).
- Shared state machines for sessions, choices, AI, and connections (lifecycle only, not rules semantics): [`src/shared/stateMachines/*.ts`](src/shared/stateMachines/gameSession.ts:1), documented in [`docs/STATE_MACHINES.md`](docs/STATE_MACHINES.md).

**5. Python mirror and contract tests (derived mirror over TS engine)**

- Python game engine and board utilities: [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py:1), [`ai-service/app/board_manager.py`](ai-service/app/board_manager.py:1).
- Python rules modules (validators/mutators and helpers): [`ai-service/app/rules/*.py`](ai-service/app/rules/core.py:1), [`ai-service/app/rules/mutators/*.py`](ai-service/app/rules/mutators/capture.py:1), [`ai-service/app/rules/validators/*.py`](ai-service/app/rules/validators/territory.py:1).
- Python default engine wrapper and shadow contracts: [`ai-service/app/rules/default_engine.py`](ai-service/app/rules/default_engine.py:1).
- Contract and parity tests (TS ↔ Python): [`tests/contracts/contractVectorRunner.test.ts`](tests/contracts/contractVectorRunner.test.ts:1) and [`ai-service/tests/contracts/test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py:1), plus broader parity suites documented in [`docs/PYTHON_PARITY_REQUIREMENTS.md`](docs/PYTHON_PARITY_REQUIREMENTS.md).

**6. Operational / rollout SSoTs (derived over engine + hosts)**

- Orchestrator rollout and legacy shutdown: [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md) and [`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`](docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md).
- Strict invariant soaks and termination guarantees: [`docs/STRICT_INVARIANT_SOAKS.md`](docs/STRICT_INVARIANT_SOAKS.md), orchestrator soak harness in [`scripts/run-orchestrator-soak.ts`](scripts/run-orchestrator-soak.ts:1) with summaries under [`results/orchestrator_soak_summary.json`](results/orchestrator_soak_summary.json:1).
- Shared engine and aggregate design: [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md), [`docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`](docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md), [`docs/DOMAIN_AGGREGATE_DESIGN.md`](docs/DOMAIN_AGGREGATE_DESIGN.md).

**7. Implementation status / meta-docs (status only, non-semantic)**

- Implementation status and gaps: [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md).
- Rules implementation mapping and traceability: [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md).
- Historical and archival analyses under [`archive/`](archive/FINAL_RULES_AUDIT_REPORT.md:1).

In conflicts about **rules semantics**, the order of authority is: rules spec → shared TS engine + orchestrator → contract tests. Host adapters and Python must follow these.

---

## 2. Canonical TS Rules & Orchestrator Entry Points

Hosts must treat the turn orchestrator surface as the **only canonical execution path** for applying moves and querying legality in live games.

**Canonical orchestrator API (TS)**  
_Location_: [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts:1), described in detail in [`docs/CANONICAL_ENGINE_API.md` §3.9](docs/CANONICAL_ENGINE_API.md).

- Turn processing:
  - [`processTurn(state, move)`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – pure, synchronous entry point for applying a `Move` and all automatic consequences (captures, lines, territory, victory).
  - [`processTurnAsync(state, move, delegates)`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – canonical host-facing entry, using [`TurnProcessingDelegates`](src/shared/engine/orchestration/types.ts:1) for decision resolution and side-channel integration (choices, timeouts, AI).
- Validation and enumeration helpers:
  - [`validateMove(state, move)`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – canonical legality check for a concrete `Move`.
  - [`getValidMoves(state)`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – canonical legal move enumeration for `state.currentPlayer` / `state.currentPhase`.
  - [`hasValidMoves(state)`](src/shared/engine/orchestration/turnOrchestrator.ts:1) – convenience predicate used by hosts and diagnostics.

**Execution semantics**

- The orchestrator calls domain aggregates in deterministic order for a full turn (placement → movement/capture/chain_capture → line_processing → territory_processing → victory), using the shared turn/phase state machine in [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts:1) and [`turnLogic.ts`](src/shared/engine/turnLogic.ts:135).
- All rule semantics for placement, movement, capture, lines, territory, and victory **must** flow through the aggregates listed in §1.2, not through host-specific reimplementations.
- Aggregates may be used **directly** only in:
  - Shared engine unit tests and contract-vector generation.
  - Diagnostics scripts and offline analysis tooling.
  - Very narrow adapter glue where orchestrator is intentionally bypassed for previews (e.g. `previewMove` in [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1)).
- Production hosts (backend `GameEngine`, sandbox `ClientSandboxEngine`) must **not** implement alternative turn pipelines that bypass the orchestrator surface for live games.

**Alignment with docs and mapping**

- [`docs/CANONICAL_ENGINE_API.md`](docs/CANONICAL_ENGINE_API.md) defines these functions as the **canonical engine API** and explicitly states that hosts must integrate via `processTurn` / `processTurnAsync` + validation/enumeration helpers.
- [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md) treats `src/shared/engine/**` (helpers → aggregates → orchestrator → contracts) as the rules/invariants semantics SSoT; backend, sandbox, and Python hosts are classified there as adapters over this surface.

---

## 3. Host Integration Map

### 3.1 Backend host (Node/TS)

**Canonical production path (orchestrator-first)**

1. **Transport:** WebSocket or HTTP handler receives a move submission (`player_move` or `player_move_by_id`), handled by [`WebSocketInteractionHandler.ts`](src/server/game/WebSocketInteractionHandler.ts:1) and [`GameSession.ts`](src/server/game/GameSession.ts:1).
2. **Game host:** `GameSession` calls backend host [`GameEngine`](src/server/game/GameEngine.ts:1) to apply the move.
3. **Adapter selection:** In production (`config.isTest === false`) and under the orchestrator rollout posture described in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:1), [`GameEngine`](src/server/game/GameEngine.ts:1) routes all moves through [`processMoveViaAdapter`](src/server/game/GameEngine.ts:1), which delegates to [`TurnEngineAdapter`](src/server/game/turn/TurnEngineAdapter.ts:1). Legacy, non-adapter paths are restricted to tests/diagnostics (see §5).
4. **Backend adapter:** [`TurnEngineAdapter`](src/server/game/turn/TurnEngineAdapter.ts:1):
   - Reads the current `GameState` via a backend `StateAccessor`.
   - Builds [`TurnProcessingDelegates`](src/shared/engine/orchestration/types.ts:1) that:
     - Present orchestrator [`PendingDecision`](src/shared/engine/orchestration/types.ts:1) objects as `PlayerChoice` messages over WebSockets.
     - Resolve AI decisions via backend AI services or local fallbacks.
   - Calls [`processTurnAsync`](src/shared/engine/orchestration/turnOrchestrator.ts:1) with the submitted `Move` and delegates.
   - Writes back the resulting `GameState`, emits `game_state` / `game_ended` events, and maps orchestrator `VictoryState` into backend [`GameResult`](src/shared/types/game.ts:1).
5. **Validation and enumeration:**
   - For pre-move checks (e.g. move preview, UI `validMoves`), backend code uses `TurnEngineAdapter.validateMoveOnly`, `TurnEngineAdapter.getValidMovesFor`, and `TurnEngineAdapter.hasAnyValidMoves`, which delegate directly to [`validateMove`](src/shared/engine/orchestration/turnOrchestrator.ts:1), [`getValidMoves`](src/shared/engine/orchestration/turnOrchestrator.ts:1), and [`hasValidMoves`](src/shared/engine/orchestration/turnOrchestrator.ts:1).

**Legacy/diagnostic backend paths**

- **Legacy `GameEngine` pipeline** – historical turn loop inside [`GameEngine.ts`](src/server/game/GameEngine.ts:1) that:
  - Applies moves via shared aggregates and helpers (movement, capture, lines, territory) but manages phase transitions and some forced-elimination/LPS logic locally.
  - Is now guarded by `config.isTest` and explicit `disableOrchestratorAdapter()` calls and is **not used in production** per [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:793).
  - Intended status: `test-only / diagnostics-only`.
- **Legacy `RuleEngine` orchestration helpers** – methods such as `processMove`, `processLineFormation`, `processTerritoryDisconnection` in [`RuleEngine.ts`](src/server/game/RuleEngine.ts:46) predate the orchestrator and are now documented as **DIAGNOSTICS-ONLY (legacy backend pipeline)** in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.3](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:882).
  - In steady state they must not be called from `GameEngine` or WebSocket handlers; only validation/enumeration entry points (`validateMove`, `getValidMoves`, `checkGameEnd`) remain part of the supported backend surface.
- **Diagnostics scripts** – backend debugging and parity tooling (e.g. trace replayers) may still call shared aggregates or legacy helpers directly but are fenced by SSOT scripts under [`scripts/ssot`](scripts/ssot/rules-ssot-check.ts:1).

### 3.2 Sandbox host (client TS)

**Canonical sandbox path (orchestrator-backed)**

1. **Transport/UI:** React components (`GamePage`, `BoardView`, `GameHUD`) interact with a `GameContext` that holds a [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts:137) instance for `/sandbox` sessions.
2. **Sandbox host:** [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts:137):
   - Maintains client-local `GameState` and history for sandbox games.
   - Exposes methods like `applyCanonicalMove`, `handleHumanCellClick`, and `maybeRunAITurn` that treat sandbox play as canonical rules exercise.
3. **Orchestrator adapter usage:** By default (`ORCHESTRATOR_ADAPTER_ENABLED` flag true), `ClientSandboxEngine`’s `applyCanonicalMoveInternal` uses [`processMoveViaAdapter`](src/client/sandbox/ClientSandboxEngine.ts:1746), which delegates to [`SandboxOrchestratorAdapter`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1).
4. **Sandbox adapter:** [`SandboxOrchestratorAdapter`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1):
   - Mirrors backend `TurnEngineAdapter` but in a browser-safe context.
   - Uses `processTurn` / `processTurnAsync` plus `validateMove` / `getValidMoves` / `hasValidMoves` to:
     - Apply moves and automatic consequences.
     - Surface decision phases as local `PlayerChoice`-like prompts or AI callbacks.
     - Expose preview functions (`processMoveSync`, `previewMove`) for UI hover/what-if flows, still grounded in the orchestrator.
   - Updates `GameState` via a sandbox `StateAccessor` and exposes metadata for debugging (`hashBefore`, `hashAfter`, `phasesTraversed`, `durationMs`).

**Legacy/diagnostic sandbox surfaces**

- **Legacy `ClientSandboxEngine` pipeline** – non-orchestrator branches in [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:137) that:
  - Directly orchestrate movement, capture, lines, territory, forced elimination, and LPS using shared aggregates plus sandbox helpers (`sandboxMovement`, `sandboxCaptures`, `sandboxLines`, `sandboxTerritory`, `sandboxElimination`, `sandboxVictory`).
  - Are now treated as **tests/tools-only** surfaces, used primarily in sandbox vs backend parity tests and historical diagnostics (see [`docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md` §3–4](docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md:41) and [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §4.2](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:224)).
  - Intended status: `diagnostic-only` once orchestrator-backed `/sandbox` is stable.
- **Sandbox helper modules** – adapters and analysis tools in `src/client/sandbox/**`:
  - Movement helpers: [`sandboxMovement.ts`](src/client/sandbox/sandboxMovement.ts:1) – UI/diagnostic adapter over shared movement helpers and aggregates.
  - Capture helpers and chain search: [`sandboxCaptures.ts`](src/client/sandbox/sandboxCaptures.ts:1), [`sandboxCaptureSearch.ts`](src/client/sandbox/sandboxCaptureSearch.ts:75) – diagnostics-only chain exploration on cloned boards.
  - Territory and line adapters: [`sandboxTerritory.ts`](src/client/sandbox/sandboxTerritory.ts:1), [`sandboxLines.ts`](src/client/sandbox/sandboxLines.ts:124).
  - Game-end and elimination wrappers: [`sandboxGameEnd.ts`](src/client/sandbox/sandboxGameEnd.ts:1), [`sandboxElimination.ts`](src/client/sandbox/sandboxElimination.ts:1), [`sandboxVictory.ts`](src/client/sandbox/sandboxVictory.ts:70).
  - Legacy local sandbox harness: [`localSandboxController.ts`](src/client/sandbox/localSandboxController.ts:1) – explicitly flagged in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.5](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:877) as `DIAGNOSTICS-ONLY (legacy)`; `/sandbox` no longer depends on it for live UI.
- These helpers may continue to be used for analysis and visualisations but must not be treated as authoritative rules surfaces once orchestrator-backed flows are canonical.

---

## 4. Python Mirror and Contracts

The Python implementation is a **parity-validated mirror and host** over the TS rules semantics SSoT; it is not an independent SSOT for rules.

**Architecture overview**

- Canonical Python rules engine: [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py:1).
  - Implements move generation (`get_valid_moves`) and application (`apply_move`) for all phases, mirroring TS shared engine semantics for geometry, placement, movement, capture chains, lines, territory (including Q23), forced elimination, and victory/LPS.
  - Enforces strict no-move invariants when enabled, as documented in [`docs/STRICT_INVARIANT_SOAKS.md`](docs/STRICT_INVARIANT_SOAKS.md:11).
- Python rules modules: [`ai-service/app/rules/core.py`](ai-service/app/rules/core.py:1), mutators under [`ai-service/app/rules/mutators`](ai-service/app/rules/mutators/placement.py:1), validators under [`ai-service/app/rules/validators`](ai-service/app/rules/validators/territory.py:1).
  - Act as a structured boundary mirroring TS aggregates and validators (see [`docs/PYTHON_PARITY_REQUIREMENTS.md`](docs/PYTHON_PARITY_REQUIREMENTS.md)).
- Shadow/adapter engine: [`ai-service/app/rules/default_engine.py`](ai-service/app/rules/default_engine.py:1).
  - Delegates canonical results to `GameEngine.apply_move` and uses mutators as a shadow contract, asserting parity per move but always returning the canonical `GameEngine` state.

**Contract vectors and parity requirements**

- TS orchestrator contract vectors under [`tests/fixtures/contract-vectors/v2`](tests/fixtures/contract-vectors/v2) encode language-neutral behaviour for placement, movement, capture, line detection, territory, and orchestrator-first flows (including chain capture and territory self-elimination sequences).
- TS runner [`tests/contracts/contractVectorRunner.test.ts`](tests/contracts/contractVectorRunner.test.ts:1) and Python runner [`ai-service/tests/contracts/test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py:1) must agree on:
  - `nextState` after applying a move via [`processTurn`](src/shared/engine/orchestration/turnOrchestrator.ts:1) (TS) vs `GameEngine.apply_move` (Python).
  - Full [`ProcessTurnResult`](src/shared/engine/orchestration/types.ts:1) payloads (status, pending decisions, S-invariant metadata) where applicable.
- Additional parity and invariants suites under [`ai-service/tests/parity`](ai-service/tests/parity/test_line_and_territory_scenario_parity.py:1) and [`ai-service/tests/invariants`](ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py:1) provide scenario-level confirmation that Python remains aligned with TS behaviour.

**Authority on disagreements**

- In any behavioural disagreement between TS and Python engines (contract vectors, parity fixtures, or orchestrator soaks):
  - The TS shared engine + orchestrator is authoritative.
  - Python code (`game_engine.py`, validators/mutators, training env) must be updated to match TS, with additional contract vectors or regression tests as needed.
- This precedence is explicit in [`docs/PYTHON_PARITY_REQUIREMENTS.md`](docs/PYTHON_PARITY_REQUIREMENTS.md:3) and [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:31).

---

## 5. Legacy / Diagnostic Surfaces

The table below summarises non-canonical or legacy modules that still exist in the tree and how they should be treated. It is a **classification aid** for future cleanup tasks (P17.6-CODE), not a deprecation plan on its own.

| Module / Surface                                            | Location                                                                                                                                     | Intended Status                                                                                                                                                                                                                        | Misuse Risk (if treated as canonical)                                                                                                                                                                                                                       |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Legacy backend turn pipeline                                | [`GameEngine` legacy branches](src/server/game/GameEngine.ts:1)                                                                              | `test-only / diagnostics-only` – used when tests explicitly disable the orchestrator adapter (`disableOrchestratorAdapter()`), or under specialised debugging harnesses.                                                               | Medium: behaviour may drift from orchestrator + aggregates over time, especially around decision phases, forced elimination, and LPS; using this path in production would reintroduce duplicate rules semantics and weaken invariant and parity guarantees. |
| Legacy `RuleEngine.processMove` and post-processing helpers | [`RuleEngine.ts`](src/server/game/RuleEngine.ts:46)                                                                                          | `diagnostic-only` – historical orchestration helpers around shared aggregates; no production call sites per [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.3](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:882).                                         | Medium: reusing these helpers instead of orchestrator would bypass `processTurn` and `PendingDecision` semantics, risking divergence in composite flows (capture → lines → territory) and undermining contract-vector coverage.                             |
| Sandbox legacy orchestration in `ClientSandboxEngine`       | [`ClientSandboxEngine.ts` non-adapter branches](src/client/sandbox/ClientSandboxEngine.ts:137)                                               | `diagnostic-only` – retained for historical parity tests and trace debugging; canonical sandbox path is via [`SandboxOrchestratorAdapter`](src/client/sandbox/SandboxOrchestratorAdapter.ts:1).                                        | Medium: wiring new sandbox UI or AI features against legacy paths would cause backend vs sandbox parity regressions and make orchestrator soaks less representative.                                                                                        |
| Sandbox movement and capture helpers                        | [`sandboxMovement.ts`](src/client/sandbox/sandboxMovement.ts:1), [`sandboxCaptures.ts`](src/client/sandbox/sandboxCaptures.ts:1)             | `UX / diagnostics-only` adapters over shared movement/capture aggregates, primarily used for highlighting and offline analysis.                                                                                                        | Low–Medium: as long as canonical legality and mutation flows go through orchestrator + aggregates, differences here mostly affect UI hints; using them as authoritative legality or mutation sources would reintroduce client-only rules variants.          |
| Sandbox chain-capture search helpers                        | [`sandboxCaptureSearch.ts`](src/client/sandbox/sandboxCaptureSearch.ts:75)                                                                   | `diagnostics-only (analysis tool)` – bounded DFS over capture trees on cloned boards, explicitly documented as non-SSOT.                                                                                                               | Low: used only in tests/diagnostics; risk arises only if wired into `getValidMoves` or AI legality checks.                                                                                                                                                  |
| Sandbox legacy local harness                                | [`localSandboxController.ts`](src/client/sandbox/localSandboxController.ts:1)                                                                | `diagnostics-only (legacy)` – historical browser-safe harness; `/sandbox` now uses `ClientSandboxEngine` + orchestrator exclusively.                                                                                                   | Low–Medium: reconnecting production UI flows to this harness would bypass orchestrator and shared aggregates completely.                                                                                                                                    |
| Python mutator-first `DefaultRulesEngine` path              | [`default_engine.py`](ai-service/app/rules/default_engine.py:1)                                                                              | `shadow / diagnostics-only` – uses mutators under shadow contract to check parity with canonical `GameEngine.apply_move`, but always returns the canonical engine result.                                                              | Low: as long as callers treat `GameEngine` as authoritative, divergence will surface as test failures rather than gameplay bugs; using mutator-first results directly as authoritative would weaken guarantees.                                             |
| Python strict no-move invariant soak harness                | [`run_self_play_soak.py`](ai-service/scripts/run_self_play_soak.py:1) under `RINGRIFT_STRICT_NO_MOVE_INVARIANT=1`                            | `diagnostic-only` – used to discover invariant violations and generate regression snapshots, not to define rules semantics.                                                                                                            | Low: risk is mainly operational (over-interpreting soak results without correlating with TS orchestrator soaks); semantics still come from shared TS engine + Python mirror.                                                                                |
| TS orchestrator invariant soak harness                      | [`scripts/run-orchestrator-soak.ts`](scripts/run-orchestrator-soak.ts:1)                                                                     | `diagnostic-only but near-SSOT for invariants` – canonical S-invariant and active-player-no-move checks for the TS orchestrator; informs rollout SLOs in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:876). | Low: misinterpreting soak outputs could affect rollout decisions, but not core rules semantics.                                                                                                                                                             |
| Capture-cycle diagnostics scripts                           | [`scripts/findCyclicCaptures.js`](scripts/findCyclicCaptures.js:1), [`scripts/findCyclicCapturesHex.js`](scripts/findCyclicCapturesHex.js:1) | `diagnostics-only` – offline exploration of capture cycles using shared engine helpers.                                                                                                                                                | Low: they depend on the shared engine and do not implement independent rules, but could be mistaken as prescriptive if not clearly bannered.                                                                                                                |

This table is **not exhaustive** but covers the highest-value surfaces for future cleanup and SSOT-banner hardening.

---

## 6. Drift & Ambiguity List (for Follow-up Tasks)

The items below are **known inconsistencies or ambiguities** between docs, code, and rollout state that should be addressed in follow-up work. Each bullet suggests likely task classification: P17.3-ASK (docs), P17.4-DEBUG (invariants/soaks), or P17.6-CODE (cleanup).

1. **Outdated note about orchestrator runbook location**
   - [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:81) still states that an `ORCHESTRATOR_ROLLOUT` runbook “remains planned but is not yet present under docs/runbooks”. The runbook now exists as [`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`](docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md:1).
   - _Follow-up:_ P17.3-ASK – Update that section to reference the existing runbook and align language with the rollout phases and SLOs in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:927).

2. **Historical wording about backend adapter migration**
   - [`docs/CANONICAL_ENGINE_API.md` §3.9.2](docs/CANONICAL_ENGINE_API.md:951) still describes the `TurnEngineAdapter` path as something that “will become the primary entry point as Phase 3 migration completes”, while [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.1](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:793) and [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:30) already treat the orchestrator-backed adapter as the canonical backend path (with legacy pipelines quarantined).
   - _Follow-up:_ P17.3-ASK – Refresh wording in `CANONICAL_ENGINE_API` to describe `TurnEngineAdapter` / `processTurnAsync` as the **current** canonical backend integration path, and explicitly mark `RuleEngine.processMove` and the legacy `GameEngine` pipeline as diagnostics-only.

3. **Partial-historical aggregate design doc vs implemented single-file aggregates**
   - [`docs/DOMAIN_AGGREGATE_DESIGN.md`](docs/DOMAIN_AGGREGATE_DESIGN.md:1) describes a multi-file-per-aggregate layout and labels several phases as “target design”, while the current implementation uses single-file aggregates (`*Aggregate.ts`) plus helpers. The doc header notes this is partially historical, but some sections still read as prescriptive.
   - _Follow-up:_ P17.3-ASK – Tighten the header and add a short “Current implementation vs target” summary, explicitly pointing readers to [`docs/MODULE_RESPONSIBILITIES.md`](docs/MODULE_RESPONSIBILITIES.md:1) and [`docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md`](docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md:41) for the live layout.

4. **SSOT emphasis for orchestrator vs legacy `RuleEngine` in mapping doc**
   - [`RULES_IMPLEMENTATION_MAPPING.md`](RULES_IMPLEMENTATION_MAPPING.md:43) correctly treats the shared TS engine as “Primary rules semantics”, but still gives substantial narrative weight to backend `RuleEngine` and legacy helpers that are now diagnostics-only per [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.3](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:882).
   - _Follow-up:_ P17.3-ASK / P17.6-CODE – Update the mapping’s backend sections to: (a) foreground orchestrator + aggregates + `TurnEngineAdapter` as the only canonical backend execution path, and (b) clearly tag legacy `RuleEngine` orchestration helpers as historical/diagnostic, ensuring there are no remaining production call sites.

5. **Invariant/soak story fragmented between TS and Python docs**
   - [`docs/STRICT_INVARIANT_SOAKS.md`](docs/STRICT_INVARIANT_SOAKS.md:1) focuses on Python’s strict no-move invariant and self-play soaks, while [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §6.2](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:456) and [`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md` §2–3](docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md:33) describe TS orchestrator invariant soaks and SLOs. The two views are consistent but live in separate documents with limited cross-linking.
   - _Follow-up:_ P17.3-ASK / P17.4-DEBUG – Addressed by the **Invariants & Soaks overview** subsection added to this SSOT map (see below), which cross-links TS orchestrator soaks, Python strict-invariant soaks, and the roll‑up metrics/SLOs used during rollout.

6. **Python type-level gaps vs TS engine types**
   - [`docs/PYTHON_PARITY_REQUIREMENTS.md` §2.11–§3.3](docs/PYTHON_PARITY_REQUIREMENTS.md:218) notes that some TS types/functions (e.g. `VictoryReason`, explicit `GameResult`, some core utility helpers) are only partially mirrored or inlined on the Python side. Behavioural parity is enforced via contract vectors, but type-level parity is not explicit.
   - _Follow-up:_ P17.3-ASK / P17.6-CODE – Decide whether to tighten Python type modelling for `VictoryReason`/`GameResult` and other noted gaps, or to explicitly document them as “intentionally simplified” with no behavioural impact, to avoid future confusion when evolving TS types.

7. **Sandbox diagnostics vs canonical `/sandbox` flows**
   - [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §7.2](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:811) and [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:90) describe `/sandbox` as fully orchestrator-backed, while some older analyses and tests still treat legacy sandbox engines and helpers (`sandboxTerritoryEngine`, `sandboxLinesEngine`) as active components. Those modules have been removed or demoted, but a few doc references and archived tests remain.
   - _Follow-up:_ P17.3-ASK / P17.6-CODE – Sweep docs and tests for references to removed sandbox engines, updating them to point at `ClientSandboxEngine` + `SandboxOrchestratorAdapter` and moving any remaining legacy tests into `archive/` with explicit SSOT banners.

---

This SSOT map is intended to remain relatively small and stable. When rules semantics, orchestrator behaviour, or host integration patterns change, updates should be expressed as:

- Targeted edits to the upstream SSoTs listed in §1, and
- Small adjustments here to keep the hierarchy, host maps, and drift table aligned with reality.

For new implementation work, treat this file as a **navigation aid**; the normative sources remain the rules spec, shared TS engine + orchestrator, and contract tests.

---

## 7. Invariants & Soaks Overview (TS + Python)

This section ties together the invariant and soak story across TS and Python so
that maintainers can see, at a glance, how progress/S‑invariant guarantees are
enforced across hosts and CI/rollout.

**Canonical invariant definition (S‑invariant and progress)**

- Formal rules: `RR-CANON-R120`–`R125` (progress and S‑invariant) in `RULES_CANONICAL_SPEC.md` and the commentary in `ringrift_compact_rules.md` §9.
- Executable S‑invariant helper (TS SSOT):
  - `computeProgressSnapshot(state: GameState): ProgressSnapshot` in
    `src/shared/engine/core.ts` with:
    - `S = markers + collapsedSpaces + eliminatedRings`
    - `eliminatedRings` derived from `totalRingsEliminated` /
      `board.eliminatedRings` when present.
  - Consumed by:
    - S‑invariant unit tests (`tests/unit/ProgressSnapshot.core.test.ts`,
      `tests/unit/SInvariant.seed17FinalBoard.test.ts`,
      `tests/unit/SharedMutators.invariants.test.ts`).
    - Orchestrator processing metadata (see `docs/CANONICAL_ENGINE_API.md` §3.9.3).

**TS orchestrator invariant soaks (adapter‑ON harness)**

- Harness: `scripts/run-orchestrator-soak.ts` and the `soak:orchestrator:*` npm scripts.
- Behaviour:
  - Drives backend `GameEngine` under `TurnEngineAdapter` (orchestrator‑ON) for many
    random self‑play games across board types.
  - Asserts that:
    - `S` (via `computeProgressSnapshot`) is **globally non‑decreasing**.
    - There is no long‑running “active + no‑move” stall for the current player.
  - Emits JSON summaries (`results/orchestrator_soak_smoke.json`,
    `results/orchestrator_soak_nightly.json`) used by:
    - CI jobs `orchestrator-soak-smoke` / `orchestrator-short-soak`
      (`SLO-CI-ORCH-SHORT-SOAK` in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §6.2).
    - Nightly soak workflow and dashboards/alerts (`ringrift_orchestrator_invariant_violations_total`).

**Python strict no‑move invariant soaks (mirror harness)**

- Harness: `ai-service/scripts/run_self_play_soak.py` with `RINGRIFT_STRICT_NO_MOVE_INVARIANT=1`.
- Behaviour:
  - Drives the Python `GameEngine` mirror with stricter “no‑move” invariants, treating
    any active state with no legal moves for the current player as a hard violation.
  - Used to:
    - Discover deep invariant/stall edge cases.
    - Generate regression snapshots and seeds promoted into:
      - Python invariants suites (`ai-service/tests/invariants/**`),
      - TS orchestrator S‑invariant regressions
        (`tests/unit/OrchestratorSInvariant.regression.test.ts`),
      - and v2 contract vectors (`tests/fixtures/contract-vectors/v2/**`).

**How they work together during rollout**

- TS orchestrator soaks are the **primary** source for rollout SLOs:
  - `SLO-CI-ORCH-SHORT-SOAK` (CI),
  - `SLO-STAGE-ORCH-INVARIANTS` (staging),
  - `SLO-PROD-ORCH-INVARIANTS` (production) in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §6.2–6.4.
- Python strict‑invariant soaks act as a **parity mirror and bug‑discovery tool**:
  - When they find a violation, the seed/scenario is promoted into:
    - TS orchestrator regression tests,
    - shared contract vectors (`sInvariantDelta` fields),
    - or orchestrator soak regression lists.
- Both harnesses ultimately rely on the same S‑invariant definition (`computeProgressSnapshot`)
  and feed into shared observability surfaces:
  - Metrics from `MetricsService.ts` (`ringrift_orchestrator_invariant_violations_total`,
    error/circuit‑breaker/rollout metrics).
  - Alerts and dashboards described in `docs/ALERTING_THRESHOLDS.md`.

When adding new rules that affect markers, collapsed spaces, or eliminated rings,
update `computeProgressSnapshot` and its tests first, then ensure:

- TS orchestrator soaks (`run-orchestrator-soak.ts`) still treat S as non‑decreasing, and
- Python strict‑invariant soaks and invariants suites remain consistent or are updated
  with explicit commentary where host‑level behaviour diverges by design.

---

## 8. Protected Artefacts and SSOT Guards

This section documents the automated protection system that guards critical parity
artefacts and orchestrator configuration from accidental modification without proper
validation.

### 8.1 Protection Level Overview

Protected artefacts are organised into categories with two protection levels:

| Level      | Meaning                                                                | CI Behaviour                                    |
| ---------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| **HIGH**   | Semantic authority – changes affect rules correctness across languages | Must pass full parity validation suite          |
| **MEDIUM** | Configuration/operational – changes affect rollout or deployment       | Must pass deployment validation and SSOT checks |

### 8.2 Protected Categories

**Category: `contract-vectors` (HIGH protection)**

Cross-language contract behaviour vectors defining expected TS↔Python behaviour.

- **Patterns:**
  - `tests/fixtures/contract-vectors/**/*.json`
  - `tests/fixtures/contract-vectors/v2/*.json`
- **Validation requirement:** All contract vector modifications require:
  1. TS contract tests pass (`npm run test:orchestrator-parity`)
  2. Python contract tests pass (`./scripts/run-python-contract-tests.sh --verbose`)
  3. Parity healthcheck green

**Category: `parity-infrastructure` (HIGH protection)**

Python parity healthcheck and contract test infrastructure.

- **Patterns:**
  - `ai-service/scripts/run_parity_healthcheck.py`
  - `ai-service/tests/contracts/*.py`
  - `scripts/run-python-contract-tests.sh`
- **Validation requirement:** Modifications require all Python contract and parity tests to pass.

**Category: `orchestrator-configuration` (MEDIUM protection)**

Orchestrator rollout configuration and feature flag documentation.

- **Patterns:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`
  - `docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md`
  - `src/server/config/env.ts`
  - `src/server/config/unified.ts`
  - `src/server/services/OrchestratorRolloutService.ts`
- **Validation requirement:** Orchestrator config changes require deployment validation (`npm run validate:deployment`) and SSOT checks (`npm run ssot-check`) to pass.

**Category: `rules-ssot-core` (HIGH protection)**

Core TypeScript shared engine modules that define semantic authority.

- **Patterns:**
  - `src/shared/engine/core.ts`
  - `src/shared/engine/types.ts`
  - `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `src/shared/engine/orchestration/types.ts`
  - `src/shared/engine/aggregates/*.ts`
- **Validation requirement:** Core modifications require:
  1. TS rules tests pass (`npm run test:ts-rules-engine`)
  2. Orchestrator parity pass (`npm run test:orchestrator-parity`)
  3. Python parity pass (`./scripts/run-python-contract-tests.sh --verbose`)

**Category: `contract-vector-generation` (HIGH protection)**

Scripts that generate contract vectors.

- **Patterns:**
  - `scripts/generate-extended-contract-vectors.ts`
  - `tests/scripts/generate_rules_parity_fixtures.ts`
- **Validation requirement:** Changes require regeneration and validation of all vectors.

**Category: `python-rules-modules` (MEDIUM protection)**

Python rules engine implementation (parity mirror, not SSOT).

- **Patterns:**
  - `ai-service/app/rules/*.py`
  - `ai-service/app/rules/mutators/*.py`
  - `ai-service/app/rules/validators/*.py`
  - `ai-service/app/game_engine.py`
- **Validation requirement:** Python rules modifications require all Python tests and contract parity to pass.

### 8.3 Automated Protection Mechanisms

**CI Integration**

The protection system is enforced via the `ssot-check` CI job, which:

- Runs `npm run ssot-check` on all PRs and pushes to `main`/`develop`
- Includes the `parity-protection-ssot` check that validates:
  - All protected artefact locations exist and are structurally valid
  - Contract vector files are valid JSON with expected structure
  - Required parity infrastructure files are present
  - Orchestrator configuration has expected content

**Pre-commit Warning**

A non-blocking pre-commit hook warns developers when modifying protected files:

- Runs `scripts/ssot/check-protected-artefacts.ts --staged`
- Shows which protection categories are affected
- Lists validation commands to run locally
- Does **not** block the commit (full validation happens in CI)
- Can be skipped via `SKIP_PROTECTED_ARTEFACT_CHECK=1`

### 8.4 Running Validation Locally

Before submitting changes to protected artefacts:

```bash
# Run full SSOT check suite
npm run ssot-check

# For contract vector changes
npm run test:orchestrator-parity
./scripts/run-python-contract-tests.sh --verbose

# For orchestrator config changes
npm run validate:deployment

# For rules SSOT core changes
npm run test:ts-rules-engine
npm run test:orchestrator-parity
./scripts/run-python-contract-tests.sh --verbose
```

### 8.5 Configuration Reference

The protection configuration is defined in:

- [`scripts/ssot/protected-artefacts.config.ts`](../scripts/ssot/protected-artefacts.config.ts:1) – Category definitions, patterns, and validation commands
- [`scripts/ssot/parity-protection-ssot-check.ts`](../scripts/ssot/parity-protection-ssot-check.ts:1) – CI-level structural validation
- [`scripts/ssot/check-protected-artefacts.ts`](../scripts/ssot/check-protected-artefacts.ts:1) – Pre-commit warning script

To add a new protected category or modify patterns, update `protected-artefacts.config.ts`
and ensure the CI check passes before merging.
