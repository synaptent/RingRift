# RingRift Test Suite Parity & Classification Plan

> **Doc Status (2025-11-27): Active (test meta-doc, non-semantics)**
>
> **Role:** Map the existing TS + Python test suites to the shared rules/lifecycle SSoTs and classify them (rules-level vs trace-level vs integration-level, by host/domain) so that future work can extend or retire tests systematically, and can distinguish authoritative rules suites from derived trace/parity harnesses. This is a **test/meta reference only** – it explains how suites are organised and which ones act as semantic anchors vs smoke tests, but it does **not** define game rules or lifecycle semantics.
>
> **Not a semantics SSoT:** Canonical rules and lifecycle semantics are owned by the shared TypeScript rules engine and contracts/vectors (`src/shared/engine/**`, `src/shared/engine/contracts/**`, `tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`) together with the written rules and lifecycle docs (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/CANONICAL_ENGINE_API.md`). When this file talks about “authority” or “canonical behaviour” it is always pointing back to those SSoTs and to specific rules-level test suites built on top of them.
>
> **Related docs:** `tests/README.md`, `tests/TEST_LAYERS.md`, `docs/PARITY_SEED_TRIAGE.md`, `RULES_SCENARIO_MATRIX.md`, `AI_ARCHITECTURE.md`, `docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/STRICT_INVARIANT_SOAKS.md`, and `DOCUMENTATION_INDEX.md`.
>
> **Goal:** Provide a concrete map of the existing tests so future work can systematically align **all TS + Python tests** to the **shared TypeScript rules engine** and the **rules-over-traces** strategy.
>
> This file complements `tests/README.md` (how to run tests, utilities) and `RULES_SCENARIO_MATRIX.md` (rules→scenario mapping) by focusing specifically on:
>
> - **Classification** of suites as rules-level vs trace-level vs integration-level.
> - **Host domain** (shared engine, backend adapter, sandbox adapter, Python AI-service, cross-host parity).
> - **Where each parity harness gets its semantic authority** (which rules-level suites/fixtures anchor it).
> - **Where new/missing rules-level coverage is still needed.**

---

## 0. Canonical rules sources

All semantic decisions should ultimately be anchored to:

1. **Shared TS rules engine** – `src/shared/engine/`
   - `core.ts`, movement/capture/line/territory/victory helpers, `aggregates/*.ts`, `orchestration/turnOrchestrator.ts`, `orchestration/phaseStateMachine.ts`, and contracts under `src/shared/engine/contracts/*`.
2. **Written rules** – `ringrift_complete_rules.md` (plus `RULES_SCENARIO_MATRIX.md`).
3. **Contract vectors + parity harnesses** – `tests/fixtures/contract-vectors/v2/*.json`, `tests/contracts/contractVectorRunner.test.ts`, and Python counterparts under `ai-service/tests/contracts/` and `ai-service/tests/parity/`.

Legacy TS engines (`src/server/game/*`, `src/client/sandbox/*`) and the Python rules engine (`ai-service/app/rules/*`, `ai-service/app/game_engine.py`) are **hosts/adapters** that must conform to this shared semantics.

Recorded traces (seeded games, AI-vs-AI runs, historic move logs) are **derived artifacts** and must yield when they conflict with the shared engine + contract vectors + rules docs. See `tests/README.md` for the **seed‑14 precedent** and the detailed rules‑over‑traces policy.

---

## 1. Shared‑engine–anchored rules-level suites (TS)

These suites are (or should be) treated as **authoritative** for rules semantics.

| File                                                        | Level                         | Domain                               | Notes / Alignment with `src/shared/engine`                                                                                                                                          | Gaps / TODO                                                                                                                         |
| ----------------------------------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/RefactoredEngine.test.ts`                       | Rules-level                   | TS shared engine                     | Directly exercises `src/shared/engine/GameEngine` and related types.                                                                                                                | Keep adding minimal, focused rules cases here when the change is **purely engine-level**.                                           |
| `tests/unit/RefactoredEngineParity.test.ts`                 | Rules-level + adapter parity  | TS shared engine ↔ backend/sandbox  | Verifies that backend/sandbox adapters behave like the refactored shared engine. Shared engine is canonical.                                                                        | For any failure: fix adapters or add a more targeted rules-level test; do **not** diverge shared engine to preserve adapter quirks. |
| `tests/unit/MovementAggregate.shared.test.ts`               | Rules-level                   | Shared engine (MovementAggregate)    | Authoritative for non-capture movement validation, enumeration, and mutation via `MovementAggregate` across square/hex boards.                                                      | Use as primary reference when adjusting movement rules; keep backend/sandbox movement tests aligned with this suite.                |
| `tests/unit/LineDetectionParity.rules.test.ts`              | Rules-level                   | Shared engine, BoardManager, sandbox | Authoritative for line detection semantics (markers only, blocking stacks/collapsed spaces, square vs hex geometry).                                                                | Use as primary reference when adjusting line logic anywhere (TS or Python).                                                         |
| `tests/unit/sandboxTerritory.rules.test.ts`                 | Rules-level                   | Shared engine ↔ sandbox             | Territory rules expressed via sandbox engine but logically anchored to shared semantics.                                                                                            | Ensure key cases are also covered in pure shared-engine tests over time.                                                            |
| `tests/unit/territoryProcessing.rules.test.ts`              | Rules-level                   | Shared engine ↔ backend             | Territory processing semantics aligned with rules doc.                                                                                                                              | Same as above: prefer pure shared-engine coverage where possible.                                                                   |
| `tests/unit/sandboxTerritoryEngine.rules.test.ts`           | Rules-level                   | Shared engine ↔ sandbox             | Legacy territory engine rules (now superseded by shared helpers and `ClientSandboxEngine` territory integration).                                                                   | Keep as historical/diagnostic until fully replaced by shared-engine + ClientSandboxEngine suites.                                   |
| `tests/unit/Seed14Move35LineParity.test.ts`                 | Rules-level, seed-specific    | Shared engine, backend, sandbox      | Codifies that after moves 1–34 of the historic seed‑14 trace, **no valid lines exist**, and `process_line` is invalid. This is the canonical resolution of the seed‑14 discrepancy. | Must remain green whenever line rules are modified; if it breaks, treat as a semantics regression until proven otherwise.           |
| `tests/unit/SInvariant.seed17FinalBoard.test.ts`            | Rules-level invariant         | Shared engine                        | Asserts S‑invariant properties of a particular final board.                                                                                                                         | Use as pattern for additional invariant-based rules tests.                                                                          |
| `tests/unit/ProgressSnapshot.core.test.ts`                  | Rules-level invariant         | Shared engine types                  | Validates `ProgressSnapshot` behaviour (S = markers + collapsed + eliminated) independent of hosts.                                                                                 | Extend as new invariant fields are added.                                                                                           |
| `tests/unit/ProgressSnapshot.sandbox.test.ts`               | Rules-level + adapter check   | Shared engine ↔ sandbox             | Ensures sandbox emits consistent progress snapshots matching core expectations.                                                                                                     | Use core tests as authority; adjust sandbox if they diverge.                                                                        |
| `tests/unit/sandboxTerritoryEngine.rules.test.ts`           | Rules-level                   | Shared engine ↔ sandbox             | Diagram-level coverage for territory engine.                                                                                                                                        | Mirror high-value fixtures to Python parity.                                                                                        |
| `tests/unit/LineDetectionParity.rules.test.ts`              | Rules-level                   | Shared engine ↔ backend ↔ sandbox  | (duplicate entry for emphasis)                                                                                                                                                      | —                                                                                                                                   |
| `tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts`   | Rules-level scenarios         | Legacy RuleEngine (TS backend)       | Movement rules as per rules doc. Ultimately should be mirrored/refreshed via `src/shared/engine`.                                                                                   | Over time, prefer shared-engine variants of these scenarios, keeping RuleEngine tests as adapter checks.                            |
| `tests/scenarios/RulesMatrix.*.GameEngine.test.ts`          | Rules-level scenarios         | Backend GameEngine adapter           | Scenario matrix for Movement, Territory, Elimination, Victory.                                                                                                                      | Pair each scenario with an equivalent shared-engine test where feasible.                                                            |
| `tests/scenarios/RulesMatrix.*.ClientSandboxEngine.test.ts` | Rules-level scenarios         | Sandbox adapter                      | Same as above, from sandbox perspective.                                                                                                                                            | Same guidance: use shared-engine as semantic anchor; sandbox tests verify adapter.                                                  |
| `tests/scenarios/LineAndTerritory.test.ts`                  | Rules-level compound scenario | Backend GameEngine                   | End-to-end line + territory rules.                                                                                                                                                  | Where ambiguity arises, defer to focused rules-level suites like `LineDetectionParity.rules.test.ts` and territory rules tests.     |
| `tests/scenarios/ForcedEliminationAndStalemate.test.ts`     | Rules-level scenario          | Backend GameEngine                   | Turn sequence, forced elimination, and S-invariant integration.                                                                                                                     | Ensure Python parity fixtures reflect these semantics.                                                                              |

> **Guideline:** when you discover or fix a rules issue, first add/extend a **rules-level test** here (or in a new `*.rules.test.ts` suite) against `src/shared/engine`. Then adapt backend, sandbox, and Python hosts to satisfy these tests.

---

## 2. TS trace-level parity harnesses

Trace-level suites are **smoke tests and regression nets**. They get their semantics from the rules-level suites above and may need to **retire or regenerate seeds** when semantics change.

### 2.1 Backend vs Sandbox trace parity & diagnostics

| File                                                              | Level                                          | Domain                                      | Current semantic anchors                                                                                                                                     | Policy / TODO                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`       | Trace-level parity (archived, diagnostic only) | Backend GameEngine ↔ ClientSandboxEngine   | `RefactoredEngine.test.ts`, `RefactoredEngineParity.test.ts`, `LineDetectionParity.rules.test.ts`, `Seed14Move35LineParity.test.ts`, territory rules suites. | **Archived diagnostic harness.** Seeds list is historical (e.g. seed 5). If a seed fails but all rules-level tests pass, treat the trace as **stale** and fix via rules-level/fixture tests instead; do not treat this file as a semantic authority or CI gate. |
| `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`               | Trace-level parity                             | Backend AIEngine ↔ Sandbox AI (shared RNG) | RNG wiring in `localAIMoveSelection.ts` and RNG-focused tests noted in `tests/README.md`.                                                                    | Use to verify RNG injection & parity. If semantics shift, update rules-level tests first, then adjust expectations here.                                                                                                                                        |
| `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`           | Trace-level parity (diagnostic, often skipped) | Same as above, but with deeper sequences    | Same as above.                                                                                                                                               | Keep as **diagnostic only** (e.g. `describe.skip` by default). Use when debugging deep AI parity issues.                                                                                                                                                        |
| `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`       | Trace-level coverage                           | Backend heuristic AI ↔ sandbox AI          | Shared engine semantics and AI heuristics tests.                                                                                                             | When this fails yet rules-level tests are green, assume either AI-specific mismatch or stale traces; debug using logs/diagnostics before altering rules.                                                                                                        |
| `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`          | Trace-level, seed-specific debug               | Backend ↔ sandbox                          | Same anchors as main traceParity suite.                                                                                                                      | Treat as a debug harness. Once issues are resolved and covered by rules-level tests, either trim or move to a `debug/` or `archive/` folder.                                                                                                                    |
| `archive/tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts` | Trace-level, seed-specific debug (archived)    | Backend ↔ sandbox                          | Seed‑17-specific.                                                                                                                                            | **Archived diagnostic harness.** Kept for historical debugging only; any important behaviour should be captured in shared-engine rules or scenario tests instead.                                                                                               |
| `tests/unit/ParityDebug.seed5.trace.test.ts`                      | Trace-level debug                              | Backend ↔ sandbox                          | Uses GameTrace helpers.                                                                                                                                      | Consider consolidating debug patterns; ensure any semantics uncovered here are reflected in rules-level tests.                                                                                                                                                  |
| `tests/unit/ParityDebug.seed14.trace.test.ts`                     | Trace-level debug                              | Backend ↔ sandbox                          | Superseded in semantics by `Seed14Move35LineParity.test.ts`.                                                                                                 | Keep for debugging, but treat `Seed14Move35LineParity` as the canonical semantic authority.                                                                                                                                                                     |
| `tests/unit/TraceParity.seed5.firstDivergence.test.ts`            | Trace-level diagnostic                         | Backend ↔ sandbox                          | Same as seed 5 anchors.                                                                                                                                      | Use to locate first divergence. If divergence is due to rules semantics, capture that case in a rules-level test.                                                                                                                                               |
| `tests/unit/TraceParity.seed14.firstDivergence.test.ts`           | Trace-level diagnostic                         | Backend ↔ sandbox                          | Same as seed 14 anchors.                                                                                                                                     | Historical; semantics now captured by `Seed14Move35LineParity`.                                                                                                                                                                                                 |
| `archive/tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`   | Trace-level + performance/parallel (archived)  | Backend ↔ sandbox                          | GameTrace helpers and AI RNG wiring.                                                                                                                         | **Archived diagnostic harness.** Use only as a local debugging/profiling tool; not a CI target and not a source of rules truth.                                                                                                                                 |

### 2.2 Movement / capture / territory / victory parity suites

These are parity-focused but often test specific scenarios. They should be **backed by rules-level counterparts** where semantics are subtle.

| File                                                               | Level                                       | Domain                        | Current semantic anchors                                                                                                 | Rules-level counterpart (existing or TODO)                                                                                                                                                      |
| ------------------------------------------------------------------ | ------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/Backend_vs_Sandbox.eliminationTrace.test.ts`           | Trace-level parity                          | Backend GameEngine ↔ sandbox | Elimination and territory traces derived from shared-engine helpers and `RULES_SCENARIO_MATRIX.md`.                      | Ensure elimination and territory scenarios are covered in shared-engine rules tests and contract vectors.                                                                                       |
| `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts`        | Trace-level parity (multi-domain, archived) | Backend GameEngine ↔ sandbox | Movement, capture, line, and territory traces anchored to shared-engine rules suites and contract vectors.               | **Archived diagnostic harness.** When this reveals an issue, add or extend focused shared-engine rules or fixture-based tests first; do not re-introduce it as a CI gate or semantic authority. |
| `tests/unit/Backend_vs_Sandbox.seed*.snapshotParity.test.ts`       | Trace-level snapshot parity                 | Backend GameEngine ↔ sandbox | Seed-specific snapshot parity over full-game traces (e.g. seed 1, 5, 18) using shared-engine semantics as authority.     | Treat as smoke/regression nets; add dedicated rules-level tests when a seed reveals a true semantics issue.                                                                                     |
| `archive/tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` | Trace-level parity (archived)               | Territory parity              | `GameEngine.territory.scenarios.test.ts`, `BoardManager.territoryDisconnection.*`, shared-engine territory rules tests.  | **Archived diagnostic harness.** Canonical territory semantics now live in shared decision-helper tests and RulesMatrix territory suites; keep this file for historical reference only.         |
| `tests/unit/TerritoryCore.GameEngine_vs_Sandbox.test.ts`           | Trace-level parity                          | Territory core behaviour      | Core territory processing + elimination semantics, anchored to `territoryProcessing.shared.test.ts` and related helpers. | Keep core shared-engine territory tests as the semantic authority; treat this suite as a host-parity check.                                                                                     |
| `tests/unit/TraceFixtures.sharedEngineParity.test.ts`              | Trace-level shared-engine parity            | Shared engine                 | Uses shared-engine-generated trace fixtures as a smoke test over contract vectors and helper behaviour.                  | Keep contract vector suites (`tests/contracts/*`, `tests/fixtures/contract-vectors/v2/*.json`) as the primary authority.                                                                        |

> **TODO:** For each parity suite above, maintain a short comment at the top linking to its primary rules-level counterparts (shared-engine tests and/or specific scenario suites).

---

## 3. Backend & sandbox behaviour / integration suites

These suites validate orchestration, WebSocket flows, AI service calls, and UI wiring. They should **not** redefine rules semantics, but they may depend on them.

### 3.1 Backend GameEngine behaviour

Representative files:

- `tests/unit/GameEngine.turnSequence.scenarios.test.ts`
- `tests/unit/GameEngine.chainCapture.test.ts`
- `tests/unit/GameEngine.chainCapture.triangleAndZigZagState.test.ts`
- `tests/unit/GameEngine.cyclicCapture.scenarios.test.ts`
- `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`
- `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`
- `tests/unit/GameEngine.territory.scenarios.test.ts`
- `tests/unit/GameEngine.territoryDisconnection.test.ts`
- `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`
- `tests/unit/GameEngine.victory.scenarios.test.ts`
- `tests/unit/GameEngine.decisionPhases.MoveDriven.test.ts`
- `tests/unit/GameEngine.captureDirectionChoice.test.ts`
- `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`
- `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts`
- `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`
- `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`
- `tests/unit/GameEngine.lines.scenarios.test.ts`
- `tests/unit/GameEngine.aiSimulation.test.ts`
- `tests/unit/GameEngine.aiSimulation.debug.test.ts`
- `tests/unit/GameEngine.aiSimulation.seed10.debug.test.ts`

**Classification:**

- Primarily **behaviour / integration** suites focused on backend orchestration and decision phases.
- They should:
  - Defer to shared-engine rules tests for semantic correctness.
  - Be updated when rules change, using shared-engine tests as the reference.

### 3.2 Client sandbox behaviour

Representative files (modern canonical + legacy/diagnostic suites):

- **Canonical sandbox invariants & trace structure**
  - `tests/unit/ClientSandboxEngine.invariants.test.ts`
  - `tests/unit/ClientSandboxEngine.traceStructure.test.ts`
- **Placement, forced elimination, and mixed human/AI flows**
  - `tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts`
  - `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`
- **Movement, markers, and backend parity**
  - `tests/unit/ClientSandboxEngine.movementParity.shared.test.ts`
  - `tests/unit/ClientSandboxEngine.placement.shared.test.ts`
  - `tests/unit/ClientSandboxEngine.moveParity.test.ts`
  - `tests/unit/ClientSandboxEngine.landingOnOwnMarker.test.ts`
- **Chain capture and capture decisions**
  - `tests/unit/ClientSandboxEngine.chainCapture.test.ts`
  - `tests/unit/ClientSandboxEngine.chainCapture.scenarios.test.ts`
  - `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`
- **Lines, territory, and region order**
  - `tests/unit/ClientSandboxEngine.lines.test.ts`
  - `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`
  - `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`
  - `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`
  - `tests/unit/ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts`
- **Victory / LPS cross‑interaction**
  - `tests/unit/ClientSandboxEngine.victory.LPS.crossInteraction.test.ts`
  - `tests/unit/LPS.CrossInteraction.Parity.test.ts`
- **Sandbox AI simulations and stall diagnostics (diagnostic/opt‑in)**
  - `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`
  - `tests/unit/ClientSandboxEngine.aiStallRegression.test.ts`
  - `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts`

**Classification:**

- Mix of **adapter-level rules checks** and **behaviour/integration**.
- They should:
  - Rely on shared-engine rules tests for semantics.
  - Explicitly reference corresponding rules-level suites in comments (e.g. for lines, territory, chain-captures, LPS/victory parity).

### 3.3 WebSocket, AI service, and UI integration

Representative files:

- `tests/unit/AIEngine.serviceClient.test.ts`
- `tests/unit/AIEngine.placementMetadata.test.ts`
- `tests/unit/AIInteractionHandler.test.ts`
- `tests/unit/WebSocketServer.humanDecisionById.integration.test.ts`
- `tests/unit/PlayerInteractionManager.test.ts`
- `tests/unit/WebSocketInteractionHandler.test.ts`
- `tests/unit/GameHUD.snapshot.test.tsx`
- `tests/unit/GameEventLog.snapshot.test.tsx`
- `tests/integration/FullGameFlow.test.ts`

**Classification:**

- **Integration-level** suites.
- Use shared-engine rules suites for semantics; these tests assert that the correct events/choices are transported and presented.

---

## 4. Python AI-service tests & TS↔Python parity

Python tests live under `ai-service/tests/`. From a parity perspective, there are two main categories:

### 4.1 Python core/behaviour tests (non-parity)

Representative files:

- `ai-service/tests/test_engine_correctness.py`
- `ai-service/tests/test_env_interface.py`
- `ai-service/tests/test_action_encoding.py`
- `ai-service/tests/archive/archived_test_determinism.py` (historical; archived, superseded by `test_engine_determinism.py` and `test_no_random_in_rules_core.py`; kept only as a non-canonical diagnostic reference)
- `ai-service/tests/test_model_architecture.py`
- `ai-service/tests/test_heuristic_ai.py`
- `ai-service/tests/test_minimax_ai.py`
- `ai-service/tests/test_mcts_ai.py`
- `ai-service/tests/test_heuristic_training_evaluation.py`
- `ai-service/tests/test_multi_start_evaluation.py`
- `ai-service/tests/test_eval_randomness_integration.py`

**Classification:**

- Python‑only **rules/behaviour and AI tests**.
- `test_heuristic_training_evaluation.py`, `test_multi_start_evaluation.py`, and `test_eval_randomness_integration.py` specifically anchor the **heuristic training / plateau diagnostics** story:
  - Sanity‑check the shared fitness harness (`evaluate_fitness` in `scripts/run_cmaes_optimization.py`) by comparing `heuristic_v1_balanced` to a degenerate all‑zero profile and asserting that position evaluation depends on the active weights.
  - Exercise the `eval_mode="multi-start"` / `state_pool_id` code path using temporary JSONL state pools via `app/training/eval_pools.py`, ensuring multi‑start evaluation stays deterministic and correctly wired to the `RingRiftEnv` env factory.
  - Integration-test the `eval_randomness` parameter of `evaluate_fitness` to ensure that both `eval_randomness=0.0` and small non-zero values remain seed-deterministic, guarding against hidden RNG sources or tie-breaking refactors that would reintroduce non-determinism into CMA-ES/GA evaluation.
- Should be brought into line with shared-engine semantics by consuming **TS-generated fixtures** where feasible (see below), but they are already treated as **sanity/diagnostic gates** for the heuristic harness rather than parity authorities for rules semantics.

### 4.2 Python rules parity suites

Representative files:

- `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`
- `ai-service/tests/parity/generate_test_vectors.py`
- TS counterpart: `tests/unit/Python_vs_TS.traceParity.test.ts`
- **Archived (historical):** `ai-service/tests/archive/archived_test_rules_parity.py`

**Current state:**

- Parity tests compare Python rules/engine against TS behaviour, primarily via TS-generated fixtures (`test_rules_parity_fixtures.py`, `generate_test_vectors.py`) and targeted scenario suites.
- Older subprocess-based parity harnesses (`test_rules_parity.py`) are archived under `ai-service/tests/archive/` and no longer run by default.

**Target state:**

- **Fixture-driven rules parity**:
  - TS shared engine generates canonical fixtures (see §5 below).
  - Python tests load and validate against these fixtures.
  - `Python_vs_TS.traceParity.test.ts` is treated as a **trace-level smoke test**, not a semantic authority.
  - Archived parity suites under `archive/tests/**` and `ai-service/tests/archive/**` remain available for historical/diagnostic reference but are not part of CI gates.

---

## 5. Planned TS→Python shared fixtures (outline)

This section sketches where fixture-based parity is expected to live; implementation details will be handled in a dedicated task.

### 5.1 Fixture formats & locations (TS side)

- Proposed TS fixtures directory: `tests/fixtures/rules-parity/`.
- Fixture types:
  - **State-only fixtures**: serialized `GameState` objects (as defined in `src/shared/engine/types.ts`).
  - **State+action fixtures**: `{ state, action, expectedValidatorResult, expectedNextState, expectedEvents }`.
  - **Scenario fixtures**: sequences of actions with intermediate invariant checks.
- TS generator script (planned): `tests/scripts/generate_rules_parity_fixtures.ts` that:
  - Uses `src/shared/engine/GameEngine` to produce deterministic outputs.
  - Writes versioned JSON (e.g. `v1_line_territory.json`, `v1_chain_capture.json`).

### 5.2 Python consumption

- Python fixtures directory: `ai-service/tests/parity/fixtures/`.
- `test_rules_parity.py` and `test_line_and_territory_scenario_parity.py` should:
  - Load TS-generated JSON.
  - Map JSON into Python models defined in `ai-service/app/rules/interfaces.py`.
  - Assert that Python rules engine (`ai-service/app/rules/default_engine.py` / `game_engine.py`) matches TS outcomes.

### 5.3 CI considerations

- A future CI job should:
  1. Run the TS fixture generator script.
  2. Commit or expose fixtures to the Python test environment.
  3. Run Python parity tests (`pytest -m rules_parity` or similar tag) against those fixtures.

---

## 6. How to extend or modify tests using this plan

When you add or modify tests, follow this decision path:

1. **Is this about core rules semantics?**
   - Yes → Add/extend a **rules-level** test anchored in `src/shared/engine` (and/or a scenario suite) and tag it with `.rules.` or otherwise clearly as rules-level.
2. **Is this about comparing hosts (backend vs sandbox vs Python)?**
   - Yes → Use/extend a **trace-level parity** suite, but:
     - Treat recorded traces as **derived**.
     - If a trace conflicts with rules-level tests, update or retire the trace (seed‑14 pattern) and add a dedicated rules-level scenario.
3. **Is this about transport, orchestration, or UI?**
   - Yes → Add/extend an **integration-level** test (HTTP, WebSocket, AI-service calls, React components), and lean on rules-level tests to define semantics.
4. **Are you touching Python rules?**
   - Yes → Prefer to use **TS-generated fixtures** once available, and keep `ai-service/tests/parity/*` aligned with TS shared-engine semantics.

By maintaining this file alongside `tests/README.md` and `RULES_SCENARIO_MATRIX.md`, future work on RingRift’s tests can:

- Quickly see where to add a new case.
- Decide whether a failing test indicates a **rules regression** or a **stale trace**.
- Keep TS backend, client sandbox, and Python AI-service aligned with the shared rules engine and documented rules.

## 7. Decision-surface Move model status (P0 Task 18)

This section records, at a high level, which decision surfaces are currently driven end‑to‑end by the unified Move model and which still rely on legacy choice structures or ad‑hoc flows.

### 7.1 Backend (server rules + game engine)

- Movement, overtaking captures, and chain‑capture continuation are fully Move‑driven:
  - Valid options are always exposed as concrete Moves (including continue‑chain variants) and applied via the same Move application pipeline as placements and normal movement.
  - Chain‑capture tests in the RulesMatrix suites assert over these canonical Moves and resulting states.
- Line and territory decisions are partially Move‑driven:
  - Canonical Move variants exist for line processing, line‑reward selection, territory‑region processing, and elimination‑from‑stack, and are enumerated/validated by the backend rules engine.
  - Some legacy decision paths still execute effects directly from PlayerChoice payloads (line order, line reward option, region order, ring‑elimination choices), even though each option is now annotated with a stable move identifier that can be mapped back to the canonical Move set.
  - Target state for this task is that all such choices are re‑applied by resolving the chosen move identifier into a valid Move and feeding it through the same Move pipeline, leaving PlayerChoice as a thin UI/transport wrapper.

### 7.2 Client sandbox engine

- The sandbox maintains the shared GameState shape and records a structured history of canonical Moves; parity/debug paths already apply backend‑style Moves directly through a single canonical Move applier.
- For human‑driven play, placement and movement/capture use local helpers and click handling, but their effects and history entries are aligned with the canonical Move model.
- Line processing in the sandbox offers a canonical decision surface via Move enumeration and application helpers; current sandbox UX still auto‑selects lines and defaults line‑reward behaviour, rather than prompting the user to choose among explicit Moves.
- Territory and elimination decisions have canonical enumeration and application helpers in the sandbox that mirror the backend decision Moves, but normal sandbox territory resolution still flows through RegionOrderChoice and related helpers for now, with the canonical Moves primarily exercised by parity and test harnesses.

### 7.3 WebSocket and AI integration

- Primary move selection (placement, movement, capture, chain capture) is Move‑driven for both humans and AI:
  - The backend exposes valid Moves to clients and AI, and all accepted decisions are normalised into Moves before being applied.
- Decision‑phase UX (line reward, region order, ring elimination, capture direction) still uses typed choice payloads as the transport surface:
  - Each option is enriched with a corresponding canonical move identifier.
  - Ongoing refactors aim to ensure those identifiers are always resolved back into concrete Moves and reapplied through the same Move validation and application code paths, so that the public decision surfaces are uniformly expressed in terms of Moves even when the transport shape still mentions PlayerChoice.
