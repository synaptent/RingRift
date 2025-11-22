# RingRift Test Suite Parity & Classification Plan

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
   - `types.ts`, `validators/*`, `mutators/*`, `GameEngine.ts`, `lineDetection.ts`, `territoryDetection.ts`, `initialState.ts`, `core.ts`, `notation.ts`.
2. **Written rules** – `ringrift_complete_rules.md` (plus `RULES_SCENARIO_MATRIX.md`).

Legacy TS engines (`src/server/game/*`, `src/client/sandbox/*`) and the Python rules engine (`ai-service/app/rules/*`, `ai-service/app/game_engine.py`) are **adapters/hosts** that must conform to this shared semantics.

Recorded traces (seeded games, AI-vs-AI runs, historic move logs) are **derived artifacts** and must yield when they conflict with the shared engine + rules docs. See `tests/README.md` for the **seed‑14 precedent** and the detailed rules‑over‑traces policy.

---

## 1. Shared‑engine–anchored rules-level suites (TS)

These suites are (or should be) treated as **authoritative** for rules semantics.

| File                                                        | Level                         | Domain                               | Notes / Alignment with `src/shared/engine`                                                                                                                                          | Gaps / TODO                                                                                                                         |
| ----------------------------------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/RefactoredEngine.test.ts`                       | Rules-level                   | TS shared engine                     | Directly exercises `src/shared/engine/GameEngine` and related types.                                                                                                                | Keep adding minimal, focused rules cases here when the change is **purely engine-level**.                                           |
| `tests/unit/RefactoredEngineParity.test.ts`                 | Rules-level + adapter parity  | TS shared engine ↔ backend/sandbox  | Verifies that backend/sandbox adapters behave like the refactored shared engine. Shared engine is canonical.                                                                        | For any failure: fix adapters or add a more targeted rules-level test; do **not** diverge shared engine to preserve adapter quirks. |
| `tests/unit/LineDetectionParity.rules.test.ts`              | Rules-level                   | Shared engine, BoardManager, sandbox | Authoritative for line detection semantics (markers only, blocking stacks/collapsed spaces, square vs hex geometry).                                                                | Use as primary reference when adjusting line logic anywhere (TS or Python).                                                         |
| `tests/unit/sandboxTerritory.rules.test.ts`                 | Rules-level                   | Shared engine ↔ sandbox             | Territory rules expressed via sandbox engine but logically anchored to shared semantics.                                                                                            | Ensure key cases are also covered in pure shared-engine tests over time.                                                            |
| `tests/unit/territoryProcessing.rules.test.ts`              | Rules-level                   | Shared engine ↔ backend             | Territory processing semantics aligned with rules doc.                                                                                                                              | Same as above: prefer pure shared-engine coverage where possible.                                                                   |
| `tests/unit/sandboxTerritoryEngine.rules.test.ts`           | Rules-level                   | Shared engine ↔ sandbox             | Engine-level territory behaviour, close to shared engine.                                                                                                                           | Expand if new territory gaps are found; mirror important cases in Python fixtures.                                                  |
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

| File                                                        | Level                                          | Domain                                      | Current semantic anchors                                                                                                                                     | Policy / TODO                                                                                                                                                                                                                                     |
| ----------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`         | Trace-level parity                             | Backend GameEngine ↔ ClientSandboxEngine   | `RefactoredEngine.test.ts`, `RefactoredEngineParity.test.ts`, `LineDetectionParity.rules.test.ts`, `Seed14Move35LineParity.test.ts`, territory rules suites. | Seeds list is curated (currently seed 5 only). If a seed fails but all rules-level tests pass, treat the trace as **stale**. Either regenerate trace under current rules or retire the seed and add a focused rules-level test (seed‑14 pattern). |
| `tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`         | Trace-level parity                             | Backend AIEngine ↔ Sandbox AI (shared RNG) | RNG wiring in `localAIMoveSelection.ts` and RNG-focused tests noted in `tests/README.md`.                                                                    | Use to verify RNG injection & parity. If semantics shift, update rules-level tests first, then adjust expectations here.                                                                                                                          |
| `tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`     | Trace-level parity (diagnostic, often skipped) | Same as above, but with deeper sequences    | Same as above.                                                                                                                                               | Keep as **diagnostic only** (e.g. `describe.skip` by default). Use when debugging deep AI parity issues.                                                                                                                                          |
| `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` | Trace-level coverage                           | Backend heuristic AI ↔ sandbox AI          | Shared engine semantics and AI heuristics tests.                                                                                                             | When this fails yet rules-level tests are green, assume either AI-specific mismatch or stale traces; debug using logs/diagnostics before altering rules.                                                                                          |
| `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`    | Trace-level, seed-specific debug               | Backend ↔ sandbox                          | Same anchors as main traceParity suite.                                                                                                                      | Treat as a debug harness. Once issues are resolved and covered by rules-level tests, either trim or move to a `debug/` folder.                                                                                                                    |
| `tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts`   | Trace-level, seed-specific debug               | Backend ↔ sandbox                          | Seed‑17-specific.                                                                                                                                            | Same as above; pair with rules-level or scenario tests that capture important behaviour.                                                                                                                                                          |
| `tests/unit/ParityDebug.seed5.trace.test.ts`                | Trace-level debug                              | Backend ↔ sandbox                          | Uses GameTrace helpers.                                                                                                                                      | Consider consolidating debug patterns; ensure any semantics uncovered here are reflected in rules-level tests.                                                                                                                                    |
| `tests/unit/ParityDebug.seed14.trace.test.ts`               | Trace-level debug                              | Backend ↔ sandbox                          | Superseded in semantics by `Seed14Move35LineParity.test.ts`.                                                                                                 | Keep for debugging, but treat `Seed14Move35LineParity` as the canonical semantic authority.                                                                                                                                                       |
| `tests/unit/TraceParity.seed5.firstDivergence.test.ts`      | Trace-level diagnostic                         | Backend ↔ sandbox                          | Same as seed 5 anchors.                                                                                                                                      | Use to locate first divergence. If divergence is due to rules semantics, capture that case in a rules-level test.                                                                                                                                 |
| `tests/unit/TraceParity.seed14.firstDivergence.test.ts`     | Trace-level diagnostic                         | Backend ↔ sandbox                          | Same as seed 14 anchors.                                                                                                                                     | Historical; semantics now captured by `Seed14Move35LineParity`.                                                                                                                                                                                   |
| `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`     | Trace-level + performance/parallel             | Backend ↔ sandbox                          | GameTrace helpers and AI RNG wiring.                                                                                                                         | Use only as a debugging/profiling tool. Not a source of rules truth.                                                                                                                                                                              |

### 2.2 Movement / capture / territory / victory parity suites

These are parity-focused but often test specific scenarios. They should be **backed by rules-level counterparts** where semantics are subtle.

| File                                                             | Level                                 | Domain                          | Current semantic anchors                                                                                            | Rules-level counterpart (existing or TODO)                                                                               |
| ---------------------------------------------------------------- | ------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `tests/unit/PlacementParity.RuleEngine_vs_Sandbox.test.ts`       | Trace-level parity                    | RuleEngine (backend) ↔ sandbox | Placement rules in shared engine and scenario suites.                                                               | Ensure placement scenarios are covered in shared-engine tests (e.g. dedicated `Placement.rules.test.ts` if gaps emerge). |
| `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts` | Trace-level parity                    | Movement/capture parity         | Movement/capture sections of `RULES_SCENARIO_MATRIX.md`, `RuleEngine.movementCapture.test.ts`, chain-capture tests. | Where behaviour is subtle (e.g. marker landings, blocked paths), mirror with shared-engine rules tests.                  |
| `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`       | Trace-level parity                    | Territory parity                | `GameEngine.territory.scenarios.test.ts`, `BoardManager.territoryDisconnection.*`, territory rules tests.           | Ensure a pure shared-engine territory rules suite exists and is referenced from here.                                    |
| `tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts`         | Trace-level parity                    | Victory conditions parity       | `ClientSandboxEngine.victory.test.ts`, `GameEngine.victory.scenarios.test.ts`.                                      | Add shared-engine victory rules tests if behaviour changes.                                                              |
| `tests/unit/Seed17GeometryParity.GameEngine_vs_Sandbox.test.ts`  | Trace-level (geometry-focused parity) | Backend ↔ sandbox              | Seed‑17 geometry behaviour.                                                                                         | For any discovered rules nuance, add a rules-level scenario capturing the final state/invariants.                        |
| `tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`    | Trace-level, seed-specific            | Backend ↔ sandbox              | Seed‑17 up to move 52.                                                                                              | Same pattern as seed‑14: if semantics change, retire or regenerate traces and add a focused rules-level test.            |

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

Representative files:

- `tests/unit/ClientSandboxEngine.initialPlacement.test.ts`
- `tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts`
- `tests/unit/ClientSandboxEngine.moveParity.test.ts`
- `tests/unit/ClientSandboxEngine.chainCapture.test.ts`
- `tests/unit/ClientSandboxEngine.chainCapture.scenarios.test.ts`
- `tests/unit/ClientSandboxEngine.lines.test.ts`
- `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`
- `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`
- `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`
- `tests/unit/ClientSandboxEngine.victory.test.ts`
- `tests/unit/ClientSandboxEngine.invariants.test.ts`
- `tests/unit/ClientSandboxEngine.traceStructure.test.ts`
- `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`
- `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`
- `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts`
- `tests/unit/ClientSandboxEngine.aiStallDiagnostics.test.ts`
- `tests/unit/ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts`
- `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`

**Classification:**

- Mix of **adapter-level rules checks** and **behaviour/integration**.
- They should:
  - Rely on shared-engine rules tests for semantics.
  - Explicitly reference corresponding rules-level suites in comments (e.g. for lines, territory, chain-captures).

### 3.3 WebSocket, AI service, and UI integration

Representative files:

- `tests/unit/AIEngine.serviceClient.test.ts`
- `tests/unit/AIEngine.placementMetadata.test.ts`
- `tests/unit/AIInteractionHandler.test.ts`
- `tests/unit/WebSocketServer.humanDecisionById.integration.test.ts`
- `tests/unit/WebSocketServer.aiTurn.integration.test.ts`
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
- `ai-service/tests/test_determinism.py`
- `ai-service/tests/test_model_architecture.py`
- `ai-service/tests/test_heuristic_ai.py`
- `ai-service/tests/test_minimax_ai.py`
- `ai-service/tests/test_mcts_ai.py`

**Classification:**

- Python‑only **rules/behaviour and AI tests**.
- Should be brought into line with shared-engine semantics by consuming **TS-generated fixtures** where feasible (see below).

### 4.2 Python rules parity suites

Representative files:

- `ai-service/tests/parity/test_rules_parity.py`
- `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`
- `ai-service/tests/parity/generate_test_vectors.py`
- TS counterpart: `tests/unit/Python_vs_TS.traceParity.test.ts`

**Current state:**

- Parity tests compare Python rules/engine against TS behaviour, partly via ad-hoc fixtures.

**Target state:**

- **Fixture-driven rules parity**:
  - TS shared engine generates canonical fixtures (see §5 below).
  - Python tests load and validate against these fixtures.
  - `Python_vs_TS.traceParity.test.ts` is treated as a **trace-level smoke test**, not a semantic authority.

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
