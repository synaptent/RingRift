# RingRift TODO / Task Tracker

> **Doc Status (2025-11-29): Active (execution/backlog tracker)**
>
> - Canonical high-level task/backlog tracker for near- and mid-term work.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

**Last Updated:** November 30, 2025

This file is the canonical high-level task tracker for the project.
When it disagrees with older planning docs (for example files under
`deprecated/`), this file and the status docs
[`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md)
win.

For **rules semantics themselves** (what the correct behaviour _should_ be),
the ultimate source of canonical truth is the rules documentation:
[`ringrift_complete_rules.md`](ringrift_complete_rules.md) (and, where
applicable, [`ringrift_compact_rules.md`](ringrift_compact_rules.md)).
When there is any ambiguity, parity mismatch, or question about an engine or
sandbox implementation, tests and code should be treated as converging toward
those documents, not the other way around.

Priorities:

- **P0** – Critical for rules correctness / engine parity.
- **P1** – High-value for playable, stable online games.
- **P2** – Important but can follow P0/P1.

## Phase 1.5 – Architecture Remediation (COMPLETED) ✅

**Completed:** November 26, 2025

This phase consolidated the rules engine architecture across 4 sub-phases:

### Phase 1: Architecture & Design ✅

- [x] Created canonical turn orchestrator in `src/shared/engine/orchestration/`
  - [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts) – main entry point
  - [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts) – phase transitions
  - [`types.ts`](src/shared/engine/orchestration/types.ts) – orchestration types
- [x] Added contract schemas in `src/shared/engine/contracts/`
  - [`schemas.ts`](src/shared/engine/contracts/schemas.ts), [`serialization.ts`](src/shared/engine/contracts/serialization.ts)
- [x] Created initial test vectors in `tests/fixtures/contract-vectors/v2/`

### Phase 2: Rules Engine Consolidation ✅

- [x] Wired orchestrator to all 6 domain aggregates (Placement, Movement, Capture, Line, Territory, Victory)
- [x] Added line detection and territory test vectors
- [x] 14 contract tests passing

### Phase 3: Backend/Sandbox Adapter Migration ✅

- [x] Created [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts) for backend (326 lines)
- [x] Created [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts) for client (476 lines)
- [x] 46 adapter/contract tests passing
- [x] Feature flags (`useOrchestratorAdapter`) for gradual rollout

### Phase 4: Python Contract Test Runner ✅

- [x] Created Python serialization matching TS format ([`serialization.py`](ai-service/app/rules/serialization.py))
- [x] Created contract test runner ([`test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py))
- [x] 100% cross-language parity on 12 test vectors
- [x] Python: 245 tests passing, 15 contract tests

**Documentation produced:**

- [`docs/drafts/PHASE1_REMEDIATION_PLAN.md`](docs/drafts/PHASE1_REMEDIATION_PLAN.md)
- [`docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md`](docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md)
- [`docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md`](docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md)
- [`src/shared/engine/orchestration/README.md`](src/shared/engine/orchestration/README.md)

---

## Phase 2 – Robustness & Testing (IN PROGRESS, P0)

### P0.1 – Rules/FAQ Scenario Matrix

- [x] Build and maintain a scenario matrix in
      [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) that maps
      examples from [`ringrift_complete_rules.md`](ringrift_complete_rules.md)
      and the FAQ to concrete Jest suites.
- [x] For each major rule cluster (movement, chain captures, lines,
      territory, victory), ensure there is at least one emblematic scenario
      that is tested in both:
  - Backend engine (`GameEngine` / `RuleEngine` in
    [`src/server/game`](src/server/game/))
  - Client sandbox engine (`ClientSandboxEngine` in
    [`src/client/sandbox/ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts))
- [x] Keep scenario IDs consistent across:
  - [`tests/scenarios/rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts)
  - Scenario suites under `tests/scenarios/`
  - Any parity tests that reference specific rules/FAQ cases.

### P0.2 – Backend ↔ Sandbox Parity

- [ ] Regularly run and keep green the trace-parity and heuristic coverage
      suites:
  - [`Backend_vs_Sandbox.traceParity.test.ts`](tests/unit/Backend_vs_Sandbox.traceParity.test.ts)
  - [`Sandbox_vs_Backend.seed5.traceDebug.test.ts`](tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts)
  - [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [`RefactoredEngineParity.test.ts`](tests/unit/RefactoredEngineParity.test.ts) (minimal legacy-backend vs shared-engine parity harness; currently covers a basic placement + movement sequence and should be extended over time)
- [ ] When a parity failure appears:
  - [ ] Extract the first divergence index using the
        `TraceParity.*.firstDivergence` helpers.
  - [ ] Turn the failing position into a small, focused unit test under
        `tests/unit/` (e.g., movement, capture, or placement parity).
  - [ ] Fix the underlying engine/sandbox discrepancy.
  - [ ] Mark the corresponding row in
        [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) as covered.
- [ ] Treat any genuine rules mismatch between backend
      [`GameEngine`](src/server/game/GameEngine.ts) /
      [`RuleEngine`](src/server/game/RuleEngine.ts) and
      [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts)
      as a **P0 bug**.

### P0.3 – S-Invariant & Termination

- [ ] Keep the S-invariant tests passing and expand them as new rules
      interactions are implemented:
  - [`ProgressSnapshot.core.test.ts`](tests/unit/ProgressSnapshot.core.test.ts)
  - Sandbox AI simulation diagnostics in
    [`ClientSandboxEngine.aiSimulation.test.ts`](tests/unit/ClientSandboxEngine.aiSimulation.test.ts)
  - Backend AI-style simulations in
    [`GameEngine.aiSimulation.test.ts`](tests/unit/GameEngine.aiSimulation.test.ts)
- [ ] For any new rule that can change markers, collapsed spaces, or
      eliminated rings, ensure it is reflected in
      [`computeProgressSnapshot`](src/shared/engine/core.ts) and covered by
      tests.
- [ ] Add explicit S-invariant coverage for orchestrator multi-phase flows:
  - [ ] `chain_capture` + `continue_capture_segment` (backend + sandbox)
  - [ ] `territory_processing` with mandatory self-elimination (backend + sandbox)
- [ ] Add an AI-service S-invariant parity check:
  - [ ] Mirror one or two core S-invariant scenarios in Python
        (e.g. `ai-service/tests/test_territory_forced_elimination_divergence.py`
        and line/territory parity tests) and assert the same S deltas as the
        TS engine.

### P0.4 – Unified Move model for all player-chosen decisions (backend + sandbox)

Goal: every player-chosen decision permitted by the rules – including chain
capture direction, line order & rewards, territory region order, and ring
elimination targets – is represented as a `Move` and enumerated via
`RuleEngine.getValidMoves` / `GameEngine.getValidMoves`, not only via
`PlayerChoice` flows. This enables:

- A single canonical move space for backend AI, sandbox AI, and clients.
- Stronger parity guarantees between backend and sandbox behaviour.
- Easier replay, trace analysis, and tooling around complete turn histories.

Planned work:

- [x] Extend the shared `Move` model in
      [`src/shared/types/game.ts`](src/shared/types/game.ts) with additional
      decision types, for example:
  - [x] `continue_capture_segment` for capture chain continuation segments
        (`from`, `captureTarget`, `to`) in backend `GameEngine` / `RuleEngine`.
  - [x] `process_line` / `choose_line_reward` for line order & reward
        decisions (line id, marker positions, reward option).
  - [x] `process_territory_region` for disconnected region order
        (region id, representative position, size).
  - [x] `eliminate_rings_from_stack` for explicit elimination targets
        (stack position, elimination count / cap height).
- [x] Introduce or clarify interactive phases in `GameState.currentPhase`
      where these decisions occur, e.g.:
  - [x] `chain_capture` phase for mandatory capture continuations (backend `GameEngine` / `TurnEngine` wired).
  - [x] Explicit `line_processing` interactive steps where `getValidMoves`
        returns `process_line` / `choose_line_reward` moves (backend complete; sandbox parity pending).
  - [x] Explicit `territory_processing` interactive steps where
        `getValidMoves` returns `process_territory_region` moves (backend region enumeration complete; sandbox Move/phase parity and adoption of explicit elimination Moves in WebSocket/AI and sandbox flows remain future work).
- [x] Refactor `RuleEngine.getValidMoves` to:
  - [x] Preserve existing behaviour for `ring_placement`, `movement`, and
        `capture` (`place_ring`, `skip_placement`, `move_stack`,
        `overtaking_capture`).
  - Note: In `chain_capture` phase, enumeration of follow-up capture
    segments is handled by `GameEngine.getValidMoves`, which has access to
    internal `chainCaptureState`; `RuleEngine` remains responsible for
    segment-level validation of `overtaking_capture` /
    `continue_capture_segment`.
  - [x] In `line_processing` phase, enumerate all eligible lines and
        available reward options as `process_line` / `choose_line_reward`
        moves instead of (or in addition to) `LineOrderChoice` and
        `LineRewardChoice`.
  - [x] In `territory_processing` phase, enumerate eligible disconnected
        regions as `process_territory_region` moves instead of (or in addition
        to) `RegionOrderChoice` for ordering (explicit elimination moves remain
        future work).
- [ ] Update `GameEngine` to drive these phases via `makeMove` rather than
      direct `PlayerInteractionManager` calls, so that:
  - [ ] Human clients and AI both select from the same `getValidMoves`
        result set for all interactive decisions.
  - [ ] Internal post-move processors (`lineProcessing`,
        `territoryProcessing`, `captureChainEngine`) are refactored to apply
        selected decision-moves instead of bespoke `PlayerChoice` branches.
- [ ] Adjust `PlayerInteractionManager` and WebSocket handlers so that
      frontends treat these new decision-move types as first-class actions
      (selecting among moves) rather than distinct `PlayerChoice` payloads.
- [ ] Keep `ClientSandboxEngine` in sync by:
  - [ ] Importing the same extended `Move` types.
  - [ ] Mirroring the new interactive phases and decision-move handling in
        the sandbox (e.g., chain capture, line processing, territory
        processing, elimination) using the existing sandbox helpers
        (`sandboxMovement.ts`, `sandboxElimination`).
  - [ ] Updating `sandboxAI` to select among these richer `Move` sets,
        staying in lockstep with backend `getValidMoves`.
- [ ] Align sandbox Move/phase handling with backend:
  - [ ] Route all sandbox actions (human and AI) through the canonical
        `applyCanonicalMoveInternal` path using the same `Move` types and
        `GamePhase` transitions as the backend, including `chain_capture`,
        `line_processing`, and `territory_processing`.
  - [ ] Ensure `ClientSandboxEngine` respects the same decision semantics
        for `continue_capture_segment`, `process_line`, and
        `process_territory_region`.
  - [ ] Update sandbox parity and RulesMatrix-backed tests to assert both
        the legal-move sets and resulting phases for these advanced phases.
- [ ] Extend and/or add parity tests to cover the new decision-move
      surface:
  - [ ] Backend vs sandbox parity for chain capture continuation decisions.
  - [ ] Backend vs sandbox parity for line order/reward and region
        order/elimination decisions.
  - [ ] Trace parity tests that confirm all such decisions are now
        represented as `Move`s and faithfully replayable.

**Current P0.4 status (as of 2025‑11‑19):**

- [x] Backend `GameEngine` / `RuleEngine` now model capture-chain continuation via a distinct `chain_capture` phase and `continue_capture_segment` moves.
- [x] Backend capture-sequence enumeration now uses `captureChainEngine.getCaptureOptionsFromPosition` plus shared `validateCaptureSegmentOnBoard`, keeping [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts) green across square and hex boards.
- [x] Backend territory-processing now enumerates explicit `eliminate_rings_from_stack` Moves via `RuleEngine.getValidEliminationDecisionMoves`, and `RingEliminationChoice.options[].moveId` / `RegionOrderChoice.options[].moveId` are wired to canonical `Move.id` values for elimination and disconnected-region decisions (sandbox Move/phase parity and WebSocket/AI adoption of these Move ids remain future work).
- [ ] Several scenario/parity suites are temporarily red while the new chain-capture model is wired through all surfaces:
  - [ ] [`ComplexChainCaptures.test.ts`](tests/scenarios/ComplexChainCaptures.test.ts)
  - [ ] [`RulesMatrix.ChainCapture.GameEngine.test.ts`](tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts)
  - [ ] [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts)
  - [ ] [`Backend_vs_Sandbox.aiParallelDebug.test.ts`](tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts)
  - [ ] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [ ] [`TraceParity.seed*.firstDivergence.test.ts`](tests/unit/TraceParity.seed5.firstDivergence.test.ts)
- [x] Sandbox engine and `sandboxAI` now participate in the new `chain_capture` / `continue_capture_segment` Move model for AI turns and canonical traces; remaining divergences (e.g. seed 14 trace parity) are localized and tracked via `TraceParity.seed14.*` / `ParityDebug.seed14.*` helpers.

**Near-term P0.4 tasks inferred from current test failures:**

- [ ] Finalize backend chain-capture semantics for cyclic/triangle patterns so that FAQ scenarios in [`rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts) and [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts) pass under the new model.
- [ ] Migrate `ComplexChainCaptures` and `RulesMatrix.ChainCapture` suites to drive chains via `chain_capture` + `continue_capture_segment` rather than internal while-loops.
- [ ] Mirror the chain-capture phase and `continue_capture_segment` moves into [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts) and [`sandboxAI`](src/client/sandbox/sandboxAI.ts), then restore:
  - [ ] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [ ] [`Backend_vs_Sandbox.*trace*.test.ts`](tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts)
- [ ] Once chain capture is stable across backend and sandbox, extend the same Move/phase unification to line-processing and territory-processing (as described above).
- [ ] Keep `tests/integration/FullGameFlow.test.ts` green by ensuring AI-vs-AI backend games using local AI fallback always reach a terminal `gameStatus` (e.g. `completed`/`finished`) within the configured move budget. Treat regressions here as part of P0.4 since they exercise the unified Move model end-to-end.

### P0.5 – Python AI-service rules engine parity (P0) ✅

**Note:** Phase 4 of the Architecture Remediation added contract tests that validate cross-language parity. The Python engine now passes 100% of contract test vectors with TS engine output.

- [x] Audit mismatches between Python [`GameEngine`](ai-service/app/game_engine.py) /
      [`BoardManager`](ai-service/app/board_manager.py) /
      [`models`](ai-service/app/models.py) and the canonical TS shared rules
      engine under `src/shared/engine/` (types, validators, mutators,
      `GameEngine`), plus backend [`GameEngine`](src/server/game/GameEngine.ts) /
      [`RuleEngine`](src/server/game/RuleEngine.ts) adapters, using the rules
      docs as the primary spec.
- [x] Align Python movement and overtaking captures with TS:
  - [x] Ray-based movement using board-type directions (Moore for square,
        6-dir for hex), enforcing minimum distance ≥ stack height and blocking on
        stacks/collapsed spaces while processing markers on the path.
  - [x] Cap-height-based overtaking captures that allow both own-stack and
        opponent-stack targets, with from→target→landing geometry validated
        analogously to
        [`validateCaptureSegmentOnBoard`](src/shared/engine/core.ts).
  - [x] Chain-capture application that leaves markers on departure, processes
        markers along both legs, transfers exactly one captured ring per segment
        to the bottom of the attacker, supports merging at the landing stack, and
        updates a Python [`ChainCaptureState`](ai-service/app/models.py)
        mirroring TS [`ChainCaptureState`](src/server/game/rules/captureChainEngine.ts).
- [x] Placement parity for AI-service engine:
  - [x] Support multi-ring placement on empty spaces (1–3 rings) and exactly
        1 ring when placing on existing stacks, mirroring
        [`RuleEngine.validateRingPlacement`](src/server/game/RuleEngine.ts)
        semantics.
  - [x] Enforce per-player ring caps derived from total rings in play /
        max players (Python analogue of `BOARD_CONFIGS[boardType].ringsPerPlayer`).
  - [x] Implement TS-style no-dead-placement via a hypothetical board helper
        (Python analogue of
        [`createHypotheticalBoardWithPlacement`](src/server/game/rules/placementHelpers.ts))
        plus a reachability helper equivalent to
        [`hasAnyLegalMoveOrCaptureFromOnBoard`](src/shared/engine/core.ts).
- [x] Lines parity for AI-service engine:
  - [x] Ensure Python line detection in
        [`BoardManager.find_all_lines`](ai-service/app/board_manager.py)
        matches TS [`BoardManager.findAllLines`](src/server/game/BoardManager.ts)
        in geometry and minimum length.
  - [x] Refactor Python line-processing move generation and
        [`_apply_line_formation`](ai-service/app/game_engine.py) to mirror
        [`lineProcessing`](src/server/game/rules/lineProcessing.ts) semantics
        (exact-length vs overlength options, collapsed markers, reward
        eliminations) while preserving the unified `Move` model.
- [x] Territory disconnection parity for AI-service engine:
  - [x] Extend Python
        [`BoardManager.find_disconnected_regions`](ai-service/app/board_manager.py)
        and territory-processing moves to respect the single-border-color marker
        logic, region representation rules, and cascades implemented in
        [`territoryProcessing`](src/server/game/rules/territoryProcessing.ts).
  - [x] Ensure mandatory self-elimination prerequisites and elimination
        accounting match TS behaviour.
- [x] Turn engine / forced elimination / victory semantics (AI-service):
  - [x] Align Python phase transitions and forced-elimination moves in
        [`GameEngine._update_phase`](ai-service/app/game_engine.py) and
        [`_end_turn`](ai-service/app/game_engine.py) with the TS
        [`TurnEngine`](src/server/game/turn/TurnEngine.ts) semantics
        (`skip_placement`, forced elimination only when no other actions are
        available, re-checking actions after forced elimination in the same turn).
  - [x] Bring [`GameEngine._check_victory`](ai-service/app/game_engine.py)
        into parity with
        [`RuleEngine.checkGameEnd`](src/server/game/RuleEngine.ts),
        including last-player-standing and stalemate (rings-in-hand conversion)
        tie-break ladders.
- [ ] Lint/typing hygiene for `ai-service` rules engine:
  - [ ] Gradually reduce flake8 long-line violations in
        [`game_engine.py`](ai-service/app/game_engine.py) and related files
        without sacrificing the diagnostic value of comments.
  - [ ] Optionally add targeted `# type: ignore[...]` annotations for known
        Pydantic alias patterns (e.g. `capture_target`, `collapsed_markers`) where
        Pylance cannot infer the dynamic `__init__` parameters, to keep the
        signal-to-noise ratio of editor diagnostics high.

### P0.6 – Canonical Orchestrator Production Hardening

Post-remediation work to fully enable the orchestrator in production. Code‑level wiring,
metrics, and CI gates are now in place; remaining work is primarily operational and
cleanup of legacy paths.

- [x] Introduce centralized `OrchestratorRolloutService` and env‑driven flags
      (`ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`,
      `ORCHESTRATOR_SHADOW_MODE_ENABLED`) with loud warnings when the adapter
      is disabled in `NODE_ENV=production` (see
      [`src/server/index.ts`](src/server/index.ts) and
      [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md)).
- [x] Add a dedicated TS orchestrator‑parity Jest profile
      (`npm run test:orchestrator-parity`) and CI job
      **TS Orchestrator Parity (adapter‑ON)** that runs with the adapter forced
      on and is wired as a required check for `main`.
- [x] Add SSoT guardrails and fences to keep orchestrator semantics canonical:
  - [x] `rules-ssot-check.ts` restricts direct `RuleEngine` imports to
        whitelisted diagnostics/legacy sites.
  - [x] `rules-ssot-check.ts` also enforces that diagnostics‑only sandbox helpers
        (`sandboxCaptureSearch`, etc.) are not imported from production hosts.
  - [x] `docs-link-ssot`, `ci-config-ssot`, `env-doc-ssot`, and related checks
        are green.
- [x] Add orchestrator‑specific metrics and observability:
  - [x] Session‑level metrics via `OrchestratorRolloutService` and
        `MetricsService` (e.g. `recordOrchestratorSession`, legacy vs
        orchestrator move counts).
  - [x] Client and server logging when legacy or shadow paths are exercised.
- [x] Operate the orchestrator adapter at 100% in staging/production (env/config
      change and monitoring task; code support is in place).
      _All environments (dev, staging, CI) now configured with `ORCHESTRATOR_ADAPTER_ENABLED=true`
      and `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`. Soak tests show zero invariant violations
      across all board types. Production deployment is a deployment task, not a code change._
- [x] Monitor for divergences in game outcomes and AI quality using the new
      metrics (including legacy vs orchestrator move counters and parity suites).
      _Orchestrator parity tests (13 suites) and soak tests (15 games, 0 violations) are green._
- [ ] Remove legacy turn‑processing code paths once orchestrator behaviour is
      stable and all high‑signal parity suites remain green; keep any needed
      harnesses under `archive/` and ensure SSOT checks prevent regressions.
      _Concrete deprecation/cleanup steps are documented under **Wave 5.4 – Legacy path deprecation and cleanup**._

## Wave 5 – Orchestrator Production Rollout (P0/P1)

> **Goal:** Make the canonical orchestrator + adapters the *only* production
> turn path, with safe rollout across environments and removal of legacy
> turn‑processing code once stable.

### Wave 5.1 – Staging orchestrator enablement

- [x] Enable `useOrchestratorAdapter` (and any equivalent flags) by default in
      **staging** for both:
  - [x] Backend `GameEngine` via `TurnEngineAdapter`.
  - [x] Client sandbox via `SandboxOrchestratorAdapter`.
- [x] Configure staging `NODE_ENV`, `RINGRIFT_RULES_MODE`, and orchestrator
      flags to match the Phase 1/2 presets in
      [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §8.1.1 / §8.7](docs/ORCHESTRATOR_ROLLOUT_PLAN.md).
      _See `.env.staging` for the complete configuration._
- [x] Document the staging orchestrator posture in `CURRENT_STATE_ASSESSMENT.md`
      and/or environment runbooks so operators know which adapter is active.

### Wave 5.2 – Orchestrator‑ON validation suites

Run these suites with orchestrator adapters forced ON (backend + sandbox) and
keep them green; treat regressions as P0 until resolved.

- [ ] WebSocket + session flows (backend host):
  - [ ] `tests/integration/GameReconnection.test.ts`
  - [ ] `tests/integration/GameSession.aiDeterminism.test.ts`
  - [ ] `tests/integration/AIResilience.test.ts`
  - [ ] `tests/integration/LobbyRealtime.test.ts`
- [x] Orchestrator multi‑phase scenarios (backend + sandbox):
  - [x] `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`
  - [x] `tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts`
- [ ] Decision‑heavy rules flows (line / territory / chain‑capture):
  - [ ] Line reward / line order suites (GameEngine + ClientSandboxEngine).
  - [ ] Territory processing + self‑elimination suites.
  - [ ] Complex chain‑capture scenarios.
- [x] Invariant and contract guards (already wired in CI, but re‑run locally
      when changing flags):
  - [x] Orchestrator invariant soaks via `scripts/run-orchestrator-soak.ts`.
  - [x] Contract vectors in `tests/contracts/contractVectorRunner.test.ts` and
        `ai-service/tests/contracts/test_contract_vectors.py`.
        _All contract vectors pass with orchestrator-ON as of the `test:orchestrator-parity` suite._

### Wave 5.3 – Production rollout with circuit‑breakers

- [x] Define and document production flag presets for Phases 2–4 in
      `src/server/config/env.ts` / `src/server/config/unified.ts` so they match
      the matrices in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (§8.1.1, §8.7):
  - [x] Phase 2 – legacy authoritative + orchestrator in shadow mode.
  - [x] Phase 3 – percentage‑based orchestrator rollout.
  - [x] Phase 4 – orchestrator authoritative.
      _All presets documented in `.env`, `.env.staging`, and `docs/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`.
      Production defaults are Phase 4 (orchestrator authoritative at 100%)._
- [x] Ensure `OrchestratorRolloutService` circuit‑breaker thresholds
      (error rate, window) are aligned with the rollout plan and surfaced via
      `MetricsService` (legacy vs orchestrator move counts, kill‑switch events).
      _Circuit breaker configured with 5% error threshold, 300s window. Metrics exposed._
- [x] Wire SLOs and CI gates as hard release checks:
  - [x] Keep `TS Orchestrator Parity (adapter‑ON)` CI job required for `main`.
  - [x] Ensure orchestrator invariant soaks and short parity suites are
        documented as pre‑prod gates in `docs/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`.

### Wave 5.4 – Legacy path deprecation and cleanup

- [ ] Mark remaining legacy turn‑processing paths as **deprecated** in:
  - [ ] `src/server/game/RuleEngine.ts` / `GameEngine.ts`.
  - [ ] `src/client/sandbox/ClientSandboxEngine.ts`.
  - [ ] Any sandbox‑only helpers that still implement bespoke turn/phase loops.
- [ ] Migrate tests that still exercise legacy‑only flows to:
  - [ ] Go through `TurnEngineAdapter` / `SandboxOrchestratorAdapter`, **or**
  - [ ] Target the shared engine directly (aggregates/helpers) where host
        concerns are not required.
- [ ] Once orchestrator behaviour is stable in production for a full release
      window and all high‑signal suites remain green:
  - [ ] Remove legacy turn‑processing implementations from backend and
        sandbox hosts.
  - [ ] Preserve any historically valuable harnesses under `archive/` with
        clear “legacy” annotations.
  - [ ] Update `ARCHITECTURE_ASSESSMENT.md`, `RULES_ENGINE_ARCHITECTURE.md`,
        and `STATE_MACHINES.md` so the orchestrator + adapters are the only
        described production paths.

## Phase 3 – Multiplayer Polish (P1)

### P1.1 – WebSocket Lifecycle & Reconnection

- [ ] Tighten and test WebSocket lifecycle around:
  - Reconnects and late joins in
    [`WebSocketServer`](src/server/websocket/server.ts) and
    [`WebSocketInteractionHandler`](src/server/game/WebSocketInteractionHandler.ts).
  - Consistent `game_over` handling and clearing of any pending choices.
  - Spectator join/leave flows, ensuring spectators are always read-only.
- [ ] Add focused Jest integration tests for lifecycle aspects under
      `tests/unit/`, and cross-link them from
      [`tests/README.md`](tests/README.md).

### P1.2 – Game HUD & Game Host UX

- [x] Enhance [`GameHUD`](src/client/components/GameHUD.tsx) to show:
  - Current player and phase.
  - Per-player ring counts (in hand / on board / eliminated).
  - Territory spaces per player.
  - Basic timer readout derived from `timeControl` and `timeRemaining`.
- [ ] Ensure the same HUD behaviour is shared between backend games
      (`/game/:gameId` and `/spectate/:gameId` via
      [`BackendGameHost`](src/client/pages/BackendGameHost.tsx)) and
      sandbox games (`/sandbox` via
      [`SandboxGameHost`](src/client/pages/SandboxGameHost.tsx)).
- [x] Add a minimal per-game event log in the game host shell
      (`BackendGameHost` / `SandboxGameHost`) for moves, PlayerChoices,
      and `game_over` events (using `GameEventLog` + system events).

### P1.3 – Lobby, Spectators, and Chat

- [ ] Improve lobby UX in
      [`LobbyPage`](src/client/pages/LobbyPage.tsx) with clearer status,
      filters, and navigation.
- [x] Implement a basic spectator UI (read-only board + HUD) that uses
      the same `GameContext` as players but disables input.
- [x] Wire a simple in-game chat panel to the existing `chat_message`
      events in the WebSocket server (see `GamePage` + `GameContext` chat tests).

### P1.4 – Client Component & View-Model Unit Tests

> **Context:** Pass 15 identified client component unit testing as the primary remaining weakness (~2.5/5), despite strong E2E coverage. The goal here is to raise confidence for key UI surfaces without duplicating full end-to-end flows.

- [x] Add comprehensive unit tests for the pure view-model adapters in
      [`gameViewModels.ts`](src/client/adapters/gameViewModels.ts)  
       (see `tests/unit/adapters/gameViewModels.test.ts`).
- [ ] Add focused React tests for complex HUD/log components:
  - [ ] [`GameHUD.tsx`](src/client/components/GameHUD.tsx) – verify ring/territory stats, timers, spectator badge, connection status.
  - [x] [`GameEventLog.tsx`](src/client/components/GameEventLog.tsx) – verify rendering of moves, system events, and victory messages from the view model.
  - [x] [`GameHistoryPanel.tsx`](src/client/components/GameHistoryPanel.tsx) – history list and selection behaviour.
- [ ] Add light-weight tests for supporting components:
  - [ ] [`AIDebugView.tsx`](src/client/components/AIDebugView.tsx).
  - [ ] [`LoadingSpinner.tsx`](src/client/components/LoadingSpinner.tsx) and small UI primitives under `src/client/components/ui/`.
- [ ] Add targeted unit tests for key pages that currently rely primarily on E2E coverage:
  - [ ] [`LobbyPage.tsx`](src/client/pages/LobbyPage.tsx) – lobby filters, game list, and navigation wiring.
  - [x] [`BackendGameHost.tsx`](src/client/pages/BackendGameHost.tsx) and
        [`SandboxGameHost.tsx`](src/client/pages/SandboxGameHost.tsx) – host‑level HUD + event
        log/chat/diagnostics wiring (see `tests/unit/client/BackendGameHost.test.tsx`,
        `tests/unit/BackendGameHost.boardControls.test.tsx`, and
        `tests/unit/client/SandboxGameHost.test.tsx`).

For the full gap analysis, see **Focus Area 5: Test Coverage Gaps** in
[`docs/PASS15_ASSESSMENT_REPORT.md`](docs/PASS15_ASSESSMENT_REPORT.md).

## Phase 4 – Advanced AI (P2)

### P2.1 – Stronger Opponents

- [ ] Promote at least one stronger AI implementation
      (Minimax/MCTS/NeuralNet) from
      [`ai-service/app/ai`](ai-service/app/ai/__init__.py) into the primary
      `/ai/move` path, behind a non-default `AIProfile`.
- [ ] Add tests around `AIServiceClient` to cover latency, timeouts, and
      fallback usage in more detail.

### P2.2 – AI Telemetry

- [ ] Add lightweight logging/metrics around AI calls in
      [`AIServiceClient`](src/server/services/AIServiceClient.ts) and
      [`AIInteractionHandler`](src/server/game/ai/AIInteractionHandler.ts),
      capturing:
  - Request type
  - Duration
  - Success vs failure vs fallback
- [ ] Surface basic metrics in the existing Prometheus/Grafana stack
      defined in [`docker-compose.yml`](docker-compose.yml).

### P2.3 – AI Wiring & Determinism (New)

- [ ] **Wire up MinimaxAI:** Update `_create_ai_instance` in `ai-service/app/main.py` to instantiate `MinimaxAI` for mid-high difficulties instead of falling back to `HeuristicAI`.
- [ ] **Fix RNG Determinism:**
  - [ ] Replace global `random` usage in Python AI with per-game seeded RNG instances.
  - [ ] Update `ZobristHash` to use a stable, seeded RNG instead of global `random.seed(42)`.
  - [ ] Pass RNG seeds from TS backend to Python service in `/ai/move` requests.

## Cross-Cutting – Documentation & CI

- [ ] Keep the following docs synchronized whenever behaviour changes:
  - [`README.md`](README.md)
  - [`QUICKSTART.md`](QUICKSTART.md)
  - [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md)
  - [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md)
- [ ] When adding new scenario suites or parity harnesses, update
      [`tests/README.md`](tests/README.md) and
      [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md) so newcomers
      know where to look.
- [ ] Gradually tighten Jest coverage thresholds in
      [`jest.config.js`](jest.config.js) once the scenario matrix and
      parity suites are stable.
- [ ] CI gating refinement:
  - [ ] After several consecutive green runs, decide whether to promote
        `TS Parity` and `TS Integration` lanes to required checks in the
        `main` ruleset, or document why they remain informational.
- [ ] Security/dependency audit configuration:
  - [ ] Capture current `npm audit` / `pip-audit` status in docs (including
        the applied `aiohttp` / `torch` fixes).
  - [ ] Document the intentionally deferred FastAPI/Starlette upgrade and
        Python 3.13 constraint, including how `pip-audit` is run in CI and
        which vulnerability IDs (if any) are temporarily ignored.

This file is intentionally concise; for deeper narrative context and
ongoing issue lists, refer to
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and
[`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md).

## Consolidated Execution Tracks & Plan

The following tracks and steps summarize the recommended direction from the
latest project assessment. They should be kept in sync with
`CURRENT_STATE_ASSESSMENT.md` and `STRATEGIC_ROADMAP.md` as work progresses.

### Track 1 – Rules/FAQ Scenario Matrix & Parity Hardening (P0)

- [x] Expand and maintain `RULES_SCENARIO_MATRIX.md` to map
      `ringrift_complete_rules.md` + FAQ examples to concrete Jest suites.
- [x] For each major rule cluster (movement, chain captures, lines,
      territory, victory), ensure at least one emblematic scenario is tested in
      both:
  - [x] Backend engine (`GameEngine` / `RuleEngine` under `src/server/game/`).
  - [x] Client sandbox engine (`ClientSandboxEngine` under
        `src/client/sandbox/ClientSandboxEngine.ts`).
- [x] Keep scenario IDs consistent across:
  - [x] `RULES_SCENARIO_MATRIX.md`.
  - [x] `tests/scenarios/rulesMatrix.ts`.
  - [x] Suites under `tests/scenarios/`.
- [ ] Use parity harnesses as scenario generators:
  - [ ] Run `Backend_vs_Sandbox.traceParity.test.ts`,
        `Sandbox_vs_Backend.seed5.traceDebug.test.ts`, and
        `Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` regularly.
  - [ ] When a parity failure occurs, extract the first divergence via the
        `TraceParity.*.firstDivergence` helpers.
  - [ ] Promote the failing position to a focused unit test under
        `tests/unit/` (e.g. movement/capture/placement parity).
  - [ ] Fix the underlying engine/sandbox discrepancy and mark the
        corresponding row in `RULES_SCENARIO_MATRIX.md` as covered.
- [ ] Define and document "must-cover" scenario sets per rules axis
      (movement, capture/chains, lines, territory, victory) and tie Jest
      coverage thresholds to these modules once the baseline is in place.
- [ ] For each axis ID (M*/C*/L*/T*/V\*) in `RULES_SCENARIO_MATRIX.md`,
      ensure there is at least one `rulesMatrix.ts` scenario and matching
      backend + sandbox test suite, and cross-link those suites from the
      matrix.

### Track 2 – Multiplayer Lifecycle & HUD/UX (P1)

- [ ] Tighten WebSocket lifecycle around reconnection and late joins:
  - [ ] In `WebSocketServer` + `WebSocketInteractionHandler`, ensure that
        reconnecting clients re-emit `join_game`, rehydrate state from the
        DB + `GameEngine`, and clear any stale choices.
  - [x] Add focused Jest coverage for the connection state machine and reconnection window in `WebSocketServer.connectionState.test.ts`.
  - [ ] Extend coverage with integration-style reconnect + `game_over` flows and cross-link the relevant suites from `tests/README.md`.
- [x] Clarify and enforce spectator semantics:
  - [x] Ensure spectators are always read-only at the server level.
  - [x] Provide a dedicated spectator view in the client (using
        `GameContext` but with input disabled).
  - [x] Add tests for spectator join/leave flows.
- [ ] Enhance `GameHUD` and GamePage UX:
  - [ ] Ensure `GameHUD` shows:
    - [ ] Current player and phase.
    - [ ] Per-player rings in hand / on board / eliminated.
    - [ ] Territory spaces per player.
    - [ ] (Optional) Timer readouts based on `timeControl` and
          `timeRemaining`.
  - [ ] Add a minimal per-game event log in `GamePage` for moves,
        PlayerChoices, and `game_over` events.
  - [ ] Improve phase-specific prompts and invalid-move feedback (toasts,
        subtle animations) in the frontend.

### Track 3 – Sandbox as a First-Class Rules Lab (P0–P1)

- [ ] Finish unifying sandbox canonical mutations in
      `ClientSandboxEngine`:
  - [ ] Route all AI and human actions through the canonical
        `applyCanonicalMoveInternal` path.
  - [ ] Remove bespoke mutation logic in sandbox AI so that sandbox phases
        and move types stay in lockstep with backend `GameEngine` /
        `RuleEngine`.
- [ ] Expose rules/FAQ scenarios directly in the sandbox UI:
  - [ ] Add a simple scenario selector (e.g. dropdown) in `/sandbox` backed
        by `rulesMatrix.ts` / `RULES_SCENARIO_MATRIX.md`.
  - [ ] Allow loading a named scenario into the sandbox for visual
        inspection and step-through play.
- [ ] Add visual debugging aids to the sandbox view:
  - [ ] Overlays for detected lines and their rewards.
  - [ ] Territory region highlighting and disconnection visualization.
  - [ ] Chain capture path visualization (e.g. arrows or highlighted
        segments).

### Track 4 – Incremental AI Improvements & Observability (P1–P2)

- [ ] Add lightweight metrics/logging around AI calls:
  - [ ] In `AIServiceClient` and `AIInteractionHandler`, log request type,
        latency, success/failure, and fallback usage.
  - [ ] Optionally expose these metrics via existing or new monitoring
        tooling (e.g. Prometheus if added later).
- [ ] Reflect AI mode in the HUD:
  - [ ] Display whether each AI player is using service-backed or local
        heuristic decisions.
  - [ ] Add a simple AI service health indication (if health endpoint is
        defined).
- [ ] Make targeted Heuristic AI improvements (within tight bounds):
  - [ ] Refine evaluation weights for early/mid/late game phases using
        available data (line potential, basic mobility, territory).
  - [ ] Avoid large structural changes until rules/parity work is fully
        stabilized.
- [ ] Prepare Python AI foundations (background work):
  - [ ] Implement robust `get_valid_moves(game_state)` and
        `apply_move(game_state, move)` in the AI service, per
        `AI_IMPROVEMENT_PLAN.md`.
  - [ ] Keep this evolution behind existing endpoints so the TS boundary
        remains stable.

### Track 5 – Persistence, Replays, and Stats (P2)

- [ ] Clarify and enforce game lifecycle transitions in the database:
  - [ ] Ensure consistent `status` transitions (`WAITING` → `ACTIVE` →
        `COMPLETED`), along with `startedAt` / `endedAt` timestamps.
  - [ ] Add targeted tests around lifecycle persistence.
- [ ] Surface simple history/replay in the UI:
  - [ ] Add a move list panel (based on `GameHistoryEntry`) in the
        GamePage HUD.
  - [ ] Plan a dedicated replay view powered by stored moves and backend
        `GameEngine` replays once the basics are stable.
- [ ] Build initial stats/leaderboards once game results are reliable:
  - [ ] Aggregate per-user stats (wins/losses, rating if enabled).
  - [ ] Expose a minimal leaderboard view in the client.

### Track 6 – Dev UX Evaluation & Dependency Modernization (P1)

- [ ] Operationalize dev stack startup for UX evaluation
  - [ ] Document a one-page "Start the Stack" recipe (Docker, AI service, `npm run dev`)
  - [ ] Ensure `.env` includes correct `DATABASE_URL`, `AI_SERVICE_URL`, and any orchestrator/feature flags
  - [ ] Verify that backend + client start cleanly and can be restarted without manual cleanup
- [ ] Run structured UX walkthroughs
  - [ ] Auth flows: register, login, logout, and common error cases
  - [ ] Lobby flows: game creation/joining (AI vs human), reconnection, and spectator behavior
  - [ ] Game flows: HUD clarity, event log usefulness, resignation/victory handling
  - [ ] Sandbox flows: scenario setup (placement/movement/capture/territory/victory) and debug affordances
- [ ] Synthesize UX findings into a prioritized plan
  - [ ] Compile UX issues per surface (auth, lobby, game, sandbox) with severity labels
  - [ ] Tag items that relate to orchestrator rollout, TS↔Python rules parity, or AI service behavior
  - [ ] Identify 1–2 "first slice" UX improvements that are low risk but high leverage
- [ ] Perform dependency modernization pass
  - [ ] For Node/TS (root `package.json`):
    - [x] Run `npm outdated` and capture current vs latest versions
    - [ ] Update dependencies to the latest compatible versions (prefer automated tools like `npm update` / `npm-check-updates`)
    - [ ] Re-run Jest + Playwright suites and fix any breakages introduced by tooling/runtime bumps
  - [ ] For Python (`ai-service/requirements.txt`):
    - [x] Run `pip-audit -r requirements.txt` to confirm security status
    - [x] Upgrade security‑critical packages to patched versions compatible with the current Python/runtime stack (e.g. `aiohttp`, `torch`) and re‑run pytest / contract tests.
    - [ ] Perform a broader modernization pass on remaining outdated packages and re-run pytest (rules/parity, training, and invariants) to address any regressions.
  - [x] Document any intentionally deferred major‑version upgrades and their rationale (e.g. FastAPI/Starlette constraints on `starlette` for Python 3.13, to be revisited with a future stack update)
  - [ ] Plan Jest stabilization order (post Node Wave A / Python Wave P1) and keep this list updated as suites are fixed:
    - [ ] **Auth & account flows:** `tests/unit/auth.routes.test.ts`, `tests/unit/user.delete.routes.test.ts`, plus `tests/integration/accountDeletion.test.ts`.
    - [ ] **Orchestrator multiphase & AI integration:** `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`, `tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts`, `tests/integration/GameSession.aiOrchestrator.integration.test.ts` (once re-enabled), and `tests/unit/GameSession.aiRequestState.test.ts`.
    - [ ] **Core rules semantics (high-signal):** multi-ring placement and landing-on-own-marker semantics (`tests/unit/RuleEngine.placementMultiRing.test.ts`, `tests/unit/GameEngine.landingOnOwnMarker.test.ts`, `tests/unit/MovementAggregate.shared.test.ts`, `tests/unit/movement.shared.test.ts`, `tests/unit/GameEngine.movement.shared.test.ts`).
    - [ ] **Board/UX surfaces:** `tests/unit/components/BoardView.test.tsx`, `tests/unit/GameContext.reconnect.test.tsx`, `tests/unit/LobbyPage.test.tsx`, and HUD/event-log suites.
    - [ ] **Heuristics and AI surfaces:** `tests/unit/heuristicEvaluation.test.ts`, `tests/unit/heuristicEvaluation.shared.test.ts`, `tests/unit/heuristicParity.shared.test.ts`, `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`, plus the Python side `tests/test_heuristic_ai.py` and `tests/test_heuristic_parity.py` (now green after Wave P1).

### Suggested 2–4 Week Execution Plan (Guidance)

These are not strict milestones but a suggested ordering that can be
reflected by checking off the above track items.

- [ ] **Week 1 – Lock in Rules Confidence (Tracks 1 & 3)**
  - [ ] Expand `RULES_SCENARIO_MATRIX.md` and scenario suites under
        `tests/scenarios/` for key movement, capture, line, territory, and
        victory examples.
  - [ ] Run parity suites regularly; for each failure, promote a minimal
        unit test and fix the underlying discrepancy.
  - [ ] Begin unifying sandbox canonical move handling in
        `ClientSandboxEngine`.
- [ ] **Week 2 – Multiplayer Lifecycle & HUD (Track 2 + more Track 1)**
  - [ ] Implement reconnection + spectator semantics end-to-end and add
        tests.
  - [ ] Flesh out `GameHUD` and integrate a basic event log into
        `GamePage`.
  - [ ] Continue adding rules scenarios and parity fixes.
- [ ] **Week 3 – Sandbox UX & AI Observability (Tracks 3 & 4)**
  - [ ] Add scenario picker into the sandbox UI and basic visual helpers
        for lines/territory/chain captures.
  - [ ] Add AI telemetry and, optionally, surface simple metrics.
  - [ ] Implement one or two bounded heuristic AI improvements that
        clearly improve play.
- [ ] **Week 4 – Persistence/Polish & Buffer (Track 5 + cleanup)**
  - [ ] Introduce a minimal move history/replay panel in GamePage or a
        dedicated route.
  - [ ] Update docs (`CURRENT_STATE_ASSESSMENT`, `TODO`,
        `STRATEGIC_ROADMAP`, `DOCUMENTATION_INDEX`) to reflect new coverage
        and UX.
  - [ ] Use remaining time to close lingering P0 items and fix UX
        papercuts discovered via playtesting.

This section is intentionally high-level and should be pruned or merged into
earlier phases as items are completed or re-scoped.
