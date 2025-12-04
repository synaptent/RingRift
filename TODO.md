# RingRift TODO / Task Tracker

> **Doc Status (2025-11-30): Active (execution/backlog tracker)**
>
> - Canonical high-level task/backlog tracker for near- and mid-term work.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

**Last Updated:** December 3, 2025

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
those documents, not the other way around. For **where** rules semantics are
allowed to live (shared engine vs. host adapters vs. UI/transport), defer to
the **Rules Entry Surfaces / SSoT checklist** in
[`docs/RULES_ENGINE_SURFACE_AUDIT.md`](docs/RULES_ENGINE_SURFACE_AUDIT.md#0-rules-entry-surfaces-ssot-checklist).

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
- [x] Python: 824 tests passing

**Documentation produced:**

- [`docs/drafts/PHASE1_REMEDIATION_PLAN.md`](docs/drafts/PHASE1_REMEDIATION_PLAN.md)
- [`docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md`](docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md)
- [`docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md`](docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md)
- [`src/shared/engine/orchestration/README.md`](src/shared/engine/orchestration/README.md)

---

## Phase 2 – Robustness & Testing (IN PROGRESS, P0)

> **Current Focus (Dec 2025):** Phase‑2 work is now centered on
> **engine/host lifecycle robustness** rather than core rules semantics:
>
> - **P0 – WebSocket lifecycle & reconnection window hardening** – Tighten
>   `docs/CANONICAL_ENGINE_API.md` WebSocket sections and add explicit
>   coverage rows in `RULES_SCENARIO_MATRIX.md` for reconnection windows,
>   lobby subscription/unsubscription, rematch flows, and spectator joins;
>   keep `GameReconnection.test.ts`, `LobbyRealtime.test.ts`, and
>   Playwright reconnection E2E suites green as the canonical coverage set.
> - **P0 – Host parity for advanced phases** – Finish treating backend
>   `GameEngine` / `RuleEngine`, `ClientSandboxEngine`, and
>   `ai-service/app/game_engine.py` as thin adapters over the shared
>   orchestrator/aggregates for `chain_capture`, `line_processing`,
>   `territory_processing`, and explicit self‑elimination. Any remaining
>   host‑level rules logic should either call shared helpers or be marked
>   diagnostic/legacy only.
> - **P0 – TS↔Python territory & forced‑elimination parity finish‑up** –
>   Use contract vectors and targeted territory/forced‑elimination tests to
>   close the remaining gaps between TS and Python (region detection,
>   eligibility filters, elimination ordering, host‑level forced
>   elimination). Contract vectors remain the SSoT for cross‑language
>   behaviour.

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

- [x] Keep the S-invariant tests passing and expand them as new rules
      interactions are implemented:
  - [`ProgressSnapshot.core.test.ts`](tests/unit/ProgressSnapshot.core.test.ts)
  - Sandbox AI simulation diagnostics in
    [`ClientSandboxEngine.aiSimulation.test.ts`](tests/unit/ClientSandboxEngine.aiSimulation.test.ts)
  - Backend AI-style simulations in
    [`GameEngine.aiSimulation.test.ts`](tests/unit/GameEngine.aiSimulation.test.ts)
- [x] For any new rule that can change markers, collapsed spaces, or
      eliminated rings, ensure it is reflected in
      [`computeProgressSnapshot`](src/shared/engine/core.ts) and covered by
      tests.
- [x] Add explicit S-invariant coverage for orchestrator multi-phase flows:
  - [x] `chain_capture` + `continue_capture_segment` (backend; see `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`)
  - [x] `territory_processing` with mandatory self-elimination (backend; see `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`)
  - [ ] Sandbox coverage remains optional; rely on shared-engine + sandbox S tests in `ProgressSnapshot.sandbox.test.ts`.
- [x] Add an AI-service S-invariant parity check:
  - [x] Mirror core S-invariant scenarios in Python via
        `ai-service/tests/parity/test_rules_parity_fixtures.py` and
        `ai-service/tests/parity/test_ai_plateau_progress.py`, asserting S
        parity/deltas against TS-generated fixtures.

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
        to) `RegionOrderChoice` for ordering (explicit elimination moves are
        now surfaced via `eliminate_rings_from_stack` Moves on both backend
        and sandbox hosts using the shared decision helpers + orchestrator).
- [x] Update `GameEngine` to drive these phases via `makeMove` rather than
      direct `PlayerInteractionManager` calls, so that:
  - [x] Human clients and AI both select from the same `getValidMoves`
        result set for all interactive decisions in the canonical
        orchestrator-backed path (legacy helpers remain only for
        diagnostics and archived tests).
  - [x] Internal post-move processors (`lineProcessing`,
        `territoryProcessing`, `captureChainEngine`) are refactored to apply
        selected decision-moves instead of bespoke `PlayerChoice` branches
        on the production/orchestrator path.
- [ ] Adjust `PlayerInteractionManager` and WebSocket handlers so that
      frontends treat these new decision-move types as first-class actions
      (selecting among moves) rather than distinct `PlayerChoice` payloads.
- [x] Keep `ClientSandboxEngine` in sync by:
  - [x] Importing the same extended `Move` types.
  - [x] Mirroring the new interactive phases and decision-move handling in
        the sandbox (e.g., chain capture, line processing, territory
        processing, elimination) using the existing sandbox helpers
        (`sandboxMovement.ts`, `sandboxElimination`) plus the
        `SandboxOrchestratorAdapter`.
  - [x] Updating `sandboxAI` to select among these richer `Move` sets,
        staying in lockstep with backend `getValidMoves`.
- [x] Align sandbox Move/phase handling with backend:
  - [x] Route all sandbox actions (human and AI) through the canonical
        `applyCanonicalMoveInternal` path using the same `Move` types and
        `GamePhase` transitions as the backend, including `chain_capture`,
        `line_processing`, and `territory_processing`.
  - [x] Ensure `ClientSandboxEngine` respects the same decision semantics
        for `continue_capture_segment`, `process_line`, and
        `process_territory_region`.
  - [x] Update sandbox parity and RulesMatrix-backed tests to assert both
        the legal-move sets and resulting phases for these advanced phases.
- [ ] Extend and/or add parity tests to cover the new decision-move
      surface:
  - [x] Backend vs sandbox parity for chain capture continuation decisions.
  - [x] Backend vs sandbox parity for line order/reward and region
        order/elimination decisions.
  - [ ] Trace parity tests that confirm all such decisions are now
        represented as `Move`s and faithfully replayable.

**Current P0.4 status (as of 2025‑11‑19):**

- [x] Backend `GameEngine` / `RuleEngine` now model capture-chain continuation via a distinct `chain_capture` phase and `continue_capture_segment` moves.
- [x] Backend capture-sequence enumeration now uses `captureChainEngine.getCaptureOptionsFromPosition` plus shared `validateCaptureSegmentOnBoard`, keeping [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts) green across square and hex boards.
- [x] Backend territory-processing now enumerates explicit `eliminate_rings_from_stack` Moves via `RuleEngine.getValidEliminationDecisionMoves`, and `RingEliminationChoice.options[].moveId` / `RegionOrderChoice.options[].moveId` are wired to canonical `Move.id` values for elimination and disconnected-region decisions (sandbox Move/phase parity and WebSocket/AI adoption of these Move ids have been implemented via the orchestrator-backed adapters and choice→Move mapping).
- [x] Chain-capture + heuristic coverage suites are now passing:
  - [x] [`ComplexChainCaptures.test.ts`](tests/scenarios/ComplexChainCaptures.test.ts) – chain capture scenarios pass
  - [x] [`RulesMatrix.ChainCapture.GameEngine.test.ts`](tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts) – backend chain capture passes
  - [x] [`ClientSandboxEngine.chainCapture.test.ts`](tests/unit/ClientSandboxEngine.chainCapture.test.ts) – sandbox chain capture passes
  - [x] [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts) – cyclic/triangle patterns pass
  - [x] [`Backend_vs_Sandbox.aiParallelDebug.test.ts`](archive/tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts) (archived diagnostic harness; no longer part of gating suites)
  - [x] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts) – seeds 5/14 complete to game end with parity
  - [x] [`TraceParity.seed*.firstDivergence.test.ts`](tests/unit/TraceParity.seed5.firstDivergence.test.ts) – diagnostic helpers for legacy path; skipped when orchestrator enabled (canonical path); aiHeuristicCoverage provides primary parity verification
- [x] Sandbox engine and `sandboxAI` now participate in the new `chain_capture` / `continue_capture_segment` Move model for AI turns and canonical traces; remaining divergences (e.g. seed 14 trace parity) are localized and tracked via `TraceParity.seed14.*` / `ParityDebug.seed14.*` helpers.

**Near-term P0.4 tasks inferred from current test failures (historical):**

- [x] Finalize backend chain-capture semantics for cyclic/triangle patterns so that FAQ scenarios in [`rulesMatrix.ts`](tests/scenarios/rulesMatrix.ts) and [`GameEngine.cyclicCapture.*.test.ts`](tests/unit/GameEngine.cyclicCapture.scenarios.test.ts) pass under the new model. (See `GameEngine.cyclicCapture.scenarios.test.ts` and `tests/scenarios/ComplexChainCaptures.test.ts` for the current FAQ-aligned coverage.)
- [x] Migrate `ComplexChainCaptures` and `RulesMatrix.ChainCapture` suites to drive chains via `chain_capture` + `continue_capture_segment` rather than internal while-loops. (Both suites now resolve mandatory continuations via `GameEngine.getValidMoves` in the `chain_capture` phase.)
- [x] Mirror the chain-capture phase and `continue_capture_segment` moves into [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts) and [`sandboxAI`](src/client/sandbox/sandboxAI.ts), then restore:
  - [x] [`Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`](tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts)
  - [x] `TraceParity.seed*.firstDivergence.test.ts` (trace parity helpers for seeds 5/14/17)
- [x] Once chain capture is stable across backend and sandbox, extend the same Move/phase unification to line-processing and territory-processing (as described above). Backend and sandbox now both enumerate `process_line`, `choose_line_reward`, `process_territory_region`, and `eliminate_rings_from_stack` Moves via the shared helpers + orchestrator adapters, and these paths are exercised by the existing line/territory decision suites and sandbox/backend parity tests.
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

## Wave 5 – Orchestrator Production Rollout (P0/P1) - ✅ COMPLETE

> **Goal:** Make the canonical orchestrator + adapters the _only_ production
> turn path, with safe rollout across environments and removal of legacy
> turn‑processing code once stable.
>
> **Status (2025-12-01):** ✅ **PHASE 3 COMPLETE**
>
> - Orchestrator at 100% in all environments
> - ~1,176 lines legacy code removed
> - Feature flags hardcoded/removed
> - Legacy paths deprecated and removed

### Wave 5.1 – Staging orchestrator enablement ✅

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

### Wave 5.2 – Orchestrator‑ON validation suites ✅

Run these suites with orchestrator adapters forced ON (backend + sandbox) and
keep them green; treat regressions as P0 until resolved.

- [x] WebSocket + session flows (backend host):
  - [x] `tests/integration/GameReconnection.test.ts`
  - [x] `tests/integration/GameSession.aiDeterminism.test.ts`
  - [x] `tests/integration/AIResilience.test.ts`
  - [x] `tests/integration/LobbyRealtime.test.ts`
- [x] Orchestrator multi‑phase scenarios (backend + sandbox):
  - [x] `tests/scenarios/Orchestrator.Backend.multiPhase.test.ts`
  - [x] `tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts`
- [x] Decision‑heavy rules flows (line / territory / chain‑capture):
  - [x] Line reward / line order suites (GameEngine + ClientSandboxEngine).
  - [x] Territory processing + self‑elimination suites.
  - [x] Complex chain‑capture scenarios.
- [x] Invariant and contract guards (already wired in CI, but re‑run locally
      when changing flags):
  - [x] Orchestrator invariant soaks via `scripts/run-orchestrator-soak.ts`.
  - [x] Contract vectors in `tests/contracts/contractVectorRunner.test.ts` and
        `ai-service/tests/contracts/test_contract_vectors.py`.
        _All contract vectors pass with orchestrator-ON as of the `test:orchestrator-parity` suite._

### Wave 5.3 – Production rollout with circuit‑breakers ✅

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

### Wave 5.4 – Legacy path deprecation and cleanup ✅

- [x] Mark remaining legacy turn‑processing paths as **deprecated** in:
  - [x] `src/server/game/RuleEngine.ts` / `GameEngine.ts`.
        _`processMove()`, `processChainReactions()`, `processLineFormation()`,
        `processTerritoryDisconnection()` all removed (~120 lines)._
  - [x] `src/client/sandbox/ClientSandboxEngine.ts`.
        _Legacy methods removed (786 lines)._
  - [x] Any sandbox‑only helpers that still implement bespoke turn/phase loops.
        _Tier 1 modules (sandboxTurnEngine, sandboxMovementEngine, sandboxLinesEngine,
        sandboxTerritoryEngine) deleted in PASS20._
- [x] Migrate tests that still exercise legacy‑only flows to:
  - [x] Go through `TurnEngineAdapter` / `SandboxOrchestratorAdapter`, **or**
  - [x] Target the shared engine directly (aggregates/helpers) where host
        concerns are not required.

  **Legacy‑path tests migration status (completed PASS20):**
  All identified tests have been audited and migrated:
  - Tests now use shared engine aggregates (e.g. `GameEngine.lines.scenarios.test.ts` → LineAggregate)
  - Tests now use orchestrator adapters (e.g. `Orchestrator.Backend.multiPhase.test.ts`)
  - Tests converted to stubs with proper `@deprecated` documentation (6 files)
  - Tests skipped with explanation (e.g. `GameEngine.territoryDisconnection.hex.test.ts`)

  **Historical harnesses now trimmed to stubs / contract‑vector coverage only:**
  - Obsolete test files deleted (193 lines)
  - See PASS20_COMPLETION_SUMMARY.md for full details

- [x] Legacy turn‑processing implementations removed from backend and sandbox hosts
      _PASS20 complete: ~1,176 lines removed total_
      _Phase 4 (Tier 2 sandbox cleanup) deferred to post-MVP_

## Wave 6 – Observability & Production Readiness (P0/P1) - ✅ COMPLETE

> **Goal:** Implement comprehensive observability infrastructure and validate production-scale performance.
>
> **Status (2025-12-01):** ✅ **COMPLETE**
>
> - 3 Grafana dashboards created with 22 panels
> - k6 load testing framework + 4 scenarios implemented
> - Monitoring stack runs by default
> - DOCUMENTATION_INDEX.md created

### Wave 6.1 – Grafana Dashboard Implementation ✅

- [x] Create game-performance dashboard (moves, AI latency, abnormal terminations)
- [x] Create rules-correctness dashboard (parity, invariants)
- [x] Create system-health dashboard (HTTP, WebSocket, infrastructure)
- [x] Wire dashboards to Prometheus data sources
- [x] Add provisioning configuration for automated deployment

### Wave 6.2 – Load Testing Framework ✅

- [x] Implement k6 load testing tool
- [x] Create Scenario P1: Mixed human vs AI ladder (40-60 players, 20-30 moves)
- [x] Create Scenario P2: AI-heavy concurrent games (60-100 players, 10-20 AI games)
- [x] Create Scenario P3: Reconnects and spectators (40-60 players + 20-40 spectators)
- [x] Create Scenario P4: Long-running AI games (10-20 games, 60+ moves)

### Wave 6.3 – Monitoring Infrastructure ✅

- [x] Move monitoring stack from optional profile to default
- [x] Ensure Prometheus metrics export from all services
- [x] Configure alerting thresholds
- [x] Document dashboard usage and alert response

### Wave 6.4 – Documentation & Indexing ✅

- [x] Create DOCUMENTATION_INDEX.md comprehensive index
- [x] Update all references to monitoring capabilities
- [x] Document load testing scenarios and SLOs
- [x] Add PASS21 assessment report

## Wave 7 – Production Validation & Scaling ✅ COMPLETE (Dec 3, 2025)

> **Goal:** Validate system performance at production scale and establish operational baselines.
>
> **Status:** ✅ COMPLETE (Dec 3, 2025)

### Wave 7.1 – Load Test Execution ✅ COMPLETE (Dec 3, 2025)

- [x] Run Scenario P1 against local Docker (player-moves) – 100% success, 0% errors
- [x] Run Scenario P2 against local Docker (concurrent-games) – p95 latency 10.79ms (<400ms target)
- [x] Run Scenario P3 against local Docker (websocket-stress) – 100% connection success, p95 latency 2ms
- [x] Run Scenario P4 against local Docker (game-creation) – p95 latency 15ms (<800ms target)
- [x] Document results and identify bottlenecks → [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md)

**Issue Fixed:** Game ID validation in Docker container was outdated (only accepted UUID, not CUID). Rebuilt container with updated `GameIdParamSchema` that accepts both formats.

### Wave 7.2 – Baseline Metrics Establishment ✅ COMPLETE (Dec 3, 2025)

- [x] Capture "healthy system" metric ranges from local Docker runs
  - Game creation p95: 15ms (target <800ms) – 53x headroom
  - GET /api/games/:id p95: 10.79ms (target <400ms) – 37x headroom
  - WebSocket message latency p95: 2ms (target <200ms) – 100x headroom
- [x] Document p50/p95/p99 latencies → [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md)
- [x] Establish capacity model (estimated from 5-50 VU tests):
  - Game creation: ~1.6/s sustained, ~100/min at moderate load
  - WebSocket connections: 50 tested, 500+ projected
  - Error rate: 0% at test load
- [x] Tune alert thresholds based on observed behavior → recommendations in baseline report

### Wave 7.3 – Operational Drills ✅ COMPLETE (Dec 3, 2025)

- [x] Execute secrets rotation drill – Token invalidation verified, ~30s recovery
- [x] Execute backup/restore drill – 11MB backup, full integrity verified (40K games)
- [x] Simulate incident response scenarios – AI service outage, detection <75s
- [x] Document lessons learned and refine runbooks → Added to [`docs/LOAD_TEST_BASELINE_REPORT.md`](docs/LOAD_TEST_BASELINE_REPORT.md)

**Key Findings:**

- Docker Compose doesn't auto-reload .env changes (must export vars)
- Nginx restart needed after app container recreation
- Prometheus scrape interval (15s) determines detection speed

### Wave 7.4 – Production Preview ✅ COMPLETE (Dec 3, 2025)

- [x] Deploy to production-like environment (local Docker with full stack)
- [x] Run smoke tests with real traffic patterns (k6 load scenarios)
- [x] Validate monitoring and alerting:
  - Prometheus: 3/3 targets healthy, 40 alert rules
  - Grafana: Healthy with Prometheus datasource
  - Alertmanager: ⚠️ Needs production notification config (created local config)
- [x] Execute go/no-go checklist → [`docs/GO_NO_GO_CHECKLIST.md`](docs/GO_NO_GO_CHECKLIST.md)

**Verdict: ✅ GO (with caveats)**

- System demonstrates production readiness for soft launch
- All SLOs met with 37x-100x headroom
- Alertmanager needs production notification channels before full launch

### Wave 7.5 – k6 scenario contract/protocol alignment (Code/Debug)

- [x] Fix contract / ID behavior for `GET /api/games/:gameId` under k6 scenarios (Code/Debug)
  - `concurrent-games` and `player-moves` under `tests/load/scenarios/` now create, track, and retire game IDs in a way that reflects the backend’s actual lifecycle and expiry semantics (terminal statuses, 404 expiry, bounded poll counts), so 4xx rates are driven by genuine behaviour (e.g., rate limits) rather than `GAME_INVALID_ID` from stale IDs.
  - Any remaining HTTP errors in these scenarios should be interpreted as capacity/behaviour issues surfaced by the backend, not as fundamental contract mismatches in the k6 harness.
  - References: [`GAME_PERFORMANCE.md`](docs/runbooks/GAME_PERFORMANCE.md), [`PASS22_COMPLETION_SUMMARY.md`](docs/PASS22_COMPLETION_SUMMARY.md), [`PASS22_ASSESSMENT_REPORT.md`](docs/PASS22_ASSESSMENT_REPORT.md), [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

- [x] Align WebSocket message format for `websocket-stress` with production client protocol (Code/Debug)
  - [`tests/load/scenarios/websocket-stress.js`](tests/load/scenarios/websocket-stress.js) now speaks the same Socket.IO v4 / Engine.IO v4 wire protocol as `WebSocketServer` (handshake, ping/pong, lobby events), eliminating "message parse error" closes when the server is healthy.
  - With protocol alignment in place, connection-duration and message-latency thresholds in this scenario correspond to realistic production SLOs rather than protocol mismatches.
  - References: [`GAME_PERFORMANCE.md`](docs/runbooks/GAME_PERFORMANCE.md), [`PASS21_ASSESSMENT_REPORT.md`](docs/PASS21_ASSESSMENT_REPORT.md), [`PASS22_COMPLETION_SUMMARY.md`](docs/PASS22_COMPLETION_SUMMARY.md), [`PASS22_ASSESSMENT_REPORT.md`](docs/PASS22_ASSESSMENT_REPORT.md), [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

---

## Wave 8 – Player Experience & UX Polish (P1 - IN PROGRESS)

> **Goal:** Transform developer-oriented UI into player-friendly experience suitable for public release.
>
> **Status:** 🔄 IN PROGRESS (Waves 8.1, 8.3, 8.4 Complete; 8.2 Mostly Complete)
> **Spec:** [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §Wave 8

### Wave 8.1 – First-Time Player Experience ✅ COMPLETE (Dec 4, 2025)

- [x] Create onboarding modal for first-time players (`OnboardingModal.tsx`)
  - Multi-step introduction: Welcome, Phases, Victory Conditions, Ready to Play
  - Keyboard navigation (arrow keys, Enter, Escape)
- [x] Create `useFirstTimePlayer` hook for tracking onboarding state
  - Persists to localStorage, tracks welcome seen, games completed
- [x] Enhance "Learn the Basics" preset visibility
  - Pulsing animation for first-time players
  - "Start Here" badge and indicator
- [x] Simplify sandbox presets – hide advanced options behind "Show Advanced"
  - Collapsed by default for first-time players
  - Expands Scenarios, Self-Play, and Manual Config sections
- [ ] Add contextual tooltips explaining game mechanics (deferred to 8.6)
- [ ] Redesign HUD visual hierarchy (existing HUD functional, minor polish deferred)

### Wave 8.2 – Game Flow Polish ✅ MOSTLY COMPLETE

- [x] Improve phase-specific prompts with clearer action buttons (Dec 3, 2025)
- [x] Enhanced invalid-move feedback (subtle animations, explanatory toasts) (Dec 4, 2025)
- [x] Decision-phase countdown with visual urgency (color changes, pulsing) (Dec 3, 2025)
- [x] Victory/defeat screens with game summary (Dec 3, 2025)
- [ ] Key moments replay (deferred to 8.3)

### Wave 8.3 – UI/UX Theme Polish ✅ COMPLETE (Dec 4, 2025)

- [x] Dark theme for turn number panel (`GameProgress` component)
- [x] Dark theme for player card panels (`PlayerCardFromVM`, `RingStatsFromVM`)
- [x] Extract `VictoryConditionsPanel` component for flexible placement
- [x] Move Victory panel below game info panel (sandbox layout)
- [x] Fix MoveHistory scroll to stay within container (prevent page scroll)

### Wave 8.4 – Spectator & Analysis Experience ✅ COMPLETE (Dec 4, 2025)

- [x] `SpectatorHUD` component with dedicated spectator layout
  - Player standings with ring/territory stats
  - Current player indicator with phase hints
  - Collapsible analysis section
  - Recent moves with annotations
- [x] `EvaluationGraph` component for evaluation timeline
  - Per-player evaluation lines over time
  - Click-to-jump to specific moves
  - Current move indicator
- [x] `MoveAnalysisPanel` for per-move insights
  - Move quality assessment (excellent/good/neutral/inaccuracy/mistake/blunder)
  - Evaluation change tracking
  - Think time and engine depth display
- [x] `TeachingOverlay` component for rules/FAQ scenarios
  - Contextual help for all game mechanics
  - Victory condition explanations
  - `useTeachingOverlay` hook for state management
  - `TeachingTopicButtons` quick-access component

### Wave 8.5 – Mobile & Responsive

- [ ] Responsive board rendering for mobile devices
- [ ] Touch-optimized controls for stack selection and movement
- [ ] Simplified mobile HUD layout

### Wave 8.6 – Accessibility

- [ ] Keyboard navigation for all game actions
- [ ] Screen reader support for game state announcements
- [ ] High-contrast mode option
- [ ] Colorblind-friendly player color palette

---

## Wave 9 – AI Strength & Optimization (P1 - PLANNED)

> **Goal:** Provide challenging AI opponents across all skill levels, from beginner to expert.
>
> **Status:** 📋 PLANNED
> **Spec:** [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §Wave 9

### Wave 9.1 – Production AI Ladder

- [ ] Wire MinimaxAI for medium-high difficulty levels (currently falling back to HeuristicAI)
- [ ] Expose MCTS implementation in production behind AIProfile
- [ ] Complete service-backing for remaining PlayerChoices:
  - [x] line_reward_option ✅
  - [x] ring_elimination ✅
  - [x] region_order ✅
  - [ ] line_order
  - [ ] capture_direction

### Wave 9.2 – Heuristic Weight Optimization

- [ ] Complete weight sensitivity analysis on square8, square19, hex
- [ ] Classify weights by signal strength:
  - Strong positive (>55% win rate) → Keep
  - Strong negative (<45% win rate) → Invert sign
  - Noise band (45-55%) → Prune or zero-initialize
- [ ] Run CMA-ES optimization on pruned weight set
- [ ] Validate optimized weights via tournament against baseline
- [ ] Create board-type specific profiles if results differ significantly

### Wave 9.3 – RNG Determinism

- [ ] Replace global `random` with per-game seeded RNG in Python AI
- [ ] Update ZobristHash to use stable, seeded RNG
- [ ] Pass RNG seeds from TS backend to Python service in /ai/move requests

### Wave 9.4 – Search Enhancements (Future)

- [ ] Move ordering heuristics for better alpha-beta pruning
- [ ] Transposition table for position caching
- [ ] Iterative deepening with time limits
- [ ] Opening book generation from strong AI self-play

### Wave 9.5 – AI Observability

- [ ] Per-difficulty latency tracking in Grafana
- [ ] AI quality metrics (win rate vs random, move consistency)
- [ ] Fallback rate monitoring by endpoint

---

## Wave 10 – Game Records & Training Data (P2 - PLANNED)

> **Goal:** Comprehensive game storage, notation, and replay system for analysis, training, and competitive features.
>
> **Status:** 📋 PLANNED
> **Spec:** [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §Wave 10

### Wave 10.1 – Game Record Types

- [ ] Python `GameRecord` types in `ai-service/app/models/game_record.py`
- [ ] TypeScript `GameRecord` types in `src/shared/types/gameRecord.ts`
- [ ] JSONL export format for training data pipelines
- [ ] Algebraic notation (RRN) generator and parser
- [ ] Coordinate conversion utilities for all board types

### Wave 10.2 – Database Integration

- [ ] Add `games` and `moves` tables to Prisma schema
- [ ] Create `GameRecordRepository` for CRUD operations
- [ ] Wire game storage into online game completion
- [ ] Wire game storage into self-play scripts (CMA-ES, soak tests)

### Wave 10.3 – Self-Play Recording (Track 11)

- [ ] Default-enabled game recording in `run_cmaes_optimization.py`
  - Per-run DB at `{output_dir}/games.db`
  - Rich metadata: source, generation, candidate, board_type, num_players
- [ ] State pool export utility (`scripts/export_state_pool.py`)
- [ ] Database merge utility (`scripts/merge_game_dbs.py`)
- [ ] Environment variables: `RINGRIFT_RECORD_SELFPLAY_GAMES`, `RINGRIFT_SELFPLAY_DB_PATH`

### Wave 10.4 – Replay System

- [ ] `reconstructStateAtMove(gameRecord, moveIndex)` in shared engine
- [ ] Checkpoint caching for efficient backward navigation
- [ ] `ReplayControls` UI component with play/pause/step/seek
- [ ] `MoveList` component with move annotations
- [ ] Sandbox integration for replay viewing

### Wave 10.5 – Self-Play Browser UI

- [x] API endpoints in `src/server/routes/selfplay.ts`
- [x] `SelfPlayGameService` for database access (read-only SQLite, with empty
      `games` tables filtered out so 0-game databases are not shown in the
      sandbox self-play browser dropdown).
- [x] `SelfPlayBrowser` component for game discovery
      (`src/client/components/SelfPlayBrowser.tsx`), wired into
      `SandboxGameHost` via the "Browse Self-Play Games" panel.
- [x] Filter by board type, player count, outcome, source (implemented via
      query params on `/api/selfplay/games` and client-side dropdown filters).
- [x] Fork games from replay position into sandbox by loading a selected
      self-play game as a `LoadableScenario` and initializing the local
      `ClientSandboxEngine` via `handleLoadScenario` in
      `SandboxGameHost.tsx`.

---

## Phase 3 – Multiplayer Polish (P1)

### P1.1 – WebSocket Lifecycle & Reconnection

- [x] Tighten and test WebSocket lifecycle around:
  - Reconnects and late joins in
    [`WebSocketServer`](src/server/websocket/server.ts) and
    [`WebSocketInteractionHandler`](src/server/game/WebSocketInteractionHandler.ts), including
    bounded reconnection windows (`pendingReconnections`), connection diagnostics
    (`playerConnectionStates`), and abandonment handling in
    `handleReconnectionTimeout`.
  - Consistent `game_over` handling and clearing of any pending choices on both
    server (GameSession / PlayerInteractionManager) and client
    ([`GameContext`](src/client/contexts/GameContext.tsx) clears
    `pendingChoice`/`choiceDeadline` on `game_over`).
  - Spectator join/leave flows, ensuring spectators are always read-only and do
    not receive reconnection windows (see strict spectator guards in
    `player_move` / `player_move_by_id` / `player_choice_response` handlers and
    the spectator branch in `WebSocketServer.handleDisconnect`).
- [x] Add focused Jest unit/integration/E2E tests for reconnection and lifecycle
      aspects and cross-link them from docs:
  - `tests/unit/GameConnection.reconnection.test.ts` – client
    `SocketGameConnection` status transitions and reconnect attempts.
  - `tests/unit/GameSession.reconnectFlow.test.ts`,
    `tests/unit/GameSession.reconnectDuringDecision.test.ts` – server-side
    reconnection windows, preservation of GameSession, and reconnect during a
    pending decision.
  - `tests/integration/GameReconnection.test.ts` – WebSocket reconnection window
    semantics (`player_disconnected` / `player_reconnected`) at the API edge.
  - `tests/e2e/reconnection.simulation.test.ts` – network partition and
    reconnection-window expiry (rated vs unrated abandonment), plus HUD-level
    reconnect UX.
  - See also `KNOWN_ISSUES.md` P1.2 “WebSocket Game Loop” and the reconnection
    section of `docs/CANONICAL_ENGINE_API.md` for the current lifecycle
    summary and remaining UX/documentation work.
- [ ] Remaining polish: multiplayer UX and documentation around connection
      lifecycle (multi-tab/cross-device reconnect messaging, richer HUD copy
      for “reconnecting” vs “abandoned”, and expanded end-to-end examples in
      `docs/CANONICAL_ENGINE_API.md` and lobby docs).

### P1.2 – Game HUD & Game Host UX

- [x] Enhance [`GameHUD`](src/client/components/GameHUD.tsx) to show:
  - Current player and phase.
  - Per-player ring counts (in hand / on board / eliminated).
  - Territory spaces per player.
  - Basic timer readout derived from `timeControl` and `timeRemaining`.
- [x] Ensure the same HUD behaviour is shared between backend games
      (`/game/:gameId` and `/spectate/:gameId` via
      [`BackendGameHost`](src/client/pages/BackendGameHost.tsx)) and
      sandbox games (`/sandbox` via
      [`SandboxGameHost`](src/client/pages/SandboxGameHost.tsx)) by
      routing both hosts through the shared `toHUDViewModel` adapter and
      the unified `GameHUD` view-model surface.
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
  - [x] [`GameHUD.tsx`](src/client/components/GameHUD.tsx) – verify ring/territory stats, timers, spectator badge, connection status.
  - [x] [`GameEventLog.tsx`](src/client/components/GameEventLog.tsx) – verify rendering of moves, system events, and victory messages from the view model.
  - [x] [`GameHistoryPanel.tsx`](src/client/components/GameHistoryPanel.tsx) – history list and selection behaviour.
- [ ] Add light-weight tests for supporting components:
  - [x] [`AIDebugView.tsx`](src/client/components/AIDebugView.tsx).
  - [x] [`LoadingSpinner.tsx`](src/client/components/LoadingSpinner.tsx) and small UI primitives under `src/client/components/ui/`.
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
- [x] Add tests around `AIServiceClient` to cover latency, timeouts, and
      fallback usage in more detail.

### P2.2 – AI Telemetry

- [x] Add lightweight logging/metrics around AI calls in
      [`AIServiceClient`](src/server/services/AIServiceClient.ts) and
      [`AIInteractionHandler`](src/server/game/ai/AIInteractionHandler.ts),
      capturing:
  - Request type
  - Duration
  - Success vs failure vs fallback
- [x] Surface basic metrics in the existing Prometheus/Grafana stack
      defined in [`docker-compose.yml`](docker-compose.yml).

### P2.3 – AI Wiring & Determinism (New)

- [ ] **Wire up MinimaxAI:** Update `_create_ai_instance` in `ai-service/app/main.py` to instantiate `MinimaxAI` for mid-high difficulties instead of falling back to `HeuristicAI`.
- [ ] **Fix RNG Determinism:**
  - [ ] Replace global `random` usage in Python AI with per-game seeded RNG instances.
  - [ ] Update `ZobristHash` to use a stable, seeded RNG instead of global `random.seed(42)`.
  - [ ] Pass RNG seeds from TS backend to Python service in `/ai/move` requests.

### P2.4 – AI Analysis Mode & Evaluation Panel

- [x] **Evaluation API in Python AI service:**
  - [x] Add a lightweight `/ai/evaluate_position` endpoint in `ai-service/app/main.py` that accepts a serialized `GameState` and returns per-player evaluation from a strong engine (e.g. Descent+NN or Minimax) as: - Estimated total evaluation per player (win/loss margin), and - Optional breakdown into territory vs eliminated-ring advantage.
  - [ ] Reuse existing `RingRiftEnv`/search code paths (from `evaluate_ai_models.py`) with bounded think time and deterministic seeds.
- [x] **Backend evaluation client & WebSocket event:**
  - [x] Add a small evaluation client on the Node side (e.g. extending `AIServiceClient` or a dedicated `PositionEvaluationClient`) that calls `/ai/evaluate_position` with strict timeouts and concurrency caps.
  - [x] Emit evaluation results asynchronously (e.g. `position_evaluation` events) from `GameSession`/`WebSocketServer` after moves, keyed by `gameId` + `moveNumber`.
- [ ] **Persistence & history:**
  - [ ] Store evaluation snapshots per move (Redis or DB) so reconnects and `/api/games/:gameId/history` can surface evaluation history alongside move history.
- [x] **Frontend EvaluationPanel:**
  - [ ] Extend `gameViewModels` / HUD view models with an `evaluationHistory` structure and current per-player evaluation.
  - [x] Add an `EvaluationPanel` React component to `BackendGameHost` that: - Shows a small time-series or bar chart over moves, and - Highlights current evaluation with color-coded per-player advantage.
  - [ ] Gate the panel behind an “Analysis mode” / spectator-only toggle so it does not affect normal rated play performance.

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
- [x] Expose rules/FAQ scenarios directly in the sandbox UI:
  - [x] Provide a Scenario Picker in `/sandbox` (via `ScenarioPickerModal` and `SandboxGameHost`)
        backed by curated RulesMatrix/FAQ scenarios and saved game states.
  - [x] Allow loading and resetting named scenarios into the sandbox for visual
        inspection and step-through play.
- [ ] Add visual debugging aids to the sandbox view:
  - [ ] Overlays for detected lines and their rewards.
  - [ ] Territory region highlighting and disconnection visualization.
  - [ ] Chain capture path visualization (e.g. arrows or highlighted
        segments).

### Track 4 – Incremental AI Improvements & Observability (P1–P2)

- [x] Add lightweight metrics/logging around AI calls:
  - [x] In `AIServiceClient` and `AIInteractionHandler`, log request type,
        latency, success/failure, and fallback usage.
  - [x] Optionally expose these metrics via existing or new monitoring
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
  - [x] Add scenario picker into the sandbox UI and basic visual helpers
        for lines/territory/chain captures (ScenarioPickerModal with curated
        RulesMatrix/FAQ-tagged scenarios and in-sandbox “Reset Scenario”).
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

---

## Phase 4 – AI Training & Optimization (P2)

This phase focuses on systematic AI improvement through data-driven heuristic
weight tuning and analysis. Spec: [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md).

### Track 7 – Heuristic Weight Optimization

- [ ] **Weight Sensitivity Analysis** (IN PROGRESS):
  - [x] Create axis-aligned sensitivity test script (`scripts/run_weight_sensitivity_test.py`)
  - [x] Add true random baseline option (`--use-true-random`)
  - [x] Implement random tie-breaking in HeuristicAI for non-deterministic play
  - [x] Create weight classification analysis script (`scripts/analyze_weight_sensitivity.py`)
  - [ ] Run sensitivity tests on all board types (square8, square19, hex)
  - [ ] Analyze results to identify high-signal vs noise weights
  - [ ] Classify weights by signal strength:
    - **Strong positive signal (>55% win rate):** Keep with positive sign
    - **Strong negative signal (<45% win rate):** Keep but **invert sign** for CMA-ES (these indicate features that hurt when over-weighted but help when negated)
    - **Noise band (45-55% win rate):** Candidate for pruning or zero-initialization
  - [ ] Normalize magnitudes of remaining high-signal weights

  **Analysis Workflow:**

  ```bash
  # Step 1: Run sensitivity tests (takes time)
  cd ai-service
  python scripts/run_weight_sensitivity_test.py --board square8 --games-per-weight 10 \
      --output logs/axis_aligned/sensitivity_results_square8.json

  # Step 2: Analyze results and generate CMA-ES seed weights
  python scripts/analyze_weight_sensitivity.py \
      --input logs/axis_aligned/sensitivity_results_square8.json \
      --output logs/axis_aligned/cmaes_seed_weights.json

  # Step 3: Run CMA-ES with classified seed weights
  python scripts/run_cmaes_optimization.py \
      --baseline logs/axis_aligned/cmaes_seed_weights.json \
      --generations 50 --population-size 20
  ```

- [ ] **Evaluation Pool Quality**:
  - [ ] Position diversity analysis (move number distribution, ring counts, territory)
  - [ ] Game length and outcome statistics (ring elimination vs territory vs stalemate)
  - [ ] Move branching factor analysis by game phase
  - [ ] Identify and fill coverage gaps in eval pools

- [ ] **Critical Position Mining**:
  - [x] Create critical position mining script (`scripts/mine_critical_positions.py`)
  - [ ] Mine positions near victory threshold (1-2 rings from win)
  - [ ] Mine positions from last N moves of games
  - [ ] Build training dataset from critical positions with outcomes

- [ ] **Weight Optimization**:
  - [ ] Run CMA-ES optimization on pruned weight set
  - [ ] Validate optimized weights via tournament against baseline
  - [ ] Create board-type specific profiles if results differ significantly
  - [ ] Integrate winning profile into production heuristic ladder

### Track 8 – AI Analysis Tools

- [ ] **Opening Book Generation** (post-optimization):
  - [ ] Generate opening book from strong AI self-play
  - [ ] Store opening tree with win rates per move
  - [ ] Integrate opening book lookup into HeuristicAI

- [ ] **Search Improvements** (future):
  - [ ] Move ordering heuristics for better pruning
  - [ ] Transposition table for position caching
  - [ ] Iterative deepening with time limits

---

## Phase 5 – Game Record System (P2)

Comprehensive game storage, notation, and replay system.
Spec: [`ai-service/docs/GAME_RECORD_SPEC.md`](ai-service/docs/GAME_RECORD_SPEC.md).

### Track 9 – Game Storage Infrastructure

- [x] **Specification**:
  - [x] Define game record JSON schema (metadata, moves, states, outcome)
  - [x] Define RingRift Notation (RRN) algebraic format
  - [x] Define coordinate systems for all board types
  - [x] Define database schema for game storage

- [ ] **Core Implementation**:
  - [ ] Create Python `GameRecord` types (`ai-service/app/models/game_record.py`)
  - [ ] Create TypeScript `GameRecord` types (`src/shared/types/gameRecord.ts`)
  - [ ] Implement JSONL export format for training data
  - [ ] Implement algebraic notation generator/parser
  - [ ] Implement coordinate conversion utilities

- [ ] **Database Integration**:
  - [ ] Add `games` and `moves` tables to database schema
  - [ ] Create `GameRecordRepository` for CRUD operations
  - [ ] Wire game storage into self-play scripts
  - [ ] Wire game storage into online game completion
  - [ ] Migration for existing games (if any)

### Track 10 – Replay System

- [ ] **State Reconstruction**:
  - [ ] Implement `reconstructStateAtMove(gameRecord, moveIndex)`
  - [ ] Add checkpoint caching for efficient backward navigation
  - [ ] Validate state reconstruction matches original game

- [ ] **Sandbox Integration**:
  - [ ] Create `GameReplayController` interface
  - [ ] Implement forward/backward navigation (goToMove, nextMove, previousMove)
  - [ ] Add game loading from database/file
  - [ ] Create `ReplayControls` UI component
  - [ ] Create `MoveList` display component
  - [ ] Integrate replay into sandbox page

- [ ] **Analysis Features** (future):
  - [ ] Position search/filter by criteria
  - [ ] Move statistics aggregation
  - [ ] Critical position tagging
  - [ ] Export to portable notation format

### Track 11 – Unified Self-Play Game Recording (NEW)

> **Goal:** Record ALL self-play games to SQLite for replay, training data, and CMA-ES pools.
>
> **Plan:** See `.claude/plans/memoized-cuddling-abelson.md` for full implementation details.

- [ ] **Phase 1 – CMA-ES Recording (HIGH)**:
  - [ ] Add default-enabled game recording to `run_cmaes_optimization.py`
  - [ ] Use per-run DB at `{output_dir}/games.db`
  - [ ] Add `--no-record` flag to disable
  - [ ] Record rich metadata (source, generation, candidate, board_type, num_players)

- [ ] **Phase 2 – State Pool Export**:
  - [ ] Create `scripts/export_state_pool.py` utility
  - [ ] Sample mid-game states at configurable intervals
  - [ ] Export to JSONL format for CMA-ES eval pools
  - [ ] Create `scripts/merge_game_dbs.py` for hybrid storage

- [ ] **Phase 3 – Iterative Pipeline Integration**:
  - [ ] Update `run_iterative_cmaes.py` to pass recording config
  - [ ] Per-iteration DB + optional export to shared DB

- [ ] **Phase 4 – Environment Control**:
  - [ ] Add `RINGRIFT_RECORD_SELFPLAY_GAMES` env var
  - [ ] Add `RINGRIFT_SELFPLAY_DB_PATH` env var

- [ ] **Phase 5 – Sandbox UI Integration (HIGH)**:
  - [x] Create `src/server/routes/selfplay.ts` API endpoints
  - [x] Create `src/server/services/SelfPlayGameService.ts`
  - [x] Create `src/client/components/SelfPlayBrowser.tsx` game browser
  - [ ] Create `src/client/components/ReplayControls.tsx` playback controls
  - [x] Add a dedicated "Self-Play Games" section to the sandbox configuration panel (via `SandboxGameHost.tsx`) that opens the Self-Play Browser and bridges into ReplayPanel when possible.
  - [x] Add maintenance script `scripts/cleanup-empty-selfplay-dbs.ts` to
        remove 0-game SQLite databases from `data/games` and
        `ai-service/**/games` so the sandbox self-play browser only surfaces
        databases with at least one recorded game.
