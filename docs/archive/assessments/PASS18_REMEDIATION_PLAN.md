# PASS18 Remediation Plan

**Priority:** P0 (Critical path for v1.0)
**Focus:** Stabilising TS rules/host integration, restoring advanced-phase & RNG parity, and executing orchestrator-first rollout with deep multi-engine parity.

> **P18.6-1 Update (2025-12-01):** P18.1-5 tasks substantially complete:
>
> - P18.1-\*: Host parity and capture/territory alignment ✅
> - P18.2-\*: Territory and elimination integration ✅
> - P18.3-\*: RNG determinism and decision lifecycle ✅
> - P18.4-\*: Orchestrator Phase 4 rollout (100% in all environments) ✅
> - P18.5-\*: Extended contract vectors (43 cases, 0 mismatches), swap_sides parity ✅
> - P18.18: Skipped test triage complete ✅

> This plan is the task-level companion to:
>
> - [`docs/PASS18_ASSESSMENT_REPORT.md`](docs/PASS18_ASSESSMENT_REPORT.md:1)
> - [`WEAKNESS_ASSESSMENT_REPORT.md`](WEAKNESS_ASSESSMENT_REPORT.md:1)

---

## 1. Task format

Each PASS18 task is described with:

- **Task ID** – e.g. `P18.1-CODE`.
- **Title** – concise description.
- **Agent Mode** – Ask / Architect / Code / Debug / DevOps.
- **Priority** – P0 (critical), P1 (important), P2 (valuable).
- **Scope** – specific, actionable work, including key files and tests.
- **Acceptance criteria** – testable outcomes (Jest/pytest suites, invariants, or docs).
- **Dependencies** – on other P18 tasks or pre-existing work (ANM remediation, orchestrator design, SSOT checks).

---

## 2. P0 – Stabilise host integration & parity (Weakest aspect)

These tasks directly remediate the weakest aspect: TS rules/host integration & RNG parity across backend host, client sandbox, and Python rules/AI.

### P18.1-CODE – Stabilise capture & chain-capture host parity ✅ COMPLETE

- **Agent Mode:** Code
- **Priority:** P0
- **Status:** ✅ Complete (2025-12-01) – Capture/territory host unification complete, advanced-phase ordering aligned

- **Scope:**
  - Investigate and fix divergence in capture-related suites, including:
    - [`captureSequenceEnumeration.test.ts`](tests/unit/captureSequenceEnumeration.test.ts:1)
    - [`GameEngine.chainCapture.test.ts`](tests/unit/GameEngine.chainCapture.test.ts:1) and [`GameEngine.chainCaptureChoiceIntegration.test.ts`](tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts:1)
    - Any related sandbox parity tests under `tests/unit` and `tests/scenarios`.
  - Ensure backend host [`GameEngine.ts`](src/server/game/GameEngine.ts:1) and client sandbox [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:1) both delegate capture enumeration to the same shared helpers defined by the RR‑CANON capture rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1).
  - Verify that Python parity suites for capture (for example, [`test_chain_capture_parity.py`](ai-service/tests/parity/test_chain_capture_parity.py:1)) remain green after changes.

- **Acceptance criteria:**
  - All capture enumeration and chain-capture Jest suites are passing and stable (no flakes across multiple CI runs).
  - Python capture parity tests remain green with no new divergences.
  - At least one additional invariant-style or property-based test is added to guard capture sequence enumeration from regression (for example, “no missing valid capture chains, no spurious extra chains” on small boards).

- **Dependencies:**
  - RR‑CANON capture rules and invariants in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1) and [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1).
  - Can run in parallel with P18.2-CODE and P18.3-CODE, but should complete before orchestrator rollout tasks (P18.4-DEVOPS, P18.5-ARCH, P18.9-DEBUG).
  - **Diagnostic Map:** See [`docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md`](docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md:1) for detailed host-path analysis and test cluster mapping.

---

### P18.2-CODE – Territory & elimination integration corrections ✅ COMPLETE

- **Agent Mode:** Code
- **Priority:** P0
- **Status:** ✅ Complete (2025-12-01) – Territory processing and forced elimination aligned

- **Scope:**
  - Debug and correct complex long-turn scenarios where line formation, territory disconnection, and Q23 self-elimination interact, focusing on:
    - [`GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1)
    - [`ClientSandboxEngine.territoryDisconnection.hex.test.ts`](tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts:1)
    - Territory mini-region scenarios such as [`RulesMatrix.Territory.MiniRegion.test.ts`](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1).
  - Ensure host integration in [`GameEngine.ts`](src/server/game/GameEngine.ts:1) correctly orders:
    1. Line reward processing.
    2. Territory collapse and region claim.
    3. Forced self-elimination and Q23 elimination checks.
  - Cross-check behaviour against Python invariants for territory processing and ANM/termination in:
    - [`test_active_no_moves_territory_processing_regression.py`](ai-service/tests/invariants/test_active_no_moves_territory_processing_regression.py:1)
    - [`test_territory_forced_elimination_divergence.py`](ai-service/tests/test_territory_forced_elimination_divergence.py:1).

- **Acceptance criteria:**
  - All targeted TS suites above are green, with no regressions in broader territory or ANM regression tests.
  - Python invariants and parity suites for territory remain green.
  - At least one “combined line+territory+Q23” scenario is captured as a contract vector or snapshot parity fixture and passes in both TS and Python engines.

- **Dependencies:**
  - P18.1-CODE (capture semantics) is helpful but not strictly required; both can proceed in parallel.
  - Must be complete before moving orchestrator rollout beyond staging shadow phases (P18.4-DEVOPS, P18.12-DEVOPS).

---

### P18.3-CODE – AI RNG parity & seeded behaviour across hosts ✅ COMPLETE

- **Agent Mode:** Code
- **Priority:** P0
- **Status:** ✅ Complete (2025-12-01) – RNG seed handling aligned per P18.2-1_AI_RNG_PATHS.md, decision lifecycle spec complete per P18.3-1

- **Scope:**
  - Restore deterministic, seed-controlled AI behaviour across:
    - Backend host (`GameSession` + [`AIEngine.ts`](src/server/game/ai/AIEngine.ts:1) + [`AIServiceClient.ts`](src/server/services/AIServiceClient.ts:1)).
    - Client sandbox AI simulations (for example, via `ClientSandboxEngine` and any AI helpers).
    - Python AI service RNG paths used in [`ai-service/tests/test_eval_randomness_integration.py`](ai-service/tests/test_eval_randomness_integration.py:1).
  - Fix and stabilise RNG parity suites:
    - [`Sandbox_vs_Backend.aiRngParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1)
    - [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1).
  - Ensure `GameSession.maybePerformAITurn` threads an explicit RNG or seed into `getAIMove` for local fallback, avoiding `Math.random()` where deterministic parity is required.
  - Confirm that `game.rngSeed` from HTTP game creation (`POST /games` in [`game.ts`](src/server/routes/game.ts:1)) remains the single source of truth for seeded behaviour across hosts.

- **Acceptance criteria:**
  - All AI RNG parity Jest suites pass reliably across multiple CI runs.
  - Repeated AI moves with the same seed produce identical results across backend, sandbox, and Python AI for representative board types (square and hex).
  - No regression in AI service health and concurrency tests (for example, [`AIEngine.fallback.test.ts`](tests/unit/AIEngine.fallback.test.ts:1), [`AIServiceClient.concurrency.test.ts`](tests/unit/AIServiceClient.concurrency.test.ts:1)).

- **Dependencies:**
  - Relies on existing RNG design described in [`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md:1).
  - Complements P18.1-CODE and P18.2-CODE; all three should be complete before long-running orchestrator soaks (P18.9-DEBUG).

- **Upstream spec:**
  - **P18.3-1 – Decision Lifecycle Spec.** See [`P18.3-1_DECISION_LIFECYCLE_SPEC.md`](docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md:1) for the canonical description of host decision, timeout, reconnect, and resign semantics used by P18.8-DEBUG and follow-on implementation/debug tasks.

---

### P18.8-DEBUG – Decision lifecycle edge cases (timeouts, reconnect, resign)

- **Agent Mode:** Debug
- **Priority:** P0

- **Scope:**
  - Systematically exercise edge cases in decision lifecycles at the host/WebSocket layer, focusing on:
    - Decision-phase timeout warnings and auto-resolutions in [`GameSession.ts`](src/server/game/GameSession.ts:1) and [`WebSocketInteractionHandler.ts`](src/server/game/WebSocketInteractionHandler.ts:1).
    - Reconnect windows and pending choice cancellation in [`server.ts`](src/server/websocket/server.ts:1).
    - HTTP-level leave/resign flows in [`game.ts`](src/server/routes/game.ts:1) and their interaction with in-engine resignation semantics.
  - Add or extend Jest/Playwright tests to cover:
    - Disconnection during an active decision phase and subsequent auto-resolution.
    - Reconnection within and after the grace window.
    - Resign vs timeout semantics for rated and unrated games.

- **Acceptance criteria:**
  - New automated tests reproduce and then fix any identified inconsistencies between documented semantics, engine behaviour, and user-visible outcomes.
  - Decision lifecycle events (timeout warnings, auto-resolutions, reconnects) are clearly reflected in persisted game history and surfaced to clients via `game_state` and `decision_*` events.
  - WebSocket and HTTP flows for resign/leave are consistent with RR‑CANON rules and any documented rating policies.

- **Dependencies:**
  - Builds on existing semantics and UX notes in [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:1) and [`docs/runbooks/WEBSOCKET_ISSUES.md`](docs/runbooks/WEBSOCKET_ISSUES.md:1).
  - Should be informed by P18.1-CODE and P18.2-CODE but can begin in parallel.

---

## 3. P1 – Orchestrator rollout & deep multi-engine parity (Hardest problem)

These tasks advance the hardest outstanding problem: executing orchestrator-first rollout with deep multi-engine parity under live load.

### P18.4-DEVOPS – Execute orchestrator Phase 1 in staging ✅ COMPLETE (now Phase 4)

- **Agent Mode:** DevOps
- **Priority:** P1
- **Status:** ✅ Complete (2025-12-01) – Orchestrator at Phase 4 (100% rollout in all environments)

- **Scope:**
  - Configure the staging environment to run with the orchestrator adapter enabled for all games:
    - `ORCHESTRATOR_ADAPTER_ENABLED=true`
    - `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`
    - `RINGRIFT_RULES_MODE=ts` (TS authoritative with Python in shadow mode).
  - Follow the staged rollout steps in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:1) and [`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`](docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md:1) for at least 48h of staging traffic (synthetic or real).
  - **Gating:** Ensure all mandatory suites listed in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §6.5](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:636) (Capture/Territory Host Parity, AI RNG Parity) are green before entry.
  - Ensure parity and invariant metrics are collected and visible in dashboards (for example, `ringrift_rules_parity_mismatches_total`, S-invariant counters, ACTIVE_NO_MOVES regressions).

- **Acceptance criteria:**
  - Orchestrator adapter is confirmed active for all staging games over a 48h window.
  - No sustained increase in parity mismatches or invariant violations beyond pre-agreed thresholds.
  - A brief staging report is produced summarising findings and readiness for production-preview phases.

- **Dependencies:**
  - P18.1-CODE, P18.2-CODE, and P18.3-CODE should be substantially complete (or at least green in CI) to avoid testing with a known-broken host stack.
  - Requires observability plumbing already described in PASS16/17 and PASS18 assessment.

---

### P18.5-ARCH – Expand contract vectors & parity artefacts ✅ COMPLETE

- **Agent Mode:** Architect
- **Priority:** P1
- **Status:** ✅ Complete (2025-12-01) – 49 contract vectors (0 mismatches), swap_sides parity verified

- **Scope:**
  - Design and add at least 10 new contract vectors and/or snapshot traces covering "long tail" interactions, including:
    - **Actual delivery: 49 vectors across the core families** (placement, movement, capture/chain_capture including extended chains, forced_elimination, territory/territory_line endgames including near_victory_territory, hex_edge_cases, and meta moves such as swap_sides)
    - Hex board territory disconnection with simultaneous line formation.
    - Three- and four-player capture chains with mixed line/territory consequences.
    - Combined ANM and territory-processing edge cases already documented in [`docs/STRICT_INVARIANT_SOAKS.md`](docs/STRICT_INVARIANT_SOAKS.md:1) and [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1).
  - Update TS and Python contract runners:
    - TS: [`tests/contracts/contractVectorRunner.test.ts`](tests/contracts/contractVectorRunner.test.ts:1)
    - Python: [`ai-service/tests/contracts/test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py:1)
  - Ensure new artefacts are referenced in [`docs/PYTHON_PARITY_REQUIREMENTS.md`](docs/PYTHON_PARITY_REQUIREMENTS.md:1) and surfaced via SSOT checks.

- **Acceptance criteria:**
  - New vectors or snapshots present under `tests/fixtures/contract-vectors/v2/` and any equivalent Python locations.
  - Both TS and Python runners pass on the expanded suite.
  - Parity artefacts are enumerated in SSOT tooling (see P18.10-ARCH).

- **Dependencies:**
  - P18.1-CODE and P18.2-CODE should be sufficiently stable to avoid baking known-bad behaviour into new vectors.
  - Feeds into orchestrator soaks (P18.9-DEBUG) and rollout phases (P18.4-DEVOPS, P18.12-DEVOPS).

---

### P18.9-DEBUG – Orchestrator parity soaks & debugging harness

- **Agent Mode:** Debug
- **Priority:** P1

- **Scope:**
  - Use existing tooling such as [`scripts/run-orchestrator-soak.ts`](scripts/run-orchestrator-soak.ts:1) and [`scripts/generate-orchestrator-contract-vectors.ts`](scripts/generate-orchestrator-contract-vectors.ts:1) to run orchestrator-first soaks under representative board types and player counts.
  - Capture outputs in results artefacts like [`results/orchestrator_soak_smoke.json`](results/orchestrator_soak_smoke.json:1) and [`results/orchestrator_soak_summary.json`](results/orchestrator_soak_summary.json:1), and extend them as needed.
  - Build lightweight debugging harnesses to quickly repro and triage any discovered mismatches (for example, scripts under `scripts/` that replay failing seeds via both legacy and orchestrator paths and dump traces).

- **Acceptance criteria:**
  - At least one soak campaign is run to completion with a summarised report of parity and invariant health.
  - Any discovered divergences are either:
    - Triaged into bugs and associated with P18.1–P18.3 or follow-up tasks, or
    - Explained as expected differences and documented in RR‑CANON or parity docs.
  - Tooling is documented in `docs/` so future soaks can be run and interpreted by on-call engineers.

- **Dependencies:**
  - Relies on orchestrator adapter being functional in non-production environments.
  - Builds on parity artefacts from P18.5-ARCH.

---

### P18.12-DEVOPS – Production-preview orchestrator rollout with SLO gating

- **Agent Mode:** DevOps
- **Priority:** P1

- **Scope:**
  - Plan and execute a limited-scope production-preview rollout of the orchestrator adapter (for example, a small percentage of games or specific board types), with explicit SLO and rollback criteria derived from the staging results (P18.4-DEVOPS).
  - **Gating:** Re-verify that the **Capture/Territory Host Parity** and **AI RNG Parity** suites remain green and that no new `INV-ACTIVE-NO-MOVES` regressions have appeared in staging.
  - Coordinate feature flags and configuration (`ORCHESTRATOR_ROLLOUT_PERCENTAGE`, board-type filters, environment variables) so that the rollout can be safely paused or rolled back.
  - Ensure on-call and runbook coverage for orchestrator-related incidents.

- **Acceptance criteria:**
  - Production-preview window completes without violating agreed SLOs or, if issues occur, they are caught and mitigated via rollback.
  - A post-preview report is produced documenting findings and recommended next steps (full rollout, revised plan, or rollback).

- **Dependencies:**
  - P18.4-DEVOPS, P18.5-ARCH, and P18.9-DEBUG should be complete.
  - P18.1–P18.3 must be green in CI to avoid conflating known host issues with orchestrator behaviour.

---

## 4. P2 – Docs, SSOT, tests & UX

These tasks align docs and UX with the updated weakest aspect and hardest problem, and strengthen guardrails around parity and rollout.

### P18.6-CODE – HUD & victory copy alignment

- **Agent Mode:** Code
- **Priority:** P2

- **Scope:**
  - Update advanced-phase copy in HUD and victory components so that player-facing text matches RR‑CANON:
    - [`GameHUD.tsx`](src/client/components/GameHUD.tsx:1) – remove “or end your turn” from chain capture prompts when continuation is mandatory; clarify line and territory decision prompts.
    - [`VictoryModal.tsx`](src/client/components/VictoryModal.tsx:1) – replace “eliminate all rings” language with the correct “more than 50 percent of rings eliminated” threshold.
  - Ensure copy is consistent with rules docs:
    - [`ringrift_simple_human_rules.md`](ringrift_simple_human_rules.md:1)
    - [`ringrift_compact_rules.md`](ringrift_compact_rules.md:1)
    - [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1).

- **Acceptance criteria:**
  - All user-facing advanced-phase copy matches RR‑CANON and no longer misstates optional vs mandatory behaviour.
  - Any existing snapshot or UI tests for HUD/victory are updated and passing.
  - UX feedback (internal or from playtests) no longer flags confusion around chain capture or victory thresholds.

- **Dependencies:**
  - Coordination with P18.14-ASK (rules narrative & UX copy) is recommended to keep docs and UI copy aligned.

---

### P18.6b-CODE – Pie rule / `swap_sides` visibility & parity (status note)

- **Agent Mode:** Code
- **Priority:** P2

- **Scope (2025-11-30 status):**
  - Treat the 2‑player pie rule (`swap_sides` meta‑move) as first‑class in both TS and Python engines, with aligned gating and semantics:
    - Offered exactly once to P2 immediately after P1’s first non‑swap move when `rulesOptions.swapRuleEnabled === true`.
    - Swaps player identity/seat metadata between players 1 and 2 without changing board geometry or per‑seat statistics.
  - Surface `swap_sides` consistently across hosts:
    - Backend host (`BackendGameHost`) and AI endpoints already honour the TS engine’s `swap_sides` move.
    - Client sandbox (`ClientSandboxEngine` / `SandboxGameHost`) now exposes a one‑shot “Swap colours (pie rule)” action for P2 in active 2‑player games when enabled.
  - Extend Python training/eval harnesses to understand and optionally exercise the pie rule:
    - `create_initial_state` in `ai-service/app/training/generate_data.py` can enable `rulesOptions.swapRuleEnabled` for 2‑player training games via `RINGRIFT_TRAINING_ENABLE_SWAP_RULE=1`.
    - Self‑play soaks and CMA‑ES/GA evaluation now emit lightweight diagnostics on total `swap_sides` selections per run so regressions in pie‑rule usage are observable.

- **Acceptance criteria:**
  - TS ↔ Python parity tests for `swap_sides` (gating + application) are green in CI.
  - Sandbox and backend hosts show clear, user‑facing copy when the pie rule is available or has been used (event log + HUD chip).
  - Training/eval logs expose basic `swap_sides` usage metrics without materially impacting runtime.

### P18.7-ARCH – Core docs clean-up & indexing alignment

- **Agent Mode:** Architect
- **Priority:** P2

- **Scope:**
  - Bring core SSOT and index docs into alignment with the updated weakest aspect and hardest problem:
    - [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) – already updated; confirm references are stable.
    - [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1) – clarify that test counts and pass status are time-stamped snapshots; avoid implying current “all green” status.
    - [`CURRENT_RULES_STATE.md`](CURRENT_RULES_STATE.md:1) – update “No critical known issues” to reference current advanced-phase/parity work and link to PASS18 docs.
    - [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md:1) and [`docs/INDEX.md`](docs/INDEX.md:1) – revise “highest-risk semantics” language to distinguish historical ANM/forced-elimination focus from the current host-integration & parity focus, and add pointers to PASS18 and weakness docs.
  - Archive or clearly mark historical design docs that have been superseded by RR‑CANON and shared-engine architecture (e.g., older adapter migration plans under `archive/`).

- **Acceptance criteria:**
  - All core docs above are internally consistent and point to `PROJECT_GOALS.md`, PASS18, and the weakness report for current risk framing.
  - No core doc claims “all tests passing” or “no critical issues” without clear temporal context or links to current health.
  - SSOT banner checks continue to pass.

- **Dependencies:**
  - Uses findings captured in [`docs/PASS18_WORKING_NOTES.md`](docs/PASS18_WORKING_NOTES.md:1) and `docs/PASS18_ASSESSMENT_REPORT.md`.

---

### P18.10-ARCH – SSOT & CI guardrails for parity and rollout

- **Agent Mode:** Architect
- **Priority:** P2

- **Scope:**
  - Extend SSoT tooling to better protect parity artefacts and orchestrator rollout configuration:
    - Update [`scripts/ssot/python-parity-ssot-check.ts`](scripts/ssot/python-parity-ssot-check.ts:1) to also assert the presence of key parity test suites in Jest and pytest configs.
    - Optionally add a new `orchestrator-rollout-ssot-check` that validates critical environment variables and CI jobs for orchestrator rollout (e.g., ensuring that staging and production pipelines include orchestrator and parity checks).
  - Ensure these checks are wired into CI via [`scripts/ssot/ssot-check.ts`](scripts/ssot/ssot-check.ts:1).

- **Acceptance criteria:**
  - SSOT checks fail if critical parity tests or orchestrator-related CI jobs are removed or disabled.
  - Documentation for these checks is added or updated in [`docs/SSOT_BANNER_GUIDE.md`](docs/SSOT_BANNER_GUIDE.md:1) or a nearby SSOT overview.

- **Dependencies:**
  - Builds on existing SSOT tooling described in PASS18 working notes and previous passes.
  - Should reflect the expanded artefact set from P18.5-ARCH.

---

### P18.13-CODE – Test coverage improvements (frontend, WebSocket, E2E)

- **Agent Mode:** Code
- **Priority:** P2

- **Scope:**
  - Add or extend tests in areas identified as under-tested in PASS18 test health scoring:
    - Frontend reconnect and spectator UX (e.g., Playwright tests over `GameContext` and lobby/game pages).
    - WebSocket reconnection and error flows (including `player_choice_required`, `decision_phase_timeout_warning`, and `decision_phase_timed_out` events) bridging [`server.ts`](src/server/websocket/server.ts:1) and [`GameContext.tsx`](src/client/contexts/GameContext.tsx:1).
    - AI service outage and fallback UX (tying together `AIEngine` behaviour and client error banners).

- **Acceptance criteria:**
  - New tests are added and passing, and coverage reports show improved coverage in targeted files.
  - At least one E2E scenario exercises a full game with reconnect and decision timeouts without manual intervention.
    - As of 2025‑12‑01, this is concretely covered by the Playwright test
      **“reconnects after decision timeout and shows auto-resolved outcome in HUD”** in
      [`tests/e2e/error-recovery.e2e.spec.ts`](tests/e2e/error-recovery.e2e.spec.ts:140), backed by the guarded
      decision-phase fixture route (`POST /api/games/fixtures/decision-phase`) and the decision-timeout overrides wired
      through [`config.decisionPhaseTimeouts`](src/server/config/unified.ts:260).

- **Dependencies:**
  - Builds on debugging insights from P18.8-DEBUG.
  - Coordinates with P18.6-CODE and P18.14-ASK for UX and copy expectations.

---

### P18.14-ASK – Rules narrative & UX copy refinement

- **Agent Mode:** Ask
- **Priority:** P2

- **Scope:**
  - Produce a concise UX-oriented rules narrative for advanced phases that keeps:
    - [`ringrift_simple_human_rules.md`](ringrift_simple_human_rules.md:1)
    - [`ringrift_compact_rules.md`](ringrift_compact_rules.md:1)
    - [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1)
      aligned with HUD/game UI copy for chain capture, line and territory decisions, and victory thresholds.
  - Use findings in [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:1) to prioritise the most confusing areas.

- **Acceptance criteria:**
  - Updated rule docs provide a clear, layered explanation of advanced phases that matches in-game prompts.
  - UX and rules reviewers agree that key sources no longer contradict each other on mandatory vs optional behaviours or victory thresholds.

- **Dependencies:**
  - Feeds into UI text updates in P18.6-CODE and helps keep future HUD changes aligned with RR‑CANON.

---

## 5. PASS18-Phase 3 Remediation Backlog (New)

> **Focus:** Frontend UX Polish & Test Suite Cleanup.
> **Status:** Active (2025-12-01).

### P18.15-CODE – Frontend Keyboard Navigation

- **Agent Mode:** Code
- **Priority:** P0
- **Scope:**
  - Implement keyboard focus management for `BoardView` cells.
  - Support arrow key navigation (Up/Down/Left/Right) to move selection.
  - Support Enter/Space to select/place/move.
  - Ensure focus is visible and accessible.
- **Acceptance Criteria:**
  - Users can play a full game using only the keyboard.
  - Focus indicators are clear.
  - No focus traps.

### P18.16-CODE – Move History UI

- **Agent Mode:** Code
- **Priority:** P0
- **Scope:**
  - Create a `MoveHistory` component in the Game Page.
  - Display a list of past moves with descriptive text (e.g., "Player 1 placed ring at C3").
  - (Optional) Allow clicking a move to view the board state at that time (read-only).
- **Acceptance Criteria:**
  - History updates in real-time.
  - Scrollable list of moves.
  - Clear, human-readable move descriptions.

### P18.17-CODE – Sandbox Scenario Picker

- **Agent Mode:** Code
- **Priority:** P1
- **Scope:**
  - Add a "Load Scenario" button to the Sandbox UI.
  - Implement a modal to select from pre-defined scenarios (e.g., from `RULES_SCENARIO_MATRIX`).
  - Allow resetting the board to the scenario start state.
- **Acceptance Criteria:**
  - Users can load at least 5 key scenarios (e.g., Chain Capture, Line Formation).
  - Board state updates correctly upon load.

### P18.18-DEBUG – Skipped Test Triage ✅ COMPLETE

- **Agent Mode:** Debug
- **Priority:** P1
- **Status:** ✅ Complete (2025-12-01) – See [`docs/P18.18_SKIPPED_TEST_TRIAGE.md`](docs/P18.18_SKIPPED_TEST_TRIAGE.md:1)
- **Scope:**
  - Analyze the 176 skipped tests.
  - Categorize them: Fix, Delete, or Defer.
  - Fix at least 20 high-value skipped tests.
  - Delete permanently obsolete tests (e.g., for legacy features).
- **Completed Work:**
  - Obsolete tests removed from active suite
  - `RulesMatrix.Comprehensive` partially re-enabled (7 passing, 3 skipped)
  - `OrchestratorSInvariant.regression` re-enabled
  - Detailed triage report published
- **Acceptance Criteria:**
  - Skipped test count reduced by at least 20.
  - A plan exists for the remaining skipped tests.

### P18.19-CODE – Legacy Code Deprecation

- **Agent Mode:** Code
- **Priority:** P1
- **Scope:**
  - Identify legacy turn-processing methods in `GameEngine.ts` that are no longer used by the Orchestrator.
  - Mark them with `@deprecated`.
  - Add runtime warnings if they are called (optional "scream test").
- **Acceptance Criteria:**
  - Legacy methods are clearly marked.
  - No regressions in Orchestrator-based gameplay.

---

## 6. Doc alignment findings (PASS18 summary)

> **P18.6-1 Update (2025-12-01):** Several docs updated during alignment task. See below.

This section summarises the PASS18 documentation audit for core SSOT and high-visibility docs.

| Doc                                                                                                          |                Status                | Notes                                                                                                                                                                                                                    |
| :----------------------------------------------------------------------------------------------------------- | :----------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1)                                                                     |             **Current**              | Updated to focus on host integration & deep multi-engine parity as the highest-risk area and orchestrator rollout as the hardest problem. Serves as SSoT for risk framing.                                               |
| [`README.md`](README.md:1)                                                                                   | **Current (minor refresh optional)** | Setup and quickstart instructions are accurate; does not embed strong claims about current weakest aspect or test health. Could optionally add a short pointer to PASS18/weakness docs for readers seeking risk context. |
| [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1)                                                       |             **Current**              | Canonical RR‑CANON rules for capture, lines, territory, and Q23 are consistent with current engine implementations and parity tests.                                                                                     |
| [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1)                                             |             **Current**              | Accurately describes shared TS engine, orchestrator, adapters, and Python parity role. No major drift found.                                                                                                             |
| [`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md:1)                                                                 |       **Current (watchlist)**        | Architecture and RNG strategy are correct; some language about historical Python simplifications may need softening once PASS18 parity work completes.                                                                   |
| [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1)                                               |       **✅ Updated (P18.6-1)**       | Updated with post-P18.5 parity status (43 vectors, 0 mismatches), orchestrator Phase 4 complete, risk register added, test health section aligned with P18.18.                                                           |
| [`CURRENT_RULES_STATE.md`](CURRENT_RULES_STATE.md:1)                                                         |           **Needs update**           | "No critical known issues" is no longer accurate given open advanced-phase/parity work; should explicitly reference PASS18 weakest-aspect findings.                                                                      |
| [`WEAKNESS_ASSESSMENT_REPORT.md`](WEAKNESS_ASSESSMENT_REPORT.md:1)                                           |       **✅ Updated (P18.6-1)**       | Added Section 3 "Completed Remediations" documenting P18.1-5 work, updated historical assessments with resolution notes, downgraded hardest problem severity.                                                            |
| [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md:1)                                                                       |       **✅ Updated (P18.6-1)**       | Downgraded P0.2 parity severity to LOW, added P18.5-\* resolution details, added Design Clarifications section (DC.1 for mid-phase vectors), updated historical issues.                                                  |
| [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md:1)                                                         |           **Needs update**           | Still describes ANM/forced-elimination as the "current highest-risk semantics" area; should either mark that as historical or point to PASS18/weakness docs for current risk.                                            |
| [`docs/INDEX.md`](docs/INDEX.md:1)                                                                           |           **Needs update**           | Over-emphasises ANM invariants as "highest-risk semantics"; should add notes that current weakest aspect & hardest problem are tracked in PASS18 and weakness reports.                                                   |
| [`docs/SSOT_BANNER_GUIDE.md`](docs/SSOT_BANNER_GUIDE.md:1)                                                   |             **Current**              | Correctly defines SSOT banner expectations and categories; no changes required beyond optional references to new PASS18 docs.                                                                                            |
| [`RULES_DYNAMIC_VERIFICATION.md`](RULES_DYNAMIC_VERIFICATION.md:1)                                           |    **Current (historical focus)**    | Accurately describes dynamic verification approach with a focus on ANM and earlier risk areas; remains useful as a historical and methodological reference.                                                              |
| [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:1) |             **Current**              | Documents tricky rules edge cases consistent with RR‑CANON; can be expanded with new cases found during PASS18 but not fundamentally stale.                                                                              |
| [`docs/supplementary/RULES_RULESET_CLARIFICATIONS.md`](docs/supplementary/RULES_RULESET_CLARIFICATIONS.md:1) |             **Current**              | Provides clarifications for earlier ambiguities; still aligned with current rules.                                                                                                                                       |
| [`docs/supplementary/RULES_DOCS_UX_AUDIT.md`](docs/supplementary/RULES_DOCS_UX_AUDIT.md:1)                   |     **Current (to be extended)**     | Correctly identifies UX/documentation mismatches; PASS18 UX work will likely add new findings rather than invalidate existing ones.                                                                                      |
| [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:1)                                   |             **Current**              | Already reflects Phase 4 completion status.                                                                                                                                                                              |
| [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1)                       |             **Current**              | Already aligned with extended contract vectors and parity status.                                                                                                                                                        |

These findings feed directly into P18.7-ARCH (core docs cleanup) and P18.10-ARCH (SSOT & CI guardrails), and they are summarised in the PASS18 assessment report's documentation sections.
