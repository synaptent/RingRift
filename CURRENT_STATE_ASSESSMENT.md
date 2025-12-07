# RingRift Current State Assessment

**Assessment Date:** 2025-12-05 (Post-Wave 14, Engine Refactoring Review)
**Last CI Test Runs:** 2025-12-05 (TypeScript CI-gated suites: 2,987 passed, 0 failing; Python: 836 passing)
**Last Full Jest Snapshot:** See `jest-results.json` and [`docs/archive/assessments/PASS20_ASSESSMENT.md`](docs/archive/assessments/PASS20_ASSESSMENT.md) for extended/diagnostic profile analysis
**Assessor:** Code + Test Review + CI Analysis
**Purpose:** Factual status of the codebase as it exists today

> **PASS20-21 Update (2025-12-01):** This document has been updated to reflect PASS18/19/20/21 completed work:
>
> - **PASS18 (33 tasks):** Host parity, RNG alignment, decision lifecycle, orchestrator Phase 4 rollout, extended contract vectors (54 cases)
> - **PASS19 (12 tasks):** E2E test infrastructure, game fixtures, type safety improvements, test rewrites
> - **PASS20 (25 tasks):** Phase 3 orchestrator migration complete, ~1,176 lines legacy code removed, TEST_CATEGORIES.md documentation
> - **PASS21 (Observability Infrastructure):** 3 Grafana dashboards created, k6 load testing framework implemented, monitoring stack by default
> - **Orchestrator Migration:** ✅ Phase 3 COMPLETE – See [`docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md)
> - All CI-gated TypeScript tests passing (2,987 tests), Python tests stable (836 tests)
> - Project health status: **GREEN**

> **Doc Status (2025-12-04): Active**
> Current high-level snapshot of implementation status across backend, client, shared engine, Python AI service, and tests. This document is **not** a rules or lifecycle SSoT; it reports factual status against the canonical semantics and lifecycle sources of truth.
>
> - **Rules semantics SSoT:** The canonical rules documents (`RULES_CANONICAL_SPEC.md` together with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`) are the **single source of truth** for RingRift game semantics. The shared TypeScript engine under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts) is the primary executable derivation of that spec, validated by contract vectors and runners (`tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`) and mapped in `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, and `docs/RULES_ENGINE_SURFACE_AUDIT.md`. Implementations must be kept in lockstep with the canonical rules spec.
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas under `src/shared/types/**`, `src/shared/engine/orchestration/types.ts`, and `src/shared/validation/websocketSchemas.ts` (plus `docs/API_REFERENCE.md` for transport details).
> - Historical architecture or remediation context lives in `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, and archived reports; this file should remain narrowly focused on **current factual status**.
> - **Relationship to goals:** For the canonical statement of RingRift’s product/technical goals, v1.0 success criteria, and scope boundaries, see [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1). This document reports the **current factual status** of the implementation and tests relative to those goals and to the phased roadmap in [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:1); it does not define new goals.
>
> This document is the **Single Source of Truth** for the project's _implementation status_ and for the current test counts and coverage metrics referenced by overview/goal docs such as [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) and [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md:1).
> It supersedes `IMPLEMENTATION_STATUS.md` and should be read together with:
>
> - `KNOWN_ISSUES.md` – P0/P1 issues and gaps
> - `TODO.md` – phase/task tracker
> - `STRATEGIC_ROADMAP.md` – phased roadmap to MVP
> - `docs/TEST_CATEGORIES.md` – canonical map of CI‑gated vs diagnostic/extended test categories and how to run them

The intent here is accuracy, not optimism. When in doubt, the **code and tests** win over any percentage or label.

---

## 📊 Executive Summary

**Overall:** Strong architectural foundation with consolidated rules engine; **stable beta approaching production readiness**. Project health status: **GREEN**.

- **Architecture Remediation Complete:** The 4-phase architecture remediation (November 2025) consolidated the rules engine:
  - Canonical turn orchestrator in `src/shared/engine/orchestration/`
  - Backend adapter (`TurnEngineAdapter.ts`) and sandbox adapter (`SandboxOrchestratorAdapter.ts`)
  - Contract testing framework with 100% Python parity on **54 test vectors** (extended from 12 in P18.5-\*)
  - **Orchestrator at Phase 4 (hard-ON):** The orchestrator adapter is hardcoded to `true` in `EnvSchema` and the rollout-percentage flag was removed as part of the Phase 3 cleanup. All environments run the orchestrator as the authoritative path; soak tests show zero invariant violations across all board types (square8, square19, hexagonal).

- **Rules Semantics & Training Data Hygiene (2025‑12‑05):**
  - Canonical rules spec tightened:
    - Line processing (RR‑CANON‑R120–R122) now explicitly requires **all** line effects (exact‑length and overlength) to be driven by explicit `process_line` / `choose_line_reward` decisions, never as implicit side‑effects of movement or capture.
    - RR‑CANON‑R074 mandates that every turn action and voluntary forgo decision (e.g. `skip_placement`, `skip_capture`, `skip_territory_processing`) is represented as an explicit move/choice in the game record; engines must not silently assume these.
  - Replay parity infrastructure:
    - TS sandbox replay in `traceMode` aligns with Python `GameEngine` semantics for movement/capture/line/territory when driven from canonical GameReplayDBs.
    - `ai-service/scripts/check_ts_python_replay_parity.py`, `scripts/selfplay-db-ts-replay.ts`, and `ai-service/scripts/debug_ts_python_state_diff.py` provide end‑to‑end TS↔Python trace and structural diff tooling.
  - Self‑play data and model hygiene:
    - `ai-service/TRAINING_DATA_REGISTRY.md` classifies all known GameReplayDBs and NN checkpoints as `canonical`, `legacy_noncanonical`, or `pending_gate`, and documents which artifacts are safe for new training.
    - `ai-service/scripts/run_canonical_selfplay_parity_gate.py` runs a small canonical Python self‑play soak for a given board type, records games into a fresh GameReplayDB, and gates that DB on TS↔Python replay parity (no structural issues, no semantic divergences).
    - Legacy self‑play DBs (e.g. early `selfplay_square8_2p.db`, `selfplay_square19_*.db`, `selfplay_hexagonal_*.db`, `selfplay.db`) and v1 NN checkpoints trained on them are explicitly marked `legacy_noncanonical` and are no longer used for new training runs; they are retained only for historical comparison and ablations.

- **PASS18 Complete (33 tasks, 2025-12):**
  - **P18.1-\*: Host Parity** – Capture/territory host unification, advanced-phase ordering aligned
  - **P18.2-\*: RNG Determinism** – AI RNG seed handling aligned across TS and Python
  - **P18.3-\*: Decision Lifecycle** – Timeout semantics and decision phase alignment
  - **P18.4-\*: Orchestrator Rollout** – Phase 4 complete in all environments
  - **P18.5-\*: Extended Vectors** – 54 contract vectors (placement, movement, capture/chain_capture, forced_elimination, territory/territory_line endgames, hex edge cases, meta moves including swap_sides and multi-phase turns), swap_sides parity verified
  - **P18.18: Test Triage** – Obsolete tests removed, RulesMatrix partially re-enabled

- **PASS19 Complete (12 tasks, 2025-11-30):**
  - **E2E Test Infrastructure** – MultiClientCoordinator, NetworkSimulator, TimeController for complex multiplayer testing
  - **Game Fixtures** – Near-victory (`near_victory_elimination`), chain capture, and multi-phase scenario fixtures
  - **Type Safety** – Continued `any` cast reduction (~37 explicit casts remaining)
  - **Visual Regression** – Playwright test infrastructure improvements
  - **Test Rewrites** – Region Order and LPS Cross-Interaction tests updated for orchestrator

- **PASS20 Complete (25 tasks, 2025-12-01):**
  - **Phase 3 Orchestrator Migration COMPLETE** – All critical legacy code paths removed
  - **Legacy Code Removed:** ~1,176 lines (RuleEngine methods, feature flags, ClientSandboxEngine legacy, obsolete tests)
  - **Test Suite Stabilization** – 6 critical test issues fixed
  - **Victory Detection Bug** – Fixed victory condition edge case
  - **Documentation** – [`docs/TEST_CATEGORIES.md`](docs/TEST_CATEGORIES.md), [`docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md)
  - See full summary: [`docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md)

- **PASS21 Complete (Observability Infrastructure, 2025-12-01):**
  - **Observability:** 3 Grafana dashboards created (game-performance, rules-correctness, system-health)
  - **Load Testing:** k6 framework implemented with 4 production-scale scenarios
  - **Monitoring:** Monitoring stack runs by default (moved from optional profile)
  - **Dashboards:** 22 panels across performance, correctness, and health metrics
  - **Test Coverage:** Context coverage dramatically improved (GameContext 89.52%, SandboxContext 84.21%)
  - **Observability Score:** Improved from 2.5/5 → 4.5/5
  - See full assessment: [`docs/archive/assessments/PASS21_ASSESSMENT_REPORT.md`](docs/archive/assessments/PASS21_ASSESSMENT_REPORT.md)

- **Current Focus (Post-PASS21):** With observability infrastructure in place and load testing framework implemented, the primary focus is **Production Validation** (running load tests at scale, establishing baseline metrics) and **Frontend UX Polish** (scenario picker refinement, spectator UI improvements, keyboard navigation testing).

- **Core Rules:** Movement, markers, captures (including chains), lines, territory, forced elimination, and victory are implemented in the shared TypeScript rules engine under [`src/shared/engine`](src/shared/engine/types.ts) and reused by backend and sandbox hosts. These helpers are exercised by focused Jest suites with 285+ test files providing comprehensive coverage.
- **Backend & Sandbox Hosts:** The backend `RuleEngine` / `GameEngine` and the client `ClientSandboxEngine` act as thin adapters over the shared helpers, wiring in IO (WebSockets/HTTP, persistence, AI) while delegating core game mechanics to shared validators/mutators and geometry helpers.
- **Backend Play:** WebSocket-backed games work end-to-end, including AI turns via the Python service / local fallback and server-driven PlayerChoices surfaced to the client.
- **Session Management:** `GameSessionManager` and `GameSession` provide robust, lock-protected game state access with Redis caching.
- **Frontend:** The React client has a usable lobby, backend GamePage (board + HUD + victory modal), and a rich local sandbox harness with full rules implementation.
- **Testing:** Comprehensive coverage with 285+ test files (2,987 TypeScript tests passing in CI-gated suites, 836 Python tests passing). Extended/diagnostic TypeScript suites (parity, AI simulation, trace debugging) are documented in [`docs/TEST_CATEGORIES.md`](docs/TEST_CATEGORIES.md) and tracked via `KNOWN_ISSUES.md`.
- **CI/CD:** Mature GitHub Actions workflow with separated job types (lint, test, build, security scan, Docker, E2E) and proper timeout protections.

A reasonable label for the current state is: **stable beta with consolidated architecture, suitable for developers, AI work, and comprehensive playtesting**, ready for production hardening.

---

## ✅ Verified Implementation Status

### 1. Core Game Logic & Engines

- **Shared Rules Engine (`src/shared/engine/`)**
  - **Complete:** Canonical `GameState` / `GameAction` types, validators, and mutators for all core mechanics
  - **Movement & captures:** [`movementLogic.ts`](src/shared/engine/movementLogic.ts), [`captureLogic.ts`](src/shared/engine/captureLogic.ts), with full mutator support
  - **Lines:** [`lineDetection.ts`](src/shared/engine/lineDetection.ts), [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts) with canonical Move enumeration
  - **Territory:** [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts), [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts), [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts)
  - **Victory & placement:** [`victoryLogic.ts`](src/shared/engine/victoryLogic.ts), [`placementHelpers.ts`](src/shared/engine/placementHelpers.ts:1) with full canonical placement validation (16 tests)
  - **Shared helpers:** All 4 previously stubbed modules now fully implemented:
    - [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1) – canonical movement application (13 tests)
    - [`placementHelpers.ts`](src/shared/engine/placementHelpers.ts:1) – no-dead-placement validation (16 tests)
    - [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1) – chain capture orchestration (20 tests)
    - [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1) – capture search and execution (existing comprehensive tests)
  - **Turn lifecycle:** [`turnLogic.ts`](src/shared/engine/turnLogic.ts), [`turnLifecycle.ts`](src/shared/engine/turnLifecycle.ts) with canonical phase transitions

- **Canonical Turn Orchestrator (`src/shared/engine/orchestration/`)** (NEW)
  - **Complete:** Single entry point for turn processing via `processTurn()` / `processTurnAsync()`
  - **Phase state machine:** [`phaseStateMachine.ts`](src/shared/engine/orchestration/phaseStateMachine.ts) handles phase transitions
  - **Domain aggregates:** Orchestrator calls all 6 aggregates (Placement, Movement, Capture, Line, Territory, Victory) in deterministic order
  - **Documentation:** Comprehensive usage guide in [`orchestration/README.md`](src/shared/engine/orchestration/README.md)

- **Contract Testing (`src/shared/engine/contracts/`)**
  - **Complete:** Contract schemas and deterministic serialization for cross-language parity
  - **Test vectors:** 49 vectors across 8+ categories (placement, movement, capture, line, territory, chain_capture, forced_elimination, territory_line_endgame, hex_edge_cases, near_victory_territory, meta_moves)
  - **Python parity:** 100% pass rate (0 mismatches) on contract tests between TypeScript and Python engines
  - **swap_sides (Pie Rule):** Verified across TS backend, TS sandbox, and Python per P18.5-4 report

- **BoardManager & Geometry**
  - **Complete:** Full support for 8×8, 19×19, and hexagonal boards
  - **Topology:** Position generation, adjacency, distance calculations, and pathfinding
  - **Territory detection:** Region finding and disconnection validation across all board types
  - **Line detection:** Marker line geometry with minimum length enforcement

- **Backend GameEngine & RuleEngine**
  - **Complete:** Full orchestration of turn/phase loop with WebSocket integration
  - **Phases:** `ring_placement → movement → capture → chain_capture → line_processing → territory_processing → next player`
  - **Decision integration:** Uses shared validators/mutators plus `PlayerInteractionManager` for all rule-driven decisions
  - **AI integration:** Seamless AI turns via `globalAIEngine` and `AIServiceClient`
  - **Chain captures:** Unified `chain_capture`/`continue_capture_segment` model live and tested
  - **Orchestrator adapter:** [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts) (326 lines) wraps orchestrator with session/WebSocket concerns

- **ClientSandboxEngine & Local Play**
  - **Complete:** Client-local sandbox engine as thin host over shared helpers
  - **Canonical moves:** Emits proper `Move` history for both AI and human flows
  - **Mixed games:** Supports human/AI combinations with unified turn semantics
  - **Parity:** Strong semantic alignment with backend engine, validated by comprehensive test suites
  - **Orchestrator adapter:** [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts) (476 lines) wraps orchestrator for local simulation

### 2. Backend Infrastructure

- **HTTP API & Routes**
  - **Complete:** Full authentication (`/api/auth`), game management (`/api/games`), and user endpoints (`/api/users`)
  - **Security:** JWT-based auth, rate limiting, CORS, security headers, input validation
  - **Game lifecycle:** Create/join/leave games, lobby listing, spectator support

- **WebSocket Server**
  - **Complete:** Authenticated Socket.IO server with full game event handling
  - **Events:** `join_game`, `player_move`, `player_choice_response`, `chat_message` with proper state synchronization
  - **AI turns:** Automatic AI turn processing via `maybePerformAITurn`
  - **Victory handling:** Proper game completion with `game_over` events and DB updates

- **Session Management & Persistence**
  - **Complete:** `GameSessionManager` with distributed locking (Redis-backed)
  - **Database:** Full Prisma schema with users, games, moves, ratings, and comprehensive migration history
  - **Caching:** Redis integration for session state and performance optimization

### 3. Frontend Client & UX

- **Core Components**
  - **Complete:** `BoardView` renders all board types with movement grid overlays
  - **Game contexts:** `GameContext` handles both backend WebSocket games and local sandbox
  - **UI library:** Full Tailwind CSS component system with `Button`, `Card`, `Badge`, `Input`, `Select`
  - **Responsive:** Works across desktop and mobile form factors

- **Game Interfaces**
  - **LobbyPage:** Complete game creation/joining with AI configuration, filters, and real-time updates
  - **GamePage:** Unified interface for both backend and sandbox games with `BoardView`, `GameHUD`, `ChoiceDialog`
  - **Sandbox:** Full `/sandbox` route with rules-complete client-local engine
  - **Victory:** `VictoryModal` with proper game completion flows

- **Accessibility Features (Wave 14.5)**
  - **Complete:** `AccessibilityContext` provides user preference management
  - **Keyboard navigation:** `useKeyboardNavigation` hook for board-level interactions (arrow keys, Enter, Escape, Tab)
  - **Screen reader support:** `ScreenReaderAnnouncer` component for live game state announcements
  - **Visual accessibility:** `AccessibilitySettingsPanel` with colorblind-friendly modes (deuteranopia, protanopia, tritanopia)
  - **High-contrast mode:** Available via AccessibilitySettingsPanel
  - **Reduced motion:** Respects `prefers-reduced-motion` OS setting
  - **Documentation:** Comprehensive [`docs/ACCESSIBILITY.md`](docs/ACCESSIBILITY.md) guide

- **Player Choice System**
  - **Complete:** `ChoiceDialog` renders all PlayerChoice variants (line rewards, elimination, region order, capture direction)
  - **Integration:** Seamless human choice handling via `GameContext.respondToChoice`
  - **AI choices:** Both backend and sandbox support AI decision-making for all choice types

### 4. AI Integration & Python Service

- **Python AI Service (`ai-service/`)**
  - **Complete:** FastAPI service with Random, Heuristic, Minimax, and MCTS implementations
  - **Endpoints:** `/ai/move`, `/ai/evaluate`, and choice-specific endpoints (`/ai/choice/line_reward_option`, etc.)
  - **Difficulty mapping:** Canonical 1–10 difficulty ladder with engine selection:
    - 1: RandomAI; 2: HeuristicAI; 3–6: MinimaxAI; 7–8: MCTSAI (+ NeuralNetAI backend); 9–10: DescentAI (+ NeuralNetAI backend).
    - Lobby currently exposes the numeric ladder; difficulties **7–10** are treated as a "Stronger Opponents" band and are intended for advanced/experimental play rather than default rated queues.
  - **Rules parity:** Python rules engine maintains alignment with TypeScript implementation
  - **Training infrastructure (2025-12-05):**
    - **Model versioning:** `ModelVersionManager` with architecture version, checksum, and config validation
    - **Training data registry:** `ai-service/TRAINING_DATA_REGISTRY.md` classifies DBs as canonical/legacy_noncanonical
    - **Canonical parity gate:** `scripts/run_canonical_selfplay_parity_gate.py` validates self-play DBs
    - **Dataset export:** `scripts/export_replay_dataset.py` for NPZ training data with rank-aware values

- **TypeScript AI Boundary**
  - **Complete:** `AIServiceClient` and `AIEngine` with comprehensive error handling and fallbacks
  - **Integration:** `AIInteractionHandler` delegates choices to service with local fallback
  - **Game creation:** Full AI opponent configuration in lobby with profile/difficulty selection
  - **Session integration:** Seamless AI turn execution in `GameSession` workflows

### 5. Testing & Quality Assurance

- **Test Infrastructure**
  - **Comprehensive:** 230+ test files across unit, integration, scenario, and E2E categories
  - **Test types:** Jest (unit/integration), Playwright (E2E), pytest (Python AI service)
  - **Coverage:** Structured test matrix covering rules, parity, AI boundary, and UI integration
  - **Timeout protection:** Robust test execution with proper timeout handling via scripts

- **Test Categories**
  - **Shared engine tests:** Movement, captures, lines, territory, victory with focused unit tests (100+ tests for shared helpers)
  - **Component tests:** 209 component tests including 160 core components and 49 ChoiceDialog tests
  - **Hooks tests:** 98 tests covering useGameState, useGameActions, useGameConnection
  - **Context tests:** 51 tests covering AuthContext and GameContext
  - **Service tests:** 27 HealthCheckService tests for health monitoring
  - **Parity suites:** Backend ↔ sandbox ↔ shared engine alignment validation
  - **Scenario tests:** Rules/FAQ matrix covering Q1-Q24 from `ringrift_complete_rules.md`
  - **AI boundary tests:** Service integration, fallbacks, choice delegation
  - **Integration tests:** WebSocket flows, game lifecycle, session management

- **CI/CD Pipeline**
  - **Complete:** GitHub Actions with lint, test, build, security scan, Docker build
  - **Coverage:** Codecov integration with PR comment reporting
  - **Security:** npm audit, Snyk scanning, dependency checks for both Node.js and Python
  - **Multi-stage:** Separated job types with proper dependency management and timeout protection

---

## ❌ Major Gaps & Current Limitations

### P0 – Production Hardening

- **Orchestrator migration COMPLETE (Phase 3):** The canonical orchestrator is complete and legacy code paths removed:
  - Backend and sandbox hosts via `TurnEngineAdapter` / `SandboxOrchestratorAdapter`.
  - CI gates (`orchestrator-parity`, short/long orchestrator soaks).
  - S-invariant regression suites and contract vectors (54/54 passing).
  - HTTP/load diagnostics via `scripts/orchestrator-load-smoke.ts` (see `npm run load:orchestrator:smoke`).

  **Phase 3 Complete (2025-12-01):** Orchestrator migration through Phase 3 is COMPLETE:
  - [x] Flip staging to the Phase 1 preset from `ORCHESTRATOR_ROLLOUT_PLAN.md` Table 4 and keep it there as the steady state.
        **Completed:** `.env.staging` uses the hardcoded orchestrator adapter (`ORCHESTRATOR_ADAPTER_ENABLED` coerced to `true`), `RINGRIFT_RULES_MODE=ts`, and circuit breaker enabled. The rollout-percentage flag was removed during Phase 3.
  - [x] Exercise the Phase 1 → 2 → 3 **phase completion checklist** in `ORCHESTRATOR_ROLLOUT_PLAN.md` §8.7.
        **Completed:** P18.4-\* orchestrator rollout phases validated via staging soak and extended vector soak (P18.5-3).
  - [x] Enable orchestrator for 100% of traffic in all environments.
        **Completed:** `.env` files rely on the hardcoded adapter (no rollout flag) across dev, staging, and CI.
  - [x] Remove legacy rules code paths in backend and sandbox hosts.
        **Completed (P20.7-\*):** ~1,118 lines of legacy code removed. See [`docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md).
  - [ ] Phase 4 (Tier 2 sandbox cleanup) deferred to post-MVP.

- **Environment rollout posture & presets (repo-level):**
  - **CI defaults (orchestrator‑ON, TS authoritative):** All primary TS CI jobs (`test`, `ts-rules-engine`, `ts-orchestrator-parity`, `ts-parity`, `ts-integration`, `orchestrator-soak-smoke`) run with:
    - `RINGRIFT_RULES_MODE=ts`
    - `ORCHESTRATOR_ADAPTER_ENABLED` hardwired to `true` by `EnvSchema` (runtime ignores env overrides)
    - `ORCHESTRATOR_SHADOW_MODE_ENABLED=false`  
      as defined in `.github/workflows/ci.yml`. This matches the **Phase 1 – orchestrator‑only** preset in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` Table 4 for test/CI environments.
  - **Shadow‑mode profile (diagnostic only):** A standard manual profile for TS‑authoritative + Python shadow parity runs is documented in `tests/README.md` and `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (for example:
    `RINGRIFT_RULES_MODE=shadow`, `ORCHESTRATOR_ADAPTER_ENABLED=true` (hardcoded), `ORCHESTRATOR_SHADOW_MODE_ENABLED=true`). This profile is not wired as a dedicated CI job; it is intended for ad‑hoc parity investigations and pre‑production shadow checks.
  - **Staging / production posture (out of repo scope):** This repository encodes the **intended** rollout phases and presets for staging and production in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §8, but does not track actual live environment state. Whether a given staging or production stack is currently running in Phase 0/1/2/3/4 is an operational concern outside this codebase and must be validated against deployment config and observability (SLOs, alerts, dashboards).

### P0 – Engine Parity & Rules Coverage

- **Backend ↔ Sandbox trace parity:** Major divergences DIV-001 (capture enumeration) and DIV-002 (territory processing) have been **RESOLVED** through unified shared engine helpers. Remaining semantic gaps (DIV-003 through DIV-007) are open but lower priority. DIV-008 (late-game phase/player tracking) is deferred as within tolerance.
- **Cross-language parity:** Contract tests now ensure 100% parity between TypeScript and Python engines on **54 test vectors** (extended from 12 in P18.5-\*). The extended vectors cover:
  - Chain captures with multi-segment sequences
  - Forced elimination scenarios
  - Territory/line interaction endgames
  - Hexagonal board edge cases
  - swap_sides (Pie Rule) parity (verified per P18.5-4)
- **Replay-level TS↔Python parity:** Differential replay harnesses compare Python `game_history_entries` against TS `selfplay-db-ts-replay.ts` output on recorded games:
  - `ai-service/scripts/check_ts_python_replay_parity.py` – per-game summary with:
    - Structure classification (`good`, `mid_snapshot`, `invalid`, `error`).
    - First semantic divergence point (`diverged_at`) with per-dimension tags:
      - `mismatch_kinds`: `["current_player"]`, `["current_phase"]`, `["game_status"]`, `["move_count"]`, or `["ts_missing_step"]`.
      - `mismatch_context`: `"initial_state"`, `"post_move"`, or `"global"`.
    - Aggregate `mismatch_counts_by_dimension` across all games.
  - `tests/parity/test_differential_replay.py` – diagnostic and strict tests:
    - Diagnostic run logs divergences for an existing `games.db` if present.
    - A golden-game hook (`RINGRIFT_PARITY_GOLDEN_DB` and `RINGRIFT_PARITY_GOLDEN_GAME_ID`) enforces zero divergences for specific high-value games when configured in CI.
- **Decision phase timeout guards:** Implemented for line, territory, and chain‑capture decision phases, with WebSocket events (`decision_phase_timeout_warning`, `decision_phase_timed_out`) and `DECISION_PHASE_TIMEOUT` error code wired into `GameSession` and validated by `GameSession.decisionPhaseTimeout.test.ts`.
- **Invariant metrics and alerts:** Orchestrator invariant violations are exported via `ringrift_orchestrator_invariant_violations_total{type,invariant_id}` and drive the `OrchestratorInvariantViolations*` alerts; Python strict‑invariant soaks (including AI healthchecks) export `ringrift_python_invariant_violations_total{invariant_id,type}` and drive the `PythonInvariantViolations` alert, as documented in `INVARIANTS_AND_PARITY_FRAMEWORK.md` and `ORCHESTRATOR_ROLLOUT_PLAN.md`.
- **Complex scenario coverage:** Core mechanics well-tested, but some complex composite scenarios (deeply nested capture + line + territory chains) rely on trace harnesses and TS-generated parity fixtures rather than only focused scenario tests.
- **Chain capture edge cases:** 180-degree reversal and cyclic capture patterns are supported and exercised by dedicated tests (`test_chain_capture_parity.py`, `test_chain_capture_phase_fix.py`, and default-engine equivalence tests), but additional TS fixtures and replay-backed golden games may be added over time for complete confidence.

### P1 – Multiplayer UX Polish & E2E Coverage

- **Spectator experience:** Basic spectator mode implemented; dedicated spectator layout and EvaluationPanel integration are in place, though some UX polish remains.
- **Reconnection UX:** Core reconnection and abandonment flows are implemented and now exercised end‑to‑end:
  - `multiPlayer.coordination.test.ts` – swap rule, near‑victory fixtures (elimination/territory), deep chain‑capture decisions.
  - `reconnection.simulation.test.ts` – network partition and reconnection‑window expiry, including rated vs unrated abandonment behaviour.
  - `tests/unit/GameSession.reconnectFlow.test.ts` – server-side reconnect window handling and guarantees that `handleJoinGame` always emits a fresh `game_state` snapshot on reconnect (players and spectators).
  - `tests/unit/contexts/GameContext.test.tsx` – verifies that fresh `game_state` snapshots clear stale `pendingChoice`/`choiceDeadline`/timeout warnings, so HUD decision banners never survive a reconnect.
  - `tests/unit/hooks/useGameConnection.test.tsx` – covers connection status transitions (`connecting`/`reconnecting`/`connected`) and reconnect labels used by BackendGameHost/GameHUD.
  - `tests/unit/client/BackendGameHost.test.tsx` – host-level reconnect UX from the client’s perspective: asserts that during `reconnecting` the HUD shows “Connection: Reconnecting…”, the board interaction helper reads “Reconnecting to server…”, moves are effectively read-only, and that after a fresh `game_state` snapshot the reconnect copy disappears and stale decision UI is cleared.
  - `timeout-and-ratings.e2e.spec.ts` – short time‑control timeout completions with `result.reason === 'timeout'` and rating semantics (rated vs unrated).
  - `decision-phase-timeout.e2e.spec.ts` – decision‑phase timeouts for line, territory **and chain_capture** via shortTimeoutMs fixtures, asserting deterministic `decision_phase_timed_out` payloads for both players.
  - Additional MultiClientCoordinator slices now cover back‑to‑back near‑victory fixtures (elimination and territory_control) across two players **and a spectator**, asserting consistent `game_over` reasons/winner on all three clients.
- **Chat & social:** In-game chat infrastructure present but persistence and advanced social features limited
- **Advanced matchmaking:** Limited to manually refreshed lobby; no automated queue or ELO-based matching

### P1 – AI Strength & Observability

- **AI tactical depth:** Service integration complete but still relies primarily on heuristic evaluation; advanced search and ML implementations experimental
- **Observability:** Logging and basic metrics present but no comprehensive dashboard or real-time performance monitoring
- **Choice coverage:** Most PlayerChoices service-backed, but some (`line_order`, `capture_direction`) still use local heuristics only

---

## 📋 Risk Register (Post-PASS20)

This section summarizes the risk status as of PASS20 completion (2025-12-01).

### ✅ Resolved Risks

| Risk                                 | Resolution                                                    | Evidence                                  |
| ------------------------------------ | ------------------------------------------------------------- | ----------------------------------------- |
| TS↔Python phase naming divergence    | Unified phase state machine in orchestrator                   | P18.1-\* host parity work                 |
| Capture chain ordering inconsistency | Shared `captureChainHelpers.ts` enforces deterministic order  | 54 contract vectors (0 mismatches)        |
| RNG determinism drift                | Seed handling aligned per P18.2-\*                            | AI RNG paths documented, tested           |
| Decision lifecycle timing gaps       | Timeout semantics aligned per P18.3-\*                        | `docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md` |
| swap_sides (Pie Rule) parity         | Verified across TS backend, TS sandbox, and Python            | P18.5-4 report (5/5 TS, 2/2 Python tests) |
| Test suite confusion (CI vs diag)    | Documented in TEST_CATEGORIES.md, jest-results.json clarified | PASS20 investigation complete             |
| Victory detection edge case          | Fixed victory condition bug                                   | PASS20.0-2 resolution                     |
| BoardManager config in tests         | Fixed BOARD_CONFIGS[boardType] undefined                      | 42/42 WebSocket tests passing             |

### ⚠️ Mitigated Risks

| Risk                                | Mitigation                                                    | Residual Concern                 |
| ----------------------------------- | ------------------------------------------------------------- | -------------------------------- |
| Orchestrator architecture stability | Phase 3 complete, ~1,118 lines legacy removed                 | Phase 4 (Tier 2) deferred        |
| Contract vector coverage gaps       | Extended to 54 vectors (core, chains, elimination, territory) | May need additional edge cases   |
| Test suite health                   | PASS20 fixes complete, TEST_CATEGORIES.md added               | ~170 skipped tests (intentional) |

### 🔴 Active Risks

| Risk                          | Status                                  | Next Step                                                                                                       |
| ----------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Frontend UX completeness      | P0/P1 tasks pending                     | Sandbox scenario picker refinement, spectator UI improvements                                                   |
| E2E multiplayer coverage      | Infra in place, targeted coverage added | Focus remaining work on heavier soaks / load (multi‑game sessions, longer chains) rather than basic correctness |
| Production preview validation | Not yet exercised                       | Needs real traffic validation                                                                                   |
| Hexagonal geometry edge cases | Coverage improved via test vectors      | Monitor for new edge cases in play                                                                              |
| Tier 2 sandbox cleanup        | Deferred to Phase 4 (post-MVP)          | ~1,200 lines remain, not blocking                                                                               |

### 📊 Risk Summary

- **Parity Risk:** LOW – 54 contract vectors with 0 mismatches, swap_sides verified
- **Orchestrator Risk:** LOW – Phase 3 complete, ~1,118 lines legacy code removed, all environments at 100%
- **Test Suite Risk:** LOW – CI-gated tests green (2,987 passed), diagnostic tests documented; E2E coverage for victory, resignation, abandonment, and timeout now in place
- **Frontend UX Risk:** MEDIUM – Known P0/P1 gaps pending UX polish work
- **E2E Coverage Risk:** MEDIUM – Complex multiplayer scenarios need infrastructure
- **Production Readiness Risk:** MEDIUM – Needs real traffic validation

---

## 🎯 Development Readiness Assessment

### ✅ Ready for Intensive Development & Testing

The project provides a solid foundation for:

- **Rules development:** Shared engine architecture supports rapid iteration with comprehensive test coverage
- **AI experimentation:** Full service integration with multiple AI types and difficulty levels
- **Frontend development:** Complete component system and game interfaces ready for UX improvements
- **Multiplayer testing:** Full WebSocket infrastructure with session management and real-time synchronization

### ⚠️ Approaching Production Readiness

Key remaining work for production deployment:

- **Scenario test completion:** Convert remaining diagnostic/trace tests to focused scenario coverage
- **UX polish:** Enhanced HUD, better reconnection flows, improved spectator experience
- **Performance optimization:** Load testing and scaling validation
- **Monitoring:** Production-grade observability and alerting

### 🛑 Not Yet Production-Ready

- **Security hardening:** Additional security review and dry‑run of rotation/backups needed for public deployment, even though:
  - Secrets inventory, rotation procedures, and SSoT checks (`SECRETS_MANAGEMENT.md`, `scripts/ssot/secrets-doc-ssot-check.ts`) are in place.
  - Data lifecycle and soft‑delete semantics are documented and implemented (`DATA_LIFECYCLE_AND_PRIVACY.md`, `OPERATIONS_DB.md`).
  - Operator-facing drills now exist as runbooks (`docs/runbooks/SECRETS_ROTATION_DRILL.md`, `docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`), but have not yet been exercised as part of a formal security review or incident‑response rehearsal.
- **Scale testing:** Performance under sustained high concurrent load and at production‑sized datasets is not yet validated; only:
  - Targeted orchestrator soaks (`npm run soak:orchestrator:*`) and
  - A lightweight HTTP load smoke (`npm run load:orchestrator:smoke`)
    have been run against smaller configurations.
- **Data lifecycle / backup drill:** Backup/recovery procedures and a concrete drill (`docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`) are documented, but the drill has not yet been institutionalised as a recurring operational exercise against staging/production‑like environments.

---

## 📈 Test Coverage Status

**Current Test Run (2025-12-01):** 285+ test files

- **TypeScript tests (CI-gated):** 2,987 passing, 0 failing, ~130 skipped
- **Python tests:** 836 tests passing
- **Contract tests:** 54 test vectors with 100% cross-language parity (0 mismatches)
- **Overall coverage:** ~69% lines (improved from 65.55%)
  - GameContext.tsx: 89.52% coverage (was 0%)
  - SandboxContext.tsx: 84.21% coverage (was 0%)

**Test Health (PASS20 Complete):**

- PASS20 investigation resolved jest-results.json confusion (stale Nov 21 snapshot)
- TEST_CATEGORIES.md documents CI-gated vs diagnostic test profiles
- Obsolete tests removed, archive files excluded from CI
- `structuredClone` polyfill added for Node.js 16 compatibility
- 42/42 WebSocket integration tests passing
- ~170 skipped tests (intentional): orchestrator-conditional, env-gated, diagnostic

**Test Categories:**

- **Integration tests:** ✅ Passing (AIResilience, GameReconnection, GameSession.aiDeterminism)
- **Scenario tests:** ✅ Passing (FAQ Q1-Q24 suites, RulesMatrix scenarios)
- **Unit tests:** ✅ Comprehensive coverage of core mechanics
- **Parity tests:** ✅ Passing (capture enumeration and territory integration now stable)
- **Contract tests:** ✅ 100% pass rate on 54 vectors across TypeScript and Python (extended in P18.5-\*)
- **Decision phase tests:** ✅ Timeout guards verified via `GameSession.decisionPhaseTimeout.test.ts`
- **Adapter tests:** ✅ 46 tests for orchestrator adapters
- **Component tests:** ✅ 209 tests (160 core + 49 ChoiceDialog)
- **Hooks tests:** ✅ 98 tests for client hooks
- **Context tests:** ✅ 51 tests for React contexts
- **Service tests:** ✅ 27 HealthCheckService tests
- **Shared helper tests:** ✅ 49 tests (13 movementApplication + 16 placementHelpers + 20 captureChainHelpers)

**Test Infrastructure:**

- **285+ total test files** providing comprehensive coverage
- **Timeout protection** via `scripts/run-tests-with-timeout.sh` preventing CI hangs
- **Categorized execution** with `test:core`, `test:diagnostics`, `test:ts-rules-engine` scripts
- **Test category documentation** via [`docs/TEST_CATEGORIES.md`](docs/TEST_CATEGORIES.md)
- **Coverage reporting** integrated with Codecov for PR feedback (~64% current, 80% target)
- **Contract testing** via `npm run test:contracts` and `scripts/run-python-contract-tests.sh`
- **MCTS tests:** Gated behind `ENABLE_MCTS_TESTS=1` with configurable timeout via `MCTS_TEST_TIMEOUT`
- **E2E tests:** Playwright configuration with `E2E_BASE_URL` and `PLAYWRIGHT_WORKERS` support
- **E2E Fixtures:** Near-victory (`near_victory_elimination`), chain capture, and multi-phase fixtures for game completion testing

---

## 🔄 Recommended Next Steps

Based on current state (post-PASS20, orchestrator Phase 3 complete, project health GREEN):

1. **Frontend UX Polish** - Sandbox scenario picker refinement, spectator UI improvements, HUD enhancements
2. **E2E Coverage Expansion** - Multi-context WebSocket coordination for timeout/reconnection tests, visual regression tests
3. **Production Validation** - Real traffic testing with orchestrator at 100%
4. **Coverage Threshold** - Increase from ~64% toward 80% target
5. **Phase 4 Tier 2 Cleanup** - Add SSOT banners to sandbox modules, archive diagnostic-only modules (post-MVP)
6. **Expand Contract Vectors** - Add edge case vectors as discovered in production play

See [`docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md`](docs/archive/assessments/PASS20_COMPLETION_SUMMARY.md) for detailed PASS21 recommendations.

The project has reached a mature beta state with consolidated architecture. PASS18/19/20 have stabilized the test suite, fixed critical bugs, and documented test categories. The codebase is ready for production hardening and UX polish work.

---

## 📊 Component Scores (Post-PASS21)

| Component                    | Score | Trend | Notes                                                    |
| :--------------------------- | :---: | :---: | :------------------------------------------------------- |
| **Rules Engine (Shared TS)** | 4.7/5 |   ➔   | Excellent, orchestrator at 100%, minor doc gaps          |
| **Rules Engine (Python)**    | 4.5/5 |   ➔   | Stable, 836 tests passing                                |
| **Test Suite**               | 4.0/5 |   ➔   | Stable, failures resolved, TEST_CATEGORIES.md added      |
| **Observability/Monitoring** | 4.5/5 |   ↗   | Improved from 2.5 – 3 dashboards, k6 load tests, default |
| **Documentation**            | 4.0/5 |   ➔   | Comprehensive, DOCUMENTATION_INDEX.md created            |
| **WebSocket**                | 4.0/5 |   ➔   | 42/42 tests passing with test helpers                    |
| **Frontend UX**              | 3.5/5 |   ➔   | Still needs polish work (sandbox, spectator UI)          |
| **Backend API**              | 4.2/5 |   ➔   | Solid routes, session management robust                  |
| **AI Service**               | 4.0/5 |   ➔   | Heuristic functional, training infra exists              |
| **DevOps/CI**                | 4.0/5 |   ➔   | CI jobs wired, diagnostic suites documented              |

---

## 🏗️ Engine Architecture & Refactoring Status (Dec 2025 Review)

### TypeScript Engine Architecture (Grade: A-)

The TypeScript rules engine demonstrates excellent domain-driven design with clear separation of concerns:

**Architecture Layers:**

```
┌─────────────────────────────────────────────────────────────┐
│ HOSTS (Backend GameEngine, Client Sandbox, Python AI)       │
├─────────────────────────────────────────────────────────────┤
│ ADAPTERS (TurnEngineAdapter, SandboxOrchestratorAdapter)    │
├─────────────────────────────────────────────────────────────┤
│ ORCHESTRATOR (turnOrchestrator.ts, phaseStateMachine.ts)    │
├─────────────────────────────────────────────────────────────┤
│ DOMAIN AGGREGATES (6 primary aggregates)                     │
├─────────────────────────────────────────────────────────────┤
│ SHARED HELPERS (63 files, ~22K LOC, pure functions)         │
├─────────────────────────────────────────────────────────────┤
│ CONTRACTS (schemas, serialization, validators, vectors)     │
└─────────────────────────────────────────────────────────────┘
```

**Domain Aggregates (5,553 LOC total):**
| Aggregate | LOC | Responsibility | Status |
|-----------|-----|----------------|--------|
| PlacementAggregate | 769 | Ring placement, validation, no-dead-placement | Complete |
| MovementAggregate | 739 | Non-capturing movement, reachability | Complete |
| CaptureAggregate | 918 | Overtaking captures, chain continuation | Complete |
| LineAggregate | 1199 | Line detection, collapse decisions | Complete |
| TerritoryAggregate | 1194 | Region detection, processing, elimination | Complete |
| VictoryAggregate | 734 | Victory evaluation, scoring, tiebreakers | Complete |

**Strengths:**

- ✅ Clear SSoT positioning with explicit documentation in every module
- ✅ All functions pure with no side effects; state immutable
- ✅ Full TypeScript typing with discriminated move types
- ✅ Runtime validation via Zod schemas
- ✅ No circular dependencies; clean dependency flow
- ✅ Single orchestrator entry point eliminates logic duplication

**Minor Improvement Opportunities:**

- Three files over 1000 LOC (turnOrchestrator, LineAggregate, TerritoryAggregate)
- Could split large aggregates into internal submodules for maintainability

### Python Engine Architecture (Grade: A)

The Python rules engine implements a well-architected 3-layer design:

**Architecture Layers:**

```
Layer 1: Core Models (app/models/core.py - 686 lines)
    │   Pydantic models mirroring TypeScript types
    │   Enums for GamePhase, MoveType, BoardType, AIType
    ↓
Layer 2: Board Management (app/board_manager.py - 854 lines)
    │   Static utilities for board-level queries
    │   Flood-fill for territory, line detection
    ↓
Layer 3: Game Engine (app/game_engine.py - 3843 lines)
    │   Move generation and validation
    │   Phase transitions via phase_machine.py
    └── Host adapter over canonical TS semantics
```

**Rules Engine Components:**

- **DefaultRulesEngine** (1064 lines): Delegates to GameEngine with shadow contract validation
- **Phase Machine** (224 lines): Pure phase + turn transitions mirroring TS phaseStateMachine
- **Validators** (5 files): PlacementValidator, MovementValidator, CaptureValidator, LineValidator, TerritoryValidator
- **Mutators** (6 files): Thin wrappers delegating to GameEngine methods

**Strengths:**

- ✅ Explicit SSoT policy in every module header
- ✅ Heavy use of Pydantic with frozen models and Protocol interfaces
- ✅ 120 test files (836 tests) covering unit/parity/invariants
- ✅ Shadow contract validation (mutator vs GameEngine comparison)
- ✅ Conservative defaults (slow territory path, mutators shadow-only)
- ✅ Environment flags for performance tuning (RINGRIFT_SKIP_SHADOW_CONTRACTS)

**Key Architectural Decisions:**

- Python is explicitly a **host adapter**, not an independent rules SSoT
- All rules semantics must match canonical TS engine exactly
- Zobrist hashing integrated for transposition tables and parity verification

### Cross-Engine Parity Status

**Contract Testing:** 54 vectors with 0 mismatches across:

- Placement, movement, capture, chain_capture
- Line detection, territory processing
- Forced elimination, hex edge cases
- Meta moves (swap_sides, multi-phase turns)

**Parity Test Infrastructure:**

- `tests/parity/` (TS): Backend vs Sandbox comparison
- `ai-service/tests/parity/` (Python): 16 test files for TS↔Python equivalence
- Replay-level parity via `selfplay-db-ts-replay.ts` and `check_ts_python_replay_parity.py`
- Golden replay framework in `tests/golden/` and `ai-service/tests/golden/`

### Canonical Rules Adherence

Both engines implement all major RR-CANON rules:

- **RR-CANON-R001-R003**: Board types, coordinates, adjacency
- **RR-CANON-R070-R076**: Turn phases and transitions
- **RR-CANON-R080-R103**: Placement, movement, capture semantics
- **RR-CANON-R120-R145**: Line and territory processing
- **RR-CANON-R170-R173**: Victory conditions
- **RR-CANON-R200-R209**: Active-no-moves and forced elimination

### Refactoring Recommendations

**Completed Refactoring:**

- ✅ Domain aggregate consolidation (6 aggregates)
- ✅ Canonical orchestrator implementation (100% rollout)
- ✅ Contract schema and serialization framework
- ✅ Adapter pattern for all hosts
- ✅ Legacy code removal (~1,176 lines in PASS20)

**Future Opportunities (Low Priority):**

1. Split large aggregates internally (LineAggregate, TerritoryAggregate)
2. Extract ANM resolution from turnOrchestrator to dedicated module
3. Consolidate Python phase transition logic into phase_machine.py
4. Document mutator-first mode decision criteria
