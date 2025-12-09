# RingRift Strategic Roadmap

> **Doc Status (2025-12-04): Active (roadmap & SLOs)**
>
> - Canonical phased roadmap and performance/scale SLO reference.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/architecture/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.
> - Relationship to goals: For the canonical statement of RingRift’s product/technical goals, v1.0 success criteria, and scope boundaries, see [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1). This roadmap operationalises those goals into phases, milestones, and SLOs and should be read as the **“how we plan to get there”** companion to [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1).

**Version:** 3.3
**Last Updated:** December 5, 2025
**Status:** Engine/Rules Beta (Orchestrator at 100% in CI, tests stabilized)
**Philosophy:** Robustness, Parity, and Scale

---

## 🎯 Executive Summary

**Current State:** Engine/Rules Beta with consolidated architecture. The 4-phase architecture remediation is complete:

- ✅ Canonical turn orchestrator in `src/shared/engine/orchestration/`
- ✅ Backend and sandbox adapters for gradual rollout
- ✅ Contract testing framework with 100% Python parity on 54 test vectors
- ✅ Extensive TypeScript and Python test suites validating rules, hosts, AI integration, and E2E flows (for up-to-date test counts and coverage metrics, see [`CURRENT_STATE_ASSESSMENT.md`](CURRENT_STATE_ASSESSMENT.md:1))

**Goal:** Production-Ready Multiplayer Game
**Timeline:** See [`PASS18C_ASSESSMENT_REPORT.md`](docs/archive/assessments/PASS18C_ASSESSMENT_REPORT.md) for concrete blockers and remediation plan
**Strategy:** Complete host integration parity → Enable orchestrator in production → Polish UX → Expand Multiplayer Features

> Note on rules authority: when there is any question about what the correct
> behaviour _should_ be (for example, when parity harnesses or engines
> disagree), the ultimate source of canonical truth is the rules
> documentation—[`ringrift_complete_rules.md`](ringrift_complete_rules.md)
> and, where applicable, [`ringrift_compact_rules.md`](ringrift_compact_rules.md).
> Code, tests, and parity fixtures are expected to converge toward those
> documents rather than redefine the rules.

---

## 🚀 Strategic Phases

### **PHASE 1: Core Playability (COMPLETED)** ✅

- [x] Game Engine & Rules (Movement, Capture, Lines, Territory)
- [x] Board Management (8x8, 19x19, Hex)
- [x] Basic AI Integration (Service + Fallback)
- [x] Frontend Basics (Board, Lobby, HUD, Victory)
- [x] Infrastructure (Docker, DB, WebSocket)

### **PHASE 1.5: Architecture Remediation (COMPLETED)** ✅

**Completed:** November 26, 2025

- [x] **Phase 1:** Created canonical turn orchestrator in `src/shared/engine/orchestration/`
- [x] **Phase 2:** Wired orchestrator to all 6 domain aggregates (Placement, Movement, Capture, Line, Territory, Victory)
- [x] **Phase 3:** Created backend adapter (`TurnEngineAdapter.ts`, 326 lines) and sandbox adapter (`SandboxOrchestratorAdapter.ts`, 476 lines)
- [x] **Phase 4:** Created Python contract test runner with 100% parity on 12 test vectors

**Key Deliverables:**

- Orchestrator entry points: `processTurn()`, `processTurnAsync()`
- Contract schemas and serialization for cross-language parity
- Feature flags for gradual rollout (`useOrchestratorAdapter`)
- Comprehensive documentation in `src/shared/engine/orchestration/README.md`

### **PHASE 2: Robustness & Testing (STABILIZED)**

**Priority:** P0 - CRITICAL
**Goal:** Ensure 100% rule compliance and stability
**Status:** All ~2,600+ tests passing. Orchestrator enabled at 100% in CI. Focus shifting to host integration parity.

#### 2.1 Comprehensive Scenario Testing

- [ ] Build test matrix for all FAQ edge cases (see `RULES_SCENARIO_MATRIX.md`)
- [ ] Implement scenario tests for complex chain captures (180° reversals, cycles)
- [ ] Verify all board types (especially Hexagonal edge cases)

#### 2.2 Sandbox Stage 2

- [x] Stabilize client-local sandbox with unified “place then move” turn semantics
      for both human and AI seats (including mixed games), and automatic local AI
      turns when it is an AI player’s move. Implemented in the browser-only
      sandbox via `ClientSandboxEngine` and the `/sandbox` path of `GamePage`,
      with coverage from `ClientSandboxEngine.mixedPlayers` tests.
- [ ] Ensure parity between backend and sandbox engines and improve AI-vs-AI
      termination behaviour using the sandbox AI simulation diagnostics
      (`ClientSandboxEngine.aiSimulation` with `RINGRIFT_ENABLE_SANDBOX_AI_SIM=1`),
      as tracked in P0.2 / P1.4 of `KNOWN_ISSUES.md`.

#### 2.2.1 Current P0 Focus (Dec 2025)

- **Engine/host lifecycle clarity:** Treat backend (`GameEngine` / `RuleEngine`),
  client sandbox (`ClientSandboxEngine`), and Python
  (`ai-service/app/game_engine.py`) strictly as adapters over the shared
  orchestrator/aggregates for advanced phases (`chain_capture`,
  `line_processing`, `territory_processing`, explicit self‑elimination). Any
  remaining host‑level rules logic should either call shared helpers or be
  clearly marked diagnostic/legacy.
- **WebSocket lifecycle & reconnection windows:** Tighten the canonical
  WebSocket API in `docs/CANONICAL_ENGINE_API.md` and keep reconnection,
  lobby, rematch, and spectator semantics covered by:
  `tests/integration/GameReconnection.test.ts`,
  `tests/integration/LobbyRealtime.test.ts`,
  `tests/e2e/reconnection.simulation.test.ts`, and related E2E slices.
- **TS↔Python territory / forced‑elimination parity:** Finish aligning
  territory detection, decision enumeration, and forced‑elimination sequences
  between TS and Python using contract vectors +
  `tests/unit/GameEngine.territoryDisconnection.test.ts`,
  `tests/unit/territoryDecisionHelpers.shared.test.ts`,
  and the Python parity suites under `ai-service/tests/`.

#### 2.3 Rules Engine Parity (Python/TS)

- [x] Implement `RulesBackendFacade` to abstract engine selection.
- [x] Implement `PythonRulesClient` for AI service communication.
- [x] Verify core mechanics parity (Movement, Capture, Lines, Territory) in Python engine.
- [ ] Enable `RINGRIFT_RULES_MODE=shadow` in staging/CI to collect parity metrics.

### **PHASE 3: Multiplayer Polish**

**Priority:** P1 - HIGH
**Goal:** Seamless online experience

#### 3.1 Spectator Mode

- [ ] UI for watching active games
- [ ] Real-time updates for spectators

#### 3.2 Social Features

- [ ] In-game chat
- [ ] User profiles and stats
- [ ] Leaderboards

#### 3.3 Matchmaking

- [ ] Automated matchmaking queue
- [ ] ELO-based matching

### **PHASE 4: Advanced AI**

**Priority:** P2 - MEDIUM
**Goal:** Challenging opponents for all skill levels

#### 4.1 Machine Learning

- [ ] Train neural network models
- [ ] Deploy advanced models to Python service

#### 4.2 Advanced Heuristics

- [ ] Implement MCTS/Minimax for intermediate levels

---

## 🔗 Alignment with TODO Tracks

The high-level phases above correspond to the more detailed execution tracks
and checklists in `TODO.md`:

- **Phase 2: Robustness & Testing** ↔ **Track 1** (Rules/FAQ Scenario Matrix &
  Parity Hardening) and parts of **Track 3** (Sandbox as a Rules Lab).
- **Phase 3: Multiplayer Polish** ↔ **Track 2** (Multiplayer Lifecycle &
  HUD/UX) and parts of **Track 5** (Persistence, Replays, and Stats).
- **Phase 4: Advanced AI** ↔ **Track 4** (Incremental AI Improvements &
  Observability) and **P2** items in `AI_IMPROVEMENT_PLAN.md`.

For day-to-day planning, treat `TODO.md` (including the
"Consolidated Execution Tracks & Plan" section) as the canonical, granular
list of tasks that roll up into these phases.

---

## 📊 Success Metrics for v1.0

These metrics restate and operationalise the v1.0 success criteria defined in [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1); if wording here ever appears to conflict, treat [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) as canonical and update this section to match.

1.  **Reliability:** >99.9% uptime, zero critical bugs.
2.  **Performance:** AI moves <1s, UI updates <16ms.
3.  **Engagement:** Users completing full games without errors.
4.  **Compliance:** 100% pass rate on rule scenario matrix.

---

## 📋 Implementation Roadmap (Post-Audit)

**P0: Architecture Production Hardening (Critical)**

- [x] **Complete architecture remediation (Phases 1-4)** – Canonical turn orchestrator, adapters, and contract tests are complete. See [`docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md`](docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md:1) for final status.

- [x] **Enable orchestrator adapters in staging/CI** – ✅ COMPLETE (PASS20)

- [x] **Enable orchestrator adapters in production** – ✅ COMPLETE (PASS20 - Configuration ready)

- [x] **Remove legacy turn processing code** – ✅ COMPLETE (PASS20: ~1,176 lines removed)

**P0: Rules Fidelity & Parity (Critical)**

- [x] **Fix forced elimination divergence for territory processing** – Align explicit `ELIMINATE_RINGS_FROM_STACK` moves and host-level forced elimination sequences between Python and TypeScript engines. Implementation complete; see [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1) and tests under [`ai-service/tests/test_territory_forced_elimination_divergence.py`](ai-service/tests/test_territory_forced_elimination_divergence.py:1) and [`ai-service/tests/test_generate_territory_dataset_smoke.py`](ai-service/tests/test_generate_territory_dataset_smoke.py:1). **Owner modes:** Debug + Code.

- [x] **Contract test framework for cross-language parity** – Contract test infrastructure complete with 54 test vectors across multiple categories and 100% Python parity. See [`tests/contracts/contractVectorRunner.test.ts`](tests/contracts/contractVectorRunner.test.ts:1) and [`ai-service/tests/contracts/test_contract_vectors.py`](ai-service/tests/contracts/test_contract_vectors.py:1).

- [ ] **Canonical replay/data gate (DB + goldens)** – Regenerate canonical DBs via `ai-service/scripts/generate_canonical_selfplay.py` (start with `canonical_square8.db`), archive parity/history gate summaries alongside the DBs, refresh the CI golden replay pack, and update `ai-service/TRAINING_DATA_REGISTRY.md` after gating.

- [ ] **Strengthen TS↔Python parity for territory detection and processing** – Expand parity suites to cover a matrix of board types and region patterns across [`src/shared/engine/territoryDetection.ts`](src/shared/engine/territoryDetection.ts:1), [`src/shared/engine/territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1), and the Python rules stack in [`ai-service/app/game_engine.py`](ai-service/app/game_engine.py:1) and [`ai-service/app/board_manager.py`](ai-service/app/board_manager.py:1). Include fixtures derived from [`RULES_SCENARIO_MATRIX.md`](RULES_SCENARIO_MATRIX.md:1) and existing Jest territory tests such as [`tests/unit/GameEngine.territoryDisconnection.test.ts`](tests/unit/GameEngine.territoryDisconnection.test.ts:1) and [`tests/unit/territoryDecisionHelpers.shared.test.ts`](tests/unit/territoryDecisionHelpers.shared.test.ts:1). **Owner modes:** Debug (test design) + Code (implementation).

- [ ] **Parity for territory decision enumeration and forced-elimination sequences** – Ensure that all territory-related `PlayerChoice` surfaces and generated territory-processing moves (claims, region order, explicit self-elimination, host-level forced elimination) stay in lockstep between TS and Python. Leverage `RINGRIFT_RULES_MODE=shadow` traces and extend parity diagnostics in [`src/server/utils/rulesParityMetrics.ts`](src/server/utils/rulesParityMetrics.ts:1). **Owner modes:** Debug + Code.

- [ ] **Property-based tests for territory invariants and forced elimination** – Introduce property-based tests (for example, with Hypothesis in Python and fast-check in TypeScript) that randomly generate mid/late-game `GameState` snapshots and assert invariants around territory connectivity, collapsed-space ownership, and forced elimination ordering. Ground properties in [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1) and the territory helpers listed in [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1). An initial TS harness exists under `tests/unit/territoryProcessing.property.test.ts` exercising 2×2 disconnected-region invariants on `square8`; Python/Hypothesis coverage and forced-elimination–specific properties remain future work. **Owner modes:** Debug (property design) + Code (harnesses).

- [x] **Dataset-level validation for territory / combined-margin training data** – Add validation passes for datasets produced by [`ai-service/app/training/generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1), checking target ranges, per-player combined margins consistency, and metadata completeness (`engine_mode`, `num_players`, `ai_type_pN`, `ai_difficulty_pN`). Initial validation helpers and tests live in [`ai-service/app/training/territory_dataset_validation.py`](ai-service/app/training/territory_dataset_validation.py:1) and [`ai-service/tests/test_territory_dataset_validation.py`](ai-service/tests/test_territory_dataset_validation.py:1), and are documented in [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1). CI wiring for full training pipelines remains future work. **Owner modes:** Debug + Code (validation helpers implemented).

- [ ] **Rules observability and divergence diagnostics** – Expand logging and metrics around territory phases and forced elimination, including structured events for explicit vs host-level eliminations and a clear taxonomy of divergence causes. Build on existing parity counters in [`src/server/utils/rulesParityMetrics.ts`](src/server/utils/rulesParityMetrics.ts:1) and Python-side logging in [`ai-service/app/rules/default_engine.py`](ai-service/app/rules/default_engine.py:1). **Owner modes:** Debug + Code + Architect.

**P1: AI Robustness, Training & Intelligence (High Value)**

- [x] **Grafana dashboards for observability and monitoring** – ✅ COMPLETE (PASS21: 3 dashboards with 22 panels)
  - `game-performance.json` – Core game metrics, AI latency, abnormal terminations
  - `rules-correctness.json` – Parity and correctness metrics
  - `system-health.json` – HTTP, WebSocket, AI service health, infrastructure metrics

- [x] **k6 load testing framework** – ✅ COMPLETE (PASS21: 4 production-scale scenarios)
  - Scenario P1: Mixed human vs AI ladder (40-60 players, 20-30 moves)
  - Scenario P2: AI-heavy concurrent games (60-100 players, 10-20 AI games)
  - Scenario P3: Reconnects and spectators (40-60 players + 20-40 spectators)
  - Scenario P4: Long-running AI games (10-20 games, 60+ moves)

- [x] **Monitoring stack by default** – ✅ COMPLETE (PASS21: Moved from optional profile to standard deployment)

- [ ] **Wire up Minimax/MCTS in the production ladder** – Audit and stabilise advanced AI implementations in [`ai-service/app/ai/minimax_ai.py`](ai-service/app/ai/minimax_ai.py:1) and [`ai-service/app/ai/mcts_ai.py`](ai-service/app/ai/mcts_ai.py:1), and the difficulty ladder in [`ai-service/app/main.py`](ai-service/app/main.py:1), then expose them through the canonical presets in [`src/server/game/ai/AIEngine.ts`](src/server/game/ai/AIEngine.ts:1). Ensure new types are covered by `/ai/move` tests and smoke AI-vs-AI runs. **Owner modes:** Code.

- [x] **Fix and document RNG determinism across TS and Python** – Implement and validate per-game seeding for AI decisions and rules randomness, aligned between Node and Python, so that mixed-mode runs and dataset generation are reproducible from a single seed. Contract captured in [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1) (§5) and exercised by seeded determinism tests in the TS and Python suites. **Owner modes:** Debug + Code + Architect (complete).

- [ ] **AI move rejection and fallback hardening** – Implement a tiered fallback system for invalid or timed-out AI moves in [`src/server/game/ai/AIEngine.ts`](src/server/game/ai/AIEngine.ts:1) and [`src/server/services/AIServiceClient.ts`](src/server/services/AIServiceClient.ts:1), with clear metrics and logging for each fallback path. Coordinate with the AI service's `/ai/move` error taxonomy in [`ai-service/app/main.py`](ai-service/app/main.py:1). **Owner modes:** Code + Debug.

- [ ] **Integrate territory / combined-margin datasets into heuristic or NN training** – Formalise training scripts that consume datasets from [`ai-service/app/training/generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1) and feed them into [`ai-service/app/training/train_heuristic_weights.py`](ai-service/app/training/train_heuristic_weights.py:1) and future NN pipelines such as [`ai-service/app/ai/neural_net.py`](ai-service/app/ai/neural_net.py:1). Document expected input schema and evaluation loops. **Owner modes:** Code + Architect.

**P2: UX Polish & Multiplayer (User Experience)**

- [ ] **HUD improvements:** Add phase indicators, timers, and better state visualization.
- [ ] **Victory modal:** Wire up `game_over` events to the victory modal.
- [ ] **Lobby game list:** Display waiting games correctly in the lobby.

## 🗄️ Database Operations &amp; Migrations (Ops Playbook)

For environment-specific database expectations, Prisma migration workflow, and backup/rollback procedures, see [docs/OPERATIONS_DB.md](docs/OPERATIONS_DB.md:1).

## 🔐 Security & Threat Model (S-05)

RingRift’s security posture and hardening plan are defined in the canonical threat model document:

- **Threat model & scope:** Assets, trust boundaries, attacker profiles, and major threat surfaces (auth, game access control, validation, abuse/DoS, data protection, supply chain) are documented in [`docs/SECURITY_THREAT_MODEL.md`](docs/SECURITY_THREAT_MODEL.md:1).
- **Current controls vs gaps:** Each surface maps concrete threats to existing controls (for example, [`auth`](src/server/routes/auth.ts:1), [`WebSocketServer`](src/server/websocket/server.ts:1), [`WebSocketPayloadSchemas`](src/shared/validation/websocketSchemas.ts:1), [`rateLimiter`](src/server/middleware/rateLimiter.ts:1), [`config`](src/server/config.ts:1)) and identifies documented risks.
- **Security backlog (S-05.A–F):** A prioritized, implementation-ready backlog for future work (auth & token lifecycle, game/session authZ, abuse & quotas, validation/logging hygiene, data retention & privacy, supply chain & CI hardening) that should be scheduled alongside P-0/P-1 items when planning production launches.
- **Data lifecycle & privacy (S-05.E):** The concrete data inventory, retention/anonymization policies, and account deletion/export workflows are designed in [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](docs/DATA_LIFECYCLE_AND_PRIVACY.md:1), referenced from the S-05.E backlog item.
- **Supply chain & CI/CD safeguards (S-05.F):** The supply-chain & CI/CD threat overview, current controls/gaps, and S-05.F.x implementation tracks are designed in [`docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`](docs/SUPPLY_CHAIN_AND_CI_SECURITY.md:1).

Treat [`docs/SECURITY_THREAT_MODEL.md`](docs/SECURITY_THREAT_MODEL.md:1) as the **single source of truth** for the S-05 backlog; the detailed data lifecycle and privacy design for S-05.E lives in [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](docs/DATA_LIFECYCLE_AND_PRIVACY.md:1).

## ⚙️ Performance &amp; Scalability (P-01)

This section defines the initial performance and scalability plan for RingRift prior to first production launch. It is intended to be **implementation-guiding**, not tool-specific, and should remain compatible with the existing architecture, metrics, and tests.

### 1. Scope, assumptions, and target scale

**Architecture assumptions**

- Single-region deployment for both staging and production.
- Stateless Node.js backend (HTTP + WebSocket) with Redis and Postgres, following the topology described in [`docs/OPERATIONS_DB.md`](docs/OPERATIONS_DB.md:1) and [`docker-compose.yml`](docker-compose.yml:1).
- Python AI service deployed alongside the backend, exporting Prometheus metrics via [`ai-service/app/metrics.py`](ai-service/app/metrics.py:1).
- Existing Node-side metrics from [`rulesParityMetrics`](src/server/utils/rulesParityMetrics.ts:1) and structured logging via [`logger`](src/server/utils/logger.ts:1).

**Target scale for P-01 validation**

These SLOs are defined under **nominal load**, not worst-case spikes:

- **Staging**
  - Up to ~20 concurrent active games (40–60 connected players).
  - AI enabled for a subset of games (30–60% of seats using AI).
- **Production (initial launch)**
  - Up to ~100 concurrent active games (200–300 connected players).
  - 30–70% of seats may be AI-controlled depending on lobby settings.
  - Up to ~50 additional spectator connections.

**Load-test tooling (IMPLEMENTED - PASS21)**

The k6 load testing framework has been implemented with:

- Support both HTTP and WebSocket traffic.
- Allow scenario scripting (login → create/join game → play moves → resign/finish).
- Allow configurable virtual users, ramp-up, and steady-state durations.
- Export per-endpoint and per-operation latency distributions (including p95/p99) and error counts.
  - Integrate with the existing orchestrator HTTP load smoke (`scripts/orchestrator-load-smoke.ts`, exposed via `npm run load:orchestrator:smoke`) and metrics/observability smoke (`tests/e2e/metrics.e2e.spec.ts`) as lightweight entry points, so that SLO validation can reuse the same `/api` and `/metrics` surfaces used in day‑to‑day smokes and runbooks.

### 2. Environment-aware SLOs

SLOs below are **per environment** and are defined for the target scale above. Numbers are intentionally conservative but realistic for a well-tuned single-region stack.

#### 2.1 HTTP API SLOs

Tracked via the load tool’s timing plus HTTP status codes; if HTTP latency histograms are later added via `prom-client`, they should use the same thresholds.

**Staging (nominal load)**

- `POST /api/auth/login`
  - p95 ≤ **500 ms**, p99 ≤ **1000 ms**.
  - 5xx rate &lt; **1.0%** of requests over any 15-minute window.
- `POST /api/games` (create game)
  - p95 ≤ **800 ms**, p99 ≤ **1500 ms**.
  - 5xx rate &lt; **1.0%** over 15 minutes.
- `GET /api/games/:gameId` (fetch game state)
  - p95 ≤ **400 ms**, p99 ≤ **800 ms**.
  - 5xx rate &lt; **1.0%** over 15 minutes.

**Production (initial launch)**

- `POST /api/auth/login`
  - p95 ≤ **250 ms**, p99 ≤ **500 ms**.
  - 5xx rate &lt; **0.5%** of requests over any rolling 15-minute window.
- `POST /api/games`
  - p95 ≤ **400 ms**, p99 ≤ **800 ms**.
  - 5xx rate &lt; **0.5%** over 15 minutes.
- `GET /api/games/:gameId`
  - p95 ≤ **200 ms**, p99 ≤ **400 ms**.
  - 5xx rate &lt; **0.5%** over 15 minutes.

#### 2.2 WebSocket gameplay SLOs

These SLOs are defined for **move-to-acknowledgement latency** and are validated via:

For the canonical decision on player move transport (WebSocket vs HTTP), see [`PLAYER_MOVE_TRANSPORT_DECISION.md`](docs/PLAYER_MOVE_TRANSPORT_DECISION.md:1). WebSocket is the authoritative move channel for interactive clients; any HTTP move endpoint is an internal/test harness over the same shared domain API.

- Client-side timing in the load tool (emit move → receive authoritative broadcast/ack).
- Server-side metrics:
  - `game_move_latency_ms` histogram in [`rulesParityMetrics`](src/server/utils/rulesParityMetrics.ts:1).
  - `websocket_connections_current` gauge in the same module.

**Human move submission → authoritative broadcast**

- **Staging**
  - 95% of moves: end-to-end latency ≤ **300 ms**.
  - 99% of moves: end-to-end latency ≤ **600 ms**.
  - `game_move_latency_ms` (phase label for normal moves) p95 ≤ **200 ms**, p99 ≤ **400 ms** under ≤20 concurrent games.
- **Production**
  - 95% of moves: end-to-end latency ≤ **200 ms**.
  - 99% of moves: end-to-end latency ≤ **400 ms**.
  - `game_move_latency_ms` p95 ≤ **150 ms**, p99 ≤ **300 ms** under ≤100 concurrent games.

**Stall definition (human moves)**

- A human move is considered **stalled** if no server ack/broadcast is observed within **2 seconds**.
- Stall rate (moves taking &gt;2 seconds) should remain:
  - ≤ **0.5%** in staging load tests.
  - ≤ **0.2%** in production load tests.

#### 2.3 AI turn SLOs

AI turns span both the Node backend and the Python AI service.

**Relevant metrics**

- Node backend:
  - `ai_move_latency_ms` histogram in [`rulesParityMetrics`](src/server/utils/rulesParityMetrics.ts:1).
  - `ai_fallback_total` counter (labeled by `reason`).
- Python AI service:
  - `AI_MOVE_LATENCY` histogram and `AI_MOVE_REQUESTS` counter in [`ai-service/app/metrics.py`](ai-service/app/metrics.py:1).

**CPU-only baseline (default expectation)**

Per **AI-service instance**, under up to **10 concurrent AI games** (roughly 10 in-flight `/ai/move` requests):

- **Staging**
  - `AI_MOVE_LATENCY` p95 ≤ **1.5 s**, p99 ≤ **3.0 s**.
  - `ai_move_latency_ms` p95 ≤ **1700 ms**, p99 ≤ **3200 ms**.
- **Production**
  - `AI_MOVE_LATENCY` p95 ≤ **1.0 s**, p99 ≤ **2.0 s**.
  - `ai_move_latency_ms` p95 ≤ **1200 ms**, p99 ≤ **2500 ms**.

**GPU-enabled or higher-spec instances**

- Targets above become **ceilings**; GPU-enabled deployments should typically see:
  - `AI_MOVE_LATENCY` p95 ≤ **0.5 s**.
  - `ai_move_latency_ms` p95 ≤ **700 ms**.
- SLOs may be tightened once empirical data is available.

**End-to-end AI turn latency (client perspective)**

- From “AI turn starts” (server signals AI is thinking) to authoritative move broadcast:
  - **Staging:** 95% of AI turns ≤ **3.0 s**, 99% ≤ **5.0 s**.
  - **Production:** 95% of AI turns ≤ **2.0 s**, 99% ≤ **4.0 s**.
- AI turns that take &gt;5 seconds should be rare (&lt; **1%** of AI turns) and should generally coincide with logged fallbacks or dependency issues.

**AI fallback SLO**

- Across any P-01 load test, `ai_fallback_total` increases by at most:
  - **1%** of total AI moves in staging.
  - **0.5%** of total AI moves in production.
- Fallbacks labeled as timeout or upstream error should be investigated before release.

#### 2.4 Availability and error budgets

High-level monthly targets (per environment):

- **Core gameplay surfaces**
  - Includes: WebSocket gameplay, `POST /api/auth/login`, `POST /api/games`, `GET /api/games/:gameId`.
  - **Production:** target availability **≥ 99.5%** (error budget 0.5% of requests or connections failing due to 5xx/transport errors).
  - **Staging:** no formal SLO, but during P-01 validation 5xx/transport error rates in load tests should meet or beat production thresholds.
- **Non-critical endpoints** (for example, future leaderboard or profile extras)
  - Target availability **≥ 99.0%**; brief degradations here should not block a release as long as core gameplay SLOs are met.

Availability is evaluated from:

- HTTP status codes and connection failures during load tests.
- WebSocket connection stability (drop/reconnect rates) observed via both the load tool and `websocket_connections_current`.

### 3. Canonical synthetic load scenarios

The following **3–4 reusable scenarios** are intended to be implemented once (as configurable scripts) and reused across environments.

#### Scenario P1: Mixed human vs AI ladder (baseline)

**Status:** ✅ Implemented (k6 framework, PASS21)

**Intent**

Validate HTTP and WebSocket SLOs for the most common flows (login, lobby, game creation/join, mixed human/AI play).

**Traffic model**

- 40–60 virtual players.
- Rough mix:
  - 70% of players log in, create a game, and wait for a human or AI opponent.
  - 30% join existing games.
- Game composition:
  - ~50% human vs AI.
  - ~50% human vs human.
- Each game plays through **20–30 moves** before resign or natural termination.

**Duration &amp; ramp**

- 5-minute ramp up from 0 to target player count.
- 15-minute steady-state period.

**Metrics &amp; signals**

- HTTP:
  - Latency and 5xx rate for `POST /api/auth/login`, `POST /api/games`, `GET /api/games/:gameId`.
- WebSocket:
  - Client-observed move latency (emit → ack/broadcast).
  - `game_move_latency_ms` and `websocket_connections_current`.
- AI:
  - `ai_move_latency_ms`, `ai_fallback_total`.
  - `AI_MOVE_LATENCY` and `AI_MOVE_REQUESTS`.
- Logs:
  - HTTP error logs (5xx) with `requestId`.
  - WebSocket disconnect/reconnect logs.
  - Any AI timeout or fallback error codes.

#### Scenario P2: AI-heavy concurrent games

**Status:** ✅ Implemented (k6 framework, PASS21)

**Intent**

Stress the AI service and backend integration where most seats are AI-controlled, validating AI SLOs and observing degradation behaviour.

**Traffic model**

- 60–100 virtual players, primarily creating **human vs AI** or **AI vs AI** games.
- At least **70% of seats** are AI-controlled.
- Games are shorter (10–20 moves) but many games overlap to maintain **10–20 concurrent AI games per AI-service instance**.

**Duration &amp; ramp**

- 5-minute ramp to target concurrency.
- 15–20-minute steady-state.

**Metrics &amp; signals**

- AI:
  - `AI_MOVE_LATENCY` and `ai_move_latency_ms` p95/p99.
  - `ai_fallback_total` by `reason`.
- WebSocket:
  - End-to-end AI turn latency (tool-measured).
  - `websocket_connections_current` to confirm expected player counts.
- HTTP:
  - `POST /api/games` latency and 5xx rate under AI-heavy load.
- Logs:
  - AI timeout, invalid-move, or upstream-error codes from the backend.
  - Any evidence of circuit-breakers, retries, or degraded AI behaviour.

#### Scenario P3: Reconnects and spectators

**Status:** ✅ Implemented (k6 framework, PASS21)

**Intent**

Validate WebSocket resilience, reconnection logic, and read-heavy traffic patterns from spectators.

**Traffic model**

- 40–60 virtual players actively playing in ~20–30 games.
- 20–40 additional spectator connections joining existing games (read-only).
- Each active player:
  - Plays 10–20 moves.
  - Intentionally disconnects and reconnects **1–3 times** per game.
- Spectators:
  - Join and leave games at a low churn rate (for example, every 2–3 minutes).

**Duration &amp; ramp**

- 5-minute ramp to full connections.
- 10–15-minute steady-state.

**Metrics &amp; signals**

- WebSocket:
  - `websocket_connections_current` stability (no sawtooth behaviour beyond expected reconnects).
  - `game_move_latency_ms` under reconnect churn.
- HTTP:
  - Any supporting HTTP calls for reconnection or game state resync (`GET /api/games/:gameId`) and their 5xx rates.
- Logs:
  - Connection and reconnection log messages keyed by `gameId` and `userId`.
  - Errors related to stale sessions or authorization failures on reconnect.

#### Scenario P4 (optional): Long-running AI games

**Status:** ✅ Implemented (k6 framework, PASS21)

**Intent**

Detect slow memory leaks, performance drift, or parity issues during long AI-heavy sessions.

**Traffic model**

- 10–20 long-running games, mostly human vs AI or AI vs AI.
- 60+ moves per game (deep into mid/late game).

**Duration**

- 30–60 minutes, low concurrency but long wall-clock time.

**Metrics &amp; signals**

- AI:
  - Drift in `AI_MOVE_LATENCY` / `ai_move_latency_ms` over time.
- WebSocket:
  - Stability of `websocket_connections_current` and move latencies late in game life.
- Rules parity:
  - Any unexpected spikes in `rules_parity_*` counters under load (if shadow rules mode is enabled).

### 4. Interpreting metrics and logs under load

Use these **checklists** when running any P-01 scenario.

#### 4.1 HTTP SLO checklist

- [ ] For each key endpoint, compute p95/p99 latency from the load tool.
  - Compare against the environment-specific thresholds in section 2.1.
- [ ] Compute 5xx rate over the steady-state window.
  - Should be below 0.5–1.0% depending on environment.
- [ ] Sample a few worst-case requests:
  - Use `requestId` in logs (via [`logger`](src/server/utils/logger.ts:1)) to find matching log entries.
  - Check for DB timeouts, Redis errors, or AI upstream errors.

#### 4.2 WebSocket gameplay checklist

- [ ] Check `game_move_latency_ms` histogram:
  - p95 and p99 for normal human moves should be within SLO.
- [ ] From the load tool, compute end-to-end move latency distributions:
  - Flag any spikes &gt;2 seconds for human moves.
- [ ] Inspect `websocket_connections_current`:
  - Confirm it matches expected virtual user counts.
  - Look for unexpected drops indicating server-side disconnects.

#### 4.3 AI service checklist

- [ ] Check `AI_MOVE_LATENCY` and `ai_move_latency_ms`:
  - p95/p99 within environment and hardware-specific SLOs.
  - No significant upward drift over the duration of a test.
- [ ] Compute fallback rate from `ai_fallback_total` and `AI_MOVE_REQUESTS`:
  - Ensure ratios stay under the thresholds in section 2.3.
- [ ] In logs, inspect representative slow/failed AI requests:
  - Use `gameId` and any AI-related error codes to distinguish timeouts, invalid moves, and upstream failures.

#### 4.4 Parity and dependency health

- [ ] If rules shadow mode is enabled, watch `rules_parity_*` counters:
  - Large spikes under load may indicate logic or dependency issues rather than pure performance problems.
- [ ] Correlate slow or failed operations with:
  - AI timeouts or fallbacks.
  - DB connection errors or migration issues (see [`docs/OPERATIONS_DB.md`](docs/OPERATIONS_DB.md:1)).
  - WebSocket transport errors.

### 5. Lifecycle integration

#### 5.1 Pre-launch performance gate

Before first production launch (and before any major gameplay/AI change):

1. **Deploy candidate build** to staging or a dedicated perf environment that mirrors production topology (backend, AI service, DB, Redis).
2. **Warm up** the stack with a small P1 run (5–10 minutes at half target load).
3. **Run full P1, P2, and P3 scenarios** at target load:
   - Use at least 10–15 minutes of steady state for each scenario.
4. **Evaluate SLOs**:
   - All HTTP, WebSocket, AI, and availability SLOs from section 2 must be met with a reasonable margin (&gt;10–20% headroom where possible).
5. **Review logs**:
   - Confirm absence of repeated AI timeouts, DB errors, or WebSocket disconnect storms.
6. **Record results** in a short checklist or ticket linked from the release notes.

If any core SLO is violated (especially WebSocket gameplay or AI SLOs), treat this as a **release blocker** until root cause is understood and a mitigation is in place.

#### 5.2 Post-release and regression checks

- **Regular cadence**
  - Run a reduced P1 scenario (for example, half scale and shorter duration) at least **weekly** against staging.
  - Run P2 on-demand after significant AI or rules-engine changes.
  - Run P3 after changes to connection handling, session logic, or lobby flows.
- **On-change triggers**
  - Any change to AI evaluation logic, move selection, or Python service topology should trigger at least one P2 run.
  - Any change to WebSocket handling or game/session orchestration should trigger P1 + P3.

If SLOs regress:

1. Identify which SLOs are broken (HTTP vs WebSocket vs AI).
2. Use the metric and log checklists to attribute cause:
   - AI latency/fallbacks vs DB latency/errors vs WebSocket transport issues.
3. File and prioritize issues accordingly (P0 if core gameplay is affected).

#### 5.3 Scaling and capacity planning hooks

Use P-01 scenarios as the **baseline capacity model**:

- When observed live traffic approaches the tested concurrency (for example, 100 concurrent games), repeat P1 and P2 at higher target loads (for example, 150–200 concurrent games) to determine safe headroom.
- If AI latency or fallback rates degrade first:
  - Scale out AI-service instances and/or introduce per-game AI concurrency limits.
  - Consider adjusting AI difficulty or evaluation depth for lower-cost modes.
- If HTTP or WebSocket SLOs degrade first:
  - Scale backend instances, revisit DB connection pooling, or optimize hot endpoints.
- Periodically revisit SLOs:
  - Tighten thresholds once empirical data shows sustained performance gains.
  - Document any major updates to SLOs or scenarios in this section and cross-reference in [`docs/INDEX.md`](docs/INDEX.md:1).
