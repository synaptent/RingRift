# RingRift Project Goals

> **Doc Status (2025-12-13): Active (project direction SSoT)**
>
> - Single authoritative source for product/technical goals and non-goals.
> - Not a rules or lifecycle SSoT; for rules semantics defer to `ringrift_complete_rules.md` + `RULES_CANONICAL_SPEC.md` + shared TS engine, and for lifecycle semantics defer to `docs/architecture/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.
>
> **Version:** 1.1  
> **Created:** 2025-11-26  
> **Last Updated:** 2025-12-13  
> **Status:** Authoritative source for project direction
>
> This document is the **single authoritative source** for RingRift’s product and technical goals. When other documents conflict with this one regarding project direction or success criteria, this document takes precedence.
>
> It is intentionally goal-focused: it describes **what we are trying to achieve**, not the live implementation status or detailed rules semantics.

---

## 1. Source of truth and related documents

- **Goals SSoT (this file):** Canonical statement of product experience goals and high-level technical objectives (engine SSOT, parity, SLOs). Use this document as the primary reference for _what success looks like_.
- **Implementation status & metrics SSoT:** For current implementation status, test counts, and coverage metrics, defer to [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1). That file is the single source of truth for live numbers and status labels.
- **Rules-specific status:** For rules-focused status and navigation across rules docs and verification/audit reports, see [`docs/rules/CURRENT_RULES_STATE.md`](docs/rules/CURRENT_RULES_STATE.md:1).
- **Specialised goal surfaces (subordinate docs):**
  - [`AI_ARCHITECTURE.md`](docs/architecture/AI_ARCHITECTURE.md:1) – AI-specific goals, architecture, and training/improvement plans, all subordinate to the product/technical goals defined here.
  - [`docs/ai/AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:1) – Canonical AI training data and pipeline objectives; describes how training/eval must respect the goals and SLOs in this file.
  - [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1) – Phased execution plan and SLO roadmap that operationalises the goals in this file; it should be read as an implementation roadmap, not as a separate goals SSoT.

---

## 2. Vision / Outcome (Purpose & Vision)

RingRift exists to deliver a polished, production-ready online implementation of the RingRift tabletop strategy game as a high-quality environment for human and AI players to explore deep, perfect-information multiplayer gameplay.

RingRift is a web-based multiplayer abstract strategy game where 2–4 players compete for territory and ring elimination through strategic stack building, marker placement, and tactical captures. The game features no chance elements and perfect information, with a focus on multi-player dynamics that create shifting alliances and emergent strategic depth. Our goal is to deliver a polished, production-ready online implementation that faithfully captures the richness of the tabletop experience while leveraging digital capabilities for AI opponents, real-time multiplayer, and accessible cross-platform play.

### 2.1 Ruleset design goals & rationale

The RingRift ruleset is intentionally designed around a **modest number of simple, universal mechanics**—stack building and movement, overtaking vs elimination captures, line formation and collapse, territory disconnection, and last-player-standing turn semantics. These mechanics are combined to achieve the following design goals:

- **High emergent complexity from simple rules.**  
  The core rules are meant to be teachable and mechanically consistent across board types and player counts, while still producing a rich decision space with deep tactical and strategic play. Complexity arises from the interaction of stacking, movement constraints, capture chains, line rewards, and territory disconnection, not from rule bloat. This is the primary reason RingRift exists as a distinct abstract strategy game rather than reusing an existing design.

- **Exciting, tense, and strategically non-trivial games.**  
  Games should remain **live and contested** for a long time: temporary leads in ring count or territory are deliberately not a straightforward proxy for eventual victory. Multiple victory paths (ring elimination, territory control, last-player-standing) plus multi-player dynamics and chain reactions are designed so that:
  - Seemingly “won” positions can still collapse through territory cascades or alliance shifts.
  - Short-term sacrifices and self-elimination can be correct play toward long-term advantage.
  - The board’s geometry (lines and region disconnections) constantly reshapes incentives and risk.

- **Human–computer competitive balance in a perfect-information, zero-sum game.**  
  Unlike many classic perfect-information strategy games where engines quickly outclass humans, RingRift is explicitly tuned so that strong human players can **compete with and sometimes outplay strong AIs**:
  - The default 3-player configuration and multi-player variants create social and political dynamics (temporary alliances, leader-punishing behaviour) that are difficult for purely algorithmic agents to model.
  - The extremely high branching factor (up to millions of choices per turn), long tactical chains (especially in captures and territory disconnections), and subtle tradeoffs between rings, markers, and territory make exhaustive search prohibitively expensive even for well-optimized engines.
  - The rules support a spectrum of AI strengths while intentionally preserving room for human creativity, intuition, long-term strategic play, coalition forming, and non-myopic tactical planning to matter at all skill levels.

Together, these goals define **how the game should feel**: simple to describe at the rules level, but with deep, emergent strategy; high tension and comeback potential; and a long-term target where humans and AIs can meaningfully co-exist as competitive opponents. Detailed rules semantics, examples, and strategy notes are defined in the authoritative rulebook [`docs/rules/COMPLETE_RULES.md`](docs/rules/COMPLETE_RULES.md:1) and the canonical specification [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1); this section records the _purpose_ those rules are serving.

---

## 3. Core objectives for current phase (v1.0)

> For the current v1.0 phase, these objectives describe what the project must deliver in terms of gameplay features, architecture, and quality. Read them together with [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1) for the phased execution plan and [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1) for factual implementation status.

### 3.0 Core objectives summary

1. Deliver a rules-complete online implementation of RingRift across the three supported board types (8×8, 19×19, hexagonal) with correct resolution of all victory conditions for 2–4 player games.
2. Provide a stable, responsive multiplayer experience over WebSocket and HTTP with real-time game state synchronisation and spectator support suitable for public beta and production use.
3. Maintain a single shared TypeScript rules engine reused by backend and client sandbox hosts, with the Python AI service kept in deterministic parity via contracts, tests, and observability.
4. Offer AI opponents across a 1–10 difficulty ladder integrated through the Python AI service and resilient local fallbacks so that AI-controlled seats never stall games.
5. Achieve v1.0 readiness on reliability, performance, and quality by meeting the documented SLOs and test coverage targets while keeping the project maintainable and observable in production.

These objectives naturally cluster into four areas that later documents use for more detailed planning and assessment:

- **Rules/engine correctness and canonical parity** (Objectives 1 & 3)
- **AI training & evaluation foundations** (Objectives 3 & 4, with details in AI docs)
- **UX/teaching & rules‑UX telemetry** (Objectives 1, 2, and quality goals below)
- **Production readiness & operations** (Objectives 2 & 5)

### 3.1 Product objectives (gameplay and UX)

| Objective                          | Description                                                                                                                                                                                                            | Rationale                                                      |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Complete rules implementation**  | Faithfully implement all game mechanics from [`docs/rules/COMPLETE_RULES.md`](docs/rules/COMPLETE_RULES.md:1) including movement, captures, chains, lines, territory disconnection, and all victory conditions         | Core game value depends on correct rules                       |
| **Multiple board types**           | Support 8×8 square (64 spaces), 19×19 square (361 spaces), and hexagonal (469 spaces) boards                                                                                                                           | Provides accessibility gradient and geometric variety          |
| **Flexible player configurations** | Support 2–4 players with any combination of human and AI participants                                                                                                                                                  | Enables solo practice, competitive play, and social gaming     |
| **AI opponents**                   | Provide AI opponents at multiple difficulty levels (1–10) with responsive move selection                                                                                                                               | Allows single-player experience and fills seats in multiplayer |
| **Real-time multiplayer**          | WebSocket-based live gameplay with synchronized state and spectator support                                                                                                                                            | Core online multiplayer experience                             |
| **Victory tracking**               | Correct implementation of all three victory paths: Ring Elimination (victoryThreshold per RR‑CANON‑R061), Territory Control (>50% board), Last Player Standing                                                         | Game cannot be complete without proper resolution              |
| **Rules teaching & UX clarity**    | Provide in-client teaching flows, weird-state explanations, and end-of-game explanations aligned with [`docs/ux/UX_RULES_TEACHING_SCENARIOS.md`](docs/ux/UX_RULES_TEACHING_SCENARIOS.md:1) and `gameEndExplanation.ts` | Ensures complex mechanics are learnable and understandable     |

### 3.2 Technical objectives (architecture, parity, performance, AI training)

| Objective                          | Description                                                                                                                                                                                         | Rationale                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Canonical rules engine**         | Single shared rules engine in TypeScript used by both backend and client sandbox                                                                                                                    | Eliminates divergence, simplifies maintenance                   |
| **Cross-language parity**          | Python AI service rules engine matches TypeScript implementation behavior                                                                                                                           | Enables ML training on same rules as production                 |
| **Consolidated architecture**      | Turn orchestrator pattern with domain aggregates and host adapters                                                                                                                                  | Clean separation of concerns, testability                       |
| **Performance targets**            | AI moves <1s, UI updates <16ms, game state sync <200ms (see SLOs in §4.1)                                                                                                                           | Responsive user experience                                      |
| **Scalable infrastructure**        | Docker-based deployment with PostgreSQL, Redis, and separate AI service                                                                                                                             | Production-ready operations                                     |
| **Canonical AI training pipeline** | Maintain an AI training and evaluation pipeline that uses only canonical replay data, enforces TS↔Python parity gates, and produces reproducible models (see `docs/ai/AI_TRAINING_AND_DATASETS.md`) | Ensures AI improvements respect rules correctness and SLO goals |

### 3.3 Quality objectives (testing, reliability, rules‑UX telemetry)

| Objective                         | Description                                                                                                                                                                                                                                                                                                          | Rationale                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Comprehensive test coverage**   | Extensive automated test suites across TypeScript and Python (unit, integration, scenario, parity, and E2E); see [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1) for live counts and coverage details.                                                                        | Confidence in correctness                                     |
| **Rules/FAQ scenario matrix**     | Test cases derived directly from [`docs/rules/COMPLETE_RULES.md`](docs/rules/COMPLETE_RULES.md:1) FAQ examples                                                                                                                                                                                                       | Ensures rules fidelity                                        |
| **Contract testing**              | Cross-language parity validated by shared contract test vectors                                                                                                                                                                                                                                                      | Guarantees engine consistency                                 |
| **Parity harnesses**              | Backend ↔ sandbox ↔ Python engine behavior validation                                                                                                                                                                                                                                                                | Catches divergence early                                      |
| **CI/CD pipeline**                | Automated testing, linting, security scanning on every change                                                                                                                                                                                                                                                        | Maintains quality bar                                         |
| **Rules‑UX telemetry & teaching** | Instrument key rules interactions (ANM, forced elimination, weird states, territory cascades) and maintain teaching overlays and scenarios per [`docs/ux/UX_RULES_TELEMETRY_SPEC.md`](docs/ux/UX_RULES_TELEMETRY_SPEC.md:1) and [`docs/ux/UX_RULES_TEACHING_SCENARIOS.md`](docs/ux/UX_RULES_TEACHING_SCENARIOS.md:1) | Observability into player confusion and real‑world rule usage |

### 3.4 Current highest-risk area (frontend UX & production validation)

> **Status (2025-12-08): Critical test/UX gaps identified; operational readiness improving**

Following PASS20–21 completion:

- ✅ **Orchestrator migration complete** (Phase 3, 100% rollout, ~1,176 lines legacy removed)
- ✅ **Observability infrastructure implemented** (3 dashboards, k6 load testing)
- ✅ **Critical context coverage improved** (GameContext ≈89.5%, SandboxContext ≈84.2%)
- ✅ **Test suite stabilised** (2,987 TS tests passing, ~130 skipped with rationale)

Remaining priorities for v1.0:

- **Client-side test coverage (critical):**  
  Earlier assessments identified client components, hooks, and services as a major test gap. React Testing Library infrastructure and focused suites around `BoardView`, `GameHUD`, `VictoryModal`, and critical hooks (for example `useGameConnection`, `useSandboxInteractions`) are required to de-risk frontend regressions.

- **Frontend UX polish (P1):**  
  The frontend needs continued improvements to timers, reconnection/spectator flows, post-game navigation, and mobile/touch ergonomics. Teaching overlays and weird-state explanations must remain aligned with the canonical rules and telemetry specs.

- **Production validation (P0):**  
  **Must execute before production launch:**
  - Run load tests at target scale (≈100 concurrent games, 200–300 players)
  - Establish baseline "healthy system" metrics from staging runs
  - Execute operational drills (secrets rotation, backup/restore, AI service degradation)
  - Validate all SLOs under real production-scale load

High-level risk framing and historical assessment for this area are summarised in [`docs/archive/assessments/WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](docs/archive/assessments/WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md:1) and [`docs/archive/assessments/WAVE3_ASSESSMENT_REPORT.md`](docs/archive/assessments/WAVE3_ASSESSMENT_REPORT.md:1).

---

## 4. Success metrics & release gate criteria (v1.0 readiness)

### 4.0 Release gate criteria (summary)

v1.0 is considered ready to ship when all of the following are true (this is the goals-level definition; current measured status lives in [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) and `docs/PRODUCTION_READINESS_CHECKLIST.md`):

- **Rules and victory correctness** is rules-complete and parity/contract validated for the supported boards and player counts (see §3 and §6; rules authority remains [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1)).
- **Performance SLOs** in §4.1 are met under production-scale load (details in `docs/testing/BASELINE_CAPACITY.md` and `docs/operations/SLO_VERIFICATION.md`).
- **Quality gates** in §4.2 are green (tests + coverage target).
- **Feature scope** in §6.1–§6.2 is playable end-to-end in both backend multiplayer and sandbox.
- **Operational readiness** gates in §4.4 are met (monitoring, runbooks, drills, and rollback posture).

### 4.1 Performance SLOs

From [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:144-149):

| Metric              | Target          | Measurement                               |
| ------------------- | --------------- | ----------------------------------------- |
| **System uptime**   | >99.9%          | Core gameplay surfaces availability       |
| **AI move latency** | <1 second (p95) | Time from AI turn start to move broadcast |
| **UI frame rate**   | <16ms updates   | Smooth 60fps rendering during gameplay    |
| **Move validation** | <200ms (p95)    | Human move submission to broadcast        |
| **HTTP API**        | <500ms (p95)    | Login, game creation, state fetch         |

These SLOs are the canonical targets for v1.0. Load-testing docs, alert thresholds, and dashboards must be kept consistent with this table and updated only when this file is updated.

### 4.2 Test and quality gates (v1.0)

| Gate / metric             | Requirement              | Where to check current status                                                        |
| ------------------------- | ------------------------ | ------------------------------------------------------------------------------------ |
| **TypeScript tests**      | All passing in CI lanes  | [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) |
| **Python tests**          | All passing in CI lanes  | [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) |
| **Contract vectors**      | 100% TS↔Python parity    | [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) |
| **Coverage target**       | ≥80% lines overall       | [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) |
| **Rules scenario matrix** | All FAQ examples covered | `docs/rules/RULES_SCENARIO_MATRIX.md`                                                |
| **Integration tests**     | Core workflows passing   | [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) |

> **Note:** Live test counts and coverage breakdowns are maintained in [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:236). This document is **not** the single source of truth for those numbers; it records only the high-level requirements and a recent snapshot.

#### 4.2.1 Metrics & test suites (qualitative overview)

To avoid duplicating live metrics, this section describes the **shape** of the test and CI surface; concrete counts and coverage percentages remain in [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1).

- **TypeScript suites:** Jest unit and integration tests for the shared engine, backend hosts, frontend client, and sandbox; Playwright E2E tests for core auth and gameplay flows.
- **Python suites:** pytest suites for the AI service covering rules engine behaviour, parity tests, training/heuristic harnesses, and invariants.
- **Cross-language & CI gates:** Contract-vector tests (TS and Python) plus orchestrator and host-parity suites that must be green in CI before promotion to staging/production.

### 4.3 Feature completion criteria

These are **release gate criteria**, not a live progress checklist. For current status, see [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1).

- [ ] All 24 FAQ scenarios from the rules document have corresponding tests
- [ ] All three board types (8×8, 19×19, hex) fully playable end-to-end
- [ ] All victory conditions (Ring Elimination, Territory Control, Last Player Standing) correctly implemented and surfaced via game-end explanations
- [ ] AI difficulty ladder (1–10) functional with fallback handling
- [ ] Backend-driven games playable end-to-end via the web client
- [ ] Sandbox mode provides rules-complete local play
- [ ] Spectator mode allows read-only game viewing

### 4.4 Release gate criteria: rollout & operations readiness

Environment posture and rollout discipline are first-class parts of v1.0 readiness, not an afterthought. At a high level, v1.0 is considered **environment‑ready** when:

- **SLOs are measured and validated** under target load (see §4.1 and `docs/testing/BASELINE_CAPACITY.md`).
- **Monitoring and alerting are in place** for critical gameplay, parity, and system-health signals (see `docs/operations/ALERTING_THRESHOLDS.md`).
- **Runbooks exist and have been exercised** for key operational failure modes (backup/restore, secrets rotation, AI service degradation, rollback).
- **Rollout posture is documented** (environment presets, rollback strategy, and orchestrator-on invariants) in `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` and `STRATEGIC_ROADMAP.md`.

Detailed rollout tables, environment presets, and alert thresholds live in `STRATEGIC_ROADMAP.md`, `CURRENT_STATE_ASSESSMENT.md`, `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md`, and `docs/operations/ALERTING_THRESHOLDS.md`.

For the current completion status of these gates, see [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) and `docs/PRODUCTION_READINESS_CHECKLIST.md`.

---

## 5. Key risks, dependencies & assumptions

This section captures goal-level risks and assumptions that shape scope and success criteria. Live issue tracking remains in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and current measured status remains in [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md).

These goals assume the following technical and operational dependencies, which are defined in more detail in the referenced architecture and operations documents:

- **Canonical rules specification as rules SSoT.** All gameplay semantics are defined, first and foremost, by the written canonical rules (`RULES_CANONICAL_SPEC.md` together with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`). The shared TypeScript engine under `src/shared/engine/**` (helpers → aggregates → orchestrator) is the primary executable derivation of that spec, validated by its contract tests and parity suites. Backend hosts, the client sandbox, and the Python rules engine are adapters over this shared surface; changes to rules must converge on the canonical rules spec and then be reflected in the shared engine. See [`docs/architecture/MODULE_RESPONSIBILITIES.md`](docs/architecture/MODULE_RESPONSIBILITIES.md:1), [`docs/architecture/DOMAIN_AGGREGATE_DESIGN.md`](docs/architecture/DOMAIN_AGGREGATE_DESIGN.md:1), and [`docs/architecture/CANONICAL_ENGINE_API.md`](docs/architecture/CANONICAL_ENGINE_API.md:1) for the canonical module catalog, aggregate design, and Move/orchestrator/WebSocket lifecycle that implement these goals.
- **Python AI service as the primary tactical engine.** Higher AI difficulties (3–10) depend on the Python `ai-service` for Minimax, MCTS, and Descent-style search as described in [`AI_ARCHITECTURE.md`](docs/architecture/AI_ARCHITECTURE.md:1) and AI docs under `docs/ai/` and `ai-service/docs/`. The TypeScript fallback AI exists for resilience and low-difficulty play, not as the sole long-term AI.
- **Canonical AI training data and pipelines.** Neural and heuristic training must use canonical replay databases and parity-gated datasets as described in [`docs/ai/AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:1) and `ai-service/TRAINING_DATA_REGISTRY.md`. Legacy or non-canonical data may be used for ablation but must never be treated as production-default.
- **Single-region, single-app-instance topology for v1.0.** The production topology assumes a single Node.js app instance per environment (`RINGRIFT_APP_TOPOLOGY=single`) backed by PostgreSQL and Redis, as described in [`docs/planning/DEPLOYMENT_REQUIREMENTS.md`](docs/planning/DEPLOYMENT_REQUIREMENTS.md:1) and [`docs/architecture/TOPOLOGY_MODES.md`](docs/architecture/TOPOLOGY_MODES.md:1). Horizontal scaling beyond this and multi-region deployments are explicitly post‑v1.0 concerns.
- **PostgreSQL, Redis, and WebSocket infrastructure.** Game lifecycle, session management, and real-time multiplayer depend on PostgreSQL, Redis, and a WebSocket server as described in [`README.md`](README.md:1) and [`docs/operations/OPERATIONS_DB.md`](docs/operations/OPERATIONS_DB.md:1).
- **Observability stack and load-testing tooling.** Meeting the SLOs in §4 and validating production readiness requires the Prometheus/Grafana/Alertmanager stack and the k6 load scenarios defined in [`docs/operations/ALERTING_THRESHOLDS.md`](docs/operations/ALERTING_THRESHOLDS.md:1), [`docs/planning/DEPLOYMENT_REQUIREMENTS.md`](docs/planning/DEPLOYMENT_REQUIREMENTS.md:1), and [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1).
- **Canonical board topologies.** Only the three documented board types (square8, square19, hexagonal) are in scope for v1.0; rules semantics, tests, and AI training pipelines assume the geometry contracts in [`docs/architecture/TOPOLOGY_MODES.md`](docs/architecture/TOPOLOGY_MODES.md:1).

If any of these assumptions change materially (for example, a different deployment topology or AI service design), this goals document should be updated **first** and downstream roadmaps and assessments should be adjusted to match.

---

## 6. In-scope (v1.0) and scope boundaries

Sections 6–8 collectively define what is in scope for the current v1.0 phase (required features and user flows), what is intentionally deferred to later phases, and what is explicitly out of scope or constrained for this project.

**Scope note:** The codebase may contain early or partial implementations of some post‑v1.0 features. Unless a feature is listed under §6.1–§6.2 as required for v1.0, it is not treated as a v1.0 release gate.

### 6.1 Required features (v1.0)

**Game mechanics (must have)**

- Ring placement (1–3 rings on empty, 1 ring on stacks)
- Stack movement with minimum distance rules
- Overtaking captures with mandatory chain continuation
- Line detection and processing (4+ for 8×8, 4+ for 19×19/hex)
- Territory disconnection detection and processing
- Graduated line rewards (Option 1 vs Option 2 for long lines)
- Forced elimination when blocked
- All three victory conditions with proper resolution
- Player choice system for all decision points

**Board types (must have)**

- 8×8 square board (18 rings/player, 64 spaces)
- 19×19 square board (72 rings/player, 361 spaces)
- Hexagonal board (96 rings/player, 469 spaces, 13 per side)

**Game modes (must have)**

- Backend multiplayer games (2–4 players via WebSocket)
- AI opponents (difficulty 1–10)
- Local sandbox mode (browser-only, full rules)
- Mixed human/AI player configurations

**Infrastructure (must have)**

- Docker-based deployment
- PostgreSQL database with Prisma ORM
- Redis for session caching and locking
- WebSocket server for real-time communication
- Python AI service with FastAPI
- JWT authentication

### 6.2 Required user flows (v1.0)

1. **Account creation and login** – Register, authenticate, maintain session
2. **Game creation** – Choose board type, player count, AI configuration
3. **Game joining** – Browse lobby, join waiting games
4. **Gameplay** – Move selection, choice dialogs, state synchronization
5. **Game completion** – Victory display, return to lobby, access to replay/spectate where available

---

## 7. Future scope (post‑v1.0)

### 7.1 Planned for later phases

| Feature                   | Description                                                              | Phase     |
| ------------------------- | ------------------------------------------------------------------------ | --------- |
| **Advanced AI**           | Neural network-based AI, MCTS/minimax and Descent at higher difficulties | Phase 4   |
| **Rating system**         | ELO-based player rankings with matchmaking                               | Phase 3   |
| **Automated matchmaking** | Queue-based matching by rating and preferences                           | Phase 3   |
| **Leaderboards**          | Global and periodic rankings                                             | Phase 3   |
| **In-game chat**          | Real-time messaging during games                                         | Phase 3   |
| **Game replays**          | Move-by-move replay viewer                                               | Phase 5   |
| **Tournament support**    | Structured competition format                                            | Future    |
| **Mobile optimisation**   | Touch-friendly UI, responsive design polish                              | Post-v1.0 |
| **Game timers**           | Configurable time controls with UI display                               | Post-v1.0 |

### 7.2 Nice-to-haves (deferred)

- Interactive tutorial system
- Video demonstrations of rules
- Strategy guides within the application
- Achievement/badge system
- Social features (friends, invitations)
- Game analysis tools
- Multi-language support

---

## 8. Out-of-scope / Non-goals (explicitly out of scope)

### 8.1 What RingRift will **not** do

| Non-goal                       | Rationale                                                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Native mobile apps**         | Web-based approach ensures universal accessibility; native apps add maintenance burden without proportional value |
| **Random/chance elements**     | Core game design principle – RingRift is a perfect information game                                               |
| **Real-money wagering**        | Regulatory complexity; focus is on gameplay quality                                                               |
| **Social media integration**   | Privacy concerns; not core to gameplay experience                                                                 |
| **Blockchain/NFT features**    | Adds complexity without gameplay value                                                                            |
| **Embedded video/voice chat**  | Existing external solutions are adequate                                                                          |
| **Offline-first architecture** | Real-time multiplayer is core; local sandbox covers offline needs                                                 |

### 8.2 Boundaries and constraints

- **Single-region deployment** for v1.0 (multi-region is post‑v1.0)
- **Single app instance topology** is the supported production configuration
- **Browser-only client** (no desktop/mobile native apps planned)
- **English-only** for initial release
- **Three board types only** (8×8, 19×19, hex) – no custom board sizes
- **Four players maximum** per game

---

## 9. Relationship to other docs

This goals document sits at the top of the planning stack for **project direction**:

- [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) (this file) defines the high-level product and technical goals, success criteria, and scope boundaries for the current phase.
- [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1) translates those goals into a phased implementation and SLO roadmap; when direction or success criteria change, update this file first and then adjust the roadmap.
- [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1) reports factual, code-verified implementation status relative to these goals and the roadmap; it does not define new goals.
- [`docs/archive/FINAL_ARCHITECT_REPORT.md`](docs/archive/FINAL_ARCHITECT_REPORT.md:1) and other archived reports provide historical context; where they disagree with this document or the current roadmap/state assessment on direction, treat them as superseded.

The tables below group key related documents by role so readers can quickly jump between **goals**, **plan**, and **current reality**.

### 9.1 Implementation & execution

| Document                                                                               | Purpose                                   |
| -------------------------------------------------------------------------------------- | ----------------------------------------- |
| [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1)                         | Phased execution plan, SLOs, milestones   |
| [`TODO.md`](TODO.md:1)                                                                 | Task-level tracking, priority assignments |
| [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1) | Code-verified implementation status       |
| [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md:1)                                                 | Active bugs and gaps with priorities      |

### 9.2 Rules & design authority

| Document                                                                       | Purpose                                           |
| ------------------------------------------------------------------------------ | ------------------------------------------------- |
| [`docs/rules/COMPLETE_RULES.md`](docs/rules/COMPLETE_RULES.md:1)               | **Authoritative rulebook** – canonical game rules |
| [`docs/rules/COMPACT_RULES.md`](docs/rules/COMPACT_RULES.md:1)                 | Implementation-oriented rules summary             |
| [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1)                         | Technical specification of rule semantics         |
| [`docs/rules/RULES_SCENARIO_MATRIX.md`](docs/rules/RULES_SCENARIO_MATRIX.md:1) | Test scenario mapping to rules/FAQ                |

### 9.3 Architecture & technical

| Document                                                                                         | Purpose                                                                                                       |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| [`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md:1)                                     | High-level system architecture review and historical remediation context subordinate to the goals in this doc |
| [`docs/architecture/CANONICAL_ENGINE_API.md`](docs/architecture/CANONICAL_ENGINE_API.md:1)       | Canonical Move/orchestrator/WebSocket lifecycle and engine public API that implements the rules engine goals  |
| [`RULES_ENGINE_ARCHITECTURE.md`](docs/architecture/RULES_ENGINE_ARCHITECTURE.md:1)               | Detailed rules engine design and TS↔Python parity mapping                                                     |
| [`docs/architecture/MODULE_RESPONSIBILITIES.md`](docs/architecture/MODULE_RESPONSIBILITIES.md:1) | Module catalog for the shared TypeScript engine helpers, aggregates, and orchestrator                         |
| [`docs/architecture/DOMAIN_AGGREGATE_DESIGN.md`](docs/architecture/DOMAIN_AGGREGATE_DESIGN.md:1) | Aggregate-level design reference for the shared engine                                                        |
| [`docs/architecture/TOPOLOGY_MODES.md`](docs/architecture/TOPOLOGY_MODES.md:1)                   | Supported board topologies and geometry constraints                                                           |
| [`AI_ARCHITECTURE.md`](docs/architecture/AI_ARCHITECTURE.md:1)                                   | AI service architecture, difficulty ladder, and training/parity plans subordinate to the goals defined here   |
| [`src/shared/engine/orchestration/README.md`](src/shared/engine/orchestration/README.md:1)       | Turn orchestrator implementation guide                                                                        |

### 9.4 Operations & development

| Document                               | Purpose                    |
| -------------------------------------- | -------------------------- |
| [`README.md`](README.md:1)             | Project overview and setup |
| [`QUICKSTART.md`](QUICKSTART.md:1)     | Getting started guide      |
| [`CONTRIBUTING.md`](CONTRIBUTING.md:1) | Contribution guidelines    |
| [`docs/INDEX.md`](docs/INDEX.md:1)     | Documentation index        |

---

## 10. Open questions / owner decisions required

This section records goal-level questions that are not fully specified by the current documentation and require explicit owner decisions. Until resolved, implementers should treat these as constraints on making irreversible changes, not as implicit commitments.

1. **AI difficulty ladder positioning in v1.0 experience.**  
   The current docs agree that the 1–10 difficulty ladder exists and that difficulties 7–10 are more experimental or advanced (see [`CURRENT_STATE_ASSESSMENT.md`](docs/archive/historical/CURRENT_STATE_ASSESSMENT.md:1) and [`AI_ARCHITECTURE.md`](docs/architecture/AI_ARCHITECTURE.md:1)), but they do not clearly state whether v1.0's primary experience should emphasise the beginner–intermediate band (1–6) in rated queues while keeping 7–10 as opt-in expert modes.
   - **Option A:** Treat 1–6 as the canonical supported ladder for public/rated queues at v1.0, with 7–10 flagged as experimental or unrated.
   - **Option B:** Treat the full 1–10 ladder as in scope for rated play at v1.0, accepting higher variance in AI strength and latency at the top difficulties.
   - **Implication:** Affects UX copy, lobby defaults, and how strictly we gate SLOs and regression budgets for higher difficulties.

2. **Initial public launch concurrency target.**  
   [`STRATEGIC_ROADMAP.md`](docs/planning/STRATEGIC_ROADMAP.md:1) defines load-test scenarios around ~100 concurrent active games and 200–300 players, but the goals docs do not explicitly state whether this is the minimum acceptable production scale for v1.0 or a stretch target.
   - **Option A:** Declare the documented P‑01 target (≈100 concurrent games, 200–300 players) as the baseline concurrency that must be demonstrated before launch.
   - **Option B:** Allow a smaller initial public rollout (for example, friends-and-family scale) with the same SLO shape but lower absolute concurrency, treating the P‑01 numbers as follow-up stretch goals.
   - **Implication:** Affects acceptance criteria for "production-ready" and how we interpret completion of Wave 7 / P‑01 validation.

3. **Long-term emphasis: competitive ladder vs casual sandbox.**  
   The docs describe both a robust sandbox for rules exploration and AI work, and plans for rated matchmaking and leaderboards, but they do not explicitly prioritise one as the primary long-term focus for design and engineering trade-offs.
   - **Option A:** Optimise first for a high-quality competitive ladder (ratings, time controls, production SLOs), treating the sandbox primarily as a developer/designer/analysis tool.
   - **Option B:** Optimise first for a rich exploratory sandbox and AI testbed, with competitive ladder features as secondary.
   - **Implication:** Affects where to invest limited UX and feature capacity (for example, tutorialisation and analysis tools versus rating UX, anti-abuse, and matchmaking sophistication).

---

## Appendix A: Game design principles

_Derived from [`docs/rules/COMPLETE_RULES.md`](docs/rules/COMPLETE_RULES.md:153-177)_

RingRift's design is guided by these core principles:

1. **Perfect information** – No hidden information, no random elements. All game state is visible to all players at all times.
2. **Deterministic resolution** – Given the same inputs, the same outcomes always result. This enables reproducibility, AI training, and replay verification.
3. **Emergent complexity** – Simple rules create complex interactions. Stack building, marker flipping, line formation, and territory disconnection interweave to produce deep strategic possibilities.
4. **Multi-player dynamics** – Designed for 3 players (extensible to 2–4), creating natural alliance formation and leader-balancing behavior that pure 2-player games lack.
5. **Dual victory paths** – Ring elimination (tactical) and territory control (strategic) provide multiple routes to victory, rewarding different play styles.
6. **Incremental learning** – The 8×8 simplified version provides an accessible entry point; 19×19 and hexagonal versions offer increased depth for experienced players.

---

## Appendix B: Version history

| Version | Date       | Changes                                                                                                                                                                                   |
| ------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | 2025-11-26 | Initial creation consolidating goals from `README.md`, `STRATEGIC_ROADMAP.md`, `CURRENT_STATE_ASSESSMENT.md`, `TODO.md`, and `ringrift_complete_rules.md`                                 |
| 1.1     | 2025-12-10 | Promoted to repo root as the canonical goals SSoT; clarified clustering of objectives; added explicit rules‑UX telemetry and AI training pipeline objectives; fixed several broken links. |

---

_This document should be reviewed and updated whenever project direction changes significantly. It is intended to be read by all project contributors and stakeholders to ensure alignment on objectives and priorities._
