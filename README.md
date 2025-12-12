# RingRift - Multiplayer Strategy Game

<!-- CI Status Badge -->

![RingRift CI/CD](https://github.com/an0mium/RingRift/actions/workflows/ci.yml/badge.svg)
![Parity CI Gate](https://github.com/an0mium/RingRift/actions/workflows/parity-ci.yml/badge.svg)

**Doc Status (2025-12-07): Active (project overview & navigation)**

- High-level project overview, setup, and API surface.
- Not a rules or lifecycle SSoT. For rules semantics, defer to `RULES_CANONICAL_SPEC.md` plus the shared TypeScript rules engine under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts + v2 contract vectors). For lifecycle semantics (move/decision/WebSocket), defer to `docs/architecture/CANONICAL_ENGINE_API.md` plus shared TS/WebSocket types and schemas.

⚠️ **PROJECT STATUS: STABLE BETA – ENGINE COMPLETE, PRODUCTION VALIDATION IN PROGRESS** ⚠️

> **Important:** The canonical turn orchestrator is 100% rolled out with 54 contract vectors at 100% TS↔Python parity. All 14 development waves are complete. Current focus is on **production validation**, **scaling tests**, and **documentation cleanup**. See [CODEBASE_REVIEW_2025_12_11.md](./docs/CODEBASE_REVIEW_2025_12_11.md) and [NEXT_STEPS_2025_12_11.md](./docs/NEXT_STEPS_2025_12_11.md) for code‑verified status.

A web-based multiplayer implementation of the RingRift strategy game supporting 2-4 players with flexible human/AI combinations across multiple board configurations.

## 📋 Current Status

**Last checked:** 2025-12-07 (aligned with current scripts/config; rerun below commands for fresh numbers)

### Key Metrics

| Metric                      | Value                      |
| --------------------------- | -------------------------- |
| TypeScript tests (CI-gated) | 2,987 passing              |
| Python tests                | 836 passing                |
| Contract vectors            | 54 (100% TS↔Python parity) |
| Line coverage               | ~69%                       |
| Canonical phases            | 8                          |
| Development waves complete  | 14/14                      |

### Recent Updates (2025-12-07)

- Added a **DecisionUI PlayerChoice harness** (`tests/unit/DecisionUI.choiceDialog.harness.test.tsx`) to exercise `capture_direction`, `line_reward_option`, `region_order`, `ring_elimination`, and `line_order` choice routing (with countdown handling) without the brittle BackendGameHost wiring. Client/component coverage is still very low but now has an initial anchor suite.
- Backfilled a **hex line → territory FAQ scenario** (`tests/scenarios/FAQ_Q20_line_then_territory.hex.test.ts`) and recorded it in `docs/rules/RULES_SCENARIO_MATRIX.md` to keep the scenario matrix aligned with the rules/FAQ set.
- Added **square8/square19 line → territory FAQ variants** (`tests/scenarios/FAQ_Q20_line_then_territory.square8.test.ts`) to cover the combined line_reward + territory region pipeline on square boards (both variants in one suite) and keep the scenario matrix current.

### Commands

- JS/TS: `npm test` (see `tests/README.md` for suite layout) or the focused `npm run test:p0-robustness` / `npm run test:orchestrator-parity` profiles for rules + parity signals.
- Python AI service: `cd ai-service && pytest`
- Replay parity + canonical history: `npm run test:ts-parity` (TS traces), `cd ai-service && python -m scripts.check_ts_python_replay_parity --db <path>`, and `python -m scripts.check_canonical_phase_history --db <path>` for recorded DBs.
- For a code-verified snapshot, see `CURRENT_STATE_ASSESSMENT.md`; for open issues, see `KNOWN_ISSUES.md`.

### ✅ What's Working

- ✅ Project infrastructure (Docker, database, Redis, WebSocket)
- ✅ **Session Management** - Robust `GameSessionManager` with distributed locking
- ✅ **Rules Facade** - `RulesBackendFacade` abstracting Python/TS engine parity
- ✅ TypeScript type system and architecture
- ✅ Canonical rules engine SSoT in TypeScript (`src/shared/engine/**`) with eight canonical phases (`ring_placement`, `movement`, `capture`, `chain_capture`, `line_processing`, `territory_processing`, `forced_elimination`, `game_over`); Python mirror passes v2 contract vectors (54, 0 mismatches) and parity CI gate
- ✅ Comprehensive game rules documentation
- ✅ Server and client scaffolding
- ✅ **Marker system** - Placement, flipping, collapsing fully functional
- ✅ **Movement validation** - Distance rules, path checking working
- ✅ **Basic and chained captures** - Overtaking captures and mandatory continuation implemented
- ✅ **Line detection** - All board types (8x8, 19x19, hexagonal)
- ✅ **Territory disconnection** - Detection and processing implemented
- ✅ **Phase transitions & forced elimination** - Explicit eight-phase turn flow (`ring_placement` → `game_over`), with forced elimination surfaced as its own move/choice
- ✅ **Player state tracking** - Ring counts, eliminations, territory
- ✅ **Hexagonal board support** - Full 469-space board (13 per side) validated
- ✅ **Client-local sandbox engine** - `/sandbox` uses `ClientSandboxEngine` plus `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxTerritory.ts`, and `sandboxVictory.ts` (with line and territory processing now wired directly to shared helpers) to run full games in the browser (movement, captures, lines, territory, and ring/territory victories) with dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts`.
- ✅ **Orchestrator at 100%** - Shared turn orchestrator handles all turn processing; legacy paths deprecated
- ✅ **Accessibility features (Wave 14)** – Keyboard navigation for board and dialogs, screen reader announcements via `ScreenReaderAnnouncer`, high-contrast and colorblind-friendly themes via `AccessibilitySettingsPanel` / `AccessibilityContext`, reduced-motion support, and over 50 ARIA/role attributes across core UI surfaces; see [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md) for details.
- ✅ **Move history panel** - `GameHistoryPanel` integrated into backend games with expandable move details
- ✅ **AI difficulty ladder (Waves 9)** – Canonical 1–10 difficulty ladder (Random/Heuristic/Minimax/MCTS/Descent) with service-backed PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`)
- ✅ **Observability stack (Wave 6)** – 3 Grafana dashboards (game-performance, rules-correctness, system-health), Prometheus alerts, k6 load testing with 4 production-scale scenarios
- ✅ **Matchmaking & Ratings (Wave 12)** – ELO-based rating system with `RatingService.ts`, leaderboard page
- ✅ **Multi-player support (Wave 13)** – 3-4 player games fully supported with evaluation pools
- ✅ **Game records & training data (Wave 10)** – GameRecord types, self-play recording, training data registry, parity gate infrastructure
- ✅ **Golden replays & test hardening (Wave 11)** – 29 golden game candidates, hash parity infrastructure, schema v6 with available moves enumeration

### ⚠️ Remaining Gaps

- ⚠️ **Client/component test coverage** – React components, hooks, and contexts have growing test coverage (100+ component test files exist for BoardView, GameHUD, ChoiceDialog, hooks, and contexts per [CODEBASE_REVIEW_2025_12_11.md](./docs/CODEBASE_REVIEW_2025_12_11.md)), though additional scenario coverage is ongoing.
- ⚠️ **Rules/FAQ scenario backfill is ongoing** – The scenario matrix (`docs/rules/RULES_SCENARIO_MATRIX.md`) now includes a hex line → territory FAQ path; square/19x variants and additional multi-choice turn sequences still need coverage to harden late-game interactions.
- ⚠️ **UX polish and resilience** – Timers, reconnection/spectator flows, post-game navigation, chat ergonomics, and Victory modal copy/telemetry still need refinement; mobile/touch responsiveness remains desktop-first.
- ⚠️ **AI boundary observability** – Service-backed PlayerChoices work, but timeout/fallback tests and lightweight latency/error metrics for `AIServiceClient`/`AIEngine` should be strengthened.
- ⚠️ **Production validation & security hardening** – Sustained production-scale testing, operational drills (backup/restore, secrets rotation), and the security hardening review (`docs/security/SECURITY_THREAT_MODEL.md`) remain to be completed; ML-backed AI agents are still future work.

### 🎯 What This Means

**Can Do (today):**

- Create and play full games via the HTTP API and React lobby (including AI opponent configuration at difficulty 1-10).
- Play backend-driven games end-to-end using the React client with click-to-move and server-validated moves.
- Have AI opponents take turns via the Python AI Service with canonical difficulty ladder (Random/Heuristic/Minimax/MCTS/Descent).
- Process all game mechanics: placement, movement, captures, chain captures, lines, territory disconnection, forced elimination.
- Play on all board types (8x8, 19x19, hexagonal) with 2-4 players.
- Run full games in `/sandbox` with client-local engine, AI opponents, and replay capabilities.
- Track ratings (ELO), view leaderboards, watch replays, and spectate games.
- Access 3 Grafana dashboards for monitoring and run k6 load tests against defined SLOs.

**Cannot Reliably Do (yet):**

- Guarantee mobile/touch UX is polished (desktop-first design).
- Use ML-backed AI agents (neural network scaffolding exists but not integrated).
- Claim production-hardened security (threat model exists, review pending).

**For complete assessment, see [CODEBASE_REVIEW_2025_12_11.md](./docs/CODEBASE_REVIEW_2025_12_11.md) and [NEXT_STEPS_2025_12_11.md](./docs/NEXT_STEPS_2025_12_11.md)**
**For detailed issues, see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md)**
**For roadmap, see [TODO.md](./TODO.md) and [docs/planning/STRATEGIC_ROADMAP.md](./docs/planning/STRATEGIC_ROADMAP.md)**

---

## 📚 Documentation Map & Canonical Sources

To understand the project and know which documents are authoritative for each area, use these documents. When information in older planning or analysis docs conflicts with the files listed below, these **canonical sources win**.

### For Players & Designers (Rules)

- `RULES_CANONICAL_SPEC.md` – Canonical semantics SSoT (phase/move contracts, forced elimination, and replay requirements).
- `ringrift_complete_rules.md` – **The authoritative rulebook.** Full, narrative rules for players and designers.
- `ringrift_compact_rules.md` – Compact, implementation-oriented spec for engine/AI authors.
- `archive/RULES_ANALYSIS_PHASE2.md` – Consistency and strategic assessment of the rules.

### For Developers (Architecture, Status, Setup)

- **Status & Roadmap (Canonical, Living)**
  - `docs/CODEBASE_REVIEW_2025_12_11.md` – First-principles codebase audit (code-verified).
  - `docs/NEXT_STEPS_2025_12_11.md` – Current architectural assessment and next steps.
  - `docs/PRODUCTION_READINESS_CHECKLIST.md` – Launch criteria (58/67 complete).
  - `TODO.md` – Phase-structured task tracker.
  - `docs/planning/STRATEGIC_ROADMAP.md` – Phased roadmap to production.
  - `KNOWN_ISSUES.md` – Current P0/P1 bugs and gaps.

- **Architecture & Design**
  - `ARCHITECTURE_ASSESSMENT.md` – Comprehensive architecture review and future design plans.
  - `AI_ARCHITECTURE.md` – AI Service architecture, assessment, and improvement plans (with cross-links to training pipelines and incidents).
  - `RULES_ENGINE_ARCHITECTURE.md` – Python Rules Engine architecture and rollout strategy.

- **Subsystem Guides**
  - `tests/README.md` – Jest setup, test structure, and the rules/FAQ → scenario test matrix.
  - `docs/rules/RULES_SCENARIO_MATRIX.md` – Canonical mapping of rules/FAQ sections to specific Jest test suites.
  - `ai-service/README.md` – Python AI Service (Random/Heuristic AI, endpoints, setup).
  - [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1) – AI Service training pipelines, self-play and Territory dataset generation CLIs, JSONL schema, and seed behaviour.
  - `ai-service/TRAINING_DATA_REGISTRY.md` – Canonical vs legacy replay DB inventory, plus the parity/history gates (`ai-service/scripts/generate_canonical_selfplay.py`, `ai-service/scripts/check_ts_python_replay_parity.py`, `ai-service/scripts/check_canonical_phase_history.py`).
  - `CONTRIBUTING.md` – Contribution workflow and historical phase breakdown.

- **Historical Plans & Evaluations**
  - [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1) – Incident report for the TerritoryMutator vs `GameEngine.apply_move()` divergence and its fix, with tests and follow-up tasks.
  - Docs under `archive/` – Earlier architecture and improvement plans, preserved for context only.

### 🔗 Developer Quick Links

- **Start Here:** [docs/INDEX.md](./docs/INDEX.md) – Concise entry point for new contributors.
- **Getting Started:** [QUICKSTART.md](./QUICKSTART.md)
- **Contributing:** [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Testing Guide:** [tests/README.md](./tests/README.md)

## 🎯 Project Goals & Scope

For a concise statement of why RingRift exists, what the current v1.0 phase is trying to achieve, and what is in or out of scope, see [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1).

[`PROJECT_GOALS.md`](PROJECT_GOALS.md:1) is the authoritative source for:

- Purpose & vision of RingRift.
- Current phase objectives.
- v1.0 success criteria and readiness metrics.
- MVP scope, future scope, and explicit non-goals.

This README focuses on project overview, architecture, and day-to-day development workflows. When in doubt about **what success looks like** for the project, prefer [`PROJECT_GOALS.md`](PROJECT_GOALS.md:1); use this README and the roadmap/state docs to understand **how we plan to get there** and **where we are now**.

## 🎯 Overview

RingRift is a sophisticated turn-based strategy game featuring:

- **Multiple Board Types**: 8x8 square, 19x19 square, and hexagonal layouts
- **Flexible Player Support**: 2-4 players with human/AI combinations
- **Real-time Multiplayer**: WebSocket-based live gameplay (engine and transport implemented; UX still evolving)
- **Spectator Mode**: Watch games in progress when `allowSpectators` is enabled on a game
- **Rating System**: ELO-based player rankings (rating updates implemented; calibration/tuning still ongoing)
- **Time Controls**: Configurable game timing (time control stored per game; display/UI still minimal)
- **Cross-platform**: Web-based for universal accessibility

## 🏗️ Architecture

### Technology Stack

#### Backend

- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js with comprehensive middleware
- **Database**: PostgreSQL with Prisma ORM
- **Real-time**: Socket.IO for WebSocket communication
- **Caching**: Redis for caching and game-related data
- **Authentication**: JWT-based with bcrypt password hashing
- **Validation**: Zod schemas for type-safe data validation
- **Logging**: Winston for structured logging
- **Security**: Helmet, CORS, rate limiting

#### Frontend

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Routing**: React Router for SPA navigation
- **State Management**: React Query for server state, Context API for client state
- **Styling**: Tailwind CSS for utility-first styling
- **WebSocket**: Socket.IO client for real-time communication
- **HTTP Client**: Axios with interceptors for API communication

#### Infrastructure

- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development and basic deployment
- **Database**: PostgreSQL with connection pooling
- **Caching**: Redis for high-performance data access
- **Monitoring Stack**: Prometheus + Grafana + Alertmanager run by default with production-ready dashboards for game performance, rules correctness, and system health
- **Environment**: Environment-based configuration via `.env`

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │    │   Express API   │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ • Game UI       │◄──►│ • REST Routes   │◄──►│ • Game Data     │
│ • Real-time     │    │ • WebSocket     │    │ • User Profiles │
│ • State Mgmt    │    │ • Game Engine   │    │ • Match History │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │      Redis      │              │
         └──────────────►│                 │◄─────────────┘
                        │ • Caching       │
                        │ • Game Cache    │
                        │ • Future queue  │
                        └─────────────────┘
```

### Canonical Turn Orchestrator

The rules engine has been consolidated into a single canonical implementation with clean adapter patterns:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Canonical Turn Orchestrator                      │
│           src/shared/engine/orchestration/                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┬──────────┬──────────┬──────────┬────────────────┐ │
│  │Placement │Movement  │Capture   │Line      │Territory       │ │
│  │Aggregate │Aggregate │Aggregate │Aggregate │Aggregate       │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
 ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
 │TurnEngineAdapter│ │SandboxAdapter │   │Python Contract    │
 │   (Backend)   │   │   (Client)    │   │   Test Runner     │
 └───────────────┘   └───────────────┘   └───────────────────┘
```

- **Single Source of Truth**: All rules logic in [`src/shared/engine/`](src/shared/engine/)
- **Orchestration Layer**: [`src/shared/engine/orchestration/`](src/shared/engine/orchestration/) contains [`turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts)
- **Backend Adapter**: [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts) wraps orchestrator for WebSocket/session concerns
- **Client Adapter**: [`SandboxOrchestratorAdapter.ts`](src/client/sandbox/SandboxOrchestratorAdapter.ts) wraps orchestrator for local simulation
- **Cross-Language Parity**: Contract test vectors validate Python engine matches TypeScript behavior

### Contract Tests

Contract tests ensure parity between TypeScript and Python implementations:

```bash
# TypeScript contract tests over the v2 contract vectors
npm test -- tests/contracts/contractVectorRunner.test.ts

# Python contract tests over the same v2 contract vectors
./scripts/run-python-contract-tests.sh
```

The canonical contract vectors live in [`tests/fixtures/contract-vectors/v2/`](tests/fixtures/contract-vectors/v2/). They currently cover placement, movement, capture, line detection, and territory scenarios and are kept in sync with the shared TypeScript engine and Python rules engine.

## ⚠️ Development Notice

**This application is not yet production-ready.** The current codebase includes a working backend game loop and a minimal but functional React client for playing backend-driven games, but the overall UX, multiplayer flows, observability, and test coverage are still incomplete. In its current state, the project is best suited for engine/AI development, rules validation, and early playtesting rather than public release.

The codebase currently provides:

- Infrastructure setup and configuration (Docker, database, Redis, WebSockets, logging, authentication)
- Fully typed shared game state and rules data structures
- A shared TypeScript rules engine (helpers → aggregates → turn orchestrator) with a backend `GameEngine` host wired via `TurnEngineAdapter`; the legacy `RuleEngine` remains only for fenced diagnostics and archived tests
- A Python AI Service wired into backend AI turns via `AIEngine` / `AIServiceClient`
- A React client (LobbyPage + GamePage + BoardView + ChoiceDialog + GameHUD) that can:
  - Create backend games (including AI opponents) via the HTTP API and lobby UI
  - Connect to backend games over WebSockets and play via click-to-move
  - Surface server-driven PlayerChoices for humans (e.g. line rewards, ring elimination)
- A growing Jest test suite around core rules, interaction flows, AI turns, territory disconnection, and sandbox parity

**To contribute or continue development, please review:**

1. [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) - Factual, code-verified analysis of the current state
2. [ARCHITECTURE_ASSESSMENT.md](./ARCHITECTURE_ASSESSMENT.md) - Architecture and refactoring axes (supersedes older codebase evaluation docs)
3. [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Specific bugs, missing features, and prioritization
4. [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) - Phased implementation plan and milestones
5. [CONTRIBUTING.md](./CONTRIBUTING.md) - Development priorities and guidelines

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm 9+
- Docker and Docker Compose
- PostgreSQL 14+ (or use Docker)
- Redis 6+ (or use Docker)

### Development Setup

1. **Clone and Install**

```bash
git clone <repository-url>
cd ringrift
npm install
```

2. **Environment Configuration**

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Database Setup**

```bash
# Start services with Docker (database + redis)
docker-compose up -d postgres redis

# Setup database
npm run db:migrate
npm run db:generate
```

4. **Start Development**

```bash
# Start both frontend and backend (concurrently)
npm run dev

# Or start individually
npm run dev:server  # Backend on :3000 (configurable via PORT)
npm run dev:client  # Frontend on :5173 by default (Vite)
```

#### Dev environment (ports & processes)

- **Backend API + WebSocket server**: `npm run dev:server` → http://localhost:3000 (or `PORT` from `.env`).
- **Frontend (Vite dev server)**: `npm run dev:client` → http://localhost:5173.
- **Python AI Service**: Must be run from the `ai-service/` directory:
  ```bash
  cd ai-service
  source ../.venv/bin/activate  # if using project venv
  uvicorn app.main:app --port 8001 --reload
  # or use the run script:
  ./run.sh
  ```
  → http://localhost:8001 (API docs at `/docs`)

To avoid flaky behaviour in `/game/:gameId` and WebSocket tests, ensure that you only have **one** Node.js backend process listening on port `3000` at a time. Use `npm run dev:server` (or `docker compose up` for the `app` service) as the canonical entrypoint, and avoid starting additional ad‑hoc servers that also bind `3000`.

### Production Deployment

```bash
# Build application
npm run build

# Start with Docker (full stack including monitoring)
# Includes: app, nginx, postgres, redis, ai-service, prometheus, alertmanager, grafana
docker-compose up -d

# Or manual deployment
npm start
```

#### Deployment Topology (Backend)

The current backend is designed and tested for a **single app instance** per environment:

- Game sessions and engines live **in-process** inside the Node app.
- Redis is used for **per-instance** locking and coordination, not as a shared, authoritative game state store.
- Running multiple independent app instances against the same PostgreSQL + Redis cluster **without strong sticky sessions or a shared game-state layer is not supported**.

The topology is controlled via the `RINGRIFT_APP_TOPOLOGY` environment variable:

- `single` (default, safe, supported):
  - The backend assumes it is the **only** app instance talking to this database and Redis for authoritative game sessions.
  - This is the default when `RINGRIFT_APP_TOPOLOGY` is unset.
- `multi-unsafe` (explicitly unsupported multi-instance):
  - Signals an operator is intentionally running multiple app instances **without** infrastructure-enforced sticky sessions or a shared game-state layer.
  - In `NODE_ENV=production`, the server logs a fatal error and **refuses to start** with this topology.
  - In non-production environments (e.g. `development`, `test`), the server logs a **strong warning** and continues, for experimentation only.
- `multi-sticky` (multi-instance with external sticky sessions; still risky):
  - Signals that infrastructure-enforced sticky sessions (HTTP + WebSocket) are in place so that all game-affecting traffic for a given game is routed to the same app instance.
  - The server starts in all environments but logs a **high-visibility warning** that correctness is not guaranteed if sticky sessions are misconfigured.

The default Docker Compose stack (`docker-compose.yml`) runs a **single** `app` instance and sets:

- `RINGRIFT_APP_TOPOLOGY=single`
- `deploy.replicas: 1` for the `app` service

If you manually scale the app service (for example with `docker compose up --scale app=2` or in an orchestrator that increases replicas), you are entering a **non-default, higher-risk mode**. You must:

1. Set `RINGRIFT_APP_TOPOLOGY` to either `multi-unsafe` or `multi-sticky` to acknowledge the risk, and
2. Ensure your infrastructure provides the guarantees required for that mode (especially sticky sessions for WebSocket and game-affecting HTTP traffic in `multi-sticky` deployments).

For more detailed environment and workflow guidance, see [QUICKSTART.md](./QUICKSTART.md).

## 🧪 Sandbox Usage & Local Testing

The `/sandbox` route is backed by a fully rules-complete, client-local engine (`ClientSandboxEngine`) and is the fastest way to experiment with RingRift’s rules without involving the backend.

- **Where:** `/sandbox` in the React client.
- **Who:** 2–4 players with any mix of humans and simple sandbox AIs (sandbox AI chooses randomly among legal options for any `PlayerChoice`).
- **What’s enforced:**
  - Movement, marker behaviour, and overtaking captures.
  - Mandatory chain captures (with `capture_direction` choices).
  - Line detection & processing with graduated rewards.
  - Territory disconnection on square + hex boards (including color-disconnection and self-elimination prerequisite).
  - Ring-elimination and Territory-control victories, surfaced via the shared `VictoryModal`.
  - **How it’s tested:** Dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts` cover chain captures, placement/forced elimination, line processing, Territory disconnection (square + hex), region order, and victory conditions.

Use the sandbox when:

- You want to quickly verify a rule interaction or FAQ scenario visually.
- You’re developing new UI/HUD features that should apply equally to backend games and the sandbox.
- You’re iterating on rules-related code and want a local harness that won’t affect backend persistence.

For backend-driven games, continue to use `/game/:gameId` via the lobby/API; both backend and sandbox views share the same BoardView/ChoiceDialog/VictoryModal stack so improvements to one benefit the other.

### 🧪 Sandbox AI debugging & stall diagnostics

The local sandbox also has **AI stall diagnostics** that are useful when investigating AI-only games that appear to stop making progress:

- Engine-level diagnostics live in [`sandboxAI.maybeRunAITurnSandbox`](src/client/sandbox/sandboxAI.ts:118). When enabled, they:
  - Detect repeated no-op AI turns (unchanged `GameState` hash while the same AI player remains to move in an active game).
  - Emit `[Sandbox AI Stall Diagnostic]` and `[Sandbox AI Stall Detector]` warnings to the console.
  - Append structured entries to `window.__RINGRIFT_SANDBOX_TRACE__` describing each AI turn and any detected stall.

- The `/sandbox` UI includes a **local stall watchdog** (in [`GamePage`](src/client/pages/GamePage.tsx:1585)):
  - When an AI player is to move and the sandbox `GameState` has not advanced for several seconds, a banner appears indicating a potential stall.
  - The banner exposes a “Copy AI trace” action, which serialises `window.__RINGRIFT_SANDBOX_TRACE__` to JSON and copies it to the clipboard (or logs it to the console as a fallback). This trace can then be pasted into a file and used to construct or refine Jest repro tests.

To enable the most detailed sandbox AI stall diagnostics when running tests or a dev server, use:

```bash
# Targeted stall-repro test for a known problematic seed:
RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1 \
RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS=1 \
npm test -- ClientSandboxEngine.aiStall.seed1
```

The `RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS` flag also activates the engine-level stall detector used by the `/sandbox` UI for deeper analysis of long-running AI-vs-AI games.

## 🗺️ Near-Term Focus (High-Level)

For contributors looking for the most impactful work, the near-term focus areas are:

1. **Client component test coverage (CRITICAL)**
   - Anchor RTL coverage with the new DecisionUI harness and extend to `VictoryModal`, `GameHUD`/`GamePage` happy paths, and choice flows. Extract small subcomponents from monolithic files (e.g. `BoardView`, `SandboxGameHost`) to make testing tractable.
   - Guard telemetry (Victory modal, rules UX) for duplicate sends and ensure countdown/reconnection banners render correctly.

2. **Rules/FAQ scenario backfill**
   - Continue expanding `tests/scenarios/**` and `docs/rules/RULES_SCENARIO_MATRIX.md` for multi-choice turns (chain → line → territory) across square8/19 and hex. Add square/19x variants for the new line → territory FAQ scenario.

3. **AI boundary hardening**
   - Add timeout/fallback tests and lightweight latency/error metrics for `AIServiceClient`/`AIEngine`; ensure service-unavailable flows still resolve PlayerChoices cleanly.

4. **UX polish & resilience**
   - Improve timers, reconnection/spectator flows, post-game navigation, chat ergonomics, and Victory modal copy/telemetry; continue mobile/touch responsiveness work.

5. **Production validation & data hygiene**
   - Run sustained load/scaling tests, complete the security hardening review, and keep replay DBs gated via parity/history checks (backup/restore and secrets-rotation drills should stay active). ML-backed AI agents remain future work once the above is stable.

For a detailed, task-level view, see `TODO.md` and `STRATEGIC_ROADMAP.md`.

## 🎮 Game Features

### Core Gameplay

- **Ring Placement**: Strategic positioning of rings on the board
- **Movement Phase**: Tactical ring repositioning
- **Row Formation**: Create rows of markers to remove opponent rings
- **Victory Conditions**: Remove required number of opponent rings and/or win via territory control

### Board Configurations

- **8x8 Square**: Compact tactical gameplay
- **19x19 Square**: Extended strategic depth
- **Hexagonal**: Unique geometric challenges

### Multiplayer Features

- **Real-time Synchronization**: Instant move updates over WebSockets
- **Spectator Mode**: Watch games where `allowSpectators` is enabled with spectator HUD
- **Chat System**: In-game chat per game room
- **Reconnection**: Socket.IO reconnection with reconnection windows and abandonment handling
- **Time Controls**: Persisted per-game time controls
- **Lobby**: Real-time game listing, creation, joining with filters and sorting
- **Ratings**: ELO-based player rankings with leaderboard

### AI Integration

- **Difficulty Levels**: Canonical 1–10 difficulty ladder (Random, Heuristic, Minimax, MCTS, Descent) mirrored between `AIEngine` and the Python AI Service
- **Smart Opponents**: AI moves and PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) routed through the Python AI Service with local heuristics as fallback
- **Mixed Games**: Human-AI combinations supported for 2-4 players
- **Training Infrastructure**: Self-play data generation, heuristic weight optimization (CMA-ES, genetic), model versioning
- **Future Work**: ML-backed agents (NeuralNetAI) for higher difficulty levels

## 🔧 API Documentation

This section describes the **current** HTTP and WebSocket surface exposed by the TypeScript backend as of the latest assessment. For deeper details, see `src/server/routes/*.ts` and `src/server/websocket/server.ts`.

### Authentication Endpoints (`/api/auth`)

```http
POST /api/auth/register       # User registration
POST /api/auth/login          # User authentication
POST /api/auth/refresh        # Exchange refresh token for new access + refresh
POST /api/auth/logout         # Revoke a specific refresh token (best-effort)
POST /api/auth/logout-all     # Revoke all refresh tokens for the current user

POST /api/auth/verify-email   # Placeholder – returns "not implemented yet"
POST /api/auth/forgot-password  # Placeholder – returns "not implemented yet"
POST /api/auth/reset-password   # Placeholder – returns "not implemented yet"
```

### User Endpoints (`/api/users` – authenticated)

```http
GET  /api/users/profile       # Get current user profile
PUT  /api/users/profile       # Update current user profile (username only for now)

GET  /api/users/stats         # Rating, gamesPlayed/gamesWon, recent games, win rate
GET  /api/users/games         # Paginated game history for the current user
GET  /api/users/search        # Search active users by username
GET  /api/users/leaderboard   # Paginated rating leaderboard
```

### Game Management (`/api/games` – authenticated)

```http
GET  /api/games                       # List games the user participates in (filterable by status)
POST /api/games                       # Create new game (boardType, maxPlayers, timeControl, isRated, aiOpponents, etc.)
GET  /api/games/:gameId               # Get full game details + moves for a specific game
POST /api/games/:gameId/join          # Join a waiting game (fills player2–4 slots as available)
POST /api/games/:gameId/leave         # Leave a waiting game or resign from an active one
GET  /api/games/:gameId/moves         # List moves for a game (subject to spectating/participant rules)
GET  /api/games/lobby/available       # List joinable games (waiting status, not already joined)
```

> **Note:** There is **no REST endpoint** for making moves. Moves are made via the WebSocket `player_move` event described below; the server validates and persists moves using `GameEngine` as the single source of truth.

### WebSocket Events

WebSockets are served from the same Node process. Events are defined and handled in `src/server/websocket/server.ts` and `src/server/game/WebSocketInteractionHandler.ts`.

#### Client → Server

```text
join_game             # Join a specific game room (requires auth token)
leave_game            # Leave a game room
player_move           # Submit a move for validation and application
chat_message          # Send in-game chat message
player_choice_response # Respond to a pending PlayerChoice (line reward, ring elimination, etc.)
```

#### Server → Client

```text
game_state            # Game state update (BoardState + GameState + validMoves)
game_over             # Game ended, includes final GameState + GameResult
player_joined         # Notification that another player joined the game
player_left           # Notification that a player left the game
player_disconnected   # Notification that a player disconnected
chat_message          # Broadcast chat message
player_choice_required # Request that a specific player respond to a PlayerChoice
error                 # Generic error message
```

The WebSocket layer sits on top of `GameEngine` and `PlayerInteractionManager`, so all moves and PlayerChoices ultimately flow through the same rules engine as HTTP-triggered flows.

## 🛡️ Security Features

### Authentication & Authorization

- JWT token-based authentication for HTTP and WebSocket connections
- Secure password hashing with bcrypt
- Role field on users (future extension point for admin/moderator roles)
- Rating and statistics fields for players

### API Security

- Rate limiting per endpoint group (auth/game/global)
- CORS configuration (configurable via `CORS_ORIGIN`)
- Helmet security headers
- Input validation with Zod schemas
- SQL injection prevention with Prisma

### Game Security

- Move validation on server via `GameEngine` + `TurnEngineAdapter` over the shared TypeScript rules engine (legacy `RuleEngine` is retained only for fenced diagnostics and archived tests)
- Game state integrity checks
- WebSocket authentication via JWT

## 📊 Performance & Observability

### Backend Optimizations

- Database connection pooling via Prisma/pg
- Centralized logging with Winston (ingested by CI/log tooling)
- Efficient game state serialization through typed GameState / BoardState
- Redis caching for session and game data
- Prometheus metrics via `MetricsService` with HTTP, AI, rules, and orchestrator metrics

### Load Test Results (Wave 7)

- Game creation p95: 15ms (target <800ms) – 53x headroom
- GET /api/games/:id p95: 10.79ms (target <400ms) – 37x headroom
- WebSocket message latency p95: 2ms (target <200ms) – 100x headroom

### Frontend Optimizations (current/planned)

- Vite-based build for fast HMR and optimized production bundles
- React Query for server-state caching (in use, coverage expanding)
- Targeted memoization and derived-state calculations in BoardView
- Future: virtualized lists, service-worker caching, and broader performance profiling

## 🧪 Testing Strategy

> **Test Status (2025-12-07):** 2,987 TypeScript tests passing (CI-gated), 836 Python tests passing, 54 contract vectors at 100% parity. React component coverage remains effectively zero; a DecisionUI harness now anchors PlayerChoice coverage, but broader client/UI suites (BoardView/GameHUD/VictoryModal) are still needed. See `CURRENT_STATE_ASSESSMENT.md` for up-to-date coverage details.

### Backend Testing

```bash
npm test                        # Run all Jest tests
npm run test:watch              # Watch mode
npm run test:coverage           # Coverage report
npm run test:unit               # Unit tests (tests/unit)
npm run test:integration        # Integration tests (tests/integration)

# Rules- and orchestrator-focused lanes
npm run test:ts-rules-engine    # TS rules-level suites (shared engine + orchestrator)
npm run test:orchestrator-parity # Curated orchestrator-ON parity bundle (shared tests, contracts, RulesMatrix, FAQ, key territory tests)

# Additional CI lanes
npm run test:ts-parity          # Trace/host parity and RNG-oriented suites (diagnostic)
npm run test:ts-integration     # Integration suites (WebSocket, routes, full game flows)

# Single-source-of-truth (SSoT) guardrails
npm run ssot-check              # Docs/env/CI/rules SSoT checks against the canonical rules spec + shared TS engine, and legacy-path fences

# Orchestrator invariant soaks (TS shared engine, backend host)
npm run soak:orchestrator       # Multi-game invariant soak (S, structure, ACTIVE-no-move)
npm run soak:orchestrator:smoke # Single short backend game on square8, fails fast on violations

# Orchestrator S-invariant regression harness (seeded, diagnostic)
npm run test:orchestrator:s-invariant  # Replays seeded backend games promoted from soak S_INVARIANT_DECREASED traces
```

### Trace parity & GameTrace

A set of AI-heavy suites compare backend and sandbox behaviour step-by-step using a shared **GameTrace** format and replay helpers:

- `GameHistoryEntry` / `GameTrace` live in `src/shared/types/game.ts` and are the canonical event-sourcing types for both engines.
- `tests/utils/traces.ts` provides:
  - `runSandboxAITrace(boardType, numPlayers, seed, maxSteps)` → generate a sandbox AI-vs-AI trace using `ClientSandboxEngine`.
  - `replayTraceOnBackend(trace)` → rebuild a backend `GameEngine` from the trace’s initial state and replay the same canonical moves.
  - `replayTraceOnSandbox(trace)` → rebuild a fresh `ClientSandboxEngine` and re-apply the same canonical moves.
- Core parity/debug suites:
  - `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`
  - `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`
  - `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`

Diagnostics can be enabled via:

- `RINGRIFT_TRACE_DEBUG` – write structured JSON snapshots for trace parity runs to `logs/ai/trace-parity.log` (sandbox opening sequence, backend mismatches, S/hashes, valid move lists).
- `RINGRIFT_AI_DEBUG` – mirror AI diagnostics to the console and enable extra sandbox AI debug logs, in addition to the existing `logs/ai/*.log` streams used by AI simulation tests.

For more details and usage patterns, see `tests/README.md`.

### Frontend / Sandbox Testing

```bash
# Currently all tests are Jest-based and live under tests/unit
# including client-local sandbox engine tests.
npm test
```

### Test Coverage Goals

- Unit tests for game logic (shared TS engine aggregates + GameEngine host; legacy `RuleEngine` only where needed for archived diagnostics)
- Integration tests for API endpoints and WebSocket flows
- AI boundary tests (AIEngine, AIServiceClient, AIInteractionHandler)
- UI-level tests for critical components (BoardView, GamePage, ChoiceDialog, VictoryModal)
- Scenario-driven tests derived from `ringrift_complete_rules.md` and the FAQ

## 📈 Monitoring & Analytics

### Application Monitoring

- **Logging**: Structured logging with Winston
- **Metrics**: Prometheus metrics via `MetricsService` (HTTP, AI, rules, lifecycle, orchestrator)
- **Dashboards**: 3 Grafana dashboards (game-performance, rules-correctness, system-health) with 22 panels
- **Alerts**: Prometheus alert rules in `monitoring/prometheus/alerts.yml`
- **Load Testing**: k6 framework with 4 production-scale scenarios
- **CI/CD**: GitHub Actions with Jest coverage publishing to Codecov

### Game Analytics

- Rating/ELO tracking with `RatingService.ts`
- Game records with move history and metadata
- Self-play data for AI training
- Replay system with DB storage

## 🔄 Development Workflow

### Code Quality

- TypeScript for type safety
- ESLint for code standards (`npm run lint`)
- Prettier for formatting
- Husky + lint-staged for git hooks (`npm run prepare` installs hooks)
- Conventional commits recommended

### CI/CD Pipeline

- GitHub Actions workflow in `.github/workflows/ci.yml` with jobs for:
  - **Lint and Type Check** – ESLint + TypeScript compilation for server/client.
  - **TS Rules Engine (rules-level)** – runs `npm run test:ts-rules-engine` with the shared TS engine + orchestrator adapter forced ON.
  - **TS Orchestrator Parity (adapter-ON)** – runs `npm run test:orchestrator-parity` over a curated, high-signal subset of suites (shared tests, contract vectors, RulesMatrix, FAQ scenarios, critical territory/disconnection tests). This job is intended to be a required gate for `main`.
  - **TS Parity / TS Integration** – additional Jest lanes for trace/host parity and integration tests (WebSocket, routes, full game flows).
  - **SSoT Drift Guards** – runs `npm run ssot-check` to enforce docs/env/CI/rules SSoT and fence legacy `RuleEngine` / sandbox helpers.
  - **Python lanes** – Python rules-parity, core tests, and dependency audit for `ai-service`.
  - **E2E Tests** – Playwright E2E tests against a full Dockerised stack.
  - **Build (server + client)** and Docker build jobs for packaging.
  - Security scans (npm audit + Snyk)
- Docker build verification via `docker/build-push-action`

## 📚 Additional Resources

### Game Rules

- Complete rule documentation in `ringrift_complete_rules.md`
- Compact rules summary in `ringrift_compact_rules.md`
- Future: interactive tutorial system, strategy guides, and video demonstrations

### Development Guides

- Architecture assessment: `ARCHITECTURE_ASSESSMENT.md`
- Current state assessment: `CURRENT_STATE_ASSESSMENT.md`
- Strategic roadmap: `STRATEGIC_ROADMAP.md`
- Playable game implementation plan (historical): `archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md`
- Test documentation: `tests/README.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for project-specific guidelines and priority areas.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

This repository is currently geared toward engine, AI, and rules development. When hosted on a public Git provider, this section should be updated with real URLs for:

- Issues (bug reports, feature requests)
- Discussions (design questions, strategy, and rules clarifications)

For now, please use the documents listed in the **Documentation Map** above to understand the current state, roadmap, and contribution priorities.

---

Built by the RingRift Team
