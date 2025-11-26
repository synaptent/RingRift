# RingRift - Multiplayer Strategy Game

⚠️ **PROJECT STATUS: ENGINE/AI-FOCUSED BETA – BACKEND PLAY & AI TURNS WORK; UX & SCENARIO TESTS STILL IN PROGRESS** ⚠️

> **Important:** Core game mechanics are largely implemented, and there is now a **playable backend game flow**: the server’s `GameEngine` drives rules, WebSocket-backed games use it as the source of truth, the React client renders boards and submits moves, and AI opponents can make moves via the Python AI service. In addition, a **client-local sandbox engine** (`ClientSandboxEngine`) powers the `/sandbox` route with strong rules parity and dedicated Jest suites for movement, captures, lines, territory, and victory checks. However, the UI/UX is still evolving and there is not yet a comprehensive scenario matrix for every rule/FAQ example. See [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) for code‑verified status.

A web-based multiplayer implementation of the RingRift strategy game supporting 2-4 players with flexible human/AI combinations across multiple board configurations.

## 📋 Current Status

**Last Updated:** November 22, 2025
**Verification:** Code-verified assessment (see `CURRENT_STATE_ASSESSMENT.md`)
**Overall Progress:** Strong foundation with critical gaps; see `CURRENT_STATE_ASSESSMENT.md` for the latest high-level summary.

### ✅ What's Working

- ✅ Project infrastructure (Docker, database, Redis, WebSocket)
- ✅ **Session Management** - Robust `GameSessionManager` with distributed locking
- ✅ **Rules Facade** - `RulesBackendFacade` abstracting Python/TS engine parity
- ✅ TypeScript type system and architecture
- ✅ Comprehensive game rules documentation
- ✅ Server and client scaffolding
- ✅ **Marker system** - Placement, flipping, collapsing fully functional
- ✅ **Movement validation** - Distance rules, path checking working
- ✅ **Basic and chained captures** - Overtaking captures and mandatory continuation implemented
- ✅ **Line detection** - All board types (8x8, 19x19, hexagonal)
- ✅ **Territory disconnection** - Detection and processing implemented
- ✅ **Phase transitions** - Correct game flow through all phases
- ✅ **Player state tracking** - Ring counts, eliminations, territory
- ✅ **Hexagonal board support** - Full 331-space board validated
- ✅ **Client-local sandbox engine** - `/sandbox` uses `ClientSandboxEngine` plus `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxLinesEngine.ts`, `sandboxTerritoryEngine.ts`, and `sandboxVictory.ts` to run full games in the browser (movement, captures, lines, territory, and ring/territory victories) with dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts`.

### ⚠️ Critical Gaps (Blocks Production-Quality Play)

- ⚠️ **Player choice system is implemented but not yet deeply battle-tested** – Shared types and `PlayerInteractionManager` exist and GameEngine uses them for line order, line reward, ring elimination, region order, and capture direction. `WebSocketInteractionHandler`, `GameContext`, and `ChoiceDialog` wire these choices to human clients for backend-driven games, and `AIInteractionHandler` answers choices for AI players via local heuristics and (for some choices) the Python AI service. What’s missing is broad scenario coverage (all FAQ/rules examples), polished UX around errors/timeouts, and full test coverage of complex multi-choice turns.
- ⚠️ **Chain captures enforced engine-side; more edge-case tests still needed** – GameEngine maintains internal chain-capture state and uses `CaptureDirectionChoice` via `PlayerInteractionManager` to drive mandatory continuation when multiple follow-up captures exist. Core behaviour is covered by focused unit/integration tests, but additional rule/FAQ scenarios (e.g. complex 180° and cyclic patterns) and full UI/AI flows still need to be exercised and encoded as named scenarios.
- ⚠️ **UI is functional but not polished** – Board rendering, a local sandbox, and backend game mode exist for 8x8, 19x19, and hex boards. Backend games now support “click source, click highlighted destination” moves and server-driven choices, and AI opponents can take turns. However, the HUD, timers, post-game flows, and multiplayer UX (spectators, reconnection, chat) still need refinement.
- ⚠️ **Testing is still evolving relative to rules complexity** – Dedicated Jest suites now cover the client-local sandbox engine (movement, captures, lines, territory, victory), backend engine/interaction paths, and a rules/FAQ scenario matrix (`RULES_SCENARIO_MATRIX.md`) including dedicated FAQ suites under `tests/scenarios/FAQ_*.test.ts`. Several seeded trace-parity suites and multi-host/AI fuzz harnesses remain diagnostic or partial rather than hard CI gates; see `CURRENT_STATE_ASSESSMENT.md` and `KNOWN_ISSUES.md` for details.
- ⚠️ **AI Service integration is move- and choice-focused but still evolving** – The Python AI Service is integrated into the turn loop via `AIEngine`/`AIServiceClient` and `WebSocketServer.maybePerformAITurn`, so AI players can select moves in backend games. The service is also used for several PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) behind `globalAIEngine`/`AIInteractionHandler`, with remaining choices currently answered via local heuristics. Higher-difficulty tactical behaviour and ML-based agents are future work.

### 🎯 What This Means

**Can Do (today):**

- Create games via the HTTP API and from the React lobby (including AI opponent configuration).
- Play backend-driven games end-to-end using the React client (BoardView + GamePage) with click-to-move and server-validated moves.
- Have AI opponents take turns in backend games via the Python AI Service and/or local heuristics, using the unified `AIProfile` / `aiOpponents` pipeline.
- Process lines and Territory disconnection, forced elimination, and hex boards through the shared GameEngine.
- Track full game state (phases, players, rings, Territory, timers) and broadcast updates over WebSockets.
- Run full, rules-complete games in the `/sandbox` route using the client-local `ClientSandboxEngine` with simple random-choice AI for all PlayerChoices, reusing the same BoardView/ChoiceDialog/VictoryModal patterns as backend games.

**Cannot Reliably Do (yet):**

- Rely on tests for full rule coverage (scenario/edge-case tests and coverage are still incomplete).
- Guarantee every chain capture and PlayerChoice edge case from the rules/FAQ is battle-tested and bug-free.
- Offer a fully polished UX (HUD, timers, post-game flows, and lobby/matchmaking are still basic).
- Use the AI Service for all PlayerChoice decisions (several choices are still answered via local heuristics only).
- Provide production-grade multiplayer with robust lobbies, matchmaking, reconnection UX, spectators, and chat.

**For complete assessment, see [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)**  
**For detailed issues, see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md)**  
**For roadmap, see [TODO.md](./TODO.md) and [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md)**

---

## 📚 Documentation Map & Canonical Sources

To understand the project and know which documents are authoritative for each area, use these documents. When information in older planning or analysis docs conflicts with the files listed below, these **canonical sources win**.

### For Players & Designers (Rules)

- `ringrift_complete_rules.md` – **The authoritative rulebook.** Full, narrative rules for players and designers.
- `ringrift_compact_rules.md` – Compact, implementation-oriented spec for engine/AI authors.
- `archive/RULES_ANALYSIS_PHASE2.md` – Consistency and strategic assessment of the rules.

### For Developers (Architecture, Status, Setup)

- **Status & Roadmap (Canonical, Living)**
  - `CURRENT_STATE_ASSESSMENT.md` – Factual, code-verified current state (includes implementation status).
  - `TODO.md` – Phase-structured task tracker.
  - `STRATEGIC_ROADMAP.md` – Phased roadmap to production.
  - `KNOWN_ISSUES.md` – Current P0/P1 bugs and gaps.

- **Architecture & Design**
  - `ARCHITECTURE_ASS

essment.md` – Comprehensive architecture review and future design plans.

- `AI_ARCHITECTURE.md` – AI Service architecture, assessment, and improvement plans (with cross-links to training pipelines and incidents).
- `RULES_ENGINE_ARCHITECTURE.md` – Python Rules Engine architecture and rollout strategy.

- **Subsystem Guides**
  - `tests/README.md` – Jest setup, test structure, and the rules/FAQ → scenario test matrix.
  - `RULES_SCENARIO_MATRIX.md` – Canonical mapping of rules/FAQ sections to specific Jest test suites.
  - `ai-service/README.md` – Python AI Service (Random/Heuristic AI, endpoints, setup).
  - [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1) – AI Service training pipelines, self-play and Territory dataset generation CLIs, JSONL schema, and seed behaviour.
  - `CONTRIBUTING.md` – Contribution workflow and historical phase breakdown.

- **Historical Plans & Evaluations**
  - [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1) – Incident report for the TerritoryMutator vs `GameEngine.apply_move()` divergence and its fix, with tests and follow-up tasks.
  - Docs under `archive/` and `deprecated/` – Earlier architecture and improvement plans, preserved for context only.

### 🔗 Developer Quick Links

- **Start Here:** [docs/INDEX.md](./docs/INDEX.md) – Concise entry point for new contributors.
- **Getting Started:** [QUICKSTART.md](./QUICKSTART.md)
- **Contributing:** [CONTRIBUTING.md](./CONTRIBUTING.md)
- **Testing Guide:** [tests/README.md](./tests/README.md)

## 🎯 Overview

RingRift is a sophisticated turn-based strategy game featuring:

- **Multiple Board Types**: 8x8 square, 19x19 square, and hexagonal layouts
- **Flexible Player Support**: 2-4 players with human/AI combinations
- **Real-time Multiplayer**: WebSocket-based live gameplay (engine and transport implemented; UX still evolving)
- **Spectator Mode**: Watch games in progress when `allowSpectators` is enabled on a game
- **Rating System**: ELO-based player rankings (rating fields and stats exist; full rating algorithms WIP)
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
- **Monitoring Stack**: Prometheus + Grafana containers are scaffolded in `docker-compose.yml` (application-level metrics wiring still future work)
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
# TypeScript contract tests (14 vectors)
npm test -- tests/contracts/contractVectorRunner.test.ts

# Python contract tests (15 tests, 12 vectors)
./scripts/run-python-contract-tests.sh
```

Test vectors are located in [`tests/fixtures/contract-vectors/v2/`](tests/fixtures/contract-vectors/v2/) and cover:

- Placement (3 vectors)
- Movement (3 vectors)
- Capture (2 vectors)
- Line detection (2 vectors)
- Territory (2 vectors)

## ⚠️ Development Notice

**This application is not yet production-ready.** The current codebase includes a working backend game loop and a minimal but functional React client for playing backend-driven games, but the overall UX, multiplayer flows, observability, and test coverage are still incomplete. In its current state, the project is best suited for engine/AI development, rules validation, and early playtesting rather than public release.

The codebase currently provides:

- Infrastructure setup and configuration (Docker, database, Redis, WebSockets, logging, authentication)
- Fully typed shared game state and rules data structures
- A largely implemented GameEngine + BoardManager + RuleEngine for all board types
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
- **Python AI Service**: From the project root, `cd ai-service && ./run.sh` → http://localhost:8001.

To avoid flaky behaviour in `/game/:gameId` and WebSocket tests, ensure that you only have **one** Node.js backend process listening on port `3000` at a time. Use `npm run dev:server` (or `docker compose up` for the `app` service) as the canonical entrypoint, and avoid starting additional ad‑hoc servers that also bind `3000`.

### Production Deployment

```bash
# Build application
npm run build

# Start with Docker (full stack: app, nginx, postgres, redis, prometheus, grafana)
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

1. **Scenario-driven tests (rules & FAQ parity)**
   - Add Jest tests that encode specific examples from `ringrift_complete_rules.md` and the FAQ (Q1–Q24), especially:
     - Complex chain capture patterns (180° reversals, cycles) on 8×8 and 19×19.
     - Combined line + Territory situations that involve multiple PlayerChoices in one turn.
     - Hex-board edge cases for lines, Territory, and forced elimination.
     - Mirror high-value Rust tests from `RingRift Rust/ringrift/tests/` (starting with chain capture and Territory) into Jest where feasible.
     2. **HUD & game lifecycle polish (GamePage/GameHUD)**
        - Implement a richer HUD in `GameHUD` for both backend and sandbox games:
          - Clear current player + phase indicators.
          - Ring counts (in hand/on board/eliminated) per player.
          - Territory-space counts, driven from `board.collapsedSpaces` and GameState.
          - Timers (display-only for now) based on `timeControl` and per-player `timeRemaining`.

   - Improve end-of-game UX using `VictoryModal` for both backend and sandbox modes, with a clear route back to the lobby.

2. **AI boundary hardening & observability**
   - Extend `AIEngine`/`AIServiceClient` tests to cover failures, timeouts, and fallback behaviour for move and choice endpoints.
   - Add lightweight logging/metrics around AI calls so we can see latency, error rates, and fallback usage in development.

For a detailed, task-level view, see `TODO.md` (especially Phase 0/1/3S near-term checklists) and `STRATEGIC_ROADMAP.md`.

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

### Multiplayer Features _(planned/partially implemented)_

- **Real-time Synchronization**: Instant move updates over WebSockets
- **Spectator Mode**: Watch games where `allowSpectators` is enabled
- **Chat System**: In-game chat per game room
- **Reconnection**: Basic Socket.IO reconnection; dedicated reconnection UX still future work
- **Time Controls**: Persisted per-game time controls, with more UI polish planned

### AI Integration _(planned/partially implemented)_

- **Difficulty Levels**: AI profiles with difficulty ratings
- **Smart Opponents**: Python AI Service-backed RandomAI + HeuristicAI (production-ready)
- **Experimental AI**: Minimax and MCTS implementations exist but are experimental (see `ai-service/README.md`)
- **Mixed Games**: Human-AI combinations supported
- **Future Work**: Stronger tactical AI and learning algorithms

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

- Move validation on server via `RuleEngine` and `GameEngine`
- Game state integrity checks
- WebSocket authentication via JWT

## 📊 Performance & Observability

### Backend Optimizations (current)

- Database connection pooling via Prisma/pg
- Centralized logging with Winston (ingested by CI/log tooling)
- Efficient game state serialization through typed GameState / BoardState

### Planned / Early-Stage Work

- Redis caching for frequently accessed data (infrastructure in place, usage expanding)
- More aggressive game-state diffing and delta updates over WebSockets
- Application-level Prometheus metrics and dashboards using the existing Prometheus/Grafana stack in `docker-compose.yml`

### Frontend Optimizations (current/planned)

- Vite-based build for fast HMR and optimized production bundles
- React Query for server-state caching (in use, coverage expanding)
- Targeted memoization and derived-state calculations in BoardView
- Future: virtualized lists, service-worker caching, and broader performance profiling

## 🧪 Testing Strategy

> The test setup described here reflects the **intended structure**. As of the latest assessment, there is a **growing suite of unit and integration tests** across the engine, WebSocket, AI boundary, and client-local sandbox, but coverage is still incomplete. See `CURRENT_STATE_ASSESSMENT.md` for up-to-date coverage details and gaps.

### Backend Testing

```bash
npm test                   # Run all Jest tests
npm run test:watch         # Watch mode
npm run test:coverage      # Coverage report
npm run test:unit          # Unit tests (tests/unit)
npm run test:integration   # Integration tests (tests/integration if present)
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

- Unit tests for game logic (BoardManager, RuleEngine, GameEngine)
- Integration tests for API endpoints and WebSocket flows
- AI boundary tests (AIEngine, AIServiceClient, AIInteractionHandler)
- UI-level tests for critical components (BoardView, GamePage, ChoiceDialog, VictoryModal)
- Scenario-driven tests derived from `ringrift_complete_rules.md` and the FAQ

## 📈 Monitoring & Analytics _(future/partially implemented)_

### Application Monitoring

- Structured logging with Winston (currently active)
- CI pipeline with Jest coverage publishing to Codecov (`.github/workflows/ci.yml`)
- Planned: error tracking, runtime metrics, and dashboards using Prometheus/Grafana

### Game Analytics (planned)

- Player behavior tracking
- Game duration statistics
- Move pattern analysis
- Rating system metrics
- User engagement data

## 🔄 Development Workflow

### Code Quality

- TypeScript for type safety
- ESLint for code standards (`npm run lint`)
- Prettier for formatting
- Husky + lint-staged for git hooks (`npm run prepare` installs hooks)
- Conventional commits recommended

### CI/CD Pipeline

- GitHub Actions workflow in `.github/workflows/ci.yml` with jobs for:
  - Lint + type check
  - Jest tests + coverage upload to Codecov
  - Build (server + client)
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
- Playable game implementation plan (historical): `deprecated/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md`
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
