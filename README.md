# RingRift - Multiplayer Strategy Game

⚠️ **PROJECT STATUS: CORE LOGIC ~70–75% COMPLETE – BACKEND PLAY AND AI TURNS WORK, UI & TESTING STILL EARLY** ⚠️

> **Important:** Core game mechanics are largely implemented (~72%), and there is now a **playable backend game flow**: the server’s `GameEngine` drives rules, WebSocket-backed games use it as the source of truth, the React client renders boards and submits moves, and AI opponents can make moves via the Python AI service. In addition, a **client-local sandbox engine** (`ClientSandboxEngine`) powers the `/sandbox` route with strong rules parity and dedicated Jest suites for movement, captures, lines, territory, and victory checks. However, the UI is still minimal, end-to-end UX is rough, and test coverage is low. See [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) for a verified breakdown.

A web-based multiplayer implementation of the RingRift strategy game supporting 2-4 players with flexible human/AI combinations across multiple board configurations.

## 📋 Current Status

**Last Updated:** November 14, 2025  
**Verification:** Code-verified assessment (see `CURRENT_STATE_ASSESSMENT.md`)  
**Overall Progress:** 58% Complete (strong foundation, critical gaps remain)

### ✅ What's Working (75% of Core Logic)
- ✅ Project infrastructure (Docker, database, Redis, WebSocket)
- ✅ TypeScript type system and architecture (100%)
- ✅ Comprehensive game rules documentation
- ✅ Server and client scaffolding
- ✅ **Marker system** - Placement, flipping, collapsing fully functional
- ✅ **Movement validation** - Distance rules, path checking working
- ✅ **Basic captures** - Single captures work correctly
- ✅ **Line detection** - All board types (8x8, 19x19, hexagonal)
- ✅ **Territory disconnection** - Detection and processing implemented
- ✅ **Phase transitions** - Correct game flow through all phases
- ✅ **Player state tracking** - Ring counts, eliminations, territory
- ✅ **Hexagonal board support** - Full 331-space board validated
- ✅ **Client-local sandbox engine** - `/sandbox` uses `ClientSandboxEngine` plus `sandboxMovement.ts`, `sandboxCaptures.ts`, `sandboxLinesEngine.ts`, `sandboxTerritoryEngine.ts`, and `sandboxVictory.ts` to run full games in the browser (movement, captures, lines, territory, and ring/territory victories) with dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts`.

### ⚠️ Critical Gaps (Blocks Production-Quality Play)
- ⚠️ **Player choice system is implemented but not yet deeply battle-tested** – Shared types and `PlayerInteractionManager` exist and GameEngine now uses them for line order, line reward, ring elimination, region order, and capture direction. `WebSocketInteractionHandler`, `GameContext`, and `ChoiceDialog` wire these choices to human clients for backend-driven games, and `AIInteractionHandler` answers choices for AI players via local heuristics. What’s missing is broad scenario coverage (all FAQ/rules examples), polished UX around errors/timeouts, and – optionally – AI-service–backed choice decisions.
- ⚠️ **Chain captures enforced engine-side; more edge-case tests still needed** – GameEngine maintains internal chain-capture state and uses `CaptureDirectionChoice` via `PlayerInteractionManager` to drive mandatory continuation when multiple follow-up captures exist. Core behaviour is covered by focused unit/integration tests, but additional rule/FAQ scenarios (e.g. complex 180° and cyclic patterns) and full UI/AI flows still need to be exercised.
- ⚠️ **UI is functional but minimal** – Board rendering, a local sandbox, and backend game mode exist for 8x8, 19x19, and hex boards. Backend games now support “click source, click highlighted destination” moves and server-driven choices, and AI opponents can take turns. However, the HUD, polish, and game lifecycle UX are still early.
- ❌ **Limited testing** – Dedicated Jest suites now cover the client-local sandbox engine (movement, captures, lines, territory, victory) and several backend engine/interaction paths, but overall coverage is still low and there is no comprehensive scenario suite derived from the rules/FAQ.
- ⚠️ **AI service integration is move- and choice-focused but still evolving** – The Python AI microservice is integrated into the turn loop via `AIEngine`/`AIServiceClient` and `WebSocketServer.maybePerformAITurn`, so AI players can select moves in backend games. The service is also used for several PlayerChoices (`line_reward_option`, `ring_elimination`, `region_order`) behind `globalAIEngine`/`AIInteractionHandler`, with remaining choices currently answered via local heuristics. Higher-difficulty tactical behaviour still depends on future work.

### 🎯 What This Means
**Can Do (today):**
- Create games via the HTTP API and from the React lobby (including AI opponent configuration).
- Play backend-driven games end-to-end using the React client (BoardView + GamePage) with click-to-move and server-validated moves.
- Have AI opponents take turns in backend games via the Python AI service, using the unified `AIProfile` / `aiOpponents` pipeline.
- Process lines and territory disconnection, forced elimination, and hex boards through the shared GameEngine.
- Track full game state (phases, players, rings, territory, timers) and broadcast updates over WebSockets.
- Run full, rules-complete games in the `/sandbox` route using the client-local `ClientSandboxEngine` with simple random-choice AI for all PlayerChoices, reusing the same BoardView/ChoiceDialog/VictoryModal patterns as backend games.

**Cannot Do (yet):**
- Rely on tests for full rule coverage (scenario/edge-case tests and coverage are still incomplete).
- Guarantee every chain capture and PlayerChoice edge case from the rules/FAQ is battle-tested and bug-free.
- Enjoy a fully polished UX (HUD, timers, post-game flows, and lobby/matchmaking are still basic).
- Use the AI service for all PlayerChoice decisions (several choices are still answered via local heuristics only).
- Play production-grade multiplayer with lobbies, matchmaking, reconnection UX, and spectators.

### 📊 Component Status
| Component | Status | Completion |
|-----------|--------|-----------|
| Type System | ✅ Complete | 100% |
| Board Manager | ✅ Complete | 90% |
| Game Engine | ⚠️ Partial | 75% |
| Rule Engine | ⚠️ Partial | 60% |
| Frontend UI | ⚠️ Basic board & choice UI + client-local sandbox engine | 30% |
| AI Integration | ⚠️ Moves + some choices service-backed | 60% |
| Testing | ⚠️ Growing but incomplete | 10% |
| Multiplayer | ⚠️ Basic backend play, no full lobby yet | 30% |

**For complete assessment, see [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)**  
**For detailed issues, see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md)**  
**For roadmap, see [TODO.md](./TODO.md)**

---

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

## ⚠️ Development Notice

**This application is not yet production-ready.** The current codebase includes a working backend game loop and a minimal but functional React client for playing backend-driven games, but the overall UX, multiplayer flows, observability, and test coverage are still incomplete. In its current state, the project is best suited for engine/AI development, rules validation, and early playtesting rather than public release.

The codebase currently provides:
- Infrastructure setup and configuration (Docker, database, Redis, WebSockets, logging, authentication)
- Fully typed shared game state and rules data structures
- A largely implemented GameEngine + BoardManager + RuleEngine for all board types
- A Python AI service wired into backend AI turns via `AIEngine` / `AIServiceClient`
- A minimal React client (LobbyPage + GamePage + BoardView + ChoiceDialog) that can:
  - Create backend games (including AI opponents) via the HTTP API and lobby UI
  - Connect to backend games over WebSockets and play via click-to-move
  - Surface server-driven PlayerChoices for humans (e.g. line rewards, ring elimination)
- A growing but still limited Jest test suite around core rules, interaction flows, AI turns, and territory disconnection

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

### Production Deployment

```bash
# Build application
npm run build

# Start with Docker (full stack: app, nginx, postgres, redis, prometheus, grafana)
docker-compose up -d

# Or manual deployment
npm start
```

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
  - Ring-elimination and territory-control victories, surfaced via the shared `VictoryModal`.
- **How it’s tested:** Dedicated Jest suites under `tests/unit/ClientSandboxEngine.*.test.ts` cover chain captures, placement/forced elimination, line processing, territory disconnection (square + hex), region order, and victory conditions.

Use the sandbox when:
- You want to quickly verify a rule interaction or FAQ scenario visually.
- You’re developing new UI/HUD features that should apply equally to backend games and the sandbox.
- You’re iterating on rules-related code and want a local harness that won’t affect backend persistence.

For backend-driven games, continue to use `/game/:gameId` via the lobby/API; both backend and sandbox views share the same BoardView/ChoiceDialog/VictoryModal stack so improvements to one benefit the other.

## 🗺️ Near-Term Focus (High-Level)

For contributors looking for the most impactful work, the near-term focus areas are:

1. **Scenario-driven tests (rules & FAQ parity)**
   - Add Jest tests that encode specific examples from `ringrift_complete_rules.md` and the FAQ (Q1–Q24), especially:
     - Complex chain capture patterns (180° reversals, cycles) on 8×8 and 19×19.
     - Combined line + territory situations that involve multiple PlayerChoices in one turn.
     - Hex-board edge cases for lines, territory, and forced elimination.
   - Mirror high-value Rust tests from `RingRift Rust/ringrift/tests/` (starting with chain capture and territory) into Jest where feasible.

2. **HUD & game lifecycle polish (GamePage/GameHUD)**
   - Implement a richer HUD in `GameHUD` for both backend and sandbox games:
     - Clear current player + phase indicators.
     - Ring counts (in hand/on board/eliminated) per player.
     - Territory-space counts, driven from `board.collapsedSpaces` and GameState.
     - Timers (display-only for now) based on `timeControl` and per-player `timeRemaining`.
   - Improve end-of-game UX using `VictoryModal` for both backend and sandbox modes, with a clear route back to the lobby.

3. **AI boundary hardening & observability**
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

### Multiplayer Features *(planned/partially implemented)*
- **Real-time Synchronization**: Instant move updates over WebSockets
- **Spectator Mode**: Watch games where `allowSpectators` is enabled
- **Chat System**: In-game chat per game room
- **Reconnection**: Basic Socket.IO reconnection; dedicated reconnection UX still future work
- **Time Controls**: Persisted per-game time controls, with more UI polish planned

### AI Integration *(planned/partially implemented)*
- **Difficulty Levels**: AI profiles with difficulty ratings
- **Smart Opponents**: Python service-backed RandomAI + HeuristicAI
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

## 📈 Monitoring & Analytics *(future/partially implemented)*

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
- Playable game implementation plan: `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md`
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

- Issues: [GitHub Issues](link-to-issues)
- Discussions: [GitHub Discussions](link-to-discussions)

---

Built by the RingRift Team
