# RingRift Quick Start Guide

**Doc Status (2025-11-27): Active (developer quickstart)**

- Step-by-step setup and deployment guide for the TS backend, React client, and Python AI service.
- Not a rules or lifecycle SSoT; for rules semantics defer to `RULES_CANONICAL_SPEC.md` + shared TS engine (`src/shared/engine/**`), and for lifecycle semantics defer to `docs/CANONICAL_ENGINE_API.md` and shared WebSocket types/schemas.

## Game Completion

When a game ends, you'll see a comprehensive victory screen showing:

- **Winner and Victory Condition**: Clear indication of who won and how (ring elimination, territory majority, last player standing, or stalemate)
- **Final Statistics**: Complete comparison of all players including:
  - Rings on board (still in play)
  - Rings lost (eliminated from play)
  - Territory spaces controlled
  - Total moves made
- **Game Summary**: Board type, total turns played, player count, and rated status

### Available Actions

After viewing the victory screen, you can:

- **Return to Lobby**: Start a new game or join an existing one
- **Request Rematch**: Challenge the same players to another game (multiplayer only)
- **Close**: View the final board state

### Victory Conditions

Games can end in several ways:

- **Ring Elimination** (🏆): Eliminate more than 50% of total rings in play
- **Territory Majority** (🏰): Control more than 50% of the board as collapsed Territory
- **Last Player Standing** (👑): Be the only player who can still make legal moves
- **Stalemate Draw** (🤝): When no moves are possible, the winner is determined by:
  - Most Territory spaces (higher priority)
  - Most rings eliminated (if Territory is tied)
  - Most markers remaining (if still tied)
  - Last player to complete a valid action (final tiebreaker)

---

# RingRift Quick Start Guide

## Understanding the Game HUD

The heads-up display (HUD) provides real-time information about the game state:

### Phase Indicator

The colored banner at the top shows the current game phase with an icon and description:

- 🎯 **Blue - Placement Phase**: Place your rings on the board
- ⚡ **Green - Movement Phase**: Move a stack or capture opponent pieces
- ⚔️ **Orange - Capture Phase**: Execute a capture move
- 🔗 **Orange - Chain Capture**: Continue capturing or end your turn
- 📏 **Purple - Line Reward**: Choose how to process your line
- 🏰 **Pink - Territory Claim**: Choose regions to collapse

### Turn Counter

Displays the current turn number and move count to track game progress.

### Player Cards

Each player card shows:

- **Name and Color**: Player identifier with color indicator
- **AI Indicator**: Shows if player is AI-controlled, with difficulty level and type
- **Current Turn Badge**: Highlights the active player
- **Timer**: Countdown timer for timed games (turns red when under 1 minute)
- **Ring Statistics**:
  - In Hand: Rings available for placement
  - On Board: Rings currently in play
  - Lost: Permanently eliminated rings
- **Territory Count**: Number of spaces controlled as Territory

### Connection Status

Top bar shows WebSocket connection state and whether you're spectating.

This guide focuses on getting a **local development environment** running quickly (backend + frontend) and then wiring up the **Python AI Service** used for AI turns and some PlayerChoices.

If you only want the AI Service, skip to **[AI Service Quick Start](#ai-service-quick-start)**.

---

## 1. Core App: Backend + Frontend

### 1.1 Prerequisites

- Node.js **18+** and npm **9+**
- Docker + Docker Compose (for PostgreSQL and Redis)
- Python 3.11+ (for the AI service, optional for initial setup)

### 1.2 Clone & Install

```bash
git clone <repository-url>
cd RingRift
npm install
```

### 1.3 Environment Configuration

Create a local `.env` based on the example:

```bash
cp .env.example .env
# Then edit .env as needed
```

Key values (defaults are usually fine for local dev):

- `DATABASE_URL` – PostgreSQL connection string (matches docker-compose defaults by default).
- `REDIS_URL` – Redis connection string.
- `JWT_SECRET`, `JWT_REFRESH_SECRET` – any non-empty secrets for local usage.
- `CORS_ORIGIN` – usually `http://localhost:5173` for the Vite dev client.
- `AI_SERVICE_URL` – URL for the Python AI Service. For local dev without Docker: `http://localhost:8001`. When running the full Docker Compose stack, the `app` service is configured to talk to `http://ai-service:8001` inside the Docker network.

### 1.4 Start Database & Redis (Docker)

From the project root:

```bash
# Start just PostgreSQL and Redis for development
docker compose up -d postgres redis
```

This matches the services defined in `docker-compose.yml`:

- PostgreSQL: `postgres://ringrift:<DB_PASSWORD>@localhost:5432/ringrift`
- Redis: `redis://localhost:6379`

### 1.5 Apply Database Migrations

```bash
npm run db:migrate
npm run db:generate
# Optional: seed once you have a seed script
# npm run db:seed
```

This uses Prisma (see `prisma/schema.prisma`) to create/update the local database.

### 1.6 Run the App in Development

```bash
# Run server + client concurrently
npm run dev

# Or run them separately
npm run dev:server   # Backend API + WebSockets on PORT (default 3000)
npm run dev:client   # React client via Vite on http://localhost:5173
```

Once running, you should have:

- **Backend API & WebSocket**: `http://localhost:3000` (`/api`, `/health`, Socket.IO)
- **React client**: `http://localhost:5173`

> **Important:** In local development there should be exactly **one** Node.js backend process listening on port `3000`:
>
> - Use `npm run dev:server` (or the `app` service in `docker compose up`) as the canonical backend entrypoint.
> - Avoid starting additional ad-hoc scripts that also bind `3000` (for example, older WebSocket test harnesses or `ts-node src/server/index.ts` in a second terminal), as they can cause confusing WebSocket and `/game/:gameId` behaviour.
>
> The backend’s supported v1 deployment topology is a **single app instance** per environment. This is enforced via the `RINGRIFT_APP_TOPOLOGY` environment variable, which defaults to `single` in development. See “Deployment Topology (Backend)” in `README.md` for details.

You can:

- Register/login via the client (Auth routes: `/api/auth/*`).
- Create games from the lobby (Game routes: `/api/games/*`), including games against AI opponents.
- Join games and play via the board UI and WebSocket-backed `GamePage`.
- Use `/sandbox` in the client to run a **client-local, rules-complete sandbox** (`ClientSandboxEngine`).

### Playing Against AI

The lobby UI now supports creating games with AI opponents:

1. **Navigate to the Lobby**: Visit `/lobby` after logging in
2. **Configure AI opponents**:
   - Set "Number of AI opponents" (0-3)
   - Choose AI difficulty (1-10):
     - **1-2**: Beginner (RandomAI)
     - **3-5**: Intermediate (HeuristicAI)
     - **6-8**: Advanced (MinimaxAI)
     - **9-10**: Expert (MCTSAI)
   - Optionally override AI type and control mode
3. **Create Game**: Games with AI opponents auto-start immediately
4. **Play**: The AI will make moves automatically during its turns

AI games are currently unrated. The AI thinking indicator shows when it's the AI's turn.

### Finding and Joining Games

The lobby shows all available games in real-time with powerful filtering and discovery features:

**Filtering Options:**

- **Board type**: Filter by square8, square19, or hexagonal boards
- **Rated vs unrated**: Show only rated games or unrated games
- **Player count**: Filter by 2, 3, or 4 player games
- **Search**: Find games by creator name or game ID

**Game Information:**
Each game card displays:

- Creator's name and rating
- Board type and time control settings
- Current players vs maximum capacity
- Rated status indicator
- Game status (waiting or in progress)

**Available Actions:**

- **Join**: Enter a waiting game (button disabled if game is full or you're the creator)
- **Watch**: Spectate any game to observe gameplay
- **Cancel**: Remove your own waiting game from the lobby

**Real-Time Updates:**
The lobby automatically updates when:

- New games are created and appear in the list
- Players join games (player count updates)
- Games start (removed from lobby)
- Games are cancelled (removed from lobby)

**Sorting Options:**

- Newest First (default)
- Most Players
- Board Type
- Rated First

No manual refresh needed - the lobby stays synchronized across all connected clients via WebSocket.

To run tests while you work:

```bash
npm test                  # All Jest tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
```

> For a deeper understanding of what is implemented and where the gaps are, see `CURRENT_STATE_ASSESSMENT.md` and `README.md`.

### Privacy and Data Management (GDPR Compliance)

RingRift includes privacy-focused features for GDPR compliance:

- **Account Deletion**: Users can delete their accounts via `DELETE /api/users/me`. This soft-deletes the account, anonymizes personal data (email and username), and revokes all authentication tokens.
- **Data Export**: Users can export their data via `GET /api/users/me/export`. The export includes profile information, statistics, and game history in a downloadable JSON format.
- **Data Retention**: Configurable retention policies clean up expired tokens, unverified accounts, and soft-deleted users. See [`docs/ENVIRONMENT_VARIABLES.md`](docs/ENVIRONMENT_VARIABLES.md) for configuration options.

For implementation details, see [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](docs/DATA_LIFECYCLE_AND_PRIVACY.md).

### 1.7 Configuration Reference (Core Environment Variables)

Most configuration is controlled via a root `.env` file for local development and
environment variables passed into containers via `docker-compose.yml`. The most
important variables are:

- `DATABASE_URL` – PostgreSQL connection string.
  - Local dev + Docker: defaults to the `postgres` service
    (`postgresql://ringrift:${DB_PASSWORD:-password}@postgres:5432/ringrift`).
- `REDIS_URL` – Redis connection string.
  - Local dev + Docker: defaults to the `redis` service (`redis://redis:6379`).
- `JWT_SECRET`, `JWT_REFRESH_SECRET` – secrets used to sign access and refresh tokens.
- `CORS_ORIGIN` – allowed origin for the browser client (usually `http://localhost:5173` in dev).
- `AI_SERVICE_URL` – base URL for the Python AI Service:
  - Local dev without Docker: `http://localhost:8001`
  - Docker Compose: `http://ai-service:8001` inside the `app` container (preconfigured in `docker-compose.yml`).
- `RINGRIFT_RULES_MODE` – Controls the rules engine authority:
  - `ts` (default): TypeScript engine is authoritative.
  - `shadow`: TypeScript is authoritative, but Python engine runs in parallel for parity checks.
  - `python`: Python engine is authoritative for validation (experimental).
- `RINGRIFT_APP_TOPOLOGY` – Controls the backend deployment topology:
  - `single` (default): Single app instance per environment; supported and tested mode.
  - `multi-unsafe`: Multiple app instances **without** sticky sessions or shared state; unsupported in production (the server will refuse to start when `NODE_ENV=production`).
  - `multi-sticky`: Multiple app instances with **infrastructure-enforced sticky sessions** for all game-affecting HTTP + WebSocket traffic; still risky and intended for operators who understand and accept the trade-offs.
- `PORT` – port the Node backend listens on (default `3000` in dev).
- `VITE_ERROR_REPORTING_ENABLED` – when set to `"true"`, enables client-side error reporting in the React SPA (recommended for staging/production).
- `VITE_ERROR_REPORTING_ENDPOINT` – HTTP endpoint used by the SPA to POST error events (defaults to `/api/client-errors`).
- `VITE_ERROR_REPORTING_MAX_EVENTS` – optional per-page-load cap on the number of error reports sent by the SPA (default `50`).

---

## 2. AI Service Quick Start

The AI Service is a separate **Python FastAPI microservice** in `ai-service/`. It is used by the backend through `AIEngine` and `AIServiceClient` for:

- AI move selection in backend games.
- Several PlayerChoices (e.g. line reward, ring elimination, region order), alongside local heuristics.

You have two primary ways to run it: **Python virtualenv (recommended for development)** or **Docker**.

For training pipelines and dataset generation (including the Territory/combined-margin generator used for heuristic and ML training), see the canonical reference in [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1).

### 2.1 Option A – Python Virtual Environment (Recommended for Dev)

**Prerequisites:** Python **3.11+** installed.

```bash
cd ai-service

# One-time setup
./setup.sh

# Start the service (hot reload, uses venv)
./run.sh
```

If you see a "permission denied" error, make the scripts executable:

```bash
chmod +x ai-service/setup.sh ai-service/run.sh
```

Once running, the AI Service is available at:

- **Base URL:** http://localhost:8001
- **API docs (Swagger):** http://localhost:8001/docs
- **Health:** http://localhost:8001/health

Make sure your root `.env` points `AI_SERVICE_URL` to this value:

```env
AI_SERVICE_URL=http://localhost:8001
```

> The Node backend will call this service via `AIServiceClient` (see `src/server/services/AIServiceClient.ts`), and the AI turn loop is triggered from `WebSocketServer.maybePerformAITurn`.

### 2.2 Option B – Dockerized AI Service

The root `docker-compose.yml` now defines an `ai-service` service that will be
started automatically when you run `docker compose up` (see section 3). If you
want to run the AI Service by itself (without the rest of the stack), you can
still build and run the container directly from the `ai-service` directory:

```bash
cd ai-service

# Build image
docker build -t ringrift-ai-service .

# Run container on port 8001
docker run --rm -p 8001:8001 ringrift-ai-service
```

Then set in your root `.env` if you are running the Node backend on the host:

```env
AI_SERVICE_URL=http://localhost:8001
```

> When you use the full Docker Compose stack, you generally do **not** need to
> run this container manually. The `ai-service` service is started as part of
> `docker compose up`, and the `app` service is already configured with
> `AI_SERVICE_URL=http://ai-service:8001`.

### 2.3 Verifying the AI Service

Once the service is running (via `run.sh` or Docker):

```bash
# Health check
curl http://localhost:8001/health
# → {"status":"healthy"}

# Service info
curl http://localhost:8001/
```

Or visit `http://localhost:8001/docs` in a browser and use the interactive Swagger UI to:

- Call `POST /ai/move` with a test `GameState` payload.
- Call `POST /ai/evaluate` for position evaluation.

---

## 3. Running the Full Stack with Docker Compose

The root `docker-compose.yml` currently defines the **main application stack**:

- `app` – Node.js backend (builds from `Dockerfile` and exposes port 3000; HTTP API and Socket.IO WebSockets share this port)
- `nginx` – Reverse proxy (80/443) using `nginx.conf`
- `postgres` – PostgreSQL database
- `redis` – Redis instance
- `ai-service` – Python FastAPI AI microservice (exposes port 8001; used by `AIServiceClient`)
- `prometheus` – Prometheus TSDB
- `grafana` – Grafana dashboards

> The AI Service is now included as the `ai-service` service; you do not need to
> start it manually when using `docker compose up`. The `app` container is
> configured with `AI_SERVICE_URL=http://ai-service:8001`.

### 3.1 Start the Core Stack

From the project root:

```bash
# Build and run the main stack
docker compose up
# or in the background
docker compose up -d
```

By default you’ll have:

- **App container (`app`)** listening on `http://localhost:3000`
  - Runs as a **single instance** by default (`deploy.replicas: 1`) with `RINGRIFT_APP_TOPOLOGY=single`, meaning it assumes it is the only app instance talking to this database and Redis for authoritative game sessions.
- **Nginx** proxy on `http://localhost/` (80) and `https://localhost/` (443), if configured
- **PostgreSQL** on port `5432`
- **Redis** on port `6379`
- **Prometheus** on `http://localhost:9090` (requires `--profile monitoring`)
- **Grafana** on `http://localhost:3002` (requires `--profile monitoring`)

> If you manually scale the `app` service (for example, `docker compose up --scale app=2`), you are leaving the default, supported topology. Update `RINGRIFT_APP_TOPOLOGY` and ensure infrastructure-enforced sticky sessions for WebSocket + game-affecting HTTP traffic before doing so.

### 3.2 Logs & Shutdown

```bash
# Tail logs (all services)
docker compose logs -f

# Tail a single service
docker compose logs -f app

# Stop & remove containers
docker compose down
```

> When using Docker for everything, the `app` service is already configured with
> `AI_SERVICE_URL=http://ai-service:8001`. If you choose to run the AI Service
> outside Docker instead, override this value in your `.env` (for example,
> `http://host.docker.internal:8001`).

---

## 4. Development Quality-of-Life Tips

### 4.1 VS Code Docker Extension

If you prefer a graphical interface for containers, the **Docker** extension for VS Code works well with this project:

- Install the **Docker** extension (whale icon sidebar).
- Open this repo in VS Code; the `docker-compose.yml` will appear under **COMPOSE**.
- Use **Compose Up/Down** commands to manage the stack.
- Inspect logs directly from the **CONTAINERS** tree.

### 4.2 Running Tests While Developing

Use Jest for backend, engine, WebSocket, AI boundary, and sandbox tests:

```bash
npm test                  # All tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
npm run test:unit         # Unit tests (tests/unit)
```

For more context on **what is already covered** and where we’re headed with tests, see:

- `CURRENT_STATE_ASSESSMENT.md` – coverage and feature completeness
- `STRATEGIC_ROADMAP.md` – higher-level plan
- `archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` – historical concrete steps toward a playable MVP; for current, code‑verified status and tasks, defer to `CURRENT_STATE_ASSESSMENT.md`, `KNOWN_ISSUES.md`, and `TODO.md`.

---

## 5. Forward-Looking: Recommended Next Steps for New Contributors

Once you have the app and AI service running, the most impactful next steps are:

1. **Exercise the Sandbox and Backend Games**
   - Use `/sandbox` to explore rules, chain captures, lines, and territory in a local-only environment.
   - Create a backend game from the lobby and play with an AI opponent (requires AI Service).
   2. **Start Adding Scenario Tests**
      - Mirror examples from `ringrift_complete_rules.md` into Jest tests in `tests/unit/`.
      - Focus on chain capture edge cases, complex line/Territory interactions, and hex-board quirks.
   3. **Improve HUD & Game UX**
      - Enhance `GameHUD` and `GamePage` to show phase, current player, ring counts, and Territory spaces.
      - Wire `VictoryModal` consistently for both backend and sandbox games.

2. **Harden the AI Boundary**
   - Extend tests around `AIEngine`, `AIServiceClient`, and `AIInteractionHandler` to cover failure and timeout behaviour.
   - Add minimal logging/metrics for AI calls.

For more detailed plans, see:

- `README.md` – high-level status and API surface
- `ARCHITECTURE_ASSESSMENT.md` – structural overview and refactoring axes
- `STRATEGIC_ROADMAP.md` – milestone-oriented roadmap
- `TODO.md` – phase-structured, granular task list

---

## 6. Troubleshooting

### 6.1 Common Python / AI Service Issues

**"command not found: python3"**

```bash
brew install python@3.11
```

**"zsh: permission denied: ./setup.sh"**

```bash
chmod +x ai-service/setup.sh ai-service/run.sh
```

**"pip install" errors**

```bash
cd ai-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**"Port 8001 already in use"**

```bash
lsof -i :8001
kill -9 <PID>
```

### 6.2 Docker Issues

**"command not found: docker"**

- Install Docker Desktop, OrbStack, or Colima.
- See `docker-compose.yml` and `Dockerfile` for Docker configuration.

**"Cannot connect to the Docker daemon"**

```bash
# Check if Docker is running
docker info

# Start Colima if you use it
colima start
```

**Database connection issues from Node**

- Make sure `docker compose up -d postgres redis` is running.
- Verify `DATABASE_URL` in `.env` matches the credentials in `docker-compose.yml`.

---

## 7. File Structure Overview (Quick)

```text
RingRift/
├── ai-service/                 # Python AI microservice (FastAPI)
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── models/            # Pydantic models (see models/core.py; mirrors src/shared/types/game.ts)
│   │   └── ai/                # AI implementations (random, heuristic, MCTS, descent, etc.)
│   ├── setup.sh               # Create Python venv + install deps
│   ├── run.sh                 # Start AI Service with hot reload
│   ├── Dockerfile             # AI Service Docker image
│   └── requirements.txt       # Python dependencies
├── src/server/                # Node backend (Express + Socket.IO + GameEngine)
├── src/client/                # React + Vite frontend
├── tests/                     # Jest tests (engine, WebSocket, AI, sandbox)
├── docker-compose.yml         # App + DB + Redis + observability
├── Dockerfile                 # Node app build
├── README.md                  # High-level overview & API
└── QUICKSTART.md              # This file
```

If you run into issues beyond what’s covered here, check the other docs in the project root—they are kept in sync with the current codebase and include more detailed plans and assessments.

---

## 8. Staging Deployment (Docker Compose)

This section describes how to run a **single-node staging stack** (backend, client, AI Service, PostgreSQL, Redis, and observability) using Docker Compose.

### 8.1 Prerequisites

- Docker + Docker Compose installed and running.
- A populated `.env` file with non-placeholder secrets (you can start from `.env.staging`):

  ```bash
  cp .env.staging .env
  # then edit .env to set strong JWT_* secrets and any DB/Redis passwords
  ```

At minimum, ensure **all** of the following are set to non-placeholder values in `.env`:

- `JWT_SECRET`
- `JWT_REFRESH_SECRET`
- `DB_PASSWORD`
- `REDIS_PASSWORD` (if you enable Redis auth)
- `AI_SERVICE_URL` (only needed when running the AI Service outside Docker)

The default `.env.staging` values are suitable for **local-only staging** but should be rotated for any externally exposed deployment.

### 8.2 Starting the Staging Stack

From the project root:

```bash
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up --build
```

This will start:

- `app` – Node.js backend + built React client (served on `http://localhost:3000`)
- `ai-service` – Python FastAPI AI microservice (internal port `8001`)
- `postgres` – PostgreSQL database (`5432`)
- `redis` – Redis cache (`6379`)
- `prometheus` – Prometheus TSDB (`9090`)
- `grafana` – Grafana dashboards (`3001`, mapped from Grafana’s `3000`)
- `nginx` – Optional reverse proxy (80/443) if you have `nginx.conf`/TLS configured

The staging overlay [`docker-compose.staging.yml`](docker-compose.staging.yml:1) adds:

- A startup command for the `app` service that runs:
  - `npx prisma migrate deploy` (applies Prisma migrations against `DATABASE_URL`)
  - Then starts `node dist/server/index.js`
- Health-check–based dependencies so `app` waits for:
  - `postgres` (via `pg_isready`)
  - `redis` (via `redis-cli ping`)
  - `ai-service` (via its internal `/health` endpoint and Dockerfile `HEALTHCHECK`)

### 8.3 Expected Endpoints

Once the stack is healthy, you should have:

- **Client + API + WebSocket** (served by `app`):
  - Backend & client: `http://localhost:3000`
  - Health: `http://localhost:3000/health`
  - Metrics: `http://localhost:3000/metrics`
- **AI Service**:
  - Internal base URL (from inside Docker): `http://ai-service:8001`
  - From the host (for debugging): `http://localhost:8001`
  - Health: `http://localhost:8001/health`
  - Docs: `http://localhost:8001/docs`
- **Database**:
  - PostgreSQL: `localhost:5432`
- **Redis**:
  - Redis: `localhost:6379`
- **Observability** (requires `--profile monitoring`):
  - Prometheus: `http://localhost:9090`
  - Grafana: `http://localhost:3002`

### 8.4 Verifying the Staging Stack

1. **Check container health:**

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.staging.yml ps
   ```

   All core services (`app`, `ai-service`, `postgres`, `redis`) should report `healthy` or `running`.

2. **Hit health endpoints:**

   ```bash
   curl http://localhost:3000/health
   curl http://localhost:8001/health
   ```

   Both should return simple JSON with a `"status"` field.

3. **Run a basic AI game flow:**
   - Open `http://localhost:3000` in a browser.
   - Register/login, go to the lobby, and create a game with at least one AI opponent.
   - Start the game and make a few moves, verifying that:
     - The AI takes its turns without backend errors.
     - Territory/line events and victory conditions behave normally.

4. **Shut down the stack:**

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.staging.yml down
   ```

This staging setup is intentionally single-node and **non-TLS**. For production-grade hardening (HTTPS termination, WAF, centralized logging/metrics, backups, and multi-instance topology with sticky sessions), additional infrastructure work is required beyond this local staging configuration.
