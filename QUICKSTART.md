# RingRift Quick Start Guide

This guide focuses on getting a **local development environment** running quickly (backend + frontend) and then wiring up the **Python AI service** used for AI turns and some PlayerChoices.

If you only want the AI microservice, skip to **[AI Service Quick Start](#ai-service-quick-start)**.

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
- `AI_SERVICE_URL` – URL for the Python AI service (see below). For local dev without Docker: `http://localhost:8001`.

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

You can:

- Register/login via the client (Auth routes: `/api/auth/*`).
- Create games from the lobby (Game routes: `/api/games/*`).
- Join games and play via the board UI and WebSocket-backed `GamePage`.
- Use `/sandbox` in the client to run a **client-local, rules-complete sandbox** (`ClientSandboxEngine`).

To run tests while you work:

```bash
npm test                  # All Jest tests
npm run test:watch        # Watch mode
npm run test:coverage     # Coverage report
```

> For a deeper understanding of what is implemented and where the gaps are, see `CURRENT_STATE_ASSESSMENT.md` and `README.md`.

---

## 2. AI Service Quick Start

The AI service is a separate **Python FastAPI microservice** in `ai-service/`. It is used by the backend through `AIEngine` and `AIServiceClient` for:

- AI move selection in backend games.
- Several PlayerChoices (e.g. line reward, ring elimination, region order), alongside local heuristics.

You have two primary ways to run it: **Python virtualenv (recommended for development)** or **Docker**.

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

Once running, the AI service is available at:

- **Base URL:** http://localhost:8001
- **API docs (Swagger):** http://localhost:8001/docs
- **Health:** http://localhost:8001/health

Make sure your root `.env` points `AI_SERVICE_URL` to this value:

```env
AI_SERVICE_URL=http://localhost:8001
```

> The Node backend will call this service via `AIServiceClient` (see `src/server/services/AIServiceClient.ts`), and the AI turn loop is triggered from `WebSocketServer.maybePerformAITurn`.

### 2.2 Option B – Dockerized AI Service

There is **no dedicated `ai-service` entry in the root `docker-compose.yml`** at the moment. To run the AI service via Docker, build and run the container directly from the `ai-service` directory:

```bash
cd ai-service

# Build image
docker build -t ringrift-ai-service .

# Run container on port 8001
docker run --rm -p 8001:8001 ringrift-ai-service
```

Then set in your root `.env`:

```env
AI_SERVICE_URL=http://localhost:8001
```

You can later add an `ai-service` block to `docker-compose.yml` if you want the AI service managed alongside the main stack; the current configuration focuses on the Node app, database, Redis, and observability stack.

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
- `prometheus` – Prometheus TSDB
- `grafana` – Grafana dashboards

> The AI service is **not** yet included here and should be run separately (see above) if you want AI opponents in Docker-based environments.

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
- **Nginx** proxy on `http://localhost/` (80) and `https://localhost/` (443), if configured
- **PostgreSQL** on port `5432`
- **Redis** on port `6379`
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3001`

### 3.2 Logs & Shutdown

```bash
# Tail logs (all services)
docker compose logs -f

# Tail a single service
docker compose logs -f app

# Stop & remove containers
docker compose down
```

> When using Docker for everything, remember to point `AI_SERVICE_URL` inside the `app` container to the correct AI service address (e.g., `http://host.docker.internal:8001` or a future `ai-service` container name).

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
- `PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` – concrete steps toward a playable MVP

---

## 5. Forward-Looking: Recommended Next Steps for New Contributors

Once you have the app and AI service running, the most impactful next steps are:

1. **Exercise the Sandbox and Backend Games**
   - Use `/sandbox` to explore rules, chain captures, lines, and territory in a local-only environment.
   - Create a backend game from the lobby and play with an AI opponent (requires AI service).

2. **Start Adding Scenario Tests**
   - Mirror examples from `ringrift_complete_rules.md` into Jest tests in `tests/unit/`.
   - Focus on chain capture edge cases, complex line/territory interactions, and hex-board quirks.

3. **Improve HUD & Game UX**
   - Enhance `GameHUD` and `GamePage` to show phase, current player, ring counts, and territory spaces.
   - Wire `VictoryModal` consistently for both backend and sandbox games.

4. **Harden the AI Boundary**
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
- See `deprecated/DOCKER_SETUP.md` for historical Docker notes if needed.

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
│   │   ├── models.py          # Pydantic models
│   │   └── ai/                # AI implementations (random, heuristic, etc.)
│   ├── setup.sh               # Create Python venv + install deps
│   ├── run.sh                 # Start AI service with hot reload
│   ├── Dockerfile             # AI service Docker image
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
