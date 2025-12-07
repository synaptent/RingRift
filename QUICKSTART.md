# RingRift Quick Start Guide

**Last Updated:** 2025-12-06  
**Doc Status:** Active (developer quick start aligned with current scripts and docker-compose)

This guide gets the TypeScript backend + React client and the Python AI service running locally. For rules semantics, use `RULES_CANONICAL_SPEC.md` and the shared engine under `src/shared/engine/**`.

## Fast path (backend + frontend)

1. Install Node.js >= 18 and npm >= 9 (see `package.json`), Python 3.11+ for the AI service, and Docker + Docker Compose for Postgres/Redis.
2. Install dependencies: `npm install`.
3. Copy env and set secrets: `cp .env.example .env`; set `DATABASE_URL`, `REDIS_URL`, JWT secrets, and `AI_SERVICE_URL` (use `http://localhost:8001` for a locally running AI service).
4. Start Postgres + Redis: `docker compose up -d postgres redis` (add `ai-service` here if you want the containerised AI service).
5. Apply database schema: `npm run db:migrate && npm run db:generate`.
6. Run the app with hot reload: `npm run dev` (or `npm run dev:server` and `npm run dev:client` in separate terminals).
7. Verify: backend health at `http://localhost:3000/health`, client at `http://localhost:5173`.

## AI Service Quick Start (Python FastAPI)

- From repo root: `cd ai-service`.
- One-time setup: `./setup.sh` (creates `venv` and installs `requirements.txt`).
- Run the service: `./run.sh` (starts uvicorn on `http://localhost:8001`).
- Health/doc routes: `GET /health`, `GET /` for service info, and interactive docs at `http://localhost:8001/docs`.
- Docker Compose alternative: `docker compose up -d ai-service` (backend defaults to `AI_SERVICE_URL=http://ai-service:8001` in the compose network).

## Full docker-compose stack (optional)

- To run backend, AI service, Postgres, and Redis together: `docker compose up -d app ai-service postgres redis`.
- Monitoring (Prometheus/Alertmanager/Grafana) is enabled by default in `docker-compose.yml`; remove/comment those services if you want a lighter local stack.
- Prefer `npm run dev` for hot reload during development; the compose stack uses the image built from `Dockerfile`.

## Useful scripts (TypeScript)

- `npm run dev:app` — free port 3000 then start backend + client.
- `npm run dev:free-port` / `npm run dev:free-all` — clear common dev ports (3000/3001/5173).
- `npm run dev:sandbox:diagnostics` — quick sandbox sanity helpers.
- `npm run ssot-check` — checks key SSOT contracts/types.

## Testing

- Core Jest suite: `npm test` (see `tests/README.md` for layer breakdown).
- Focused TS rules/parity: `npm run test:ts-parity` or `npm run test:orchestrator-parity`.
- Python AI service: `cd ai-service && pytest`.
- Replay parity (Python harness): `cd ai-service && python -m scripts.check_ts_python_replay_parity --db <path>`.

## Troubleshooting

- Backend health: `GET /health` and `GET /ready` on port 3000.
- AI service health: `GET /health` on port 8001.
- If ports are busy, run `npm run dev:free-port` before `npm run dev`.
