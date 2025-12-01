#!/usr/bin/env bash
set -euo pipefail

# dev-db.sh
#
# Helper script for RingRift development:
# - Starts a Postgres container via docker-compose
# - Waits for the database to become ready
# - Runs Prisma migrations against the local dev database
#
# Usage (from repo root):
#   bash scripts/dev-db.sh
#
# Or via npm (after wiring package.json):
#   npm run dev:db
#
# This assumes:
# - docker or docker-compose is installed and running
# - .env (or environment) provides DATABASE_URL matching the defaults:
#     postgresql://ringrift:password@localhost:5432/ringrift

echo "[dev-db] Starting local Postgres via docker-compose..."

# Prefer docker-compose if available, otherwise fall back to `docker compose`.
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  echo "[dev-db] ERROR: Neither docker-compose nor docker is available on PATH." >&2
  echo "[dev-db] Install Docker Desktop or docker-compose and try again." >&2
  exit 1
fi

"${COMPOSE_CMD}" up -d postgres

echo "[dev-db] Waiting for Postgres to become ready (service: postgres)..."

ready=0
for i in {1..30}; do
  if "${COMPOSE_CMD}" exec -T postgres pg_isready -U ringrift -d ringrift >/dev/null 2>&1; then
    ready=1
    echo "[dev-db] Postgres is ready."
    break
  fi
  echo "[dev-db] Postgres not ready yet, retrying ($i/30)..."
  sleep 2
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[dev-db] ERROR: Postgres did not become ready in time." >&2
  echo "[dev-db] You can inspect logs with:" >&2
  echo "  ${COMPOSE_CMD} logs postgres" >&2
  exit 1
fi

echo "[dev-db] Running Prisma migrations (npx prisma migrate dev)..."
npx prisma migrate dev

echo "[dev-db] Local development database is ready."
echo "[dev-db] You can now start the backend with: npm run dev:server"

