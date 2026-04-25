#!/usr/bin/env bash
# Launch the fastest reviewer-facing RingRift play path.
#
# Default mode starts only the Vite client and opens the anonymous sandbox
# Human-vs-AI preset. That path does not require Docker, Postgres, Redis, or the
# Python AI service; the sandbox falls back to browser-local AI if backend AI is
# unavailable. Use --full-stack when you also want the local backend and DB.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_BASE="${RINGRIFT_PLAY_CLIENT_BASE:-http://localhost:5173}"
CLIENT_URL="${RINGRIFT_PLAY_URL:-${CLIENT_BASE}/sandbox?preset=hex8-1h-1ai}"
SERVER_HEALTH_URL="${RINGRIFT_PLAY_SERVER_HEALTH_URL:-http://localhost:3000/health}"
OPEN_BROWSER="${RINGRIFT_PLAY_OPEN_BROWSER:-1}"
FULL_STACK="${RINGRIFT_PLAY_FULL_STACK:-0}"
SERVER_PID=""
CLIENT_PID=""

usage() {
  cat <<EOF
Usage: npm run play [-- --full-stack] [-- --no-open]

Launches RingRift directly into a no-account Human-vs-AI sandbox game.

Options:
  --full-stack  Start Postgres, Redis, backend, and Vite client.
  --no-open     Print the demo URL without opening a browser.
  -h, --help    Show this help.

Environment:
  RINGRIFT_PLAY_URL              Demo URL (default: ${CLIENT_URL})
  RINGRIFT_PLAY_OPEN_BROWSER=0   Disable browser launch.
  RINGRIFT_PLAY_FULL_STACK=1     Same as --full-stack.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-stack)
      FULL_STACK=1
      shift
      ;;
    --no-open)
      OPEN_BROWSER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$CLIENT_PID" ]]; then
    kill "$CLIENT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"

  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[play] ${label} is ready: ${url}"
      return 0
    fi
    sleep 1
  done

  echo "[play] ERROR: ${label} did not become ready at ${url}" >&2
  return 1
}

open_url() {
  local url="$1"
  if [[ "$OPEN_BROWSER" == "0" || "$OPEN_BROWSER" == "false" ]]; then
    return 0
  fi

  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  else
    python3 -m webbrowser "$url" >/dev/null 2>&1 || true
  fi
}

ensure_node_modules() {
  if [[ ! -d "$ROOT_DIR/node_modules" ]]; then
    echo "[play] ERROR: node_modules is missing. Run npm install first." >&2
    exit 1
  fi
}

ensure_local_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    return 0
  fi

  echo "[play] Creating minimal local .env"
  cat >"$ROOT_DIR/.env" <<'EOF'
NODE_ENV=development
PORT=3000
DATABASE_URL=postgresql://ringrift:password@localhost:5432/ringrift
REDIS_URL=redis://localhost:6379
JWT_SECRET=dev-jwt-secret-for-local-demo-only-32chars
JWT_REFRESH_SECRET=dev-refresh-secret-for-local-demo-32chars
AI_SERVICE_URL=http://localhost:8001
ENABLE_SANDBOX_AI_ENDPOINTS=true
RINGRIFT_RULES_MODE=ts
RINGRIFT_APP_TOPOLOGY=single
EOF
}

detect_compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
  else
    echo "[play] ERROR: --full-stack requires Docker Compose." >&2
    exit 1
  fi
}

start_full_stack_dependencies() {
  detect_compose
  echo "[play] Starting Postgres and Redis"
  "${COMPOSE_CMD[@]}" up -d postgres redis

  for i in {1..45}; do
    if "${COMPOSE_CMD[@]}" exec -T postgres pg_isready -U ringrift -d ringrift >/dev/null 2>&1; then
      echo "[play] Postgres is ready"
      break
    fi
    if [[ "$i" -eq 45 ]]; then
      echo "[play] ERROR: Postgres did not become ready." >&2
      exit 1
    fi
    sleep 2
  done

  for i in {1..30}; do
    if "${COMPOSE_CMD[@]}" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
      echo "[play] Redis is ready"
      break
    fi
    if [[ "$i" -eq 30 ]]; then
      echo "[play] ERROR: Redis did not become ready." >&2
      exit 1
    fi
    sleep 1
  done

  echo "[play] Applying Prisma schema"
  DATABASE_URL="${DATABASE_URL:-postgresql://ringrift:password@localhost:5432/ringrift}" \
    npx prisma generate
  DATABASE_URL="${DATABASE_URL:-postgresql://ringrift:password@localhost:5432/ringrift}" \
    npx prisma migrate deploy
}

monitor_processes() {
  while true; do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      set +e
      wait "$SERVER_PID"
      local status=$?
      set -e
      echo "[play] Backend exited with status ${status}" >&2
      exit "$status"
    fi

    if [[ -n "$CLIENT_PID" ]] && ! kill -0 "$CLIENT_PID" >/dev/null 2>&1; then
      set +e
      wait "$CLIENT_PID"
      local status=$?
      set -e
      echo "[play] Vite client exited with status ${status}" >&2
      exit "$status"
    fi

    sleep 2
  done
}

cd "$ROOT_DIR"
ensure_node_modules
ensure_local_env

if [[ "$FULL_STACK" == "1" || "$FULL_STACK" == "true" ]]; then
  start_full_stack_dependencies
  echo "[play] Starting backend"
  npm run dev:server &
  SERVER_PID=$!
  wait_for_url "$SERVER_HEALTH_URL" "Backend" 90
else
  echo "[play] Starting browser-only sandbox demo"
  echo "[play] Use npm run play:full if you also want Postgres, Redis, and backend."
fi

echo "[play] Starting Vite client"
npm run dev:client -- --host 0.0.0.0 &
CLIENT_PID=$!
wait_for_url "$CLIENT_BASE" "Vite client" 90

cat <<EOF

[play] RingRift demo is ready:
  ${CLIENT_URL}

[play] This opens a no-account Human-vs-AI sandbox preset.
[play] Press Ctrl-C here to stop the local dev processes.
EOF

open_url "$CLIENT_URL"
monitor_processes
