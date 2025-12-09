#!/bin/bash
# Start CMA-ES worker on AWS staging server
#
# This script is designed to be run on the AWS staging instance.
# It starts the cluster_worker.py in the background with proper environment.
#
# Usage:
#   ./start_aws_worker.sh           # Start worker in foreground
#   ./start_aws_worker.sh --daemon  # Start worker as background daemon
#   ./start_aws_worker.sh --stop    # Stop running worker
#   ./start_aws_worker.sh --status  # Check worker status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGRIFT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
AI_SERVICE_ROOT="$RINGRIFT_ROOT/ai-service"
VENV_PATH="$AI_SERVICE_ROOT/venv"
PID_FILE="/tmp/ringrift-worker.pid"
LOG_FILE="$AI_SERVICE_ROOT/logs/worker.log"

# Default to port 8766 on AWS (8765 is used by main app)
PORT="${RINGRIFT_WORKER_PORT:-8766}"
WORKER_ID="${RINGRIFT_WORKER_ID:-aws-staging}"

# Activate virtual environment if it exists
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    else
        echo "Warning: Virtual environment not found at $VENV_PATH"
        echo "Worker will use system Python"
    fi
}

start_foreground() {
    echo "Starting RingRift CMA-ES worker..."
    echo "  Port: $PORT"
    echo "  Worker ID: $WORKER_ID"
    echo "  Log: stdout"
    echo ""

    activate_venv

    cd "$AI_SERVICE_ROOT"
    export PYTHONPATH="$AI_SERVICE_ROOT"
    export RINGRIFT_SKIP_SHADOW_CONTRACTS=true

    exec python scripts/cluster_worker.py \
        --port "$PORT" \
        --worker-id "$WORKER_ID"
}

start_daemon() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Worker already running (PID: $OLD_PID)"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting RingRift CMA-ES worker as daemon..."
    echo "  Port: $PORT"
    echo "  Worker ID: $WORKER_ID"
    echo "  Log: $LOG_FILE"
    echo "  PID file: $PID_FILE"

    mkdir -p "$(dirname "$LOG_FILE")"

    activate_venv

    cd "$AI_SERVICE_ROOT"
    export PYTHONPATH="$AI_SERVICE_ROOT"
    export RINGRIFT_SKIP_SHADOW_CONTRACTS=true

    nohup python scripts/cluster_worker.py \
        --port "$PORT" \
        --worker-id "$WORKER_ID" \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Worker started (PID: $(cat "$PID_FILE"))"
    echo ""
    echo "To view logs: tail -f $LOG_FILE"
    echo "To stop: $0 --stop"
}

stop_worker() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Worker may not be running."
        exit 0
    fi

    PID=$(cat "$PID_FILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping worker (PID: $PID)..."
        kill "$PID"

        # Wait for process to exit
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done

        if kill -0 "$PID" 2>/dev/null; then
            echo "Worker didn't stop gracefully, forcing..."
            kill -9 "$PID" 2>/dev/null || true
        fi

        rm -f "$PID_FILE"
        echo "Worker stopped."
    else
        echo "Worker process not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
}

show_status() {
    echo "RingRift CMA-ES Worker Status"
    echo "=============================="

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Status: RUNNING (PID: $PID)"
        else
            echo "Status: STOPPED (stale PID file)"
        fi
    else
        echo "Status: STOPPED"
    fi

    echo ""
    echo "Checking health endpoint..."
    if curl -s --connect-timeout 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Health: OK"
        curl -s "http://localhost:$PORT/health" | python3 -m json.tool 2>/dev/null || \
            curl -s "http://localhost:$PORT/health"
    else
        echo "Health: NOT RESPONDING (port $PORT)"
    fi
}

# Parse command
case "${1:-}" in
    --daemon|-d)
        start_daemon
        ;;
    --stop|-s)
        stop_worker
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        echo "Usage: $0 [--daemon|--stop|--status|--help]"
        echo ""
        echo "Options:"
        echo "  (none)     Start worker in foreground"
        echo "  --daemon   Start worker as background daemon"
        echo "  --stop     Stop running worker"
        echo "  --status   Check worker status"
        echo ""
        echo "Environment variables:"
        echo "  RINGRIFT_WORKER_PORT  Worker port (default: 8765)"
        echo "  RINGRIFT_WORKER_ID    Worker ID (default: aws-staging)"
        ;;
    "")
        start_foreground
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information."
        exit 1
        ;;
esac
