#!/bin/bash
# RingRift P2P Keepalive Script
# Ensures P2P orchestrator is always running
#
# Usage: Add to crontab:
#   * * * * * /path/to/ringrift/ai-service/scripts/ops/p2p_keepalive.sh
#
# Environment variables:
#   RINGRIFT_NODE_ID - Node identifier (default: hostname)
#   RINGRIFT_P2P_PORT - P2P port (default: 8770)
#   RINGRIFT_AI_SERVICE - Path to ai-service directory (default: script location)

set -e

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGRIFT_AI_SERVICE="${RINGRIFT_AI_SERVICE:-$(dirname $(dirname "$SCRIPT_DIR"))}"
RINGRIFT_PATH="${RINGRIFT_AI_SERVICE%/ai-service}"

# Configuration with defaults
NODE_ID="${RINGRIFT_NODE_ID:-$(hostname -s)}"
P2P_PORT="${RINGRIFT_P2P_PORT:-8770}"
LOGFILE="$RINGRIFT_AI_SERVICE/logs/p2p_keepalive.log"

# Default peer seeds (voter nodes)
P2P_SEEDS="${RINGRIFT_P2P_SEEDS:-http://89.169.112.47:8770,http://135.181.39.239:8770,http://135.181.39.201:8770,http://208.167.249.164:8770}"

# Ensure log directory exists
mkdir -p "$RINGRIFT_AI_SERVICE/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOGFILE"
}

# Check if P2P is running and healthy
if curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1; then
    # Already running and healthy
    exit 0
fi

log "P2P not running or unhealthy, starting..."

# Kill any zombie process
pkill -9 -f "p2p_orchestrator.py" 2>/dev/null || true
sleep 1

# Find Python executable
if [ -f "$RINGRIFT_AI_SERVICE/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_AI_SERVICE/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="/usr/bin/python3"
fi

# Start P2P in background
cd "$RINGRIFT_AI_SERVICE"
export PYTHONPATH="$RINGRIFT_AI_SERVICE"

setsid $PYTHON scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$P2P_PORT" \
    --peers "$P2P_SEEDS" \
    --ringrift-path "$RINGRIFT_PATH" \
    >> "$RINGRIFT_AI_SERVICE/logs/p2p.log" 2>&1 &

PID=$!
log "P2P started with PID $PID"

# Wait briefly and verify it started
sleep 3
if kill -0 $PID 2>/dev/null; then
    log "P2P process running"
else
    log "ERROR: P2P process died immediately"
    exit 1
fi
