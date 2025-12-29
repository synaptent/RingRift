#!/bin/bash
# RingRift P2P Keepalive Script
# Ensures P2P orchestrator is always running
#
# Usage:
#   ./scripts/p2p_keepalive.sh
#
# Add to crontab for automatic recovery:
#   */5 * * * * /path/to/ringrift/ai-service/scripts/p2p_keepalive.sh
#
# Environment variables:
#   RINGRIFT_NODE_ID - Override auto-detected node ID
#   RINGRIFT_P2P_PORT - Override default port (8770)
#   RINGRIFT_P2P_PEERS - Override default peer list

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGRIFT_PATH="${SCRIPT_DIR%/scripts}"
P2P_PORT="${RINGRIFT_P2P_PORT:-8770}"
LOGFILE="$RINGRIFT_PATH/logs/p2p_keepalive.log"

# Auto-detect node ID from hostname if not set
if [ -n "$RINGRIFT_NODE_ID" ]; then
    NODE_ID="$RINGRIFT_NODE_ID"
else
    NODE_ID="$(hostname -s)"
fi

# Default peers (coordinator nodes)
DEFAULT_PEERS="http://100.78.101.123:8770,http://100.88.176.74:8770,http://100.107.168.125:8770"
PEERS="${RINGRIFT_P2P_PEERS:-$DEFAULT_PEERS}"

# Ensure log directory exists
mkdir -p "$RINGRIFT_PATH/logs"

# Check if P2P is running and healthy
check_health() {
    curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1
}

if check_health; then
    # Already running and healthy
    exit 0
fi

# Log restart attempt
echo "[$(date)] P2P not running or unhealthy, starting..." >> "$LOGFILE"

# Kill any zombie process
pkill -9 -f "p2p_orchestrator.py" 2>/dev/null || true
sleep 1

# Find Python interpreter
if [ -f "$RINGRIFT_PATH/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/venv/bin/python"
elif [ -f "$RINGRIFT_PATH/.venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/.venv/bin/python"
else
    PYTHON="$(which python3)"
fi

# Verify Python exists
if [ ! -x "$PYTHON" ]; then
    echo "[$(date)] ERROR: Python not found" >> "$LOGFILE"
    exit 1
fi

# Start P2P in background
cd "$RINGRIFT_PATH"
export PYTHONPATH="$RINGRIFT_PATH"

nohup "$PYTHON" scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$P2P_PORT" \
    --peers "$PEERS" \
    --ringrift-path "${RINGRIFT_PATH%/ai-service}" \
    >> "$RINGRIFT_PATH/logs/p2p.log" 2>&1 &

P2P_PID=$!
echo "[$(date)] P2P started with PID $P2P_PID" >> "$LOGFILE"

# Wait briefly and verify it started
sleep 3
if check_health; then
    echo "[$(date)] P2P health check passed" >> "$LOGFILE"
else
    echo "[$(date)] WARNING: P2P started but health check failed" >> "$LOGFILE"
fi
