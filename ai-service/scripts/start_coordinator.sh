#!/bin/bash
# Start P2P orchestrator in lightweight coordinator-only mode
#
# This mode disables all CPU/GPU intensive operations:
# - No selfplay
# - No training
# - No gauntlet/evaluation
# - No data export
#
# It only runs:
# - P2P mesh coordination (leader election, job dispatch)
# - Sync daemon orchestration
# - Health monitoring
# - Cluster status API
#
# Usage:
#   ./scripts/start_coordinator.sh              # Foreground
#   ./scripts/start_coordinator.sh --background # Background with screen

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$AI_SERVICE_DIR"

# Coordinator mode - disables heavy compute
export RINGRIFT_IS_COORDINATOR=true
export PYTHONPATH="$AI_SERVICE_DIR"

# Node configuration
NODE_ID="${RINGRIFT_NODE_ID:-local-mac}"
PORT="${RINGRIFT_P2P_PORT:-8770}"

# Known peers (backbone nodes for initial discovery)
PEERS="${RINGRIFT_P2P_PEERS:-nebius-backbone-1:8770,hetzner-cpu1:8770,hetzner-cpu2:8770}"

echo "=== RingRift P2P Coordinator (Lightweight Mode) ==="
echo "Node ID: $NODE_ID"
echo "Port: $PORT"
echo "Peers: $PEERS"
echo ""
echo "Coordinator mode enabled:"
echo "  - Selfplay: DISABLED"
echo "  - Training: DISABLED"
echo "  - Gauntlet: DISABLED"
echo "  - Export: DISABLED"
echo ""

# Check if already running
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "ERROR: Port $PORT is already in use"
    echo "Check with: lsof -i :$PORT"
    exit 1
fi

# Background mode with screen
if [[ "$1" == "--background" || "$1" == "-b" ]]; then
    echo "Starting in background (screen session: p2p-coordinator)..."
    screen -dmS p2p-coordinator bash -c "
        cd $AI_SERVICE_DIR
        export RINGRIFT_IS_COORDINATOR=true
        export PYTHONPATH=$AI_SERVICE_DIR
        python3 scripts/p2p_orchestrator.py \
            --node-id $NODE_ID \
            --port $PORT \
            --peers $PEERS \
            2>&1 | tee logs/p2p_coordinator.log
    "
    sleep 2

    # Check if started successfully
    if curl -s --connect-timeout 3 http://localhost:$PORT/status >/dev/null 2>&1; then
        echo "Coordinator started successfully!"
        echo ""
        echo "Commands:"
        echo "  Status:  curl -s http://localhost:$PORT/status | python3 -m json.tool"
        echo "  Attach:  screen -r p2p-coordinator"
        echo "  Stop:    screen -S p2p-coordinator -X quit"
    else
        echo "Warning: Coordinator may still be starting up..."
        echo "Check with: screen -r p2p-coordinator"
    fi
    exit 0
fi

# Foreground mode
echo "Starting coordinator (Ctrl+C to stop)..."
echo ""

python3 scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$PORT" \
    --peers "$PEERS"
