#!/bin/bash
# Setup cron-based watchdog for P2P orchestrator (for containers without systemd)
# Usage: setup-cron-p2p.sh <node_id> [ringrift_path] [coordinator_url]

set -e

NODE_ID="${1:-$(hostname)}"
RINGRIFT_PATH="${2:-$HOME/ringrift}"
COORDINATOR_URL="${3:-http://localhost:8770}"  # Set to your coordinator URL
P2P_PORT="${4:-8770}"

echo "[setup] Configuring P2P orchestrator cron watchdog for node: $NODE_ID"

# Create config directory
mkdir -p /etc/ringrift /var/log/ringrift

# Write node configuration
cat > /etc/ringrift/node.conf << EOF
NODE_ID=$NODE_ID
P2P_PORT=$P2P_PORT
COORDINATOR_URL=$COORDINATOR_URL
RINGRIFT_PATH=$RINGRIFT_PATH
EOF

echo "[setup] Created /etc/ringrift/node.conf"

# Create watchdog script
cat > /etc/ringrift/p2p-watchdog.sh << 'WATCHDOG'
#!/bin/bash
# P2P Orchestrator Watchdog - restarts if not running or not responding

source /etc/ringrift/node.conf

LOG="/var/log/ringrift/p2p.log"
PID_FILE="/var/run/ringrift-p2p.pid"

# Check if process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "$(date): Process $PID not running, restarting" >> "$LOG"
        rm -f "$PID_FILE"
    else
        # Process running, check health
        if curl -sf --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1; then
            # Healthy, nothing to do
            exit 0
        else
            echo "$(date): Health check failed, killing and restarting" >> "$LOG"
            kill -9 "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            sleep 2
        fi
    fi
fi

# Start orchestrator
cd "$RINGRIFT_PATH/ai-service"
PY="$RINGRIFT_PATH/ai-service/venv/bin/python"
if [ ! -x "$PY" ]; then PY="/usr/bin/python3"; fi

echo "$(date): Starting P2P orchestrator (node=$NODE_ID)" >> "$LOG"
nohup "$PY" scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$P2P_PORT" \
    --peers "$COORDINATOR_URL" \
    --ringrift-path "$RINGRIFT_PATH" >> "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "$(date): Started with PID $!" >> "$LOG"
WATCHDOG

chmod +x /etc/ringrift/p2p-watchdog.sh

# Add to crontab (every minute)
CRON_LINE="* * * * * /etc/ringrift/p2p-watchdog.sh"
(crontab -l 2>/dev/null | grep -v "p2p-watchdog" ; echo "$CRON_LINE") | crontab -

echo "[setup] Cron watchdog installed (runs every minute)"

# Run watchdog immediately to start the service
/etc/ringrift/p2p-watchdog.sh

sleep 5
echo "[setup] Checking health..."
curl -s --connect-timeout 5 "localhost:$P2P_PORT/health" | head -c 100 || echo "(health check pending)"
