#!/bin/bash
# Setup node resilience on a RingRift cluster node
#
# Usage:
#   ./setup_node_resilience.sh <node-id> <coordinator-url>
#
# Example:
#   ./setup_node_resilience.sh vast-5090-quad http://192.222.53.22:8770

set -e

NODE_ID="${1:?Usage: $0 <node-id> <coordinator-url>}"
COORDINATOR_URL="${2:?Usage: $0 <node-id> <coordinator-url>}"
RINGRIFT_DIR="${RINGRIFT_DIR:-/root/ringrift/ai-service}"
LOG_DIR="/var/log/ringrift"
P2P_PORT="${P2P_PORT:-8770}"
SSH_PORT="${SSH_PORT:-}"

echo "Setting up node resilience for $NODE_ID"
echo "Coordinator: $COORDINATOR_URL"
echo "RingRift dir: $RINGRIFT_DIR"

# Try to infer SSH port from distributed_hosts.yaml if not explicitly provided.
if [ -z "$SSH_PORT" ]; then
    SSH_PORT="$(python3 - "$NODE_ID" "$RINGRIFT_DIR" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    sys.exit(1)

node_id = sys.argv[1]
ringrift_dir = Path(sys.argv[2])
cfg = ringrift_dir / "config" / "distributed_hosts.yaml"
if not cfg.exists():
    print("22")
    sys.exit(0)

data = yaml.safe_load(cfg.read_text()) or {}
hosts = data.get("hosts", {}) or {}
node = hosts.get(node_id, {}) or {}
port = node.get("ssh_port", 22) or 22
print(int(port))
PY
)"
fi
SSH_PORT="${SSH_PORT:-22}"

# Create directories
mkdir -p /etc/ringrift
mkdir -p "$LOG_DIR"

# Write node configuration
cat > /etc/ringrift/node.conf << EOF
NODE_ID=$NODE_ID
COORDINATOR_URL=$COORDINATOR_URL
RINGRIFT_DIR=$RINGRIFT_DIR
P2P_PORT=$P2P_PORT
SSH_PORT=$SSH_PORT
EOF

if [ -n "${RINGRIFT_CLUSTER_AUTH_TOKEN:-}" ]; then
    echo "RINGRIFT_CLUSTER_AUTH_TOKEN=$RINGRIFT_CLUSTER_AUTH_TOKEN" >> /etc/ringrift/node.conf
fi

echo "Created /etc/ringrift/node.conf"

# Install systemd services if available
if command -v systemctl &> /dev/null && [ -d /etc/systemd/system ]; then
    echo "Installing systemd services..."

    # Copy service files
    cp "$RINGRIFT_DIR/deploy/systemd/ringrift-p2p.service" /etc/systemd/system/
    cp "$RINGRIFT_DIR/deploy/systemd/ringrift-resilience.service" /etc/systemd/system/

    # Reload and enable
    systemctl daemon-reload
    systemctl enable ringrift-p2p.service
    systemctl enable ringrift-resilience.service

    echo "Systemd services installed and enabled"
else
    echo "Systemd not available, using cron fallback"
fi

# Setup cron jobs as fallback/backup
CRON_FILE="/etc/cron.d/ringrift"
cat > "$CRON_FILE" << EOF
# RingRift node resilience cron jobs
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
PYTHONPATH=$RINGRIFT_DIR
P2P_PORT=$P2P_PORT
SSH_PORT=$SSH_PORT
RINGRIFT_CLUSTER_AUTH_TOKEN=${RINGRIFT_CLUSTER_AUTH_TOKEN:-}

# Health check and reconnection every 5 minutes
*/5 * * * * root python3 $RINGRIFT_DIR/scripts/node_resilience.py --node-id $NODE_ID --coordinator $COORDINATOR_URL --ai-service-dir $RINGRIFT_DIR --p2p-port $P2P_PORT --once >> $LOG_DIR/cron.log 2>&1

# Node registration every 10 minutes
*/10 * * * * root python3 $RINGRIFT_DIR/scripts/register_node.py --node-id $NODE_ID --coordinator $COORDINATOR_URL --auto-ip --port $SSH_PORT >> $LOG_DIR/registration.log 2>&1

# Log rotation daily
0 0 * * * root find $LOG_DIR -name "*.log" -size +100M -exec truncate -s 10M {} \;
EOF

chmod 644 "$CRON_FILE"
echo "Cron jobs installed at $CRON_FILE"

# Create watchdog script
cat > /usr/local/bin/ringrift-watchdog << 'EOF'
#!/bin/bash
# Quick watchdog to restart services if they crash
source /etc/ringrift/node.conf

# Check P2P health
if ! curl -s --connect-timeout 5 "http://localhost:${P2P_PORT}/health" > /dev/null 2>&1; then
    echo "$(date): P2P orchestrator not responding, attempting restart"

    # Kill any zombie processes
    pkill -f "p2p_orchestrator.py" 2>/dev/null || true
    sleep 2

    # Start fresh
    cd "$RINGRIFT_DIR"
    PYTHONPATH="$RINGRIFT_DIR" nohup python3 scripts/p2p_orchestrator.py --node-id "$NODE_ID" --port "$P2P_PORT" --peers "$COORDINATOR_URL" --ringrift-path "$RINGRIFT_DIR/.." >> /var/log/ringrift/p2p.log 2>&1 &
    echo "$(date): P2P orchestrator restarted (PID $!)"
fi
EOF
chmod +x /usr/local/bin/ringrift-watchdog

# Add watchdog to crontab if not using systemd
if ! command -v systemctl &> /dev/null; then
    echo "*/2 * * * * root /usr/local/bin/ringrift-watchdog >> $LOG_DIR/watchdog.log 2>&1" >> "$CRON_FILE"
fi

echo ""
echo "Node resilience setup complete!"
echo ""
echo "To start services now:"
if command -v systemctl &> /dev/null; then
    echo "  systemctl start ringrift-p2p"
    echo "  systemctl start ringrift-resilience"
else
    echo "  python3 $RINGRIFT_DIR/scripts/p2p_orchestrator.py --node-id $NODE_ID --port $P2P_PORT --peers $COORDINATOR_URL &"
    echo "  python3 $RINGRIFT_DIR/scripts/node_resilience.py --node-id $NODE_ID --coordinator $COORDINATOR_URL &"
fi
echo ""
echo "To check status:"
echo "  curl http://localhost:$P2P_PORT/health"
echo "  tail -f $LOG_DIR/resilience.log"
