#!/bin/bash
# Setup systemd service for P2P orchestrator auto-restart
# Usage: setup-systemd-p2p.sh <node_id> [ringrift_path] [coordinator_url]

set -e

NODE_ID="${1:-$(hostname)}"
RINGRIFT_PATH="${2:-/home/ubuntu/ringrift}"
COORDINATOR_URL="${3:-http://localhost:8770}"  # Set to your coordinator URL
P2P_PORT="${4:-8770}"

echo "[setup] Configuring P2P orchestrator systemd service for node: $NODE_ID"

# Create config directory
sudo mkdir -p /etc/ringrift /var/log/ringrift

# Write node configuration
sudo tee /etc/ringrift/node.conf > /dev/null << EOF
NODE_ID=$NODE_ID
P2P_PORT=$P2P_PORT
COORDINATOR_URL=$COORDINATOR_URL
RINGRIFT_PATH=$RINGRIFT_PATH
EOF

echo "[setup] Created /etc/ringrift/node.conf"

# Copy service file
sudo cp "$RINGRIFT_PATH/ai-service/deploy/systemd/ringrift-p2p-universal.service" /etc/systemd/system/ringrift-p2p.service

# Reload systemd
sudo systemctl daemon-reload

# Stop any existing orchestrator processes
sudo pkill -9 -f p2p_orchestrator.py 2>/dev/null || true
sleep 2

# Enable and start service
sudo systemctl enable ringrift-p2p
sudo systemctl start ringrift-p2p

echo "[setup] Service enabled and started"

# Check status
sleep 3
if sudo systemctl is-active --quiet ringrift-p2p; then
    echo "[setup] SUCCESS: ringrift-p2p service is running"
    curl -s --connect-timeout 5 localhost:$P2P_PORT/health | head -c 100 || echo "(health check pending)"
else
    echo "[setup] WARNING: Service may have failed to start"
    sudo journalctl -u ringrift-p2p -n 20 --no-pager
fi
