#!/bin/bash
# RingRift Vast.ai Instance Onstart Script
# Add this to the "On-start Script" field when creating a new Vast.ai instance
#
# This script:
# 1. Installs Tailscale for mesh networking
# 2. Clones/updates the RingRift repository
# 3. Starts the universal keepalive daemon
# 4. Starts the P2P orchestrator
#
# Prerequisites:
# - Set TAILSCALE_AUTH_KEY environment variable in Vast instance config
# - Set RINGRIFT_SLACK_WEBHOOK for notifications (optional)

set -e

LOG_FILE="/tmp/ringrift_onstart.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] Starting RingRift onstart script..."

# Get node ID from hostname or instance ID
NODE_ID="${VAST_CONTAINERLABEL:-$(hostname)}"
echo "[$(date)] Node ID: $NODE_ID"

# ============================================
# 1. Install and configure Tailscale
# ============================================
install_tailscale() {
    echo "[$(date)] Installing Tailscale..."

    if command -v tailscale &> /dev/null; then
        echo "[$(date)] Tailscale already installed"
    else
        curl -fsSL https://tailscale.com/install.sh | sh
    fi

    # Start tailscaled
    if ! pgrep -x tailscaled > /dev/null; then
        echo "[$(date)] Starting tailscaled..."
        tailscaled --state=/var/lib/tailscale/tailscaled.state &
        sleep 3
    fi

    # Authenticate with Tailscale
    if [ -n "$TAILSCALE_AUTH_KEY" ]; then
        echo "[$(date)] Authenticating Tailscale..."
        tailscale up --authkey="$TAILSCALE_AUTH_KEY" --hostname="$NODE_ID" --accept-routes || true
    else
        echo "[$(date)] WARNING: TAILSCALE_AUTH_KEY not set, Tailscale not authenticated"
    fi

    # Wait for Tailscale to connect
    for i in {1..30}; do
        if tailscale status --json 2>/dev/null | grep -q '"BackendState":"Running"'; then
            echo "[$(date)] Tailscale connected"
            tailscale ip -4
            return 0
        fi
        sleep 2
    done

    echo "[$(date)] WARNING: Tailscale connection timeout"
}

# ============================================
# 2. Clone/update RingRift repository
# ============================================
setup_ringrift() {
    echo "[$(date)] Setting up RingRift..."

    RINGRIFT_DIR="$HOME/ringrift"

    if [ -d "$RINGRIFT_DIR/.git" ]; then
        echo "[$(date)] Updating existing repo..."
        cd "$RINGRIFT_DIR"
        git fetch origin
        git reset --hard origin/main
    else
        echo "[$(date)] Cloning repository..."
        rm -rf "$RINGRIFT_DIR"
        git clone --depth 1 https://github.com/an0mium/RingRift.git "$RINGRIFT_DIR"
    fi

    cd "$RINGRIFT_DIR/ai-service"

    # Install Python dependencies if requirements.txt exists
    if [ -f requirements.txt ]; then
        echo "[$(date)] Installing Python dependencies..."
        pip install -q -r requirements.txt 2>/dev/null || true
    fi

    # Create log directories
    mkdir -p logs data/selfplay

    echo "[$(date)] RingRift setup complete"
}

# ============================================
# 3. Start keepalive daemon
# ============================================
start_keepalive() {
    echo "[$(date)] Starting keepalive daemon..."

    cd "$HOME/ringrift/ai-service"

    # Kill any existing keepalive
    pkill -f universal_keepalive || true
    sleep 1

    # Start keepalive in background
    nohup python3 scripts/universal_keepalive.py \
        --node-id "$NODE_ID" \
        --daemon \
        >> /tmp/ringrift_keepalive.log 2>&1 &

    echo "[$(date)] Keepalive daemon started (PID: $!)"

    # Add to bashrc for persistence
    if ! grep -q "universal_keepalive" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << 'KEEPALIVE_EOF'
# RingRift Keepalive auto-start
if ! pgrep -f universal_keepalive > /dev/null; then
  cd ~/ringrift/ai-service 2>/dev/null && nohup python3 scripts/universal_keepalive.py --node-id "$(hostname)" --daemon >> /tmp/ringrift_keepalive.log 2>&1 &
fi
KEEPALIVE_EOF
    fi
}

# ============================================
# 4. Start P2P orchestrator
# ============================================
install_p2p_systemd() {
    echo "[$(date)] Installing P2P systemd service..."

    # Check if systemd is available
    if ! command -v systemctl &> /dev/null; then
        echo "[$(date)] systemd not available, skipping systemd install"
        return 1
    fi

    # Create systemd service file
    cat > /tmp/ringrift-p2p.service << EOF
[Unit]
Description=RingRift P2P Orchestrator
After=network.target tailscaled.service
Wants=tailscaled.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ringrift/ai-service
ExecStart=/usr/bin/python3 scripts/p2p_orchestrator.py --node-id $NODE_ID --port 8770 --peers 100.94.174.19:8770,100.67.131.72:8770,100.126.21.102:8770
Restart=always
RestartSec=10
StandardOutput=append:$HOME/ringrift/ai-service/logs/p2p_orchestrator.log
StandardError=append:$HOME/ringrift/ai-service/logs/p2p_orchestrator.log
Environment=PYTHONPATH=$HOME/ringrift/ai-service

[Install]
WantedBy=multi-user.target
EOF

    # Install service (requires sudo)
    if sudo cp /tmp/ringrift-p2p.service /etc/systemd/system/ringrift-p2p.service 2>/dev/null; then
        sudo systemctl daemon-reload
        sudo systemctl enable ringrift-p2p
        echo "[$(date)] P2P systemd service installed and enabled"
        return 0
    else
        echo "[$(date)] Failed to install systemd service (no sudo access)"
        return 1
    fi
}

start_p2p() {
    echo "[$(date)] Starting P2P orchestrator..."

    cd "$HOME/ringrift/ai-service"

    # Wait for Tailscale IP
    TAILSCALE_IP=""
    for i in {1..30}; do
        TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || true)
        if [ -n "$TAILSCALE_IP" ]; then
            break
        fi
        sleep 2
    done

    if [ -z "$TAILSCALE_IP" ]; then
        echo "[$(date)] WARNING: No Tailscale IP, P2P may not work properly"
    fi

    # Kill any existing P2P
    pkill -f p2p_orchestrator || true
    sleep 2

    # Seed peers: Hetzner CPU nodes (always online, stable P2P voters)
    SEED_PEERS="100.94.174.19:8770,100.67.131.72:8770,100.126.21.102:8770"

    # Try systemd first for persistence, fall back to nohup
    if install_p2p_systemd; then
        echo "[$(date)] Starting P2P via systemd..."
        sudo systemctl start ringrift-p2p
        echo "[$(date)] P2P orchestrator started via systemd (persistent)"
    else
        # Fallback to nohup with seed peers for cluster discovery
        echo "[$(date)] Starting P2P via nohup (non-persistent)..."
        nohup python3 scripts/p2p_orchestrator.py \
            --node-id "$NODE_ID" \
            --port 8770 \
            --peers "$SEED_PEERS" \
            >> logs/p2p_orchestrator.log 2>&1 &
        echo "[$(date)] P2P orchestrator started (PID: $!) with seed peers"
    fi
}

# ============================================
# 5. Bootstrap models from cluster
# ============================================
bootstrap_models() {
    echo "[$(date)] Bootstrapping canonical models from cluster..."
    cd "$HOME/ringrift/ai-service"
    mkdir -p models

    # Try to sync models from Hetzner nodes
    for HOST in 100.94.174.19 100.67.131.72 100.126.21.102; do
        if rsync -avz --timeout=60 -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
            root@$HOST:~/ringrift/ai-service/models/canonical_*.pth models/ 2>/dev/null; then
            echo "[$(date)] Models synced from $HOST"
            cd models
            for f in canonical_*.pth; do
                [ -f "$f" ] && ln -sf "$f" "${f/canonical_/ringrift_best_}" 2>/dev/null
            done
            return 0
        fi
    done
    echo "[$(date)] Model sync failed (non-fatal)"
}

# ============================================
# 6. Health check
# ============================================
health_check() {
    echo "[$(date)] Running health check..."

    sleep 10  # Wait for services to start

    # Check keepalive
    if pgrep -f universal_keepalive > /dev/null; then
        echo "[$(date)] Keepalive: RUNNING"
    else
        echo "[$(date)] Keepalive: NOT RUNNING"
    fi

    # Check P2P
    if pgrep -f p2p_orchestrator > /dev/null; then
        echo "[$(date)] P2P Orchestrator: RUNNING"
    else
        echo "[$(date)] P2P Orchestrator: NOT RUNNING"
    fi

    # Check Tailscale
    if tailscale status --json 2>/dev/null | grep -q '"BackendState":"Running"'; then
        echo "[$(date)] Tailscale: CONNECTED ($(tailscale ip -4))"
    else
        echo "[$(date)] Tailscale: NOT CONNECTED"
    fi

    # Check P2P health endpoint
    if curl -s http://localhost:8770/health > /dev/null 2>&1; then
        echo "[$(date)] P2P Health Endpoint: OK"
    else
        echo "[$(date)] P2P Health Endpoint: NOT RESPONDING"
    fi
}

# ============================================
# Main execution
# ============================================
main() {
    echo "[$(date)] =========================================="
    echo "[$(date)] RingRift Vast.ai Onstart Script"
    echo "[$(date)] =========================================="

    install_tailscale
    setup_ringrift
    start_keepalive
    start_p2p
    bootstrap_models
    health_check

    echo "[$(date)] =========================================="
    echo "[$(date)] Onstart script complete!"
    echo "[$(date)] =========================================="
}

main "$@"
