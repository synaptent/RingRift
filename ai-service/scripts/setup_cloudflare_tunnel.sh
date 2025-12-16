#!/bin/bash
# ==============================================================================
# Cloudflare Tunnel Setup for RingRift P2P
# ==============================================================================
#
# This script sets up a Cloudflare quick tunnel for the P2P orchestrator.
# Quick tunnels are free and don't require a Cloudflare account, but the URL
# changes on restart.
#
# For persistent URLs, create a Cloudflare account and use named tunnels:
#   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps
#
# Usage:
#   ./setup_cloudflare_tunnel.sh [port]
#
# ==============================================================================

set -e

P2P_PORT="${1:-8770}"
LOG_FILE="/var/log/ringrift/cloudflared.log"
TUNNEL_URL_FILE="/etc/ringrift/tunnel_url"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    log_info "Installing cloudflared..."
    curl -fsSL https://pkg.cloudflare.com/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    dpkg -i /tmp/cloudflared.deb || apt-get install -f -y
    rm /tmp/cloudflared.deb
fi

# Check if P2P is running
if ! curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1; then
    log_error "P2P orchestrator not running on port $P2P_PORT"
    exit 1
fi

# Kill any existing quick tunnels
log_info "Stopping any existing quick tunnels..."
pkill -f "tunnel --url http://localhost:$P2P_PORT" 2>/dev/null || true
pkill -f "tunnel --url http://127.0.0.1:$P2P_PORT" 2>/dev/null || true
sleep 2

# Start quick tunnel
log_info "Starting Cloudflare quick tunnel for port $P2P_PORT..."
mkdir -p "$(dirname $LOG_FILE)" "$(dirname $TUNNEL_URL_FILE)"
nohup cloudflared tunnel --url "http://127.0.0.1:$P2P_PORT" > "$LOG_FILE" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
log_info "Waiting for tunnel URL..."
for i in {1..30}; do
    TUNNEL_URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" "$LOG_FILE" 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    log_error "Failed to get tunnel URL. Check $LOG_FILE"
    exit 1
fi

# Save tunnel URL
echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"

# Test tunnel
log_info "Testing tunnel..."
sleep 3
if curl -s --connect-timeout 10 "$TUNNEL_URL/health" > /dev/null 2>&1; then
    log_info "Tunnel is working!"
else
    log_warn "Tunnel may take a few more seconds to propagate"
fi

log_info "=============================================="
log_info "Cloudflare Tunnel Setup Complete"
log_info "=============================================="
log_info "Tunnel URL: $TUNNEL_URL"
log_info "Tunnel PID: $TUNNEL_PID"
log_info "Log file: $LOG_FILE"
log_info ""
log_info "To use this tunnel from vast nodes, update /etc/ringrift/node.conf:"
log_info "  PEERS=$TUNNEL_URL"
log_info ""
log_info "Note: Quick tunnel URLs change on restart."
log_info "For persistent URLs, set up a Cloudflare account."
log_info "=============================================="
