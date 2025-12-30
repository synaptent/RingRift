#!/bin/bash
# Setup Cloudflare Tunnel for RingRift Cluster
#
# Usage: ./setup_cloudflare_tunnel.sh <TUNNEL_TOKEN>

set -e

TUNNEL_TOKEN="${1:-}"

if [ -z "$TUNNEL_TOKEN" ]; then
    echo "Usage: $0 <TUNNEL_TOKEN>"
    echo ""
    echo "Get the tunnel token from Cloudflare Dashboard:"
    echo "  1. Go to https://one.dash.cloudflare.com/"
    echo "  2. Networks -> Tunnels -> Create tunnel"
    echo "  3. Name: ringrift-cluster"
    echo "  4. Copy the token (starts with eyJ...)"
    exit 1
fi

# Nodes
PRIMARY_HOST="100.110.28.41"  # nebius-backbone-1
BACKUP_HOST="100.94.174.19"   # hetzner-cpu1
P2P_LEADER="100.94.201.92"    # vultr-a100-20gb

echo "Setting up Cloudflare Tunnel for RingRift..."

for NODE in "ubuntu@$PRIMARY_HOST:nebius-backbone-1" "root@$BACKUP_HOST:hetzner-cpu1"; do
    IFS=':' read -r USERHOST NAME <<< "$NODE"
    echo ">>> Setting up $NAME..."
    
    ssh -o ConnectTimeout=10 $USERHOST "
        sudo systemctl stop cloudflared 2>/dev/null || true
        sudo cloudflared service uninstall 2>/dev/null || true
        sudo cloudflared service install $TUNNEL_TOKEN
        sudo systemctl enable cloudflared
        sudo systemctl start cloudflared
        sleep 2
        sudo systemctl status cloudflared --no-pager | head -10
    " || echo "Warning: Failed to setup $NAME"
done

echo ""
echo "Setup complete! Test with:"
echo "  curl https://dashboard.ringrift.ai/"
echo "  curl https://p2p.ringrift.ai/status"
