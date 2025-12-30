#!/bin/bash
# Setup Cloudflare Tunnel for RingRift Cluster
#
# This script sets up cloudflared tunnel on additional backup nodes.
# The tunnel credentials are already configured and should be copied from the primary node.
#
# Primary node: nebius-backbone-1 (100.110.28.41)
# Backup node: hetzner-cpu1 (100.94.174.19)
#
# Tunnel Details:
#   Name: ringrift-dashboard
#   ID: 02a725cf-3519-474f-ab16-d40c3aa1edae
#   Backend: P2P leader at 100.94.201.92:8770
#
# Public URLs:
#   - https://dashboard.ringrift.ai/
#   - https://p2p.ringrift.ai/status
#   - https://queue.ringrift.ai/

set -e

TUNNEL_ID="02a725cf-3519-474f-ab16-d40c3aa1edae"
P2P_LEADER="100.94.201.92"
P2P_PORT="8770"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."
    if [ -f /etc/debian_version ]; then
        curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        dpkg -i cloudflared.deb
        rm cloudflared.deb
    elif [ -f /etc/redhat-release ]; then
        curl -L --output cloudflared.rpm https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm
        rpm -i cloudflared.rpm
        rm cloudflared.rpm
    else
        echo "Unsupported OS. Please install cloudflared manually."
        exit 1
    fi
fi

# Create directories
mkdir -p /etc/cloudflared
mkdir -p ~/.cloudflared

# Check for credentials file
CREDS_FILE="$HOME/.cloudflared/${TUNNEL_ID}.json"
if [ ! -f "$CREDS_FILE" ]; then
    echo "ERROR: Credentials file not found at $CREDS_FILE"
    echo ""
    echo "Please copy the credentials from the primary node:"
    echo "  scp ubuntu@100.110.28.41:~/.cloudflared/${TUNNEL_ID}.json ~/.cloudflared/"
    exit 1
fi

# Create config file
cat > /etc/cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: ${CREDS_FILE}

ingress:
  - hostname: dashboard.ringrift.ai
    service: http://${P2P_LEADER}:${P2P_PORT}
  - hostname: p2p.ringrift.ai
    service: http://${P2P_LEADER}:${P2P_PORT}
  - hostname: queue.ringrift.ai
    service: http://${P2P_LEADER}:${P2P_PORT}
  - service: http_status:404
EOF

echo "Created /etc/cloudflared/config.yml"

# Create systemd service
cat > /etc/systemd/system/cloudflared.service << 'EOF'
[Unit]
Description=cloudflared tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/cloudflared tunnel --config /etc/cloudflared/config.yml run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Handle different cloudflared install locations
if [ -f /usr/local/bin/cloudflared ] && [ ! -f /usr/bin/cloudflared ]; then
    sed -i 's|/usr/bin/cloudflared|/usr/local/bin/cloudflared|g' /etc/systemd/system/cloudflared.service
fi

echo "Created /etc/systemd/system/cloudflared.service"

# Enable and start service
systemctl daemon-reload
systemctl enable cloudflared
systemctl restart cloudflared

echo ""
echo "Cloudflare tunnel service started. Checking status..."
sleep 3
systemctl status cloudflared --no-pager | head -15

echo ""
echo "Test with:"
echo "  curl https://dashboard.ringrift.ai/"
echo "  curl https://p2p.ringrift.ai/status"
