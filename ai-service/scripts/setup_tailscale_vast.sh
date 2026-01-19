#!/bin/bash
# Setup Tailscale on Vast.ai nodes for P2P mesh connectivity
#
# Usage:
#   TAILSCALE_AUTH_KEY=tskey-auth-xxx ./setup_tailscale_vast.sh
#
# Or run remotely:
#   ssh root@vast-node 'TAILSCALE_AUTH_KEY=tskey-auth-xxx bash -s' < setup_tailscale_vast.sh
#
# Jan 19, 2026: Part of P2P reliability improvements for CGNAT bypass

set -e

echo "=== Tailscale Setup for Vast.ai Node ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Warning: Not running as root, some operations may fail"
fi

# Check for auth key
if [ -z "$TAILSCALE_AUTH_KEY" ]; then
    echo "ERROR: TAILSCALE_AUTH_KEY environment variable not set"
    echo "Get an auth key from: https://login.tailscale.com/admin/settings/keys"
    echo "Use a reusable, ephemeral key for Vast nodes"
    exit 1
fi

# Check if Tailscale is already installed and running
if command -v tailscale &> /dev/null; then
    echo "Tailscale is already installed"

    # Check if already connected
    if tailscale status &> /dev/null; then
        TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
        if [ -n "$TAILSCALE_IP" ]; then
            echo "Tailscale is already connected: $TAILSCALE_IP"
            echo "Status:"
            tailscale status
            exit 0
        fi
    fi

    echo "Tailscale installed but not connected, will authenticate..."
else
    echo "Installing Tailscale..."

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        curl -fsSL https://tailscale.com/install.sh | sh
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        curl -fsSL https://tailscale.com/install.sh | sh
    elif command -v apk &> /dev/null; then
        # Alpine
        apk add tailscale
    else
        echo "ERROR: Unsupported package manager"
        exit 1
    fi

    echo "Tailscale installed successfully"
fi

# Start tailscaled if not running
echo "Starting tailscaled..."
if ! pgrep -x tailscaled > /dev/null; then
    # Try systemd first
    if command -v systemctl &> /dev/null; then
        systemctl start tailscaled 2>/dev/null || true
        systemctl enable tailscaled 2>/dev/null || true
    fi

    # If systemd didn't work, start directly
    if ! pgrep -x tailscaled > /dev/null; then
        echo "Starting tailscaled directly..."
        tailscaled --state=/var/lib/tailscale/tailscaled.state &
        sleep 3
    fi
fi

# Verify tailscaled is running
if ! pgrep -x tailscaled > /dev/null; then
    echo "ERROR: Failed to start tailscaled"
    exit 1
fi

echo "tailscaled is running"

# Authenticate with Tailscale
echo "Authenticating with Tailscale..."

# Get hostname for Tailscale node name
NODE_NAME=$(hostname | sed 's/[^a-zA-Z0-9-]/-/g')

# Use tailscale up with auth key
# --accept-routes: Accept routes advertised by other nodes
# --accept-dns=false: Don't use Tailscale DNS (avoid conflicts in containers)
# --hostname: Set a recognizable hostname
tailscale up \
    --authkey="$TAILSCALE_AUTH_KEY" \
    --hostname="vast-${NODE_NAME}" \
    --accept-routes \
    --accept-dns=false \
    --reset

# Wait for connection
echo "Waiting for Tailscale connection..."
sleep 5

# Verify connection
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
if [ -z "$TAILSCALE_IP" ]; then
    echo "ERROR: Failed to get Tailscale IP"
    tailscale status
    exit 1
fi

echo ""
echo "=== Tailscale Setup Complete ==="
echo "Tailscale IP: $TAILSCALE_IP"
echo "Hostname: vast-${NODE_NAME}"
echo ""
echo "Status:"
tailscale status

# Output for automation
echo ""
echo "# Add to distributed_hosts.yaml:"
echo "vast-${NODE_NAME}:"
echo "  tailscale_ip: $TAILSCALE_IP"
echo "  tailscale_enabled: true"
