#!/bin/bash
#
# Bootstrap P2P networking on container nodes (Vast.ai, RunPod, etc.)
#
# Container environments cannot use kernel Tailscale. This script sets up
# userspace Tailscale with SOCKS5 proxy for P2P mesh connectivity.
#
# Usage:
#   # With auth key in environment
#   TAILSCALE_AUTH_KEY=tskey-xxx ./bootstrap_container_p2p.sh
#
#   # Or pass auth key as argument
#   ./bootstrap_container_p2p.sh tskey-xxx
#
# Requirements:
#   - Root access (for starting tailscaled)
#   - curl (for installing Tailscale)
#
# After running this script:
#   - Tailscale runs in userspace mode
#   - SOCKS5 proxy available at localhost:1055
#   - RINGRIFT_SOCKS_PROXY environment variable set
#
# December 2025

set -e

# Configuration
SOCKS5_PORT="${RINGRIFT_TAILSCALE_SOCKS_PORT:-1055}"
STATE_DIR="/var/lib/tailscale"
RUN_DIR="/var/run/tailscale"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get auth key from argument or environment
AUTH_KEY="${1:-$TAILSCALE_AUTH_KEY}"

# Check if running in container
is_container() {
    if [ -f "/.dockerenv" ]; then
        return 0
    fi
    if [ -f "/proc/1/cgroup" ] && grep -q -E "(docker|lxc|kubepods)" /proc/1/cgroup 2>/dev/null; then
        return 0
    fi
    if [ -f "/run/.containerenv" ]; then
        return 0
    fi
    if [ -n "$VAST_CONTAINERLABEL" ] || [ -n "$RUNPOD_POD_ID" ]; then
        return 0
    fi
    return 1
}

# Check if Tailscale is installed
check_tailscale_installed() {
    command -v tailscale &> /dev/null && command -v tailscaled &> /dev/null
}

# Install Tailscale
install_tailscale() {
    log_info "Installing Tailscale..."

    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
        curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list | tee /etc/apt/sources.list.d/tailscale.list
        apt-get update -qq
        apt-get install -y tailscale
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        yum install -y yum-utils
        yum-config-manager --add-repo https://pkgs.tailscale.com/stable/centos/8/tailscale.repo
        yum install -y tailscale
    else
        # Fallback to official install script
        curl -fsSL https://tailscale.com/install.sh | sh
    fi

    if ! check_tailscale_installed; then
        log_error "Tailscale installation failed"
        exit 1
    fi

    log_info "Tailscale installed successfully"
}

# Check if tailscaled is running
is_tailscaled_running() {
    pgrep -x tailscaled > /dev/null 2>&1
}

# Start tailscaled in userspace mode
start_tailscaled() {
    if is_tailscaled_running; then
        log_info "tailscaled already running"
        return 0
    fi

    log_info "Starting tailscaled in userspace mode..."

    # Create directories
    mkdir -p "$STATE_DIR" "$RUN_DIR"

    # Start tailscaled with userspace networking and SOCKS5 proxy
    tailscaled \
        --tun=userspace-networking \
        --socks5-server="localhost:${SOCKS5_PORT}" \
        --state="${STATE_DIR}/tailscaled.state" \
        --socket="${RUN_DIR}/tailscaled.sock" \
        > /var/log/tailscaled.log 2>&1 &

    # Wait for startup
    for i in {1..10}; do
        if is_tailscaled_running; then
            log_info "tailscaled started (PID: $(pgrep -x tailscaled))"
            return 0
        fi
        sleep 1
    done

    log_error "tailscaled failed to start. Check /var/log/tailscaled.log"
    exit 1
}

# Authenticate with Tailscale
authenticate_tailscale() {
    if [ -z "$AUTH_KEY" ]; then
        log_warn "No TAILSCALE_AUTH_KEY provided"
        log_info "Checking if already authenticated..."

        if tailscale status --json 2>/dev/null | grep -q '"Online":true'; then
            log_info "Already authenticated and connected"
            return 0
        fi

        log_error "Not authenticated. Please provide TAILSCALE_AUTH_KEY"
        exit 1
    fi

    log_info "Authenticating with Tailscale..."

    tailscale up \
        --auth-key="$AUTH_KEY" \
        --accept-routes \
        --hostname="$(hostname)-container"

    # Verify connection
    for i in {1..30}; do
        if tailscale status --json 2>/dev/null | grep -q '"Online":true'; then
            log_info "Successfully connected to Tailscale network"
            return 0
        fi
        sleep 1
    done

    log_error "Failed to connect to Tailscale network"
    exit 1
}

# Verify SOCKS5 proxy is working
verify_socks5() {
    log_info "Verifying SOCKS5 proxy on port ${SOCKS5_PORT}..."

    for i in {1..10}; do
        if nc -z 127.0.0.1 "$SOCKS5_PORT" 2>/dev/null || \
           (echo > /dev/tcp/127.0.0.1/"$SOCKS5_PORT") 2>/dev/null; then
            log_info "SOCKS5 proxy is available"
            return 0
        fi
        sleep 1
    done

    log_error "SOCKS5 proxy not available on port ${SOCKS5_PORT}"
    exit 1
}

# Export environment variables
export_env() {
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "unknown")

    log_info "Setting environment variables..."

    # Export for current session
    export RINGRIFT_SOCKS_PROXY="socks5://localhost:${SOCKS5_PORT}"
    export RINGRIFT_TAILSCALE_IP="$TAILSCALE_IP"

    # Add to profile for persistence
    PROFILE_FILE="${HOME}/.bashrc"
    if [ -f "${HOME}/.profile" ]; then
        PROFILE_FILE="${HOME}/.profile"
    fi

    if ! grep -q "RINGRIFT_SOCKS_PROXY" "$PROFILE_FILE" 2>/dev/null; then
        echo "" >> "$PROFILE_FILE"
        echo "# RingRift P2P networking (added by bootstrap_container_p2p.sh)" >> "$PROFILE_FILE"
        echo "export RINGRIFT_SOCKS_PROXY=\"socks5://localhost:${SOCKS5_PORT}\"" >> "$PROFILE_FILE"
    fi

    log_info "Environment configured:"
    log_info "  RINGRIFT_SOCKS_PROXY=socks5://localhost:${SOCKS5_PORT}"
    log_info "  Tailscale IP: $TAILSCALE_IP"
}

# Add to startup script for persistence across container restarts
add_to_startup() {
    # For Vast.ai: onstart.sh
    if [ -f "/opt/onstart.sh" ] || [ -n "$VAST_CONTAINERLABEL" ]; then
        STARTUP_FILE="/opt/onstart.sh"
        if [ -f "$STARTUP_FILE" ]; then
            if ! grep -q "bootstrap_container_p2p.sh" "$STARTUP_FILE" 2>/dev/null; then
                log_info "Adding to Vast.ai onstart.sh..."
                echo "" >> "$STARTUP_FILE"
                echo "# Start Tailscale for P2P networking" >> "$STARTUP_FILE"
                echo "if [ -f /workspace/ringrift/ai-service/scripts/bootstrap_container_p2p.sh ]; then" >> "$STARTUP_FILE"
                echo "    /workspace/ringrift/ai-service/scripts/bootstrap_container_p2p.sh" >> "$STARTUP_FILE"
                echo "fi" >> "$STARTUP_FILE"
            fi
        fi
    fi

    # For RunPod: check for startup script location
    if [ -n "$RUNPOD_POD_ID" ]; then
        log_info "RunPod detected - add to /start.sh or /workspace/start.sh manually"
    fi
}

# Main
main() {
    log_info "=== RingRift Container P2P Bootstrap ==="

    # Verify container environment
    if ! is_container; then
        log_warn "Not running in container - native Tailscale should work"
        log_info "If you still need userspace mode, continue with RINGRIFT_NEEDS_USERSPACE_TAILSCALE=1"
    fi

    # Install Tailscale if needed
    if ! check_tailscale_installed; then
        install_tailscale
    else
        log_info "Tailscale already installed"
    fi

    # Start tailscaled
    start_tailscaled

    # Authenticate
    authenticate_tailscale

    # Verify SOCKS5
    verify_socks5

    # Export environment
    export_env

    # Add to startup for persistence
    add_to_startup

    log_info "=== P2P Bootstrap Complete ==="
    log_info "Container is ready to join P2P network"
    log_info ""
    log_info "To start P2P orchestrator:"
    log_info "  cd /workspace/ringrift/ai-service"
    log_info "  PYTHONPATH=. python scripts/p2p_orchestrator.py"
}

main "$@"
