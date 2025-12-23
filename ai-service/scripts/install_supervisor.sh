#!/bin/bash
# Install RingRift Cluster Supervisor on a node
# Usage: ./install_supervisor.sh [--enable] [--start]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$AI_SERVICE_DIR/config/systemd/ringrift-supervisor.service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    error "Please run with sudo: sudo $0 $*"
fi

# Parse arguments
ENABLE=false
START=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable) ENABLE=true; shift ;;
        --start) START=true; shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Check for systemd
if ! command -v systemctl &> /dev/null; then
    error "systemd not found. This script requires systemd."
fi

# Check for service file
if [ ! -f "$SERVICE_FILE" ]; then
    error "Service file not found: $SERVICE_FILE"
fi

# Get the actual user (not root)
REAL_USER="${SUDO_USER:-ubuntu}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

info "Installing RingRift Supervisor for user: $REAL_USER"

# Create logs directory
LOGS_DIR="$AI_SERVICE_DIR/logs"
mkdir -p "$LOGS_DIR"
chown "$REAL_USER:$REAL_USER" "$LOGS_DIR"

# Customize service file for this system
TEMP_SERVICE="/tmp/ringrift-supervisor.service"
sed -e "s|User=ubuntu|User=$REAL_USER|g" \
    -e "s|Group=ubuntu|Group=$REAL_USER|g" \
    -e "s|/home/ubuntu/ringrift|$REAL_HOME/ringrift|g" \
    "$SERVICE_FILE" > "$TEMP_SERVICE"

# Install service file
info "Installing systemd service..."
cp "$TEMP_SERVICE" /etc/systemd/system/ringrift-supervisor.service
rm "$TEMP_SERVICE"

# Reload systemd
info "Reloading systemd daemon..."
systemctl daemon-reload

# Install sdnotify if not present (for systemd watchdog)
if ! python3 -c "import sdnotify" 2>/dev/null; then
    info "Installing sdnotify for systemd integration..."
    pip3 install sdnotify --quiet || warn "Could not install sdnotify (optional)"
fi

# Enable service
if [ "$ENABLE" = true ]; then
    info "Enabling service to start on boot..."
    systemctl enable ringrift-supervisor
fi

# Start service
if [ "$START" = true ]; then
    info "Starting supervisor service..."
    systemctl start ringrift-supervisor
    sleep 2

    # Check status
    if systemctl is-active --quiet ringrift-supervisor; then
        info "Supervisor started successfully!"
        systemctl status ringrift-supervisor --no-pager
    else
        error "Failed to start supervisor. Check logs: journalctl -u ringrift-supervisor"
    fi
fi

echo ""
info "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start ringrift-supervisor    # Start supervisor"
echo "  sudo systemctl stop ringrift-supervisor     # Stop supervisor"
echo "  sudo systemctl restart ringrift-supervisor  # Restart supervisor"
echo "  sudo systemctl status ringrift-supervisor   # Check status"
echo "  journalctl -u ringrift-supervisor -f        # View logs"
echo "  curl http://localhost:8775/status           # HTTP status"
echo ""

# Quick status check
if [ "$ENABLE" = false ] && [ "$START" = false ]; then
    echo "To enable and start the service:"
    echo "  sudo systemctl enable --now ringrift-supervisor"
fi
