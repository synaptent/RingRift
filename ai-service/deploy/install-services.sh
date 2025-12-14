#!/bin/bash
# Install RingRift AI services on a Linux host
# Usage: ./install-services.sh [service_name]
# If no service_name is provided, installs all services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

# Services to install
SERVICES=(
    "unified-ai-loop"
    "streaming-data-collector"
    "shadow-tournament"
    "model-promoter"
)

install_service() {
    local service_name=$1
    local service_file="${SCRIPT_DIR}/systemd/${service_name}.service"

    if [[ ! -f "$service_file" ]]; then
        echo "Error: Service file not found: $service_file"
        return 1
    fi

    echo "Installing ${service_name}..."
    sudo cp "$service_file" "${SYSTEMD_DIR}/"
    sudo systemctl daemon-reload
    echo "  Installed: ${service_name}.service"
}

enable_service() {
    local service_name=$1
    echo "Enabling ${service_name}..."
    sudo systemctl enable "${service_name}.service"
}

start_service() {
    local service_name=$1
    echo "Starting ${service_name}..."
    sudo systemctl start "${service_name}.service"
    sudo systemctl status "${service_name}.service" --no-pager
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    # Install all services
    echo "Installing all RingRift AI services..."
    for service in "${SERVICES[@]}"; do
        install_service "$service"
    done

    echo ""
    echo "Services installed. To enable and start:"
    echo "  sudo systemctl enable --now unified-ai-loop"
    echo ""
    echo "Or for standalone components:"
    echo "  sudo systemctl enable --now streaming-data-collector"
    echo "  sudo systemctl enable --now shadow-tournament"
    echo "  sudo systemctl enable --now model-promoter"
else
    # Install specific service
    for service in "$@"; do
        install_service "$service"
    done
fi

echo ""
echo "Done. Check status with:"
echo "  sudo systemctl status unified-ai-loop"
echo "  journalctl -u unified-ai-loop -f"
