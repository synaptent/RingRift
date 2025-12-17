#!/bin/bash
# ============================================================================
# RingRift LaunchD Services Installer
# ============================================================================
#
# This script installs RingRift daemon services as macOS LaunchAgents.
# Compatible with macOS default bash (3.x) and newer versions.
#
# Usage:
#   ./scripts/install_launchd_services.sh [--service NAME] [--uninstall]
#
# Services:
#   - data-aggregator: Syncs selfplay data from cluster to local/S3
#   - p2p-orchestrator: P2P networking for distributed selfplay
#
# Examples:
#   ./scripts/install_launchd_services.sh --service data-aggregator
#   ./scripts/install_launchd_services.sh --service all
#   ./scripts/install_launchd_services.sh --service data-aggregator --uninstall
#
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$AI_SERVICE_DIR/.." && pwd)"
LAUNCHD_DIR="$AI_SERVICE_DIR/config/launchd"
LAUNCHAGENTS_DIR="$HOME/Library/LaunchAgents"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"
NODE_ID="${NODE_ID:-$(hostname -s)}"

# Service definitions (bash 3.x compatible)
SERVICE_NAMES="data-aggregator p2p-orchestrator"

get_service_label() {
    local name="$1"
    case "$name" in
        data-aggregator)
            echo "com.ringrift.data-aggregator"
            ;;
        p2p-orchestrator)
            echo "com.ringrift.p2p-orchestrator"
            ;;
        *)
            echo ""
            ;;
    esac
}

usage() {
    echo "Usage: $0 [--service NAME] [--uninstall]"
    echo ""
    echo "Options:"
    echo "  --service NAME   Service to install (data-aggregator, p2p-orchestrator, all)"
    echo "  --uninstall      Uninstall the service instead of installing"
    echo "  --status         Show status of all services"
    echo "  --python PATH    Path to Python executable"
    echo "  --node-id ID     Node ID for P2P orchestrator"
    echo ""
    exit 1
}

install_service() {
    local name="$1"
    local label
    label=$(get_service_label "$name")

    if [ -z "$label" ]; then
        echo "Unknown service: $name"
        exit 1
    fi

    local plist_src="$LAUNCHD_DIR/$label.plist"
    local plist_dst="$LAUNCHAGENTS_DIR/$label.plist"

    if [ ! -f "$plist_src" ]; then
        echo "Error: Plist not found: $plist_src"
        exit 1
    fi

    echo "Installing $name service..."

    # Create logs directory
    mkdir -p "$AI_SERVICE_DIR/logs"

    # Stop existing service if running
    launchctl unload "$plist_dst" 2>/dev/null || true

    # Copy and configure plist
    mkdir -p "$LAUNCHAGENTS_DIR"

    # Replace placeholders
    sed -e "s|__RINGRIFT_AI_SERVICE_DIR__|$AI_SERVICE_DIR|g" \
        -e "s|__RINGRIFT_REPO_ROOT__|$REPO_ROOT|g" \
        -e "s|__RINGRIFT_PYTHON__|$PYTHON_PATH|g" \
        -e "s|__RINGRIFT_P2P_PYTHON__|$PYTHON_PATH|g" \
        -e "s|__RINGRIFT_NODE_ID__|$NODE_ID|g" \
        "$plist_src" > "$plist_dst"

    # Load service
    launchctl load "$plist_dst"

    echo "  -> Installed: $plist_dst"
    echo "  -> Python: $PYTHON_PATH"
    echo "  -> Working dir: $AI_SERVICE_DIR"

    # Check if running
    sleep 1
    if launchctl list 2>/dev/null | grep -q "$label"; then
        echo "  -> Status: RUNNING"
    else
        echo "  -> Status: NOT RUNNING (check logs)"
    fi

    echo ""
}

uninstall_service() {
    local name="$1"
    local label
    label=$(get_service_label "$name")
    local plist_dst="$LAUNCHAGENTS_DIR/$label.plist"

    echo "Uninstalling $name service..."

    # Unload if running
    launchctl unload "$plist_dst" 2>/dev/null || true

    # Remove plist
    rm -f "$plist_dst"

    echo "  -> Removed: $plist_dst"
    echo ""
}

show_status() {
    echo "RingRift Service Status"
    echo "========================"
    echo ""

    for name in $SERVICE_NAMES; do
        local label
        label=$(get_service_label "$name")
        local plist_dst="$LAUNCHAGENTS_DIR/$label.plist"

        printf "%-20s " "$name:"

        if [ -f "$plist_dst" ]; then
            if launchctl list 2>/dev/null | grep -q "$label"; then
                echo "[RUNNING]"
            else
                echo "[INSTALLED but NOT RUNNING]"
            fi
        else
            echo "[NOT INSTALLED]"
        fi
    done

    echo ""
}

# Parse arguments
SERVICE=""
UNINSTALL=false
STATUS=false

while [ $# -gt 0 ]; do
    case "$1" in
        --service)
            SERVICE="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --status)
            STATUS=true
            shift
            ;;
        --python)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --node-id)
            NODE_ID="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Show status
if $STATUS; then
    show_status
    exit 0
fi

# Validate
if [ -z "$SERVICE" ]; then
    echo "No service specified."
    echo ""
    show_status
    echo "Use --service NAME to install a service."
    exit 0
fi

# Handle 'all' service
if [ "$SERVICE" = "all" ]; then
    for name in $SERVICE_NAMES; do
        if $UNINSTALL; then
            uninstall_service "$name"
        else
            install_service "$name"
        fi
    done
else
    if $UNINSTALL; then
        uninstall_service "$SERVICE"
    else
        install_service "$SERVICE"
    fi
fi

echo "Done!"
