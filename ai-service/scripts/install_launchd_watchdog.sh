#!/bin/bash
#
# Install RingRift Master Loop Watchdog as a launchd service
#
# This script installs the launchd plist that keeps master_loop.py running.
# Run this once to enable automatic restart of the master loop.
#
# Usage:
#   ./scripts/install_launchd_watchdog.sh          # Install and start
#   ./scripts/install_launchd_watchdog.sh --status # Check status
#   ./scripts/install_launchd_watchdog.sh --uninstall # Remove
#
# December 2025: Created for 48-hour autonomous operation enablement.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_SRC="$AI_SERVICE_DIR/config/launchd/com.ringrift.master-loop.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.ringrift.master-loop.plist"
LABEL="com.ringrift.master-loop"

# Ensure log directory exists
mkdir -p "$AI_SERVICE_DIR/logs"

case "${1:-install}" in
    install)
        echo "Installing RingRift Master Loop Watchdog..."

        # Check if source plist exists
        if [ ! -f "$PLIST_SRC" ]; then
            echo "ERROR: Source plist not found at $PLIST_SRC"
            exit 1
        fi

        # Create LaunchAgents directory if needed
        mkdir -p "$HOME/Library/LaunchAgents"

        # Unload existing if present
        if launchctl list "$LABEL" &>/dev/null; then
            echo "Unloading existing service..."
            launchctl unload "$PLIST_DST" 2>/dev/null || true
        fi

        # Copy plist
        cp "$PLIST_SRC" "$PLIST_DST"
        echo "Installed plist to $PLIST_DST"

        # Load the service
        launchctl load "$PLIST_DST"
        echo "Loaded launchd service"

        # Verify it's running
        sleep 2
        if launchctl list "$LABEL" &>/dev/null; then
            echo "SUCCESS: Master loop watchdog is now active"
            launchctl list "$LABEL"
        else
            echo "WARNING: Service loaded but may not be running yet"
        fi
        ;;

    --status|status)
        echo "Checking RingRift Master Loop Watchdog status..."
        if launchctl list "$LABEL" &>/dev/null; then
            echo "Service is LOADED"
            launchctl list "$LABEL"

            # Check if process is running
            if pgrep -f "master_loop.py" >/dev/null; then
                echo ""
                echo "Process is RUNNING:"
                ps aux | grep "[m]aster_loop.py" | head -1
            else
                echo ""
                echo "Process is NOT RUNNING (launchd will restart it)"
            fi
        else
            echo "Service is NOT LOADED"
            echo "Run: $0 install"
        fi
        ;;

    --uninstall|uninstall)
        echo "Uninstalling RingRift Master Loop Watchdog..."

        if launchctl list "$LABEL" &>/dev/null; then
            launchctl unload "$PLIST_DST"
            echo "Unloaded service"
        fi

        if [ -f "$PLIST_DST" ]; then
            rm "$PLIST_DST"
            echo "Removed plist"
        fi

        echo "SUCCESS: Watchdog uninstalled"
        ;;

    --help|help|-h)
        echo "Usage: $0 [install|--status|--uninstall]"
        echo ""
        echo "Commands:"
        echo "  install      Install and start the watchdog (default)"
        echo "  --status     Check current status"
        echo "  --uninstall  Stop and remove the watchdog"
        ;;

    *)
        echo "Unknown option: $1"
        echo "Run: $0 --help"
        exit 1
        ;;
esac
