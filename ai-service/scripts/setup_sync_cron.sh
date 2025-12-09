#!/bin/bash
# setup_sync_cron.sh - Set up automated periodic sync of selfplay data
#
# This script configures a launchd job (macOS) or cron job (Linux) to
# periodically run sync_selfplay_data.sh, pulling game databases from
# all configured remote instances.
#
# Usage:
#   ./scripts/setup_sync_cron.sh [OPTIONS]
#
# Options:
#   --interval MINS   Sync interval in minutes (default: 30)
#   --enable          Enable/install the sync job
#   --disable         Disable/remove the sync job
#   --status          Show current status
#   --run-now         Run sync immediately (in addition to scheduled)
#   -h, --help        Show this help message
#
# Examples:
#   ./scripts/setup_sync_cron.sh --enable --interval 15
#   ./scripts/setup_sync_cron.sh --status
#   ./scripts/setup_sync_cron.sh --disable
#
# Notes:
#   - On macOS, uses launchd (~/Library/LaunchAgents)
#   - On Linux, uses user crontab
#   - Logs are written to ai-service/logs/sync/
#   - Sync only runs when the machine is on (not wake-from-sleep)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$AI_SERVICE_DIR/logs/sync"

# Job identifiers
LAUNCHD_LABEL="com.ringrift.selfplay-sync"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

# Default interval (minutes)
INTERVAL_MINS=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_usage() {
    head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
}

# Parse arguments
ACTION=""
RUN_NOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval) INTERVAL_MINS="$2"; shift 2 ;;
        --enable) ACTION="enable"; shift ;;
        --disable) ACTION="disable"; shift ;;
        --status) ACTION="status"; shift ;;
        --run-now) RUN_NOW=true; shift ;;
        --help|-h) print_usage; exit 0 ;;
        *) log_error "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$ACTION" && "$RUN_NOW" == "false" ]]; then
    print_usage
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Detect OS
is_macos() {
    [[ "$(uname)" == "Darwin" ]]
}

# ============================================
# macOS launchd functions
# ============================================

create_launchd_plist() {
    local interval_secs=$((INTERVAL_MINS * 60))

    cat > "$LAUNCHD_PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCHD_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${SCRIPT_DIR}/sync_selfplay_data.sh</string>
        <string>--merge</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${AI_SERVICE_DIR}</string>

    <key>StartInterval</key>
    <integer>${interval_secs}</integer>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/sync_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/sync_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>

    <key>RunAtLoad</key>
    <false/>

    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF
}

enable_launchd() {
    log_info "Creating launchd plist at $LAUNCHD_PLIST"
    create_launchd_plist

    log_info "Loading launchd job..."
    launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
    launchctl load "$LAUNCHD_PLIST"

    log_success "Sync job enabled (every $INTERVAL_MINS minutes)"
    log_info "Logs: $LOG_DIR/sync_stdout.log"
}

disable_launchd() {
    if [[ -f "$LAUNCHD_PLIST" ]]; then
        log_info "Unloading launchd job..."
        launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
        rm -f "$LAUNCHD_PLIST"
        log_success "Sync job disabled"
    else
        log_warning "No sync job found"
    fi
}

status_launchd() {
    echo "=== Sync Job Status (macOS launchd) ==="
    echo ""

    if [[ -f "$LAUNCHD_PLIST" ]]; then
        echo "Plist: $LAUNCHD_PLIST"
        echo "Status: $(launchctl list | grep -q "$LAUNCHD_LABEL" && echo "LOADED" || echo "NOT LOADED")"

        # Extract interval from plist
        if command -v plutil &>/dev/null; then
            local interval=$(plutil -extract StartInterval raw "$LAUNCHD_PLIST" 2>/dev/null || echo "?")
            echo "Interval: $((interval / 60)) minutes"
        fi
    else
        echo "Status: NOT CONFIGURED"
        echo ""
        echo "Run with --enable to set up periodic sync"
    fi

    echo ""
    echo "=== Recent Sync Logs ==="
    if [[ -f "$LOG_DIR/sync_stdout.log" ]]; then
        tail -20 "$LOG_DIR/sync_stdout.log"
    else
        echo "(no logs yet)"
    fi
}

# ============================================
# Linux cron functions
# ============================================

enable_cron() {
    local cron_entry="*/$INTERVAL_MINS * * * * cd $AI_SERVICE_DIR && $SCRIPT_DIR/sync_selfplay_data.sh --merge >> $LOG_DIR/sync.log 2>&1"

    # Remove existing entry if present
    (crontab -l 2>/dev/null | grep -v "sync_selfplay_data.sh") | crontab -

    # Add new entry
    (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -

    log_success "Cron job added (every $INTERVAL_MINS minutes)"
    log_info "Logs: $LOG_DIR/sync.log"
}

disable_cron() {
    (crontab -l 2>/dev/null | grep -v "sync_selfplay_data.sh") | crontab -
    log_success "Cron job removed"
}

status_cron() {
    echo "=== Sync Job Status (Linux cron) ==="
    echo ""

    local cron_line=$(crontab -l 2>/dev/null | grep "sync_selfplay_data.sh" || true)
    if [[ -n "$cron_line" ]]; then
        echo "Cron entry: $cron_line"
    else
        echo "Status: NOT CONFIGURED"
        echo ""
        echo "Run with --enable to set up periodic sync"
    fi

    echo ""
    echo "=== Recent Sync Logs ==="
    if [[ -f "$LOG_DIR/sync.log" ]]; then
        tail -20 "$LOG_DIR/sync.log"
    else
        echo "(no logs yet)"
    fi
}

# ============================================
# Main logic
# ============================================

case "$ACTION" in
    enable)
        if is_macos; then
            enable_launchd
        else
            enable_cron
        fi
        ;;
    disable)
        if is_macos; then
            disable_launchd
        else
            disable_cron
        fi
        ;;
    status)
        if is_macos; then
            status_launchd
        else
            status_cron
        fi
        ;;
esac

# Run sync now if requested
if [[ "$RUN_NOW" == "true" ]]; then
    log_info "Running sync now..."
    "$SCRIPT_DIR/sync_selfplay_data.sh" --merge
fi
