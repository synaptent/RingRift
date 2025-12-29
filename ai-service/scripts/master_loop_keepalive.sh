#!/bin/bash
# Master Loop Keepalive Script
#
# Simple shell script for cron-based master loop monitoring.
# Use this on systems without systemd or as a backup to the Python watchdog.
#
# Usage:
#   # Run manually
#   ./scripts/master_loop_keepalive.sh
#
#   # Add to cron (every 2 minutes)
#   */2 * * * * /path/to/ai-service/scripts/master_loop_keepalive.sh >> /var/log/master_loop_keepalive.log 2>&1
#
# December 2025: Created as part of 48-hour autonomous operation plan.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$AI_SERVICE_DIR/data/coordination/master_loop.pid"
STATE_DB="$AI_SERVICE_DIR/data/coordination/master_loop_state.db"
HEARTBEAT_THRESHOLD=${RINGRIFT_HEARTBEAT_STALE_THRESHOLD:-90}
LOG_TAG="[MasterLoopKeepalive]"

# Logging helper
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_TAG $1"
}

# Check if master loop is running via PID
check_pid_alive() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi

    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null)

    if [[ -z "$pid" ]]; then
        return 1
    fi

    # Check if process exists
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Get heartbeat age from SQLite
get_heartbeat_age() {
    if [[ ! -f "$STATE_DB" ]]; then
        echo "-1"
        return
    fi

    # Query last_beat from heartbeat table
    local last_beat
    last_beat=$(sqlite3 "$STATE_DB" "SELECT last_beat FROM heartbeat WHERE id = 1;" 2>/dev/null || echo "")

    if [[ -z "$last_beat" ]]; then
        echo "-1"
        return
    fi

    # Calculate age
    local now
    now=$(date +%s)
    local age
    age=$(echo "$now - $last_beat" | bc 2>/dev/null || echo "-1")

    echo "${age%.*}"  # Remove decimal part
}

# Check if heartbeat is stale
is_heartbeat_stale() {
    local age
    age=$(get_heartbeat_age)

    if [[ "$age" == "-1" ]]; then
        # No heartbeat data - assume OK (first run)
        return 1
    fi

    if [[ "$age" -gt "$HEARTBEAT_THRESHOLD" ]]; then
        return 0  # Stale
    else
        return 1  # Fresh
    fi
}

# Kill stale master loop process
kill_stale_process() {
    if [[ ! -f "$PID_FILE" ]]; then
        return
    fi

    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null)

    if [[ -z "$pid" ]]; then
        return
    fi

    log "Killing stale master loop (PID $pid)..."

    # Try SIGTERM first
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5

    # If still running, SIGKILL
    if kill -0 "$pid" 2>/dev/null; then
        log "Process still running, sending SIGKILL..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 2
    fi

    # Remove PID file
    rm -f "$PID_FILE"

    # Remove lock file
    rm -f "$AI_SERVICE_DIR/data/coordination/ringrift_master_loop.lock"
}

# Start master loop
start_master_loop() {
    log "Starting master loop..."

    cd "$AI_SERVICE_DIR"

    # Activate virtualenv if present
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    # Start in background
    PYTHONPATH="$AI_SERVICE_DIR" nohup python scripts/master_loop.py \
        >> "$AI_SERVICE_DIR/logs/master_loop.log" 2>&1 &

    local new_pid=$!
    log "Master loop started (PID $new_pid)"

    # Wait a moment and verify it's running
    sleep 3
    if kill -0 "$new_pid" 2>/dev/null; then
        log "Master loop is running"
        return 0
    else
        log "ERROR: Master loop failed to start"
        return 1
    fi
}

# Main logic
main() {
    log "Checking master loop health..."

    # Check if PID is alive
    if ! check_pid_alive; then
        log "Master loop not running (no PID or process dead)"
        start_master_loop
        exit $?
    fi

    # Check heartbeat freshness
    if is_heartbeat_stale; then
        local age
        age=$(get_heartbeat_age)
        log "Heartbeat stale (${age}s > ${HEARTBEAT_THRESHOLD}s threshold)"

        kill_stale_process
        sleep 2
        start_master_loop
        exit $?
    fi

    log "Master loop is healthy (heartbeat age: $(get_heartbeat_age)s)"
    exit 0
}

# Ensure logs directory exists
mkdir -p "$AI_SERVICE_DIR/logs"

# Run main
main "$@"
