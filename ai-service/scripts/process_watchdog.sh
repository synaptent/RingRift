#!/bin/bash
# process_watchdog.sh - Monitor and limit selfplay process counts
#
# Prevents runaway process accumulation that can overload nodes.
# Run via cron: */5 * * * * /path/to/process_watchdog.sh >> /var/log/process_watchdog.log 2>&1
#
# Usage:
#   ./scripts/process_watchdog.sh [OPTIONS]
#
# Options:
#   --max-python MAX   Maximum Python processes before alert (default: 50)
#   --max-load MAX     Maximum load average before action (default: 100)
#   --kill             Kill excess processes (default: alert only)
#   --slack URL        Send alerts to Slack webhook
#   -h, --help         Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Defaults
MAX_PYTHON_PROCS=50
MAX_LOAD=100
KILL_MODE=false
SLACK_WEBHOOK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-python) MAX_PYTHON_PROCS="$2"; shift 2 ;;
        --max-load) MAX_LOAD="$2"; shift 2 ;;
        --kill) KILL_MODE=true; shift ;;
        --slack) SLACK_WEBHOOK="$2"; shift 2 ;;
        --help|-h)
            head -20 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get current metrics
PYTHON_COUNT=$(pgrep -c python 2>/dev/null || echo "0")
LOAD_1M=$(awk '{print $1}' /proc/loadavg 2>/dev/null || uptime | awk -F'[, ]+' '{print $(NF-2)}')

# Convert load to integer for comparison
LOAD_INT=${LOAD_1M%.*}

send_alert() {
    local message="$1"
    local severity="$2"

    echo "[$TIMESTAMP] [$severity] $message"

    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="warning"
        [[ "$severity" == "CRITICAL" ]] && color="danger"
        [[ "$severity" == "OK" ]] && color="good"

        curl -s -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Process Watchdog: $severity\",
                    \"text\": \"$message\",
                    \"footer\": \"$HOSTNAME | $TIMESTAMP\"
                }]
            }" >/dev/null 2>&1 || true
    fi
}

# Check for problems
PROBLEMS=()

if [[ "$PYTHON_COUNT" -gt "$MAX_PYTHON_PROCS" ]]; then
    PROBLEMS+=("Python processes: $PYTHON_COUNT (max: $MAX_PYTHON_PROCS)")
fi

if [[ "$LOAD_INT" -gt "$MAX_LOAD" ]]; then
    PROBLEMS+=("Load average: $LOAD_1M (max: $MAX_LOAD)")
fi

# Report status
if [[ ${#PROBLEMS[@]} -eq 0 ]]; then
    echo "[$TIMESTAMP] [OK] Host: $HOSTNAME | Python: $PYTHON_COUNT | Load: $LOAD_1M"
    exit 0
fi

# Problems detected
PROBLEM_MSG="${PROBLEMS[*]}"
send_alert "$HOSTNAME: $PROBLEM_MSG" "WARNING"

# Kill mode - try to recover
if [[ "$KILL_MODE" == "true" ]]; then
    echo "[$TIMESTAMP] [ACTION] Attempting to kill excess processes..."

    # Kill selfplay processes first (most likely culprit)
    SELFPLAY_PIDS=$(pgrep -f "run_self_play" 2>/dev/null || true)
    if [[ -n "$SELFPLAY_PIDS" ]]; then
        SELFPLAY_COUNT=$(echo "$SELFPLAY_PIDS" | wc -l)
        if [[ "$SELFPLAY_COUNT" -gt 10 ]]; then
            echo "[$TIMESTAMP] [ACTION] Killing $SELFPLAY_COUNT selfplay processes..."
            echo "$SELFPLAY_PIDS" | xargs kill -9 2>/dev/null || true
            send_alert "$HOSTNAME: Killed $SELFPLAY_COUNT selfplay processes" "ACTION"
        fi
    fi

    # If still overloaded, kill all Python processes
    sleep 5
    NEW_LOAD=$(awk '{print $1}' /proc/loadavg 2>/dev/null || uptime | awk -F'[, ]+' '{print $(NF-2)}')
    NEW_LOAD_INT=${NEW_LOAD%.*}

    if [[ "$NEW_LOAD_INT" -gt "$MAX_LOAD" ]]; then
        echo "[$TIMESTAMP] [CRITICAL] Load still high ($NEW_LOAD), killing all Python..."
        pkill -9 python 2>/dev/null || true
        send_alert "$HOSTNAME: CRITICAL - Killed all Python processes (load: $NEW_LOAD)" "CRITICAL"
    fi
fi

# Exit with error code to indicate problems were found
exit 1
