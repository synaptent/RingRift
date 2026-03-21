#!/bin/bash
# watchdog_master_loop.sh - Auto-restart master_loop when CPU > 90% for 10+ min
# Runs every 5 min via cron. Checks CPU, restarts if persistently high.
# Mar 20, 2026: Created because multiple daemon paths can cause CPU burn.
export PATH="/opt/homebrew/bin:$PATH"

STATE_FILE="/tmp/ringrift-ml-watchdog.state"
ML_PID=$(pgrep -f "master_loop.py" | head -1)

if [ -z "$ML_PID" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) master_loop not running"
    rm -f "$STATE_FILE"
    exit 0
fi

CPU=$(ps -o pcpu= -p "$ML_PID" 2>/dev/null | xargs)
if [ -z "$CPU" ]; then exit 0; fi

HIGH=$(echo "$CPU > 90" | bc 2>/dev/null || echo 0)

if [ "$HIGH" = "1" ]; then
    # CPU is high - check if it's been high for 2 consecutive checks (10 min)
    if [ -f "$STATE_FILE" ]; then
        PREV_TIME=$(cat "$STATE_FILE")
        NOW=$(date +%s)
        DIFF=$((NOW - PREV_TIME))
        if [ "$DIFF" -gt 300 ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WATCHDOG: master_loop at ${CPU}% CPU for ${DIFF}s - RESTARTING"
            launchctl kickstart -k gui/$(id -u)/com.ringrift.master-loop
            rm -f "$STATE_FILE"
            exit 0
        fi
    else
        # First high reading - record timestamp
        date +%s > "$STATE_FILE"
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WATCHDOG: master_loop at ${CPU}% CPU - monitoring"
    fi
else
    # CPU is normal - clear state
    rm -f "$STATE_FILE"
fi
