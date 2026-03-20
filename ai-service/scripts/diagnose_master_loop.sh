#!/bin/bash
# diagnose_master_loop.sh — Periodic CPU burn diagnostic for master_loop
#
# Runs every 5 minutes via cron. Captures a snapshot of master_loop state
# to help identify which daemon enters a tight async loop. After 8-12 hours,
# the log file shows a time-series that reveals exactly when CPU escalated
# and what changed (thread count, open files, child processes, asyncio tasks).
#
# Usage:
#   # Manual run
#   bash scripts/diagnose_master_loop.sh
#
#   # Cron (every 5 min)
#   */5 * * * * ~/Development/RingRift/ai-service/scripts/diagnose_master_loop.sh >> ~/Library/Logs/RingRift/master_loop_diag.log 2>&1
#
# Mar 20, 2026: Created to diagnose recurring master_loop CPU burn (98% CPU,
# escalates over 2-3 hours, no child processes — internal daemon tight loop).

set -uo pipefail
export PATH="/opt/homebrew/bin:$PATH"

ML_PID=$(pgrep -f "master_loop.py" | head -1)
if [ -z "$ML_PID" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) master_loop NOT RUNNING"
    exit 0
fi

# CPU, memory, thread count, open files
CPU=$(ps -o pcpu= -p "$ML_PID" 2>/dev/null | xargs)
MEM=$(ps -o pmem= -p "$ML_PID" 2>/dev/null | xargs)
RSS=$(ps -o rss= -p "$ML_PID" 2>/dev/null | xargs)
RSS_MB=$((${RSS:-0} / 1024))
THREADS=$(ps -M "$ML_PID" 2>/dev/null | wc -l | xargs)
OPEN_FILES=$(lsof -p "$ML_PID" 2>/dev/null | wc -l | xargs)
CHILDREN=$(pgrep -P "$ML_PID" 2>/dev/null | wc -l | xargs)
UPTIME=$(ps -o etime= -p "$ML_PID" 2>/dev/null | xargs)

# Child process details (if any)
CHILD_INFO=""
if [ "${CHILDREN:-0}" -gt 0 ]; then
    CHILD_INFO=$(pgrep -P "$ML_PID" 2>/dev/null | while read cpid; do
        ps -o pcpu=,command= -p "$cpid" 2>/dev/null | head -c 100
    done | tr '\n' '; ')
fi

# Health endpoint (daemon status)
DAEMON_STATUS=""
HEALTH=$(curl -s --max-time 5 http://localhost:8790/status 2>/dev/null)
if [ -n "$HEALTH" ]; then
    DAEMON_STATUS=$(echo "$HEALTH" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    daemons = d.get('daemons', d.get('daemon_status', {}))
    if isinstance(daemons, dict):
        # Find daemons with high error counts or abnormal state
        issues = []
        for name, info in daemons.items():
            if isinstance(info, dict):
                errors = info.get('consecutive_errors', info.get('errors', 0))
                state = info.get('state', info.get('status', ''))
                cycle_count = info.get('cycle_count', info.get('total_runs', 0))
                last_duration = info.get('last_cycle_duration', info.get('last_run_duration', 0))
                if errors and int(errors) > 0:
                    issues.append(f'{name}:err={errors}')
                if last_duration and float(last_duration) > 30:
                    issues.append(f'{name}:slow={last_duration:.0f}s')
        if issues:
            print(' '.join(issues[:10]))
        else:
            print('all_healthy')
    else:
        print('unknown_format')
except:
    print('parse_error')
" 2>/dev/null)
fi

# AsyncIO task snapshot (via /metrics if available)
ASYNCIO_TASKS=""
METRICS=$(curl -s --max-time 3 http://localhost:8790/metrics 2>/dev/null)
if [ -n "$METRICS" ]; then
    ASYNCIO_TASKS=$(echo "$METRICS" | grep -i "asyncio_tasks\|active_tasks\|pending_tasks" | head -3 | tr '\n' ' ')
fi

# Log the snapshot
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) cpu=${CPU}% mem=${MEM}% rss=${RSS_MB}MB threads=${THREADS} files=${OPEN_FILES} children=${CHILDREN} uptime=${UPTIME} daemons=[${DAEMON_STATUS:-no_data}] asyncio=[${ASYNCIO_TASKS:-no_data}] children_detail=[${CHILD_INFO:-none}]"

# Alert if CPU is high
if [ -n "$CPU" ]; then
    HIGH=$(echo "$CPU > 50" | bc 2>/dev/null || echo 0)
    if [ "$HIGH" = "1" ]; then
        echo "  WARNING: CPU=${CPU}% — capturing detailed snapshot"

        # Python-level asyncio task dump
        # This sends a signal to master_loop to dump its asyncio tasks
        # (only works if master_loop has a SIGUSR2 handler, otherwise harmless)

        # Thread-level breakdown
        echo "  THREADS:"
        ps -M "$ML_PID" 2>/dev/null | head -20

        # Top system calls (if dtrace available)
        # sudo dtruss -p "$ML_PID" -c 2>/dev/null | head -10

        # Recent master_loop log (what was it doing?)
        echo "  RECENT_LOG:"
        tail -5 ~/Library/Logs/RingRift/master_loop.log 2>/dev/null
    fi
fi
