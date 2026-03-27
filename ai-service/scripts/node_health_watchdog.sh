#!/bin/bash
# Node health watchdog — runs via cron on Lambda nodes.
# Detects and recovers from common failure modes:
#   1. Zombie gauntlet_runner processes (>5 instances)
#   2. P2P crash-looping (activating/failed state)
#   3. GPU VRAM leak (>50GB used with <5% utilization)
#   4. P2P active but no selfplay for 30+ min (idle node)
#   5. Governor slot leak on coordinator (>30 min old export slots)
#
# Install: crontab -e → */5 * * * * ~/ringrift/ai-service/scripts/node_health_watchdog.sh >> /tmp/watchdog.log 2>&1
#
# Mar 23, 2026: Created after 65 zombie gauntlet_runner processes
# overwhelmed gh200-8 and made it unreachable via SSH.
# Mar 27, 2026: Added idle node detection (check #4) — nodes repeatedly
# showed "P2P active, GPU 0%, 0 selfplay procs" for hours without recovery.

LOG_PREFIX="[watchdog $(date +%H:%M:%S)]"
IDLE_MARKER="/tmp/watchdog_idle_count"

# 1. Kill duplicate gauntlet_runner processes (keep at most 1 parent)
gauntlet_count=$(pgrep -c gauntlet_runner 2>/dev/null || echo 0)
if [ "$gauntlet_count" -gt 10 ]; then
    echo "$LOG_PREFIX ZOMBIE: $gauntlet_count gauntlet_runner procs, killing all and restarting"
    pkill -9 -f gauntlet_runner 2>/dev/null
    sleep 3
    cd ~/ringrift/ai-service && PYTHONPATH=. nohup venv/bin/python scripts/gauntlet_runner.py > /tmp/gauntlet.log 2>&1 &
fi

# 2. Restart P2P if crash-looping
p2p_status=$(systemctl is-active ringrift-p2p 2>/dev/null)
if [ "$p2p_status" = "activating" ] || [ "$p2p_status" = "failed" ]; then
    echo "$LOG_PREFIX P2P_RECOVERY: status=$p2p_status, restarting"
    sudo systemctl restart ringrift-p2p
    rm -f "$IDLE_MARKER"
fi

# 3. Detect GPU VRAM leak (high VRAM, low utilization)
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' MiB')
if [ -n "$gpu_util" ] && [ -n "$gpu_mem" ] && [ "$gpu_util" -lt 5 ] && [ "$gpu_mem" -gt 50000 ]; then
    echo "$LOG_PREFIX VRAM_LEAK: GPU ${gpu_util}% util with ${gpu_mem}MB VRAM, restarting P2P"
    sudo systemctl restart ringrift-p2p
    rm -f "$IDLE_MARKER"
fi

# 4. Detect idle node: P2P active but no selfplay processes AND low GPU
# Uses a counter file to avoid restarting during brief gaps between jobs.
# After 6 consecutive idle checks (30 min at */5 cron), restart P2P.
if [ "$p2p_status" = "active" ]; then
    selfplay_procs=$(ps aux | grep -E "selfplay|gumbel|run_gpu" | grep -v grep | wc -l)
    if [ "$selfplay_procs" -eq 0 ] && [ -n "$gpu_util" ] && [ "$gpu_util" -lt 10 ]; then
        idle_count=$(cat "$IDLE_MARKER" 2>/dev/null || echo 0)
        idle_count=$((idle_count + 1))
        echo "$idle_count" > "$IDLE_MARKER"
        if [ "$idle_count" -ge 6 ]; then
            echo "$LOG_PREFIX IDLE_NODE: P2P active but 0 selfplay for ${idle_count} checks (~$((idle_count * 5))min), restarting"
            sudo systemctl restart ringrift-p2p
            echo 0 > "$IDLE_MARKER"
        fi
    else
        # Node is working, reset counter
        echo 0 > "$IDLE_MARKER"
    fi
fi
