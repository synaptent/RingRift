#!/bin/bash
# Node health watchdog — runs via cron on Lambda nodes.
# Detects and recovers from common failure modes:
#   1. Zombie gauntlet_runner processes (>5 instances)
#   2. P2P crash-looping (activating state)
#   3. GPU VRAM leak (>90GB used with 0% utilization)
#
# Install: crontab -e → */5 * * * * /home/ubuntu/ringrift/ai-service/scripts/node_health_watchdog.sh >> /tmp/watchdog.log 2>&1
#
# Mar 23, 2026: Created after 65 zombie gauntlet_runner processes
# overwhelmed gh200-8 and made it unreachable via SSH.

LOG_PREFIX="[watchdog $(date +%H:%M:%S)]"

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
fi

# 3. Detect GPU VRAM leak (high VRAM, low utilization)
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' MiB')
if [ -n "$gpu_util" ] && [ -n "$gpu_mem" ] && [ "$gpu_util" -lt 5 ] && [ "$gpu_mem" -gt 50000 ]; then
    echo "$LOG_PREFIX VRAM_LEAK: GPU ${gpu_util}% util with ${gpu_mem}MB VRAM, restarting P2P"
    sudo systemctl restart ringrift-p2p
fi
