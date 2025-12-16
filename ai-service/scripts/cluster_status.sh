#!/bin/bash
# Quick cluster status check

echo "=== CLUSTER STATUS $(date '+%Y-%m-%d %H:%M') ==="
echo ""

# GPU utilization
echo "GPU Utilization:"
for host in lambda-gh200-a lambda-gh200-b lambda-gh200-c lambda-gh200-d lambda-gh200-e lambda-gh200-g lambda-gh200-h lambda-gh200-i lambda-gh200-k lambda-gh200-l lambda-2xh100; do
    printf "  %-20s " "$host:"
    result=$(ssh -o ConnectTimeout=3 -o BatchMode=yes $host 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null' 2>/dev/null)
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "offline"
    fi
done

echo ""
echo "Training DB (gh200-a):"
ssh -o ConnectTimeout=5 lambda-gh200-a 'sqlite3 ~/ringrift/ai-service/data/games/merged_training.db "SELECT COUNT(*) || \" games\" FROM games" 2>/dev/null' 2>/dev/null || echo "  unavailable"

echo ""
echo "Active Processes:"
for host in lambda-gh200-a lambda-gh200-b lambda-gh200-c; do
    count=$(ssh -o ConnectTimeout=3 $host 'ps aux | grep -E "(train|selfplay)" | grep -v grep | wc -l' 2>/dev/null)
    echo "  $host: $count workers"
done
