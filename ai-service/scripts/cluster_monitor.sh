#!/bin/bash
# DEPRECATED: Use cluster_monitor_unified.sh instead (December 2025)
# Migration: ./scripts/cluster_monitor_unified.sh --duration 10
echo "WARNING: This script is deprecated. Use cluster_monitor_unified.sh instead."
echo "For equivalent functionality: ./scripts/cluster_monitor_unified.sh --duration 10"
echo ""

# Cluster Monitor - runs every 2 minutes for 10 hours
# Checks: node connectivity, GPU utilization, training loop, benchmarks

LOG_FILE="/tmp/cluster_monitor.log"
ALERT_FILE="/tmp/cluster_alerts.log"
START_TIME=$(date +%s)
DURATION_HOURS=10
DURATION_SECONDS=$((DURATION_HOURS * 3600))
CHECK_INTERVAL=120  # 2 minutes

# Node definitions
LAMBDA_NODES=(
    "ubuntu@192.222.51.29:GH200-A"
    "ubuntu@192.222.51.162:GH200-C"
    "ubuntu@192.222.58.122:GH200-D"
    "ubuntu@192.222.57.162:GH200-E"
    "ubuntu@192.222.57.178:GH200-F"
    "ubuntu@192.222.57.79:GH200-G"
    "ubuntu@192.222.56.123:GH200-H"
    "ubuntu@192.222.50.112:GH200-I"
    "ubuntu@192.222.51.150:GH200-K"
    "ubuntu@192.222.51.233:GH200-L"
    "ubuntu@192.222.50.219:GH200-M"
    "ubuntu@192.222.51.204:GH200-N"
    "ubuntu@192.222.51.161:GH200-B"
    "ubuntu@192.222.51.92:GH200-O"
    "ubuntu@192.222.51.215:GH200-P"
)

VAST_NODES=(
    "root@ssh3.vast.ai|19940|Vast-4080S"
    "root@ssh5.vast.ai|18168|Vast-8x5090"
    "root@ssh1.vast.ai|15166|Vast-5090"
    "root@ssh8.vast.ai|38742|Vast-A40"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $1" | tee -a "$ALERT_FILE" | tee -a "$LOG_FILE"
}

check_lambda_node() {
    local conn="$1"
    local name="$2"
    local user_host="${conn}"

    # Check connectivity and get status
    result=$(ssh -i ~/.ssh/id_cluster -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$user_host" \
        "echo 'ONLINE'; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1; ps aux | grep -E 'selfplay|train|benchmark' | grep -v grep | wc -l" 2>/dev/null)

    if [ -z "$result" ]; then
        alert "$name: OFFLINE/UNREACHABLE"
        return 1
    fi

    online=$(echo "$result" | head -1)
    gpu_info=$(echo "$result" | sed -n '2p')
    proc_count=$(echo "$result" | tail -1)

    if [ "$online" = "ONLINE" ]; then
        if [ -n "$gpu_info" ]; then
            gpu_util=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
            mem_used=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' ')
            mem_total=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
            log "$name: OK | GPU: ${gpu_util}% | VRAM: ${mem_used}/${mem_total}MB | Procs: $proc_count"

            # Alert if GPU idle but should be working
            if [ "$gpu_util" -lt 10 ] && [ "$proc_count" -eq 0 ]; then
                alert "$name: GPU idle and no active processes"
            fi
        else
            log "$name: OK (no GPU info) | Procs: $proc_count"
        fi
    fi
    return 0
}

check_vast_node() {
    local conn="$1"
    local port="$2"
    local name="$3"

    result=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -p "$port" "$conn" \
        "echo 'ONLINE'; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1; ps aux | grep -E 'selfplay|train|benchmark' | grep -v grep | wc -l" 2>/dev/null)

    if [ -z "$result" ]; then
        alert "$name: OFFLINE/UNREACHABLE"
        return 1
    fi

    online=$(echo "$result" | head -1)
    gpu_info=$(echo "$result" | sed -n '2p')
    proc_count=$(echo "$result" | tail -1)

    if [ "$online" = "ONLINE" ]; then
        if [ -n "$gpu_info" ]; then
            gpu_util=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
            log "$name: OK | GPU: ${gpu_util}% | Procs: $proc_count"
        else
            log "$name: OK | Procs: $proc_count"
        fi
    fi
    return 0
}

check_benchmarks() {
    log "--- Benchmark Status ---"

    # Check key benchmark nodes
    for node_info in "ubuntu@192.222.51.162:sq8_2p" "ubuntu@192.222.57.79:sq8_4p" "ubuntu@192.222.56.123:hex8_2p"; do
        host="${node_info%:*}"
        bench="${node_info#*:}"

        status=$(ssh -i ~/.ssh/id_cluster -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" \
            "if pgrep -f benchmark_search > /dev/null; then echo 'RUNNING'; tail -1 /tmp/benchmark_${bench}.log 2>/dev/null; else if [ -f /tmp/benchmark_${bench}.log ]; then echo 'COMPLETE'; grep 'RANKINGS' /tmp/benchmark_${bench}.log 2>/dev/null | head -1; else echo 'NOT_FOUND'; fi; fi" 2>/dev/null)

        if [ -n "$status" ]; then
            log "Benchmark $bench: $status"
        fi
    done
}

check_training_loop() {
    log "--- Training Loop Status ---"

    # Check if unified loop is running on any node
    for node in "${LAMBDA_NODES[@]}"; do
        conn="${node%:*}"
        name="${node#*:}"

        loop_status=$(ssh -i ~/.ssh/id_cluster -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$conn" \
            "pgrep -f 'unified_loop' > /dev/null && echo 'RUNNING' || echo 'NOT_RUNNING'" 2>/dev/null)

        if [ "$loop_status" = "RUNNING" ]; then
            log "Training loop active on $name"
            return 0
        fi
    done

    log "No active training loop detected"
}

# Main monitoring loop
log "=========================================="
log "Cluster Monitor Started"
log "Duration: $DURATION_HOURS hours"
log "Check interval: $CHECK_INTERVAL seconds"
log "=========================================="

iteration=0
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - START_TIME))

    if [ $elapsed -ge $DURATION_SECONDS ]; then
        log "Monitoring complete after $DURATION_HOURS hours"
        break
    fi

    remaining_hours=$(( (DURATION_SECONDS - elapsed) / 3600 ))
    remaining_mins=$(( ((DURATION_SECONDS - elapsed) % 3600) / 60 ))

    iteration=$((iteration + 1))
    log ""
    log "========== Check #$iteration (${remaining_hours}h ${remaining_mins}m remaining) =========="

    # Check Lambda nodes
    log "--- Lambda GH200 Nodes ---"
    online_count=0
    for node in "${LAMBDA_NODES[@]}"; do
        conn="${node%:*}"
        name="${node#*:}"
        if check_lambda_node "$conn" "$name"; then
            online_count=$((online_count + 1))
        fi
    done
    log "Lambda nodes online: $online_count/${#LAMBDA_NODES[@]}"

    # Check Vast nodes
    log "--- Vast.ai Nodes ---"
    vast_online=0
    for node in "${VAST_NODES[@]}"; do
        conn=$(echo "$node" | cut -d'|' -f1)
        port=$(echo "$node" | cut -d'|' -f2)
        name=$(echo "$node" | cut -d'|' -f3)
        if check_vast_node "$conn" "$port" "$name"; then
            vast_online=$((vast_online + 1))
        fi
    done
    log "Vast nodes online: $vast_online/${#VAST_NODES[@]}"

    # Check benchmarks
    check_benchmarks

    # Check training loop
    check_training_loop

    # Sleep until next check
    sleep $CHECK_INTERVAL
done
