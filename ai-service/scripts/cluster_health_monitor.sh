#!/bin/bash
# Cluster Health Monitor - runs for 10 hours, checking every 3-5 minutes
# Monitors Lambda nodes and ensures selfplay is running

LAMBDA_NODES=(
    "100.65.88.62"
    "100.79.109.120"
    "100.117.177.83"
    "100.99.27.56"
    "100.97.98.26"
    "100.66.65.33"
    "100.104.126.58"
    "100.83.234.82"
)

LOG_FILE="${HOME}/cluster_monitor.log"
DURATION_HOURS=10
INTERVAL_SECONDS=240  # 4 minutes

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_node_health() {
    local ip=$1
    # Check SSH connectivity (timeout 10s)
    if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes ubuntu@${ip} "echo OK" >/dev/null 2>&1; then
        echo "ok"
    else
        echo "unreachable"
    fi
}

check_gpu_status() {
    local ip=$1
    # Check CUDA availability
    timeout 30 ssh ubuntu@${ip} "cd ~/ringrift/ai-service && source venv/bin/activate && python -c 'import torch; print(\"CUDA:\", torch.cuda.is_available())'" 2>/dev/null | grep -q "CUDA: True"
    if [ $? -eq 0 ]; then
        echo "ok"
    else
        echo "no_cuda"
    fi
}

check_selfplay_running() {
    local ip=$1
    # Check if any selfplay process is running
    local count=$(timeout 10 ssh ubuntu@${ip} "pgrep -f 'run.*selfplay' | wc -l" 2>/dev/null)
    echo "${count:-0}"
}

start_selfplay() {
    local ip=$1
    local board_type=$2  # hexagonal or square19
    log "Starting ${board_type} selfplay on ${ip}"
    ssh ubuntu@${ip} "cd ~/ringrift/ai-service && source venv/bin/activate && nohup python scripts/run_random_selfplay.py --board ${board_type} --num-games 1000 --output-dir data/selfplay/${board_type}_$(date +%Y%m%d_%H%M%S) > /tmp/selfplay.log 2>&1 &" &
}

run_health_check() {
    log "=== Starting health check cycle ==="
    
    local healthy=0
    local unhealthy=0
    local selfplay_running=0
    
    for ip in "${LAMBDA_NODES[@]}"; do
        # Check connectivity
        status=$(check_node_health "$ip")
        
        if [ "$status" = "ok" ]; then
            ((healthy++))
            
            # Check selfplay
            sp_count=$(check_selfplay_running "$ip")
            if [ "$sp_count" -gt 0 ]; then
                ((selfplay_running++))
                log "  ${ip}: healthy, ${sp_count} selfplay processes"
            else
                log "  ${ip}: healthy, no selfplay - consider starting"
                # Auto-start selfplay on idle nodes (alternate between hex and square19)
                if [ $((RANDOM % 2)) -eq 0 ]; then
                    start_selfplay "$ip" "hexagonal"
                else
                    start_selfplay "$ip" "square19"
                fi
            fi
        else
            ((unhealthy++))
            log "  ${ip}: UNREACHABLE"
        fi
    done
    
    log "Summary: ${healthy}/${#LAMBDA_NODES[@]} healthy, ${selfplay_running} running selfplay"
    log "=== Health check complete ==="
}

# Main monitoring loop
log "Starting 10-hour cluster monitoring (interval: ${INTERVAL_SECONDS}s)"
END_TIME=$(($(date +%s) + DURATION_HOURS * 3600))

while [ $(date +%s) -lt $END_TIME ]; do
    run_health_check
    
    # Sleep with random jitter (3-5 minutes)
    SLEEP_TIME=$((180 + RANDOM % 120))
    log "Sleeping for ${SLEEP_TIME} seconds..."
    sleep $SLEEP_TIME
done

log "Monitoring complete after ${DURATION_HOURS} hours"
