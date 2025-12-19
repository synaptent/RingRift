#!/bin/bash
# DEPRECATED: Use cluster_monitor_unified.sh instead (December 2025)
# Migration: ./scripts/cluster_monitor_unified.sh --duration 10
echo "WARNING: This script is deprecated. Use cluster_monitor_unified.sh instead."
echo "For equivalent functionality: ./scripts/cluster_monitor_unified.sh --duration 10"
echo ""

#
# Cluster Monitor Daemon
# Monitors cluster health every 2 minutes for 10 hours
# Alerts on issues and attempts auto-recovery
#

LOG_DIR="/Users/armand/Development/RingRift/ai-service/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cluster_monitor_$(date +%Y%m%d_%H%M%S).log"
ALERT_LOG="$LOG_DIR/cluster_alerts.log"

# Configuration
MONITOR_INTERVAL=120  # 2 minutes
TOTAL_DURATION=$((10 * 60 * 60))  # 10 hours in seconds
START_TIME=$(date +%s)
END_TIME=$((START_TIME + TOTAL_DURATION))

# Thresholds
MIN_GPU_UTIL=10       # Minimum GPU utilization %
MIN_ACTIVE_NODES=10   # Minimum number of active nodes
IDLE_THRESHOLD=300    # Seconds before considering node idle

# Node lists
LAMBDA_NODES="100.83.234.82 100.88.176.74 100.104.165.116 100.104.126.58 100.65.88.62 100.99.27.56 100.96.142.42 100.76.145.60 100.97.104.89"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $1" | tee -a "$LOG_FILE" >> "$ALERT_LOG"
}

check_lambda_nodes() {
    local online=0
    local total=0
    local low_util=""
    
    for ip in $LAMBDA_NODES; do
        ((total++))
        result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "ubuntu@$ip" '
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            procs=$(ps aux | grep -E "(selfplay|train)" | grep -v grep | wc -l)
            echo "$gpu_util|$mem_used|$procs"
        ' 2>/dev/null)
        
        if [ -n "$result" ]; then
            ((online++))
            gpu_util=$(echo "$result" | cut -d'|' -f1)
            procs=$(echo "$result" | cut -d'|' -f3)
            
            if [ -n "$gpu_util" ] && [ "$gpu_util" -lt "$MIN_GPU_UTIL" ] && [ "$procs" -eq 0 ]; then
                low_util="$low_util $ip"
            fi
        fi
    done
    
    echo "$online|$total|$low_util"
}

check_vast_nodes() {
    local online=0
    local total=0
    
    # Check running Vast instances via vastai CLI
    vast_output=$(vastai show instances 2>/dev/null | grep "running" | wc -l)
    echo "$vast_output"
}

check_training_jobs() {
    # Check if any training jobs are running on the cluster
    for ip in $LAMBDA_NODES; do
        result=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "ubuntu@$ip" '
            ps aux | grep -E "train_nnue|run_training" | grep -v grep | head -1
        ' 2>/dev/null)
        if [ -n "$result" ]; then
            echo "Training active on $ip"
            return 0
        fi
    done
    echo "No training jobs found"
    return 1
}

check_selfplay_jobs() {
    local count=0
    for ip in $LAMBDA_NODES; do
        result=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "ubuntu@$ip" '
            ps aux | grep -E "selfplay|gpu_selfplay" | grep -v grep | wc -l
        ' 2>/dev/null)
        if [ -n "$result" ]; then
            count=$((count + result))
        fi
    done
    echo "$count"
}

start_selfplay_on_idle() {
    local ip=$1
    log "Starting selfplay on idle node $ip..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$ip" '
        cd ~/ringrift/ai-service
        nohup python scripts/run_gpu_selfplay.py --board hex --num-players 2 --num-games 10000 --engine-mode nnue-guided > /tmp/selfplay_auto.log 2>&1 &
        echo "Started selfplay PID: $!"
    ' 2>/dev/null
}

# Main monitoring loop
log "=========================================="
log "Cluster Monitor Started"
log "Duration: 10 hours"
log "Interval: 2 minutes"
log "End time: $(date -r $END_TIME '+%Y-%m-%d %H:%M:%S')"
log "=========================================="

iteration=0
while [ $(date +%s) -lt $END_TIME ]; do
    ((iteration++))
    current_time=$(date +%s)
    elapsed=$((current_time - START_TIME))
    remaining=$((END_TIME - current_time))
    hours_remaining=$((remaining / 3600))
    mins_remaining=$(((remaining % 3600) / 60))
    
    log ""
    log "=== Iteration $iteration | Remaining: ${hours_remaining}h ${mins_remaining}m ==="
    
    # Check Lambda nodes
    lambda_status=$(check_lambda_nodes)
    lambda_online=$(echo "$lambda_status" | cut -d'|' -f1)
    lambda_total=$(echo "$lambda_status" | cut -d'|' -f2)
    lambda_idle=$(echo "$lambda_status" | cut -d'|' -f3)
    
    log "Lambda Nodes: $lambda_online/$lambda_total online"
    
    if [ "$lambda_online" -lt "$MIN_ACTIVE_NODES" ]; then
        alert "Low Lambda node count: $lambda_online (min: $MIN_ACTIVE_NODES)"
    fi
    
    # Check for idle nodes and start selfplay
    if [ -n "$lambda_idle" ]; then
        for idle_ip in $lambda_idle; do
            alert "Node $idle_ip is idle (GPU < ${MIN_GPU_UTIL}%)"
            start_selfplay_on_idle "$idle_ip"
        done
    fi
    
    # Check Vast nodes
    vast_running=$(check_vast_nodes)
    log "Vast.ai Nodes: $vast_running running"
    
    # Check selfplay jobs
    selfplay_count=$(check_selfplay_jobs)
    log "Selfplay processes: $selfplay_count"
    
    if [ "$selfplay_count" -eq 0 ]; then
        alert "No selfplay processes running!"
    fi
    
    # Check training jobs
    training_status=$(check_training_jobs)
    log "Training: $training_status"
    
    # Get GPU utilization summary
    log "GPU Utilization:"
    for ip in $LAMBDA_NODES; do
        util=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "ubuntu@$ip" 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1' 2>/dev/null)
        if [ -n "$util" ]; then
            host=$(tailscale status 2>/dev/null | grep "$ip" | awk '{print $2}' | head -1)
            log "  $host ($ip): $util"
        fi
    done
    
    # Sleep until next iteration
    log "Next check in $MONITOR_INTERVAL seconds..."
    sleep $MONITOR_INTERVAL
done

log ""
log "=========================================="
log "Cluster Monitor Completed"
log "Total iterations: $iteration"
log "=========================================="
