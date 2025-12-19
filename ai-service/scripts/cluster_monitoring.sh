#!/bin/bash
# DEPRECATED: Use cluster_monitor_unified.sh instead (December 2025)
# Migration: ./scripts/cluster_monitor_unified.sh --p2p-api http://localhost:8770
echo "WARNING: This script is deprecated. Use cluster_monitor_unified.sh instead."
echo "For equivalent functionality: ./scripts/cluster_monitor_unified.sh"
echo ""

# Cluster Monitoring Script - Run every 2 minutes for 10 hours
# Monitors: P2P cluster health, work queue, node utilization, training progress

# Configuration
LEADER_API="http://100.78.101.123:8770"  # P2P orchestrator
FASTAPI_URL="http://100.78.101.123:8000"
ADMIN_KEY="ringrift-admin-2024-secret"
SSH_OPTS="-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -i ~/.ssh/id_cluster"

LOG_FILE="logs/cluster_monitor_$(date +%Y%m%d_%H%M%S).log"
INTERVAL_SECONDS=120  # 2 minutes
DURATION_HOURS=10
MAX_ITERATIONS=$((DURATION_HOURS * 60 * 60 / INTERVAL_SECONDS))  # 300 iterations

# Sample nodes for GPU checks
GPU_NODES=(
    "lambda-2xh100|ubuntu@192.222.53.22"
    "lambda-gh200-c|ubuntu@192.222.51.162"
    "lambda-gh200-a|ubuntu@192.222.51.29"
)

mkdir -p logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

check_p2p_cluster() {
    log "--- P2P Cluster Status ---"

    # Check orchestrator health - parse JSON response
    health=$(curl -s --max-time 10 "$LEADER_API/health" 2>/dev/null || echo '{}')

    # Extract key metrics from health response
    is_healthy=$(echo "$health" | grep -o '"healthy":[^,}]*' | cut -d: -f2 | tr -d ' ')
    node_id=$(echo "$health" | grep -o '"node_id":"[^"]*"' | cut -d'"' -f4)
    role=$(echo "$health" | grep -o '"role":"[^"]*"' | cut -d'"' -f4)
    active_peers=$(echo "$health" | grep -o '"active_peers":[0-9]*' | cut -d: -f2)
    gpu_util=$(echo "$health" | grep -o '"gpu_util":[0-9.]*' | cut -d: -f2)
    leader_id=$(echo "$health" | grep -o '"leader_id":"[^"]*"' | cut -d'"' -f4)

    if [ "$is_healthy" = "true" ]; then
        log "  [OK] P2P Orchestrator: HEALTHY ($node_id)"
    else
        log "  [WARN] P2P Orchestrator: $node_id (healthy=$is_healthy)"
    fi

    # Peer count from health endpoint
    if [ -n "$active_peers" ] && [ "$active_peers" -ge 25 ]; then
        log "  [OK] Active Peers: $active_peers"
    elif [ -n "$active_peers" ] && [ "$active_peers" -ge 15 ]; then
        log "  [WARN] Active Peers: $active_peers (below target of 30)"
    else
        log "  [ERR] Active Peers: ${active_peers:-0} (critical)"
    fi

    log "  [INFO] Role: $role, Leader: $leader_id"
    log "  [INFO] Cluster GPU Util: ${gpu_util:-N/A}%"
}

check_work_queue() {
    log "--- Work Queue Status ---"

    # Get health which includes job counts
    health=$(curl -s --max-time 10 "$LEADER_API/health" 2>/dev/null || echo '{}')

    selfplay_jobs=$(echo "$health" | grep -o '"selfplay_jobs":[0-9]*' | cut -d: -f2 || echo "0")
    training_jobs=$(echo "$health" | grep -o '"training_jobs":[0-9]*' | cut -d: -f2 || echo "0")
    selfplay_rate=$(echo "$health" | grep -o '"selfplay_rate":[0-9]*' | cut -d: -f2 || echo "0")

    log "  [INFO] Selfplay jobs: ${selfplay_jobs:-0}, Training jobs: ${training_jobs:-0}"
    log "  [INFO] Selfplay rate: ${selfplay_rate:-0} games/day"
}

check_gpu_utilization() {
    log "--- GPU Utilization (sample nodes) ---"

    for node_info in "${GPU_NODES[@]}"; do
        IFS='|' read -r name host <<< "$node_info"
        util=$(ssh $SSH_OPTS "$host" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null || echo "N/A")

        if [ "$util" = "N/A" ]; then
            log "  [$name] GPU: N/A (unreachable)"
        elif [ "$util" -gt 50 ]; then
            log "  [$name] GPU: ${util}% (active)"
        elif [ "$util" -gt 0 ]; then
            log "  [$name] GPU: ${util}% (low)"
        else
            log "  [$name] GPU: ${util}% (idle)"
        fi
    done
}

check_elo_progress() {
    log "--- Training Progress (Elo Velocity) ---"

    velocity=$(curl -s --max-time 15 "$FASTAPI_URL/admin/velocity" -H "X-Admin-Key: $ADMIN_KEY" 2>/dev/null || echo '{}')

    if echo "$velocity" | grep -q '"configs"'; then
        configs_met=$(echo "$velocity" | grep -o '"configs_met":[0-9]*' | cut -d: -f2 || echo "0")
        configs_unmet=$(echo "$velocity" | grep -o '"configs_unmet":[0-9]*' | cut -d: -f2 || echo "0")
        top_elo=$(echo "$velocity" | grep -o '"current_elo":[0-9.]*' | head -1 | cut -d: -f2 || echo "0")

        log "  [INFO] Configs at target: $configs_met / $((configs_met + configs_unmet))"
        log "  [INFO] Best Elo: $top_elo (target: 2000)"
    else
        log "  [WARN] Could not retrieve velocity data"
    fi
}

# Main monitoring loop
log "Starting cluster monitoring for $DURATION_HOURS hours (every $((INTERVAL_SECONDS/60)) minutes)"
log "Log file: $LOG_FILE"
log "========================================"

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    log ""
    log "=== CHECK $((iteration+1))/$MAX_ITERATIONS ==="

    check_p2p_cluster
    check_work_queue
    check_gpu_utilization
    check_elo_progress

    iteration=$((iteration + 1))
    if [ $iteration -lt $MAX_ITERATIONS ]; then
        log ""
        log "Next check in $((INTERVAL_SECONDS/60)) minutes..."
        sleep $INTERVAL_SECONDS
    fi
done

log ""
log "========================================"
log "Monitoring completed after $DURATION_HOURS hours"
