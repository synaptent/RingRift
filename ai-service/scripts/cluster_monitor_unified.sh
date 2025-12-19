#!/bin/bash
#
# Unified Cluster Monitor (December 2025)
# Consolidates: cluster_health_monitor.sh, cluster_monitor.sh,
#               cluster_monitor_daemon.sh, cluster_monitoring.sh
#
# Usage:
#   ./cluster_monitor_unified.sh                    # Default: 10 hours, all checks
#   ./cluster_monitor_unified.sh --duration 2      # Run for 2 hours
#   ./cluster_monitor_unified.sh --interval 60     # Check every 60 seconds
#   ./cluster_monitor_unified.sh --no-recovery     # Disable auto-recovery
#   ./cluster_monitor_unified.sh --local-only      # Only local checks (no SSH)
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${AI_SERVICE_DIR}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/cluster_monitor_$(date +%Y%m%d_%H%M%S).log"
ALERT_FILE="${LOG_DIR}/cluster_alerts.log"

# Defaults
DURATION_HOURS=${DURATION_HOURS:-10}
CHECK_INTERVAL=${CHECK_INTERVAL:-120}  # 2 minutes
AUTO_RECOVERY=${AUTO_RECOVERY:-true}
LOCAL_ONLY=${LOCAL_ONLY:-false}
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_cluster}"
SSH_OPTS="-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"

# P2P Orchestrator API (optional)
P2P_API="${P2P_API:-http://localhost:8770}"
FASTAPI_URL="${FASTAPI_URL:-http://localhost:8000}"
ADMIN_KEY="${ADMIN_KEY:-ringrift-admin-2024-secret}"

# Thresholds
MIN_GPU_UTIL=10
MIN_ACTIVE_NODES=5
CRITICAL_DISK_PERCENT=90

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION_HOURS="$2"
            shift 2
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --no-recovery)
            AUTO_RECOVERY=false
            shift
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --p2p-api)
            P2P_API="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --duration N     Run for N hours (default: 10)"
            echo "  --interval S     Check every S seconds (default: 120)"
            echo "  --no-recovery    Disable auto-recovery of idle nodes"
            echo "  --local-only     Only run local checks (no SSH)"
            echo "  --p2p-api URL    P2P orchestrator API URL"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Node Definitions
# =============================================================================

# Lambda GH200 nodes (SSH access)
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

# Vast.ai nodes (SSH with port)
VAST_NODES=(
    "root@ssh3.vast.ai|19940|Vast-4080S"
    "root@ssh5.vast.ai|18168|Vast-8x5090"
    "root@ssh1.vast.ai|15166|Vast-5090"
    "root@ssh8.vast.ai|38742|Vast-A40"
)

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_ok() {
    log "[OK] $1"
}

log_warn() {
    log "[WARN] $1"
}

log_err() {
    log "[ERR] $1"
}

alert() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $1"
    echo "$msg" | tee -a "$ALERT_FILE" | tee -a "$LOG_FILE"
}

# =============================================================================
# Local Health Checks
# =============================================================================

check_local_services() {
    log "--- Local Services ---"

    # Check AI service
    if curl -sf --max-time 5 "http://localhost:8001/health" > /dev/null 2>&1; then
        log_ok "AI Service: healthy"
    else
        log_warn "AI Service: not responding"
    fi

    # Check P2P orchestrator
    if curl -sf --max-time 5 "$P2P_API/health" > /dev/null 2>&1; then
        log_ok "P2P Orchestrator: healthy"
    else
        log_warn "P2P Orchestrator: not responding"
    fi
}

check_local_resources() {
    log "--- Local Resources ---"

    # CPU load
    load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    log "Load Average: $load"

    # Disk usage
    if [ -d "$AI_SERVICE_DIR" ]; then
        disk_use=$(df -h "$AI_SERVICE_DIR" 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
        log "Disk Usage: ${disk_use}%"
        if [ -n "$disk_use" ] && [ "$disk_use" -ge "$CRITICAL_DISK_PERCENT" ]; then
            alert "Disk usage critical: ${disk_use}%"
        fi
    fi

    # GPU if available locally
    if command -v nvidia-smi &> /dev/null; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "Local GPU: ${gpu_util}% util, ${gpu_mem} MB"
    fi

    # Training processes
    training_procs=$(pgrep -f "train.py|generate_data.py|selfplay" 2>/dev/null | wc -l | tr -d ' ')
    log "Training Processes: $training_procs"
}

# =============================================================================
# P2P Cluster Checks (via API)
# =============================================================================

check_p2p_cluster() {
    log "--- P2P Cluster Status ---"

    health=$(curl -sf --max-time 10 "$P2P_API/health" 2>/dev/null || echo '{}')

    if [ "$health" = "{}" ]; then
        log_warn "Could not reach P2P orchestrator"
        return
    fi

    # Parse health response
    is_healthy=$(echo "$health" | grep -o '"healthy":[^,}]*' | cut -d: -f2 | tr -d ' ')
    node_id=$(echo "$health" | grep -o '"node_id":"[^"]*"' | cut -d'"' -f4)
    role=$(echo "$health" | grep -o '"role":"[^"]*"' | cut -d'"' -f4)
    active_peers=$(echo "$health" | grep -o '"active_peers":[0-9]*' | cut -d: -f2)
    gpu_util=$(echo "$health" | grep -o '"gpu_util":[0-9.]*' | cut -d: -f2)

    if [ "$is_healthy" = "true" ]; then
        log_ok "P2P Orchestrator: $node_id (role: $role)"
    else
        log_warn "P2P Orchestrator unhealthy: $node_id"
    fi

    if [ -n "$active_peers" ]; then
        if [ "$active_peers" -ge 20 ]; then
            log_ok "Active Peers: $active_peers"
        elif [ "$active_peers" -ge 10 ]; then
            log_warn "Active Peers: $active_peers (below target)"
        else
            alert "Active Peers critically low: $active_peers"
        fi
    fi

    [ -n "$gpu_util" ] && log "Cluster GPU Util: ${gpu_util}%"
}

check_work_queue() {
    log "--- Work Queue ---"

    health=$(curl -sf --max-time 10 "$P2P_API/health" 2>/dev/null || echo '{}')

    selfplay_jobs=$(echo "$health" | grep -o '"selfplay_jobs":[0-9]*' | cut -d: -f2)
    training_jobs=$(echo "$health" | grep -o '"training_jobs":[0-9]*' | cut -d: -f2)

    [ -n "$selfplay_jobs" ] && log "Selfplay jobs: $selfplay_jobs"
    [ -n "$training_jobs" ] && log "Training jobs: $training_jobs"
}

# =============================================================================
# Remote Node Checks (via SSH)
# =============================================================================

check_lambda_node() {
    local conn="$1"
    local name="$2"

    result=$(ssh $SSH_OPTS -i "$SSH_KEY" "$conn" \
        'echo "ONLINE"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; ps aux | grep -E "selfplay|train|benchmark" | grep -v grep | wc -l' 2>/dev/null)

    if [ -z "$result" ]; then
        echo "OFFLINE"
        return 1
    fi

    online=$(echo "$result" | head -1)
    gpu_info=$(echo "$result" | sed -n '2p')
    proc_count=$(echo "$result" | tail -1)

    if [ "$online" = "ONLINE" ] && [ -n "$gpu_info" ]; then
        gpu_util=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
        echo "OK|${gpu_util}|${proc_count}"
        return 0
    fi

    echo "OK|0|${proc_count}"
    return 0
}

check_vast_node() {
    local conn="$1"
    local port="$2"

    result=$(ssh $SSH_OPTS -p "$port" "$conn" \
        'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1' 2>/dev/null)

    if [ -n "$result" ]; then
        echo "OK|${result}|0"
        return 0
    fi
    echo "OFFLINE"
    return 1
}

check_remote_nodes() {
    log "--- Lambda GH200 Nodes ---"

    local online_count=0
    local idle_nodes=()

    for node in "${LAMBDA_NODES[@]}"; do
        conn="${node%:*}"
        name="${node#*:}"

        status=$(check_lambda_node "$conn" "$name")
        state=$(echo "$status" | cut -d'|' -f1)
        gpu=$(echo "$status" | cut -d'|' -f2)
        procs=$(echo "$status" | cut -d'|' -f3)

        if [ "$state" = "OK" ]; then
            online_count=$((online_count + 1))
            if [ -n "$gpu" ] && [ "$gpu" -lt "$MIN_GPU_UTIL" ] && [ "$procs" -eq 0 ]; then
                log_warn "$name: GPU ${gpu}% (idle)"
                idle_nodes+=("$conn")
            else
                log "$name: GPU ${gpu}%, Procs: $procs"
            fi
        elif [ "$state" = "OFFLINE" ]; then
            log_warn "$name: OFFLINE"
        fi
    done

    log "Lambda nodes online: $online_count/${#LAMBDA_NODES[@]}"

    if [ "$online_count" -lt "$MIN_ACTIVE_NODES" ]; then
        alert "Low Lambda node count: $online_count (min: $MIN_ACTIVE_NODES)"
    fi

    # Auto-recovery for idle nodes
    if [ "$AUTO_RECOVERY" = true ] && [ ${#idle_nodes[@]} -gt 0 ]; then
        for idle_conn in "${idle_nodes[@]}"; do
            log "Starting selfplay on idle node: $idle_conn"
            ssh $SSH_OPTS -i "$SSH_KEY" "$idle_conn" '
                cd ~/ringrift/ai-service 2>/dev/null || cd ~/ai-service 2>/dev/null
                if [ -f scripts/run_gpu_selfplay.py ]; then
                    nohup python scripts/run_gpu_selfplay.py --board square8 --num-players 2 --num-games 5000 > /tmp/selfplay_auto.log 2>&1 &
                fi
            ' 2>/dev/null || true
        done
    fi

    # Check Vast nodes
    log "--- Vast.ai Nodes ---"
    local vast_online=0

    for node in "${VAST_NODES[@]}"; do
        conn=$(echo "$node" | cut -d'|' -f1)
        port=$(echo "$node" | cut -d'|' -f2)
        name=$(echo "$node" | cut -d'|' -f3)

        status=$(check_vast_node "$conn" "$port")
        state=$(echo "$status" | cut -d'|' -f1)
        gpu=$(echo "$status" | cut -d'|' -f2)

        if [ "$state" = "OK" ]; then
            vast_online=$((vast_online + 1))
            log "$name: GPU ${gpu}%"
        else
            log_warn "$name: OFFLINE"
        fi
    done

    log "Vast nodes online: $vast_online/${#VAST_NODES[@]}"
}

# =============================================================================
# Training Progress
# =============================================================================

check_training_progress() {
    log "--- Training Progress ---"

    velocity=$(curl -sf --max-time 15 "$FASTAPI_URL/admin/velocity" -H "X-Admin-Key: $ADMIN_KEY" 2>/dev/null || echo '{}')

    if echo "$velocity" | grep -q '"configs"'; then
        configs_met=$(echo "$velocity" | grep -o '"configs_met":[0-9]*' | cut -d: -f2)
        top_elo=$(echo "$velocity" | grep -o '"current_elo":[0-9.]*' | head -1 | cut -d: -f2)

        [ -n "$configs_met" ] && log "Configs at target: $configs_met"
        [ -n "$top_elo" ] && log "Best Elo: $top_elo"
    fi
}

# =============================================================================
# Main Loop
# =============================================================================

START_TIME=$(date +%s)
DURATION_SECONDS=$((DURATION_HOURS * 3600))
END_TIME=$((START_TIME + DURATION_SECONDS))

log "========================================"
log "Unified Cluster Monitor"
log "========================================"
log "Duration: $DURATION_HOURS hours"
log "Interval: $CHECK_INTERVAL seconds"
log "Auto-recovery: $AUTO_RECOVERY"
log "Local-only: $LOCAL_ONLY"
log "Log: $LOG_FILE"
log "Alerts: $ALERT_FILE"
log "========================================"

iteration=0
while true; do
    current_time=$(date +%s)
    if [ $current_time -ge $END_TIME ]; then
        break
    fi

    remaining=$((END_TIME - current_time))
    hours_remaining=$((remaining / 3600))
    mins_remaining=$(((remaining % 3600) / 60))

    iteration=$((iteration + 1))
    log ""
    log "=== Check #$iteration | ${hours_remaining}h ${mins_remaining}m remaining ==="

    # Always run local checks
    check_local_services
    check_local_resources

    # P2P checks if available
    check_p2p_cluster
    check_work_queue

    # Remote SSH checks (unless local-only)
    if [ "$LOCAL_ONLY" != true ]; then
        check_remote_nodes
    fi

    # Training progress
    check_training_progress

    # Sleep until next check
    if [ $current_time -lt $((END_TIME - CHECK_INTERVAL)) ]; then
        log ""
        log "Next check in $((CHECK_INTERVAL / 60)) minutes..."
        sleep $CHECK_INTERVAL
    fi
done

log ""
log "========================================"
log "Monitoring completed after $DURATION_HOURS hours"
log "Total checks: $iteration"
log "Alerts logged to: $ALERT_FILE"
log "========================================"

# Summary
if [ -f "$ALERT_FILE" ]; then
    alert_count=$(wc -l < "$ALERT_FILE" 2>/dev/null | tr -d ' ')
    log "Total alerts during run: $alert_count"
fi
