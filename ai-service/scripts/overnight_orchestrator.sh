#!/bin/bash
# Overnight Orchestrator - Keeps AI training pipeline running across all hosts
# Run with: nohup ./scripts/overnight_orchestrator.sh >> logs/overnight/run.log 2>&1 &
#
# CONFIGURATION:
#   Copy config/orchestrator_hosts.example.sh to config/orchestrator_hosts.sh
#   and fill in your host details. The config file is gitignored.

set -uo pipefail  # Removed -e to prevent exit on SSH failures
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/overnight"
LOCKFILE="/tmp/overnight_orchestrator.lock"
CONFIG_FILE="$PROJECT_DIR/config/orchestrator_hosts.sh"

# ============================================
# Load Host Configuration
# ============================================

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Copy config/orchestrator_hosts.example.sh to config/orchestrator_hosts.sh"
    echo "and fill in your host details."
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

# Build connection strings from config (user@host format)
LAMBDA_H100="${LAMBDA_H100_USER:-ubuntu}@${LAMBDA_H100_HOST:-disabled}"
LAMBDA_A10="${LAMBDA_A10_USER:-ubuntu}@${LAMBDA_A10_HOST:-disabled}"
AWS_STAGING="${AWS_STAGING_USER:-ubuntu}@${AWS_STAGING_HOST:-disabled}"
VAST_3090="${VAST_3090_USER:-root}@${VAST_3090_HOST:-disabled}"
VAST_5090_DUAL="${VAST_5090_DUAL_USER:-root}@${VAST_5090_DUAL_HOST:-disabled}"
VAST_5090_QUAD="${VAST_5090_QUAD_USER:-root}@${VAST_5090_QUAD_HOST:-disabled}"
MAC_STUDIO="${MAC_STUDIO_USER:-}@${MAC_STUDIO_HOST:-disabled}"
MBP_16GB="${MBP_16GB_USER:-}@${MBP_16GB_HOST:-disabled}"
MBP_64GB="${MBP_64GB_USER:-}@${MBP_64GB_HOST:-disabled}"

# Set defaults for optional config values
LAMBDA_H100_MIN_JOBS="${LAMBDA_H100_MIN_JOBS:-3}"
LAMBDA_A10_MIN_JOBS="${LAMBDA_A10_MIN_JOBS:-2}"
AWS_STAGING_MIN_JOBS="${AWS_STAGING_MIN_JOBS:-2}"
AWS_STAGING_KEY="${AWS_STAGING_KEY:-}"
VAST_3090_PORT="${VAST_3090_PORT:-22}"
VAST_3090_MIN_JOBS="${VAST_3090_MIN_JOBS:-2}"
VAST_5090_DUAL_PORT="${VAST_5090_DUAL_PORT:-22}"
VAST_5090_DUAL_MIN_JOBS="${VAST_5090_DUAL_MIN_JOBS:-3}"
VAST_5090_QUAD_PORT="${VAST_5090_QUAD_PORT:-22}"
VAST_5090_QUAD_MIN_JOBS="${VAST_5090_QUAD_MIN_JOBS:-4}"
MAC_STUDIO_MIN_JOBS="${MAC_STUDIO_MIN_JOBS:-2}"
MAC_STUDIO_KEY="${MAC_STUDIO_KEY:-}"
MBP_16GB_MIN_JOBS="${MBP_16GB_MIN_JOBS:-1}"
MBP_64GB_MIN_JOBS="${MBP_64GB_MIN_JOBS:-2}"

# ============================================
# Timing Configuration
# ============================================
CHECK_INTERVAL=300      # 5 minutes normal interval
RETRY_INTERVAL=60       # Retry failed hosts after 60 seconds
SSH_TIMEOUT=15          # Timeout for SSH commands
MAX_RETRIES=3           # Max consecutive failures before backing off

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >> "$LOG_DIR/orchestrator.log"
    echo "$msg"  # Also to stdout for nohup capture
}

# Prevent multiple instances
if [ -f "$LOCKFILE" ]; then
    if kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
        log "Orchestrator already running (PID $(cat $LOCKFILE)), exiting"
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# ============================================
# SSH Helper Functions
# ============================================

# Standard SSH (no custom port/key)
check_ssh() {
    local host=$1
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "echo ok" >/dev/null 2>&1
}

run_ssh() {
    local host=$1
    local cmd=$2
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "$cmd" 2>/dev/null
}

# SSH with custom port (for Vast.ai)
check_ssh_port() {
    local host=$1
    local port=$2
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "$host" "echo ok" >/dev/null 2>&1
}

run_ssh_port() {
    local host=$1
    local port=$2
    local cmd=$3
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "$host" "$cmd" 2>/dev/null
}

# SSH with custom key (for AWS)
check_ssh_key() {
    local host=$1
    local key=$2
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$key" "$host" "echo ok" >/dev/null 2>&1
}

run_ssh_key() {
    local host=$1
    local key=$2
    local cmd=$3
    timeout $SSH_TIMEOUT ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$key" "$host" "$cmd" 2>/dev/null
}

# ============================================
# Job Counting Functions
# ============================================

count_python_jobs() {
    local host=$1
    local result
    result=$(run_ssh "$host" "ps aux | grep -E 'python.*selfplay|python.*train|python.*cmaes' | grep -v grep | wc -l")
    echo "${result:-0}"
}

count_python_jobs_port() {
    local host=$1
    local port=$2
    local result
    result=$(run_ssh_port "$host" "$port" "ps aux | grep -E 'python.*selfplay|python.*train|python.*cmaes' | grep -v grep | wc -l")
    echo "${result:-0}"
}

count_python_jobs_key() {
    local host=$1
    local key=$2
    local result
    result=$(run_ssh_key "$host" "$key" "ps aux | grep -E 'python.*selfplay|python.*train|python.*cmaes' | grep -v grep | wc -l")
    echo "${result:-0}"
}

# ============================================
# Selfplay Restart Functions
# ============================================

restart_selfplay_linux() {
    local host=$1
    local host_name=$2
    local min_jobs=$3
    local path=$4
    local ssh_func=$5  # "run_ssh", "run_ssh_port:PORT", or "run_ssh_key:KEY"

    local current_jobs
    if [[ "$ssh_func" == run_ssh_port:* ]]; then
        local port="${ssh_func#run_ssh_port:}"
        current_jobs=$(count_python_jobs_port "$host" "$port")
    elif [[ "$ssh_func" == run_ssh_key:* ]]; then
        local key="${ssh_func#run_ssh_key:}"
        current_jobs=$(count_python_jobs_key "$host" "$key")
    else
        current_jobs=$(count_python_jobs "$host")
    fi

    log "$host_name: $current_jobs Python jobs running (min: $min_jobs)"

    if [ "$current_jobs" -lt "$min_jobs" ]; then
        log "$host_name: Restarting selfplay jobs..."

        local seed_base=$((RANDOM * 1000 + $(date +%s) % 10000))
        local cmd="cd $path && \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 1000 --board-type square8 --num-players 2 --output-dir data/selfplay/sq8_2p --seed $seed_base > /tmp/selfplay_sq8_2p.log 2>&1 & \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 500 --board-type square8 --num-players 3 --output-dir data/selfplay/sq8_3p --seed $((seed_base+1)) > /tmp/selfplay_sq8_3p.log 2>&1 & \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 500 --board-type hex --num-players 2 --output-dir data/selfplay/hex_2p --seed $((seed_base+2)) > /tmp/selfplay_hex_2p.log 2>&1 & \
            echo 'Restarted selfplay jobs'"

        local result
        if [[ "$ssh_func" == run_ssh_port:* ]]; then
            local port="${ssh_func#run_ssh_port:}"
            result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "$host" "$cmd" 2>&1) || result="SSH timeout"
        elif [[ "$ssh_func" == run_ssh_key:* ]]; then
            local key="${ssh_func#run_ssh_key:}"
            result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$key" "$host" "$cmd" 2>&1) || result="SSH timeout"
        else
            result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "$cmd" 2>&1) || result="SSH timeout"
        fi
        log "$host_name: $result"
    fi
}

restart_selfplay_mac() {
    local host=$1
    local host_name=$2
    local min_jobs=$3
    local path=$4
    local key=${5:-}  # Optional SSH key

    local current_jobs
    if [ -n "$key" ]; then
        current_jobs=$(run_ssh_key "$host" "$key" "ps aux | grep -E 'python.*selfplay' | grep -v grep | wc -l") || current_jobs="0"
    else
        current_jobs=$(run_ssh "$host" "ps aux | grep -E 'python.*selfplay' | grep -v grep | wc -l") || current_jobs="0"
    fi
    current_jobs="${current_jobs:-0}"

    log "$host_name: $current_jobs selfplay jobs running (min: $min_jobs)"

    if [ "$current_jobs" -lt "$min_jobs" ]; then
        log "$host_name: Restarting selfplay jobs..."

        local seed_base=$((RANDOM * 1000 + $(date +%s) % 10000))
        # Mac uses different activation and path structure
        local result
        local ssh_opts="-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"
        if [ -n "$key" ]; then
            ssh_opts="$ssh_opts -i $key"
        fi
        result=$(timeout 45 ssh $ssh_opts "$host" "cd $path && \
            source venv/bin/activate && \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 500 --board-type square8 --num-players 2 --output-dir data/selfplay/sq8_2p --seed $seed_base > /tmp/selfplay_sq8_2p.log 2>&1 & \
            echo 'Restarted selfplay'" 2>&1) || result="SSH timeout"
        log "$host_name: $result"
    fi
}

# ============================================
# GPU-Specific Functions (H100 only currently)
# ============================================

check_nnue_training() {
    local host=$1

    local nnue_running
    nnue_running=$(run_ssh "$host" "ps aux | grep train_nnue | grep -v grep | wc -l") || nnue_running="0"
    nnue_running="${nnue_running:-0}"

    if [ "$nnue_running" -eq 0 ] 2>/dev/null; then
        local model_exists
        model_exists=$(run_ssh "$host" "ls -la /home/ubuntu/ringrift/ai-service/models/nnue/*.pt 2>/dev/null | wc -l") || model_exists="0"
        model_exists="${model_exists:-0}"

        if [ "$model_exists" -gt 0 ] 2>/dev/null; then
            log "H100: NNUE training completed! Model exists."
            log "H100: Starting NNUE training for square8_3p..."
            local result
            result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "cd /home/ubuntu/ringrift/ai-service && \
                nohup python3 scripts/train_nnue.py \
                    --db data/games/lambda_sq8_3p_1.db data/games/lambda_sq8_3p_2.db data/games/lambda_sq8_3p_3.db \
                    --board-type square8 --num-players 3 --epochs 50 --batch-size 1024 \
                    --save-path models/nnue/square8_3p_v1.pt \
                    > logs/nnue/train_sq8_3p.log 2>&1 & echo 'NNUE training started'" 2>&1) || result="SSH timeout"
            log "H100: $result"
        else
            log "H100: NNUE training not running and no model found - may have crashed"
        fi
    else
        log "H100: NNUE training still running"
    fi
}

check_cmaes() {
    local host=$1

    local cmaes_running
    cmaes_running=$(run_ssh "$host" "ps aux | grep cmaes | grep -v grep | wc -l") || cmaes_running="0"
    cmaes_running="${cmaes_running:-0}"

    if [ "$cmaes_running" -eq 0 ] 2>/dev/null; then
        log "H100: CMA-ES not running, starting ITERATIVE version..."
        local result
        result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "cd /home/ubuntu/ringrift/ai-service && \
            mkdir -p logs/cmaes/iterative && \
            nohup python3 scripts/run_iterative_cmaes.py \
                --board square8 --num-players 2 \
                --generations-per-iter 15 --max-iterations 20 \
                --population-size 20 --games-per-eval 10 \
                --improvement-threshold 0.55 \
                --output-dir logs/cmaes/iterative/overnight_\$(date +%Y%m%d) \
                > logs/cmaes/iterative_overnight.log 2>&1 & echo 'CMA-ES started'" 2>&1) || result="SSH timeout"
        log "H100: $result"
    else
        log "H100: CMA-ES still running"
    fi
}

check_improvement_loop() {
    local host=$1

    local loop_running
    loop_running=$(run_ssh "$host" "ps aux | grep improvement_loop | grep -v grep | wc -l") || loop_running="0"
    loop_running="${loop_running:-0}"

    if [ "$loop_running" -eq 0 ] 2>/dev/null; then
        local model_exists
        model_exists=$(run_ssh "$host" "ls -la /home/ubuntu/ringrift/ai-service/models/nnue/*.pt 2>/dev/null | wc -l") || model_exists="0"
        model_exists="${model_exists:-0}"

        if [ "$model_exists" -gt 0 ] 2>/dev/null; then
            log "H100: Starting AlphaZero improvement loop (model found)..."
            local result
            result=$(timeout 45 ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "cd /home/ubuntu/ringrift/ai-service && \
                mkdir -p logs/improvement && \
                nohup python3 scripts/run_improvement_loop.py \
                    --board square8 --players 2 \
                    --iterations 100 --games-per-iter 50 \
                    --promotion-threshold 0.55 \
                    --resume \
                    > logs/improvement/overnight.log 2>&1 & echo 'Improvement loop started'" 2>&1) || result="SSH timeout"
            log "H100: $result"
        else
            log "H100: No NNUE model yet - waiting for initial training to complete before starting improvement loop"
        fi
    else
        log "H100: Improvement loop still running"
    fi
}

# ============================================
# Vast.ai GPU Training Functions
# ============================================

check_vast_nnue_training() {
    local host=$1
    local port=$2
    local host_name=$3
    local path=$4

    local nnue_running
    nnue_running=$(run_ssh_port "$host" "$port" "ps aux | grep train_nnue | grep -v grep | wc -l") || nnue_running="0"
    nnue_running="${nnue_running:-0}"

    if [ "$nnue_running" -eq 0 ] 2>/dev/null; then
        log "$host_name: NNUE training not running - could start training here if needed"
    else
        log "$host_name: NNUE training running"
    fi
}

# ============================================
# Data Sync Function
# ============================================

sync_data() {
    log "Starting data sync from remote instances..."
    cd "$PROJECT_DIR"

    if ./scripts/sync_selfplay_data.sh >> "$LOG_DIR/sync.log" 2>&1; then
        log "Data sync completed successfully"
    else
        log "Data sync failed (code $?)"
    fi
}

# ============================================
# Main Loop
# ============================================

log "=== Overnight Orchestrator Started ==="
log "Check interval: ${CHECK_INTERVAL}s"
log "Managing: Lambda H100/A10, AWS Staging, Vast 3090/5090-Dual/5090-Quad, Mac Cluster"

# Failure counters for each host
lambda_h100_failures=0
lambda_a10_failures=0
aws_staging_failures=0
vast_3090_failures=0
vast_5090_dual_failures=0
vast_5090_quad_failures=0
mac_studio_failures=0
mbp_16gb_failures=0
mbp_64gb_failures=0

iteration=0
while true; do
    iteration=$((iteration + 1))
    log "--- Iteration $iteration ---"

    # ============================================
    # Lambda H100 - Primary GPU (selfplay + NNUE + CMA-ES + improvement)
    # ============================================
    if [ "${LAMBDA_H100_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh "$LAMBDA_H100"; then
            lambda_h100_failures=0
            restart_selfplay_linux "$LAMBDA_H100" "Lambda-H100" "$LAMBDA_H100_MIN_JOBS" "$LAMBDA_H100_PATH" "run_ssh"
            if [ "${LAMBDA_H100_ROLE:-}" = "primary" ]; then
                check_nnue_training "$LAMBDA_H100"
                check_cmaes "$LAMBDA_H100"
                check_improvement_loop "$LAMBDA_H100"
            fi
        else
            lambda_h100_failures=$((lambda_h100_failures + 1))
            log "Lambda-H100: SSH failed (attempt $lambda_h100_failures)"
        fi
    fi

    # ============================================
    # Lambda A10 - Secondary GPU (selfplay only)
    # ============================================
    if [ "${LAMBDA_A10_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh "$LAMBDA_A10"; then
            lambda_a10_failures=0
            restart_selfplay_linux "$LAMBDA_A10" "Lambda-A10" "$LAMBDA_A10_MIN_JOBS" "$LAMBDA_A10_PATH" "run_ssh"
        else
            lambda_a10_failures=$((lambda_a10_failures + 1))
            log "Lambda-A10: SSH failed (attempt $lambda_a10_failures)"
        fi
    fi

    # ============================================
    # AWS Staging - CPU selfplay (needs key)
    # ============================================
    if [ "${AWS_STAGING_ROLE:-disabled}" != "disabled" ] && [ -n "$AWS_STAGING_KEY" ]; then
        if [ -f "$AWS_STAGING_KEY" ]; then
            if check_ssh_key "$AWS_STAGING" "$AWS_STAGING_KEY"; then
                aws_staging_failures=0
                restart_selfplay_linux "$AWS_STAGING" "AWS-Staging" "$AWS_STAGING_MIN_JOBS" "$AWS_STAGING_PATH" "run_ssh_key:$AWS_STAGING_KEY"
            else
                aws_staging_failures=$((aws_staging_failures + 1))
                log "AWS-Staging: SSH failed (attempt $aws_staging_failures)"
            fi
        else
            if [ $((iteration % 12)) -eq 1 ]; then  # Log only every hour
                log "AWS-Staging: SSH key not found at $AWS_STAGING_KEY - skipping"
            fi
        fi
    fi

    # ============================================
    # Vast.ai 3090 - GPU training/selfplay
    # ============================================
    if [ "${VAST_3090_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh_port "$VAST_3090" "$VAST_3090_PORT"; then
            vast_3090_failures=0
            restart_selfplay_linux "$VAST_3090" "Vast-3090" "$VAST_3090_MIN_JOBS" "$VAST_3090_PATH" "run_ssh_port:$VAST_3090_PORT"
        else
            vast_3090_failures=$((vast_3090_failures + 1))
            if [ $vast_3090_failures -lt $MAX_RETRIES ]; then
                log "Vast-3090: SSH failed (attempt $vast_3090_failures)"
            fi
        fi
    fi

    # ============================================
    # Vast.ai 5090 Dual - GPU training/selfplay
    # ============================================
    if [ "${VAST_5090_DUAL_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh_port "$VAST_5090_DUAL" "$VAST_5090_DUAL_PORT"; then
            vast_5090_dual_failures=0
            restart_selfplay_linux "$VAST_5090_DUAL" "Vast-5090-Dual" "$VAST_5090_DUAL_MIN_JOBS" "$VAST_5090_DUAL_PATH" "run_ssh_port:$VAST_5090_DUAL_PORT"
        else
            vast_5090_dual_failures=$((vast_5090_dual_failures + 1))
            if [ $vast_5090_dual_failures -lt $MAX_RETRIES ]; then
                log "Vast-5090-Dual: SSH failed (attempt $vast_5090_dual_failures)"
            fi
        fi
    fi

    # ============================================
    # Vast.ai 5090 Quad - Primary GPU training
    # ============================================
    if [ "${VAST_5090_QUAD_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh_port "$VAST_5090_QUAD" "$VAST_5090_QUAD_PORT"; then
            vast_5090_quad_failures=0
            restart_selfplay_linux "$VAST_5090_QUAD" "Vast-5090-Quad" "$VAST_5090_QUAD_MIN_JOBS" "$VAST_5090_QUAD_PATH" "run_ssh_port:$VAST_5090_QUAD_PORT"
        else
            vast_5090_quad_failures=$((vast_5090_quad_failures + 1))
            if [ $vast_5090_quad_failures -lt $MAX_RETRIES ]; then
                log "Vast-5090-Quad: SSH failed (attempt $vast_5090_quad_failures)"
            fi
        fi
    fi

    # ============================================
    # Mac Studio - MPS training capable (selfplay for now)
    # ============================================
    if [ "${MAC_STUDIO_ROLE:-disabled}" != "disabled" ]; then
        # Use consistent pattern with other hosts - direct check without intermediate variable
        mac_studio_ssh_ok=false
        if [ -n "${MAC_STUDIO_KEY:-}" ]; then
            check_ssh_key "$MAC_STUDIO" "$MAC_STUDIO_KEY" && mac_studio_ssh_ok=true
        else
            check_ssh "$MAC_STUDIO" && mac_studio_ssh_ok=true
        fi

        if [ "$mac_studio_ssh_ok" = "true" ]; then
            mac_studio_failures=0
            restart_selfplay_mac "$MAC_STUDIO" "Mac-Studio" "$MAC_STUDIO_MIN_JOBS" "$MAC_STUDIO_PATH" "${MAC_STUDIO_KEY:-}"
        else
            mac_studio_failures=$((mac_studio_failures + 1))
            if [ $mac_studio_failures -lt $MAX_RETRIES ]; then
                log "Mac-Studio: SSH failed (attempt $mac_studio_failures) - is Tailscale connected?"
            fi
        fi
    fi

    # ============================================
    # MacBook Pro 16GB - Light selfplay
    # ============================================
    if [ "${MBP_16GB_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh "$MBP_16GB"; then
            mbp_16gb_failures=0
            restart_selfplay_mac "$MBP_16GB" "MBP-16GB" "$MBP_16GB_MIN_JOBS" "$MBP_16GB_PATH"
        else
            mbp_16gb_failures=$((mbp_16gb_failures + 1))
            if [ $mbp_16gb_failures -lt $MAX_RETRIES ]; then
                log "MBP-16GB: SSH failed (attempt $mbp_16gb_failures) - is Tailscale connected?"
            fi
        fi
    fi

    # ============================================
    # MacBook Pro 64GB - Selfplay
    # ============================================
    if [ "${MBP_64GB_ROLE:-disabled}" != "disabled" ]; then
        if check_ssh "$MBP_64GB"; then
            mbp_64gb_failures=0
            restart_selfplay_mac "$MBP_64GB" "MBP-64GB" "$MBP_64GB_MIN_JOBS" "$MBP_64GB_PATH"
        else
            mbp_64gb_failures=$((mbp_64gb_failures + 1))
            if [ $mbp_64gb_failures -lt $MAX_RETRIES ]; then
                log "MBP-64GB: SSH failed (attempt $mbp_64gb_failures) - is Tailscale connected?"
            fi
        fi
    fi

    # ============================================
    # Data Sync (every 6 iterations = 30 minutes)
    # ============================================
    if [ $((iteration % 6)) -eq 0 ]; then
        sync_data &
    fi

    # ============================================
    # Determine Sleep Interval
    # ============================================
    # Shorter interval if any host needs retry
    any_failures=0
    for failures in $lambda_h100_failures $lambda_a10_failures $aws_staging_failures \
                    $vast_3090_failures $vast_5090_dual_failures $vast_5090_quad_failures \
                    $mac_studio_failures $mbp_16gb_failures $mbp_64gb_failures; do
        if [ "$failures" -gt 0 ] && [ "$failures" -lt $MAX_RETRIES ]; then
            any_failures=1
            break
        fi
    done

    if [ $any_failures -eq 1 ]; then
        log "Retrying failed hosts in ${RETRY_INTERVAL}s..."
        sleep "$RETRY_INTERVAL"
    else
        log "Sleeping for ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    fi
done
