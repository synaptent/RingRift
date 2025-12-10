#!/bin/bash
# Overnight Orchestrator - Keeps AI training pipeline running
# Run with: nohup ./scripts/overnight_orchestrator.sh >> logs/overnight.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/overnight"
LOCKFILE="/tmp/overnight_orchestrator.lock"

# Configuration
LAMBDA_H100="ubuntu@209.20.157.81"
LAMBDA_A10="ubuntu@150.136.65.197"
CHECK_INTERVAL=300  # 5 minutes
MAX_RETRIES=3

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/orchestrator.log"
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

check_ssh() {
    local host=$1
    timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" >/dev/null 2>&1
}

count_python_jobs() {
    local host=$1
    ssh -o ConnectTimeout=10 "$host" "ps aux | grep -E 'python.*selfplay|python.*train|python.*cmaes' | grep -v grep | wc -l" 2>/dev/null || echo "0"
}

restart_selfplay_if_needed() {
    local host=$1
    local host_name=$2
    local min_jobs=$3
    
    local current_jobs=$(count_python_jobs "$host")
    log "$host_name: $current_jobs Python jobs running (min: $min_jobs)"
    
    if [ "$current_jobs" -lt "$min_jobs" ]; then
        log "$host_name: Restarting selfplay jobs..."
        
        local seed_base=$((RANDOM * 1000))
        ssh -o ConnectTimeout=30 "$host" "cd /home/ubuntu/ringrift/ai-service && \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 1000 --board-type square8 --num-players 2 --output-dir data/selfplay/sq8_2p --seed $seed_base > /tmp/selfplay_sq8_2p.log 2>&1 & \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 500 --board-type square8 --num-players 3 --output-dir data/selfplay/sq8_3p --seed $((seed_base+1)) > /tmp/selfplay_sq8_3p.log 2>&1 & \
            nohup python3 scripts/run_hybrid_selfplay.py --num-games 500 --board-type hex --num-players 2 --output-dir data/selfplay/hex_2p --seed $((seed_base+2)) > /tmp/selfplay_hex_2p.log 2>&1 & \
            echo 'Restarted selfplay jobs'" 2>&1 | while read line; do log "$host_name: $line"; done
    fi
}

check_nnue_training() {
    local host=$1
    
    # Check if NNUE training is running
    local nnue_running=$(ssh -o ConnectTimeout=10 "$host" "ps aux | grep train_nnue | grep -v grep | wc -l" 2>/dev/null || echo "0")
    
    if [ "$nnue_running" -eq 0 ]; then
        # Check if model was created (training completed)
        local model_exists=$(ssh -o ConnectTimeout=10 "$host" "ls -la /home/ubuntu/ringrift/ai-service/models/nnue/*.pt 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        
        if [ "$model_exists" -gt 0 ]; then
            log "H100: NNUE training completed! Model exists."
            # Start training on next config
            log "H100: Starting NNUE training for square8_3p..."
            ssh -o ConnectTimeout=30 "$host" "cd /home/ubuntu/ringrift/ai-service && \
                nohup python3 scripts/train_nnue.py \
                    --db data/games/lambda_sq8_3p_1.db data/games/lambda_sq8_3p_2.db data/games/lambda_sq8_3p_3.db \
                    --board-type square8 --num-players 3 --epochs 50 --batch-size 1024 \
                    --save-path models/nnue/square8_3p_v1.pt \
                    > logs/nnue/train_sq8_3p.log 2>&1 &" 2>&1 | while read line; do log "H100: $line"; done
        else
            log "H100: NNUE training not running and no model found - may have crashed"
        fi
    else
        log "H100: NNUE training still running"
    fi
}

check_cmaes() {
    local host=$1

    local cmaes_running=$(ssh -o ConnectTimeout=10 "$host" "ps aux | grep cmaes | grep -v grep | wc -l" 2>/dev/null || echo "0")

    if [ "$cmaes_running" -eq 0 ]; then
        log "H100: CMA-ES not running, starting ITERATIVE version..."
        ssh -o ConnectTimeout=30 "$host" "cd /home/ubuntu/ringrift/ai-service && \
            mkdir -p logs/cmaes/iterative && \
            nohup python3 scripts/run_iterative_cmaes.py \
                --board square8 --num-players 2 \
                --generations-per-iter 15 --max-iterations 20 \
                --population-size 20 --games-per-eval 10 \
                --improvement-threshold 0.55 \
                --output-dir logs/cmaes/iterative/overnight_$(date +%Y%m%d) \
                > logs/cmaes/iterative_overnight.log 2>&1 &" 2>&1 | while read line; do log "H100: $line"; done
    else
        log "H100: CMA-ES still running"
    fi
}

sync_data() {
    log "Starting data sync from remote instances..."
    cd "$PROJECT_DIR"

    if ./scripts/sync_selfplay_data.sh >> "$LOG_DIR/sync.log" 2>&1; then
        log "Data sync completed successfully"
    else
        log "Data sync failed (code $?)"
    fi
}

check_improvement_loop() {
    local host=$1

    # Check if improvement loop is running
    local loop_running=$(ssh -o ConnectTimeout=10 "$host" "ps aux | grep improvement_loop | grep -v grep | wc -l" 2>/dev/null || echo "0")

    if [ "$loop_running" -eq 0 ]; then
        # Check if there's a trained model to build on
        local model_exists=$(ssh -o ConnectTimeout=10 "$host" "ls -la /home/ubuntu/ringrift/ai-service/models/nnue/*.pt 2>/dev/null | wc -l" 2>/dev/null || echo "0")

        if [ "$model_exists" -gt 0 ]; then
            log "H100: Starting AlphaZero improvement loop (model found)..."
            ssh -o ConnectTimeout=30 "$host" "cd /home/ubuntu/ringrift/ai-service && \
                mkdir -p logs/improvement && \
                nohup python3 scripts/run_improvement_loop.py \
                    --board square8 --players 2 \
                    --iterations 100 --games-per-iter 50 \
                    --promotion-threshold 0.55 \
                    --resume \
                    > logs/improvement/overnight.log 2>&1 &" 2>&1 | while read line; do log "H100: $line"; done
        else
            log "H100: No NNUE model yet - waiting for initial training to complete before starting improvement loop"
        fi
    else
        log "H100: Improvement loop still running"
    fi
}

# Main loop
log "=== Overnight Orchestrator Started ==="
log "Check interval: ${CHECK_INTERVAL}s"

iteration=0
while true; do
    iteration=$((iteration + 1))
    log "--- Iteration $iteration ---"
    
    # Check H100
    if check_ssh "$LAMBDA_H100"; then
        restart_selfplay_if_needed "$LAMBDA_H100" "H100" 3
        check_nnue_training "$LAMBDA_H100"
        check_cmaes "$LAMBDA_H100"
        check_improvement_loop "$LAMBDA_H100"
    else
        log "H100: SSH connection failed"
    fi
    
    # Check A10
    if check_ssh "$LAMBDA_A10"; then
        restart_selfplay_if_needed "$LAMBDA_A10" "A10" 2
    else
        log "A10: SSH connection failed"
    fi
    
    # Sync data every 6 iterations (30 minutes)
    if [ $((iteration % 6)) -eq 0 ]; then
        sync_data &
    fi
    
    # Log status summary
    log "Sleeping for ${CHECK_INTERVAL}s..."
    sleep "$CHECK_INTERVAL"
done
