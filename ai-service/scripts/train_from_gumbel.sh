#!/bin/bash
# Train NNUE models from Gumbel MCTS selfplay data ON CLUSTER
# All intensive work runs on cluster nodes, not locally

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Local directories (lightweight data only)
COLLECTED_DATA_DIR="$PROJECT_ROOT/data/selfplay_collected"

# Cluster training node (use a square8 node with good GPU)
TRAINING_NODE="100.65.88.62"
REMOTE_PROJECT="/home/ubuntu/ringrift/ai-service"

# Training parameters
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_data() {
    log "=== Checking Available Data (Local) ==="

    for board in square8 square19 hexagonal hex8; do
        local agg="${COLLECTED_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl"
        if [ -f "$agg" ]; then
            local count=$(wc -l < "$agg")
            echo "  $board: $count games"
        else
            echo "  $board: 0 games (no aggregate file)"
        fi
    done
}

push_data_to_cluster() {
    local board=$1
    local input_file="${COLLECTED_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl"

    if [ ! -f "$input_file" ]; then
        log "No data file for $board, skipping"
        return 1
    fi

    local game_count=$(wc -l < "$input_file")
    if [ "$game_count" -lt 10 ]; then
        log "Insufficient data for $board ($game_count games < 10), skipping"
        return 1
    fi

    log "Pushing $game_count $board games to cluster..."
    ssh -o ConnectTimeout=10 ubuntu@${TRAINING_NODE} "mkdir -p ${REMOTE_PROJECT}/data/selfplay_collected"
    scp -o ConnectTimeout=30 "$input_file" ubuntu@${TRAINING_NODE}:${REMOTE_PROJECT}/data/selfplay_collected/

    return 0
}

import_on_cluster() {
    local board=$1

    log "Importing $board data to SQLite on cluster..."

    ssh -o ConnectTimeout=30 ubuntu@${TRAINING_NODE} "cd ${REMOTE_PROJECT} && source venv/bin/activate && \
        mkdir -p data/training_dbs && \
        rm -f data/training_dbs/gumbel_${board}_2p.db && \
        PYTHONPATH=. python scripts/import_gpu_selfplay_to_db.py \
            --input data/selfplay_collected/gumbel_${board}_2p_aggregate.jsonl \
            --output data/training_dbs/gumbel_${board}_2p.db \
            --skip-expansion \
            --skip-invalid-moves"

    # Verify import
    local db_count=$(ssh -o ConnectTimeout=10 ubuntu@${TRAINING_NODE} \
        "sqlite3 ${REMOTE_PROJECT}/data/training_dbs/gumbel_${board}_2p.db 'SELECT COUNT(*) FROM games;'" 2>/dev/null || echo "0")
    log "  Imported $db_count games to cluster database"

    return 0
}

train_on_cluster() {
    local board=$1

    log "Starting training for $board on cluster (backgrounded)..."

    # Run training in background on cluster
    ssh -o ConnectTimeout=30 ubuntu@${TRAINING_NODE} "cd ${REMOTE_PROJECT} && source venv/bin/activate && \
        PYTHONPATH=. nohup python scripts/train_nnue_policy.py \
            --db data/training_dbs/gumbel_${board}_2p.db \
            --board-type ${board} \
            --num-players 2 \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LEARNING_RATE} \
            --save-interval 10 \
            > /tmp/train_${board}.log 2>&1 &"

    log "  Training started. Monitor with: ssh ubuntu@${TRAINING_NODE} 'tail -f /tmp/train_${board}.log'"
    return 0
}

check_training_status() {
    log "=== Training Status on Cluster ==="

    local procs=$(ssh -o ConnectTimeout=10 ubuntu@${TRAINING_NODE} "pgrep -f 'train_nnue_policy' | wc -l" 2>/dev/null || echo "0")
    echo "Active training processes: $procs"

    for board in square8 square19 hexagonal hex8; do
        local log_file="/tmp/train_${board}.log"
        local status=$(ssh -o ConnectTimeout=10 ubuntu@${TRAINING_NODE} \
            "if [ -f ${log_file} ]; then tail -1 ${log_file}; else echo 'Not started'; fi" 2>/dev/null)
        echo "  $board: $status"
    done
}

pull_models() {
    log "=== Pulling Trained Models from Cluster ==="

    mkdir -p "$PROJECT_ROOT/models/nnue"

    for board in square8 square19 hexagonal hex8; do
        local remote_model="${REMOTE_PROJECT}/models/nnue/nnue_policy_${board}_2p.pt"
        local exists=$(ssh -o ConnectTimeout=10 ubuntu@${TRAINING_NODE} "[ -f ${remote_model} ] && echo yes || echo no" 2>/dev/null)

        if [ "$exists" = "yes" ]; then
            log "Pulling model for $board..."
            scp -o ConnectTimeout=60 ubuntu@${TRAINING_NODE}:${remote_model} "$PROJECT_ROOT/models/nnue/"
            echo "  $board: downloaded"
        else
            echo "  $board: no model yet"
        fi
    done
}

main() {
    case "${1:-help}" in
        check)
            check_data
            ;;
        push)
            log "=== Pushing Data to Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                push_data_to_cluster "$board" || true
            done
            ;;
        import)
            log "=== Importing Data on Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                import_on_cluster "$board" || true
            done
            ;;
        train)
            log "=== Starting Training on Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                train_on_cluster "$board" || true
            done
            ;;
        status)
            check_training_status
            ;;
        pull)
            pull_models
            ;;
        all)
            log "=== Full Pipeline (runs on cluster) ==="

            check_data

            log ""
            log "=== Step 1: Push Data to Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                push_data_to_cluster "$board" || true
            done

            log ""
            log "=== Step 2: Import Data on Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                import_on_cluster "$board" || true
            done

            log ""
            log "=== Step 3: Start Training on Cluster ==="
            for board in square8 square19 hexagonal hex8; do
                train_on_cluster "$board" || true
            done

            log ""
            log "=== Pipeline Started ==="
            log "Training is running on cluster. Use 'status' to check progress."
            log "Use 'pull' to download trained models when complete."
            ;;
        *)
            echo "Usage: $0 [check|push|import|train|status|pull|all]"
            echo ""
            echo "Commands (all intensive work on cluster):"
            echo "  check   - Show available local data"
            echo "  push    - Push data to cluster node"
            echo "  import  - Import JSONL to SQLite ON CLUSTER"
            echo "  train   - Start training ON CLUSTER (backgrounded)"
            echo "  status  - Check training progress on cluster"
            echo "  pull    - Download trained models from cluster"
            echo "  all     - Run full pipeline on cluster"
            echo ""
            echo "Training node: ${TRAINING_NODE}"
            echo ""
            echo "Environment variables:"
            echo "  EPOCHS=50         - Number of training epochs"
            echo "  BATCH_SIZE=256    - Training batch size"
            echo "  LEARNING_RATE=0.001 - Learning rate"
            exit 1
            ;;
    esac
}

main "$@"
