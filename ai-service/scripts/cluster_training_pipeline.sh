#!/bin/bash
# Cluster Training Pipeline
# Runs self-play generation and curriculum training in optimal order
#
# Usage:
#   ./scripts/cluster_training_pipeline.sh [command]
#
# Commands:
#   selfplay     - Start self-play on all available nodes
#   train-sq8    - Train curriculum model for square8 on H100
#   train-hex    - Train curriculum model for hexagonal on H100
#   train-sq19   - Train curriculum model for square19 on H100
#   ab-test      - Run A/B test for trained model
#   status       - Check status of running jobs
#   all          - Run complete pipeline (selfplay + train + test)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Primary training host (via Tailscale)
TRAIN_HOST="ubuntu@100.78.101.123"  # lambda-h100
TRAIN_PATH="~/ringrift/ai-service"

# SSH options
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes -i ~/.ssh/id_cluster"

# GH200 nodes for self-play (via Tailscale)
GH200_HOSTS=(
    "ubuntu@100.123.183.70"   # lambda-gh200-a
    "ubuntu@100.88.35.19"     # lambda-gh200-c
    "ubuntu@100.75.84.47"     # lambda-gh200-d
    "ubuntu@100.88.176.74"    # lambda-gh200-e
    "ubuntu@100.104.165.116"  # lambda-gh200-f
    "ubuntu@100.104.126.58"   # lambda-gh200-g
    "ubuntu@100.65.88.62"     # lambda-gh200-h
    "ubuntu@100.99.27.56"     # lambda-gh200-i
    "ubuntu@100.96.142.42"    # lambda-gh200-k
    "ubuntu@100.76.145.60"    # lambda-gh200-l
)

start_selfplay() {
    echo "=========================================="
    echo "Starting Self-Play on GH200 Nodes"
    echo "=========================================="

    # Self-play command - priority on hex and square19 (we need more data)
    # Uses --canonical-export for canonical format output (December 2025)
    SELFPLAY_CMD='cd ~/ringrift/ai-service && source venv/bin/activate && \
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && \
        mkdir -p data/games logs && \
        pkill -f "run_gpu_selfplay.py" 2>/dev/null || true && \
        sleep 2 && \
        for config in "hexagonal:2" "hexagonal:3" "square19:2" "square19:3" "square8:2"; do \
            board=$(echo $config | cut -d: -f1); \
            players=$(echo $config | cut -d: -f2); \
            nohup python3 scripts/run_gpu_selfplay.py \
                --board $board \
                --num-players $players \
                --num-games 500 \
                --canonical-export \
                --skip-resource-check \
                --output-dir data/games/selfplay_${board}_${players}p \
                > logs/selfplay_${board}_${players}p.log 2>&1 & \
            sleep 1; \
        done; \
        echo "Started 5 selfplay jobs with canonical export"'

    for host in "${GH200_HOSTS[@]}"; do
        (
            echo "  Starting on $host..."
            ssh $SSH_OPTS $host "$SELFPLAY_CMD" 2>&1 || echo "  Failed: $host"
        ) &
    done

    wait
    echo ""
    echo "Self-play started on ${#GH200_HOSTS[@]} nodes"
    echo "Monitor with: ./scripts/cluster_training_pipeline.sh status"
}

train_curriculum() {
    local BOARD_TYPE="$1"
    local NUM_PLAYERS="${2:-2}"

    echo "=========================================="
    echo "Training Curriculum Model: ${BOARD_TYPE} ${NUM_PLAYERS}p"
    echo "=========================================="
    echo "Host: $TRAIN_HOST"
    echo ""

    # First aggregate any new JSONL data into the training DB
    echo "[1/3] Aggregating training data..."
    ssh $SSH_OPTS $TRAIN_HOST "cd $TRAIN_PATH && source venv/bin/activate && \
        python3 scripts/aggregate_jsonl_to_db.py \
            --input-dir data/games \
            --output-db data/games/all_jsonl_training.db \
            --incremental 2>&1 | tail -5"

    # Check available data
    echo ""
    echo "[2/3] Checking available data..."
    ssh $SSH_OPTS $TRAIN_HOST "cd $TRAIN_PATH && python3 -c \"
import sqlite3
conn = sqlite3.connect('data/games/all_jsonl_training.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM games WHERE board_type = ?', ('$BOARD_TYPE',))
count = cur.fetchone()[0]
print(f'Available games for $BOARD_TYPE: {count:,}')
if count < 100:
    print('WARNING: Low game count. Consider running more self-play first.')
\""

    # Run curriculum training
    echo ""
    echo "[3/3] Starting curriculum training (runs in background)..."
    ssh $SSH_OPTS $TRAIN_HOST "cd $TRAIN_PATH && source venv/bin/activate && \
        mkdir -p logs models/curriculum && \
        nohup python3 scripts/train_nnue_policy_curriculum.py \
            --db data/games/all_jsonl_training.db \
            --board-type $BOARD_TYPE \
            --num-players $NUM_PLAYERS \
            --output-dir models/curriculum/${BOARD_TYPE}_${NUM_PLAYERS}p \
            > logs/curriculum_${BOARD_TYPE}_${NUM_PLAYERS}p.log 2>&1 &
        echo 'Training started. PID:' \$!
        sleep 2
        tail -20 logs/curriculum_${BOARD_TYPE}_${NUM_PLAYERS}p.log 2>/dev/null || echo 'Log not ready yet'"

    echo ""
    echo "Training running in background on $TRAIN_HOST"
    echo "Monitor with: ssh $TRAIN_HOST 'tail -f $TRAIN_PATH/logs/curriculum_${BOARD_TYPE}_${NUM_PLAYERS}p.log'"
}

run_ab_test() {
    local MODEL_PATH="${1:-models/curriculum/square8_2p/stage_full/nnue_policy_square8_2p.pt}"

    echo "=========================================="
    echo "Running A/B Test"
    echo "=========================================="
    echo "Model: $MODEL_PATH"
    echo ""

    ssh $SSH_OPTS $TRAIN_HOST "cd $TRAIN_PATH && source venv/bin/activate && \
        python3 scripts/ab_test_policy_models.py \
            --model-a '$MODEL_PATH' \
            --model-b none \
            --multi-time \
            --multi-time-values 50 100 200 500 \
            --num-games 100 \
            --output data/training/ab_test_curriculum_$(date +%Y%m%d_%H%M%S).json"
}

check_status() {
    echo "=========================================="
    echo "Cluster Job Status"
    echo "=========================================="

    echo ""
    echo "=== Training Host (H100) ==="
    ssh $SSH_OPTS $TRAIN_HOST "cd $TRAIN_PATH && \
        echo 'Running processes:' && \
        ps aux | grep -E 'train_nnue|ab_test' | grep -v grep || echo '  No training jobs' && \
        echo '' && \
        echo 'Recent logs:' && \
        ls -lt logs/*.log 2>/dev/null | head -5 || echo '  No logs'"

    echo ""
    echo "=== Self-Play Nodes (sample of 3) ==="
    for host in "${GH200_HOSTS[@]:0:3}"; do
        echo ""
        echo "--- $host ---"
        ssh $SSH_OPTS $host "cd ~/ringrift/ai-service && \
            ps aux | grep 'run_gpu_selfplay' | grep -v grep | wc -l | xargs -I{} echo 'Selfplay processes: {}' && \
            find data/games -name '*.jsonl' -mmin -60 2>/dev/null | wc -l | xargs -I{} echo 'JSONL files updated in last hour: {}'" 2>&1 || echo "  Unreachable"
    done
}

case "${1:-help}" in
    selfplay)
        start_selfplay
        ;;
    train-sq8)
        train_curriculum "square8" 2
        ;;
    train-hex)
        train_curriculum "hexagonal" 2
        ;;
    train-sq19)
        train_curriculum "square19" 2
        ;;
    ab-test)
        run_ab_test "$2"
        ;;
    status)
        check_status
        ;;
    all)
        echo "Running complete pipeline..."
        start_selfplay
        echo ""
        echo "Waiting 5 minutes for initial data generation..."
        sleep 300
        train_curriculum "square8" 2
        ;;
    *)
        echo "Usage: $0 {selfplay|train-sq8|train-hex|train-sq19|ab-test|status|all}"
        echo ""
        echo "Commands:"
        echo "  selfplay   - Start self-play on all GH200 nodes"
        echo "  train-sq8  - Train curriculum model for square8"
        echo "  train-hex  - Train curriculum model for hexagonal"
        echo "  train-sq19 - Train curriculum model for square19"
        echo "  ab-test    - Run A/B test (optional: model path)"
        echo "  status     - Check status of running jobs"
        echo "  all        - Run complete pipeline"
        ;;
esac
