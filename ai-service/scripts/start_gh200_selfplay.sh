#!/bin/bash
# GH200 Selfplay Startup Script
#
# Designed for Lambda Labs GH200 instances (72 vCPUs, 480GB GPU, 200GB RAM)
# Runs a balanced mix of GPU-accelerated and neural network selfplay
#
# Usage:
#   ./scripts/start_gh200_selfplay.sh
#
# Stop with:
#   pkill -f "run_self_play_soak"

set -e
cd "$(dirname "$0")/.."

# Configuration
HOSTNAME=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/selfplay_${HOSTNAME}"
DATA_DIR="data/selfplay/${HOSTNAME}"

mkdir -p "$LOG_DIR" "$DATA_DIR"

echo "Starting GH200 selfplay on $HOSTNAME at $TIMESTAMP"
echo "Log dir: $LOG_DIR"
echo "Data dir: $DATA_DIR"

# Kill any existing selfplay processes
pkill -f "run_self_play_soak" 2>/dev/null || true
sleep 2

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ============================================================================
# GPU-Accelerated Heuristic Selfplay (fast data generation for warmup)
# - Uses CUDA for parallel game simulation
# - Only supports square8 2p with heuristic engines
# - High throughput: ~1000+ games/minute
# ============================================================================
echo "Starting GPU-accelerated heuristic selfplay..."

nohup python3 scripts/run_self_play_soak.py \
    --board-type square8 \
    --num-players 2 \
    --engine-mode heuristic-only \
    --gpu \
    --gpu-batch-size 512 \
    --log-jsonl "${DATA_DIR}/gpu_heuristic_sq8_2p.jsonl" \
    --record-db "${DATA_DIR}/gpu_heuristic.db" \
    --lean-db \
    > "${LOG_DIR}/gpu_heuristic.log" 2>&1 &

echo "  GPU heuristic selfplay started (PID: $!)"

# ============================================================================
# Neural Network Selfplay (high quality training data)
# - Uses best NN model for both players (self-play)
# - Lower throughput but higher quality positions
# - Run 4 parallel workers to utilize CPU cores
# ============================================================================
echo "Starting NN selfplay workers..."

for i in 1 2 3 4; do
    nohup python3 scripts/run_self_play_soak.py \
        --board-type square8 \
        --num-players 2 \
        --engine-mode nn-only \
        --log-jsonl "${DATA_DIR}/nn_sq8_2p_worker${i}.jsonl" \
        --record-db "${DATA_DIR}/nn_selfplay.db" \
        --lean-db \
        --watch-model-updates \
        --model-reload-interval 50 \
        > "${LOG_DIR}/nn_sq8_2p_worker${i}.log" 2>&1 &

    echo "  NN worker $i started (PID: $!)"
    sleep 1
done

# ============================================================================
# Mixed Engine Selfplay (diverse training data)
# - Samples from full engine ladder (random, heuristic, minimax, mcts, nn)
# - Important for robust training
# - Run 2 parallel workers
# ============================================================================
echo "Starting mixed engine selfplay..."

for config in "square8:2" "square8:3" "square8:4"; do
    board=$(echo $config | cut -d: -f1)
    players=$(echo $config | cut -d: -f2)

    nohup python3 scripts/run_self_play_soak.py \
        --board-type $board \
        --num-players $players \
        --engine-mode mixed \
        --log-jsonl "${DATA_DIR}/mixed_${board}_${players}p.jsonl" \
        --record-db "${DATA_DIR}/mixed_selfplay.db" \
        --lean-db \
        > "${LOG_DIR}/mixed_${board}_${players}p.log" 2>&1 &

    echo "  Mixed $board ${players}p started (PID: $!)"
    sleep 1
done

# ============================================================================
# Summary
# ============================================================================
sleep 3
echo ""
echo "========================================="
echo "GH200 Selfplay Started"
echo "========================================="
echo "Processes running:"
ps aux | grep "run_self_play_soak" | grep -v grep | wc -l
echo ""
echo "GPU utilization:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "N/A"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  watch 'wc -l ${DATA_DIR}/*.jsonl'"
echo ""
echo "Stop with:"
echo "  pkill -f run_self_play_soak"
