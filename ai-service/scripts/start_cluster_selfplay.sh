#!/bin/bash
# Start DIVERSE selfplay on all cluster nodes with priority on hex and square19
# Uses run_diverse_selfplay.py for high-quality varied AI matchups
# Run from ai-service directory

set -e

# GH200 nodes (via Tailscale IPs)
GH200_HOSTS=(
    "100.123.183.70"  # lambda-gh200-a
    "100.104.34.73"   # lambda-gh200-b
    "100.88.35.19"    # lambda-gh200-c
    "100.75.84.47"    # lambda-gh200-d
    "100.88.176.74"   # lambda-gh200-e
    "100.104.165.116" # lambda-gh200-f
    "100.104.126.58"  # lambda-gh200-g
    "100.65.88.62"    # lambda-gh200-h
    "100.99.27.56"    # lambda-gh200-i
    "100.96.142.42"   # lambda-gh200-k
    "100.76.145.60"   # lambda-gh200-l
)

# Other Lambda instances
LAMBDA_HOSTS=(
    "100.91.25.13"    # lambda-a10
    "100.78.101.123"  # lambda-h100
    "100.97.104.89"   # lambda-2xh100
)

# Vast instances
VAST_HOSTS=(
    "100.118.201.85"  # vast-5090-quad-new
    "100.100.242.64"  # vast-rtx4060ti
    "100.74.154.36"   # vast-rtx3070-a-new
    "100.75.98.13"    # vast-rtx2060s
)

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "=============================================="
echo "Starting Cluster DIVERSE Selfplay - Priority: hex/square19"
echo "=============================================="

# Diverse selfplay command - uses varied AI matchups (NNUE, NN-MCTS, NN-Minimax, heuristic)
# This generates higher quality training data than GPU-only selfplay
SELFPLAY_CMD='cd ~/ringrift/ai-service && source venv/bin/activate && \
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && \
    nohup python3 scripts/run_diverse_selfplay.py \
    --priority-configs \
    --games-per-matchup 100 \
    --output-dir data/games \
    > logs/diverse_selfplay_$(hostname)_$(date +%Y%m%d_%H%M%S).log 2>&1 &'

# Alternative: run specific hex/sq19 configs only with diverse AI
HEX_SQ19_CMD='cd ~/ringrift/ai-service && source venv/bin/activate && \
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && \
    mkdir -p data/games logs && \
    for config in "hexagonal:2" "hexagonal:3" "hexagonal:4" "square19:2" "square19:3" "square19:4"; do \
        board=$(echo $config | cut -d: -f1); \
        players=$(echo $config | cut -d: -f2); \
        nohup python3 scripts/run_diverse_selfplay.py \
            --board $board \
            --players $players \
            --games-per-matchup 100 \
            --output-dir data/games \
            > logs/diverse_selfplay_${board}_${players}p_$(hostname).log 2>&1 & \
        sleep 1; \
    done; \
    echo "Started 6 diverse selfplay jobs for hex and square19"'

echo ""
echo "Starting selfplay on GH200 nodes..."
for host in "${GH200_HOSTS[@]}"; do
    (
        echo "  Starting on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "$HEX_SQ19_CMD"
    ) &
done

echo ""
echo "Starting selfplay on Lambda nodes..."
for host in "${LAMBDA_HOSTS[@]}"; do
    (
        echo "  Starting on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "$HEX_SQ19_CMD"
    ) &
done

echo ""
echo "Starting selfplay on Vast instances..."
for host in "${VAST_HOSTS[@]}"; do
    (
        echo "  Starting on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster root@$host "$HEX_SQ19_CMD"
    ) &
done

wait

echo ""
echo "=============================================="
echo "Selfplay started on all nodes!"
echo "=============================================="
echo ""
echo "Each node is running 6 selfplay processes:"
echo "  - hexagonal 2p, 3p, 4p"
echo "  - square19 2p, 3p, 4p"
echo ""
echo "Monitor progress with:"
echo "  ssh ubuntu@<node> 'ps aux | grep run_gpu_selfplay | wc -l'"
echo "  ssh ubuntu@<node> 'wc -l ~/ringrift/ai-service/data/games/gpu_selfplay/*/*.jsonl'"
