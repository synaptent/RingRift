#!/bin/bash
# Launch distributed gauntlet across multiple nodes
#
# Usage:
#   ./scripts/launch_distributed_gauntlet.sh hex8_2p models/canonical_hex8_2p.pth
#   ./scripts/launch_distributed_gauntlet.sh square8_3p models/canonical_square8_3p.pth --mcts

set -e

CONFIG=$1
MODEL=$2
USE_MCTS=${3:-""}

if [ -z "$CONFIG" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 <config> <model_path> [--mcts]"
    echo "  config: hex8_2p, hex8_3p, square8_2p, square8_3p, etc."
    echo "  model_path: path to model file"
    echo "  --mcts: use MCTS instead of policy-only (slower but more accurate)"
    exit 1
fi

# Parse config into board_type and num_players
BOARD_TYPE=$(echo $CONFIG | sed 's/_[0-9]p$//')
NUM_PLAYERS=$(echo $CONFIG | grep -oE '[0-9]+p$' | tr -d 'p')

echo "=== Distributed Gauntlet: $CONFIG ==="
echo "Model: $MODEL"
echo "Board: $BOARD_TYPE, Players: $NUM_PLAYERS"
echo "Mode: ${USE_MCTS:-policy-only}"
echo ""

# Available idle nodes (checked earlier)
NODES=("lambda-gh200-9" "lambda-gh200-11" "nebius-backbone-1" "nebius-h100-3")
NUM_NODES=${#NODES[@]}

# Games per opponent
GAMES_PER_OPPONENT=20
OPPONENTS=("random" "heuristic")

MCTS_FLAG=""
if [ "$USE_MCTS" == "--mcts" ]; then
    MCTS_FLAG="--use-mcts"
fi

# Create results directory
RESULTS_DIR="/tmp/gauntlet_${CONFIG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "Distributing across ${NUM_NODES} nodes..."
echo "Results directory: $RESULTS_DIR"
echo ""

# First, update all nodes to latest code
echo "Updating nodes..."
for node in "${NODES[@]}"; do
    ssh $node "cd ~/ringrift && git fetch origin && git reset --hard origin/main" 2>/dev/null &
done
wait
echo "Nodes updated."
echo ""

# Launch sharded gauntlets
JOB_COUNT=0
for opponent in "${OPPONENTS[@]}"; do
    # Distribute shards across nodes
    for shard in $(seq 0 $((NUM_NODES - 1))); do
        node=${NODES[$shard]}
        output_file="shard_${opponent}_${shard}.json"

        echo "Starting shard $shard on $node (vs $opponent)..."

        ssh $node "cd ~/ringrift/ai-service && \
            nohup bash -c 'PYTHONPATH=. python scripts/sharded_gauntlet.py \
                --model $MODEL \
                --board-type $BOARD_TYPE \
                --num-players $NUM_PLAYERS \
                --opponent $opponent \
                --games $GAMES_PER_OPPONENT \
                --shard $shard \
                --total-shards $NUM_NODES \
                $MCTS_FLAG \
                --output /tmp/$output_file' \
            > /tmp/gauntlet_${opponent}_shard${shard}.log 2>&1 &" &

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

wait
echo ""
echo "Launched $JOB_COUNT gauntlet shards across $NUM_NODES nodes"
echo ""
echo "To check progress:"
for node in "${NODES[@]}"; do
    echo "  ssh $node 'tail -20 /tmp/gauntlet_*.log'"
done
echo ""
echo "To collect results when complete:"
echo "  for node in ${NODES[@]}; do scp \$node:/tmp/shard_*.json $RESULTS_DIR/; done"
echo "  PYTHONPATH=. python scripts/sharded_gauntlet.py --aggregate --results-dir $RESULTS_DIR"
