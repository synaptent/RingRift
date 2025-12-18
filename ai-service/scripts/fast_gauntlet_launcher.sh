#!/bin/bash
# Fast Multi-Stage Gauntlet Launcher
# Launches sharded gauntlet across cluster nodes for ~37 min completion

set -e

BOARD="${1:-square8}"
PLAYERS="${2:-2}"
NUM_SHARDS=8

# Available GH200 nodes
NODES=(
    "lambda-gh200-f"
    "lambda-gh200-g"
    "lambda-gh200-h"
    "lambda-gh200-i"
    "lambda-gh200-k"
    "lambda-gh200-l"
    "lambda-gh200-m"
    "lambda-gh200-n"
)

echo "=== FAST MULTI-STAGE GAUNTLET ==="
echo "Board: $BOARD, Players: $PLAYERS"
echo "Shards: $NUM_SHARDS across ${#NODES[@]} nodes"
echo "Expected completion: ~37 minutes"
echo ""

# Launch shards
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    SHARD=$i

    if [ $SHARD -ge $NUM_SHARDS ]; then
        break
    fi

    echo "Launching shard $SHARD on $NODE..."

    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE" "
        cd ~/ringrift/ai-service && \
        source venv/bin/activate && \
        nohup python scripts/two_stage_gauntlet.py \
            --run \
            --board $BOARD \
            --players $PLAYERS \
            --stage1-games 5 \
            --stage2-games 20 \
            --parallel 48 \
            --shard $SHARD \
            --num-shards $NUM_SHARDS \
            --no-record \
            > logs/gauntlet_shard${SHARD}_${BOARD}_${PLAYERS}p.log 2>&1 &
        echo 'Started shard $SHARD PID: '\$!
    " &
done

wait

echo ""
echo "All shards launched. Monitor with:"
echo "  tail -f logs/gauntlet_shard*_${BOARD}_${PLAYERS}p.log"
echo ""
echo "Aggregate results after completion with:"
echo "  python scripts/two_stage_gauntlet.py --aggregate --board $BOARD --players $PLAYERS"
