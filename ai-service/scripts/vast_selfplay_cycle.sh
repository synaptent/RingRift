#!/bin/bash
# Vast.ai Selfplay Cycle Script
# Runs limited batches, transfers to Lambda, cleans up, repeats
#
# Usage: ./vast_selfplay_cycle.sh [--games-per-batch 50] [--cycles 10]

set -e

# Configuration
GAMES_PER_BATCH=${GAMES_PER_BATCH:-30}  # Games per worker per batch (small to avoid disk fill)
WORKERS_PER_INSTANCE=${WORKERS_PER_INSTANCE:-8}
CYCLES=${CYCLES:-100}
LAMBDA_HOST="lambda-gpu"
LAMBDA_DATA_DIR="/home/ubuntu/ringrift/ai-service/data/collected"

# Vast.ai instance SSH commands
VAST_INSTANCES=(
    "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14133 root@ssh3.vast.ai"  # 4x RTX 5090
    "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14861 root@ssh6.vast.ai"  # 2x RTX 5090
    "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 15093 root@ssh7.vast.ai"  # 1x RTX 3090
)
VAST_NAMES=("vast4x5090" "vast2x5090" "vast1x3090")
VAST_WORKERS=(12 8 8)  # Workers per instance

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games-per-batch) GAMES_PER_BATCH="$2"; shift 2 ;;
        --cycles) CYCLES="$2"; shift 2 ;;
        --workers) WORKERS_PER_INSTANCE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Vast.ai Selfplay Cycle ==="
echo "Games per worker per batch: $GAMES_PER_BATCH"
echo "Cycles: $CYCLES"
echo "Lambda host: $LAMBDA_HOST"
echo ""

# Ensure Lambda collection directory exists
ssh $LAMBDA_HOST "mkdir -p $LAMBDA_DATA_DIR"

run_batch_on_instance() {
    local idx=$1
    local ssh_cmd="${VAST_INSTANCES[$idx]}"
    local name="${VAST_NAMES[$idx]}"
    local workers="${VAST_WORKERS[$idx]}"
    local cycle=$2
    local base_seed=$((1000000 + cycle * 10000 + idx * 1000))

    echo "[$(date '+%H:%M:%S')] Starting batch on $name ($workers workers, $GAMES_PER_BATCH games each)"

    # Start workers and wait for completion
    $ssh_cmd "
        cd ~/ringrift/ai-service
        source venv/bin/activate 2>/dev/null || true
        mkdir -p logs/selfplay data/games

        # Clear old data
        rm -f data/games/batch_*.db logs/selfplay/batch_*.jsonl

        # Start workers
        for i in \$(seq 1 $workers); do
            python3 scripts/run_self_play_soak.py \\
                --num-games $GAMES_PER_BATCH \\
                --board-type square8 \\
                --engine-mode descent-only \\
                --num-players 2 \\
                --max-moves 500 \\
                --seed \$((${base_seed} + i)) \\
                --log-jsonl logs/selfplay/batch_\${i}.jsonl \\
                --no-record-db \\
                --include-training-data \\
                >> logs/selfplay/batch.log 2>&1 &
        done

        echo 'Workers started, waiting for completion...'
        wait
        echo 'Batch complete'

        # Show results
        wc -l logs/selfplay/batch_*.jsonl 2>/dev/null | tail -1
    " 2>&1
}

transfer_from_instance() {
    local idx=$1
    local ssh_cmd="${VAST_INSTANCES[$idx]}"
    local name="${VAST_NAMES[$idx]}"
    local cycle=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    echo "[$(date '+%H:%M:%S')] Transferring data from $name to Lambda..."

    # Create temp archive on vast instance
    $ssh_cmd "
        cd ~/ringrift/ai-service
        tar czf /tmp/batch_data.tar.gz logs/selfplay/batch_*.jsonl 2>/dev/null || true
        ls -lh /tmp/batch_data.tar.gz
    " 2>&1

    # Transfer via local machine (vast -> local -> lambda)
    local local_tmp="/tmp/${name}_${timestamp}.tar.gz"
    scp -o ConnectTimeout=15 -P ${ssh_cmd##*-p } ${ssh_cmd##* }:/tmp/batch_data.tar.gz "$local_tmp" 2>/dev/null || true

    if [ -f "$local_tmp" ]; then
        scp "$local_tmp" "${LAMBDA_HOST}:${LAMBDA_DATA_DIR}/" 2>/dev/null
        ssh $LAMBDA_HOST "cd $LAMBDA_DATA_DIR && tar xzf ${name}_${timestamp}.tar.gz && rm ${name}_${timestamp}.tar.gz" 2>/dev/null
        rm "$local_tmp"
        echo "[$(date '+%H:%M:%S')] Transfer complete for $name"
    else
        echo "[$(date '+%H:%M:%S')] No data to transfer from $name"
    fi
}

cleanup_instance() {
    local idx=$1
    local ssh_cmd="${VAST_INSTANCES[$idx]}"
    local name="${VAST_NAMES[$idx]}"

    $ssh_cmd "
        cd ~/ringrift/ai-service
        rm -f logs/selfplay/batch_*.jsonl data/games/batch_*.db /tmp/batch_data.tar.gz
        echo 'Cleaned up'
    " 2>&1 | head -1
}

# Main loop
for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "=========================================="
    echo "CYCLE $cycle / $CYCLES - $(date)"
    echo "=========================================="

    # Run batches on all instances in parallel
    for idx in "${!VAST_INSTANCES[@]}"; do
        run_batch_on_instance $idx $cycle &
    done
    wait

    echo ""
    echo "[$(date '+%H:%M:%S')] All batches complete, transferring data..."

    # Transfer data from each instance
    for idx in "${!VAST_INSTANCES[@]}"; do
        transfer_from_instance $idx $cycle
        cleanup_instance $idx
    done

    # Show Lambda disk usage
    echo ""
    ssh $LAMBDA_HOST "echo 'Lambda collected data:' && ls -lh $LAMBDA_DATA_DIR/*.jsonl 2>/dev/null | wc -l && echo 'files' && du -sh $LAMBDA_DATA_DIR" 2>/dev/null

    echo ""
    echo "[$(date '+%H:%M:%S')] Cycle $cycle complete"
done

echo ""
echo "=== All cycles complete ==="
echo "Data collected on Lambda at: $LAMBDA_DATA_DIR"
