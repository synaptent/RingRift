#!/bin/bash
# Unified Selfplay Cycle Script
# Runs batches on Vast.ai (JSONL only) and AWS (lean DB), transfers to Lambda
#
# Usage: ./selfplay_cycle.sh [--cycles 100] [--vast-games 30] [--aws-games 200]

set -e

# Configuration
VAST_GAMES_PER_BATCH=${VAST_GAMES_PER_BATCH:-30}   # Small batches for Vast (16GB disk)
AWS_GAMES_PER_BATCH=${AWS_GAMES_PER_BATCH:-200}    # Larger batches for AWS (~78GB disk)
AWS_MAX_DB_SIZE_MB=${AWS_MAX_DB_SIZE_MB:-500}      # Max DB size before forced transfer (lean mode ~500KB/game = ~1000 games)
CYCLES=${CYCLES:-100}
LAMBDA_HOST="lambda-gpu"
LAMBDA_DATA_DIR="/home/ubuntu/ringrift/ai-service/data/collected"

# Instance configurations
declare -A INSTANCES=(
    # Vast.ai instances (JSONL only, no DB)
    ["vast4x5090"]="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14133 root@ssh3.vast.ai"
    ["vast2x5090"]="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14861 root@ssh6.vast.ai"
    ["vast3090"]="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 15093 root@ssh7.vast.ai"
    # AWS instances (lean DB recording)
    ["aws_staging"]="ssh ringrift-staging"
    ["aws_extra"]="ssh ringrift-selfplay-extra"
)

declare -A WORKERS=(
    ["vast4x5090"]=12
    ["vast2x5090"]=8
    ["vast3090"]=8
    ["aws_staging"]=1
    ["aws_extra"]=1
)

declare -A INSTANCE_TYPE=(
    ["vast4x5090"]="vast"
    ["vast2x5090"]="vast"
    ["vast3090"]="vast"
    ["aws_staging"]="aws"
    ["aws_extra"]="aws"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cycles) CYCLES="$2"; shift 2 ;;
        --vast-games) VAST_GAMES_PER_BATCH="$2"; shift 2 ;;
        --aws-games) AWS_GAMES_PER_BATCH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Unified Selfplay Cycle ==="
echo "Vast.ai games per worker: $VAST_GAMES_PER_BATCH"
echo "AWS games per worker: $AWS_GAMES_PER_BATCH"
echo "Cycles: $CYCLES"
echo "Lambda host: $LAMBDA_HOST"
echo ""

# Ensure Lambda collection directory exists
ssh $LAMBDA_HOST "mkdir -p $LAMBDA_DATA_DIR/jsonl $LAMBDA_DATA_DIR/db"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

run_vast_batch() {
    local name=$1
    local ssh_cmd="${INSTANCES[$name]}"
    local workers="${WORKERS[$name]}"
    local cycle=$2
    local base_seed=$((1000000 + cycle * 10000 + RANDOM % 1000))

    log "Starting Vast batch on $name ($workers workers, $VAST_GAMES_PER_BATCH games each)"

    $ssh_cmd "
        cd ~/ringrift/ai-service
        source venv/bin/activate 2>/dev/null || true
        mkdir -p logs/selfplay

        # Clear old batch data
        rm -f logs/selfplay/batch_*.jsonl

        # Start workers
        for i in \$(seq 1 $workers); do
            python3 scripts/run_self_play_soak.py \\
                --num-games $VAST_GAMES_PER_BATCH \\
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

        wait
        echo 'done'
        wc -l logs/selfplay/batch_*.jsonl 2>/dev/null | tail -1 || echo '0 games'
    " 2>&1
}

run_aws_batch() {
    local name=$1
    local ssh_cmd="${INSTANCES[$name]}"
    local cycle=$2
    local base_seed=$((2000000 + cycle * 10000 + RANDOM % 1000))
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log "Starting AWS batch on $name ($AWS_GAMES_PER_BATCH games, lean DB)"

    $ssh_cmd "
        cd ~/ringrift/ai-service
        source venv/bin/activate
        mkdir -p logs/selfplay data/games

        # Run selfplay with lean DB
        python3 scripts/run_self_play_soak.py \\
            --num-games $AWS_GAMES_PER_BATCH \\
            --board-type square8 \\
            --engine-mode descent-only \\
            --num-players 2 \\
            --max-moves 500 \\
            --seed ${base_seed} \\
            --log-jsonl logs/selfplay/batch_${timestamp}.jsonl \\
            --record-db data/games/batch_${timestamp}.db \\
            --lean-db \\
            >> logs/selfplay/batch.log 2>&1

        echo 'done'
        ls -lh data/games/batch_${timestamp}.db 2>/dev/null || echo 'No DB'
        wc -l logs/selfplay/batch_${timestamp}.jsonl 2>/dev/null || echo '0 games'
    " 2>&1
}

transfer_vast_data() {
    local name=$1
    local ssh_cmd="${INSTANCES[$name]}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log "Transferring JSONL from $name to Lambda..."

    # Create archive on vast
    $ssh_cmd "
        cd ~/ringrift/ai-service
        tar czf /tmp/batch_${name}.tar.gz logs/selfplay/batch_*.jsonl 2>/dev/null || true
    " 2>&1

    # Transfer via local
    local local_tmp="/tmp/${name}_${timestamp}.tar.gz"
    local port=$(echo "$ssh_cmd" | grep -oP '\-p \K\d+')
    local host=$(echo "$ssh_cmd" | grep -oP 'root@\S+')

    scp -o ConnectTimeout=15 -P $port $host:/tmp/batch_${name}.tar.gz "$local_tmp" 2>/dev/null || true

    if [ -f "$local_tmp" ]; then
        scp "$local_tmp" "${LAMBDA_HOST}:${LAMBDA_DATA_DIR}/jsonl/" 2>/dev/null
        ssh $LAMBDA_HOST "cd $LAMBDA_DATA_DIR/jsonl && tar xzf ${name}_${timestamp}.tar.gz --strip-components=2 && rm ${name}_${timestamp}.tar.gz" 2>/dev/null
        rm "$local_tmp"
        log "Transferred JSONL from $name"
    fi

    # Cleanup vast
    $ssh_cmd "rm -f logs/selfplay/batch_*.jsonl /tmp/batch_${name}.tar.gz" 2>&1 >/dev/null
}

transfer_aws_data() {
    local name=$1
    local ssh_cmd="${INSTANCES[$name]}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log "Transferring DB and JSONL from $name to Lambda..."

    # Get list of batch files
    local files=$($ssh_cmd "cd ~/ringrift/ai-service && ls data/games/batch_*.db logs/selfplay/batch_*.jsonl 2>/dev/null" 2>&1 | tr '\n' ' ')

    if [ -n "$files" ]; then
        # Transfer DBs directly (they're small with lean mode)
        for db in $($ssh_cmd "ls ~/ringrift/ai-service/data/games/batch_*.db 2>/dev/null"); do
            local basename=$(basename $db)
            scp "${name#aws_}:$db" "/tmp/${name}_${basename}" 2>/dev/null || true
            if [ -f "/tmp/${name}_${basename}" ]; then
                scp "/tmp/${name}_${basename}" "${LAMBDA_HOST}:${LAMBDA_DATA_DIR}/db/" 2>/dev/null
                rm "/tmp/${name}_${basename}"
            fi
        done

        # Transfer JSONLs
        $ssh_cmd "
            cd ~/ringrift/ai-service
            tar czf /tmp/batch_jsonl.tar.gz logs/selfplay/batch_*.jsonl 2>/dev/null || true
        " 2>&1

        scp "${name#aws_}:~/ringrift/ai-service/tmp/batch_jsonl.tar.gz" "/tmp/${name}_jsonl.tar.gz" 2>/dev/null || true
        if [ -f "/tmp/${name}_jsonl.tar.gz" ]; then
            scp "/tmp/${name}_jsonl.tar.gz" "${LAMBDA_HOST}:${LAMBDA_DATA_DIR}/jsonl/" 2>/dev/null
            ssh $LAMBDA_HOST "cd $LAMBDA_DATA_DIR/jsonl && tar xzf ${name}_jsonl.tar.gz --strip-components=2 && rm ${name}_jsonl.tar.gz" 2>/dev/null
            rm "/tmp/${name}_jsonl.tar.gz"
        fi

        log "Transferred data from $name"

        # Cleanup AWS
        $ssh_cmd "rm -f ~/ringrift/ai-service/data/games/batch_*.db ~/ringrift/ai-service/logs/selfplay/batch_*.jsonl /tmp/batch_jsonl.tar.gz" 2>&1 >/dev/null
    fi
}

# Main loop
for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "=========================================="
    echo "CYCLE $cycle / $CYCLES - $(date)"
    echo "=========================================="

    # Run batches on all instances in parallel
    pids=()
    for name in "${!INSTANCES[@]}"; do
        type="${INSTANCE_TYPE[$name]}"
        if [ "$type" = "vast" ]; then
            run_vast_batch "$name" $cycle &
            pids+=($!)
        else
            run_aws_batch "$name" $cycle &
            pids+=($!)
        fi
    done

    # Wait for all batches
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    log "All batches complete, transferring data..."

    # Transfer data from each instance
    for name in "${!INSTANCES[@]}"; do
        type="${INSTANCE_TYPE[$name]}"
        if [ "$type" = "vast" ]; then
            transfer_vast_data "$name"
        else
            transfer_aws_data "$name"
        fi
    done

    # Show Lambda stats
    echo ""
    ssh $LAMBDA_HOST "
        echo 'Lambda collected data:'
        echo -n 'JSONL files: ' && ls $LAMBDA_DATA_DIR/jsonl/*.jsonl 2>/dev/null | wc -l
        echo -n 'DB files: ' && ls $LAMBDA_DATA_DIR/db/*.db 2>/dev/null | wc -l
        echo -n 'Total JSONL size: ' && du -sh $LAMBDA_DATA_DIR/jsonl 2>/dev/null | cut -f1
        echo -n 'Total DB size: ' && du -sh $LAMBDA_DATA_DIR/db 2>/dev/null | cut -f1
    " 2>/dev/null

    log "Cycle $cycle complete"
done

echo ""
echo "=== All cycles complete ==="
echo "Data collected on Lambda at: $LAMBDA_DATA_DIR"
