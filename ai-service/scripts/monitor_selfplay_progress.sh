#!/usr/bin/env bash
# Monitor selfplay progress across cluster and trigger import when threshold reached
# Usage: ./scripts/monitor_selfplay_progress.sh [--import-threshold 1000] [--interval 60]

set -e
cd "$(dirname "$0")/.."

# Configuration
IMPORT_THRESHOLD=${IMPORT_THRESHOLD:-1000}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
LOG_FILE="logs/selfplay_monitor_$(date +%Y%m%d_%H%M%S).log"
IMPORT_TRIGGERED=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --import-threshold) IMPORT_THRESHOLD="$2"; shift 2 ;;
        --interval) CHECK_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p logs

# All hosts
GH200_HOSTS="100.123.183.70 100.104.34.73 100.88.35.19 100.75.84.47 100.88.176.74 100.104.165.116 100.104.126.58 100.65.88.62 100.99.27.56 100.96.142.42 100.76.145.60"
LAMBDA_HOSTS="100.91.25.13 100.78.101.123 100.97.104.89"
VAST_HOSTS="100.118.201.85 100.100.242.64 100.74.154.36 100.75.98.13"

SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_game_counts() {
    local host=$1
    local user=$2
    ssh $SSH_OPTS -i ~/.ssh/id_cluster ${user}@${host} '
        for config in hexagonal_2p hexagonal_3p hexagonal_4p square19_2p square19_3p square19_4p; do
            f=~/ringrift/ai-service/data/games/gpu_selfplay/${config}/games.jsonl
            if [ -f "$f" ]; then
                count=$(wc -l < "$f")
            else
                count=0
            fi
            echo "${config}:${count}"
        done
    ' 2>/dev/null || echo "ERROR:0"
}

check_cluster_progress() {
    log "=== Checking cluster progress ==="

    # Initialize totals
    total_hex2p=0; total_hex3p=0; total_hex4p=0
    total_sq19_2p=0; total_sq19_3p=0; total_sq19_4p=0
    active_nodes=0

    # Check GH200 nodes
    for host in $GH200_HOSTS; do
        result=$(get_game_counts "$host" "ubuntu")
        if [[ "$result" != "ERROR:0" ]]; then
            ((active_nodes++)) || true
            while IFS=: read -r config count; do
                case $config in
                    hexagonal_2p) total_hex2p=$((total_hex2p + count)) ;;
                    hexagonal_3p) total_hex3p=$((total_hex3p + count)) ;;
                    hexagonal_4p) total_hex4p=$((total_hex4p + count)) ;;
                    square19_2p) total_sq19_2p=$((total_sq19_2p + count)) ;;
                    square19_3p) total_sq19_3p=$((total_sq19_3p + count)) ;;
                    square19_4p) total_sq19_4p=$((total_sq19_4p + count)) ;;
                esac
            done <<< "$result"
        fi
    done

    # Check Lambda nodes
    for host in $LAMBDA_HOSTS; do
        result=$(get_game_counts "$host" "ubuntu")
        if [[ "$result" != "ERROR:0" ]]; then
            ((active_nodes++)) || true
            while IFS=: read -r config count; do
                case $config in
                    hexagonal_2p) total_hex2p=$((total_hex2p + count)) ;;
                    hexagonal_3p) total_hex3p=$((total_hex3p + count)) ;;
                    hexagonal_4p) total_hex4p=$((total_hex4p + count)) ;;
                    square19_2p) total_sq19_2p=$((total_sq19_2p + count)) ;;
                    square19_3p) total_sq19_3p=$((total_sq19_3p + count)) ;;
                    square19_4p) total_sq19_4p=$((total_sq19_4p + count)) ;;
                esac
            done <<< "$result"
        fi
    done

    # Check Vast nodes
    for host in $VAST_HOSTS; do
        result=$(get_game_counts "$host" "root")
        if [[ "$result" != "ERROR:0" ]]; then
            ((active_nodes++)) || true
            while IFS=: read -r config count; do
                case $config in
                    hexagonal_2p) total_hex2p=$((total_hex2p + count)) ;;
                    hexagonal_3p) total_hex3p=$((total_hex3p + count)) ;;
                    hexagonal_4p) total_hex4p=$((total_hex4p + count)) ;;
                    square19_2p) total_sq19_2p=$((total_sq19_2p + count)) ;;
                    square19_3p) total_sq19_3p=$((total_sq19_3p + count)) ;;
                    square19_4p) total_sq19_4p=$((total_sq19_4p + count)) ;;
                esac
            done <<< "$result"
        fi
    done

    # Print summary
    log "Active nodes: $active_nodes/18"
    log "  hexagonal_2p: $total_hex2p"
    log "  hexagonal_3p: $total_hex3p"
    log "  hexagonal_4p: $total_hex4p"
    log "  square19_2p: $total_sq19_2p"
    log "  square19_3p: $total_sq19_3p"
    log "  square19_4p: $total_sq19_4p"

    total_games=$((total_hex2p + total_hex3p + total_hex4p + total_sq19_2p + total_sq19_3p + total_sq19_4p))
    log "Total games: $total_games"

    # Find minimum
    min_count=$total_hex2p
    for c in $total_hex3p $total_hex4p $total_sq19_2p $total_sq19_3p $total_sq19_4p; do
        if [[ $c -lt $min_count ]]; then
            min_count=$c
        fi
    done
    log "Min config count: $min_count (threshold: $IMPORT_THRESHOLD)"

    # Check threshold
    if [[ $min_count -ge $IMPORT_THRESHOLD ]] && [[ "$IMPORT_TRIGGERED" == "false" ]]; then
        log ">>> THRESHOLD REACHED! Triggering import... <<<"
        IMPORT_TRIGGERED=true
        trigger_import
    fi
}

trigger_import() {
    log "Starting import process..."

    # Sync games from cluster
    log "Syncing games from cluster..."
    ./scripts/sync_cluster_games.sh 2>&1 | tee -a "$LOG_FILE"

    # Import to training DB
    log "Importing to training database..."
    source venv/bin/activate 2>/dev/null || true

    for config in hexagonal_2p hexagonal_3p hexagonal_4p square19_2p square19_3p square19_4p; do
        board=$(echo $config | cut -d_ -f1)
        players=$(echo $config | cut -d_ -f2 | tr -d 'p')

        log "Importing $config..."
        python3 scripts/import_gpu_selfplay_to_db.py \
            --input-dir "data/games/gpu_selfplay/$config" \
            --output-db "data/training/gpu_${config}.db" \
            --board-type "$board" \
            --num-players "$players" \
            2>&1 | tee -a "$LOG_FILE" || log "Warning: Import failed for $config"
    done

    log "Import complete!"
}

# Main loop
log "=============================================="
log "Selfplay Progress Monitor Started"
log "=============================================="
log "Import threshold: $IMPORT_THRESHOLD games per config"
log "Check interval: ${CHECK_INTERVAL}s"
log "Log file: $LOG_FILE"
log ""

while true; do
    check_cluster_progress

    if [[ "$IMPORT_TRIGGERED" == "true" ]]; then
        log "Import triggered. Exiting monitor."
        break
    fi

    log "Sleeping ${CHECK_INTERVAL}s..."
    log ""
    sleep $CHECK_INTERVAL
done
