#!/bin/bash
# Collect Gumbel MCTS selfplay data from Lambda GH200 cluster
# Aggregates data from all nodes for local training

set -e

# Lambda node IPs by board type
SQUARE8_NODES="100.65.88.62 100.79.109.120"
SQUARE19_NODES="100.117.177.83 100.99.27.56"
HEXAGONAL_NODES="100.97.98.26 100.66.65.33"
HEX8_NODES="100.104.126.58 100.83.234.82"

ALL_NODES="$SQUARE8_NODES $SQUARE19_NODES $HEXAGONAL_NODES $HEX8_NODES"

# Local data directory
LOCAL_DATA_DIR="/Users/armand/Development/RingRift/ai-service/data/selfplay_collected"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

collect_from_node() {
    local ip=$1
    local board=$2
    local output_file="${LOCAL_DATA_DIR}/gumbel_${board}_2p_${ip//./}_${TIMESTAMP}.jsonl"

    log "Collecting from $ip ($board)..."

    # Check if node is reachable
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@${ip} "echo ok" >/dev/null 2>&1; then
        log "WARNING: Cannot reach $ip, skipping"
        return 1
    fi

    # Get remote file info
    local remote_file="~/ringrift/ai-service/data/selfplay/gumbel_${board}_2p.jsonl"
    local remote_lines=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ${remote_file} 2>/dev/null | awk '{print \$1}'" 2>/dev/null)

    if [ -z "$remote_lines" ] || [ "$remote_lines" = "0" ]; then
        log "  No data on $ip"
        return 0
    fi

    log "  Found $remote_lines games on $ip"

    # Copy file locally
    scp -o ConnectTimeout=30 ubuntu@${ip}:${remote_file} "${output_file}" 2>/dev/null

    local local_lines=$(wc -l < "${output_file}")
    log "  Copied $local_lines games to ${output_file##*/}"

    return 0
}

aggregate_data() {
    local board=$1
    local aggregate_file="${LOCAL_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl"

    log "Aggregating $board data..."

    # Find all files for this board type
    local files=$(ls ${LOCAL_DATA_DIR}/gumbel_${board}_2p_*.jsonl 2>/dev/null | grep -v aggregate || true)

    if [ -z "$files" ]; then
        log "  No $board data files found"
        return 0
    fi

    # Concatenate all files (removing duplicates based on game hash if available)
    cat $files | sort -u > "${aggregate_file}.tmp"
    mv "${aggregate_file}.tmp" "${aggregate_file}"

    local total=$(wc -l < "${aggregate_file}")
    log "  Aggregated: $total unique games in $aggregate_file"
}

show_status() {
    log "=== Current Cluster Status ==="
    for ip in $ALL_NODES; do
        local board=""
        case $ip in
            100.65.88.62|100.79.109.120) board="square8" ;;
            100.117.177.83|100.99.27.56) board="square19" ;;
            100.97.98.26|100.66.65.33) board="hexagonal" ;;
            100.104.126.58|100.83.234.82) board="hex8" ;;
        esac

        local procs=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | wc -l" 2>/dev/null || echo "?")
        local games=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_2p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")

        echo "  $ip ($board): procs=$procs games=$games"
    done
}

main() {
    mkdir -p "$LOCAL_DATA_DIR"

    case "${1:-collect}" in
        status)
            show_status
            ;;
        collect)
            log "=== Collecting Gumbel MCTS Selfplay Data ==="

            # Collect from each node
            for ip in $SQUARE8_NODES; do
                collect_from_node "$ip" "square8" || true
            done

            for ip in $SQUARE19_NODES; do
                collect_from_node "$ip" "square19" || true
            done

            for ip in $HEXAGONAL_NODES; do
                collect_from_node "$ip" "hexagonal" || true
            done

            for ip in $HEX8_NODES; do
                collect_from_node "$ip" "hex8" || true
            done

            log ""
            log "=== Aggregating Data ==="
            aggregate_data "square8"
            aggregate_data "square19"
            aggregate_data "hexagonal"
            aggregate_data "hex8"

            log ""
            log "=== Summary ==="
            for board in square8 square19 hexagonal hex8; do
                local agg="${LOCAL_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl"
                if [ -f "$agg" ]; then
                    local count=$(wc -l < "$agg")
                    echo "  $board: $count games"
                else
                    echo "  $board: 0 games"
                fi
            done
            ;;
        loop)
            log "Starting continuous collection (every 30 minutes)..."
            while true; do
                main collect
                log "Sleeping 30 minutes until next collection..."
                sleep 1800
            done
            ;;
        *)
            echo "Usage: $0 [status|collect|loop]"
            echo "  status  - Show cluster status"
            echo "  collect - Collect data once"
            echo "  loop    - Collect continuously every 30 minutes"
            exit 1
            ;;
    esac
}

main "$@"
