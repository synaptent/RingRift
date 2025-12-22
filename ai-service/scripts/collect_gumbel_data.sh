#!/bin/bash
# Collect Gumbel MCTS selfplay data from Lambda GH200 cluster
# Aggregates data from all nodes for local training
# Supports 2P, 3P, and 4P games across all board types

set -e

# ============================================================================
# NODE CONFIGURATION - 22 nodes total
# ============================================================================

# 2P nodes (7 nodes)
SQUARE8_2P_NODES="100.65.88.62 100.79.109.120"
SQUARE19_2P_NODES="100.117.177.83 100.99.27.56"
HEXAGONAL_2P_NODES="100.66.65.33"
HEX8_2P_NODES="100.104.126.58 100.83.234.82"

# 3P nodes (8 nodes)
SQUARE8_3P_NODES="100.123.183.70 100.88.35.19"
SQUARE19_3P_NODES="100.75.84.47 100.88.176.74"
HEXAGONAL_3P_NODES="100.104.165.116 100.96.142.42"
HEX8_3P_NODES="100.76.145.60 100.85.106.113"

# 4P nodes (7 nodes)
SQUARE8_4P_NODES="100.106.0.3 100.81.5.33"
SQUARE19_4P_NODES="100.78.101.123 100.97.104.89"
HEXAGONAL_4P_NODES="100.91.25.13 100.101.45.4"
HEX8_4P_NODES="100.78.55.103"

ALL_2P_NODES="$SQUARE8_2P_NODES $SQUARE19_2P_NODES $HEXAGONAL_2P_NODES $HEX8_2P_NODES"
ALL_3P_NODES="$SQUARE8_3P_NODES $SQUARE19_3P_NODES $HEXAGONAL_3P_NODES $HEX8_3P_NODES"
ALL_4P_NODES="$SQUARE8_4P_NODES $SQUARE19_4P_NODES $HEXAGONAL_4P_NODES $HEX8_4P_NODES"
ALL_NODES="$ALL_2P_NODES $ALL_3P_NODES $ALL_4P_NODES"

# Local data directory
LOCAL_DATA_DIR="/Users/armand/Development/RingRift/ai-service/data/selfplay_collected"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

collect_from_node() {
    local ip=$1
    local board=$2
    local players=$3
    local output_file="${LOCAL_DATA_DIR}/gumbel_${board}_${players}p_${ip//./}_${TIMESTAMP}.jsonl"

    log "Collecting from $ip ($board ${players}P)..."

    # Check if node is reachable
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@${ip} "echo ok" >/dev/null 2>&1; then
        log "WARNING: Cannot reach $ip, skipping"
        return 1
    fi

    # Get remote file info
    local remote_file="~/ringrift/ai-service/data/selfplay/gumbel_${board}_${players}p.jsonl"
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
    local players=$2
    local aggregate_file="${LOCAL_DATA_DIR}/gumbel_${board}_${players}p_aggregate.jsonl"

    log "Aggregating $board ${players}P data..."

    # Find all files for this board type and player count
    local files=$(ls ${LOCAL_DATA_DIR}/gumbel_${board}_${players}p_*.jsonl 2>/dev/null | grep -v aggregate || true)

    if [ -z "$files" ]; then
        log "  No $board ${players}P data files found"
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

    log ""
    log "--- 2P Nodes (7) ---"
    for ip in $ALL_2P_NODES; do
        local board=""
        case $ip in
            100.65.88.62|100.79.109.120) board="square8" ;;
            100.117.177.83|100.99.27.56) board="square19" ;;
            100.66.65.33) board="hexagonal" ;;
            100.104.126.58|100.83.234.82) board="hex8" ;;
        esac
        local procs=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | head -1" 2>/dev/null || echo "")
        local games=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_2p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
        [ -n "$procs" ] && status="✓" || status="✗"
        printf "  %-18s %-10s %s games=%s\n" "$ip" "($board)" "$status" "$games"
    done

    log ""
    log "--- 3P Nodes (8) ---"
    for ip in $ALL_3P_NODES; do
        local board=""
        case $ip in
            100.123.183.70|100.88.35.19) board="square8" ;;
            100.75.84.47|100.88.176.74) board="square19" ;;
            100.104.165.116|100.96.142.42) board="hexagonal" ;;
            100.76.145.60|100.85.106.113) board="hex8" ;;
        esac
        local procs=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | head -1" 2>/dev/null || echo "")
        local games=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_3p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
        [ -n "$procs" ] && status="✓" || status="✗"
        printf "  %-18s %-10s %s games=%s\n" "$ip" "($board)" "$status" "$games"
    done

    log ""
    log "--- 4P Nodes (7) ---"
    for ip in $ALL_4P_NODES; do
        local board=""
        case $ip in
            100.106.0.3|100.81.5.33) board="square8" ;;
            100.78.101.123|100.97.104.89) board="square19" ;;
            100.91.25.13|100.101.45.4) board="hexagonal" ;;
            100.78.55.103) board="hex8" ;;
        esac
        local procs=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | head -1" 2>/dev/null || echo "")
        local games=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_${board}_4p.jsonl 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "0")
        [ -n "$procs" ] && status="✓" || status="✗"
        printf "  %-18s %-10s %s games=%s\n" "$ip" "($board)" "$status" "$games"
    done
}

main() {
    mkdir -p "$LOCAL_DATA_DIR"

    case "${1:-collect}" in
        status)
            show_status
            ;;
        collect)
            log "=== Collecting Gumbel MCTS Selfplay Data (2P/3P/4P) ==="

            # ---- 2P Collection ----
            log ""
            log "--- Collecting 2P Games ---"
            for ip in $SQUARE8_2P_NODES; do collect_from_node "$ip" "square8" 2 || true; done
            for ip in $SQUARE19_2P_NODES; do collect_from_node "$ip" "square19" 2 || true; done
            for ip in $HEXAGONAL_2P_NODES; do collect_from_node "$ip" "hexagonal" 2 || true; done
            for ip in $HEX8_2P_NODES; do collect_from_node "$ip" "hex8" 2 || true; done

            # ---- 3P Collection ----
            log ""
            log "--- Collecting 3P Games ---"
            for ip in $SQUARE8_3P_NODES; do collect_from_node "$ip" "square8" 3 || true; done
            for ip in $SQUARE19_3P_NODES; do collect_from_node "$ip" "square19" 3 || true; done
            for ip in $HEXAGONAL_3P_NODES; do collect_from_node "$ip" "hexagonal" 3 || true; done
            for ip in $HEX8_3P_NODES; do collect_from_node "$ip" "hex8" 3 || true; done

            # ---- 4P Collection ----
            log ""
            log "--- Collecting 4P Games ---"
            for ip in $SQUARE8_4P_NODES; do collect_from_node "$ip" "square8" 4 || true; done
            for ip in $SQUARE19_4P_NODES; do collect_from_node "$ip" "square19" 4 || true; done
            for ip in $HEXAGONAL_4P_NODES; do collect_from_node "$ip" "hexagonal" 4 || true; done
            for ip in $HEX8_4P_NODES; do collect_from_node "$ip" "hex8" 4 || true; done

            log ""
            log "=== Aggregating Data ==="
            for players in 2 3 4; do
                for board in square8 square19 hexagonal hex8; do
                    aggregate_data "$board" "$players"
                done
            done

            log ""
            log "=== Summary ==="
            printf "%-12s %8s %8s %8s\n" "Board" "2P" "3P" "4P"
            printf "%-12s %8s %8s %8s\n" "--------" "------" "------" "------"
            for board in square8 square19 hexagonal hex8; do
                count_2p=0; count_3p=0; count_4p=0
                [ -f "${LOCAL_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl" ] && count_2p=$(wc -l < "${LOCAL_DATA_DIR}/gumbel_${board}_2p_aggregate.jsonl")
                [ -f "${LOCAL_DATA_DIR}/gumbel_${board}_3p_aggregate.jsonl" ] && count_3p=$(wc -l < "${LOCAL_DATA_DIR}/gumbel_${board}_3p_aggregate.jsonl")
                [ -f "${LOCAL_DATA_DIR}/gumbel_${board}_4p_aggregate.jsonl" ] && count_4p=$(wc -l < "${LOCAL_DATA_DIR}/gumbel_${board}_4p_aggregate.jsonl")
                printf "%-12s %8d %8d %8d\n" "$board" "$count_2p" "$count_3p" "$count_4p"
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
            echo "  status  - Show cluster status (22 nodes)"
            echo "  collect - Collect data once from all nodes"
            echo "  loop    - Collect continuously every 30 minutes"
            exit 1
            ;;
    esac
}

main "$@"
