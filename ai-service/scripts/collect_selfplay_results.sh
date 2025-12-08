#!/bin/bash
# Collect selfplay results from all distributed workers
# Usage: ./scripts/collect_selfplay_results.sh [--mode lan|aws|all] [--output-dir DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_OUTPUT_DIR="$AI_SERVICE_DIR/logs/selfplay/collected"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
MODE="all"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mode lan|aws|all] [--output-dir DIR]"
            echo ""
            echo "Options:"
            echo "  --mode      Worker mode: lan (local macs), aws (cloud), all (default)"
            echo "  --output-dir  Directory to store collected results (default: logs/selfplay/collected)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

echo "=== Selfplay Result Collection ==="
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to collect from a host
collect_from_host() {
    local name="$1"
    local ssh_host="$2"
    local ssh_key="$3"
    local remote_path="$4"
    local ssh_user="${5:-}"

    # Expand ~ in ssh_key
    ssh_key="${ssh_key/#\~/$HOME}"

    echo "Collecting from $name ($ssh_host)..."

    # Build SSH command
    local ssh_opts="-o ConnectTimeout=10 -o BatchMode=yes"
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="$ssh_opts -i $ssh_key"
    fi

    local user_host="$ssh_host"
    if [[ -n "$ssh_user" ]]; then
        user_host="${ssh_user}@${ssh_host}"
    fi

    # Create host-specific subdirectory
    local host_dir="$OUTPUT_DIR/$name"
    mkdir -p "$host_dir"

    # Try to copy JSONL files
    if scp $ssh_opts "${user_host}:${remote_path}/ai-service/logs/selfplay/*.jsonl" "$host_dir/" 2>/dev/null; then
        local count=$(ls -1 "$host_dir"/*.jsonl 2>/dev/null | wc -l)
        echo "  ✓ Collected $count files from $name"
    else
        echo "  ⚠ No files or connection failed for $name"
    fi

    # Also try to copy DB files if they exist
    if scp $ssh_opts "${user_host}:${remote_path}/ai-service/data/games/*.db" "$host_dir/" 2>/dev/null; then
        local db_count=$(ls -1 "$host_dir"/*.db 2>/dev/null | wc -l)
        echo "  ✓ Collected $db_count database files from $name"
    fi
}

# LAN hosts
if [[ "$MODE" == "lan" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== LAN Workers ==="

    # Mac Studio
    collect_from_host "mac-studio" "Mac-Studio.local" "~/.ssh/id_cluster" "~/Development/RingRift"

    # MacBook Pro 64GB
    collect_from_host "mbp-64gb" "MacBook-Pro-3.local" "~/.ssh/id_cluster" "~/Development/RingRift"

    # MacBook Pro 16GB (optional)
    collect_from_host "mbp-16gb" "MacBook-Pro.local" "~/.ssh/id_cluster" "~/Development/RingRift"
fi

# AWS hosts
if [[ "$MODE" == "aws" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== AWS Workers ==="

    # AWS staging
    collect_from_host "aws-staging" "54.198.219.106" "~/.ssh/ringrift-staging-key.pem" "/home/ubuntu/ringrift" "ubuntu"
fi

# Generate summary
echo ""
echo "=== Collection Summary ==="

total_games=0
for host_dir in "$OUTPUT_DIR"/*/; do
    if [[ -d "$host_dir" ]]; then
        host_name=$(basename "$host_dir")
        file_count=$(ls -1 "$host_dir"/*.jsonl 2>/dev/null | wc -l)

        if [[ $file_count -gt 0 ]]; then
            game_count=$(cat "$host_dir"/*.jsonl 2>/dev/null | wc -l)
            total_games=$((total_games + game_count))
            echo "$host_name: $file_count files, $game_count games"
        fi
    fi
done

echo ""
echo "Total games collected: $total_games"
echo "Results saved to: $OUTPUT_DIR"

# Optionally merge all results
if [[ $total_games -gt 0 ]]; then
    merged_file="$OUTPUT_DIR/merged_${TIMESTAMP}.jsonl"
    cat "$OUTPUT_DIR"/*/*.jsonl > "$merged_file" 2>/dev/null || true
    merged_count=$(wc -l < "$merged_file")
    echo "Merged file: $merged_file ($merged_count games)"
fi
