#!/bin/bash
# Local Streaming Relay Script
# Pulls data incrementally from Vast.ai and pushes to Lambda in real-time
#
# Usage: ./local_streaming_relay.sh [--sync-interval 30] [--instance vast4x5090]
#
# This script runs on your LOCAL machine and:
# 1. Monitors JSONL files on a Vast.ai instance
# 2. Pulls new content incrementally
# 3. Pushes to Lambda in real-time

set -e

# Configuration
SYNC_INTERVAL=${SYNC_INTERVAL:-15}  # Seconds between syncs
LAMBDA_HOST="lambda-gpu"
LAMBDA_DATA_DIR="/home/ubuntu/ringrift/ai-service/data/collected/streaming"
INSTANCE=${INSTANCE:-"vast4x5090"}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sync-interval) SYNC_INTERVAL="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Instance SSH commands (bash 3.2 compatible - no associative arrays)
get_ssh_cmd() {
    case "$1" in
        vast4x5090) echo "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14133 root@ssh3.vast.ai" ;;
        vast2x5090) echo "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 14861 root@ssh6.vast.ai" ;;
        vast3090) echo "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p 15093 root@ssh7.vast.ai" ;;
        *) echo "" ;;
    esac
}

SSH_CMD=$(get_ssh_cmd "$INSTANCE")
if [ -z "$SSH_CMD" ]; then
    echo "Unknown instance: $INSTANCE"
    echo "Available: vast4x5090, vast2x5090, vast3090"
    exit 1
fi

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Local Streaming Relay ==="
log "Instance: $INSTANCE"
log "Sync interval: ${SYNC_INTERVAL}s"
log "Lambda target: $LAMBDA_HOST:$LAMBDA_DATA_DIR"

# Ensure Lambda directory exists
ssh $LAMBDA_HOST "mkdir -p $LAMBDA_DATA_DIR" 2>/dev/null || log "Warning: Could not create Lambda directory"

# Track sync positions for each file (local tracking)
SYNC_STATE_DIR="/tmp/streaming_state_${INSTANCE}"
mkdir -p "$SYNC_STATE_DIR"

# Cleanup on exit
cleanup() {
    log "Shutting down..."
    rm -rf "$SYNC_STATE_DIR"
}
trap cleanup EXIT

# Main sync loop
log "Starting sync loop..."
TOTAL_GAMES=0

while true; do
    # Get list of all JSONL files on Vast.ai (batch_*, extra_*, c*_w*, etc.)
    FILES=$($SSH_CMD "ls ~/ringrift/ai-service/logs/selfplay/*.jsonl 2>/dev/null" 2>&1 | grep -v "Welcome\|Have fun\|No such file" || true)

    if [ -n "$FILES" ]; then
        for remote_file in $FILES; do
            filename=$(basename "$remote_file")
            state_file="$SYNC_STATE_DIR/${filename}.pos"

            # Get current remote file size
            remote_size=$($SSH_CMD "stat -c%s $remote_file 2>/dev/null || echo 0" 2>&1 | grep -E '^[0-9]+$' | tail -1)

            # Get last synced position
            last_pos=$(cat "$state_file" 2>/dev/null || echo "0")

            if [ "$remote_size" -gt "$last_pos" ] 2>/dev/null; then
                new_bytes=$((remote_size - last_pos))

                # Pull new content and push to Lambda
                $SSH_CMD "tail -c +$((last_pos + 1)) $remote_file" 2>/dev/null | \
                    ssh $LAMBDA_HOST "cat >> ${LAMBDA_DATA_DIR}/${INSTANCE}_${filename}" 2>/dev/null

                if [ $? -eq 0 ]; then
                    echo "$remote_size" > "$state_file"
                    # Count new lines
                    new_games=$($SSH_CMD "tail -c +$((last_pos + 1)) $remote_file | wc -l" 2>&1 | grep -E '^[0-9]+' | head -1 || echo "0")
                    TOTAL_GAMES=$((TOTAL_GAMES + new_games))
                    log "Synced $new_bytes bytes (~$new_games games) from $filename. Total: $TOTAL_GAMES games"
                fi
            fi
        done
    fi

    sleep "$SYNC_INTERVAL"
done
