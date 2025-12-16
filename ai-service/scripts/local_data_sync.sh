#!/bin/bash
# RingRift Local Data Sync
# Runs from local machine, syncs game data from all Vast.ai nodes
# Add to crontab: */15 * * * * /path/to/local_data_sync.sh >> /tmp/ringrift_sync.log 2>&1
#
# This script syncs:
#   1. JSONL files → aggregated to selfplay_stats.db (for monitoring)
#   2. Canonical DBs → data/selfplay/remote_sync/ (for training)
#
# NOTE: For training, use data/selfplay/diverse/*.db or data/canonical/*.db
#       NOT the selfplay_stats.db (which lacks game_moves table)

set -e

# Configuration
LOCAL_DIR="/Users/armand/Development/RingRift/ai-service/data/games"
SCRIPT_DIR="/Users/armand/Development/RingRift/ai-service/scripts"
LOG_FILE="/tmp/ringrift_sync.log"

# Disk usage thresholds (consistent with p2p_orchestrator.py and unified_ai_loop.py)
MAX_DISK_USAGE_PERCENT=${RINGRIFT_MAX_DISK_PERCENT:-70}

# Node configurations: name:ip:remote_path
declare -a NODES=(
    "vast-3060ti:100.117.81.49:/root/ringrift/ai-service/data/games"
    "vast-2060s:100.75.98.13:/root/RingRift/ai-service/data/games"
    "vast-3070:100.74.154.36:/root/ringrift/ai-service/data/games"
    "vast-512cpu:100.118.201.85:/workspace/ringrift/ai-service/data/games"
    "vast-4080s:100.79.143.125:/root/ringrift/ai-service/data/games"
    "vast-5070:100.116.197.108:/root/ringrift/ai-service/data/games"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

mkdir -p "$LOCAL_DIR"

# Check disk usage before syncing
DISK_USAGE=$(df -P "$LOCAL_DIR" 2>/dev/null | awk 'NR==2 {gsub(/%/, ""); print $5}')
if [[ "$DISK_USAGE" -ge "$MAX_DISK_USAGE_PERCENT" ]]; then
    log "[ERROR] Disk usage ${DISK_USAGE}% exceeds limit ${MAX_DISK_USAGE_PERCENT}% - aborting sync"
    log "[INFO] Run disk cleanup: python scripts/disk_monitor.py --force"
    exit 1
fi

log "Starting data sync from ${#NODES[@]} nodes..."

total_synced=0
online_nodes=0

for node_entry in "${NODES[@]}"; do
    IFS=':' read -r name ip path <<< "$node_entry"

    # Check if node is online
    if ! timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes "root@$ip" "echo ok" >/dev/null 2>&1; then
        log "  $name: OFFLINE"
        continue
    fi

    online_nodes=$((online_nodes + 1))

    # Create node-specific directory to avoid filename collisions
    node_dir="$LOCAL_DIR/$name"
    mkdir -p "$node_dir"

    # Sync JSONL files
    synced=$(rsync -avz --progress -e "ssh -o ConnectTimeout=10" \
        "root@$ip:$path/*.jsonl" "$node_dir/" 2>/dev/null | grep -c "\.jsonl$" || echo "0")

    if [ "$synced" -gt 0 ]; then
        log "  $name: synced $synced files"
        total_synced=$((total_synced + synced))
    else
        log "  $name: no new files"
    fi
done

log "Sync complete: $online_nodes nodes online, $total_synced files synced"

# Also sync canonical DBs (with game_moves table) for training
log "Syncing canonical training DBs..."
TRAINING_SYNC_DIR="/Users/armand/Development/RingRift/ai-service/data/selfplay/remote_sync"
mkdir -p "$TRAINING_SYNC_DIR"

for node_entry in "${NODES[@]}"; do
    IFS=':' read -r name ip path <<< "$node_entry"

    # Skip if node is offline
    if ! timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes "root@$ip" "echo ok" >/dev/null 2>&1; then
        continue
    fi

    # Sync any DBs that have game_moves table (canonical format)
    node_train_dir="$TRAINING_SYNC_DIR/$name"
    mkdir -p "$node_train_dir"

    # Rsync canonical DBs from selfplay directory
    rsync -avz --progress -e "ssh -o ConnectTimeout=30" \
        --include="*/" --include="*.db" --exclude="*" \
        "root@$ip:${path%/games}/selfplay/" "$node_train_dir/" 2>/dev/null || true

    # Also sync from canonical subdirectory (new canonical selfplay location)
    rsync -avz --progress -e "ssh -o ConnectTimeout=30" \
        "root@$ip:${path%/games}/selfplay/canonical/" "$node_train_dir/canonical/" 2>/dev/null || true

    # Count synced DBs
    db_count=$(find "$node_train_dir" -name "*.db" -type f 2>/dev/null | wc -l)
    if [ "$db_count" -gt 0 ]; then
        log "  $name: synced $db_count canonical DBs"
    fi
done

# Clean null bytes from sparse files
log "Cleaning sparse files..."
python3 << 'CLEAN_EOF'
import os
import sys

data_dir = "/Users/armand/Development/RingRift/ai-service/data/games"
cleaned = 0

for root, dirs, files in os.walk(data_dir):
    for fname in files:
        if fname.endswith('.jsonl'):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'rb') as f:
                    content = f.read()
                if b'\x00' in content:
                    clean_content = content.replace(b'\x00', b'')
                    with open(fpath, 'wb') as f:
                        f.write(clean_content)
                    cleaned += 1
            except Exception as e:
                pass

if cleaned > 0:
    print(f"Cleaned {cleaned} files")
CLEAN_EOF

# Run aggregation
log "Running aggregation..."
cd /Users/armand/Development/RingRift/ai-service
python3 scripts/aggregate_games.py 2>&1 | tail -10

log "=== Sync cycle complete ==="
