#!/bin/bash
# Sync selfplay data from GH200s to local, then to H100 for training

set -e
LOCAL_DATA_DIR="/Users/armand/Development/RingRift/ai-service/data/gh200_sync"
mkdir -p "$LOCAL_DATA_DIR"

echo "=== Syncing GH200 selfplay data ==="

# GH200 hosts
GH200_HOSTS=(
    "ubuntu@192.222.51.29"
    "ubuntu@192.222.51.167"
    "ubuntu@192.222.51.162"
    "ubuntu@192.222.58.122"
)

for host in "${GH200_HOSTS[@]}"; do
    node_name=$(echo "$host" | cut -d@ -f2 | tr '.' '_')
    echo "Syncing from $host..."
    
    # Sync GPU selfplay data
    rsync -avz --progress \
        "$host:~/ringrift/ai-service/data/selfplay/gpu/" \
        "$LOCAL_DATA_DIR/${node_name}/" 2>/dev/null || echo "  Warning: sync failed for $host"
done

echo "=== Sync complete ==="
echo "Data location: $LOCAL_DATA_DIR"
ls -la "$LOCAL_DATA_DIR"
