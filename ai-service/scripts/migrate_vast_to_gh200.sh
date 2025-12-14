#!/bin/bash
# Migrate all Vast.ai selfplay data to GH200 shared NFS
# Run from local machine (has SSH access to all nodes)

set -e
SHARED_NFS="/lambda/nfs/RingRift/vast_migration"
GH200_HOST="ubuntu@192.222.51.29"
LOCAL_STAGING="/tmp/vast_migration"

mkdir -p "$LOCAL_STAGING"

VAST_HOSTS=(
    "vast-5090-quad:211.72.13.202:45875"
    "vast-5090-a:104.188.118.187:45180"
    "vast-5090-b:77.104.167.149:50612"
    "vast-3090-a:79.116.93.241:47070"
    "vast-3080-dual:211.21.106.81:30847"
    "vast-3070-a:211.21.106.81:35066"
    "vast-3060ti:116.102.207.205:24469"
    "vast-3070-b:211.21.106.81:36401"
    "vast-3090-b:79.112.1.66:20153"
)

echo "=== Creating destination on GH200 shared NFS ==="
ssh $GH200_HOST "mkdir -p $SHARED_NFS"

for entry in "${VAST_HOSTS[@]}"; do
    name=$(echo "$entry" | cut -d: -f1)
    host=$(echo "$entry" | cut -d: -f2)
    port=$(echo "$entry" | cut -d: -f3)
    
    echo
    echo "=== Migrating $name ($host:$port) ==="
    
    # Create local staging dir
    mkdir -p "$LOCAL_STAGING/$name"
    
    # Sync from Vast to local
    echo "  Pulling from Vast..."
    rsync -avz --progress \
        -e "ssh -p $port -o ConnectTimeout=10 -o StrictHostKeyChecking=no" \
        "root@$host:~/ringrift/ai-service/data/" \
        "$LOCAL_STAGING/$name/" 2>/dev/null || echo "  Warning: sync failed for $name"
    
    # Push to GH200 shared NFS
    echo "  Pushing to GH200 shared NFS..."
    rsync -avz --progress \
        "$LOCAL_STAGING/$name/" \
        "$GH200_HOST:$SHARED_NFS/$name/" 2>/dev/null || echo "  Warning: push failed for $name"
    
    echo "  Done with $name"
done

echo
echo "=== Migration complete ==="
echo "Data location: $GH200_HOST:$SHARED_NFS"
ssh $GH200_HOST "du -sh $SHARED_NFS/*"
