#!/bin/bash
# Sync completed models from cluster nodes to coordinator
# Run periodically via: while true; do ./scripts/sync_cluster_models.sh; sleep 300; done

MODELS_DIR="/Users/armand/Development/RingRift/ai-service/models"
SSH_KEY="$HOME/.ssh/id_cluster"
LOG_FILE="/Users/armand/Development/RingRift/ai-service/logs/model_sync.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

sync_from_node() {
    local name=$1
    local user=$2
    local host=$3
    local port=${4:-22}
    local remote_path=$5
    
    log "Syncing models from $name..."
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        "$user@$host:$remote_path/models/*_cluster.pth" \
        "$MODELS_DIR/" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log "  ✅ $name sync complete"
    else
        log "  ⚠️ $name sync failed or no new models"
    fi
}

log "=== Starting model sync from cluster ==="

# Sync from each cluster node
sync_from_node "nebius-h100-3" "ubuntu" "89.169.111.139" "22" "~/ringrift/ai-service"
sync_from_node "nebius-backbone-1" "ubuntu" "89.169.112.47" "22" "~/ringrift/ai-service"
sync_from_node "vultr-a100" "root" "208.167.249.164" "22" "/root/ringrift/ai-service"

# List synced models
log "=== Synced cluster models ==="
ls -la "$MODELS_DIR"/*_cluster.pth 2>/dev/null | while read line; do
    log "  $line"
done

log "=== Sync complete ==="
