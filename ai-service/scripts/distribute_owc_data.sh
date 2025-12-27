#!/bin/bash
# Distribute game data from OWC drive (via mac-studio) to training nodes
# Uses this machine as a relay since mac-studio isn't on Tailscale
#
# Usage: ./distribute_owc_data.sh [target_host] [board_type]
# Example: ./distribute_owc_data.sh nebius-h100-3 hex8
#          ./distribute_owc_data.sh all  # distribute to all training nodes

set -e

# Mac Studio OWC data paths
MAC_STUDIO="mac-studio"
OWC_GAMES="/Volumes/RingRift-Data/canonical_games"
OWC_MODELS="/Volumes/RingRift-Data/canonical_models"
OWC_DATA="/Volumes/RingRift-Data/canonical_data"

# Target nodes with their connection details
declare -A NODES
NODES["nebius-h100-3"]="ubuntu@89.169.110.128:~/ringrift/ai-service:~/.ssh/id_cluster"
NODES["nebius-h100-1"]="ubuntu@89.169.111.139:~/ringrift/ai-service:~/.ssh/id_cluster"
NODES["vultr-a100"]="root@208.167.249.164:/root/ringrift/ai-service:~/.ssh/id_ed25519"
NODES["runpod-h100"]="root@102.210.171.65:/workspace/ringrift/ai-service:~/.runpod/ssh/RunPod-Key-Go:30755"
NODES["vast-29118471"]="root@ssh8.vast.ai:~/ringrift/ai-service:~/.ssh/id_cluster:38470"

# Parse target and board type
TARGET="${1:-all}"
BOARD_TYPE="${2:-}"

sync_to_node() {
    local node_name="$1"
    local node_info="${NODES[$node_name]}"

    if [ -z "$node_info" ]; then
        echo "Unknown node: $node_name"
        return 1
    fi

    # Parse connection info
    IFS=':' read -r user_host remote_path ssh_key port <<< "$node_info"
    port="${port:-22}"

    echo "=== Syncing to $node_name ==="
    echo "  User@Host: $user_host"
    echo "  Remote path: $remote_path"

    # Build SSH options
    SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -i $ssh_key"
    [ "$port" != "22" ] && SSH_OPTS="$SSH_OPTS -p $port"

    RSYNC_OPTS="-avz --progress"
    [ "$port" != "22" ] && RSYNC_OPTS="$RSYNC_OPTS -e 'ssh $SSH_OPTS'"

    # Sync canonical databases (filtered by board type if specified)
    echo "  Syncing databases..."
    if [ -n "$BOARD_TYPE" ]; then
        rsync $RSYNC_OPTS -e "ssh $SSH_OPTS" \
            "$MAC_STUDIO:$OWC_GAMES/canonical_${BOARD_TYPE}*.db" \
            "$user_host:$remote_path/data/games/" 2>/dev/null || true
    else
        rsync $RSYNC_OPTS -e "ssh $SSH_OPTS" \
            "$MAC_STUDIO:$OWC_GAMES/canonical_*.db" \
            "$user_host:$remote_path/data/games/" 2>/dev/null || true
    fi

    # Sync models
    echo "  Syncing models..."
    if [ -n "$BOARD_TYPE" ]; then
        rsync $RSYNC_OPTS -e "ssh $SSH_OPTS" \
            "$MAC_STUDIO:$OWC_MODELS/canonical_${BOARD_TYPE}*.pth" \
            "$user_host:$remote_path/models/" 2>/dev/null || true
    else
        rsync $RSYNC_OPTS -e "ssh $SSH_OPTS" \
            "$MAC_STUDIO:$OWC_MODELS/canonical_*.pth" \
            "$user_host:$remote_path/models/" 2>/dev/null || true
    fi

    echo "  Done: $node_name"
}

if [ "$TARGET" = "all" ]; then
    for node in "${!NODES[@]}"; do
        sync_to_node "$node" &
    done
    wait
    echo "=== All nodes synced ==="
else
    sync_to_node "$TARGET"
fi
