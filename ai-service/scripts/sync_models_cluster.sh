#!/bin/bash
# Sync latest models across the cluster
# Run from any node with SSH access to all others
# Add to cron: */30 * * * * /path/to/sync_models_cluster.sh >> /tmp/model_sync.log 2>&1

set -e

SOURCE_NODE="lambda-2xh100"
TARGET_NODES="lambda-gh200-i lambda-gh200-k-ts lambda-gh200-l-ts"
MODELS_DIR="ringrift/ai-service/models"
TMP_DIR="/tmp/model_sync_$$"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting model sync..."

# Create temp dir
mkdir -p "$TMP_DIR"

# Download latest best models from source
echo "  Downloading from $SOURCE_NODE..."
rsync -az --include='ringrift_best*.pth' \
         --include='hex*.pth' \
         --include='sq19*.pth' \
         --include='sq8_*_nn_baseline.pth' \
         --exclude='*' \
         "${SOURCE_NODE}:~/${MODELS_DIR}/" "$TMP_DIR/" 2>/dev/null || true

MODEL_COUNT=$(ls "$TMP_DIR"/*.pth 2>/dev/null | wc -l)
echo "  Found $MODEL_COUNT models to sync"

if [ "$MODEL_COUNT" -gt 0 ]; then
    for target in $TARGET_NODES; do
        echo "  Syncing to $target..."
        rsync -az "$TMP_DIR/" "${target}:~/${MODELS_DIR}/" 2>/dev/null || echo "    (failed)"
    done
fi

# Cleanup
rm -rf "$TMP_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model sync complete"
