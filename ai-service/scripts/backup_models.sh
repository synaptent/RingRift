#!/bin/bash
# Backup models from Lambda to local storage
# Usage: ./scripts/backup_models.sh [--all]
#   --all: Backup all models (default: only latest 10)

set -e

LAMBDA_HOST="ubuntu@192.222.53.22"
REMOTE_MODELS_DIR="~/ringrift/ai-service/models"
LOCAL_BACKUP_DIR="/Users/armand/Development/RingRift/ai-service/backups/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOCAL_BACKUP_DIR"

echo "[$(date)] Starting model backup from Lambda..."

if [ "$1" == "--all" ]; then
    echo "Backing up ALL models..."
    rsync -avz --progress \
        "$LAMBDA_HOST:$REMOTE_MODELS_DIR/*.pth" \
        "$LOCAL_BACKUP_DIR/"
else
    echo "Backing up latest 10 models..."
    # Get list of latest 10 models
    LATEST_MODELS=$(ssh "$LAMBDA_HOST" "ls -t $REMOTE_MODELS_DIR/*.pth 2>/dev/null | head -10")

    for model in $LATEST_MODELS; do
        model_name=$(basename "$model")
        if [ ! -f "$LOCAL_BACKUP_DIR/$model_name" ]; then
            echo "  Downloading: $model_name"
            scp "$LAMBDA_HOST:$model" "$LOCAL_BACKUP_DIR/"
        else
            echo "  Already exists: $model_name"
        fi
    done
fi

# Count backups
BACKUP_COUNT=$(ls -1 "$LOCAL_BACKUP_DIR"/*.pth 2>/dev/null | wc -l)
echo "[$(date)] Backup complete. Total models in backup: $BACKUP_COUNT"

# Show latest models
echo ""
echo "Latest backed up models:"
ls -lht "$LOCAL_BACKUP_DIR"/*.pth 2>/dev/null | head -5
