#!/bin/bash
# Automated S3 backup for Lambda nodes.
# Backs up game databases and selfplay data incrementally.
# Uses aws s3 sync (only uploads changed files) to minimize bandwidth.
#
# Install: crontab -e → 0 */4 * * * ~/ringrift/ai-service/scripts/s3_data_backup.sh >> /tmp/s3_backup.log 2>&1
#
# Mar 24, 2026: Created after discovering 153GB of gauntlet data and
# 39GB+ of selfplay data per node were at risk of loss.

set -e
NODE_ID="${NODE_ID:-$(hostname)}"
BUCKET="ringrift-models-20251214"
AI_DIR="${HOME}/ringrift/ai-service"
LOG_PREFIX="[s3-backup $(date +%H:%M)]"

# Skip if no AWS credentials
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "$LOG_PREFIX ERROR: No AWS credentials"
    exit 1
fi

echo "$LOG_PREFIX Starting backup for $NODE_ID"

# 1. Back up game databases (selfplay + gauntlet)
if [ -d "$AI_DIR/data/games" ]; then
    echo "$LOG_PREFIX Syncing game DBs..."
    aws s3 sync "$AI_DIR/data/games/" "s3://$BUCKET/nodes/$NODE_ID/games/" \
        --storage-class STANDARD_IA \
        --exclude "*.db-journal" \
        --exclude "*.db-wal" \
        --exclude "*.db-shm" \
        --size-only \
        --quiet
fi

# 2. Back up models (canonical + candidate)
if [ -d "$AI_DIR/models" ]; then
    echo "$LOG_PREFIX Syncing models..."
    aws s3 sync "$AI_DIR/models/" "s3://$BUCKET/nodes/$NODE_ID/models/" \
        --storage-class STANDARD_IA \
        --include "*.pth" \
        --include "*.sha256" \
        --exclude "*" \
        --size-only \
        --quiet
fi

# 3. Back up gauntlet results
if [ -d "$AI_DIR/data/gauntlet_results" ]; then
    echo "$LOG_PREFIX Syncing gauntlet results..."
    aws s3 sync "$AI_DIR/data/gauntlet_results/" "s3://$BUCKET/nodes/$NODE_ID/gauntlet_results/" \
        --storage-class STANDARD_IA \
        --quiet
fi

echo "$LOG_PREFIX Backup complete for $NODE_ID"
