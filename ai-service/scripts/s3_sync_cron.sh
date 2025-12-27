#!/bin/bash
# S3 Backup Cron Script for RingRift AI
# Runs hourly to sync models, databases, and state to S3
#
# This script is called by cron: 0 * * * * ~/Development/RingRift/ai-service/scripts/s3_sync_cron.sh
#
# Backs up:
# - Best models (all configs)
# - NNUE models
# - Game databases (training data)
# - Promotion history and state

set -e

# Add custom paths for AWS CLI (cron environment has minimal PATH)
export PATH="/Users/armand/local/aws-cli:/usr/local/bin:/usr/bin:/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$AI_SERVICE_ROOT/logs/s3_backup.log"

# Ensure log directory exists
mkdir -p "$AI_SERVICE_ROOT/logs"

# Timestamp for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "=== Starting S3 Backup ==="

# Change to AI service root
cd "$AI_SERVICE_ROOT"

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check AWS CLI is available
if ! command -v aws &> /dev/null; then
    log "ERROR: aws CLI not found. Please install AWS CLI."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    log "ERROR: AWS credentials not configured or expired."
    exit 1
fi

# Run the S3 backup script
PYTHONPATH=. python3 scripts/s3_backup.py 2>&1 | while read line; do
    log "$line"
done

BACKUP_STATUS=${PIPESTATUS[0]}

if [ $BACKUP_STATUS -eq 0 ]; then
    log "=== S3 Backup Completed Successfully ==="
else
    log "=== S3 Backup Failed with status $BACKUP_STATUS ==="
fi

# Rotate logs if too large (>10MB)
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 10485760 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
    log "Log rotated"
fi

exit $BACKUP_STATUS
