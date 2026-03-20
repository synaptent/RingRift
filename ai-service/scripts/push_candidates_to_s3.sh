#!/bin/bash
# push_candidates_to_s3.sh — Periodic S3 candidate push for Lambda nodes
#
# Runs every 10 minutes via systemd timer or cron. Pushes any local
# candidate_*.pth models to S3 so the coordinator can discover and
# evaluate them.
#
# This replaces the broken TRAINING_COMPLETED event chain which depends
# on model sync from Lambda→coordinator succeeding (often fails due to
# network timeouts, P2P event loop starvation, or rsync errors).
#
# Mar 19, 2026: Created after discovering Lambda nodes produce candidates
# but never push them to S3 automatically.

set -euo pipefail

# Source node config for AWS credentials
if [ -f /etc/ringrift/node.conf ]; then
    source /etc/ringrift/node.conf
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION
fi

S3_BUCKET="${RINGRIFT_S3_BUCKET:-ringrift-models-20251214}"
S3_PREFIX="consolidated/models"
MODELS_DIR="${RINGRIFT_AI_PATH:-$HOME/ringrift/ai-service}/models"
LOCK_FILE="/tmp/ringrift-s3-push.lock"

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || stat -f %m "$LOCK_FILE" 2>/dev/null || echo 0) ))
    if [ "$lock_age" -lt 300 ]; then
        exit 0  # Another instance running, skip
    fi
    rm -f "$LOCK_FILE"  # Stale lock
fi
touch "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

pushed=0
for candidate in "$MODELS_DIR"/candidate_*p.pth; do
    [ -f "$candidate" ] || continue

    basename=$(basename "$candidate")

    # Skip timestamped versions (only push the latest symlink/copy)
    if echo "$basename" | grep -qE '_[0-9]{8}_[0-9]{6}\.pth$'; then
        continue
    fi

    # Check if S3 already has a newer or same-size version
    local_size=$(stat -c %s "$candidate" 2>/dev/null || stat -f %z "$candidate" 2>/dev/null)
    s3_size=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/$basename" 2>/dev/null | awk '{print $3}')

    if [ "$local_size" = "$s3_size" ]; then
        continue  # Already pushed
    fi

    # Push to S3
    if aws s3 cp "$candidate" "s3://$S3_BUCKET/$S3_PREFIX/$basename" \
        --storage-class STANDARD --quiet 2>/dev/null; then
        pushed=$((pushed + 1))
    fi
done

if [ "$pushed" -gt 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Pushed $pushed candidates to s3://$S3_BUCKET/$S3_PREFIX/"
fi
