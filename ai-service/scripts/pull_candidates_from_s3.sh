#!/bin/bash
# pull_candidates_from_s3.sh — Pull fresh candidates from S3 to coordinator
#
# Runs every 10 minutes on the coordinator (mac-studio). Pulls candidate
# models uploaded by Lambda nodes, making them available for gauntlet
# evaluation by the evaluation daemon.
#
# Mar 19, 2026: Created as the coordinator-side complement to
# push_candidates_to_s3.sh. Together they form a reliable S3-mediated
# candidate flow that doesn't depend on P2P rsync or event chains.

set -euo pipefail

export PATH="/opt/homebrew/bin:$PATH"

S3_BUCKET="${RINGRIFT_S3_BUCKET:-ringrift-models-20251214}"
S3_PREFIX="consolidated/models"
MODELS_DIR="${RINGRIFT_AI_PATH:-$HOME/Development/RingRift/ai-service}/models"
LOCK_FILE="/tmp/ringrift-s3-pull.lock"

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    lock_age=$(( $(date +%s) - $(stat -f %m "$LOCK_FILE" 2>/dev/null || stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0) ))
    if [ "$lock_age" -lt 300 ]; then
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
touch "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

pulled=0
for config in hex8_2p hex8_3p hex8_4p hexagonal_2p hexagonal_3p hexagonal_4p square19_2p square19_3p square19_4p square8_2p square8_3p square8_4p; do
    s3_path="s3://$S3_BUCKET/$S3_PREFIX/candidate_${config}.pth"
    local_path="$MODELS_DIR/candidate_${config}.pth"

    # Get S3 size
    s3_size=$(aws s3 ls "$s3_path" 2>/dev/null | awk '{print $3}')
    [ -z "$s3_size" ] && continue

    # Get local size
    local_size=0
    [ -f "$local_path" ] && local_size=$(stat -f %z "$local_path" 2>/dev/null || stat -c %s "$local_path" 2>/dev/null || echo 0)

    # Pull if different
    if [ "$s3_size" != "$local_size" ]; then
        if aws s3 cp "$s3_path" "$local_path" --quiet 2>/dev/null; then
            pulled=$((pulled + 1))
        fi
    fi
done

if [ "$pulled" -gt 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Pulled $pulled fresh candidates from S3"
fi
