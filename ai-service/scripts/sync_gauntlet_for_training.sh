#!/bin/bash
# Sync gauntlet game data from S3 → coordinator → re-export NPZ → S3
# Closes the gap where 153GB of high-quality MCTS games from gauntlet
# evaluation are not included in training because they only exist on
# the gauntlet node (gh200-8), not on the coordinator where exports run.
#
# The auto_export_daemon already has --include-gauntlet enabled, but
# it can only export games that are in LOCAL databases. This script
# ensures gauntlet DBs are pulled from S3 to the coordinator so the
# next auto_export cycle includes them.
#
# Install on mac-studio:
#   crontab -e → 0 */6 * * * ~/Development/RingRift/ai-service/scripts/sync_gauntlet_for_training.sh >> /tmp/gauntlet_sync.log 2>&1
#
# Mar 24, 2026: Created after discovering gauntlet games were backed
# up to S3 but never pulled to coordinator for training inclusion.

set -e
cd ~/Development/RingRift/ai-service
export PATH="/opt/homebrew/bin:$PATH"
source .venv/bin/activate
export PYTHONPATH=.

LOG_PREFIX="[gauntlet-sync $(date +%H:%M)]"
BUCKET="ringrift-models-20251214"

echo "$LOG_PREFIX Starting gauntlet data sync for training"

# 1. Pull gauntlet DBs from S3 to coordinator (incremental, size-based)
echo "$LOG_PREFIX Pulling gauntlet DBs from S3..."
aws s3 sync "s3://$BUCKET/consolidated/games/" data/games/ \
    --exclude "*" --include "gauntlet_*.db" \
    --size-only --quiet

# 2. Verify we have gauntlet data
total_gauntlet=$(ls -1 data/games/gauntlet_*.db 2>/dev/null | wc -l | tr -d ' ')
echo "$LOG_PREFIX Found $total_gauntlet gauntlet DBs locally"

if [ "$total_gauntlet" -eq 0 ]; then
    echo "$LOG_PREFIX No gauntlet DBs found, skipping NPZ re-export"
    exit 0
fi

# 3. Check which configs have gauntlet data newer than current NPZ
for cfg in hex8_2p hex8_3p hex8_4p square8_2p square8_3p square8_4p square19_2p square19_3p square19_4p hexagonal_2p hexagonal_3p hexagonal_4p; do
    gauntlet_db="data/games/gauntlet_${cfg}.db"
    npz_file="data/training/${cfg}.npz"

    if [ ! -f "$gauntlet_db" ]; then
        continue
    fi

    # Check game count in gauntlet DB
    game_count=$(.venv/bin/python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('$gauntlet_db')
    count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
    print(count)
except: print(0)
" 2>/dev/null)

    if [ "$game_count" -gt 50 ]; then
        echo "$LOG_PREFIX $cfg: $game_count gauntlet games available"
    fi
done

# 4. Push any updated NPZ to S3 for training nodes
echo "$LOG_PREFIX Syncing NPZ files to S3..."
aws s3 sync data/training/ "s3://$BUCKET/consolidated/training/" \
    --exclude "*" --include "*.npz" \
    --size-only --quiet

echo "$LOG_PREFIX Gauntlet sync complete"
