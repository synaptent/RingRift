#!/bin/bash
# Cron job for syncing selfplay data from remote instances
# Add to crontab: */30 * * * * /path/to/cron_sync_selfplay.sh >> /tmp/sync.log 2>&1

set -e
cd "$(dirname "$0")/.."

# Only run if not already running
LOCKFILE="/tmp/selfplay_sync.lock"
if [ -f "$LOCKFILE" ]; then
    if kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
        echo "$(date): Sync already running, skipping"
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

echo "$(date): Starting selfplay sync..."
./scripts/sync_selfplay_data.sh || echo "$(date): Sync failed with code $?"
echo "$(date): Sync completed"
