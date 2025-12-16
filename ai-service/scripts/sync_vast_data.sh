#!/bin/bash
# Sync game data from Vast.ai instances to Lambda
# Run periodically via cron to collect distributed selfplay data

set -e

LAMBDA_HOST="lambda-a10"
LAMBDA_DB="/home/ubuntu/ringrift/ai-service/data/games/selfplay.db"
MERGE_SCRIPT="/home/ubuntu/ringrift/ai-service/scripts/merge_game_dbs.py"
TEMP_DIR="/tmp/vast_sync_$$"
LOG_FILE="/var/log/ringrift/vast_sync.log"

# Vast instances - format: "ssh_host:port"
VAST_INSTANCES=(
    "ssh5.vast.ai:14364"
    "ssh2.vast.ai:14370"
    "ssh8.vast.ai:19942"
    "ssh7.vast.ai:14398"
    "ssh1.vast.ai:14400"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

mkdir -p "$TEMP_DIR"
mkdir -p "$(dirname $LOG_FILE)"

log "Starting Vast.ai data sync"

COLLECTED_DBS=""
for instance in "${VAST_INSTANCES[@]}"; do
    host="${instance%:*}"
    port="${instance#*:}"

    log "Checking $host:$port..."

    # Try to copy the DB
    db_file="$TEMP_DIR/vast_${host}_${port}.db"
    if timeout 30 scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -P "$port" \
        "root@${host}:/root/RingRift/ai-service/data/games/vast_selfplay.db" "$db_file" 2>/dev/null; then

        # Check if it has games
        count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM games" 2>/dev/null || echo "0")
        if [ "$count" -gt 0 ]; then
            log "  Found $count games on $host"
            COLLECTED_DBS="$COLLECTED_DBS --db $db_file"
        else
            log "  No games on $host"
            rm -f "$db_file"
        fi
    else
        log "  Could not connect to $host:$port"
    fi
done

if [ -n "$COLLECTED_DBS" ]; then
    log "Merging collected databases into $LAMBDA_DB"
    ssh "$LAMBDA_HOST" "cd /home/ubuntu/ringrift/ai-service && python3 $MERGE_SCRIPT --output $LAMBDA_DB --dedupe-by-game-id $COLLECTED_DBS"
    log "Sync complete"
else
    log "No new games collected"
fi

# Cleanup
rm -rf "$TEMP_DIR"

log "Vast.ai sync finished"
