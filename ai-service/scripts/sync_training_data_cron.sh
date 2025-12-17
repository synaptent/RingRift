#!/bin/bash
# Automated training data sync from GPU cluster
# Run via cron: */15 * * * * /path/to/sync_training_data_cron.sh
#
# Configuration: Set SYNC_HOSTS in config/sync_hosts.env or as environment variables
# Example:
#   SYNC_PRIMARY_HOST=ubuntu@gpu-primary
#   SYNC_FALLBACK_HOST=ubuntu@gpu-secondary

set -e
cd "$(dirname "$0")/.."

LOG_FILE="logs/sync_cron.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Load hosts from config file if exists
if [ -f "config/sync_hosts.env" ]; then
  source config/sync_hosts.env
fi

# Default to placeholder hosts if not configured
SYNC_PRIMARY_HOST="${SYNC_PRIMARY_HOST:-ubuntu@gpu-primary}"
SYNC_FALLBACK_HOST="${SYNC_FALLBACK_HOST:-}"

echo "[$TIMESTAMP] Starting data sync..." >> "$LOG_FILE"

# Try primary, then fallback
sync_success=false
for ssh_host in "$SYNC_PRIMARY_HOST" "$SYNC_FALLBACK_HOST"; do
  [ -z "$ssh_host" ] && continue
  if ssh -o ConnectTimeout=5 $ssh_host "test -f ~/ringrift/ai-service/data/games/jsonl_aggregated.db" 2>/dev/null; then
    echo "[$TIMESTAMP] Using host $ssh_host for sync" >> "$LOG_FILE"
    rsync -avz --timeout=120 $ssh_host:~/ringrift/ai-service/data/games/jsonl_aggregated.db \
      data/games/cluster_synced.db >> "$LOG_FILE" 2>&1
    sync_success=true
    break
  fi
done

if [ "$sync_success" = false ]; then
  echo "[$TIMESTAMP] Sync failed - no accessible host" >> "$LOG_FILE"
fi

# Update symlinks
ln -sf cluster_synced.db data/games/all_jsonl_training.db
ln -sf cluster_synced.db data/games/jsonl_aggregated.db

# Sync MCTS data if available
mkdir -p data/selfplay/mcts_cluster
for ssh_host in "$SYNC_PRIMARY_HOST" "$SYNC_FALLBACK_HOST"; do
  [ -z "$ssh_host" ] && continue
  rsync -avz --timeout=60 $ssh_host:~/ringrift/ai-service/data/selfplay/mcts_*/games.jsonl \
    data/selfplay/mcts_cluster/ >> "$LOG_FILE" 2>&1 && break || true
done

echo "[$TIMESTAMP] Sync complete" >> "$LOG_FILE"
