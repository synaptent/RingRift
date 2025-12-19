#!/bin/bash
# RingRift Data Sync Orchestrator
# Runs from local machine, syncs data from all nodes to coordinator
# Long-term automated solution - add to crontab
#
# Configuration: Set these environment variables or create config/sync_hosts.env:
#   COORDINATOR_IP - Tailscale IP of coordinator node
#   COORDINATOR_PATH - Path to data/games on coordinator
#   NODES - Space-separated list of "name:ip:path" entries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config/sync_hosts.env"

if [[ -z "${RINGRIFT_LEGACY_SYNC:-}" ]]; then
    echo "[sync_all_nodes] Deprecated; use cluster_sync_coordinator.py --mode full"
    python3 "$SCRIPT_DIR/cluster_sync_coordinator.py" --mode full
    exit $?
fi

# Load config if exists
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# Default values (override in sync_hosts.env)
COORDINATOR_IP="${COORDINATOR_IP:-}"
COORDINATOR_PATH="${COORDINATOR_PATH:-/workspace/ringrift/ai-service/data/games}"

# Node configurations - set in sync_hosts.env or as environment variable
# Format: "name:ip:path name2:ip2:path2 ..."
if [[ -z "${NODES_STRING:-}" ]]; then
    echo "[sync_all_nodes] Error: No nodes configured"
    echo "[sync_all_nodes] Create config/sync_hosts.env with:"
    echo "  COORDINATOR_IP=<your_coordinator_tailscale_ip>"
    echo "  NODES_STRING=\"node1:10.0.0.1:/path node2:10.0.0.2:/path\""
    echo "[sync_all_nodes] See config/sync_hosts.env.example for template"
    exit 1
fi

# Convert space-separated string to array
IFS=' ' read -ra NODES <<< "$NODES_STRING"

if [[ -z "$COORDINATOR_IP" ]]; then
    echo "[sync_all_nodes] Error: COORDINATOR_IP not set"
    exit 1
fi

TEMP_DIR="/tmp/ringrift_sync_$$"
mkdir -p "$TEMP_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Ensure coordinator has database
log "Initializing coordinator database..."
ssh -o ConnectTimeout=10 root@$COORDINATOR_IP "cd $COORDINATOR_PATH && python3 -c \"
import sqlite3
conn = sqlite3.connect('selfplay.db')
conn.execute('''CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    board_type TEXT,
    num_players INTEGER,
    moves TEXT,
    winner INTEGER,
    final_scores TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_node TEXT
)''')
conn.commit()
print('DB ready')
\"" 2>/dev/null || log "Warning: Could not init coordinator DB"

total_synced=0

for node_entry in "${NODES[@]}"; do
    IFS=':' read -r name ip path <<< "$node_entry"
    log "Syncing from $name..."
    
    # Try to get games from SQLite first
    games=$(ssh -o ConnectTimeout=10 -o BatchMode=yes root@$ip "cd $path 2>/dev/null && python3 -c \"
import sqlite3, json, sys
try:
    conn = sqlite3.connect('selfplay.db')
    cursor = conn.execute('SELECT game_id, board_type, num_players, moves, winner, final_scores, metadata FROM games')
    games = [dict(zip(['game_id','board_type','num_players','moves','winner','final_scores','metadata'], row)) for row in cursor]
    print(json.dumps(games))
except Exception as e:
    print('[]', file=sys.stderr)
    sys.exit(0)
\" 2>/dev/null" 2>/dev/null || echo "[]")
    
    if [ "$games" != "[]" ] && [ -n "$games" ]; then
        # Insert into coordinator
        count=$(ssh -o ConnectTimeout=30 root@$COORDINATOR_IP "cd $COORDINATOR_PATH && python3 -c \"
import sqlite3, json, sys
games = json.loads('''$games''')
conn = sqlite3.connect('selfplay.db')
count = 0
for g in games:
    try:
        conn.execute('INSERT OR IGNORE INTO games (game_id, board_type, num_players, moves, winner, final_scores, metadata, source_node) VALUES (?,?,?,?,?,?,?,?)',
            (g['game_id'], g['board_type'], g['num_players'], g['moves'], g['winner'], g['final_scores'], g['metadata'], '$name'))
        count += 1
    except: pass
conn.commit()
print(count)
\"" 2>/dev/null || echo "0")
        log "  $name: synced $count games"
        total_synced=$((total_synced + count))
    else
        # Try JSONL fallback
        log "  $name: no SQLite data, trying JSONL..."
    fi
done

# Print final stats
log "=== Sync complete: $total_synced games synced ==="
ssh -o ConnectTimeout=10 root@$COORDINATOR_IP "cd $COORDINATOR_PATH && python3 -c \"
import sqlite3
conn = sqlite3.connect('selfplay.db')
print('Current totals:')
for row in conn.execute('SELECT board_type, num_players, COUNT(*) FROM games GROUP BY 1,2 ORDER BY 3 DESC'):
    print(f'  {row[0]} {row[1]}p: {row[2]}')
total = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
print(f'TOTAL: {total} games')
\"" 2>/dev/null

rm -rf "$TEMP_DIR"
