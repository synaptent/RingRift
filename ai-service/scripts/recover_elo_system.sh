#!/bin/bash
#
# RingRift Elo System Recovery Script
# Run this on mac-studio to restore Elo tracking and start re-evaluation
#
# Usage:
#   ./scripts/recover_elo_system.sh           # Full recovery
#   ./scripts/recover_elo_system.sh --dry-run # Preview only
#

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No changes will be made ==="
fi

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"

echo "========================================"
echo "RingRift Elo System Recovery"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo "Date: $(date)"
echo ""

# Step 1: Check current state
echo "=== Step 1: Checking current state ==="

MAIN_ELO_DB="/Volumes/RingRift-Data/ai-service/data/games/unified_elo.db"
BACKUP_ELO_DB="/Volumes/RingRift-Data/nebius_backup_20260108/ringrift-h100-3/data/unified_elo.db"

if [[ -f "$MAIN_ELO_DB" ]]; then
    MAIN_COUNT=$(sqlite3 "$MAIN_ELO_DB" "SELECT COUNT(*) FROM match_history;" 2>/dev/null || echo "0")
    echo "Main Elo DB matches: $MAIN_COUNT"
else
    MAIN_COUNT=0
    echo "Main Elo DB: NOT FOUND"
fi

if [[ -f "$BACKUP_ELO_DB" ]]; then
    BACKUP_COUNT=$(sqlite3 "$BACKUP_ELO_DB" "SELECT COUNT(*) FROM match_history;" 2>/dev/null || echo "0")
    echo "Backup Elo DB matches: $BACKUP_COUNT"
else
    BACKUP_COUNT=0
    echo "Backup Elo DB: NOT FOUND"
fi

# Step 2: Sync Elo database if needed
echo ""
echo "=== Step 2: Sync Elo database ==="

if [[ "$BACKUP_COUNT" -gt "$MAIN_COUNT" ]]; then
    echo "Backup has more matches ($BACKUP_COUNT vs $MAIN_COUNT)"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would copy: $BACKUP_ELO_DB -> $MAIN_ELO_DB"
    else
        echo "Copying backup to main location..."
        mkdir -p "$(dirname "$MAIN_ELO_DB")"
        cp "$BACKUP_ELO_DB" "$MAIN_ELO_DB"
        echo "Done. New match count: $(sqlite3 "$MAIN_ELO_DB" "SELECT COUNT(*) FROM match_history;")"
    fi
else
    echo "Main DB is up to date (or backup not available)"
fi

# Step 3: Check P2P status
echo ""
echo "=== Step 3: Check P2P network ==="

P2P_STATUS=$(curl -s --connect-timeout 5 http://localhost:8770/status 2>/dev/null || echo "")
if [[ -n "$P2P_STATUS" ]]; then
    echo "P2P is running:"
    echo "$P2P_STATUS" | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"  Leader: {d.get(\"leader_id\", \"unknown\")}")
print(f"  Alive peers: {d.get(\"alive_peers\", 0)}")
print(f"  Training jobs: {len(d.get(\"training_jobs\", []))}")
'
else
    echo "P2P is NOT running"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would start P2P orchestrator"
    else
        echo "Starting P2P orchestrator..."
        nohup python scripts/p2p_orchestrator.py > logs/p2p_orchestrator.log 2>&1 &
        P2P_PID=$!
        echo "Started P2P with PID: $P2P_PID"
        sleep 5
        if curl -s --connect-timeout 5 http://localhost:8770/status > /dev/null 2>&1; then
            echo "P2P started successfully"
        else
            echo "WARNING: P2P may still be starting up..."
        fi
    fi
fi

# Step 4: Configure StaleEvaluationDaemon
echo ""
echo "=== Step 4: Configure StaleEvaluationDaemon ==="

ENV_FILE=".env.local"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would add stale eval config to $ENV_FILE"
else
    # Add stale eval config if not present
    if ! grep -q "RINGRIFT_STALE_EVAL_ENABLED" "$ENV_FILE" 2>/dev/null; then
        echo "" >> "$ENV_FILE"
        echo "# Stale Elo evaluation config (added $(date))" >> "$ENV_FILE"
        echo "RINGRIFT_STALE_EVAL_ENABLED=true" >> "$ENV_FILE"
        echo "RINGRIFT_STALE_EVAL_AGE_DAYS=7" >> "$ENV_FILE"
        echo "RINGRIFT_STALE_EVAL_INTERVAL=21600" >> "$ENV_FILE"
        echo "RINGRIFT_STALE_EVAL_MAX_PER_CYCLE=10" >> "$ENV_FILE"
        echo "Added stale eval config to $ENV_FILE"
    else
        echo "Stale eval config already exists in $ENV_FILE"
    fi
fi

# Step 5: Preview stale entries
echo ""
echo "=== Step 5: Preview stale entries ==="

python3 scripts/refresh_stale_elo.py --age-days 7 --stats-only 2>/dev/null || echo "Could not run refresh script (might need database path)"

# Step 6: Queue re-evaluation (if not dry run)
echo ""
echo "=== Step 6: Queue re-evaluation ==="

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would run: python scripts/refresh_stale_elo.py --age-days 7 --execute"
else
    read -p "Queue stale entries for GPU Gumbel MCTS re-evaluation? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/refresh_stale_elo.py --age-days 7 --execute
    else
        echo "Skipped. Run manually with: python scripts/refresh_stale_elo.py --age-days 7 --execute"
    fi
fi

echo ""
echo "========================================"
echo "Recovery complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Monitor P2P: curl -s http://localhost:8770/status | python3 -m json.tool"
echo "2. Check evaluation queue: python -c 'from app.coordination.evaluation_queue import get_evaluation_queue; print(get_evaluation_queue().get_queue_status())'"
echo "3. Run master loop for full automation: python scripts/master_loop.py"
