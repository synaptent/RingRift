#!/bin/bash
# Collect selfplay DBs from GH200 hosts and merge into main selfplay.db
# Usage: ./scripts/collect_and_merge_selfplay.sh [--dry-run]

set -e

SELFPLAY_HOSTS=(
  "GH200-a:ubuntu@192.222.51.29:hexagonal_4p"
  "GH200-b:ubuntu@192.222.51.167:square19_4p"
  "GH200-c:ubuntu@192.222.51.162:hexagonal_2p"
  "GH200-d:ubuntu@192.222.58.122:square19_2p"
  "GH200-e:ubuntu@192.222.57.162:hexagonal_3p"
  "GH200-f:ubuntu@192.222.57.178:square19_3p"
)

COLLECT_DIR="data/games/collected_$(date +%Y%m%d_%H%M%S)"
MAIN_DB="data/games/selfplay.db"
DRY_RUN=false

if [ "${1:-}" == "--dry-run" ]; then
  DRY_RUN=true
  echo "[DRY RUN MODE]"
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

echo "=============================================="
echo "  SELFPLAY DATA COLLECTION & MERGE"
echo "=============================================="
echo ""

# Create collection directory
mkdir -p "$COLLECT_DIR"

# Check completion status and collect DBs
echo "Step 1: Checking job completion and collecting DBs..."
echo ""

COLLECTED=0
STILL_RUNNING=0

for entry in "${SELFPLAY_HOSTS[@]}"; do
  IFS=':' read -r name host config <<< "$entry"

  # Check if job is still running
  running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f "run_self_play_soak|generate_canonical_selfplay"' 2>/dev/null || true)

  if [ -n "$running" ]; then
    echo "⏳ $name ($config): STILL RUNNING - skipping"
    ((STILL_RUNNING++))
    continue
  fi

  # Check if DB exists and has games
  games=$(ssh -o ConnectTimeout=5 "$host" "sqlite3 ~/ringrift/ai-service/data/games/new_${config}.db 'SELECT COUNT(*) FROM games' 2>/dev/null" 2>/dev/null || echo "0")

  if [ "$games" == "0" ] || [ -z "$games" ]; then
    echo "⚠️  $name ($config): No games in DB - skipping"
    continue
  fi

  echo "✓ $name ($config): $games games - collecting..."

  if [ "$DRY_RUN" == "false" ]; then
    # Download the DB
    scp -o ConnectTimeout=10 "${host}:~/ringrift/ai-service/data/games/new_${config}.db" \
        "${COLLECT_DIR}/${config}.db" 2>/dev/null

    if [ $? -eq 0 ]; then
      echo "  → Downloaded to ${COLLECT_DIR}/${config}.db"
      ((COLLECTED++))
    else
      echo "  ✗ Failed to download"
    fi
  else
    echo "  → Would download to ${COLLECT_DIR}/${config}.db"
    ((COLLECTED++))
  fi
done

echo ""
echo "Collection summary: $COLLECTED DBs collected, $STILL_RUNNING jobs still running"

if [ "$COLLECTED" == "0" ]; then
  echo ""
  echo "No DBs to merge. Exiting."
  exit 0
fi

# Merge into main DB
echo ""
echo "Step 2: Merging collected DBs into main selfplay.db..."
echo ""

if [ "$DRY_RUN" == "false" ]; then
  # Build merge command
  MERGE_CMD="python scripts/merge_game_dbs.py --output $MAIN_DB --dedupe-by-game-id"
  for db in "$COLLECT_DIR"/*.db; do
    if [ -f "$db" ]; then
      MERGE_CMD="$MERGE_CMD --db $db"
    fi
  done

  echo "Running: $MERGE_CMD"
  $MERGE_CMD

  # Show final stats
  echo ""
  echo "Merge complete. Final DB stats:"
  total_games=$(sqlite3 "$MAIN_DB" "SELECT COUNT(*) FROM games" 2>/dev/null || echo "?")
  echo "  Total games in $MAIN_DB: $total_games"
else
  echo "[DRY RUN] Would merge ${COLLECTED} DBs into $MAIN_DB"
fi

echo ""
echo "=============================================="
echo "  COLLECTION COMPLETE"
echo "=============================================="
