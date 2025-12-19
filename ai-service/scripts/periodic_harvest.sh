#!/bin/bash
# Periodic Training Data Harvest
# Run via cron: 0 */6 * * * /home/ubuntu/ringrift/ai-service/scripts/periodic_harvest.sh
#
# Harvests high-quality training data every 6 hours and triggers training if enough new data.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$AI_SERVICE_DIR/logs"
DATA_DIR="$AI_SERVICE_DIR/data"
HARVEST_DIR="$DATA_DIR/harvested"
TRAINING_DIR="$DATA_DIR/training"

HOSTNAME=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/periodic_harvest_${TIMESTAMP}.log"

# Configuration
MIN_QUALITY=0.7
MAX_GAMES_PER_HARVEST=25000
MIN_NEW_GAMES_FOR_TRAINING=10000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting periodic harvest on $HOSTNAME ==="

cd "$AI_SERVICE_DIR"
source venv/bin/activate
export PYTHONPATH="$AI_SERVICE_DIR"

# Create directories
mkdir -p "$HARVEST_DIR" "$LOG_DIR"

# Count existing harvested games
EXISTING_GAMES=0
if [ -f "$HARVEST_DIR/accumulated_square8_2p.jsonl" ]; then
    EXISTING_GAMES=$(wc -l < "$HARVEST_DIR/accumulated_square8_2p.jsonl")
fi
log "Existing accumulated games: $EXISTING_GAMES"

# Harvest new games
HARVEST_OUTPUT="$HARVEST_DIR/harvest_${HOSTNAME}_${TIMESTAMP}.jsonl"
log "Harvesting new games to $HARVEST_OUTPUT"

python scripts/harvest_local_training_data.py \
    --board-type square8 \
    --num-players 2 \
    --min-quality $MIN_QUALITY \
    --max-games $MAX_GAMES_PER_HARVEST \
    --output "$HARVEST_OUTPUT" 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$HARVEST_OUTPUT" ]; then
    log "ERROR: Harvest failed - no output file"
    exit 1
fi

NEW_GAMES=$(wc -l < "$HARVEST_OUTPUT")
log "Harvested $NEW_GAMES new games"

# Accumulate with deduplication
ACCUMULATED="$HARVEST_DIR/accumulated_square8_2p.jsonl"
TEMP_ACCUMULATED="$HARVEST_DIR/temp_accumulated.jsonl"

if [ -f "$ACCUMULATED" ]; then
    # Merge and deduplicate
    cat "$ACCUMULATED" "$HARVEST_OUTPUT" | python3 -c "
import json
import sys

seen = set()
for line in sys.stdin:
    try:
        game = json.loads(line.strip())
        game_id = game.get('game_id', str(hash(line))[:20])
        if game_id not in seen:
            seen.add(game_id)
            print(line.strip())
    except:
        pass
" > "$TEMP_ACCUMULATED"
    mv "$TEMP_ACCUMULATED" "$ACCUMULATED"
else
    cp "$HARVEST_OUTPUT" "$ACCUMULATED"
fi

TOTAL_GAMES=$(wc -l < "$ACCUMULATED")
ADDED_GAMES=$((TOTAL_GAMES - EXISTING_GAMES))
log "Total accumulated games: $TOTAL_GAMES (added $ADDED_GAMES new unique games)"

# Clean up individual harvest file
rm -f "$HARVEST_OUTPUT"

# Check if we should trigger training
if [ $ADDED_GAMES -ge $MIN_NEW_GAMES_FOR_TRAINING ]; then
    log "Sufficient new games ($ADDED_GAMES >= $MIN_NEW_GAMES_FOR_TRAINING), triggering training..."

    # Convert to NPZ
    NPZ_OUTPUT="$TRAINING_DIR/accumulated_square8_2p_${TIMESTAMP}.npz"
    log "Converting to NPZ: $NPZ_OUTPUT"

    python scripts/jsonl_to_npz.py \
        --input "$ACCUMULATED" \
        --output "$NPZ_OUTPUT" \
        --board-type square8 \
        --num-players 2 \
        --sample-every 1 2>&1 | tee -a "$LOG_FILE"

    if [ -f "$NPZ_OUTPUT" ]; then
        log "NPZ conversion complete: $(ls -lh "$NPZ_OUTPUT" | awk '{print $5}')"

        # Trigger training (background)
        log "Starting training job..."
        nohup python scripts/train_model.py \
            --config square8_2p \
            --data "$NPZ_OUTPUT" \
            --use-hyperparameters \
            > "$LOG_DIR/training_${TIMESTAMP}.log" 2>&1 &

        log "Training started in background (PID: $!)"
    else
        log "ERROR: NPZ conversion failed"
    fi
else
    log "Not enough new games for training ($ADDED_GAMES < $MIN_NEW_GAMES_FOR_TRAINING)"
fi

log "=== Periodic harvest complete ==="
