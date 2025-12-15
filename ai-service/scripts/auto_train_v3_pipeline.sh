#!/bin/bash
# Automated V3 training pipeline
# Monitors for export completion, trains model, runs evaluation

set -e

cd ~/ringrift/ai-service
source venv/bin/activate
export PYTHONPATH=.

BOARD_TYPE="${1:-hexagonal}"
NUM_PLAYERS="${2:-2}"
DATA_FILE="data/training/${BOARD_TYPE}_${NUM_PLAYERS}p_v3.npz"
LOG_DIR="logs/v3_pipeline_${BOARD_TYPE}_${NUM_PLAYERS}p"
MODEL_DIR="models/v3_${BOARD_TYPE}_${NUM_PLAYERS}p"

mkdir -p "$LOG_DIR" "$MODEL_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline.log"
}

log "Starting V3 training pipeline for $BOARD_TYPE ${NUM_PLAYERS}p"

# Step 1: Wait for export to complete
log "Waiting for $DATA_FILE to be ready..."
while [ ! -f "$DATA_FILE" ]; do
    sleep 30
    log "  Still waiting for export..."
done

# Verify file is complete (size stable for 30s)
PREV_SIZE=0
while true; do
    CURR_SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
    if [ "$CURR_SIZE" -eq "$PREV_SIZE" ] && [ "$CURR_SIZE" -gt 0 ]; then
        break
    fi
    PREV_SIZE=$CURR_SIZE
    sleep 30
    log "  File size: $CURR_SIZE bytes, waiting for stability..."
done

log "Export complete! File size: $CURR_SIZE bytes"

# Verify channel count
CHANNELS=$(python3 -c "import numpy as np; d=np.load('$DATA_FILE', allow_pickle=True); print(d['features'].shape[1])" 2>/dev/null)
log "Feature channels: $CHANNELS"

if [ "$BOARD_TYPE" = "hexagonal" ] && [ "$CHANNELS" != "64" ]; then
    log "ERROR: Expected 64 channels for hex V3, got $CHANNELS"
    exit 1
fi

# Step 2: Train V3 model
log "Starting V3 training..."
python scripts/run_nn_training_baseline.py \
    --board "$BOARD_TYPE" \
    --num-players "$NUM_PLAYERS" \
    --data-path "$DATA_FILE" \
    --run-dir "$MODEL_DIR" \
    --model-version v3 \
    --epochs 100 \
    2>&1 | tee "$LOG_DIR/training.log"

log "Training complete!"

# Step 3: Find the trained model
MODEL_PATH=$(ls -t "$MODEL_DIR"/*.pth 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ]; then
    log "ERROR: No model found after training"
    exit 1
fi
log "Trained model: $MODEL_PATH"

# Step 4: Run evaluation tournament
log "Starting evaluation tournament..."
python scripts/run_model_elo_tournament.py \
    --board-type "$BOARD_TYPE" \
    --num-players "$NUM_PLAYERS" \
    --games-per-pair 50 \
    --db "data/elo_leaderboard.db" \
    2>&1 | tee "$LOG_DIR/tournament.log"

log "Tournament complete!"

# Step 5: Show results
log "Final Elo standings:"
python scripts/track_elo_improvement.py --hours 2 2>&1 | tee -a "$LOG_DIR/pipeline.log"

log "V3 pipeline complete for $BOARD_TYPE ${NUM_PLAYERS}p"
