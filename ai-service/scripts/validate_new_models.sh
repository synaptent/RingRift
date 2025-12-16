#!/bin/bash
# Run Elo tournaments to validate new v4_large models against existing best models
# Usage: ./validate_new_models.sh

MODEL_DIR=/lambda/nfs/RingRift/models/v4_large
BEST_MODEL_DIR=/home/ubuntu/ringrift/ai-service/models
LOG_DIR=/tmp/elo_tournaments
GAMES_PER_MATCH=50

mkdir -p $LOG_DIR

cd ~/ringrift/ai-service
source venv/bin/activate

declare -A BOARD_CONFIGS=(
    ["sq8"]="square8"
    ["sq19"]="square19"
    ["hex"]="hexagonal"
)

for model_file in $MODEL_DIR/*_v4_large.pth; do
    [ -f "$model_file" ] || continue
    
    # Extract config from filename (e.g., sq8_2p from sq8_2p_v4_large.pth)
    filename=$(basename "$model_file" _v4_large.pth)
    board_short=$(echo "$filename" | cut -d_ -f1)
    num_players=$(echo "$filename" | grep -oE "[234]p" | tr -d p)
    board_type=${BOARD_CONFIGS[$board_short]}
    
    # Find existing best model
    best_model="$BEST_MODEL_DIR/ringrift_best_${filename}.pth"
    if [ ! -f "$best_model" ]; then
        best_model=$(ls $BEST_MODEL_DIR/*${board_short}*${num_players}p*.pth 2>/dev/null | head -1)
    fi
    
    if [ -z "$best_model" ] || [ ! -f "$best_model" ]; then
        echo "[$(date)] No baseline model found for $filename, skipping tournament"
        continue
    fi
    
    log_file="$LOG_DIR/elo_${filename}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$(date)] Starting Elo tournament: $filename"
    echo "  New model: $model_file"
    echo "  Baseline: $best_model"
    echo "  Board: $board_type, Players: $num_players"
    
    PYTHONPATH=. python scripts/run_model_elo_tournament.py \
        --board "$board_type" \
        --players "$num_players" \
        --models "$best_model" "$model_file" \
        --games "$GAMES_PER_MATCH" \
        2>&1 | tee "$log_file"
    
    echo "[$(date)] Tournament complete for $filename"
    echo ""
done

echo "[$(date)] All tournaments complete"
