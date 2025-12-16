#!/bin/bash
# Train a large model (20 blocks, 256 filters) with v3 architecture
# Usage: ./train_large_model.sh <board_type> <num_players> <npz_path>

BOARD_TYPE=$1
NUM_PLAYERS=$2
NPZ_PATH=$3
OUTPUT_DIR=/lambda/nfs/RingRift/models/v4_large

mkdir -p $OUTPUT_DIR

cd ~/ringrift/ai-service
source venv/bin/activate

echo "Starting training: $BOARD_TYPE ${NUM_PLAYERS}p"
echo "Data: $NPZ_PATH"
echo "Output: $OUTPUT_DIR/${BOARD_TYPE}_${NUM_PLAYERS}p_v4_large.pth"

PYTHONPATH=. python -m app.training.train \
    --data-path "$NPZ_PATH" \
    --board-type "$BOARD_TYPE" \
    --num-players "$NUM_PLAYERS" \
    --model-version v3 \
    --num-res-blocks 20 \
    --num-filters 256 \
    --epochs 50 \
    --lr-scheduler cosine \
    --early-stopping-patience 10 \
    --save-path "$OUTPUT_DIR/${BOARD_TYPE}_${NUM_PLAYERS}p_v4_large.pth" \
    2>&1 | tee "/tmp/train_${BOARD_TYPE}_${NUM_PLAYERS}p.log"
