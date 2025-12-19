#!/bin/bash
# D10 Retraining Script with Fresh Gumbel Selfplay Data
#
# This script automates the D10 retraining workflow:
# 1. Converts JSONL selfplay data to NPZ training format
# 2. Trains with integrated enhancements (augmentation, background eval)
# 3. Runs tier evaluation on the trained model
#
# Usage:
#   ./scripts/retrain_d10_fresh.sh [--dry-run] [--skip-convert] [--skip-train]

set -e

# Configuration
SELFPLAY_DIR="data/gumbel_selfplay"
TRAINING_DIR="data/training"
MODELS_DIR="models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BOARD_TYPE="square8"
NUM_PLAYERS="2"

# Find the latest selfplay file
SELFPLAY_FILE=$(ls -t ${SELFPLAY_DIR}/sq8_gumbel_fresh_*.jsonl 2>/dev/null | head -1)
if [ -z "$SELFPLAY_FILE" ]; then
    echo "ERROR: No selfplay JSONL file found in ${SELFPLAY_DIR}"
    exit 1
fi

# Output paths
NPZ_FILE="${TRAINING_DIR}/d10_fresh_${TIMESTAMP}.npz"
MODEL_FILE="${MODELS_DIR}/sq8_2p_d10_fresh_${TIMESTAMP}.pth"

echo "=============================================="
echo "D10 Retraining with Fresh Data"
echo "=============================================="
echo "Selfplay file: ${SELFPLAY_FILE}"
echo "NPZ output: ${NPZ_FILE}"
echo "Model output: ${MODEL_FILE}"
echo ""

# Parse arguments
DRY_RUN=false
SKIP_CONVERT=false
SKIP_TRAIN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --skip-convert) SKIP_CONVERT=true ;;
        --skip-train) SKIP_TRAIN=true ;;
    esac
done

# Count games in selfplay file
GAME_COUNT=$(wc -l < "$SELFPLAY_FILE" | tr -d ' ')
echo "Games in selfplay file: ${GAME_COUNT}"

if [ "$GAME_COUNT" -lt 100 ]; then
    echo "WARNING: Less than 100 games. Consider waiting for more data."
    if [ "$DRY_RUN" = false ]; then
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Step 1: Convert JSONL to NPZ
if [ "$SKIP_CONVERT" = false ]; then
    echo ""
    echo "Step 1: Converting JSONL to NPZ..."
    mkdir -p "${TRAINING_DIR}"

    CMD="PYTHONPATH=. python scripts/jsonl_to_npz.py \
        --input ${SELFPLAY_FILE} \
        --output ${NPZ_FILE} \
        --board-type ${BOARD_TYPE} \
        --num-players ${NUM_PLAYERS}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: $CMD"
    else
        eval $CMD
        echo "NPZ file created: ${NPZ_FILE}"
    fi
else
    echo "Skipping conversion (--skip-convert)"
    # Use existing NPZ file
    NPZ_FILE=$(ls -t ${TRAINING_DIR}/d10_fresh_*.npz 2>/dev/null | head -1)
    if [ -z "$NPZ_FILE" ]; then
        echo "ERROR: No existing NPZ file found"
        exit 1
    fi
    echo "Using existing NPZ: ${NPZ_FILE}"
fi

# Step 2: Train with integrated enhancements
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "Step 2: Training with integrated enhancements..."

    CMD="python -m app.training.train \
        --data-path ${NPZ_FILE} \
        --save-path ${MODEL_FILE} \
        --use-integrated-enhancements \
        --enable-augmentation \
        --enable-background-eval \
        --auto-tune-batch-size \
        --board-type ${BOARD_TYPE} \
        --num-players ${NUM_PLAYERS} \
        --epochs 50 \
        --early-stopping-patience 10"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: $CMD"
    else
        eval $CMD
        echo "Model saved: ${MODEL_FILE}"
    fi
else
    echo "Skipping training (--skip-train)"
fi

# Step 3: Run tier evaluation
echo ""
echo "Step 3: Run tier evaluation..."
CANDIDATE_ID=$(basename "${MODEL_FILE}" .pth)

CMD="python scripts/select_best_checkpoint_by_elo.py \
    --candidate-id ${CANDIDATE_ID} \
    --games 20 \
    --board-type ${BOARD_TYPE} \
    --num-players ${NUM_PLAYERS} \
    --copy-best"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run: $CMD"
else
    echo "Run this command to evaluate checkpoints:"
    echo "  $CMD"
fi

echo ""
echo "=============================================="
echo "D10 Retraining Workflow Complete"
echo "=============================================="
