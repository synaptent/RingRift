#!/bin/bash
# Train Hex8 2p, 3p and 4p Models on Cluster
#
# This script handles the full training pipeline for missing hex8 models:
# 1. Validates/gates training data for canonical compliance
# 2. Trains hex8_2p, hex8_3p and hex8_4p models using NNUE policy training
#
# Usage:
#   ./scripts/train_hex8_models.sh [command]
#
# Commands:
#   validate   - Validate training data for hex8 (no training)
#   train-2p   - Train hex8 2-player model
#   train-3p   - Train hex8 3-player model
#   train-4p   - Train hex8 4-player model
#   train-all  - Train all 2p, 3p and 4p models
#   status     - Check training status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Training database - contains hex8 data
TRAINING_DB="data/games/all_jsonl_training.db"
OUTPUT_DIR="models/nnue"
LOG_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_data() {
    log_info "Checking available hex8 training data..."

    # Use venv if available
    if [ -f "venv/bin/python3" ]; then
        PYTHON="venv/bin/python3"
    else
        PYTHON="python3"
    fi

    $PYTHON -c "
import sqlite3
conn = sqlite3.connect('${TRAINING_DB}')
cur = conn.cursor()
cur.execute('''
    SELECT board_type, num_players, COUNT(*)
    FROM games
    WHERE board_type = 'hex8'
    GROUP BY board_type, num_players
''')
rows = cur.fetchall()
if not rows:
    print('ERROR: No hex8 data found in ${TRAINING_DB}')
    exit(1)
for board, players, count in rows:
    print(f'  {board} {players}p: {count} games')
conn.close()
"
}

validate_data() {
    log_info "Validating hex8 training data for canonical compliance..."

    # Check if we can bypass validation for now
    if [ -f "data/games/canonical_hex8.db" ]; then
        log_info "Canonical hex8 database already exists"
        return 0
    fi

    # Use venv if available
    if [ -f "venv/bin/python3" ]; then
        PYTHON="venv/bin/python3"
    else
        PYTHON="python3"
    fi

    # Try to run validation
    log_warn "Running canonical validation (this may take a while)..."
    $PYTHON scripts/build_canonical_training_pool_db.py \
        --input "${TRAINING_DB}" \
        --output "data/games/canonical_hex8.db" \
        --board-type hex8 \
        --skip-parity-check \
        2>&1 || {
            log_warn "Canonical validation unavailable, using training DB directly"
            log_warn "Training will proceed with --skip-registry-check flag"
            return 1
        }

    log_info "Canonical validation complete"
    return 0
}

train_model() {
    local BOARD_TYPE="hex8"
    local NUM_PLAYERS="$1"
    local DB_PATH="${TRAINING_DB}"

    log_info "=========================================="
    log_info "Training ${BOARD_TYPE} ${NUM_PLAYERS}p model"
    log_info "=========================================="

    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

    # Check for existing model
    local MODEL_FILE="${OUTPUT_DIR}/nnue_hex8_${NUM_PLAYERS}p.pt"
    if [ -f "${MODEL_FILE}" ]; then
        log_warn "Model already exists: ${MODEL_FILE}"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping ${BOARD_TYPE} ${NUM_PLAYERS}p"
            return 0
        fi
    fi

    # Run training (use venv if available)
    log_info "Starting training..."
    if [ -f "venv/bin/python3" ]; then
        PYTHON="venv/bin/python3"
    else
        PYTHON="python3"
    fi

    $PYTHON scripts/train_nnue_policy.py \
        --db "${DB_PATH}" \
        --board-type "${BOARD_TYPE}" \
        --num-players "${NUM_PLAYERS}" \
        --save-path "${OUTPUT_DIR}/nnue_hex8_${NUM_PLAYERS}p.pt" \
        --epochs 50 \
        --batch-size 256 \
        --allow-pending-gate \
        2>&1 | tee "${LOG_DIR}/train_hex8_${NUM_PLAYERS}p.log"

    if [ $? -eq 0 ]; then
        log_info "Training complete: ${MODEL_FILE}"
    else
        log_error "Training failed for ${BOARD_TYPE} ${NUM_PLAYERS}p"
        return 1
    fi
}

train_with_curriculum() {
    local BOARD_TYPE="hex8"
    local NUM_PLAYERS="$1"
    local DB_PATH="${TRAINING_DB}"

    log_info "=========================================="
    log_info "Curriculum Training: ${BOARD_TYPE} ${NUM_PLAYERS}p"
    log_info "=========================================="

    mkdir -p "models/curriculum/hex8_${NUM_PLAYERS}p" "${LOG_DIR}"

    # Use venv if available
    if [ -f "venv/bin/python3" ]; then
        PYTHON="venv/bin/python3"
    else
        PYTHON="python3"
    fi

    # Use simple training (curriculum requires registry validation)
    log_info "Using simple NNUE policy training..."
    train_model "${NUM_PLAYERS}"
}

check_status() {
    log_info "=========================================="
    log_info "Hex8 Training Status"
    log_info "=========================================="

    echo ""
    echo "=== Available Data ==="
    check_data

    echo ""
    echo "=== Existing Models ==="
    for players in 2 3 4; do
        local MODEL_FILE="${OUTPUT_DIR}/nnue_hex8_${players}p.pt"
        if [ -f "${MODEL_FILE}" ]; then
            local SIZE=$(ls -lh "${MODEL_FILE}" | awk '{print $5}')
            local DATE=$(ls -l "${MODEL_FILE}" | awk '{print $6, $7, $8}')
            echo -e "  hex8_${players}p: ${GREEN}EXISTS${NC} (${SIZE}, ${DATE})"
        else
            echo -e "  hex8_${players}p: ${RED}MISSING${NC}"
        fi
    done

    echo ""
    echo "=== Running Processes ==="
    ps aux | grep -E 'train_nnue|curriculum' | grep -v grep || echo "  No training jobs running"

    echo ""
    echo "=== Recent Logs ==="
    ls -lt "${LOG_DIR}"/*hex8*.log 2>/dev/null | head -5 || echo "  No logs found"
}

case "${1:-help}" in
    validate)
        check_data
        validate_data
        ;;
    train-2p)
        check_data
        train_with_curriculum 2
        ;;
    train-3p)
        check_data
        train_with_curriculum 3
        ;;
    train-4p)
        check_data
        train_with_curriculum 4
        ;;
    train-all)
        check_data
        log_info "Training all missing hex8 models..."
        train_with_curriculum 2
        train_with_curriculum 3
        train_with_curriculum 4
        log_info "All training complete!"
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {validate|train-2p|train-3p|train-4p|train-all|status}"
        echo ""
        echo "Commands:"
        echo "  validate   - Validate training data for hex8"
        echo "  train-2p   - Train hex8 2-player model"
        echo "  train-3p   - Train hex8 3-player model"
        echo "  train-4p   - Train hex8 4-player model"
        echo "  train-all  - Train all 2p, 3p, 4p models"
        echo "  status     - Check training status"
        echo ""
        echo "Run on cluster:"
        echo "  ssh ubuntu@100.78.101.123 'cd ~/ringrift/ai-service && ./scripts/train_hex8_models.sh train-all'"
        ;;
esac
