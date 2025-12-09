#!/bin/bash
# Run iterative CMA-ES across all board/player configurations
# This script runs each config sequentially, with iterative self-play improvement
#
# Usage:
#   ./run_all_iterative_cmaes.sh [MODE]
#
# Modes:
#   lan      - Local Mac cluster only (default)
#   aws      - AWS staging only (good for square8 configs)
#   hybrid   - Both LAN and AWS workers (maximum parallelism)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse mode argument
MODE="${1:-lan}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="logs/cmaes/iterative_multiconfig_${TIMESTAMP}"
LOG_FILE="${OUTPUT_BASE}/runner.log"

# Worker URLs by mode
LAN_WORKERS="http://Mac-Studio.local:8765,http://10.0.0.193:8765"
AWS_WORKERS="http://3.236.54.231:8766"

case "$MODE" in
    lan)
        WORKERS="$LAN_WORKERS"
        ;;
    aws)
        WORKERS="$AWS_WORKERS"
        ;;
    hybrid)
        WORKERS="${LAN_WORKERS},${AWS_WORKERS}"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [lan|aws|hybrid]"
        exit 1
        ;;
esac

# Config: generations per iteration, max iterations, population, games per eval
GENS_PER_ITER=15
MAX_ITERATIONS=8
POPULATION=20
GAMES_PER_EVAL=12

mkdir -p "$OUTPUT_BASE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=============================================="
log "ITERATIVE MULTI-CONFIG CMA-ES TRAINING"
log "=============================================="
log "Mode: $MODE"
log "Output: $OUTPUT_BASE"
log "Workers: $WORKERS"
log "Generations per iter: $GENS_PER_ITER"
log "Max iterations: $MAX_ITERATIONS"
log "Population: $POPULATION"
log "Games per eval: $GAMES_PER_EVAL"
log ""

# Define all configurations
CONFIGS=(
    "square8:2"
    "square8:3"
    "square8:4"
    "square19:2"
    "square19:3"
    "square19:4"
    "hex:2"
    "hex:3"
    "hex:4"
)

TOTAL_CONFIGS=${#CONFIGS[@]}
COMPLETED=0

for config in "${CONFIGS[@]}"; do
    BOARD="${config%%:*}"
    PLAYERS="${config##*:}"
    CONFIG_DIR="${OUTPUT_BASE}/${BOARD}_${PLAYERS}p"
    CONFIG_LOG="${CONFIG_DIR}/training.log"

    ((COMPLETED++))
    log ""
    log "=============================================="
    log "CONFIG ${COMPLETED}/${TOTAL_CONFIGS}: ${BOARD} ${PLAYERS}p"
    log "=============================================="
    log "Output: $CONFIG_DIR"

    mkdir -p "$CONFIG_DIR"

    PYTHONPATH=. RINGRIFT_SKIP_SHADOW_CONTRACTS=true python scripts/run_iterative_cmaes.py \
        --board "$BOARD" \
        --num-players "$PLAYERS" \
        --generations-per-iter "$GENS_PER_ITER" \
        --max-iterations "$MAX_ITERATIONS" \
        --population-size "$POPULATION" \
        --games-per-eval "$GAMES_PER_EVAL" \
        --sigma 0.5 \
        --output-dir "$CONFIG_DIR" \
        --distributed \
        --workers "$WORKERS" \
        2>&1 | tee "$CONFIG_LOG"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "CONFIG ${BOARD}_${PLAYERS}p: COMPLETED SUCCESSFULLY"
    else
        log "CONFIG ${BOARD}_${PLAYERS}p: FAILED (exit code $EXIT_CODE)"
    fi

    # Brief pause between configs
    sleep 5
done

log ""
log "=============================================="
log "ALL CONFIGURATIONS COMPLETED"
log "=============================================="
log "Total configs: $TOTAL_CONFIGS"
log "Output directory: $OUTPUT_BASE"
