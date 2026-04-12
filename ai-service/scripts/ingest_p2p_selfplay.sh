#!/usr/bin/env bash
# Ingest P2P-generated selfplay JSONL into the minimal loop training pipeline.
#
# P2P's generate_gumbel_selfplay.py already records MCTS policies in JSONL files.
# This script finds those JSONL files, converts them to NPZ, and drops them into
# the minimal loop's work directory where they'll be picked up by the next
# training iteration's data window.
#
# Usage:
#   # One-shot: ingest all pending JSONL files
#   ./scripts/ingest_p2p_selfplay.sh
#
#   # Daemon mode: watch for new JSONL files every N seconds
#   ./scripts/ingest_p2p_selfplay.sh --watch 300
#
# The script reads /etc/ringrift/training.conf to determine the board type,
# num players, and work directory for the local training loop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${AI_DIR}/venv/bin/python"

# Read training config
if [ -f /etc/ringrift/training.conf ]; then
    source /etc/ringrift/training.conf
else
    echo "ERROR: /etc/ringrift/training.conf not found"
    exit 1
fi

BOARD_TYPE="${TRAINING_BOARD_TYPE}"
NUM_PLAYERS="${TRAINING_NUM_PLAYERS}"
WORK_DIR="${AI_DIR}/${TRAINING_WORK_DIR}"
INGEST_DIR="${WORK_DIR}/p2p_ingest"
PROCESSED_DIR="${INGEST_DIR}/processed"

mkdir -p "${INGEST_DIR}" "${PROCESSED_DIR}"

WATCH_INTERVAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch) WATCH_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

ingest_once() {
    local config_key="${BOARD_TYPE}_${NUM_PLAYERS}p"
    local found=0

    # Find JSONL files from P2P gumbel selfplay matching our config
    for jsonl in $(find "${AI_DIR}/data/selfplay" -name "gumbel_${config_key}.jsonl" -newer "${PROCESSED_DIR}/.last_ingest" 2>/dev/null || \
                   find "${AI_DIR}/data/selfplay" -name "gumbel_${config_key}.jsonl" 2>/dev/null); do
        local games=$(wc -l < "$jsonl" 2>/dev/null || echo 0)
        if [ "$games" -lt 5 ]; then
            continue  # Skip tiny files
        fi

        local basename=$(basename "$(dirname "$jsonl")")
        local npz_name="p2p_${basename}.npz"
        local npz_path="${WORK_DIR}/${npz_name}"

        # Skip if already processed
        if [ -f "${PROCESSED_DIR}/${npz_name}.done" ]; then
            continue
        fi

        echo "Ingesting: $jsonl ($games games) -> $npz_path"

        # Convert JSONL to NPZ
        cd "${AI_DIR}"
        PYTHONPATH="${AI_DIR}" "${VENV_PYTHON}" scripts/jsonl_to_npz.py \
            --input "$jsonl" \
            --output "$npz_path" \
            --board-type "${BOARD_TYPE}" \
            --num-players "${NUM_PLAYERS}" \
            --gpu-selfplay 2>&1 | tail -3

        if [ -f "$npz_path" ]; then
            # Rename to iter_PNNN.npz so the minimal loop's glob("iter_*.npz") picks it up
            local seq=$(ls "${WORK_DIR}"/iter_*.npz 2>/dev/null | wc -l)
            local iter_name="iter_p2p_$(printf '%03d' $seq).npz"
            mv "$npz_path" "${WORK_DIR}/${iter_name}"
            touch "${PROCESSED_DIR}/${npz_name}.done"
            echo "  -> ${iter_name} ($(du -h "${WORK_DIR}/${iter_name}" | cut -f1))"
            found=$((found + 1))
        else
            echo "  FAILED: NPZ conversion produced no output"
        fi
    done

    touch "${PROCESSED_DIR}/.last_ingest"

    if [ "$found" -gt 0 ]; then
        echo "Ingested $found P2P selfplay files"
    fi
}

if [ "$WATCH_INTERVAL" -gt 0 ]; then
    echo "Watching for P2P selfplay data every ${WATCH_INTERVAL}s (board=${BOARD_TYPE}, players=${NUM_PLAYERS})"
    while true; do
        ingest_once
        sleep "$WATCH_INTERVAL"
    done
else
    ingest_once
fi
