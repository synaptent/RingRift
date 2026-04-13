#!/usr/bin/env bash
# Ingest policy-bearing selfplay JSONL into a trainer's supplemental NPZ lane.
#
# This wrapper preserves the original entrypoint name but routes through the
# safe supplemental path instead of creating iter_p2p_*.npz files that hijack
# the minimal loop's lexical train window.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${AI_DIR}/venv/bin/python"

if [ ! -f /etc/ringrift/training.conf ]; then
  echo "ERROR: /etc/ringrift/training.conf not found" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/ringrift/training.conf

BOARD_TYPE="${TRAINING_BOARD_TYPE}"
NUM_PLAYERS="${TRAINING_NUM_PLAYERS}"
WORK_DIR="${AI_DIR}/${TRAINING_WORK_DIR}"
SUPPLEMENTAL_DIR="${TRAINING_SUPPLEMENTAL_DATA_DIR:-${WORK_DIR}/supplemental}"
STATE_DIR="${WORK_DIR}/supplemental_state"

WATCH_INTERVAL=0
INPUT_PATHS=("${AI_DIR}/data/selfplay")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH_INTERVAL="$2"; shift 2 ;;
    --input) INPUT_PATHS+=("$2"); shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

run_once() {
  local cmd=(
    "${VENV_PYTHON}" "${AI_DIR}/scripts/ingest_policy_selfplay.py"
    --board-type "${BOARD_TYPE}"
    --num-players "${NUM_PLAYERS}"
    --output-dir "${SUPPLEMENTAL_DIR}"
    --state-dir "${STATE_DIR}"
  )
  for input_path in "${INPUT_PATHS[@]}"; do
    cmd+=(--input "${input_path}")
  done
  (cd "${AI_DIR}" && PYTHONPATH="${AI_DIR}" "${cmd[@]}")
}

if [[ "${WATCH_INTERVAL}" -gt 0 ]]; then
  echo "Watching policy-bearing selfplay for ${BOARD_TYPE}_${NUM_PLAYERS}p every ${WATCH_INTERVAL}s"
  while true; do
    run_once || true
    sleep "${WATCH_INTERVAL}"
  done
else
  run_once
fi
