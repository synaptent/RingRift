#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_quick_eval.sh <hex8_2p|square8_2p> [options]

Options:
  --games N       Number of games to run (default: 12)
  --output PATH   Output JSON path (default: ai-service/results/quick_eval_<config>.json)
  --python PATH   Python interpreter to use
  --print-only    Print the resolved command without running it
  --help          Show this message

This is the fastest outsider-facing model check: a canonical neural policy
checkpoint versus the built-in baseline heuristic, with color/seat rotation.
It is not a publication-grade Elo run; it is a reproducible smoke/eval path.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT_DIR/ai-service"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "Neither shasum nor sha256sum is available for checkpoint verification." >&2
    exit 2
  fi
}

verify_checkpoint_sidecar() {
  local model_path="$1"
  local absolute_model="$AI_DIR/$model_path"
  local sidecar="$absolute_model.sha256"

  if [[ ! -f "$absolute_model" ]]; then
    echo "Missing checkpoint: $absolute_model" >&2
    echo "Publish or download the model before running quick evaluation." >&2
    exit 2
  fi

  if [[ ! -f "$sidecar" ]]; then
    echo "Missing checkpoint checksum sidecar: $sidecar" >&2
    echo "Quick evaluation requires a verified .sha256 sidecar." >&2
    exit 2
  fi

  local expected actual
  expected="$(awk 'NF {print $1; exit}' "$sidecar")"
  actual="$(sha256_file "$absolute_model")"

  if [[ "$actual" != "$expected" ]]; then
    cat >&2 <<EOF
Checkpoint checksum mismatch for $model_path
  expected: $expected
  actual:   $actual

Do not bypass this check for public evidence. Publish the verified checkpoint
or update the sidecar only after provenance is confirmed.
EOF
    exit 2
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG="$1"
shift

GAMES=12
OUTPUT=""
PRINT_ONLY=0
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games)
      GAMES="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --print-only|--dry-run)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$AI_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$AI_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

case "$CONFIG" in
  square8_2p)
    BOARD="square8"
    NUM_PLAYERS=2
    MODEL_PATH="models/canonical_square8_2p.pth"
    MAX_MOVES=500
    ;;
  hex8_2p)
    BOARD="hex8"
    NUM_PLAYERS=2
    MODEL_PATH="models/canonical_hex8_2p.pth"
    MAX_MOVES=600
    ;;
  *)
    echo "Unsupported config: $CONFIG" >&2
    usage
    exit 1
    ;;
esac

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="results/quick_eval_${CONFIG}.json"
fi

EVAL_CMD=(
  "$PYTHON_BIN"
  scripts/evaluate_ai_models.py
  --player1 policy_only
  --player2 baseline_heuristic
  --games "$GAMES"
  --board "$BOARD"
  --num-players "$NUM_PLAYERS"
  --checkpoint "$MODEL_PATH"
  --output "$OUTPUT"
  --seed 42
  --max-moves "$MAX_MOVES"
)

echo "RingRift quick evaluation"
echo "  config:   $CONFIG"
echo "  python:   $PYTHON_BIN"
echo "  model:    $MODEL_PATH"
echo "  games:    $GAMES"
echo "  output:   $OUTPUT"
echo
printf 'Resolved command:\n  '
printf '%q ' "${EVAL_CMD[@]}"
printf '\n'

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  exit 0
fi

verify_checkpoint_sidecar "$MODEL_PATH"

cd "$AI_DIR"
export PYTHONPATH=.
mkdir -p "$(dirname "$OUTPUT")"
"${EVAL_CMD[@]}"
