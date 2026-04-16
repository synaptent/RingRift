#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_proven_experiment.sh <hex8_2p|square8_2p> [options]

Options:
  --iterations N     Number of loop iterations to run (default: 10)
  --work-dir DIR     Work directory relative to ai-service/ or absolute path
  --python PATH      Python interpreter to use
  --print-only       Print the resolved command without running it
  --help             Show this message

Notes:
  - This script launches the supported minimal-loop configurations used for the
    published RingRift training results.
  - Artifacts are written to ai-service/data/proven_experiments/<config>/ by
    default.
  - A single iteration proves the pipeline. Multiple iterations are required
    before promotions become likely.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT_DIR/ai-service"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG="$1"
shift

ITERATIONS=10
WORK_DIR=""
PRINT_ONLY=0
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
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

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="data/proven_experiments/$CONFIG"
fi

case "$CONFIG" in
  hex8_2p)
    MODEL_PATH="models/canonical_hex8_2p.pth"
    LOOP_ARGS=(
      --board-type hex8
      --num-players 2
      --games-per-iter 100
      --selfplay-budget 200
      --eval-budget 128
      --lr 5e-5
      --lr-schedule fixed
      --train-lr-scheduler none
      --train-window 5
      --iterations "$ITERATIONS"
    )
    ;;
  square8_2p)
    MODEL_PATH="models/canonical_square8_2p.pth"
    LOOP_ARGS=(
      --board-type square8
      --num-players 2
      --games-per-iter 100
      --selfplay-budget 128
      --eval-budget 128
      --lr 5e-5
      --lr-schedule fixed
      --train-lr-scheduler none
      --train-window 3
      --iterations "$ITERATIONS"
    )
    ;;
  *)
    echo "Unsupported config: $CONFIG" >&2
    usage
    exit 1
    ;;
esac

LOOP_CMD=(
  "$PYTHON_BIN"
  scripts/minimal_alphazero_loop.py
  --model "$MODEL_PATH"
  --work-dir "$WORK_DIR"
  "${LOOP_ARGS[@]}"
)

echo "RingRift proven experiment"
echo "  config:      $CONFIG"
echo "  python:      $PYTHON_BIN"
echo "  work_dir:    $WORK_DIR"
echo "  ai_service:  $AI_DIR"
echo "  model:       $MODEL_PATH"
echo
printf 'Resolved command:\n  '
printf '%q ' "${LOOP_CMD[@]}"
printf '\n'

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  exit 0
fi

cd "$AI_DIR"
export PYTHONPATH=.

"${LOOP_CMD[@]}"

WORK_PATH="$WORK_DIR"
if [[ "$WORK_PATH" != /* ]]; then
  WORK_PATH="$AI_DIR/$WORK_PATH"
fi

METRICS_PATH="$WORK_PATH/metrics.jsonl"
SUMMARY_PATH="$WORK_PATH/summary.json"
BEST_MODEL_PATH="$WORK_PATH/models/best.pth"

if [[ ! -f "$METRICS_PATH" ]]; then
  echo "Expected metrics file was not created: $METRICS_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY' "$CONFIG" "$WORK_PATH" "$METRICS_PATH" "$BEST_MODEL_PATH" "$SUMMARY_PATH"
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

config, work_dir, metrics_path, best_model_path, summary_path = sys.argv[1:6]

metrics_lines = [line for line in Path(metrics_path).read_text().splitlines() if line.strip()]
if not metrics_lines:
    raise SystemExit(f"No metrics rows found in {metrics_path}")

latest = json.loads(metrics_lines[-1])
evaluation = latest.get("evaluation", {})

summary = {
    "config": config,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "work_dir": str(Path(work_dir).resolve()),
    "metrics_path": str(Path(metrics_path).resolve()),
    "best_model_path": str(Path(best_model_path).resolve()),
    "latest_iteration": latest.get("iteration"),
    "estimated_elo": latest.get("estimated_elo"),
    "total_promotions": latest.get("total_promotions"),
    "promoted": latest.get("promoted"),
    "win_rate": evaluation.get("win_rate"),
    "games_played": evaluation.get("games_played", evaluation.get("eval_games")),
    "decision": evaluation.get("decision"),
    "decision_stage": evaluation.get("decision_stage"),
    "selfplay_budget": latest.get("selfplay_budget"),
    "eval_budget": latest.get("eval_budget"),
    "base_lr": latest.get("base_lr"),
    "effective_lr": latest.get("effective_lr"),
    "train_lr_scheduler": latest.get("train_lr_scheduler"),
    "git_sha": latest.get("git_sha"),
}

Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo
echo "Artifacts:"
echo "  metrics:  $METRICS_PATH"
echo "  summary:  $SUMMARY_PATH"
echo "  best:     $BEST_MODEL_PATH"
