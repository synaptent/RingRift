#!/usr/bin/env bash
#
# Convenience wrapper for running a square8 2-player Descent vs MCTS
# training experiment using scripts/run_descent_vs_mcts_experiment.py.
#
# Usage (from ai-service/):
#   chmod +x scripts/run_sq8_descent_vs_mcts.sh
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
#     scripts/run_sq8_descent_vs_mcts.sh
#
# Optional environment overrides:
#   GAMES       - self-play games per engine (default: 500)
#   EPOCHS      - training epochs per model (default: 20)
#   EVAL_GAMES  - evaluation games per matchup (default: 100)
#   SEED        - base random seed (default: 42)
#   OUTPUT_DIR  - root experiments dir (default: experiments/descent_vs_mcts_sq8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GAMES="${GAMES:-500}"
EPOCHS="${EPOCHS:-20}"
EVAL_GAMES="${EVAL_GAMES:-100}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/descent_vs_mcts_sq8}"

echo "Running square8 2p Descent vs MCTS experiment:"
echo "  games per engine : ${GAMES}"
echo "  epochs per model : ${EPOCHS}"
echo "  eval games       : ${EVAL_GAMES}"
echo "  seed             : ${SEED}"
echo "  output dir       : ${OUTPUT_DIR}"
echo

PYTHONPATH="${PYTHONPATH:-.}" \
python scripts/run_descent_vs_mcts_experiment.py \
  --games "${GAMES}" \
  --epochs "${EPOCHS}" \
  --eval-games "${EVAL_GAMES}" \
  --board-type square8 \
  --num-players 2 \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}"

