#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT_DIR/ai-service"

echo "==> Script smoke"
bash "$ROOT_DIR/scripts/run_proven_experiment.sh" hex8_2p --print-only >/dev/null
bash "$ROOT_DIR/scripts/run_proven_experiment.sh" square8_2p --print-only >/dev/null

echo "==> Results artifact refresh"
python3 "$ROOT_DIR/scripts/refresh_results_artifacts.py" --dry-run >/dev/null

echo "==> TypeScript supported-path gates"
cd "$ROOT_DIR"
npm run test:ts-rules-engine
npm run test:orchestrator-parity

echo "==> Python minimal-loop gates"
cd "$AI_DIR"
PYTHONPATH=. "${PYTHON:-python3}" -m pytest \
  tests/unit/scripts/test_minimal_alphazero_loop.py \
  tests/unit/training/test_train_cli.py \
  -q

echo "Supported path checks passed."
