#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT_DIR/ai-service"
COVERAGE_DIR="$ROOT_DIR/coverage"
COVERAGE_FILE_PATH="$COVERAGE_DIR/.coverage.python-training-contracts"
COVERAGE_JSON="$COVERAGE_DIR/python-training-contracts.json"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$AI_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$AI_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

INCLUDE_FILES=(
  "app/training/board_encoding_contract.py"
  "app/training/encoding.py"
  "app/training/feature_registry.py"
  "app/training/model_versioning.py"
  "scripts/minimal_alphazero_loop.py"
  "scripts/lib/training_probes.py"
)

TESTS=(
  "tests/contracts/test_board_encoding_contract.py"
  "tests/contracts/test_channel_contract_matrix.py"
  "tests/contracts/test_training_pipeline_contracts.py"
  "tests/unit/training/test_board_encoding_contract.py"
  "tests/unit/training/test_encoding.py"
  "tests/unit/training/test_feature_registry.py"
  "tests/unit/training/test_model_versioning.py"
  "tests/unit/scripts/test_minimal_alphazero_loop.py"
  "tests/unit/scripts/test_training_probes.py"
)

mkdir -p "$COVERAGE_DIR"

INCLUDE_CSV="$(IFS=,; echo "${INCLUDE_FILES[*]}")"

cd "$AI_DIR"
rm -f "$COVERAGE_FILE_PATH" "$COVERAGE_JSON"

# Avoid pytest-cov/source preloading here: local PyTorch 2.6 import behavior can
# conflict with pytest-cov's collection path. coverage.py direct mode records the
# same tests without pre-importing the measured modules.
PYTHONPATH=. COVERAGE_FILE="$COVERAGE_FILE_PATH" "$PYTHON_BIN" -m coverage run \
  -m pytest "${TESTS[@]}" -q --timeout=120

COVERAGE_FILE="$COVERAGE_FILE_PATH" "$PYTHON_BIN" -m coverage report \
  --precision=2 \
  --include="$INCLUDE_CSV"

COVERAGE_FILE="$COVERAGE_FILE_PATH" "$PYTHON_BIN" -m coverage json \
  -o "$COVERAGE_JSON" \
  --include="$INCLUDE_CSV" >/dev/null

"$PYTHON_BIN" - "$COVERAGE_JSON" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

coverage_json = Path(sys.argv[1])
data = json.loads(coverage_json.read_text(encoding="utf-8"))

thresholds = {
    "TOTAL": 62.0,
    "app/training/board_encoding_contract.py": 74.0,
    "app/training/encoding.py": 72.0,
    "app/training/feature_registry.py": 93.0,
    "app/training/model_versioning.py": 46.0,
    "scripts/lib/training_probes.py": 86.0,
    "scripts/minimal_alphazero_loop.py": 51.0,
}

actuals = {"TOTAL": float(data["totals"]["percent_covered"])}
actuals.update(
    {
        filename: float(file_data["summary"]["percent_covered"])
        for filename, file_data in data["files"].items()
    }
)

failures: list[str] = []
for filename, minimum in thresholds.items():
    actual = actuals.get(filename)
    if actual is None:
        failures.append(f"{filename}: missing from coverage report")
        continue
    if actual < minimum:
        failures.append(f"{filename}: {actual:.2f}% < {minimum:.2f}%")

if failures:
    joined = "\n".join(failures)
    raise SystemExit(f"Python training contract coverage below ratchet:\n{joined}")

print("Python training contract coverage ratchet passed:")
for filename, minimum in thresholds.items():
    print(f"  {filename}: {actuals[filename]:.2f}% >= {minimum:.2f}%")
PY
