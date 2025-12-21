#!/bin/bash
# Run hexagonal ANM parity diagnosis on cluster
# This script generates fresh hex games and analyzes ANM divergences

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/data/anm_diagnosis_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Hexagonal ANM Parity Diagnosis"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Step 1: Generate fresh hex games with parity validation
echo "Step 1: Generating 5 hex games with parity validation..."
python "${SCRIPT_DIR}/generate_gumbel_selfplay.py" \
    --board hexagonal \
    --num-players 2 \
    --games 5 \
    --validate-parity \
    --emit-state-bundles-dir "${OUTPUT_DIR}/bundles" \
    --output "${OUTPUT_DIR}/games.db" \
    2>&1 | tee "${OUTPUT_DIR}/generation.log"

# Step 2: Run parity check with detailed output
echo ""
echo "Step 2: Running parity check..."
python "${SCRIPT_DIR}/check_ts_python_replay_parity.py" \
    --db "${OUTPUT_DIR}/games.db" \
    --emit-state-bundles-dir "${OUTPUT_DIR}/bundles" \
    --json-output "${OUTPUT_DIR}/parity_results.json" \
    2>&1 | tee "${OUTPUT_DIR}/parity.log" || true

# Step 3: Analyze ANM divergences
echo ""
echo "Step 3: Analyzing ANM divergences..."
if [ -f "${OUTPUT_DIR}/parity_results.json" ]; then
    python "${SCRIPT_DIR}/diagnose_anm_divergence.py" \
        --parity-gate "${OUTPUT_DIR}/parity_results.json" \
        --verbose \
        2>&1 | tee "${OUTPUT_DIR}/diagnosis.log"
else
    echo "No parity results to analyze."
fi

# Step 4: Summary
echo ""
echo "============================================================"
echo "Diagnosis Complete"
echo "============================================================"
echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "To review results:"
echo "  cat ${OUTPUT_DIR}/diagnosis.log"
echo ""
echo "To analyze state bundles:"
echo "  ls ${OUTPUT_DIR}/bundles/"
