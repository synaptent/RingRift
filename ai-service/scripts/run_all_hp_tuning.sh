#!/bin/bash
# Run all 9 hyperparameter tuning configurations sequentially on a single host
# This ensures we have the selfplay.db data available
#
# Usage: ./scripts/run_all_hp_tuning.sh [host]
# Example: ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && ./scripts/run_all_hp_tuning.sh'

set -e

TRIALS=50
EPOCHS=15
DB="data/games/selfplay.db"
OUTPUT_BASE="logs/hp_tuning"

# Activate venv if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "COMPREHENSIVE HYPERPARAMETER TUNING"
echo "=========================================="
echo "Trials per config: $TRIALS"
echo "Epochs per trial: $EPOCHS"
echo "Database: $DB"
echo ""

# Define all configs with their trial counts
# More trials for configs with more data
declare -a CONFIGS=(
    "square8:2:50"    # 1014 games
    "square8:3:50"    # 2442 games
    "square8:4:50"    # 2076 games
    "square19:2:30"   # 201 games
    "square19:3:30"   # 320 games
    "square19:4:30"   # 126 games
    "hexagonal:2:30"  # 133 games
    "hexagonal:3:30"  # 301 games
    "hexagonal:4:20"  # 32 games
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r board players trials <<< "$config"
    config_key="${board}_${players}p"

    echo ""
    echo "=========================================="
    echo "TUNING: $config_key ($trials trials)"
    echo "=========================================="
    echo "Started: $(date)"

    python3 scripts/tune_hyperparameters.py \
        --board "$board" \
        --players "$players" \
        --trials "$trials" \
        --epochs "$EPOCHS" \
        --db "$DB" \
        --output-dir "${OUTPUT_BASE}/${config_key}" \
        2>&1 | tee "${OUTPUT_BASE}/${config_key}.log"

    echo "Completed: $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "ALL TUNING COMPLETE"
echo "=========================================="
echo ""

# Show summary
echo "Results summary:"
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r board players trials <<< "$config"
    config_key="${board}_${players}p"

    if [ -f "${OUTPUT_BASE}/${config_key}/tuning_session.json" ]; then
        best=$(python3 -c "import json; d=json.load(open('${OUTPUT_BASE}/${config_key}/tuning_session.json')); print(f'{d.get(\"best_score\", -1):.4f}')" 2>/dev/null || echo "?")
        echo "  $config_key: best_score=$best"
    else
        echo "  $config_key: no results"
    fi
done

echo ""
echo "Updated hyperparameters saved to config/hyperparameters.json"
