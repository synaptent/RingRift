#!/bin/bash
# Mutation testing wrapper
# Usage: ./scripts/run_mutation_testing.sh
#
# This script runs mutation testing on critical modules.
# Currently uses mutmut which works but has some limitations with mutmut 3.x.
# See https://github.com/boxed/mutmut/issues for known issues.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODULE="app/ai/neural_losses.py"
TEST_FILE="tests/unit/ai/test_neural_losses.py"

echo "=== Mutation Testing ==="
echo "Module: $MODULE"
echo "Tests:  $TEST_FILE"
echo ""

# Clean up any previous mutation testing artifacts
rm -rf .mutmut-cache mutants

# Verify the test passes first
echo "Step 1: Running baseline test..."
python -m pytest "$TEST_FILE" -x -q --timeout=30
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline test failed. Fix tests before running mutation testing."
    exit 1
fi
echo "Baseline test passed."
echo ""

# Run mutmut
echo "Step 2: Running mutation testing..."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Use mutmut with explicit settings
mutmut run \
    --paths-to-mutate="$MODULE" \
    --tests-dir=tests/ \
    --runner="python -m pytest $TEST_FILE -x -q --tb=no --timeout=30" \
    --no-progress 2>&1 || true

echo ""
echo "=== Mutation Testing Results ==="
mutmut results 2>/dev/null || echo "No results available"

# Calculate survival rate
RESULTS=$(mutmut results 2>/dev/null || echo "")
if [ -n "$RESULTS" ]; then
    SURVIVED=$(echo "$RESULTS" | grep -oP 'Survived: \K\d+' || echo "0")
    KILLED=$(echo "$RESULTS" | grep -oP 'Killed: \K\d+' || echo "0")
    TOTAL=$((SURVIVED + KILLED))

    if [ "$TOTAL" -gt 0 ]; then
        KILL_RATE=$((KILLED * 100 / TOTAL))
        echo ""
        echo "Summary:"
        echo "  Killed:    $KILLED"
        echo "  Survived:  $SURVIVED"
        echo "  Kill rate: ${KILL_RATE}%"

        if [ "$KILL_RATE" -lt 70 ]; then
            echo ""
            echo "WARNING: Kill rate below 70%. Consider adding more thorough tests."
        fi
    fi
fi

echo ""
echo "Mutation testing complete."
echo ""
echo "To see surviving mutants:"
echo "  mutmut results"
echo "  mutmut show <id>  # Show specific mutant"
