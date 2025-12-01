#!/bin/bash
# RingRift Test Summary Script
# Outputs a summary of test health across TypeScript and Python test suites

set -e

echo "=== RingRift Test Summary ==="
echo ""

# TypeScript/Jest tests
echo "📦 TypeScript Tests (Jest):"
if command -v npm &> /dev/null; then
    # Run tests in JSON mode and extract summary
    npm test -- --json --passWithNoTests 2>/dev/null | tail -1 | jq -r '
        "  Passed: \(.numPassedTests // 0)",
        "  Failed: \(.numFailedTests // 0)", 
        "  Skipped: \(.numPendingTests // 0)",
        "  Total: \(.numTotalTests // 0)"
    ' 2>/dev/null || echo "  (Run 'npm test' for detailed results)"
else
    echo "  npm not found - skipping TypeScript tests"
fi

echo ""

# Python tests
echo "🐍 Python Tests (pytest):"
if [ -d "ai-service" ]; then
    cd ai-service
    if command -v pytest &> /dev/null; then
        pytest --collect-only -q 2>/dev/null | tail -1 || echo "  (Run 'cd ai-service && pytest' for detailed results)"
    elif [ -f "requirements.txt" ]; then
        echo "  pytest not installed - run 'pip install -r requirements.txt'"
    fi
    cd ..
else
    echo "  ai-service directory not found"
fi

echo ""

# Contract vectors count
echo "📋 Contract Vectors:"
if [ -d "tests/fixtures/contract-vectors/v2" ]; then
    vector_count=$(find tests/fixtures/contract-vectors/v2 -name "*.json" | wc -l | tr -d ' ')
    echo "  Count: $vector_count vectors"
else
    echo "  Contract vectors directory not found"
fi

echo ""
echo "=== End Summary ==="