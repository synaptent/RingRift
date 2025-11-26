#!/bin/bash
# ============================================================================
# Python Contract Test Runner
# ============================================================================
# Run Python rules engine contract tests against shared test vectors.
# This script validates cross-language parity between TypeScript and Python.
#
# Usage:
#   ./scripts/run-python-contract-tests.sh
#   ./scripts/run-python-contract-tests.sh --verbose
#   ./scripts/run-python-contract-tests.sh --quick
#
# Exit codes:
#   0 - All tests passed
#   1 - Some tests failed
#   2 - Setup/configuration error
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_SERVICE_DIR="$PROJECT_ROOT/ai-service"

# Default settings
VERBOSE=false
QUICK=false
TIMEOUT=120

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Show detailed test output"
            echo "  -q, --quick      Skip slow tests (none currently)"
            echo "  -t, --timeout    Set test timeout in seconds (default: 120)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 2
            ;;
    esac
done

# Header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE} Python Contract Test Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if ai-service directory exists
if [ ! -d "$AI_SERVICE_DIR" ]; then
    echo -e "${RED}Error: ai-service directory not found${NC}"
    exit 2
fi

# Check if test vectors exist
VECTORS_DIR="$PROJECT_ROOT/tests/fixtures/contract-vectors/v2"
if [ ! -d "$VECTORS_DIR" ]; then
    echo -e "${RED}Error: Test vectors directory not found: $VECTORS_DIR${NC}"
    exit 2
fi

# Count test vectors
VECTOR_COUNT=$(find "$VECTORS_DIR" -name "*.vectors.json" -exec cat {} \; | \
    grep -o '"id"' | wc -l | tr -d ' ')
echo -e "${BLUE}Found ${VECTOR_COUNT} test vectors in:${NC}"
echo "  $VECTORS_DIR"
echo ""

# Change to ai-service directory
cd "$AI_SERVICE_DIR"

# Check if virtual environment or Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 2
fi

# Build pytest arguments
PYTEST_ARGS=("tests/contracts/test_contract_vectors.py")

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v" "--tb=short")
else
    PYTEST_ARGS+=("-v" "--tb=line")
fi

PYTEST_ARGS+=("--timeout=$TIMEOUT")

# Run tests
echo -e "${BLUE}Running Python contract tests...${NC}"
echo ""

if python -m pytest "${PYTEST_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN} All Python contract tests passed!${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED} Some Python contract tests failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Check if Python rules engine matches TypeScript semantics"
    echo "  2. Review the failed assertions in the output above"
    echo "  3. Compare with TypeScript test results:"
    echo "     npm test -- tests/contracts/contractVectorRunner.test.ts"
    exit 1
fi