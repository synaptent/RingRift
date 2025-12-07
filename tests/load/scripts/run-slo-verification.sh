#!/bin/bash
set -euo pipefail

#
# RingRift SLO Verification Pipeline
#
# Runs load test and verifies SLOs in one complete workflow.
# Generates console output, JSON report, and HTML dashboard.
#
# Usage:
#   ./run-slo-verification.sh [target] [--env environment] [--skip-test] [--results-file FILE]
#
# Arguments:
#   target        Environment target: local | staging (default: staging)
#   --env         SLO environment: staging | production (default: staging)
#   --skip-test   Skip load test and use existing results file
#   --results-file Use specific results file (requires --skip-test)
#
# Examples:
#   ./run-slo-verification.sh staging
#   ./run-slo-verification.sh local --env production
#   ./run-slo-verification.sh --skip-test --results-file results/baseline_staging_20251207.json
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"

# Default values
TARGET="staging"
SLO_ENV="staging"
SKIP_TEST=false
RESULTS_FILE=""

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}❌${NC} $1"
}

log_step() {
    echo -e "${CYAN}▶${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            SLO_ENV="$2"
            shift 2
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --results-file)
            RESULTS_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [target] [--env environment] [--skip-test] [--results-file FILE]"
            echo ""
            echo "Run load test and verify SLOs in one workflow."
            echo ""
            echo "Arguments:"
            echo "  target          Environment target: local | staging (default: staging)"
            echo "  --env           SLO environment: staging | production (default: staging)"
            echo "  --skip-test     Skip load test and use existing results"
            echo "  --results-file  Specific results file (requires --skip-test)"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 staging"
            echo "  $0 local --env production"
            echo "  $0 --skip-test --results-file results/baseline_staging_20251207.json"
            exit 0
            ;;
        local|staging)
            TARGET="$1"
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$LOAD_DIR/results"
mkdir -p "$RESULTS_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           RingRift SLO Verification Pipeline                       ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║  Target:          $TARGET"
echo "║  SLO Environment: $SLO_ENV"
echo "║  Skip Test:       $SKIP_TEST"
echo "║  Timestamp:       $TIMESTAMP"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Run load test (unless skipped)
if [ "$SKIP_TEST" = true ]; then
    log_step "Step 1: Skipping load test (using existing results)"
    
    if [ -n "$RESULTS_FILE" ]; then
        LATEST_RESULT="$RESULTS_FILE"
    else
        # Find the latest results file
        LATEST_RESULT=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -1 || true)
    fi
    
    if [ -z "$LATEST_RESULT" ] || [ ! -f "$LATEST_RESULT" ]; then
        log_error "No results file found. Run without --skip-test or specify --results-file"
        exit 1
    fi
    
    log_info "Using results file: $LATEST_RESULT"
else
    log_step "Step 1: Running load test..."
    echo ""
    
    # Check if run-baseline.sh exists and is executable
    BASELINE_SCRIPT="$SCRIPT_DIR/run-baseline.sh"
    if [ ! -f "$BASELINE_SCRIPT" ]; then
        log_error "Baseline script not found: $BASELINE_SCRIPT"
        exit 1
    fi
    
    chmod +x "$BASELINE_SCRIPT"
    
    # Run the baseline load test
    if ! "$BASELINE_SCRIPT" "$TARGET"; then
        log_warning "Load test completed with threshold violations"
    fi
    
    # Find the latest results file
    LATEST_RESULT=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -1 || true)
    
    if [ -z "$LATEST_RESULT" ]; then
        log_error "No load test results found in $RESULTS_DIR"
        exit 1
    fi
fi

echo ""
log_success "Results file: $LATEST_RESULT"

# Step 2: Verify SLOs
echo ""
log_step "Step 2: Verifying SLOs..."
echo ""

SLO_VERIFY_SCRIPT="$SCRIPT_DIR/verify-slos.js"
if [ ! -f "$SLO_VERIFY_SCRIPT" ]; then
    log_error "SLO verification script not found: $SLO_VERIFY_SCRIPT"
    exit 1
fi

# Run SLO verification
SLO_RESULT=0
node "$SLO_VERIFY_SCRIPT" "$LATEST_RESULT" console --env "$SLO_ENV" || SLO_RESULT=$?

# Get the report file path
SLO_REPORT="${LATEST_RESULT%.json}_slo_report.json"

# Step 3: Generate dashboard
echo ""
log_step "Step 3: Generating HTML dashboard..."

DASHBOARD_SCRIPT="$SCRIPT_DIR/generate-slo-dashboard.js"
if [ -f "$DASHBOARD_SCRIPT" ]; then
    if [ -f "$SLO_REPORT" ]; then
        node "$DASHBOARD_SCRIPT" "$SLO_REPORT"
        DASHBOARD_FILE="${SLO_REPORT%.json}.html"
        log_success "Dashboard: $DASHBOARD_FILE"
    else
        log_warning "SLO report not found, skipping dashboard generation"
    fi
else
    log_warning "Dashboard generator not found: $DASHBOARD_SCRIPT"
fi

# Step 4: Generate JSON summary
echo ""
log_step "Step 4: Generating JSON summary..."

if [ -f "$SLO_REPORT" ]; then
    # Create a summary with key metrics
    SUMMARY_FILE="${LATEST_RESULT%.json}_slo_summary.json"
    node -e "
const fs = require('fs');
const report = JSON.parse(fs.readFileSync('$SLO_REPORT', 'utf8'));

const summary = {
    timestamp: report.timestamp,
    environment: report.environment,
    all_passed: report.all_passed,
    passed_count: report.passed_count,
    total_count: report.total_count,
    critical_breaches: Object.values(report.slos).filter(s => !s.passed && s.priority === 'critical').length,
    high_breaches: Object.values(report.slos).filter(s => !s.passed && s.priority === 'high').length,
    key_metrics: {
        availability: report.slos.availability?.actual,
        error_rate: report.slos.error_rate?.actual,
        api_latency_p95: report.slos.latency_api_p95?.actual,
        move_latency_p95: report.slos.latency_move_e2e?.actual,
        concurrent_players: report.slos.concurrent_players?.actual
    }
};

fs.writeFileSync('$SUMMARY_FILE', JSON.stringify(summary, null, 2));
console.log('Summary saved to: $SUMMARY_FILE');
"
    log_success "Summary: $SUMMARY_FILE"
else
    log_warning "No SLO report found for summary generation"
fi

# Final summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    SLO Verification Complete                       ║"
echo "╠════════════════════════════════════════════════════════════════════╣"

if [ $SLO_RESULT -eq 0 ]; then
    echo -e "║  Status: ${GREEN}✅ ALL SLOs PASSED${NC}"
else
    echo -e "║  Status: ${RED}❌ SLO BREACHES DETECTED${NC}"
fi

echo "║"
echo "║  Output Files:"
echo "║    📊 Results:   $LATEST_RESULT"
echo "║    📋 SLO Report: ${SLO_REPORT:-N/A}"
echo "║    🌐 Dashboard:  ${DASHBOARD_FILE:-N/A}"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Return appropriate exit code
if [ $SLO_RESULT -eq 0 ]; then
    log_success "SLO Verification PASSED"
    exit 0
else
    log_error "SLO Verification FAILED"
    log_info "Review the SLO report and dashboard for details."
    exit 1
fi