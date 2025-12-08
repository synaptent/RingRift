#!/bin/bash
set -euo pipefail

#
# RingRift Target Scale Load Test Runner
#
# Tests system at production target: 100 concurrent games with 300 players.
# Aligned with PROJECT_GOALS.md targets for production-scale validation.
#
# Usage:
#   ./tests/load/scripts/run-target-scale.sh [--local|--staging]
#   ./tests/load/scripts/run-target-scale.sh local
#   ./tests/load/scripts/run-target-scale.sh staging
#
# Environment Variables:
#   BASE_URL       - Override the base URL for the target environment
#   STAGING_URL    - URL for staging environment (default: http://localhost:3000)
#   K6_EXTRA_ARGS  - Additional arguments to pass to k6
#   SKIP_CONFIRM   - Set to 'true' to skip the confirmation prompt
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"

# Default to staging
TARGET="${1:-staging}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$LOAD_DIR/results"
RESULT_FILE="$RESULTS_DIR/BCAP_SQ8_3P_TARGET_100G_300P_${TARGET}_${TIMESTAMP}.json"
SUMMARY_FILE="$RESULTS_DIR/BCAP_SQ8_3P_TARGET_100G_300P_${TARGET}_${TIMESTAMP}_summary.json"
CONFIG_FILE="$LOAD_DIR/configs/target-scale.json"

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

# Set base URL based on target
case "$TARGET" in
  --local|local)
    BASE_URL="${BASE_URL:-http://localhost:3001}"
    WS_URL="${WS_URL:-ws://localhost:3001}"
    THRESHOLD_ENV="staging"
    ;;
  --staging|staging)
    BASE_URL="${STAGING_URL:-${BASE_URL:-http://localhost:3000}}"
    WS_URL="${WS_URL:-$(echo "$BASE_URL" | sed 's/^http/ws/')}"
    THRESHOLD_ENV="production"
    ;;
  --production|production)
    log_error "Production target scale tests should be run with extreme caution!"
    log_error "Use --staging for pre-production validation."
    exit 1
    ;;
  --help|-h)
    echo "Usage: $0 [--local|--staging]"
    echo ""
    echo "Run target scale load tests to validate production capacity."
    echo "Target: 100 concurrent games, 300 concurrent players"
    echo ""
    echo "Options:"
    echo "  --local     Run against local development server (localhost:3001)"
    echo "  --staging   Run against staging environment (default)"
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BASE_URL       Override the target URL"
    echo "  STAGING_URL    URL for staging environment"
    echo "  K6_EXTRA_ARGS  Additional k6 arguments"
    echo "  SKIP_CONFIRM   Set to 'true' to skip confirmation prompt"
    echo ""
    echo "Duration: Approximately 30 minutes"
    echo "Resource Requirements: ~8GB RAM, 4+ CPU cores"
    exit 0
    ;;
  *)
    log_error "Unknown target: $TARGET"
    echo "Usage: $0 [--local|--staging]"
    exit 1
    ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║     RingRift TARGET SCALE Load Test                    ║"
echo "║     Target: 100 Concurrent Games / 300 Players         ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Configuration: $(basename "$CONFIG_FILE")"
echo "║  Target:        $TARGET"
echo "║  Base URL:      $BASE_URL"
echo "║  WS URL:        $WS_URL"
echo "║  Results:       $(basename "$RESULT_FILE")"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check for k6 installation
if ! command -v k6 &> /dev/null; then
    log_error "k6 is not installed. Please install it first:"
    echo "  macOS:   brew install k6"
    echo "  Linux:   sudo apt-get install k6  (or use snap/deb package)"
    echo "  Docker:  docker run -i grafana/k6 run - < script.js"
    exit 1
fi

log_info "k6 version: $(k6 version)"

# Pre-flight health check
echo ""
log_info "Running pre-flight checks..."

# 1. Health check
HEALTH_URL="$BASE_URL/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Server is healthy at $BASE_URL"
else
    log_error "Health check failed at $HEALTH_URL"
    log_error "Is the server running? Start with: npm run dev"
    exit 1
fi

# 2. Optionally check AI service health if configured
AI_HEALTH_URL="${AI_SERVICE_URL:-http://localhost:8000}/health"
if curl -sf "$AI_HEALTH_URL" > /dev/null 2>&1; then
    log_success "AI service is healthy"
else
    log_warning "AI service not responding at $AI_HEALTH_URL (optional)"
fi

# 3. Verify staging resources (if staging)
if [[ "$TARGET" == *"staging"* ]]; then
    log_info "Staging environment detected"
    log_warning "Ensure sufficient resources (~8GB RAM, 4+ CPU cores) for target scale test"
    
    # Check if Docker containers are running (if using staging)
    if command -v docker &> /dev/null; then
        CONTAINER_COUNT=$(docker ps --filter "name=ringrift" --format "{{.Names}}" 2>/dev/null | wc -l || echo "0")
        if [[ "$CONTAINER_COUNT" -gt 0 ]]; then
            log_success "Found $CONTAINER_COUNT RingRift containers running"
        fi
    fi
fi

# Warning and confirmation
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ⚠️  WARNING: HIGH LOAD TEST                           ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  This test will apply significant load to the system:  ║"
echo "║    • 300 virtual users (concurrent players)            ║"
echo "║    • 100 concurrent games                              ║"
echo "║    • ~30 minute duration                               ║"
echo "║                                                        ║"
echo "║  Phases:                                               ║"
echo "║    1. Warmup:      2m  (30 VUs)                        ║"
echo "║    2. Ramp to 50%: 5m  (150 VUs)                       ║"
echo "║    3. Steady 50%:  5m  (150 VUs)                       ║"
echo "║    4. Ramp to 100%:5m  (300 VUs)                       ║"
echo "║    5. Steady 100%: 10m (300 VUs) <- TARGET VALIDATION  ║"
echo "║    6. Ramp down:   3m  (0 VUs)                         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [[ "${SKIP_CONFIRM:-false}" != "true" ]]; then
    read -p "Continue with target scale test? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Test aborted by user."
        exit 0
    fi
fi

# Run the target scale test
echo ""
log_info "🚀 Starting target scale load test..."
log_info "This will take approximately 30 minutes."
log_info "Monitor system resources during the test."
echo ""

# Determine which scenario to run - use concurrent-games as the primary scenario
SCENARIO_FILE="$LOAD_DIR/scenarios/concurrent-games.js"

if [[ ! -f "$SCENARIO_FILE" ]]; then
    log_error "Scenario file not found: $SCENARIO_FILE"
    exit 1
fi

# Build k6 arguments for target scale
K6_ARGS=(
    "--env" "BASE_URL=$BASE_URL"
    "--env" "WS_URL=$WS_URL"
    "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
    "--env" "LOAD_PROFILE=target_scale"
    "--tag" "test=target-scale"
    "--tag" "target=$TARGET"
    "--tag" "timestamp=$TIMESTAMP"
    "--out" "json=$RESULT_FILE"
)

# Add summary directory for the scenario's built-in summary
K6_ARGS+=("--env" "K6_SUMMARY_DIR=$RESULTS_DIR")

# Override VUs and duration for target scale
# Total duration: 2m + 5m + 5m + 5m + 10m + 3m = 30m
K6_ARGS+=(
    "--stage" "2m:30"     # Warmup
    "--stage" "5m:150"    # Ramp to 50%
    "--stage" "5m:150"    # Steady at 50%
    "--stage" "5m:300"    # Ramp to 100%
    "--stage" "10m:300"   # Steady at 100% (target validation)
    "--stage" "3m:0"      # Ramp down
)

# Add any extra arguments
if [[ -n "${K6_EXTRA_ARGS:-}" ]]; then
    IFS=' ' read -ra EXTRA_ARGS <<< "$K6_EXTRA_ARGS"
    K6_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Record start time
START_TIME=$(date +%s)

# Run the test
k6 run "${K6_ARGS[@]}" "$SCENARIO_FILE"
K6_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MINS=$((DURATION / 60))

echo ""

if [[ $K6_EXIT_CODE -eq 0 ]]; then
    log_success "Target scale test completed successfully"
else
    log_warning "Target scale test completed with threshold violations (exit code: $K6_EXIT_CODE)"
fi

log_info "Test duration: ${DURATION_MINS} minutes"

# Run the results analyzer
echo ""
log_info "📊 Analyzing target scale results..."

ANALYZER_SCRIPT="$SCRIPT_DIR/analyze-target-scale.js"
if [[ -f "$ANALYZER_SCRIPT" ]]; then
    if command -v node &> /dev/null; then
        node "$ANALYZER_SCRIPT" "$RESULT_FILE" "$SUMMARY_FILE"
        ANALYZER_EXIT=$?
    else
        log_warning "Node.js not available, skipping detailed analysis"
        ANALYZER_EXIT=0
    fi
else
    log_warning "Target scale analyzer script not found at $ANALYZER_SCRIPT"
    log_info "Using generic analyzer..."
    GENERIC_ANALYZER="$SCRIPT_DIR/analyze-results.js"
    if [[ -f "$GENERIC_ANALYZER" ]] && command -v node &> /dev/null; then
        node "$GENERIC_ANALYZER" "$RESULT_FILE" "$SUMMARY_FILE"
        ANALYZER_EXIT=$?
    else
        ANALYZER_EXIT=0
    fi
fi

# Run SLO verifier if available
VERIFIER_SCRIPT="$SCRIPT_DIR/verify-slos.js"
SLO_EXIT_CODE=0
if [[ -f "$VERIFIER_SCRIPT" ]]; then
    if command -v node &> /dev/null; then
        echo ""
        log_info "Verifying SLOs against $THRESHOLD_ENV thresholds..."
        if node "$VERIFIER_SCRIPT" "$RESULT_FILE" console --env "$THRESHOLD_ENV"; then
            log_success "SLO verification passed"
            SLO_EXIT_CODE=0
        else
            log_warning "SLO verification reported failures (see output above)"
            SLO_EXIT_CODE=1
        fi
    else
        log_warning "Node.js not available, skipping SLO verification"
    fi
else
    log_warning "SLO verifier not found at $VERIFIER_SCRIPT"
fi

# Print final summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Target Scale Test Complete                ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Duration:    ${DURATION_MINS} minutes"
echo "║  Results:     $RESULT_FILE"
echo "║  Summary:     $SUMMARY_FILE"
echo "╠════════════════════════════════════════════════════════╣"
if [[ $K6_EXIT_CODE -eq 0 ]] && [[ ${ANALYZER_EXIT:-0} -eq 0 ]]; then
    echo "║  Status:      ✅ TARGET SCALE VALIDATED               ║"
else
    echo "║  Status:      ❌ VALIDATION FAILED - Review results   ║"
fi
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Remind about documentation
if [[ -f "$PROJECT_ROOT/docs/BASELINE_CAPACITY.md" ]]; then
    log_info "If this establishes a new baseline, update: docs/BASELINE_CAPACITY.md"
fi

# Exit with combined status
if [[ $K6_EXIT_CODE -ne 0 ]]; then
    exit $K6_EXIT_CODE
elif [[ ${ANALYZER_EXIT:-0} -ne 0 ]]; then
    exit $ANALYZER_EXIT
elif [[ $SLO_EXIT_CODE -ne 0 ]]; then
    exit $SLO_EXIT_CODE
fi

exit 0
