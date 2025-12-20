#!/bin/bash
set -euo pipefail

#
# RingRift Target Scale SHORT Load Test Runner
#
# Runs a condensed 12-minute version of the target-scale test that stays
# within the JWT token TTL (15 minutes), avoiding token expiration issues.
#
# Target: 100 concurrent games with 300 players (same as full target-scale)
# Duration: 12 minutes (vs 30 minutes for full test)
#
# Usage:
#   ./tests/load/scripts/run-target-scale-short.sh [--local|--staging]
#
# Environment Variables:
#   Same as run-target-scale.sh - see that file for full documentation.
#   SKIP_PREFLIGHT_CHECKS - Set to 'true' to skip extended preflight validation
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"
SCENARIO_ID_DEFAULT="BCAP_SQ8_3P_TARGET_SHORT_100G_300P"
SCENARIO_ID="${SCENARIO_ID:-$SCENARIO_ID_DEFAULT}"
EXPECTED_VUS=300
EXPECTED_DURATION_S=720

# Default to staging
TARGET="${1:-staging}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$LOAD_DIR/results"
RESULT_FILE="$RESULTS_DIR/${SCENARIO_ID}_${TARGET}_${TIMESTAMP}.json"
SUMMARY_FILE="$RESULTS_DIR/${SCENARIO_ID}_${TARGET}_${TIMESTAMP}_summary.json"

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
  --help|-h)
    echo "Usage: $0 [--local|--staging]"
    echo ""
    echo "Run SHORT target scale load test (12 minutes, within token TTL)."
    echo "Target: 100 concurrent games, 300 concurrent players"
    echo ""
    echo "This is a condensed version of the full 30-minute target-scale test."
    echo "Use this when you need results within the 15-minute JWT token TTL."
    echo ""
    echo "Options:"
    echo "  --local     Run against local development server (localhost:3001)"
    echo "  --staging   Run against staging environment (default)"
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SKIP_PREFLIGHT_CHECKS  Skip extended preflight validation"
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
echo "║     RingRift TARGET SCALE (SHORT) Load Test            ║"
echo "║     Target: 100 Concurrent Games / 300 Players         ║"
echo "║     Duration: ~12 minutes (within token TTL)           ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Scenario ID:  $SCENARIO_ID"
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
    exit 1
fi

log_info "k6 version: $(k6 version)"

# Pre-flight checks
echo ""
log_info "Running pre-flight checks..."

if [[ "${SKIP_PREFLIGHT_CHECKS:-false}" != "true" ]]; then
    log_info "Running extended preflight validation..."
    BASE_URL="$BASE_URL" AI_SERVICE_URL="${AI_SERVICE_URL:-}" \
        node "$LOAD_DIR/scripts/preflight-check.js" \
        --expected-vus "$EXPECTED_VUS" \
        --expected-duration-s "$EXPECTED_DURATION_S"
else
    log_warning "Skipping extended preflight checks (SKIP_PREFLIGHT_CHECKS=true)"
fi

# Health check
HEALTH_URL="$BASE_URL/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Server is healthy at $BASE_URL"
else
    log_error "Health check failed at $HEALTH_URL"
    exit 1
fi

# Login pre-flight
if [[ -z "${LOADTEST_EMAIL:-}" || -z "${LOADTEST_PASSWORD:-}" ]]; then
    log_error "LOADTEST_EMAIL and LOADTEST_PASSWORD must be set."
    exit 1
fi

LOGIN_PAYLOAD=$(jq -n --arg email "$LOADTEST_EMAIL" --arg pass "$LOADTEST_PASSWORD" '{email: $email, password: $pass}')
LOGIN_STATUS=$(curl -s -o /dev/null -w '%{http_code}' \
  -X POST "${BASE_URL}/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "$LOGIN_PAYLOAD") || LOGIN_STATUS=000

if [[ "$LOGIN_STATUS" != "200" ]]; then
    log_error "Login pre-flight failed (status=${LOGIN_STATUS})"
    exit 1
else
    log_success "Login pre-flight succeeded for ${LOADTEST_EMAIL}"
fi

# Phases info
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  SHORT TARGET SCALE TEST (~12 minutes)                 ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Phases (condensed to fit within 15min token TTL):     ║"
echo "║    1. Warmup:       1m  (30 VUs)                       ║"
echo "║    2. Ramp to 50%:  2m  (150 VUs)                      ║"
echo "║    3. Ramp to 100%: 2m  (300 VUs)                      ║"
echo "║    4. Steady 100%:  5m  (300 VUs) <- TARGET VALIDATION ║"
echo "║    5. Ramp down:    2m  (0 VUs)                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [[ "${SKIP_CONFIRM:-false}" != "true" ]]; then
    read -p "Continue with SHORT target scale test? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Test aborted by user."
        exit 0
    fi
fi

# Run the test
echo ""
log_info "🚀 Starting SHORT target scale load test..."
log_info "This will take approximately 12 minutes."
echo ""

SCENARIO_FILE="$LOAD_DIR/scenarios/concurrent-games.js"

if [[ ! -f "$SCENARIO_FILE" ]]; then
    log_error "Scenario file not found: $SCENARIO_FILE"
    exit 1
fi

# Build k6 arguments for short target scale
K6_ARGS=(
    "--env" "BASE_URL=$BASE_URL"
    "--env" "WS_URL=$WS_URL"
    "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
    "--env" "LOAD_PROFILE=target_scale"
    "--env" "SCENARIO_ID=$SCENARIO_ID"
    "--tag" "test=target-scale-short"
    "--tag" "scenario_id=$SCENARIO_ID"
    "--tag" "target=$TARGET"
    "--tag" "timestamp=$TIMESTAMP"
    "--out" "json=$RESULT_FILE"
)

# Add summary directory
K6_ARGS+=("--env" "K6_SUMMARY_DIR=$RESULTS_DIR")

# Short stages: 12 minutes total (within 15-minute token TTL)
# 1m + 2m + 2m + 5m + 2m = 12m
K6_ARGS+=(
    "--stage" "1m:30"     # Warmup
    "--stage" "2m:150"    # Ramp to 50%
    "--stage" "2m:300"    # Ramp to 100%
    "--stage" "5m:300"    # Steady at 100% (target validation)
    "--stage" "2m:0"      # Ramp down
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
    log_success "SHORT target scale test completed successfully"
else
    log_warning "SHORT target scale test completed with threshold violations (exit code: $K6_EXIT_CODE)"
fi

log_info "Test duration: ${DURATION_MINS} minutes"

# Run the results analyzer
echo ""
log_info "📊 Analyzing results..."

ANALYZER_SCRIPT="$SCRIPT_DIR/analyze-results.js"
if [[ -f "$ANALYZER_SCRIPT" ]] && command -v node &> /dev/null; then
    node "$ANALYZER_SCRIPT" "$RESULT_FILE" "$SUMMARY_FILE"
    ANALYZER_EXIT=$?
else
    ANALYZER_EXIT=0
fi

# Print final summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║        SHORT Target Scale Test Complete                ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Scenario:    $SCENARIO_ID"
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

# Exit with combined status
if [[ $K6_EXIT_CODE -ne 0 ]]; then
    exit $K6_EXIT_CODE
elif [[ ${ANALYZER_EXIT:-0} -ne 0 ]]; then
    exit $ANALYZER_EXIT
fi

exit 0
