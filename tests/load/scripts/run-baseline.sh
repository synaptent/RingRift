#!/bin/bash
set -euo pipefail

#
# RingRift Baseline Load Test Runner
#
# Measures current system capacity to establish baseline metrics.
# Aligned with PROJECT_GOALS.md targets: 100+ concurrent games, 300 players, <500ms p95 latency.
#
# Usage:
#   ./tests/load/scripts/run-baseline.sh [--local|--staging]
#   ./tests/load/scripts/run-baseline.sh --local
#   ./tests/load/scripts/run-baseline.sh --staging
#
# Environment Variables:
#   BASE_URL       - Override the base URL for the target environment
#   STAGING_URL    - URL for staging environment (default: http://localhost:3000)
#   K6_EXTRA_ARGS  - Additional arguments to pass to k6
#   SCENARIO_ID    - Scenario identifier/tag (default: BCAP_STAGING_BASELINE_20G_60P)
#   SMOKE          - Set to 1/true to run with the short smoke profile
#   SKIP_WS_COMPANION - Set to 1/true to skip the WebSocket companion run
#   SEED_LOADTEST_USERS - If 'true', seed load-test users before running (uses scripts/seed-loadtest-users.js)
#   LOADTEST_USER_COUNT / LOADTEST_USER_DOMAIN / LOADTEST_USER_OFFSET / LOADTEST_USER_PASSWORD / LOADTEST_USER_ROLE - Seeding overrides
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"

SCENARIO_ID_DEFAULT="BCAP_STAGING_BASELINE_20G_60P"
SCENARIO_ID="${SCENARIO_ID:-$SCENARIO_ID_DEFAULT}"

# Default to local
TARGET="${1:-local}"
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
    # For staging, default to the Docker-based staging stack (app on :3000).
    # STAGING_URL can be used to point at a remote staging host instead.
    BASE_URL="${STAGING_URL:-${BASE_URL:-http://localhost:3000}}"
    WS_URL="${WS_URL:-$(echo "$BASE_URL" | sed 's/^http/ws/') }"
    THRESHOLD_ENV="staging"
    ;;
  --production|production)
    log_error "Production baseline tests should be run with extreme caution!"
    log_error "Use --staging for pre-production validation."
    exit 1
    ;;
  --help|-h)
    echo "Usage: $0 [--local|--staging]"
    echo ""
    echo "Run baseline load tests to establish system capacity."
    echo ""
    echo "Options:"
    echo "  --local     Run against local development server (default)"
    echo "  --staging   Run against staging environment"
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BASE_URL       Override the target URL"
    echo "  STAGING_URL    URL for staging environment"
    echo "  K6_EXTRA_ARGS  Additional k6 arguments"
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
echo "║       RingRift Baseline Load Test Runner               ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Scenario:   $SCENARIO_ID"
echo "║  Target:     $TARGET"
echo "║  Base URL:   $BASE_URL"
echo "║  WS URL:     $WS_URL"
echo "║  Results:    $RESULT_FILE"
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

# Pre-flight checks
echo ""
log_info "Running pre-flight checks..."
HEALTH_URL="$BASE_URL/health"

if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Server is healthy at $BASE_URL"
else
    log_error "Health check failed at $HEALTH_URL"
    log_error "Is the server running? Start with: npm run dev"
    exit 1
fi

# Optionally check AI service health if configured
AI_HEALTH_URL="${AI_SERVICE_URL:-http://localhost:8000}/health"
if curl -sf "$AI_HEALTH_URL" > /dev/null 2>&1; then
    log_success "AI service is healthy"
else
    log_warning "AI service not responding at $AI_HEALTH_URL (optional)"
fi

# Login pre-flight for load-test user
if [[ -z "${LOADTEST_EMAIL:-}" || -z "${LOADTEST_PASSWORD:-}" ]]; then
    log_error "LOADTEST_EMAIL and LOADTEST_PASSWORD must be set for login pre-flight."
    log_error "Example: LOADTEST_EMAIL=baseline_k6_user@loadtest.local LOADTEST_PASSWORD='BaselineTest123!' npm run load:baseline:staging"
    exit 1
fi

LOGIN_STATUS=$(curl -s -o /dev/null -w '%{http_code}' \
  -X POST "${BASE_URL}/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"${LOADTEST_EMAIL}\",\"password\":\"${LOADTEST_PASSWORD}\"}") || LOGIN_STATUS=000

if [[ "$LOGIN_STATUS" != "200" ]]; then
    log_error "Login pre-flight failed (status=${LOGIN_STATUS}) for ${LOADTEST_EMAIL} at ${BASE_URL}/api/auth/login"
    log_error "Ensure load-test user is seeded (npm run load:seed-users) and auth environment variables are correct for the target environment."
    exit 1
else
    log_success "Login pre-flight succeeded for ${LOADTEST_EMAIL}"
fi

# Run the baseline test
echo ""
log_info "Starting baseline load test..."
log_info "This will take approximately 10 minutes (1m warmup + 3m ramp + 5m steady + 1m down)"
echo ""

# Optionally seed load-test users to ensure sufficient accounts exist.
if [[ "${SEED_LOADTEST_USERS:-false}" == "true" ]]; then
    echo ""
    log_info "Seeding load-test users (LOADTEST_USER_COUNT=${LOADTEST_USER_COUNT:-400}, domain=${LOADTEST_USER_DOMAIN:-loadtest.local}, offset=${LOADTEST_USER_OFFSET:-0})..."
    (cd "$PROJECT_ROOT" && npm run load:seed-users) || log_warning "User seeding failed; continuing without seeding"
fi

# Determine which scenario to run - use concurrent-games as the primary baseline
SCENARIO_FILE="$LOAD_DIR/scenarios/concurrent-games.js"

if [[ ! -f "$SCENARIO_FILE" ]]; then
    log_error "Scenario file not found: $SCENARIO_FILE"
    exit 1
fi

# Run k6 with baseline profile
# We use the 'load' profile which has more conservative VU targets suitable for baseline
K6_ARGS=(
    "--env" "BASE_URL=$BASE_URL"
    "--env" "WS_URL=$WS_URL"
    "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
    "--env" "LOAD_PROFILE=load"
    "--env" "SMOKE=${SMOKE:-0}"
    "--env" "SCENARIO_ID=$SCENARIO_ID"
    "--tag" "test=baseline"
    "--tag" "scenario_id=$SCENARIO_ID"
    "--tag" "target=$TARGET"
    "--tag" "timestamp=$TIMESTAMP"
    "--out" "json=$RESULT_FILE"
)

# Add summary directory for the scenario's built-in summary
K6_ARGS+=("--env" "K6_SUMMARY_DIR=$RESULTS_DIR")

# Add any extra arguments
if [[ -n "${K6_EXTRA_ARGS:-}" ]]; then
    IFS=' ' read -ra EXTRA_ARGS <<< "$K6_EXTRA_ARGS"
    K6_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Run the test
k6 run "${K6_ARGS[@]}" "$SCENARIO_FILE"
K6_EXIT_CODE=$?

echo ""

if [[ $K6_EXIT_CODE -eq 0 ]]; then
    log_success "Baseline test completed successfully"
else
    log_warning "Baseline test completed with threshold violations (exit code: $K6_EXIT_CODE)"
fi

# Run the results analyzer
echo ""
log_info "Analyzing results..."

ANALYZER_SCRIPT="$SCRIPT_DIR/analyze-results.js"
if [[ -f "$ANALYZER_SCRIPT" ]]; then
    if command -v node &> /dev/null; then
        node "$ANALYZER_SCRIPT" "$RESULT_FILE" "$SUMMARY_FILE"
    else
        log_warning "Node.js not available, skipping analysis"
    fi
else
    log_warning "Analyzer script not found at $ANALYZER_SCRIPT"
fi

# Optional: run a WebSocket-only baseline (preset=baseline, peak ~60 connections)
WS_SCENARIO_FILE="$LOAD_DIR/scenarios/websocket-stress.js"
if [[ -f "$WS_SCENARIO_FILE" ]]; then
    echo ""
    log_info "Starting WebSocket baseline companion run (preset=baseline, peak ~60 connections)..."
    if [[ "${SKIP_WS_COMPANION:-false}" == "true" || "${SKIP_WS_COMPANION:-0}" == "1" ]]; then
        log_info "SKIP_WS_COMPANION set; skipping WebSocket companion run"
    else
        WS_RESULT_FILE="$RESULTS_DIR/websocket_${SCENARIO_ID}_${TARGET}_${TIMESTAMP}.json"
        WS_SCENARIO_ID="${SCENARIO_ID}-websocket"
    WS_K6_ARGS=(
        "--env" "BASE_URL=$BASE_URL"
        "--env" "WS_URL=$WS_URL"
        "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
        "--env" "WS_SCENARIO_PRESET=baseline"
        "--env" "SCENARIO_ID=$WS_SCENARIO_ID"
        "--tag" "test=websocket-baseline"
        "--tag" "scenario_id=$WS_SCENARIO_ID"
        "--tag" "target=$TARGET"
        "--tag" "timestamp=$TIMESTAMP"
        "--out" "json=$WS_RESULT_FILE"
    )
    k6 run "${WS_K6_ARGS[@]}" "$WS_SCENARIO_FILE" || log_warning "WebSocket baseline run exited non-zero (thresholds may have failed)"
    fi
else
    log_warning "WebSocket scenario not found at $WS_SCENARIO_FILE; skipping WebSocket baseline run"
fi

# Print summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                    Test Complete                       ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Scenario: $SCENARIO_ID"
echo "║  Results:   $RESULT_FILE"
echo "║  Summary:   $SUMMARY_FILE"
if [[ -n "${WS_RESULT_FILE:-}" ]]; then
    echo "║  WS Results: $WS_RESULT_FILE"
fi
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Remind about baseline documentation
if [[ -f "$PROJECT_ROOT/docs/testing/BASELINE_CAPACITY.md" ]]; then
    log_info "Update baseline documentation: docs/testing/BASELINE_CAPACITY.md"
fi

exit $K6_EXIT_CODE
