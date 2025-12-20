#!/bin/bash
set -euo pipefail

#
# RingRift Stress Test - Find Breaking Point Beyond Target Scale
#
# Ramps beyond the target (100 games / 300 players) to find system limits.
# This test pushes the system until it fails to identify capacity ceilings.
#
# Usage:
#   ./tests/load/scripts/run-stress-test.sh [--local|--staging]
#   ./tests/load/scripts/run-stress-test.sh staging
#
# Environment Variables:
#   BASE_URL       - Override the base URL for the target environment
#   STAGING_URL    - URL for staging environment (default: http://localhost:3000)
#   MAX_VUS        - Maximum VUs to stress test to (default: 500)
#   SKIP_CONFIRM   - Set to 'true' to skip the confirmation prompt
#   SKIP_PREFLIGHT_CHECKS - Set to 'true' to skip extended preflight validation
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"

# Default to staging
TARGET="${1:-staging}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$LOAD_DIR/results"
RESULT_FILE="$RESULTS_DIR/stress_${TARGET}_${TIMESTAMP}.json"
SUMMARY_FILE="$RESULTS_DIR/stress_${TARGET}_${TIMESTAMP}_summary.json"

# Maximum VUs for stress test - configurable
MAX_VUS="${MAX_VUS:-500}"
EXPECTED_VUS="$MAX_VUS"
EXPECTED_DURATION_S=1800

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
    echo "Run stress tests beyond target scale to find system limits."
    echo "Ramps from 0 to $MAX_VUS VUs to identify breaking point."
    echo ""
    echo "Options:"
    echo "  --local     Run against local development server"
    echo "  --staging   Run against staging environment (default)"
    echo "  --help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BASE_URL       Override the target URL"
    echo "  STAGING_URL    URL for staging environment"
    echo "  MAX_VUS        Maximum VUs to stress test to (default: 500)"
    echo "  SKIP_CONFIRM   Set to 'true' to skip confirmation prompt"
    echo "  SKIP_PREFLIGHT_CHECKS  Skip extended preflight validation"
    echo ""
    echo "Duration: Approximately 30 minutes"
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
echo "║     RingRift STRESS TEST - Find Breaking Point         ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Target:       $TARGET"
echo "║  Base URL:     $BASE_URL"
echo "║  Max VUs:      $MAX_VUS"
echo "║  Results:      $(basename "$RESULT_FILE")"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check for k6 installation
if ! command -v k6 &> /dev/null; then
    log_error "k6 is not installed. Please install it first:"
    echo "  macOS:   brew install k6"
    echo "  Linux:   sudo apt-get install k6"
    exit 1
fi

log_info "k6 version: $(k6 version)"

# Pre-flight health check
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

HEALTH_URL="$BASE_URL/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Server is healthy at $BASE_URL"
else
    log_error "Health check failed at $HEALTH_URL"
    log_error "Is the server running?"
    exit 1
fi

# Warning and confirmation
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ⚠️  WARNING: EXTREME STRESS TEST                      ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  This test WILL push the system to failure!            ║"
echo "║                                                        ║"
echo "║  The test ramps beyond target to find limits:          ║"
echo "║    • Phase 1: 0→100 VUs (5m) - Baseline                ║"
echo "║    • Phase 2: 100→200 VUs (5m) - Near target           ║"
echo "║    • Phase 3: 200→300 VUs (5m) - AT TARGET             ║"
echo "║    • Phase 4: 300→400 VUs (5m) - Beyond target         ║"
echo "║    • Phase 5: 400→${MAX_VUS} VUs (5m) - Stress zone      ║"
echo "║    • Phase 6: ${MAX_VUS}→0 VUs (5m) - Recovery            ║"
echo "║                                                        ║"
echo "║  Expect errors as the system reaches its limits.       ║"
echo "║  A 10% error rate is acceptable in stress testing.     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [[ "${SKIP_CONFIRM:-false}" != "true" ]]; then
    read -p "Continue with stress test? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Test aborted by user."
        exit 0
    fi
fi

# Determine scenario file
SCENARIO_FILE="$LOAD_DIR/scenarios/concurrent-games.js"

if [[ ! -f "$SCENARIO_FILE" ]]; then
    log_error "Scenario file not found: $SCENARIO_FILE"
    exit 1
fi

echo ""
log_info "🚀 Starting stress test..."
log_info "This will take approximately 30 minutes."
log_info "Monitor system resources closely for breaking point detection."
echo ""

# Record start time
START_TIME=$(date +%s)

# Build k6 arguments for stress test
K6_ARGS=(
    "--env" "BASE_URL=$BASE_URL"
    "--env" "WS_URL=$WS_URL"
    "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
    "--env" "LOAD_PROFILE=stress"
    "--tag" "test=stress"
    "--tag" "target=$TARGET"
    "--tag" "timestamp=$TIMESTAMP"
    "--out" "json=$RESULT_FILE"
)

# Stress test stages - ramp to find breaking point
# Total: 5m + 5m + 5m + 5m + 5m + 5m = 30m
K6_ARGS+=(
    "--stage" "5m:100"     # Phase 1: Ramp to baseline (100 VUs)
    "--stage" "5m:200"     # Phase 2: Ramp to 200 VUs
    "--stage" "5m:300"     # Phase 3: Ramp to target (300 VUs)
    "--stage" "5m:400"     # Phase 4: Beyond target (400 VUs)
    "--stage" "5m:${MAX_VUS}"   # Phase 5: Stress zone (500 VUs)
    "--stage" "5m:0"       # Phase 6: Recovery ramp down
)

# Add more permissive thresholds for stress testing
# We expect errors, so we use higher tolerances
K6_ARGS+=(
    "--env" "STRESS_TEST_MODE=true"
)

# Run the test
k6 run "${K6_ARGS[@]}" "$SCENARIO_FILE"
K6_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MINS=$((DURATION / 60))

echo ""

if [[ $K6_EXIT_CODE -eq 0 ]]; then
    log_success "Stress test completed"
else
    log_warning "Stress test completed with threshold violations (exit code: $K6_EXIT_CODE)"
    log_info "This is expected in stress testing - we're looking for the breaking point."
fi

log_info "Test duration: ${DURATION_MINS} minutes"

# Analyze results for breaking point
echo ""
log_info "📊 Analyzing stress test results..."

# Use generic analyzer for stress test
ANALYZER_SCRIPT="$SCRIPT_DIR/analyze-results.js"
if [[ -f "$ANALYZER_SCRIPT" ]] && command -v node &> /dev/null; then
    node "$ANALYZER_SCRIPT" "$RESULT_FILE" "$SUMMARY_FILE" || true
else
    log_warning "Analyzer not available, skipping detailed analysis"
fi

# Generate stress test specific summary
if command -v node &> /dev/null; then
    node -e "
const fs = require('fs');
const path = process.argv[1];
const content = fs.readFileSync(path, 'utf8');
const lines = content.split('\n').filter(l => l.trim());
const metrics = {
  max_vus: 0,
  total_requests: 0,
  failed_requests: 0,
  latencies: [],
  error_rates_by_vu: []
};

lines.forEach(line => {
  try {
    const entry = JSON.parse(line);
    if (entry.type !== 'Point') return;
    const { metric, data } = entry;
    if (metric === 'vus' && data.value > metrics.max_vus) {
      metrics.max_vus = data.value;
    }
    if (metric === 'http_req_duration') {
      metrics.latencies.push(data.value);
      metrics.total_requests++;
    }
    if (metric === 'http_req_failed' && data.value === 1) {
      metrics.failed_requests++;
    }
  } catch {}
});

const sorted = metrics.latencies.sort((a, b) => a - b);
const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
const errorRate = metrics.total_requests > 0 
  ? (metrics.failed_requests / metrics.total_requests * 100).toFixed(2) 
  : 0;

console.log('');
console.log('╔════════════════════════════════════════════════════════════════════╗');
console.log('║           STRESS TEST BREAKING POINT ANALYSIS                      ║');
console.log('╠════════════════════════════════════════════════════════════════════╣');
console.log('║  Max VUs Sustained:  ' + String(metrics.max_vus).padEnd(10) + '                              ║');
console.log('║  Total Requests:     ' + String(metrics.total_requests).padEnd(10) + '                              ║');
console.log('║  Failed Requests:    ' + String(metrics.failed_requests).padEnd(10) + '                              ║');
console.log('║  Error Rate:         ' + String(errorRate + '%').padEnd(10) + '                              ║');
console.log('║  p95 Latency:        ' + String(p95.toFixed(0) + 'ms').padEnd(10) + '                              ║');
console.log('╠════════════════════════════════════════════════════════════════════╣');

// Determine breaking point heuristic
let breakingPoint = 'Unknown';
if (errorRate > 10) {
  breakingPoint = 'System under severe stress';
} else if (errorRate > 5) {
  breakingPoint = 'Approaching limits';
} else if (p95 > 2000) {
  breakingPoint = 'Latency degradation detected';
} else {
  breakingPoint = 'System handled stress well';
}

console.log('║  Assessment: ' + breakingPoint.padEnd(45) + '║');
console.log('╚════════════════════════════════════════════════════════════════════╝');
console.log('');
" "$RESULT_FILE"
fi

# Print final summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Stress Test Complete                      ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Duration:    ${DURATION_MINS} minutes"
echo "║  Results:     $RESULT_FILE"
echo "║  Summary:     $SUMMARY_FILE"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Review results to identify:                           ║"
echo "║    • At what VU count did errors increase?             ║"
echo "║    • At what VU count did latency spike?               ║"
echo "║    • What was the maximum sustainable load?            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

log_info "Use these results to set appropriate capacity limits and alerts."

# Exit with k6 exit code (stress test failures are expected)
exit 0
