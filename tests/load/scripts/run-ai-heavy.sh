#!/bin/bash
set -euo pipefail

#
# RingRift AI-Heavy Capacity Probe Runner
#
# Scenario: BCAP_SQ8_4P_AI_HEAVY_75G_300P
#
# AI-heavy capacity probe with ~75 concurrent 4-player square8 games and
# ~300 players (3 AI seats per game). This is a staging-focused scenario
# that uses staging thresholds for non-AI metrics and production SLOs for
# AI latency/fallbacks at the SLO verification layer.
#
# Usage:
#   ./tests/load/scripts/run-ai-heavy.sh [--local|--staging]
#   ./tests/load/scripts/run-ai-heavy.sh staging
#
# Environment Variables:
#   BASE_URL       - Override the base URL for the target environment
#   STAGING_URL    - URL for staging environment (default: http://localhost:3000)
#   K6_EXTRA_ARGS  - Additional arguments to pass to k6
#   SKIP_CONFIRM   - Set to 'true' to skip the confirmation prompt
#   SEED_LOADTEST_USERS - If 'true', seed load-test users before running (uses scripts/seed-loadtest-users.js)
#   LOADTEST_USER_COUNT / LOADTEST_USER_DOMAIN / LOADTEST_USER_OFFSET / LOADTEST_USER_PASSWORD / LOADTEST_USER_ROLE - Seeding overrides
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$LOAD_DIR")")"

SCENARIO_ID="BCAP_SQ8_4P_AI_HEAVY_75G_300P"

# Default to staging for this capacity probe
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
    # For local runs we still use staging thresholds for k6, SLO script
    # will escalate AI-related SLOs to production via bcap-scenarios.json.
    THRESHOLD_ENV="staging"
    ;;
  --staging|staging)
    BASE_URL="${STAGING_URL:-${BASE_URL:-http://localhost:3000}}"
    WS_URL="${WS_URL:-$(echo "$BASE_URL" | sed 's/^http/ws/')}"
    # Staging thresholds for in-test k6 thresholds; SLO verification will
    # apply production targets for AI metrics only.
    THRESHOLD_ENV="staging"
    ;;
  --production|production)
    log_error "AI-heavy BCAP probe should not be run directly against production!"
    log_error "Use --staging (and remote staging endpoints) for validation."
    exit 1
    ;;
  --help|-h)
    echo "Usage: $0 [--local|--staging]"
    echo ""
    echo "Run AI-heavy capacity probe (~75 games / 300 players, square8 4p with 3 AI seats)."
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
    echo "Duration: Approximately 13 minutes"
    echo "Resource Requirements: ~8GB RAM, 4+ CPU cores (for staging cluster)"
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
echo "║     RingRift AI-HEAVY Capacity Probe (BCAP v1)         ║"
echo "║     Scenario: $SCENARIO_ID"
echo "╠════════════════════════════════════════════════════════╣"
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

HEALTH_URL="$BASE_URL/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Server is healthy at $BASE_URL"
else
    log_error "Health check failed at $HEALTH_URL"
    log_error "Is the server running? Start with: npm run dev or deploy staging"
    exit 1
fi

AI_HEALTH_URL="${AI_SERVICE_URL:-http://localhost:8000}/health"
if curl -sf "$AI_HEALTH_URL" > /dev/null 2>&1; then
    log_success "AI service is healthy"
else
    log_warning "AI service not responding at $AI_HEALTH_URL (optional)"
fi

# Warning and confirmation
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ⚠️  WARNING: AI-HEAVY CAPACITY PROBE                  ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  This test applies AI-heavy load to the system:        ║"
echo "║    • ~75 concurrent games (4-player square8)           ║"
echo "║    • ~300 players (3 AI seats per game)                ║"
echo "║    • ~13 minute duration                               ║"
echo "║                                                        ║"
echo "║  Phases:                                               ║"
echo "║    1. Warmup:        2m  (25 VUs)                      ║"
echo "║    2. Ramp to 75G:   3m  (75 VUs)                      ║"
echo "║    3. Steady AI load:5m  (75 VUs)  <- validation       ║"
echo "║    4. Ramp down:     3m  (0 VUs)                       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [[ "${SKIP_CONFIRM:-false}" != "true" ]]; then
    read -p "Continue with AI-heavy capacity probe? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Test aborted by user."
        exit 0
    fi
fi

# Run the AI-heavy test
echo ""
log_info "🚀 Starting AI-heavy capacity probe..."
log_info "Scenario ID: $SCENARIO_ID"
log_info "This will take approximately 13 minutes."
echo ""

# Optionally seed load-test users to ensure sufficient accounts exist.
if [[ "${SEED_LOADTEST_USERS:-false}" == "true" ]]; then
    echo ""
    log_info "Seeding load-test users (LOADTEST_USER_COUNT=${LOADTEST_USER_COUNT:-400}, domain=${LOADTEST_USER_DOMAIN:-loadtest.local}, offset=${LOADTEST_USER_OFFSET:-0})..."
    (cd "$PROJECT_ROOT" && npm run load:seed-users) || log_warning "User seeding failed; continuing without seeding"
fi

SCENARIO_FILE="$LOAD_DIR/scenarios/concurrent-games.js"

if [[ ! -f "$SCENARIO_FILE" ]]; then
    log_error "Scenario file not found: $SCENARIO_FILE"
    exit 1
fi

# Build k6 arguments: we reuse concurrent-games.js but override stages explicitly
K6_ARGS=(
    "--env" "BASE_URL=$BASE_URL"
    "--env" "WS_URL=$WS_URL"
    "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
    "--env" "LOAD_PROFILE=target_scale"
    "--tag" "test=ai-heavy"
    "--tag" "scenario_id=$SCENARIO_ID"
    "--tag" "target=$TARGET"
    "--tag" "timestamp=$TIMESTAMP"
    "--out" "json=$RESULT_FILE"
)

# Summary output directory for handleSummary
K6_ARGS+=("--env" "K6_SUMMARY_DIR=$RESULTS_DIR")

# Stage overrides for BCAP_SQ8_4P_AI_HEAVY_75G_300P:
# In smoke mode we swap in a very short, low-intensity profile so local runs
# stay lightweight while preserving scenario IDs and SLO semantics.
if [[ "${SMOKE:-0}" == "1" || "${SMOKE:-}" == "true" ]]; then
    log_info "SMOKE mode enabled - overriding stages for AI-heavy scenario to a short, low-intensity profile (~25s, max 2 VUs)"
    K6_ARGS+=(
        "--stage" "5s:1"
        "--stage" "15s:2"
        "--stage" "5s:0"
    )
else
    #   - 2m @ 25 VUs (warmup)
    #   - 3m @ 75 VUs (ramp)
    #   - 5m @ 75 VUs (AI-heavy steady window)
    #   - 3m @ 0 VUs (ramp down)
    K6_ARGS+=(
        "--stage" "2m:25"
        "--stage" "3m:75"
        "--stage" "5m:75"
        "--stage" "3m:0"
    )
fi

# Extra arguments (if provided)
if [[ -n "${K6_EXTRA_ARGS:-}" ]]; then
    IFS=' ' read -ra EXTRA_ARGS <<< "$K6_EXTRA_ARGS"
    K6_ARGS+=("${EXTRA_ARGS[@]}")
fi

START_TIME=$(date +%s)

k6 run "${K6_ARGS[@]}" "$SCENARIO_FILE"
K6_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MINS=$((DURATION / 60))

echo ""

if [[ $K6_EXIT_CODE -eq 0 ]]; then
    log_success "AI-heavy capacity probe completed successfully"
else
    log_warning "AI-heavy probe completed with threshold violations (exit code: $K6_EXIT_CODE)"
fi

log_info "Test duration: ${DURATION_MINS} minutes"

# Analyze results
echo ""
log_info "📊 Analyzing AI-heavy results..."

ANALYZER_SCRIPT="$SCRIPT_DIR/analyze-results.js"
if [[ -f "$ANALYZER_SCRIPT" ]] && command -v node &> /dev/null; then
    node "$ANALYZER_SCRIPT" "$RESULT_FILE" "$SUMMARY_FILE" || log_warning "Analyzer reported non-zero exit; review summary for details"
else
    log_warning "Analyzer script not available; skipping detailed analysis"
fi

# Optional: run WebSocket companion pass (preset=target, peak ~300 connections)
WS_SCENARIO_FILE="$LOAD_DIR/scenarios/websocket-stress.js"
if [[ -f "$WS_SCENARIO_FILE" ]]; then
    echo ""
    log_info "Starting WebSocket companion run (preset=target, peak ~300 connections)..."
    WS_RESULT_FILE="$RESULTS_DIR/websocket_${SCENARIO_ID}_${TARGET}_${TIMESTAMP}.json"
    WS_K6_ARGS=(
        "--env" "BASE_URL=$BASE_URL"
        "--env" "WS_URL=$WS_URL"
        "--env" "THRESHOLD_ENV=$THRESHOLD_ENV"
        "--env" "WS_SCENARIO_PRESET=target"
        "--tag" "test=websocket-${SCENARIO_ID}"
        "--tag" "target=$TARGET"
        "--tag" "timestamp=$TIMESTAMP"
        "--out" "json=$WS_RESULT_FILE"
    )
    k6 run "${WS_K6_ARGS[@]}" "$WS_SCENARIO_FILE" || log_warning "WebSocket companion run exited non-zero (thresholds may have failed)"
else
    log_warning "WebSocket scenario not found at $WS_SCENARIO_FILE; skipping WebSocket companion run"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              AI-Heavy Probe Complete                   ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Scenario:   $SCENARIO_ID"
echo "║  Duration:   ${DURATION_MINS} minutes"
echo "║  Results:    $RESULT_FILE"
echo "║  Summary:    $SUMMARY_FILE"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [[ -f "$PROJECT_ROOT/docs/BASELINE_CAPACITY.md" ]]; then
    log_info "If this probe informs capacity limits, update: docs/BASELINE_CAPACITY.md"
fi

exit $K6_EXIT_CODE
