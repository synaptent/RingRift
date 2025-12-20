/**
 * RingRift k6 Load Test Configuration
 *
 * Unified configuration for running comprehensive load tests at production scale.
 * Supports 100+ concurrent games with 300 players as per PROJECT_GOALS.md targets.
 *
 * Usage:
 *   # Run with default configuration (all scenarios)
 *   k6 run --config tests/load/k6.config.js tests/load/scenarios/game-lifecycle.js
 *
 *   # Run specific scenario with this config
 *   k6 run --config tests/load/k6.config.js tests/load/scenarios/concurrent-games.js
 *
 *   # Override environment
 *   k6 run --config tests/load/k6.config.js --env THRESHOLD_ENV=production tests/load/scenarios/game-creation.js
 *
 * Environment Variables:
 *   BASE_URL           - HTTP API base (default: http://localhost:3001)
 *   WS_URL             - WebSocket base (default: derived from BASE_URL)
 *   THRESHOLD_ENV      - Threshold profile: staging | production (default: staging)
 *   LOAD_PROFILE       - Load profile: smoke | load | stress | spike | soak (default: stress)
 *   K6_SUMMARY_DIR     - Summary output directory (default: results/load)
 *
 * @see tests/load/README.md for detailed usage documentation
 * @see tests/load/config/thresholds.json for SLO definitions
 * @see tests/load/config/scenarios.json for load profiles
 */

// Load configuration from JSON files
const thresholdsConfig = JSON.parse(open('./config/thresholds.json'));
const scenariosConfig = JSON.parse(open('./config/scenarios.json'));

// Environment configuration
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const LOAD_PROFILE = __ENV.LOAD_PROFILE || 'stress';
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';

// Get environment-specific thresholds
const perfEnv = thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const loadTestEnv = thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;

// Get profile-specific scenario configuration
const profile = scenariosConfig.profiles[LOAD_PROFILE] || scenariosConfig.profiles.stress;

/**
 * Global k6 options that apply to all scenarios when using this config.
 * Individual scenario files can override these with their own options.
 */
export const options = {
  // Default to stress profile stages for concurrent games scenario
  scenarios: {
    // Production-scale concurrent games simulation
    production_scale_games: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: profile.concurrent_games?.stages || [
        { duration: '2m', target: 50 },   // Warm up
        { duration: '3m', target: 100 },  // Ramp to production target
        { duration: '5m', target: 100 },  // Sustain at target
        { duration: '2m', target: 50 },   // Gradual cooldown
        { duration: '1m', target: 0 },    // Shutdown
      ],
      gracefulRampDown: '30s',
    },
  },

  // Global thresholds aligned with thresholds.json
  thresholds: {
    // HTTP API SLOs
    'http_req_duration': [
      `p(95)<${perfEnv.http_api.game_creation.latency_p95_ms}`,
      `p(99)<${perfEnv.http_api.game_creation.latency_p99_ms}`,
    ],
    'http_req_failed': [
      `rate<${perfEnv.http_api.game_creation.error_rate_5xx_percent / 100}`,
    ],

    // WebSocket SLOs
    'websocket_connection_success_rate': [
      `rate>${perfEnv.websocket_gameplay.connection_stability.connection_success_rate_percent / 100}`,
    ],

    // Classification counters (contract vs capacity failures)
    'contract_failures_total': [
      `count<=${loadTestEnv.contract_failures_total.max}`,
    ],
    'id_lifecycle_mismatches_total': [
      `count<=${loadTestEnv.id_lifecycle_mismatches_total.max}`,
    ],
    'capacity_failures_total': [
      `rate<${loadTestEnv.capacity_failures_total.rate}`,
    ],

    // True error rate thresholds - excludes auth (401) and rate-limit (429) errors
    // This provides the real application error rate for SLO validation.
    // These errors are "noise" from expected infrastructure behavior, not true app bugs.
    'true_errors_total': [
      `rate<${loadTestEnv.true_errors?.rate || 0.005}`, // Less than 0.5% true error rate
    ],

    // WebSocket protocol errors
    'websocket_protocol_errors': [
      `count<=${loadTestEnv.websocket.protocol_errors_max}`,
    ],
    'websocket_connection_errors': [
      `count<=${loadTestEnv.websocket.connection_errors_max}`,
    ],
  },

  // Test metadata
  tags: {
    environment: THRESHOLD_ENV,
    profile: LOAD_PROFILE,
    config: 'k6.config.js',
  },

  // Output configuration
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],

  // Disable built-in summary when using custom handleSummary
  noSummary: false,
};

/**
 * Export configuration for use in scenario files.
 * Scenarios can import these to access centralized settings.
 */
export const config = {
  baseUrl: BASE_URL,
  wsUrl: __ENV.WS_URL || BASE_URL.replace(/^http/, 'ws'),
  thresholdEnv: THRESHOLD_ENV,
  loadProfile: LOAD_PROFILE,
  thresholds: thresholdsConfig,
  scenarios: scenariosConfig,
  perfEnv,
  loadTestEnv,
  profile,

  // Production scale targets from thresholds.json
  scaleTargets: perfEnv.scale_targets,

  // Move submission SLOs
  moveSubmission: perfEnv.websocket_gameplay.move_submission,

  // AI service SLOs
  aiService: perfEnv.ai_service,
};

/**
 * Pre-configured scenario stage configurations for common use cases.
 * Import these in scenario files for consistent load patterns.
 */
export const stageConfigs = {
  // Smoke test - quick validation (1-2 minutes)
  smoke: {
    stages: [
      { duration: '10s', target: 2 },
      { duration: '30s', target: 5 },
      { duration: '10s', target: 5 },
      { duration: '10s', target: 0 },
    ],
    gracefulRampDown: '10s',
  },

  // Load test - realistic production load (5 minutes)
  load: {
    stages: [
      { duration: '30s', target: 10 },
      { duration: '1m', target: 50 },
      { duration: '2m', target: 50 },
      { duration: '30s', target: 0 },
    ],
    gracefulRampDown: '30s',
  },

  // Stress test - production scale validation (13 minutes)
  stress: {
    stages: [
      { duration: '2m', target: 50 },
      { duration: '3m', target: 100 },
      { duration: '5m', target: 100 },
      { duration: '2m', target: 50 },
      { duration: '1m', target: 0 },
    ],
    gracefulRampDown: '30s',
  },

  // Spike test - sudden surge (7 minutes)
  spike: {
    stages: [
      { duration: '30s', target: 10 },
      { duration: '1m', target: 100 },
      { duration: '3m', target: 100 },
      { duration: '1m', target: 10 },
      { duration: '30s', target: 0 },
    ],
    gracefulRampDown: '30s',
  },

  // Soak test - extended duration (1 hour)
  soak: {
    stages: [
      { duration: '5m', target: 40 },
      { duration: '50m', target: 40 },
      { duration: '5m', target: 0 },
    ],
    gracefulRampDown: '1m',
  },

  // WebSocket stress - high connection count (15 minutes)
  websocketStress: {
    stages: [
      { duration: '2m', target: 100 },
      { duration: '2m', target: 300 },
      { duration: '3m', target: 500 },
      { duration: '5m', target: 500 },
      { duration: '2m', target: 100 },
      { duration: '1m', target: 0 },
    ],
    gracefulRampDown: '30s',
  },
};

/**
 * Helper to select stage configuration based on environment.
 * @param {string} profileName - Profile name (smoke, load, stress, etc.)
 * @returns {Object} Stage configuration
 */
export function getStagesForProfile(profileName) {
  return stageConfigs[profileName] || stageConfigs.stress;
}

/**
 * Export default configuration summary for logging.
 */
export function logConfig() {
  console.log('=== RingRift k6 Load Test Configuration ===');
  console.log(`Environment:    ${THRESHOLD_ENV}`);
  console.log(`Load Profile:   ${LOAD_PROFILE}`);
  console.log(`Base URL:       ${BASE_URL}`);
  console.log(`WebSocket URL:  ${config.wsUrl}`);
  console.log('');
  console.log('Scale Targets:');
  console.log(`  Max Concurrent Games:  ${config.scaleTargets.max_concurrent_games}`);
  console.log(`  Max Active Players:    ${config.scaleTargets.max_active_players}`);
  console.log(`  Max AI Controlled:     ${config.scaleTargets.max_ai_controlled_seats}`);
  console.log('');
  console.log('Move Submission SLOs:');
  console.log(`  E2E Latency p95:       ${config.moveSubmission.end_to_end_latency_p95_ms}ms`);
  console.log(`  E2E Latency p99:       ${config.moveSubmission.end_to_end_latency_p99_ms}ms`);
  console.log(`  Stall Threshold:       ${config.moveSubmission.stall_threshold_ms}ms`);
  console.log(`  Stall Rate Budget:     ${config.moveSubmission.stall_rate_percent}%`);
  console.log('=============================================');
}