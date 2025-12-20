/**
 * RingRift Load Test: Game Creation Scenario
 * 
 * Tests the game creation rate and latency under increasing load.
 * Validates production scale assumptions for game lobby operations.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3.1: Mixed Human vs AI Ladder
 * SLOs from STRATEGIC_ROADMAP.md §2.1: HTTP API targets
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { getValidToken, getBypassHeaders } from '../auth/helpers.js';
import { makeHandleSummary } from '../summary.js';

const thresholdsConfig = JSON.parse(open('../config/thresholds.json'));

// Classification metrics shared across scenarios
export const contractFailures = new Counter('contract_failures_total');
export const idLifecycleMismatches = new Counter('id_lifecycle_mismatches_total');
export const capacityFailures = new Counter('capacity_failures_total');
const authTokenExpired = new Counter('auth_token_expired_total');
const rateLimitHit = new Counter('rate_limit_hit_total');
const trueErrors = new Counter('true_errors_total');

// Custom metrics
const gameCreationErrors = new Counter('game_creation_errors');
const gameCreationSuccess = new Rate('game_creation_success_rate');
const gameCreationLatency = new Trend('game_creation_latency_ms');

// Threshold configuration derived from thresholds.json
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const perfEnv =
  thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const loadTestEnv =
  thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;
const authLoginHttp = perfEnv.http_api.auth_login;
const gameCreationHttp = perfEnv.http_api.game_creation;
const gameStateFetchHttp = perfEnv.http_api.game_state_fetch;
const trueErrorRateTarget =
  loadTestEnv &&
  loadTestEnv.true_errors &&
  typeof loadTestEnv.true_errors.rate === 'number'
    ? loadTestEnv.true_errors.rate
    : 0.005;

// Test configuration aligned with thresholds.json SLOs
export const options = {
  stages: [
    { duration: '30s', target: 10 }, // Warm up: ramp to 10 users
    { duration: '1m', target: 50 }, // Load: ramp to 50 users
    { duration: '2m', target: 50 }, // Sustain: hold at 50 users
    { duration: '30s', target: 0 }, // Ramp down
  ],

  thresholds: {
    // HTTP request duration - env-specific SLOs from thresholds.json, per endpoint.
    'http_req_duration{name:auth-login-setup}': [
      `p(95)<${authLoginHttp.latency_p95_ms}`,
      `p(99)<${authLoginHttp.latency_p99_ms}`,
    ],
    'http_req_duration{name:create-game}': [
      `p(95)<${gameCreationHttp.latency_p95_ms}`,
      `p(99)<${gameCreationHttp.latency_p99_ms}`,
    ],
    'http_req_duration{name:get-game}': [
      `p(95)<${gameStateFetchHttp.latency_p95_ms}`,
      `p(99)<${gameStateFetchHttp.latency_p99_ms}`,
    ],

    // Error rate - use the strictest 5xx error budget across these key endpoints.
    http_req_failed: [
      `rate<${
        Math.max(
          authLoginHttp.error_rate_5xx_percent,
          gameCreationHttp.error_rate_5xx_percent,
          gameStateFetchHttp.error_rate_5xx_percent
        ) / 100
      }`,
    ],

    // Custom metrics
    game_creation_success_rate: ['rate>0.99'],
    game_creation_latency_ms: [
      `p(95)<${gameCreationHttp.latency_p95_ms}`,
      `p(99)<${gameCreationHttp.latency_p99_ms}`,
    ],

    // Contract/id-lifecycle/capacity classification
    contract_failures_total: [`count<=${loadTestEnv.contract_failures_total.max}`],
    id_lifecycle_mismatches_total: [
      `count<=${loadTestEnv.id_lifecycle_mismatches_total.max}`,
    ],
    capacity_failures_total: [`rate<${loadTestEnv.capacity_failures_total.rate}`],
    true_errors_total: [`rate<${trueErrorRateTarget}`],
  },

  // Test metadata
  tags: {
    scenario: 'game-creation',
    test_type: 'load',
    environment: THRESHOLD_ENV,
  },
};

// Test configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

/**
 * Safely parse create-game API response and unwrap { success, data: { game } }.
 */
function parseCreateGameResponse(res) {
  try {
    const body = JSON.parse(res.body);
    const game = body && body.data && body.data.game ? body.data.game : null;
    return { body, game };
  } catch (error) {
    return { body: null, game: null };
  }
}

/**
 * Classify failures for POST /api/games when the request payload is expected
 * to be valid according to the CreateGameRequest contract.
 */
function classifyCreateGameFailure(res, parsed) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (res.status === 401) {
    authTokenExpired.add(1);
    return;
  }

  if (res.status === 429) {
    rateLimitHit.add(1);
    capacityFailures.add(1);
    return;
  }

  if (res.status === 400 || res.status === 403) {
    contractFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (res.status >= 500) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  // 2xx but missing/malformed body or game object.
  if (res.status >= 200 && res.status < 300 && (!parsed.body || !parsed.game || !parsed.game.id)) {
    contractFailures.add(1);
    trueErrors.add(1);
  }
}

/**
 * Classify failures for GET /api/games/:gameId immediately after creation.
 * A 404 here almost always indicates an ID lifecycle mismatch, since the
 * harness just created the game and has not yet hit any poll budget.
 */
function classifyImmediateGetGameFailure(res, gameId) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (res.status === 401) {
    authTokenExpired.add(1);
    return;
  }

  if (res.status === 429) {
    rateLimitHit.add(1);
    capacityFailures.add(1);
    return;
  }

  if (res.status === 400 || res.status === 403) {
    contractFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (res.status === 404) {
    idLifecycleMismatches.add(1);
    trueErrors.add(1);
    return;
  }

  if (res.status >= 500) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  // 2xx but missing or mismatched ID.
  if (res.status >= 200 && res.status < 300) {
    contractFailures.add(1);
    trueErrors.add(1);
  }
}

/**
 * Setup function - runs once before the test and returns shared config.
 *
 * Multi-user mode:
 *   When LOADTEST_USER_POOL_SIZE is set, each VU will authenticate as a
 *   different user from the pool. This distributes load across users to
 *   avoid per-user rate limits.
 */
export function setup() {
  console.log(`Starting game creation load test against ${BASE_URL}`);
  console.log('Target load: 50 concurrent users creating games');

  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  return { baseUrl: BASE_URL };
}

/**
 * Main test function - runs repeatedly for each VU
 */
export default function(data) {
  const baseUrl = data.baseUrl;

  // Each VU logs in as its own user (when multi-user pool is configured)
  // getValidToken handles caching and token refresh per-VU
  const { token } = getValidToken(baseUrl, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login' },
    metrics: {
      contractFailures,
      capacityFailures,
    },
  });

  const authHeaders = getBypassHeaders({
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  });

  sleep(0.5);

  // Step 3: Create Game (main scenario focus)
  const boardTypes = ['square8', 'square19', 'hexagonal'];
  const boardType = boardTypes[Math.floor(Math.random() * boardTypes.length)];
  const maxPlayersOptions = [2, 3, 4];
  const maxPlayers = maxPlayersOptions[Math.floor(Math.random() * maxPlayersOptions.length)];

  const gameConfig = {
    name: `Load Test Game ${__VU}-${__ITER}`,
    boardType: boardType,
    maxPlayers: maxPlayers,
    isPrivate: false,
    timeControl: {
      initialTime: 300,
      increment: 5,
      type: 'blitz',
    },
    isRated: true,
  };

  const startTime = Date.now();
  const createGameRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
    headers: authHeaders,
    tags: { name: 'create-game' },
  });
  const createGameDuration = Date.now() - startTime;

  const parsedCreate = parseCreateGameResponse(createGameRes);
  const gameCreated = check(createGameRes, {
    'status is 201': (r) => r.status === 201,
    'response payload parsed': () => parsedCreate.body !== null && parsedCreate.game !== null,
    'game ID returned': () => Boolean(parsedCreate.game && parsedCreate.game.id),
    'game config matches': () =>
      !!(
        parsedCreate.game &&
        parsedCreate.game.boardType === boardType &&
        parsedCreate.game.maxPlayers === maxPlayers
      ),
    'ai games are unrated when present': () => {
      if (!parsedCreate.game || !parsedCreate.game.aiOpponents) return true;
      const count = parsedCreate.game.aiOpponents.count || 0;
      if (count <= 0) return true;
      return parsedCreate.game.isRated === false;
    },
  });

  // Track metrics
  gameCreationLatency.add(createGameDuration);
  gameCreationSuccess.add(gameCreated);

  if (!gameCreated) {
    gameCreationErrors.add(1);
    classifyCreateGameFailure(createGameRes, parsedCreate);
    console.error(
      `Game creation failed for VU ${__VU}: ${createGameRes.status} - ${createGameRes.body}`
    );
    return;
  }

  const gameId = parsedCreate.game.id;

  sleep(0.5);

  // Step 4: Fetch Game State (validates read path)
  const getGameRes = http.get(`${baseUrl}${API_PREFIX}/games/${gameId}`, {
    headers: authHeaders,
    tags: { name: 'get-game' },
  });

  const gameStateOk = check(getGameRes, {
    'game state retrieved': (r) => r.status === 200,
    'game state valid': (r) => {
      try {
        const body = JSON.parse(r.body);
        const game = body && body.data && body.data.game ? body.data.game : null;
        return !!(game && game.id === gameId);
      } catch {
        return false;
      }
    },
  });

  if (!gameStateOk) {
    classifyImmediateGetGameFailure(getGameRes, gameId);

    const bodySnippet =
      typeof getGameRes.body === 'string' && getGameRes.body.length > 200
        ? `${getGameRes.body.substring(0, 200)}...`
        : getGameRes.body;
    console.error(
      `Game state fetch failed for VU ${__VU}: status=${getGameRes.status} body=${bodySnippet}`
    );
  }

  // Think time - simulates user reviewing game before next action
  sleep(1 + Math.random() * 2); // 1-3 seconds
}

/**
 * Teardown function - runs once after all iterations complete
 */
export function teardown(data) {
  console.log('Game creation load test complete');
  console.log('Review metrics in Grafana or k6 summary output');
}

export const handleSummary = makeHandleSummary('game-creation');
