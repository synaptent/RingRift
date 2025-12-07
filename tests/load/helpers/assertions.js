/**
 * RingRift Load Test - Custom Assertions
 *
 * Provides reusable assertion helpers and check functions for k6 load tests.
 * These assertions are aligned with the SLOs defined in thresholds.json.
 *
 * Usage:
 *   import { assertGameCreated, assertMoveSucceeded, assertLatencyWithinSLO } from '../helpers/assertions.js';
 */

import { check } from 'k6';

// ─────────────────────────────────────────────────────────────────────────────
// HTTP Response Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assert that an HTTP response indicates success (2xx status).
 *
 * @param {Object} res - k6 HTTP response
 * @param {string} [description='HTTP request successful'] - Check description
 * @returns {boolean} True if all checks passed
 */
export function assertHttpSuccess(res, description) {
  return check(res, {
    [description || 'HTTP request successful']: (r) => r.status >= 200 && r.status < 300,
  });
}

/**
 * Assert that an HTTP response has a specific status code.
 *
 * @param {Object} res - k6 HTTP response
 * @param {number} expectedStatus - Expected status code
 * @param {string} [description] - Check description
 * @returns {boolean} True if check passed
 */
export function assertStatus(res, expectedStatus, description) {
  return check(res, {
    [description || `status is ${expectedStatus}`]: (r) => r.status === expectedStatus,
  });
}

/**
 * Assert that an HTTP response body contains valid JSON.
 *
 * @param {Object} res - k6 HTTP response
 * @param {string} [description='response body is valid JSON'] - Check description
 * @returns {{ passed: boolean, parsed: any }}
 */
export function assertValidJson(res, description) {
  let parsed = null;
  const passed = check(res, {
    [description || 'response body is valid JSON']: (r) => {
      try {
        parsed = JSON.parse(r.body);
        return parsed !== null;
      } catch {
        return false;
      }
    },
  });
  return { passed, parsed };
}

// ─────────────────────────────────────────────────────────────────────────────
// Game Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assert that a game was created successfully.
 *
 * @param {Object} res - k6 HTTP response from POST /api/games
 * @param {Object} [expectedConfig] - Expected game configuration
 * @returns {{ passed: boolean, game: Object|null }}
 */
export function assertGameCreated(res, expectedConfig = {}) {
  let game = null;

  try {
    const body = JSON.parse(res.body);
    game = body && body.data && body.data.game ? body.data.game : null;
  } catch {
    game = null;
  }

  const checks = {
    'game creation status is 201': (r) => r.status === 201,
    'game object returned': () => game !== null,
    'game ID is present': () => !!(game && game.id),
  };

  // Add config validation if expected values provided
  if (expectedConfig.boardType) {
    checks['board type matches'] = () => game && game.boardType === expectedConfig.boardType;
  }

  if (expectedConfig.maxPlayers) {
    checks['max players matches'] = () => game && game.maxPlayers === expectedConfig.maxPlayers;
  }

  if (expectedConfig.hasAI) {
    checks['AI games are unrated'] = () => {
      if (!game || !game.aiOpponents) return true;
      if (!game.aiOpponents.count || game.aiOpponents.count <= 0) return true;
      return game.isRated === false;
    };
  }

  const passed = check(res, checks);

  return { passed, game };
}

/**
 * Assert that a game state was retrieved successfully.
 *
 * @param {Object} res - k6 HTTP response from GET /api/games/:id
 * @param {string} [expectedGameId] - Expected game ID
 * @returns {{ passed: boolean, game: Object|null }}
 */
export function assertGameStateRetrieved(res, expectedGameId) {
  let game = null;

  try {
    const body = JSON.parse(res.body);
    game = body && body.data && body.data.game ? body.data.game : null;
  } catch {
    game = null;
  }

  const checks = {
    'game state retrieved': (r) => r.status === 200,
    'game object present': () => game !== null,
  };

  if (expectedGameId) {
    checks['game ID matches'] = () => game && game.id === expectedGameId;
  }

  const passed = check(res, checks);

  return { passed, game };
}

/**
 * Assert that a game has active players.
 *
 * @param {Object} game - Game object
 * @returns {boolean} True if check passed
 */
export function assertGameHasPlayers(game) {
  return check(null, {
    'game has players': () => {
      const playerIds = [
        game && game.player1Id,
        game && game.player2Id,
        game && game.player3Id,
        game && game.player4Id,
      ].filter(Boolean);
      return playerIds.length > 0;
    },
  });
}

/**
 * Assert that a game is in an expected status.
 *
 * @param {Object} game - Game object
 * @param {string|string[]} expectedStatus - Expected status or array of acceptable statuses
 * @returns {boolean} True if check passed
 */
export function assertGameStatus(game, expectedStatus) {
  const acceptableStatuses = Array.isArray(expectedStatus) ? expectedStatus : [expectedStatus];

  return check(null, {
    [`game status is one of: ${acceptableStatuses.join(', ')}`]: () =>
      game && acceptableStatuses.includes(game.status),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Move Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assert that a move was accepted successfully.
 *
 * @param {Object} res - k6 HTTP response from move submission
 * @returns {boolean} True if check passed
 */
export function assertMoveAccepted(res) {
  return check(res, {
    'move accepted': (r) => r.status >= 200 && r.status < 300,
  });
}

/**
 * Assert that move latency is within SLO thresholds.
 *
 * @param {number} latencyMs - Move latency in milliseconds
 * @param {Object} [thresholds] - Custom thresholds
 * @param {number} [thresholds.p95=300] - p95 threshold
 * @param {number} [thresholds.stall=2000] - Stall threshold
 * @returns {{ passed: boolean, stalled: boolean }}
 */
export function assertMoveLatency(latencyMs, thresholds = {}) {
  const p95Threshold = thresholds.p95 || 300;
  const stallThreshold = thresholds.stall || 2000;

  const stalled = latencyMs > stallThreshold;

  const passed = check(null, {
    [`move latency (${latencyMs}ms) under p95 target (${p95Threshold}ms)`]: () =>
      latencyMs <= p95Threshold,
  });

  return { passed, stalled };
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assert that a WebSocket connection was established.
 *
 * @param {Object} res - k6 ws.connect result
 * @returns {boolean} True if connection succeeded
 */
export function assertWebSocketConnected(res) {
  return check(res, {
    'WebSocket connected': (r) => r && r.status === 101,
  });
}

/**
 * Assert that a Socket.IO handshake completed.
 *
 * @param {Object} parsed - Parsed Socket.IO message
 * @param {string} sioConnectType - Expected SIO_CONNECT constant ('0')
 * @returns {boolean} True if handshake message
 */
export function assertSocketIOHandshake(parsed, sioConnectType = '0') {
  return check(null, {
    'Socket.IO handshake complete': () =>
      parsed &&
      parsed.eioType === '4' && // EIO_MESSAGE
      parsed.sioType === sioConnectType,
  });
}

/**
 * Assert that WebSocket connection duration meets SLO.
 *
 * @param {number} durationMs - Connection duration in milliseconds
 * @param {number} [minDurationMs=300000] - Minimum expected duration (default: 5 minutes)
 * @returns {boolean} True if duration meets target
 */
export function assertConnectionDuration(durationMs, minDurationMs = 300000) {
  return check(null, {
    [`connection duration (${Math.round(durationMs / 1000)}s) >= target (${Math.round(minDurationMs / 1000)}s)`]: () =>
      durationMs >= minDurationMs,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Latency and SLO Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Assert that a latency measurement is within a threshold.
 *
 * @param {number} latencyMs - Measured latency in milliseconds
 * @param {number} thresholdMs - Maximum acceptable latency
 * @param {string} [label='latency'] - Description for the check
 * @returns {boolean} True if within threshold
 */
export function assertLatencyWithinSLO(latencyMs, thresholdMs, label = 'latency') {
  return check(null, {
    [`${label} (${latencyMs}ms) <= SLO (${thresholdMs}ms)`]: () => latencyMs <= thresholdMs,
  });
}

/**
 * Assert that an error rate is within budget.
 *
 * @param {number} errorCount - Number of errors
 * @param {number} totalCount - Total number of requests
 * @param {number} [maxRatePercent=1.0] - Maximum acceptable error rate percentage
 * @returns {boolean} True if within budget
 */
export function assertErrorRateWithinBudget(errorCount, totalCount, maxRatePercent = 1.0) {
  if (totalCount === 0) {
    return check(null, { 'no requests to check error rate': () => true });
  }

  const actualRate = (errorCount / totalCount) * 100;

  return check(null, {
    [`error rate (${actualRate.toFixed(2)}%) <= budget (${maxRatePercent}%)`]: () =>
      actualRate <= maxRatePercent,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Failure Classification Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Classify a failure based on HTTP response status.
 *
 * @param {Object} res - k6 HTTP response
 * @returns {{ type: string, isContract: boolean, isCapacity: boolean, isLifecycle: boolean }}
 */
export function classifyFailure(res) {
  if (!res || res.status === 0 || res.error) {
    return { type: 'capacity', isContract: false, isCapacity: true, isLifecycle: false };
  }

  const status = res.status;

  // Contract failures: client-side errors indicating API misuse
  if (status === 400 || status === 401 || status === 403) {
    return { type: 'contract', isContract: true, isCapacity: false, isLifecycle: false };
  }

  // ID lifecycle mismatch: resource not found
  if (status === 404) {
    return { type: 'lifecycle', isContract: false, isCapacity: false, isLifecycle: true };
  }

  // Capacity failures: rate limiting or server errors
  if (status === 429 || status >= 500) {
    return { type: 'capacity', isContract: false, isCapacity: true, isLifecycle: false };
  }

  // Unknown failure type
  return { type: 'unknown', isContract: false, isCapacity: false, isLifecycle: false };
}

/**
 * Assert that no contract failures occurred.
 *
 * @param {number} contractFailureCount - Number of contract failures
 * @param {number} [maxAllowed=0] - Maximum allowed contract failures
 * @returns {boolean} True if within budget
 */
export function assertNoContractFailures(contractFailureCount, maxAllowed = 0) {
  return check(null, {
    [`contract failures (${contractFailureCount}) <= max (${maxAllowed})`]: () =>
      contractFailureCount <= maxAllowed,
  });
}

/**
 * Assert that no ID lifecycle mismatches occurred.
 *
 * @param {number} lifecycleMismatchCount - Number of lifecycle mismatches
 * @param {number} [maxAllowed=0] - Maximum allowed mismatches
 * @returns {boolean} True if within budget
 */
export function assertNoLifecycleMismatches(lifecycleMismatchCount, maxAllowed = 0) {
  return check(null, {
    [`lifecycle mismatches (${lifecycleMismatchCount}) <= max (${maxAllowed})`]: () =>
      lifecycleMismatchCount <= maxAllowed,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Composite Assertions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run a comprehensive health check assertion suite.
 *
 * @param {Object} res - k6 HTTP response from health endpoint
 * @returns {boolean} True if all checks passed
 */
export function assertHealthy(res) {
  return check(res, {
    'health check successful': (r) => r.status === 200,
    'health check response time < 1s': (r) => r.timings.duration < 1000,
  });
}

/**
 * Assert that authentication succeeded.
 *
 * @param {Object} res - k6 HTTP response from login endpoint
 * @returns {{ passed: boolean, token: string|null, userId: string|null }}
 */
export function assertAuthSuccess(res) {
  let token = null;
  let userId = null;

  try {
    const body = JSON.parse(res.body);
    token = body && body.data && body.data.accessToken ? body.data.accessToken : null;
    userId = body && body.data && body.data.user && body.data.user.id ? body.data.user.id : null;
  } catch {
    token = null;
    userId = null;
  }

  const passed = check(res, {
    'login successful': (r) => r.status === 200,
    'access token present': () => typeof token === 'string' && token.length > 0,
  });

  return { passed, token, userId };
}