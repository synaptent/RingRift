/**
 * RingRift Load Test: WebSocket Gameplay Scenario
 *
 * Canonical WebSocket gameplay harness for move RTT / stall metrics.
 *
 * Implements S1/S2 from LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md:
 *  - Health check + auth
 *  - Game creation (human vs AI)
 *  - WebSocket connect + join_game
 *  - player_move_by_id over Socket.IO
 *  - game_state based RTT measurement
 */

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter, Gauge } from 'k6/metrics';
import { getValidToken, loginAndGetToken, getBypassHeaders } from '../auth/helpers.js';
import { makeHandleSummary } from '../summary.js';

const thresholdsConfig = JSON.parse(open('../config/thresholds.json'));
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const perfEnv =
  thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const moveSubmissionSlo =
  (perfEnv.websocket_gameplay && perfEnv.websocket_gameplay.move_submission) ||
  thresholdsConfig.environments.staging.websocket_gameplay.move_submission;
const connectionStabilitySlo =
  (perfEnv.websocket_gameplay && perfEnv.websocket_gameplay.connection_stability) ||
  thresholdsConfig.environments.staging.websocket_gameplay.connection_stability;
const loadTestConfig =
  thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;
const trueErrorRateTarget =
  loadTestConfig &&
  loadTestConfig.true_errors &&
  typeof loadTestConfig.true_errors.rate === 'number'
    ? loadTestConfig.true_errors.rate
    : 0.005;

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

const wsMoveRtt = new Trend('ws_move_rtt_ms');
const wsMoveSuccess = new Rate('ws_move_success_rate');
const wsMovesAttempted = new Counter('ws_moves_attempted_total');
const wsMoveStalled = new Counter('ws_move_stalled_total');

const wsConnectionSuccess = new Rate('ws_connection_success_rate');
const wsHandshakeSuccess = new Rate('ws_handshake_success_rate');
const wsReconnectAttempts = new Counter('ws_reconnect_attempts_total');
const wsReconnectSuccess = new Rate('ws_reconnect_success_rate');
const wsReconnectLatency = new Trend('ws_reconnect_latency_ms');

const wsErrorMoveRejected = new Counter('ws_error_move_rejected_total');
const wsErrorAccessDenied = new Counter('ws_error_access_denied_total');
const wsErrorInternalError = new Counter('ws_error_internal_error_total');
const authTokenExpired = new Counter('auth_token_expired_total');
const rateLimitHit = new Counter('rate_limit_hit_total');
const trueErrors = new Counter('true_errors_total');

const concurrentActiveGames = new Gauge('concurrent_active_games');

// ─────────────────────────────────────────────────────────────────────────────
// Environment configuration
// ─────────────────────────────────────────────────────────────────────────────

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// WebSocket origin used for Socket.IO connections.
// Defaults to BASE_URL with http(s) → ws(s) when WS_URL is not set.
const WS_URL = (() => {
  const explicit = __ENV.WS_URL;
  if (explicit && explicit.length > 0) {
    return explicit;
  }
  return BASE_URL.replace(/^http/, 'ws');
})();

// Scenario selection: smoke (default), baseline (10–25 games), or throughput.
// Prefer an explicit WS_GAMEPLAY_MODE when provided, but fall back to the
// legacy ENABLE_WS_GAMEPLAY_THROUGHPUT flag for compatibility.
const MODE = (() => {
  const explicit = String(__ENV.WS_GAMEPLAY_MODE || '').toLowerCase();
  if (
    explicit === 'smoke' ||
    explicit === 'baseline' ||
    explicit === 'throughput' ||
    explicit === 'target'
  ) {
    return explicit;
  }

  const raw = String(__ENV.ENABLE_WS_GAMEPLAY_THROUGHPUT || '').toLowerCase();
  if (raw === 'true' || raw === '1' || raw === 'yes' || raw === 'on') {
    return 'throughput';
  }

  return 'smoke';
})();

// Gameplay tuning (overrideable via env vars).
const GAME_MAX_MOVES = Number(__ENV.GAME_MAX_MOVES || 40);
const GAME_MAX_LIFETIME_S = Number(__ENV.GAME_MAX_LIFETIME_S || 600);
const DEFAULT_VU_MAX_GAMES = MODE === 'throughput' ? 10 : 3;
const VU_MAX_GAMES = Number(__ENV.VU_MAX_GAMES || DEFAULT_VU_MAX_GAMES);

// Baseline target concurrent games (VUs) for MODE="baseline".
// Clamped to [10, 25] to satisfy the 10–25 game baseline requirement.
const BASELINE_TARGET_VUS = Number(__ENV.WS_GAMEPLAY_BASELINE_VUS || 20);
const BASELINE_VUS_CLAMPED = Math.max(10, Math.min(BASELINE_TARGET_VUS, 25));

const TERMINAL_GAME_STATUSES = ['completed', 'abandoned', 'finished'];

const WS_GAMEPLAY_DEBUG = (() => {
  const raw = String(__ENV.WS_GAMEPLAY_DEBUG || '').toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
})();

const MOVE_STALL_THRESHOLD_MS = Number(
  __ENV.WS_MOVE_STALL_THRESHOLD_MS || moveSubmissionSlo.stall_threshold_ms || 2000
);
const MOVE_RTT_TIMEOUT_MS = Number(__ENV.WS_MOVE_RTT_TIMEOUT_MS || 10000);
const WS_RECONNECT_PROBABILITY = Number(__ENV.WS_RECONNECT_PROBABILITY || 0);
const WS_RECONNECT_MAX_PER_GAME = Number(__ENV.WS_RECONNECT_MAX_PER_GAME || 1);
const WS_RECONNECT_DELAY_MS = Number(__ENV.WS_RECONNECT_DELAY_MS || 1000);

// ─────────────────────────────────────────────────────────────────────────────
// k6 options (scenarios + thresholds)
// ─────────────────────────────────────────────────────────────────────────────

const smokeScenario = {
  executor: 'ramping-vus',
  startVUs: 0,
  stages: [
    { duration: '30s', target: 1 },
    { duration: '2m', target: 2 },
    { duration: '30s', target: 0 },
  ],
  gracefulRampDown: '30s',
};

// Baseline scenario: 10–25 concurrent games over a short steady-state window.
// This is the default profile for staging 10–25 game baselines when
// WS_GAMEPLAY_MODE=baseline.
const baselineScenario = {
  executor: 'ramping-vus',
  startVUs: 0,
  stages: [
    // Ramp up to roughly half the target to avoid sudden spikes.
    { duration: '1m', target: Math.max(5, Math.floor(BASELINE_VUS_CLAMPED / 2)) },
    // Hold between 10 and 25 concurrent games for the main measurement window.
    { duration: '4m', target: BASELINE_VUS_CLAMPED },
    // Graceful ramp down.
    { duration: '1m', target: 0 },
  ],
  gracefulRampDown: '30s',
};

 // Throughput scenario is intended for higher-scale P-01 runs.
const throughputScenario = {
  executor: 'ramping-vus',
  startVUs: 0,
  stages: [
    // Staging/perf example shape (0 → 20 → 40 VUs, then sustain).
    { duration: '5m', target: 20 },
    { duration: '5m', target: 40 },
    { duration: '10m', target: 40 },
    { duration: '5m', target: 0 },
  ],
  gracefulRampDown: '1m',
};

// Target-scale scenario: align with target-scale.json (100 games / 300 players).
const targetScenario = {
  executor: 'ramping-vus',
  startVUs: 0,
  stages: [
    { duration: '2m', target: 30 },
    { duration: '5m', target: 150 },
    { duration: '5m', target: 150 },
    { duration: '5m', target: 300 },
    { duration: '10m', target: 300 },
    { duration: '3m', target: 0 },
  ],
  gracefulRampDown: '1m',
};

// Select scenarios based on MODE.
const scenarios = {};
if (MODE === 'baseline') {
  scenarios.websocket_gameplay_baseline = baselineScenario;
} else if (MODE === 'throughput') {
  scenarios.websocket_gameplay_throughput = throughputScenario;
} else if (MODE === 'target') {
  scenarios.websocket_gameplay_target = targetScenario;
} else {
  // Default to the original smoke profile for dev/CI.
  scenarios.websocket_gameplay_smoke = smokeScenario;
}

export const options = {
  scenarios,
  thresholds: {
    ws_move_rtt_ms: [
      `p(95)<${moveSubmissionSlo.end_to_end_latency_p95_ms}`,
      `p(99)<${moveSubmissionSlo.end_to_end_latency_p99_ms}`,
    ],
    ws_move_success_rate: ['rate>0.95'],
    ws_move_stalled_total: ['count<10'],
    ws_moves_attempted_total: ['count>0'],
    ws_connection_success_rate: [
      `rate>${connectionStabilitySlo.connection_success_rate_percent / 100}`,
    ],
    ws_handshake_success_rate: [
      `rate>${connectionStabilitySlo.connection_success_rate_percent / 100}`,
    ],

    // True error rate: errors excluding auth (401) and rate limiting (429)
    // This provides the real application error rate for SLO validation.
    true_errors_total: [`rate<${trueErrorRateTarget}`],
  },
  tags: {
    scenario: 'websocket-gameplay',
    test_type: 'load',
    environment: THRESHOLD_ENV,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Engine.IO / Socket.IO framing helpers
// ─────────────────────────────────────────────────────────────────────────────

const EIO_OPEN = '0';
const EIO_CLOSE = '1';
const EIO_PING = '2';
const EIO_PONG = '3';
const EIO_MESSAGE = '4';

const SIO_CONNECT = '0';
const SIO_DISCONNECT = '1';
const SIO_EVENT = '2';
const SIO_ACK = '3';

/**
 * Parse an Engine.IO / Socket.IO message from the WebSocket transport.
 */
function parseSocketIOMessage(raw) {
  if (!raw || raw.length === 0) {
    return { error: 'empty message' };
  }

  const eioType = raw[0];

  if (eioType === EIO_OPEN) {
    try {
      const payload = raw.slice(1);
      const data = payload ? JSON.parse(payload) : {};
      return { eioType, data };
    } catch (e) {
      return { eioType, error: 'invalid open payload' };
    }
  }

  if (eioType === EIO_CLOSE) {
    return { eioType };
  }

  if (eioType === EIO_PING) {
    return { eioType, data: raw.slice(1) || null };
  }

  if (eioType === EIO_PONG) {
    return { eioType, data: raw.slice(1) || null };
  }

  if (eioType === EIO_MESSAGE) {
    if (raw.length < 2) {
      return { eioType, error: 'truncated message' };
    }

    const sioType = raw[1];
    const payload = raw.slice(2);

    if (sioType === SIO_CONNECT) {
      try {
        const data = payload ? JSON.parse(payload) : {};
        return { eioType, sioType, data };
      } catch (e) {
        return { eioType, sioType, data: payload };
      }
    }

    if (sioType === SIO_EVENT) {
      try {
        const arr = JSON.parse(payload);
        const eventName = arr[0];
        const eventData = arr.slice(1);
        return { eioType, sioType, event: eventName, data: eventData };
      } catch (e) {
        return { eioType, sioType, error: 'invalid event payload' };
      }
    }

    if (sioType === SIO_ACK) {
      return { eioType, sioType, data: payload };
    }

    if (sioType === SIO_DISCONNECT) {
      return { eioType, sioType };
    }

    return { eioType, sioType, data: payload };
  }

  return { error: 'unknown packet type: ' + eioType };
}

function buildSocketIOEvent(eventName, data) {
  return EIO_MESSAGE + SIO_EVENT + JSON.stringify([eventName, data]);
}

function buildSocketIOConnect(auth) {
  if (auth) {
    return EIO_MESSAGE + SIO_CONNECT + JSON.stringify({ auth });
  }
  return EIO_MESSAGE + SIO_CONNECT;
}

function buildEnginePong(probe) {
  return EIO_PONG + (probe || '');
}

function buildSocketIoEndpoint(wsBase, token, vu) {
  let origin = wsBase || '';
  if (origin.startsWith('http://') || origin.startsWith('https://')) {
    origin = origin.replace(/^http/, 'ws');
  }
  origin = origin.replace(/\/$/, '');

  const path = '/socket.io/';
  const params = ['EIO=4', 'transport=websocket'];
  if (typeof vu !== 'undefined') {
    params.push('vu=' + String(vu));
  }
  if (token) {
    params.push('token=' + encodeURIComponent(token));
  }

  const query = params.join('&');
  return origin + path + '?' + query;
}

// ─────────────────────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────────────────────

export function setup() {
  console.log('[websocket-gameplay] Starting WebSocket gameplay scenario');
  console.log(`[websocket-gameplay] Mode: ${MODE}`);
  console.log(`[websocket-gameplay] BASE_URL=${BASE_URL}, WS_URL=${WS_URL}`);
  console.log('[websocket-gameplay] Moves submitted over WebSockets only');

  const health = http.get(`${BASE_URL}/health`);
  check(health, { 'health check successful': (r) => r.status === 200 });

  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-websocket-gameplay' },
  });

  return { baseUrl: BASE_URL, wsUrl: WS_URL, token, userId };
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-VU state
// ─────────────────────────────────────────────────────────────────────────────

let myGameId = null;
let myMovesInCurrentGame = 0;
let myGameCreatedAt = 0;
let gamesCompletedByVu = 0;
let myLastObservedMoveNumberForPlayer = 0;
let hasActiveGame = false;
let reconnectsForGame = 0;
let pendingReconnect = false;
let reconnectStartAtMs = 0;
let reconnectDeadlineAtMs = 0;

function markGameActive() {
  if (!hasActiveGame) {
    hasActiveGame = true;
    concurrentActiveGames.add(1);
  }
}

function markGameInactive() {
  if (hasActiveGame) {
    hasActiveGame = false;
    concurrentActiveGames.add(-1);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main VU function
// ─────────────────────────────────────────────────────────────────────────────

export default function (data) {
  const baseUrl = data.baseUrl;
  const wsUrl = data.wsUrl;
  let token = data.token;
  const userId = data.userId;

  if (gamesCompletedByVu >= VU_MAX_GAMES) {
    // VU has finished its allotted games; idle with light think time.
    sleep(3);
    return;
  }

  // Step 1: Game creation (HTTP /api/games).
  if (!myGameId) {
    const createPayload = {
      boardType: 'square8',
      maxPlayers: 2,
      isPrivate: false,
      timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
      isRated: false,
      aiOpponents: {
        count: 1,
        difficulty: [5],
        mode: 'service',
        aiType: 'heuristic',
      },
    };

    // Always obtain a currently-valid token before creating a game so long
    // runs are not dominated by AUTH_TOKEN_EXPIRED responses.
    const auth = getValidToken(baseUrl, {
      apiPrefix: API_PREFIX,
      tags: { name: 'auth-login-websocket-gameplay' },
    });
    token = auth.token;

    let createRes = http.post(
      `${baseUrl}${API_PREFIX}/games`,
      JSON.stringify(createPayload),
      {
        headers: getBypassHeaders({
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        }),
        tags: { name: 'create-game-websocket' },
      }
    );

    // If the token has expired, refresh once and retry game creation with a
    // fresh token. This keeps long-running tests from being swamped by
    // AUTH_TOKEN_EXPIRED errors.
    if (createRes.status === 401) {
      let errorCode = null;
      try {
        const errorBody = JSON.parse(createRes.body);
        errorCode =
          (errorBody && errorBody.code) ||
          (errorBody && errorBody.error && errorBody.error.code) ||
          null;
      } catch (e) {
        errorCode = null;
      }

      // Classify 401 as auth token expired (not a true error)
      authTokenExpired.add(1);

      if (errorCode === 'AUTH_TOKEN_EXPIRED') {
        const refreshed = getValidToken(baseUrl, {
          apiPrefix: API_PREFIX,
          tags: { name: 'auth-login-websocket-gameplay-refresh' },
          forceRefresh: true,
        });
        token = refreshed.token;

        createRes = http.post(
          `${baseUrl}${API_PREFIX}/games`,
          JSON.stringify(createPayload),
          {
            headers: getBypassHeaders({
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            }),
            tags: { name: 'create-game-websocket' },
          }
        );
      }
    }

    // Handle rate limiting as a separate classification (not a true error)
    if (createRes.status === 429) {
      rateLimitHit.add(1);
      console.warn(`VU ${__VU}: Rate limited during game creation (429)`);
      sleep(2);
      return;
    }

    let createdGameId = null;
    try {
      const body = JSON.parse(createRes.body);
      const game = body && body.data && body.data.game ? body.data.game : null;
      createdGameId = game ? game.id : null;
    } catch (e) {
      createdGameId = null;
    }

    if (createRes.status === 201 && createdGameId) {
      myGameId = createdGameId;
      myMovesInCurrentGame = 0;
      myGameCreatedAt = Date.now();
      myLastObservedMoveNumberForPlayer = 0;
      reconnectsForGame = 0;
      pendingReconnect = false;
      reconnectStartAtMs = 0;
      reconnectDeadlineAtMs = 0;
      console.log(`VU ${__VU}: Created AI game ${myGameId} for WebSocket gameplay`);
    } else {
      // Record as true error only if not already classified as auth/rate-limit
      if (createRes.status !== 401 && createRes.status !== 429) {
        trueErrors.add(1);
      }
      console.error(
        `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
      );
      sleep(2);
      return;
    }
  }

  // Step 2: WebSocket connect and gameplay loop for current game.
  const wsEndpoint = buildSocketIoEndpoint(wsUrl, token, __VU);

  let connectionOpened = false;
  let handshakeComplete = false;
  let awaitingRttForMove = false;
  let lastSentAt = 0;
  let pendingMoveId = null;
  let myPlayerNumberInGame = null;
  let shouldRetireGame = false;
  let reconnectAttemptResolved = false;
  const isReconnectAttempt = pendingReconnect === true;

  const connectionStart = Date.now();

  const res = ws.connect(
    wsEndpoint,
    {
      headers: getBypassHeaders({
        'User-Agent': 'k6-websocket-gameplay',
      }),
      tags: {
        scenario: 'websocket-gameplay',
        gameId: String(myGameId),
      },
    },
    function (socket) {
      socket.on('open', () => {
        connectionOpened = true;
        wsConnectionSuccess.add(1);
        console.log(`VU ${__VU}: WebSocket transport connected for game ${myGameId}`);
      });

      socket.on('message', (message) => {
        const parsed = parseSocketIOMessage(message);

        if (parsed.error) {
          console.warn(`VU ${__VU}: Socket.IO protocol error - ${parsed.error}`);
          return;
        }

        if (parsed.eioType === EIO_OPEN) {
          const connectMsg = buildSocketIOConnect({ token });
          socket.send(connectMsg);
          return;
        }

        if (parsed.eioType === EIO_PING) {
          const pong = buildEnginePong(parsed.data);
          socket.send(pong);
          return;
        }

        if (parsed.eioType === EIO_CLOSE) {
          return;
        }

        if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_CONNECT) {
          handshakeComplete = true;
          wsHandshakeSuccess.add(1);
          console.log(`VU ${__VU}: Socket.IO handshake complete for game ${myGameId}`);

          if (isReconnectAttempt && !reconnectAttemptResolved) {
            const reconnectLatency =
              reconnectStartAtMs > 0 ? Date.now() - reconnectStartAtMs : null;
            if (reconnectLatency !== null) {
              wsReconnectLatency.add(reconnectLatency);
            }
            wsReconnectSuccess.add(1);
            reconnectAttemptResolved = true;
            pendingReconnect = false;
            reconnectStartAtMs = 0;
            reconnectDeadlineAtMs = 0;
          }

          const joinPayload = { gameId: myGameId };
          const joinMsg = buildSocketIOEvent('join_game', joinPayload);
          socket.send(joinMsg);

          if (
            WS_RECONNECT_PROBABILITY > 0 &&
            WS_RECONNECT_MAX_PER_GAME > 0 &&
            !pendingReconnect &&
            reconnectsForGame < WS_RECONNECT_MAX_PER_GAME
          ) {
            const roll = Math.random();
            if (roll <= WS_RECONNECT_PROBABILITY) {
              pendingReconnect = true;
              reconnectsForGame += 1;
              reconnectDeadlineAtMs = Date.now() + WS_RECONNECT_DELAY_MS;
              wsReconnectAttempts.add(1);
              if (WS_GAMEPLAY_DEBUG) {
                console.log(
                  `[websocket-gameplay] VU ${__VU} scheduled reconnect in ${WS_RECONNECT_DELAY_MS}ms for game ${myGameId}`
                );
              }
            }
          }
          return;
        }

        if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_EVENT) {
          const eventName = parsed.event;
          const eventArgs = parsed.data || [];
          const payload = eventArgs[0];

          if (eventName === 'game_state') {
            const updated = handleGameStateMessage({
              socket,
              payload,
              userId,
              awaitingRttForMove,
              lastSentAt,
              pendingMoveId,
              myPlayerNumberInGame,
              currentMoveCount: myMovesInCurrentGame,
            });
            awaitingRttForMove = updated.awaitingRttForMove;
            lastSentAt = updated.lastSentAt;
            pendingMoveId = updated.pendingMoveId;
            myPlayerNumberInGame = updated.myPlayerNumberInGame;
            myMovesInCurrentGame = updated.currentMoveCount;
            shouldRetireGame = updated.shouldRetireGame;
            return;
          }

          if (eventName === 'game_over') {
            handleGameOverMessage({ socket, payload });
            shouldRetireGame = true;
            if (!awaitingRttForMove) {
              socket.close();
            }
            return;
          }

          if (eventName === 'error') {
            const errorPayload = payload || {};
            handleErrorMessage(errorPayload, {
              clearAwaiting: () => {
                if (awaitingRttForMove) {
                  wsMoveSuccess.add(0);
                }
                awaitingRttForMove = false;
                pendingMoveId = null;
              },
            });
            return;
          }

          return;
        }
      });

      socket.on('close', (code) => {
        const durationMs = Date.now() - connectionStart;
        console.log(
          `VU ${__VU}: WebSocket closed for game ${myGameId} after ${durationMs}ms (code: ${code})`
        );

        if (pendingReconnect && reconnectStartAtMs === 0) {
          reconnectStartAtMs = Date.now();
        }

        if (awaitingRttForMove) {
          // Treat an unexpected close while a move is in flight as a failed,
          // stalled move for success-rate accounting.
          wsMoveSuccess.add(0);
          wsMoveStalled.add(1);
          awaitingRttForMove = false;
          pendingMoveId = null;
        }

        if (connectionOpened && !handshakeComplete) {
          wsHandshakeSuccess.add(0);
        }

        if (isReconnectAttempt && !reconnectAttemptResolved) {
          wsReconnectSuccess.add(0);
          reconnectAttemptResolved = true;
          pendingReconnect = false;
          reconnectStartAtMs = 0;
          reconnectDeadlineAtMs = 0;
        }
      });

      socket.on('error', (e) => {
        console.error(`VU ${__VU}: WebSocket error for game ${myGameId} - ${e}`);
      });

      // Hard cap on connection lifetime for this game.
      socket.setTimeout(() => {
        if (awaitingRttForMove) {
          // Consider this move failed for success-rate accounting.
          wsMoveSuccess.add(0);
          awaitingRttForMove = false;
          pendingMoveId = null;
        }
        shouldRetireGame = true;
        socket.close();
      }, GAME_MAX_LIFETIME_S * 1000);

      // Lightweight interval to allow retirement once caps or terminal states are reached.
      socket.setInterval(() => {
        if (shouldRetireGame && pendingReconnect) {
          wsReconnectSuccess.add(0);
          pendingReconnect = false;
          reconnectStartAtMs = 0;
          reconnectDeadlineAtMs = 0;
        }

        if (
          pendingReconnect &&
          !awaitingRttForMove &&
          reconnectDeadlineAtMs > 0 &&
          Date.now() >= reconnectDeadlineAtMs
        ) {
          socket.close();
          return;
        }

        if (shouldRetireGame && !awaitingRttForMove) {
          socket.close();
        }
      }, 1000);
    }
  );

  const ok = check(res, { 'WebSocket connected': (r) => r && r.status === 101 });

  if (!ok || !connectionOpened) {
    wsConnectionSuccess.add(0);
    console.error(
      `VU ${__VU}: Failed to establish WebSocket connection for game ${myGameId} - status ${
        res && res.status
      }`
    );
    if (isReconnectAttempt && !reconnectAttemptResolved) {
      wsReconnectSuccess.add(0);
      reconnectAttemptResolved = true;
      pendingReconnect = false;
      reconnectStartAtMs = 0;
      reconnectDeadlineAtMs = 0;
    }
    // Keep the current myGameId and retry in the next iteration.
    sleep(2);
    return;
  }

  // If the game reached a terminal state or move/lifetime caps, retire it.
  const lifetimeMs = Date.now() - myGameCreatedAt;
  if (
    lifetimeMs >= GAME_MAX_LIFETIME_S * 1000 ||
    myMovesInCurrentGame >= GAME_MAX_MOVES ||
    shouldRetireGame
  ) {
    markGameInactive();
    console.log(
      `VU ${__VU}: Retiring game ${myGameId} after ${myMovesInCurrentGame} moves and ${Math.round(
        lifetimeMs / 1000
      )}s`
    );
    myGameId = null;
    myMovesInCurrentGame = 0;
    myGameCreatedAt = 0;
    myLastObservedMoveNumberForPlayer = 0;
    reconnectsForGame = 0;
    pendingReconnect = false;
    reconnectStartAtMs = 0;
    reconnectDeadlineAtMs = 0;
    gamesCompletedByVu += 1;
    sleep(1 + Math.random() * 2);
    return;
  }

  // Short think time between WebSocket iterations when keeping the same gameId.
  sleep(1 + Math.random() * 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Message handlers
// ─────────────────────────────────────────────────────────────────────────────

function handleGameStateMessage({
  socket,
  payload,
  userId,
  awaitingRttForMove,
  lastSentAt,
  pendingMoveId,
  myPlayerNumberInGame,
  currentMoveCount,
}) {
  if (!payload || typeof payload !== 'object') {
    return {
      awaitingRttForMove,
      lastSentAt,
      pendingMoveId,
      myPlayerNumberInGame,
      currentMoveCount,
      shouldRetireGame: false,
    };
  }

  const message = payload;
  const data = message.data || {};
  const gameId = data.gameId;
  const gameState = data.gameState;
  const validMoves = data.validMoves || [];

  if (!gameState || gameId !== myGameId) {
    return {
      awaitingRttForMove,
      lastSentAt,
      pendingMoveId,
      myPlayerNumberInGame,
      currentMoveCount,
      shouldRetireGame: false,
    };
  }

  let shouldRetireGame = false;
  const status = gameState.gameStatus;
  const isTerminal = TERMINAL_GAME_STATUSES.indexOf(status) !== -1;
  const now = Date.now();

  if (!isTerminal) {
    markGameActive();
  }

  // Determine this user's player number within the game.
  if (myPlayerNumberInGame == null && Array.isArray(gameState.players)) {
    const me = gameState.players.find((p) => p && p.id === userId);
    if (me && typeof me.playerNumber === 'number') {
      myPlayerNumberInGame = me.playerNumber;

      if (WS_GAMEPLAY_DEBUG) {
        console.log(
          `[websocket-gameplay] VU ${__VU} joined game ${gameId} as player ${myPlayerNumberInGame} (userId=${userId})`
        );
      }
    } else if (WS_GAMEPLAY_DEBUG) {
      console.log(
        `[websocket-gameplay] VU ${__VU} could not determine playerNumber in game ${gameId} for userId=${userId}`
      );
    }
  }

  const hasMoveHistory =
    Array.isArray(gameState.moveHistory) && gameState.moveHistory.length > 0;

  // When not actively timing a move, keep a baseline of the last observed move
  // number for this player so we can later detect new moves as RTT completions.
  if (!awaitingRttForMove && myPlayerNumberInGame != null && hasMoveHistory) {
    const lastMyMove = findLastMoveForPlayer(gameState.moveHistory, myPlayerNumberInGame);
    if (lastMyMove && typeof lastMyMove.moveNumber === 'number') {
      if (lastMyMove.moveNumber > myLastObservedMoveNumberForPlayer) {
        myLastObservedMoveNumberForPlayer = lastMyMove.moveNumber;
      }
    }
  }

  // If we were waiting for a move RTT, either detect completion via moveHistory
  // or mark the move as stalled/failed once a timeout elapses.
  if (awaitingRttForMove && lastSentAt) {
    const elapsedMs = now - lastSentAt;

    if (elapsedMs >= MOVE_RTT_TIMEOUT_MS) {
      wsMoveSuccess.add(0);
      wsMoveStalled.add(1);
      awaitingRttForMove = false;
      pendingMoveId = null;
      lastSentAt = 0;
    } else if (hasMoveHistory && myPlayerNumberInGame != null) {
      const lastMyMove = findLastMoveForPlayer(
        gameState.moveHistory,
        myPlayerNumberInGame
      );

      if (lastMyMove) {
        const moveNumber =
          typeof lastMyMove.moveNumber === 'number' ? lastMyMove.moveNumber : null;
        const isNewForUs =
          moveNumber !== null && moveNumber > myLastObservedMoveNumberForPlayer;
        const idMatches =
          pendingMoveId && lastMyMove.id && lastMyMove.id === pendingMoveId;

        if (isNewForUs || idMatches) {
          const rtt = elapsedMs;
          wsMoveRtt.add(rtt);
          if (rtt > MOVE_STALL_THRESHOLD_MS) {
            wsMoveStalled.add(1);
          }
          wsMoveSuccess.add(1);

          awaitingRttForMove = false;
          pendingMoveId = null;
          lastSentAt = 0;

          if (moveNumber !== null && moveNumber > myLastObservedMoveNumberForPlayer) {
            myLastObservedMoveNumberForPlayer = moveNumber;
          }
        }
      }
    }
  }

  if (isTerminal) {
    shouldRetireGame = true;
    if (!awaitingRttForMove) {
      socket.close();
    }
    return {
      awaitingRttForMove,
      lastSentAt,
      pendingMoveId,
      myPlayerNumberInGame,
      currentMoveCount,
      shouldRetireGame,
    };
  }

  // If it's our turn and we're not currently timing a move, send a player_move_by_id.
  if (
    !awaitingRttForMove &&
    myPlayerNumberInGame != null &&
    gameState.currentPlayer === myPlayerNumberInGame &&
    currentMoveCount < GAME_MAX_MOVES
  ) {
    const nextMove = chooseMoveFromValidMoves(validMoves, myPlayerNumberInGame);
    if (nextMove) {
      const movePayload = {
        gameId: myGameId,
        moveId: nextMove.id,
      };
      const msg = buildSocketIOEvent('player_move_by_id', movePayload);
      socket.send(msg);
      lastSentAt = now;
      awaitingRttForMove = true;
      pendingMoveId = nextMove.id || null;
      wsMovesAttempted.add(1);
      currentMoveCount += 1;
    }
  }

  // Retire the game once we've reached the move cap.
  if (currentMoveCount >= GAME_MAX_MOVES && !awaitingRttForMove) {
    shouldRetireGame = true;
    socket.close();
  }

  return {
    awaitingRttForMove,
    lastSentAt,
    pendingMoveId,
    myPlayerNumberInGame,
    currentMoveCount,
    shouldRetireGame,
  };
}

function handleGameOverMessage({ socket, payload }) {
  if (!payload || typeof payload !== 'object') {
    return;
  }

  const data = payload.data || {};
  if (data.gameId !== myGameId) {
    return;
  }

  const status = data.gameState && data.gameState.gameStatus;
  console.log(
    `VU ${__VU}: game_over received for ${myGameId} with status ${status}`
  );
  socket.close();
}

function handleErrorMessage(errorPayload, helpers) {
  const code = errorPayload && errorPayload.code;

  if (code === 'MOVE_REJECTED') {
    wsErrorMoveRejected.add(1);
    trueErrors.add(1); // Move rejection is a true application error
    helpers.clearAwaiting();
    return;
  }

  if (code === 'ACCESS_DENIED') {
    wsErrorAccessDenied.add(1);
    // ACCESS_DENIED may be auth-related, classify separately
    authTokenExpired.add(1);
    helpers.clearAwaiting();
    return;
  }

  if (code === 'RATE_LIMITED' || code === 'TOO_MANY_REQUESTS') {
    // Rate limiting is not a true error
    rateLimitHit.add(1);
    helpers.clearAwaiting();
    return;
  }

  if (code === 'INTERNAL_ERROR') {
    wsErrorInternalError.add(1);
    trueErrors.add(1); // Internal errors are true application errors
    helpers.clearAwaiting();
    return;
  }

  // Other error codes are logged and counted as true errors
  trueErrors.add(1);
  console.warn(
    `VU ${__VU}: Unclassified WebSocket error code ${code} - message=${
      errorPayload && errorPayload.message
    }`
  );
}

function chooseMoveFromValidMoves(validMoves, myPlayerNumberInGame) {
  if (!Array.isArray(validMoves) || validMoves.length === 0) {
    return null;
  }

  const myMoves = validMoves.filter(
    (m) => m && typeof m.player === 'number' && m.player === myPlayerNumberInGame
  );

  const pool = myMoves.length > 0 ? myMoves : validMoves;

  // Deterministic selection to keep runs repeatable: always take the first
  // legal move available for this player.
  return pool[0] || null;
}

function findLastMoveForPlayer(moveHistory, playerNumber) {
  if (!Array.isArray(moveHistory) || moveHistory.length === 0 || playerNumber == null) {
    return null;
  }

  for (let i = moveHistory.length - 1; i >= 0; i -= 1) {
    const move = moveHistory[i];
    if (move && move.player === playerNumber) {
      return move;
    }
  }

  return null;
}

// Shared SLO-aware summary writer; writes
// results/load/websocket-gameplay.<env>.summary.json by default.
export const handleSummary = makeHandleSummary('websocket-gameplay');
