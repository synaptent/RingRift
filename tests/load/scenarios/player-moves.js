/**
 * RingRift Load Test: Player Move Submission Scenario
 *
 * Tests move submission latency and turn processing throughput.
 * Validates production-scale assumptions for real-time gameplay.
 *
 * Scenario from STRATEGIC_ROADMAP.md §3: Player Moves
 * SLOs from STRATEGIC_ROADMAP.md §2.2: WebSocket gameplay SLOs
 *
 * NOTE: k6 has limited WebSocket support. For full real-time testing,
 * consider supplementing with socket.io-client or Playwright tests.
 * This scenario focuses on HTTP-based move submission where available.
 *
 * Modes:
 *  - Poll-only (default): MOVE_HTTP_ENDPOINT_ENABLED unset or "false".
 *    Creates games and polls state only; no HTTP move submissions.
 *  - HTTP move harness mode: backend ENABLE_HTTP_MOVE_HARNESS=true and
 *    k6 MOVE_HTTP_ENDPOINT_ENABLED=true. Submits moves via
 *    POST /api/games/:gameId/moves and tracks move metrics.
 *
 * To run (examples):
 *   # Poll-only baseline (local dev)
 *   k6 run tests/load/scenarios/player-moves.js
 *
 *   # HTTP move harness (staging/load)
 *   ENABLE_HTTP_MOVE_HARNESS=true \\
 *   MOVE_HTTP_ENDPOINT_ENABLED=true \\
 *   k6 run tests/load/scenarios/player-moves.js
 *
 * When MOVE_HTTP_ENDPOINT_ENABLED=true but the backend harness is disabled,
 * POST /api/games/:gameId/moves will typically return 404 and move-related
 * thresholds in this scenario are expected to fail; that configuration is
 * intentionally unsupported for harness validation.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { loginAndGetToken, getValidToken, getBypassHeaders } from '../auth/helpers.js';
import { makeHandleSummary } from '../summary.js';

const thresholdsConfig = JSON.parse(open('../config/thresholds.json'));

// Classification metrics shared across load scenarios
export const contractFailures = new Counter('contract_failures_total');
export const idLifecycleMismatches = new Counter('id_lifecycle_mismatches_total');
export const capacityFailures = new Counter('capacity_failures_total');
const authTokenExpired = new Counter('auth_token_expired_total');
const rateLimitHit = new Counter('rate_limit_hit_total');
const trueErrors = new Counter('true_errors_total');

 // Custom metrics aligned with STRATEGIC_ROADMAP metrics
 const moveSubmissionLatency = new Trend('move_submission_latency_ms');
 const moveSubmissionSuccess = new Rate('move_submission_success_rate');
 const moveProcessingErrors = new Counter('move_processing_errors');
 const turnProcessingLatency = new Trend('turn_processing_latency_ms');
 // Alias for Prometheus-facing game move latency naming; this keeps
 // thresholds.json.alert_correlation aligned with the k6 metric name.
 const gameMoveLatency = new Trend('game_move_latency_ms');
 // Track stalled moves as a Rate so we can express thresholds as a fraction
 // of total moves, aligned with stall_rate_percent in thresholds.json.
 const stalledMoves = new Rate('stalled_moves_total');
 const movesAttemptedTotal = new Counter('moves_attempted_total');

 // Lifecycle tuning for how long a single gameId should be polled by this scenario.
 // MAX_POLLS_PER_GAME can be overridden via the GAME_MAX_POLLS env var without
 // changing the script.
 const TERMINAL_GAME_STATUSES = ['completed', 'abandoned', 'finished'];
 const MAX_POLLS_PER_GAME = Number(__ENV.GAME_MAX_POLLS || 60);

 // Threshold configuration derived from thresholds.json
 const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
 const perfEnv =
   thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
 const loadTestEnv =
   thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;
 const moveSubmission = perfEnv.websocket_gameplay.move_submission;
 const trueErrorRateTarget =
   loadTestEnv &&
   loadTestEnv.true_errors &&
   typeof loadTestEnv.true_errors.rate === 'number'
     ? loadTestEnv.true_errors.rate
     : 0.005;

 // Test configuration
 const moveHarnessEnabledForThresholds = (() => {
   const raw = String(__ENV.MOVE_HTTP_ENDPOINT_ENABLED || '').toLowerCase();
   return raw === 'true' || raw === '1' || raw === 'yes' || raw === 'on';
 })();
 
 const thresholds = {};
 
 if (moveHarnessEnabledForThresholds) {
   // Move submission latency - staging SLOs from STRATEGIC_ROADMAP §2.2,
   // aligned with websocket_gameplay.move_submission in thresholds.json.
   thresholds['move_submission_latency_ms'] = [
     `p(95)<${moveSubmission.end_to_end_latency_p95_ms}`,
     `p(99)<${moveSubmission.end_to_end_latency_p99_ms}`,
   ];
 
   // Stall rate - moves taking > stall_threshold_ms should be rare. We track
   // this as a Rate so we can enforce the stall_rate_percent budget directly.
   thresholds['stalled_moves_total'] = [
     `rate<${moveSubmission.stall_rate_percent / 100}`,
   ];
 
   // Success rate in harness mode; tolerate occasional failures while still
   // catching systemic transport/validation issues.
   thresholds['move_submission_success_rate'] = ['rate>0.95'];
 
   // Ensure that when MOVE_HTTP_ENDPOINT_ENABLED=true we actually exercise
   // the harness by making at least one move attempt.
   thresholds['moves_attempted_total'] = ['count>0'];
 
   // Turn processing (includes validation + state update), aligned with
   // server_processing_* thresholds.
   thresholds['turn_processing_latency_ms'] = [
     `p(95)<${moveSubmission.server_processing_p95_ms}`,
     `p(99)<${moveSubmission.server_processing_p99_ms}`,
   ];
 
   // Alias threshold for game_move_latency_ms so that alert_correlation in
   // thresholds.json can reliably map this Trend to the HighGameMoveLatency
   // alert using the same server_processing_* SLOs.
   thresholds['game_move_latency_ms'] = [
     `p(95)<${moveSubmission.server_processing_p95_ms}`,
     `p(99)<${moveSubmission.server_processing_p99_ms}`,
   ];
 }
 
 // In poll-only mode (MOVE_HTTP_ENDPOINT_ENABLED=false), no move-specific
 // thresholds are applied. The scenario remains a pure creation+polling
 // baseline and is safe to run against local dev even if the backend does
 // not expose the HTTP move harness.
 
 export const options = {
   scenarios: {
     realistic_gameplay: {
       executor: 'ramping-vus',
       startVUs: 0,
       stages: [
         { duration: '1m', target: 20 }, // Ramp up to 20 concurrent games (40 players)
         { duration: '3m', target: 40 }, // Increase to 40 games (80 players)
         { duration: '5m', target: 40 }, // Sustain realistic gameplay
         { duration: '1m', target: 0 }, // Ramp down
       ],
       gracefulRampDown: '30s',
     },
   },
 
   thresholds: {
     ...thresholds,
     // Classification counters are always enforced for this scenario so that
     // contract and capacity issues surface clearly regardless of whether the
     // HTTP move harness is enabled.
     contract_failures_total: [`count<=${loadTestEnv.contract_failures_total.max}`],
    id_lifecycle_mismatches_total: [
      `count<=${loadTestEnv.id_lifecycle_mismatches_total.max}`,
    ],
    capacity_failures_total: [`rate<${loadTestEnv.capacity_failures_total.rate}`],
    true_errors_total: [`rate<${trueErrorRateTarget}`],
  },
 
   tags: {
     scenario: 'player-moves',
     test_type: 'load',
     environment: THRESHOLD_ENV,
   },
 };

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// HTTP move submission is enabled only when the internal HTTP move harness
// endpoint is available on the backend. This flag is driven by the k6 env
// var MOVE_HTTP_ENDPOINT_ENABLED so operators can toggle between poll-only
// mode and poll+HTTP-move mode without changing the script. When false or
// unset, the scenario only creates games and polls state.
const MOVE_HTTP_ENDPOINT_ENABLED = (() => {
  const raw = String(__ENV.MOVE_HTTP_ENDPOINT_ENABLED || '').toLowerCase();
  return raw === 'true' || raw === '1' || raw === 'yes' || raw === 'on';
})();

 // Game state per VU
 let myGameId = null;
 let myGamePollCount = 0;

export function setup() {
  console.log('Starting player move submission load test');
  console.log('Focus: Move processing latency and turn throughput');

  if (MOVE_HTTP_ENDPOINT_ENABLED) {
    console.log(
      'HTTP move harness enabled; submitting moves via POST /api/games/:gameId/moves (internal endpoint)'
    );
  } else {
    console.log(
      'HTTP move harness disabled; running in poll-only mode (no move submissions from this scenario)'
    );
  }

  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper so this scenario matches the same
  // /api/auth/login contract as game-creation and concurrent-games.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
    metrics: {
      contractFailures,
      capacityFailures,
    },
  });

  return { baseUrl: BASE_URL, token, userId };
}

export default function (data) {
  const baseUrl = data.baseUrl;
  let token;

  try {
    const authResult = getValidToken(baseUrl, {
      apiPrefix: API_PREFIX,
      tags: { name: 'auth-refresh' },
      metrics: {
        contractFailures,
        capacityFailures,
      },
    });
    token = authResult.token;
  } catch (err) {
    capacityFailures.add(1);
    console.warn(`VU ${__VU}: Auth refresh failed: ${err.message}`);
    sleep(1);
    return;
  }

  if (!token) {
    capacityFailures.add(1);
    sleep(1);
    return;
  }

  // Step 1: Setup - Create game once per VU using the canonical create-game payload
  if (!myGameId) {
    const aiCount = 1; // 1 AI opponent for automated gameplay

    const createPayload = {
      boardType: 'square8', // Smaller board for faster games
      maxPlayers: 2,
      isPrivate: false,
      timeControl: {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      },
      // AI games must be unrated per backend contract
      isRated: false,
      aiOpponents: {
        count: aiCount,
        difficulty: Array(aiCount).fill(5),
        mode: 'service',
        aiType: 'heuristic',
      },
    };

    let createRes = http.post(
      `${baseUrl}${API_PREFIX}/games`,
      JSON.stringify(createPayload),
      {
        headers: getBypassHeaders({
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        }),
      }
    );

    if (createRes.status === 401) {
      authTokenExpired.add(1);
      const refreshedToken = refreshAuthToken(baseUrl, 'auth-refresh-create-game');
      if (refreshedToken) {
        token = refreshedToken;
        createRes = http.post(
          `${baseUrl}${API_PREFIX}/games`,
          JSON.stringify(createPayload),
          {
            headers: getBypassHeaders({
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            }),
          }
        );
      }
    }

    let createdGameId = null;
    try {
      const body = JSON.parse(createRes.body);
      const game = body && body.data && body.data.game ? body.data.game : null;
      createdGameId = game ? game.id : null;

      if (game) {
        // Ensure the create-game response matches the payload contract.
        if (
          game.boardType !== createPayload.boardType ||
          game.maxPlayers !== createPayload.maxPlayers
        ) {
          contractFailures.add(1);
        }

        if (game.aiOpponents && game.aiOpponents.count > 0 && game.isRated !== false) {
          contractFailures.add(1);
        }
      }
    } catch (e) {
      createdGameId = null;
    }

    if (createRes.status === 201 && createdGameId) {
      myGameId = createdGameId;
      myGamePollCount = 0;
      console.log(`VU ${__VU}: Created game ${myGameId}`);
    } else {
      classifyCreateGameFailure(createRes, createdGameId);
      console.error(
        `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
      );
      return;
    }
  }

  // Step 2: Poll game state and (optionally) submit HTTP "moves"
  if (myGameId && token) {
    // Get current game state
    const pollCountBefore = myGamePollCount;
    let stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
      headers: getBypassHeaders({ Authorization: `Bearer ${token}` }),
    });

    if (stateRes.status === 401) {
      authTokenExpired.add(1);
      const refreshedToken = refreshAuthToken(baseUrl, 'auth-refresh-get-game');
      if (refreshedToken) {
        token = refreshedToken;
        stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
          headers: getBypassHeaders({ Authorization: `Bearer ${token}` }),
        });
      }
    }

    let game = null;
    let retireGameId = false;

    if (stateRes.status === 200) {
      try {
        const body = JSON.parse(stateRes.body);
        game = body && body.data && body.data.game ? body.data.game : null;
      } catch (e) {
        game = null;
      }

      if (!game) {
        moveProcessingErrors.add(1);
        contractFailures.add(1);
        console.error(
          `VU ${__VU}: Failed to parse game state for ${myGameId} from 200 response: ${stateRes.body}`
        );
        retireGameId = true;
      } else {
        // Check if game is still active/pollable
        const status = game.status;
        const isTerminalStatus = TERMINAL_GAME_STATUSES.indexOf(status) !== -1;

        if (isTerminalStatus) {
          console.log(`VU ${__VU}: Game ${myGameId} ended with status ${status}`);
          retireGameId = true;
        } else if (status !== 'active' && status !== 'waiting' && status !== 'paused') {
          // Unknown non-terminal status; keep polling but log for diagnostics.
          console.warn(
            `VU ${__VU}: Game ${myGameId} in unexpected non-terminal status ${status}; continuing to poll`
          );
        }

        myGamePollCount += 1;

        // Guardrail: even if a game remains in a non-terminal state, stop
        // polling it after a bounded number of checks so we do not keep very
        // old IDs hot forever.
        if (!isTerminalStatus && myGamePollCount >= MAX_POLLS_PER_GAME) {
          retireGameId = true;
          console.log(
            `VU ${__VU}: Retiring game ${myGameId} after ${myGamePollCount} polls (lifecycle budget reached)`
          );
        }
      }
    } else if (stateRes.status === 404) {
      // The game ID came from POST /api/games but the row is no longer present.
      // Treat as "expired/cleaned up" and retire this ID so we stop polling it.
      moveProcessingErrors.add(1);
      retireGameId = true;
      classifyGameStateFailure(stateRes, pollCountBefore);
      console.log(
        `VU ${__VU}: Game ${myGameId} not found (404); treating as expired and stopping polling for this ID`
      );
    } else if (stateRes.status === 400) {
      // 400 implies an invalid gameId format. Since IDs originate from
      // POST /api/games, this indicates a scenario bug rather than expected
      // backend behaviour.
      moveProcessingErrors.add(1);
      retireGameId = true;
      classifyGameStateFailure(stateRes, pollCountBefore);
      console.error(
        `VU ${__VU}: Received 400 for GET /api/games/${myGameId}; invalid ID format indicates a scenario bug`
      );
    } else if (stateRes.status === 429) {
      // Rate limiting at high concurrency is expected; keep the ID and allow a
      // later iteration to retry.
      moveProcessingErrors.add(1);
      classifyGameStateFailure(stateRes, pollCountBefore);
      console.warn(
        `VU ${__VU}: Rate limited when fetching game ${myGameId} (429); backend capacity limit reached`
      );
    } else {
      moveProcessingErrors.add(1);
      classifyGameStateFailure(stateRes, pollCountBefore);
      console.error(
        `VU ${__VU}: Unexpected status ${stateRes.status} when fetching game ${myGameId} - body=${stateRes.body}`
      );
    }

    if (retireGameId) {
      myGameId = null;
      myGamePollCount = 0;
      // Simulate a player briefly reviewing the final state before starting a new game.
      sleep(5);
      return;
    }

    if (!game) {
      // We already counted an error and handled retirement above where relevant.
      sleep(2);
      return;
    }

    if (MOVE_HTTP_ENDPOINT_ENABLED) {
      // Submit a real move via the internal HTTP move harness endpoint.
      // The payload conforms to the shared MoveSchema used by both the
      // harness and the WebSocket player_move path.
      const movePayload = generateRandomMove(game);

      const moveStart = Date.now();
      movesAttemptedTotal.add(1);

      let moveRes = http.post(
        `${baseUrl}${API_PREFIX}/games/${myGameId}/moves`,
        JSON.stringify(movePayload),
        {
          headers: getBypassHeaders({
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          }),
          tags: { name: 'submit-move' },
        }
      );

      if (moveRes.status === 401) {
        authTokenExpired.add(1);
        const refreshedToken = refreshAuthToken(baseUrl, 'auth-refresh-submit-move');
        if (refreshedToken) {
          token = refreshedToken;
          moveRes = http.post(
            `${baseUrl}${API_PREFIX}/games/${myGameId}/moves`,
            JSON.stringify(movePayload),
            {
              headers: getBypassHeaders({
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`,
              }),
              tags: { name: 'submit-move' },
            }
          );
        }
      }
      const moveLatency = Date.now() - moveStart;

      // Track metrics
      moveSubmissionLatency.add(moveLatency);

      const moveSuccess = check(moveRes, {
        'move accepted': (r) => r.status >= 200 && r.status < 300,
      });

      moveSubmissionSuccess.add(moveSuccess);

      if (moveSuccess) {
        // For now we treat turn processing latency as approximately the
        // HTTP move submission latency; a future refinement can measure
        // move-to-visible-state-change latency via polling.
        turnProcessingLatency.add(moveLatency);
        gameMoveLatency.add(moveLatency);
      } else {
        moveProcessingErrors.add(1);
        classifyMoveFailure(moveRes);

        if (moveRes.status === 404) {
          console.warn(
            `VU ${__VU}: HTTP move harness returned 404 for game ${myGameId} - is ENABLE_HTTP_MOVE_HARNESS enabled on the backend?`
          );
        } else if (moveRes.status >= 500) {
          console.error(
            `VU ${__VU}: Server error submitting move for game ${myGameId} - status=${moveRes.status} body=${moveRes.body}`
          );
        } else if (moveRes.status >= 400) {
          console.warn(
            `VU ${__VU}: Move rejected for game ${myGameId} - status=${moveRes.status} body=${moveRes.body}`
          );
        }
      }

      // Track stalled moves (>2s per STRATEGIC_ROADMAP stall definition)
      if (moveLatency > moveSubmission.stall_threshold_ms) {
        stalledMoves.add(1);
        console.warn(`VU ${__VU}: Stalled move detected - ${moveLatency}ms`);
      }
    }
  }

  // Think time between moves - realistic gameplay pacing
  sleep(1 + Math.random() * 3); // 1-4 seconds between moves
}

/**
 * Generate a valid random move based on game state
 * In production, this would analyze the board and generate legal moves
 * For load testing, simplified placeholder logic
 */
function generateRandomMove(gameState) {
  // Basic, high-success-rate move generator:
  // - Always uses the canonical "place_ring" move type.
  // - Targets a random coordinate within the board bounds.
  //
  // This prioritises exercising the full move pipeline and collecting
  // meaningful latency/success metrics over strict gameplay realism.
  const boardType = gameState.boardType || 'square8';
  const size = boardType === 'square19' ? 19 : 8;

  const to = {
    x: Math.floor(Math.random() * size),
    y: Math.floor(Math.random() * size),
  };

  return {
    moveType: 'place_ring',
    position: {
      to,
    },
  };
}

/**
 * Classification helpers
 */

function classifyCreateGameFailure(res, createdGameId) {
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

  if (res.status >= 200 && res.status < 300 && !createdGameId) {
    contractFailures.add(1);
    trueErrors.add(1);
  }
}

function classifyGameStateFailure(res, pollCountBefore) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  const status = res.status;

  if (status === 401) {
    authTokenExpired.add(1);
    return;
  }

  if (status === 429) {
    rateLimitHit.add(1);
    capacityFailures.add(1);
    return;
  }

  if (status === 400 || status === 403) {
    contractFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (status === 404) {
    // Treat early disappearance of IDs as a potential lifecycle mismatch,
    // but allow later 404s to represent normal cleanup.
    if (pollCountBefore < MAX_POLLS_PER_GAME / 4) {
      idLifecycleMismatches.add(1);
      trueErrors.add(1);
    }
    return;
  }

  if (status >= 500) {
    capacityFailures.add(1);
    trueErrors.add(1);
  }
}

function classifyMoveFailure(res) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  const status = res.status;

  if (status === 401) {
    authTokenExpired.add(1);
    return;
  }

  if (status === 429) {
    rateLimitHit.add(1);
    capacityFailures.add(1);
    return;
  }

  if (status === 400 || status === 403) {
    // Move payloads are generated by the harness to conform to MoveSchema,
    // so 4xx responses here generally indicate a contract mismatch.
    contractFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (status === 404) {
    // Typically indicates that the HTTP move harness is not enabled or that
    // the game has been cleaned up unexpectedly. We treat this as a capacity /
    // environment issue rather than a pure contract failure.
    capacityFailures.add(1);
    trueErrors.add(1);
    return;
  }

  if (status >= 500) {
    capacityFailures.add(1);
    trueErrors.add(1);
  }
}

export function teardown(data) {
  console.log('Player move submission test complete');
  console.log('Key metrics to review:');
  console.log('  - move_submission_latency_ms (p95, p99)');
  console.log('  - stalled_moves_total (should be <0.5% of total moves)');
  console.log('  - move_submission_success_rate (should stay >=0.95 in harness mode)');
  console.log('  - moves_attempted_total (should be >0 when MOVE_HTTP_ENDPOINT_ENABLED=true)');
}

function refreshAuthToken(baseUrl, tagName) {
  try {
    const refreshed = getValidToken(baseUrl, {
      apiPrefix: API_PREFIX,
      tags: { name: tagName || 'auth-refresh-expired' },
      metrics: {
        contractFailures,
        capacityFailures,
      },
      forceRefresh: true,
    });
    return refreshed.token;
  } catch (err) {
    capacityFailures.add(1);
    console.warn(
      `VU ${__VU}: Auth refresh failed (${tagName || 'auth-refresh-expired'}): ${err.message}`
    );
    return null;
  }
}

export const handleSummary = makeHandleSummary('player-moves');
