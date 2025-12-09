/**
 * RingRift Load Test: Concurrent Games Scenario
 *
 * Tests system behavior with 100+ simultaneous games.
 * Validates production scale assumptions for resource usage and state management.
 *
 * Scenario from STRATEGIC_ROADMAP.md §3.2: AI-Heavy Concurrent Games
 * Target: 100+ concurrent games with 200-300 players
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Gauge, Trend } from 'k6/metrics';
import { loginAndGetToken } from '../auth/helpers.js';
import { makeHandleSummary } from '../summary.js';

const thresholdsConfig = JSON.parse(open('../config/thresholds.json'));
const scenariosConfig = JSON.parse(open('../config/scenarios.json'));

// Classification metrics shared across load scenarios
export const contractFailures = new Counter('contract_failures_total');
export const idLifecycleMismatches = new Counter('id_lifecycle_mismatches_total');
export const capacityFailures = new Counter('capacity_failures_total');

// Custom metrics
const concurrentActiveGames = new Gauge('concurrent_active_games');
const gameStateErrors = new Counter('game_state_errors');
const gameStateCheckSuccess = new Rate('game_state_check_success');
const resourceOverhead = new Trend('game_resource_overhead_ms');

// Basic lifecycle tuning for how long a single gameId should be polled.
// MAX_POLLS_PER_GAME can be overridden via the GAME_MAX_POLLS env var for
// experimentation in CI without changing the script.
const TERMINAL_GAME_STATUSES = ['completed', 'abandoned', 'finished'];
const MAX_POLLS_PER_GAME = Number(__ENV.GAME_MAX_POLLS || 60);

// Threshold / SLO configuration derived from thresholds.json
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const perfEnv =
  thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const loadTestEnv =
  thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;

const gameCreationHttp = perfEnv.http_api.game_creation;
const gameStateFetchHttp = perfEnv.http_api.game_state_fetch;

// Profile selection for expected concurrency. This defaults to the "stress"
// profile but can be overridden via LOAD_PROFILE to use the "smoke" or "load"
// shapes defined in scenarios.json.
const LOAD_PROFILE = __ENV.LOAD_PROFILE || 'stress';
const profile =
  (scenariosConfig.profiles && scenariosConfig.profiles[LOAD_PROFILE]) ||
  scenariosConfig.profiles.stress;
const concurrentProfile = profile.concurrent_games;

const peakTarget =
  concurrentProfile && concurrentProfile.stages && concurrentProfile.stages.length > 0
    ? concurrentProfile.stages.reduce(
        (max, stage) => Math.max(max, stage.target || 0),
        0
      )
    : profile.max_vus || 100;

let EXPECTED_MIN_CONCURRENT_GAMES;
if (LOAD_PROFILE === 'stress') {
  // Preserve the original intent of requiring ~100 concurrent games for the
  // explicit stress profile, but bound it by the configured peak target so we
  // do not overconstrain when profiles are adjusted.
  EXPECTED_MIN_CONCURRENT_GAMES = Math.min(peakTarget || 100, 100);
} else {
  // For smoke/load profiles, require a more modest fraction of the configured
  // peak, avoiding unrealistic minima when targets are very small.
  EXPECTED_MIN_CONCURRENT_GAMES = Math.max(Math.floor((peakTarget || 1) * 0.5), 1);
}

// Test configuration for production-scale validation
export const options = {
  scenarios: {
    concurrent_games: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 }, // Ramp up to 50 VUs (50 games)
        { duration: '3m', target: 100 }, // Ramp to 100 VUs (100+ games)
        { duration: '5m', target: 100 }, // Sustain 100+ concurrent games
        { duration: '2m', target: 50 }, // Gradual ramp down
        { duration: '1m', target: 0 }, // Complete shutdown
      ],
      gracefulRampDown: '30s',
    },
  },

  thresholds: {
    // Game state retrieval should remain fast even at scale, aligned with
    // http_api.game_state_fetch in thresholds.json.
    'http_req_duration{name:get-game}': [
      `p(95)<${gameStateFetchHttp.latency_p95_ms}`,
      `p(99)<${gameStateFetchHttp.latency_p99_ms}`,
    ],

    // Game creation overhead at scale, aligned with http_api.game_creation.
    'http_req_duration{name:create-game}': [
      `p(95)<${gameCreationHttp.latency_p95_ms}`,
      `p(99)<${gameCreationHttp.latency_p99_ms}`,
    ],

    // Error rate - must remain low even at peak concurrency. We use the
    // stricter of the configured 5xx error budgets for creation/state fetch.
    http_req_failed: [
      `rate<${Math.max(
        gameCreationHttp.error_rate_5xx_percent,
        gameStateFetchHttp.error_rate_5xx_percent
      ) / 100}`,
    ],

    // Custom thresholds
    game_state_check_success: ['rate>0.99'],

    // Note: concurrency targets are now enforced via the higher-level SLO
    // verification pipeline (see verify-slos.js), which computes
    // concurrent_games from the concurrent_active_games gauge. We intentionally
    // avoid a k6 threshold on this Gauge because only the "value" aggregation
    // is supported for Gauges, and that would key off the final post-ramp-down
    // value rather than the peak reached during the run.

    // Classification counters
    contract_failures_total: [`count<=${loadTestEnv.contract_failures_total.max}`],
    id_lifecycle_mismatches_total: [
      `count<=${loadTestEnv.id_lifecycle_mismatches_total.max}`,
    ],
    capacity_failures_total: [`rate<${loadTestEnv.capacity_failures_total.rate}`],
  },

  tags: {
    scenario: 'concurrent-games',
    test_type: 'stress',
    environment: THRESHOLD_ENV,
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// Track created games per VU for state checking
let myGameId = null;
let myGamePollCount = 0;
// Per-VU flag indicating whether this VU currently owns an "active" game
// that should contribute to concurrent_active_games.
let hasActiveGame = false;

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

/**
 * Classify failures for POST /api/games when the request payload is expected
 * to be valid according to the CreateGameRequest contract.
 */
function classifyCreateGameFailure(res, createdGameId) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    return;
  }

  if (res.status === 400 || res.status === 401 || res.status === 403) {
    contractFailures.add(1);
    return;
  }

  if (res.status === 429 || res.status >= 500) {
    capacityFailures.add(1);
    return;
  }

  // 2xx but missing/malformed body or game object.
  if (res.status >= 200 && res.status < 300 && !createdGameId) {
    contractFailures.add(1);
  }
}

/**
 * Classify failures for GET /api/games/:gameId when polling concurrently
 * created games. We treat early 404s (well below MAX_POLLS_PER_GAME) as
 * potential ID lifecycle mismatches.
 */
function classifyGameStateFailure(res, pollCountBefore) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    return;
  }

  const status = res.status;

  if (status === 400 || status === 401 || status === 403) {
    contractFailures.add(1);
    return;
  }

  if (status === 404) {
    // Only flag as an ID lifecycle mismatch when the game "disappears" very
    // early in its polling lifecycle. Later 404s are treated as normal
    // cleanup/expiry.
    if (pollCountBefore < MAX_POLLS_PER_GAME / 4) {
      idLifecycleMismatches.add(1);
    }
    return;
  }

  if (status === 429 || status >= 500) {
    capacityFailures.add(1);
  }
}

export function setup() {
  console.log('Starting concurrent games stress test');
  console.log('Target: 100+ simultaneous active games');

  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper to obtain a canonical JWT for the
  // pre-seeded load-test user. All VUs share this token to avoid
  // (re)registering users during the scenario. We also wire in the
  // classification counters so auth failures are attributed correctly.
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
  const token = data.token;

  if (!token) {
    // If setup somehow failed, treat as capacity and bail quickly so we
    // do not generate misleading contract failures.
    capacityFailures.add(1);
    sleep(1);
    return;
  }

  // Each VU creates and maintains one game at a time. When a game reaches a
  // terminal state or is cleaned up, we retire its ID and create a new one on
  // a subsequent iteration to keep the concurrent-games target realistic.
  if (!myGameId) {
    createGame(baseUrl, token);
  }

  if (myGameId) {
    pollGameState(baseUrl, token);
  }

  // Simulate realistic polling interval (players checking game state)
  sleep(2 + Math.random() * 3); // 2-5 seconds between checks
}

function createGame(baseUrl, token) {
  // Step 1: Create a game (contributes to concurrent count) using the
  // canonical create-game payload shape from the API:
  //   { boardType, maxPlayers, isPrivate, timeControl, isRated, aiOpponents? }
  const boardTypes = ['square8', 'square19', 'hexagonal'];
  const boardType = boardTypes[__VU % boardTypes.length];

  const maxPlayersOptions = [2, 3, 4];
  const maxPlayers = maxPlayersOptions[__VU % maxPlayersOptions.length];

  const aiCount = 1 + (__VU % 2); // 1-2 AI opponents
  const hasAI = aiCount > 0;

  const gameConfig = {
    boardType,
    maxPlayers,
    isPrivate: false,
    timeControl: {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    },
    // AI games must be unrated per backend contract
    isRated: hasAI ? false : true,
    ...(hasAI && {
      aiOpponents: {
        count: aiCount,
        difficulty: Array(aiCount).fill(5),
        mode: 'service',
        aiType: 'heuristic',
      },
    }),
  };

  const createStart = Date.now();
  const createRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    tags: { name: 'create-game' },
  });
  const createDuration = Date.now() - createStart;

  let createdGameId = null;
  try {
    const body = JSON.parse(createRes.body);
    const game = body && body.data && body.data.game ? body.data.game : null;
    createdGameId = game ? game.id : null;

    // Sanity-check a subset of the response contract when we do receive a game.
    if (game) {
      if (game.boardType !== boardType || game.maxPlayers !== maxPlayers) {
        contractFailures.add(1);
      }

      if (game.aiOpponents && game.aiOpponents.count > 0 && game.isRated !== false) {
        // AI games must not be rated per backend contract.
        contractFailures.add(1);
      }
    }
  } catch (e) {
    createdGameId = null;
  }

  if (createRes.status === 201 && createdGameId) {
    myGameId = createdGameId;
    myGamePollCount = 0;
    console.log(`VU ${__VU}: Created game ${myGameId} in ${createDuration}ms`);
  } else {
    classifyCreateGameFailure(createRes, createdGameId);
    console.error(
      `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
    );
    gameStateErrors.add(1);
    // Without a valid game ID there is nothing useful to poll this iteration.
    sleep(2);
  }
}

function pollGameState(baseUrl, token) {
  // Step 2: Continuously monitor game state (validates state management at scale)
  const pollCountBefore = myGamePollCount;
  const stateStart = Date.now();
  const stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
    headers: { Authorization: `Bearer ${token}` },
    tags: { name: 'get-game' },
  });
  const stateDuration = Date.now() - stateStart;

  let stateValid = false;
  let retireGameId = false;

  if (stateRes.status === 200) {
    let game = null;
    try {
      const body = JSON.parse(stateRes.body);
      game = body && body.data && body.data.game ? body.data.game : null;
    } catch (e) {
      game = null;
    }

    if (!game) {
      // 200 with an unparseable or missing game payload is a contract failure.
      contractFailures.add(1);
    } else {
      const checksOk = check(stateRes, {
        'game state retrieved': (r) => r.status === 200,
        'game ID matches': () => !!(game && game.id === myGameId),
        'game has players': () => {
          const playerIds = [
            game.player1Id,
            game.player2Id,
            game.player3Id,
            game.player4Id,
          ].filter(Boolean);
          return playerIds.length > 0;
        },
      });

      stateValid = checksOk;

      if (!checksOk) {
        contractFailures.add(1);
      }

      if (checksOk) {
        const status = game.status;
        const isTerminalStatus = TERMINAL_GAME_STATUSES.indexOf(status) !== -1;

        if (isTerminalStatus) {
          // Backend has marked this game as finished; retire the ID so we stop
          // polling it and allow a new game to be created on a later iteration.
          retireGameId = true;
          console.log(
            `VU ${__VU}: Game ${myGameId} reached terminal status ${status}; retiring from polling`
          );
        } else {
          // First successful poll of a non-terminal game: start counting it
          // towards the concurrent_active_games gauge.
          if (myGamePollCount === 0) {
            markGameActive();
          }

          myGamePollCount += 1;

          // Guardrail: even if a game remains in a non-terminal state, stop
          // polling it after a bounded number of checks so we do not keep very
          // old IDs hot forever.
          if (myGamePollCount >= MAX_POLLS_PER_GAME) {
            retireGameId = true;
            console.log(
              `VU ${__VU}: Retiring game ${myGameId} after ${myGamePollCount} polls (lifecycle budget reached)`
            );
          }
        }
      }
    }
  } else if (stateRes.status === 404) {
    // The ID was once valid (came from POST /api/games) but the row is no
    // longer present (expired/cleaned up). Treat this as "no longer pollable"
    // and retire the ID so we don't keep hammering a 404. We also allow the
    // classifier to flag obviously early disappearances as lifecycle
    // mismatches.
    stateValid = false;
    retireGameId = true;
    classifyGameStateFailure(stateRes, pollCountBefore);
    console.log(
      `VU ${__VU}: Game ${myGameId} not found (404); treating as expired and stopping polling for this ID`
    );
  } else if (stateRes.status === 400) {
    // 400 from GET /api/games/:gameId means the ID format itself is invalid.
    // Since all IDs in this scenario come from POST /api/games, this
    // indicates a scenario bug rather than expected behaviour.
    stateValid = false;
    retireGameId = true;
    classifyGameStateFailure(stateRes, pollCountBefore);
    console.error(
      `VU ${__VU}: Received 400 for GET /api/games/${myGameId}; invalid ID format indicates a scenario bug`
    );
  } else if (stateRes.status === 429) {
    // Rate limiting is an expected source of failure at high concurrency, so
    // we record it as a capacity failure but keep the ID for future polling.
    stateValid = false;
    classifyGameStateFailure(stateRes, pollCountBefore);
    console.warn(
      `VU ${__VU}: Rate limited when fetching game ${myGameId} (429); backend capacity limit reached`
    );
  } else {
    // Other 4xx/5xx responses are treated as backend or auth issues that
    // should surface as failures in the k6 metrics.
    stateValid = false;
    classifyGameStateFailure(stateRes, pollCountBefore);
    console.error(
      `VU ${__VU}: Unexpected status ${stateRes.status} for game ${myGameId} - body=${stateRes.body}`
    );
  }

  gameStateCheckSuccess.add(stateValid);
  resourceOverhead.add(stateDuration);

  if (!stateValid) {
    gameStateErrors.add(1);
  }

  if (retireGameId) {
    markGameInactive();
    myGameId = null;
    myGamePollCount = 0;
  }
}

export function teardown() {
  console.log('Concurrent games stress test complete');
  console.log('Check metrics for:');
  console.log('  - Peak concurrent games reached');
  console.log('  - Game state retrieval latency at scale');
  console.log('  - Memory/CPU resource trends (via Prometheus)');
}

export const handleSummary = makeHandleSummary('concurrent-games');