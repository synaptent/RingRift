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
import { SharedArray } from 'k6/data';
import { loginAndGetToken } from '../auth/helpers.js';

// Custom metrics
const activeGames = new Gauge('concurrent_active_games');
const gameStateErrors = new Counter('game_state_errors');
const gameStateCheckSuccess = new Rate('game_state_check_success');
const resourceOverhead = new Trend('game_resource_overhead_ms');

// Basic lifecycle tuning for how long a single gameId should be polled.
// MAX_POLLS_PER_GAME can be overridden via the GAME_MAX_POLLS env var for
// experimentation in CI without changing the script.
const TERMINAL_GAME_STATUSES = ['completed', 'abandoned', 'finished'];
const MAX_POLLS_PER_GAME = Number(__ENV.GAME_MAX_POLLS || 60);

// Test configuration for production-scale validation
export const options = {
  scenarios: {
    concurrent_games: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 VUs (50 games)
        { duration: '3m', target: 100 },  // Ramp to 100 VUs (100+ games)
        { duration: '5m', target: 100 },  // Sustain 100+ concurrent games
        { duration: '2m', target: 50 },   // Gradual ramp down
        { duration: '1m', target: 0 }     // Complete shutdown
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Game state retrieval should remain fast even at scale
    'http_req_duration{name:get-game}': [
      'p(95)<400',   // Staging: p95 ≤ 400ms for GET /api/games/:id
      'p(99)<800'    // Staging: p99 ≤ 800ms
    ],
    
    // Game creation overhead at scale
    'http_req_duration{name:create-game}': [
      'p(95)<800',
      'p(99)<1500'
    ],
    
    // Error rate - must remain low even at peak concurrency
    'http_req_failed': ['rate<0.01'],
    
    // Custom thresholds
    'game_state_check_success': ['rate>0.99'],
    'concurrent_active_games': ['value>=100'],  // Confirm we reach target
  },
  
  tags: {
    scenario: 'concurrent-games',
    test_type: 'stress',
    environment: 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// Track created games per VU for state checking
let myGameId = null;
let myGamePollCount = 0;

export function setup() {
  console.log('Starting concurrent games stress test');
  console.log('Target: 100+ simultaneous active games');

  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper to obtain a canonical JWT for the
  // pre-seeded load-test user. All VUs share this token to avoid
  // (re)registering users during the scenario.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
  });

  return { baseUrl: BASE_URL, token, userId };
}

export default function(data) {
  const baseUrl = data.baseUrl;
  const token = data.token;

  // Each VU creates and maintains one game at a time. When a game reaches a
  // terminal state or is cleaned up, we retire its ID and create a new one on
  // a subsequent iteration to keep the concurrent-games target realistic.
  if (!myGameId) {
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
    } catch {
      createdGameId = null;
    }

    if (createRes.status === 201 && createdGameId) {
      myGameId = createdGameId;
      myGamePollCount = 0;
      console.log(`VU ${__VU}: Created game ${myGameId} in ${createDuration}ms`);
    } else {
      console.error(
        `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
      );
      gameStateErrors.add(1);
      // Without a valid game ID there is nothing useful to poll this iteration.
      sleep(2);
      return;
    }
  }

  // Step 3: Continuously monitor game state (validates state management at scale)
  if (myGameId && token) {
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
      } catch {
        game = null;
      }

      const checksOk = check(stateRes, {
        'game state retrieved': (r) => r.status === 200,
        'game ID matches': () => !!(game && game.id === myGameId),
        'game has players': () => {
          if (!game) return false;
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

      if (checksOk && game) {
        const status = game.status;
        const isTerminalStatus = TERMINAL_GAME_STATUSES.indexOf(status) !== -1;

        if (isTerminalStatus) {
          // Backend has marked this game as finished; retire the ID so we stop
          // polling it and allow a new game to be created on a later iteration.
          retireGameId = true;
          console.log(
            `VU ${__VU}: Game ${myGameId} reached terminal status ${status}; retiring from polling`
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
      // The ID was once valid (came from POST /api/games) but the row is no
      // longer present (expired/cleaned up). Treat this as "no longer
      // pollable" and retire the ID so we don't keep hammering a 404.
      stateValid = false;
      retireGameId = true;
      console.log(
        `VU ${__VU}: Game ${myGameId} not found (404); treating as expired and stopping polling for this ID`
      );
    } else if (stateRes.status === 400) {
      // 400 from GET /api/games/:gameId means the ID format itself is invalid.
      // Since all IDs in this scenario come from POST /api/games, this
      // indicates a scenario bug rather than expected behaviour.
      stateValid = false;
      retireGameId = true;
      console.error(
        `VU ${__VU}: Received 400 for GET /api/games/${myGameId}; invalid ID format indicates a scenario bug`
      );
    } else if (stateRes.status === 429) {
      // Rate limiting is an expected source of failure at high concurrency, so
      // we record it as an error but keep the ID for future polling.
      stateValid = false;
      console.warn(
        `VU ${__VU}: Rate limited when fetching game ${myGameId} (429); backend capacity limit reached`
      );
    } else {
      // Other 4xx/5xx responses are treated as backend or auth issues that
      // should surface as failures in the k6 metrics.
      stateValid = false;
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
      myGameId = null;
      myGamePollCount = 0;
    } else if (myGameId) {
      // Only count games that are still considered pollable toward the
      // concurrent_active_games gauge. This keeps the metric aligned with
      // "live" games rather than long-expired IDs.
      activeGames.add(__VU);
    }
  }

  // Simulate realistic polling interval (players checking game state)
  sleep(2 + Math.random() * 3); // 2-5 seconds between checks
}

export function teardown(data) {
  console.log('Concurrent games stress test complete');
  console.log('Check metrics for:');
  console.log('  - Peak concurrent games reached');
  console.log('  - Game state retrieval latency at scale');
  console.log('  - Memory/CPU resource trends (via Prometheus)');
}