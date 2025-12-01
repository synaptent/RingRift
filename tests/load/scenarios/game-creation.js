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

// Custom metrics
const gameCreationErrors = new Counter('game_creation_errors');
const gameCreationSuccess = new Rate('game_creation_success_rate');
const gameCreationLatency = new Trend('game_creation_latency_ms');

// Test configuration aligned with production staging SLOs
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Warm up: ramp to 10 users
    { duration: '1m', target: 50 },    // Load: ramp to 50 users
    { duration: '2m', target: 50 },    // Sustain: hold at 50 users
    { duration: '30s', target: 0 }     // Ramp down
  ],
  
  thresholds: {
    // HTTP request duration - staging SLOs from STRATEGIC_ROADMAP.md §2.1
    'http_req_duration': [
      'p95<800',   // Staging: p95 ≤ 800ms for POST /api/games
      'p99<1500'   // Staging: p99 ≤ 1500ms
    ],
    
    // Error rate - staging SLO < 1.0%
    'http_req_failed': ['rate<0.01'],
    
    // Custom metrics
    'game_creation_success_rate': ['rate>0.99'],
    'game_creation_latency_ms': ['p95<800', 'p99<1500']
  },
  
  // Test metadata
  tags: {
    scenario: 'game-creation',
    test_type: 'load',
    environment: 'staging'
  }
};

// Test configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

/**
 * Setup function - runs once per VU before iterations
 */
export function setup() {
  console.log(`Starting game creation load test against ${BASE_URL}`);
  console.log('Target load: 50 concurrent users creating games');
  
  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200
  });
  
  return { baseUrl: BASE_URL };
}

/**
 * Main test function - runs repeatedly for each VU
 */
export default function(data) {
  const baseUrl = data.baseUrl;
  
  // Step 1: Register/Login
  const userId = `load-test-user-${__VU}-${Date.now()}`;
  const password = 'TestPassword123!';
  
  const registerRes = http.post(`${baseUrl}${API_PREFIX}/auth/register`, JSON.stringify({
    username: userId,
    email: `${userId}@loadtest.local`,
    password: password
  }), {
    headers: { 'Content-Type': 'application/json' },
    tags: { name: 'auth-register' }
  });
  
  const registerSuccess = check(registerRes, {
    'registration successful': (r) => r.status === 201 || r.status === 409, // 409 if already exists
  });
  
  if (!registerSuccess) {
    gameCreationErrors.add(1);
    return;
  }
  
  sleep(0.5); // Brief pause between operations
  
  // Step 2: Login
  const loginRes = http.post(`${baseUrl}${API_PREFIX}/auth/login`, JSON.stringify({
    username: userId,
    password: password
  }), {
    headers: { 'Content-Type': 'application/json' },
    tags: { name: 'auth-login' }
  });
  
  const loginSuccess = check(loginRes, {
    'login successful': (r) => r.status === 200,
    'token received': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.token && body.token.length > 0;
      } catch {
        return false;
      }
    }
  });
  
  if (!loginSuccess) {
    gameCreationErrors.add(1);
    return;
  }
  
  const token = JSON.parse(loginRes.body).token;
  const authHeaders = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };
  
  sleep(0.5);
  
  // Step 3: Create Game (main scenario focus)
  const boardTypes = ['square8', 'square19', 'hexagonal'];
  const boardType = boardTypes[Math.floor(Math.random() * boardTypes.length)];
  const playerCounts = [2, 3, 4];
  const playerCount = playerCounts[Math.floor(Math.random() * playerCounts.length)];
  
  const gameConfig = {
    name: `Load Test Game ${__VU}-${__ITER}`,
    boardType: boardType,
    playerCount: playerCount,
    isPrivate: false,
    aiPlayers: Math.floor(Math.random() * (playerCount - 1)) // 0 to playerCount-1 AI players
  };
  
  const startTime = Date.now();
  const createGameRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
    headers: authHeaders,
    tags: { name: 'create-game' }
  });
  const createGameDuration = Date.now() - startTime;
  
  const gameCreated = check(createGameRes, {
    'game created successfully': (r) => r.status === 201,
    'game ID returned': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id && body.id.length > 0;
      } catch {
        return false;
      }
    },
    'game config matches': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.boardType === boardType && body.playerCount === playerCount;
      } catch {
        return false;
      }
    }
  });
  
  // Track metrics
  gameCreationLatency.add(createGameDuration);
  gameCreationSuccess.add(gameCreated);
  
  if (!gameCreated) {
    gameCreationErrors.add(1);
    console.error(`Game creation failed for VU ${__VU}: ${createGameRes.status} - ${createGameRes.body}`);
    return;
  }
  
  const gameId = JSON.parse(createGameRes.body).id;
  
  sleep(0.5);
  
  // Step 4: Fetch Game State (validates read path)
  const getGameRes = http.get(`${baseUrl}${API_PREFIX}/games/${gameId}`, {
    headers: authHeaders,
    tags: { name: 'get-game' }
  });
  
  check(getGameRes, {
    'game state retrieved': (r) => r.status === 200,
    'game state valid': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id === gameId;
      } catch {
        return false;
      }
    }
  });
  
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