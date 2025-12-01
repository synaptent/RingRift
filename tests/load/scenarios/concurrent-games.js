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

// Custom metrics
const activeGames = new Gauge('concurrent_active_games');
const gameStateErrors = new Counter('game_state_errors');
const gameStateCheckSuccess = new Rate('game_state_check_success');
const resourceOverhead = new Trend('game_resource_overhead_ms');

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
      'p95<400',   // Staging: p95 ≤ 400ms for GET /api/games/:id
      'p99<800'    // Staging: p99 ≤ 800ms
    ],
    
    // Game creation overhead at scale
    'http_req_duration{name:create-game}': [
      'p95<800',
      'p99<1500'
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
let myToken = null;

export function setup() {
  console.log('Starting concurrent games stress test');
  console.log('Target: 100+ simultaneous active games');
  
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200
  });
  
  return { baseUrl: BASE_URL };
}

export default function(data) {
  const baseUrl = data.baseUrl;
  
  // Each VU creates and maintains one game
  if (!myGameId) {
    // Step 1: Authentication
    const userId = `stress-user-${__VU}`;
    const password = 'StressTest123!';
    
    const registerRes = http.post(`${baseUrl}${API_PREFIX}/auth/register`, JSON.stringify({
      username: userId,
      email: `${userId}@stresstest.local`,
      password: password
    }), {
      headers: { 'Content-Type': 'application/json' },
      tags: { name: 'auth-register' }
    });
    
    sleep(0.3);
    
    const loginRes = http.post(`${baseUrl}${API_PREFIX}/auth/login`, JSON.stringify({
      username: userId,
      password: password
    }), {
      headers: { 'Content-Type': 'application/json' },
      tags: { name: 'auth-login' }
    });
    
    if (loginRes.status !== 200) {
      console.error(`VU ${__VU}: Login failed - ${loginRes.status}`);
      return;
    }
    
    myToken = JSON.parse(loginRes.body).token;
    
    sleep(0.5);
    
    // Step 2: Create a game (contributes to concurrent count)
    const boardTypes = ['square8', 'square19', 'hexagonal'];
    const gameConfig = {
      name: `Stress Test Game ${__VU}`,
      boardType: boardTypes[__VU % boardTypes.length],
      playerCount: 2 + (__VU % 3), // 2, 3, or 4 players
      isPrivate: false,
      aiPlayers: 1 + (__VU % 2) // 1-2 AI players
    };
    
    const createStart = Date.now();
    const createRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${myToken}`
      },
      tags: { name: 'create-game' }
    });
    const createDuration = Date.now() - createStart;
    
    if (createRes.status === 201) {
      myGameId = JSON.parse(createRes.body).id;
      console.log(`VU ${__VU}: Created game ${myGameId} in ${createDuration}ms`);
    } else {
      console.error(`VU ${__VU}: Game creation failed - ${createRes.status}`);
      gameStateErrors.add(1);
      return;
    }
  }
  
  // Step 3: Continuously monitor game state (validates state management at scale)
  if (myGameId && myToken) {
    const stateStart = Date.now();
    const stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
      headers: { 'Authorization': `Bearer ${myToken}` },
      tags: { name: 'get-game' }
    });
    const stateDuration = Date.now() - stateStart;
    
    const stateValid = check(stateRes, {
      'game state retrieved': (r) => r.status === 200,
      'game ID matches': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.id === myGameId;
        } catch {
          return false;
        }
      },
      'game has players': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.players && body.players.length > 0;
        } catch {
          return false;
        }
      }
    });
    
    gameStateCheckSuccess.add(stateValid);
    resourceOverhead.add(stateDuration);
    
    if (!stateValid) {
      gameStateErrors.add(1);
      console.error(`VU ${__VU}: Game state check failed for ${myGameId}`);
    }
  }
  
  // Update concurrent games metric
  // Note: This is approximate as VUs may be ramping
  activeGames.add(__VU);
  
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