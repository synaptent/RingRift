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
 */

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics aligned with STRATEGIC_ROADMAP metrics
const moveSubmissionLatency = new Trend('move_submission_latency_ms');
const moveSubmissionSuccess = new Rate('move_submission_success_rate');
const moveProcessingErrors = new Counter('move_processing_errors');
const turnProcessingLatency = new Trend('turn_processing_latency_ms');
const stalledMoves = new Counter('stalled_moves_total'); // >2s threshold per STRATEGIC_ROADMAP

// Test configuration
export const options = {
  scenarios: {
    realistic_gameplay: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },   // Ramp up to 20 concurrent games (40 players)
        { duration: '3m', target: 40 },   // Increase to 40 games (80 players)
        { duration: '5m', target: 40 },   // Sustain realistic gameplay
        { duration: '1m', target: 0 }     // Ramp down
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Move submission latency - staging SLOs from STRATEGIC_ROADMAP §2.2
    'move_submission_latency_ms': [
      'p95<300',   // Staging: 95% ≤ 300ms
      'p99<600'    // Staging: 99% ≤ 600ms
    ],
    
    // Stall rate - moves taking >2s should be rare
    'stalled_moves_total': ['count<10'], // <0.5% for staging (assuming ~2000 moves)
    
    // Success rate
    'move_submission_success_rate': ['rate>0.99'],
    
    // Turn processing (includes validation + state update)
    'turn_processing_latency_ms': [
      'p95<400',
      'p99<800'
    ]
  },
  
  tags: {
    scenario: 'player-moves',
    test_type: 'load',
    environment: 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const WS_URL = __ENV.WS_URL || 'ws://localhost:3001';
const API_PREFIX = '/api';

// Game state per VU
let myGameId = null;
let myToken = null;
let myPlayerId = null;

export function setup() {
  console.log('Starting player move submission load test');
  console.log('Focus: Move processing latency and turn throughput');
  
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200
  });
  
  return { baseUrl: BASE_URL, wsUrl: WS_URL };
}

export default function(data) {
  const baseUrl = data.baseUrl;
  
  // Step 1: Setup - Create game and authenticate (once per VU)
  if (!myGameId) {
    const userId = `move-test-user-${__VU}`;
    const password = 'MoveTest123!';
    
    // Register and login
    http.post(`${baseUrl}${API_PREFIX}/auth/register`, JSON.stringify({
      username: userId,
      email: `${userId}@movetest.local`,
      password: password
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    sleep(0.3);
    
    const loginRes = http.post(`${baseUrl}${API_PREFIX}/auth/login`, JSON.stringify({
      username: userId,
      password: password
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (loginRes.status !== 200) {
      console.error(`VU ${__VU}: Login failed`);
      return;
    }
    
    myToken = JSON.parse(loginRes.body).token;
    
    sleep(0.5);
    
    // Create a game for move testing
    const createRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify({
      name: `Move Test Game ${__VU}`,
      boardType: 'square8', // Smaller board for faster games
      playerCount: 2,
      isPrivate: false,
      aiPlayers: 1 // 1 AI opponent for automated gameplay
    }), {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${myToken}`
      }
    });
    
    if (createRes.status === 201) {
      const gameData = JSON.parse(createRes.body);
      myGameId = gameData.id;
      
      // Find our player ID
      if (gameData.players && gameData.players.length > 0) {
        myPlayerId = gameData.players[0].id;
      }
      
      console.log(`VU ${__VU}: Created game ${myGameId}`);
    } else {
      console.error(`VU ${__VU}: Game creation failed - ${createRes.status}`);
      return;
    }
  }
  
  // Step 2: Submit moves through the game
  if (myGameId && myToken) {
    // Get current game state
    const stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
      headers: { 'Authorization': `Bearer ${myToken}` }
    });
    
    if (stateRes.status !== 200) {
      moveProcessingErrors.add(1);
      sleep(2);
      return;
    }
    
    const gameState = JSON.parse(stateRes.body);
    
    // Check if game is still active
    if (gameState.status !== 'active' && gameState.status !== 'waiting') {
      console.log(`VU ${__VU}: Game ${myGameId} ended with status ${gameState.status}`);
      // Reset to create a new game next iteration
      myGameId = null;
      sleep(5);
      return;
    }
    
    // Simulate move submission via HTTP endpoint
    // Note: Actual implementation may use WebSocket. Adjust based on API design.
    const movePayload = {
      gameId: myGameId,
      playerId: myPlayerId,
      action: generateRandomMove(gameState) // Helper function for valid moves
    };
    
    const moveStart = Date.now();
    const moveRes = http.post(`${baseUrl}${API_PREFIX}/games/${myGameId}/moves`, 
      JSON.stringify(movePayload), 
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${myToken}`
        },
        tags: { name: 'submit-move' }
      }
    );
    const moveLatency = Date.now() - moveStart;
    
    // Track metrics
    moveSubmissionLatency.add(moveLatency);
    
    const moveSuccess = check(moveRes, {
      'move accepted': (r) => r.status === 200 || r.status === 201,
      'move processed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.success || body.gameState;
        } catch {
          return false;
        }
      }
    });
    
    moveSubmissionSuccess.add(moveSuccess);
    
    if (moveSuccess) {
      turnProcessingLatency.add(moveLatency);
    } else {
      moveProcessingErrors.add(1);
    }
    
    // Track stalled moves (>2s per STRATEGIC_ROADMAP stall definition)
    if (moveLatency > 2000) {
      stalledMoves.add(1);
      console.warn(`VU ${__VU}: Stalled move detected - ${moveLatency}ms`);
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
  const moveTypes = ['PLACE_RING', 'MOVE_RING', 'PLACE_MARKER'];
  const moveType = moveTypes[Math.floor(Math.random() * moveTypes.length)];
  
  // Simplified move generation - actual implementation needs game rules
  return {
    type: moveType,
    position: {
      q: Math.floor(Math.random() * 8),
      r: Math.floor(Math.random() * 8)
    },
    // Add fields based on move type
    ...(moveType === 'MOVE_RING' && {
      from: {
        q: Math.floor(Math.random() * 8),
        r: Math.floor(Math.random() * 8)
      }
    })
  };
}

export function teardown(data) {
  console.log('Player move submission test complete');
  console.log('Key metrics to review:');
  console.log('  - move_submission_latency_ms (p95, p99)');
  console.log('  - stalled_moves_total (should be <0.5% of total moves)');
  console.log('  - move_submission_success_rate');
}