/**
 * RingRift Load Test: WebSocket Connection Stress Test
 * 
 * Tests WebSocket connection limits, stability, and reconnection behavior.
 * Validates production-scale assumptions for real-time connection handling.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3.3: Reconnects and Spectators
 * Alert thresholds from monitoring/prometheus/alerts.yml (>1000 connections)
 * 
 * NOTE: k6 WebSocket support is functional but may require k6 v0.46+
 * for advanced features. See k6.io/docs/using-k6/protocols/websockets
 * 
 * PROTOCOL: This scenario speaks Socket.IO v4 / Engine.IO v4 protocol:
 *   - Engine.IO packets: 0=open, 1=close, 2=ping, 3=pong, 4=message
 *   - Socket.IO packets: 0=CONNECT, 1=DISCONNECT, 2=EVENT, 3=ACK
 *   - Full Socket.IO message: "4" + socketio_type + JSON payload
 *   - Example EVENT: "42" + JSON.stringify(["eventName", data])
 */

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Gauge, Trend } from 'k6/metrics';
import { loginAndGetToken } from '../auth/helpers.js';
import { makeHandleSummary } from '../summary.js';

const thresholdsConfig = JSON.parse(open('../config/thresholds.json'));

// Classification metrics shared across load scenarios
export const contractFailures = new Counter('contract_failures_total');
export const idLifecycleMismatches = new Counter('id_lifecycle_mismatches_total');
export const capacityFailures = new Counter('capacity_failures_total');
const authTokenExpired = new Counter('auth_token_expired_total');
const rateLimitHit = new Counter('rate_limit_hit_total');
const trueErrors = new Counter('true_errors_total');

// Custom metrics
const wsConnections = new Gauge('websocket_connections_active');
const wsConnectionSuccess = new Rate('websocket_connection_success_rate');
const wsConnectionErrors = new Counter('websocket_connection_errors');
const wsReconnections = new Counter('websocket_reconnections_total');
const wsMessageLatency = new Trend('websocket_message_latency_ms');
const wsConnectionDuration = new Trend('websocket_connection_duration_ms');
const wsHandshakeSuccess = new Rate('websocket_handshake_success_rate');
const wsProtocolErrors = new Counter('websocket_protocol_errors');

 // Test configuration - stress test for connection limits
export const options = {
  scenarios: {
    websocket_stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },   // Ramp to 100 connections
        { duration: '2m', target: 300 },   // Ramp to 300 connections
        { duration: '3m', target: 500 },   // Reach 500+ connections
        { duration: '5m', target: 500 },   // Sustain high connection count
        { duration: '2m', target: 100 },   // Gradual ramp down
        { duration: '1m', target: 0 }      // Complete shutdown
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Connection success rate - should remain high even at scale. Use the
    // environment-specific target from thresholds.json.
    'websocket_connection_success_rate': [
      `rate>${(
        (thresholdsConfig.environments[__ENV.THRESHOLD_ENV || 'staging'] ||
          thresholdsConfig.environments.staging
        ).websocket_gameplay.connection_stability.connection_success_rate_percent / 100
      ).toFixed(4)}`,
    ],
    
    // Handshake success rate (Socket.IO protocol level). There is no explicit
    // SLO for this in thresholds.json yet, so we keep the existing constant.
    'websocket_handshake_success_rate': ['rate>0.98'],
    
    // Connection errors should be minimal; derive maximum counts from the
    // load_tests.websocket subsection for the current environment.
    'websocket_connection_errors': [
      `count<=${(
        thresholdsConfig.load_tests[__ENV.THRESHOLD_ENV || 'staging'] ||
        thresholdsConfig.load_tests.staging
      ).websocket.connection_errors_max}`,
    ],
    
    // Protocol errors (message parse failures) should be zero or extremely rare.
    'websocket_protocol_errors': [
      `count<=${(
        thresholdsConfig.load_tests[__ENV.THRESHOLD_ENV || 'staging'] ||
        thresholdsConfig.load_tests.staging
      ).websocket.protocol_errors_max}`,
    ],
    
    // Message latency - real-time feel (kept as explicit transport SLOs).
    'websocket_message_latency_ms': [
      'p(95)<200',  // Most messages arrive quickly
      'p(99)<500'   // Even slow messages acceptable
    ],
    
    // Connection stability - should maintain for 5+ minutes per STRATEGIC_ROADMAP
    'websocket_connection_duration_ms': ['p(50)>300000'], // Median >5 minutes

    // Classification counters.
    contract_failures_total: [
      `count<=${(
        thresholdsConfig.load_tests[__ENV.THRESHOLD_ENV || 'staging'] ||
        thresholdsConfig.load_tests.staging
      ).contract_failures_total.max}`,
    ],
    id_lifecycle_mismatches_total: [
      `count<=${(
        thresholdsConfig.load_tests[__ENV.THRESHOLD_ENV || 'staging'] ||
        thresholdsConfig.load_tests.staging
      ).id_lifecycle_mismatches_total.max}`,
    ],
    capacity_failures_total: [
      `rate<${(
        thresholdsConfig.load_tests[__ENV.THRESHOLD_ENV || 'staging'] ||
        thresholdsConfig.load_tests.staging
      ).capacity_failures_total.rate}`,
    ],
    true_errors_total: [`rate<${trueErrorRateTarget}`],
  },
  
  tags: {
    scenario: 'websocket-stress',
    test_type: 'stress',
    environment: __ENV.THRESHOLD_ENV || 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
/**
 * WebSocket origin used for Socket.IO connections.
 *
 * This may be provided as either:
 *   - WS_URL=ws://host:port
 *   - WS_URL=http://host:port
 *
 * When not set, it falls back to BASE_URL so local runs behave like the
 * browser client, which derives its socket origin from VITE_WS_URL /
 * VITE_API_URL (see src/client/utils/socketBaseUrl.ts and LobbyPage).
 */
const WS_BASE = __ENV.WS_URL || BASE_URL;
const API_PREFIX = '/api';

// Target per-connection session duration. Defaults to 6 minutes, which
// comfortably covers the 5-minute sustain period in the scenario stages.
const TARGET_SESSION_DURATION_MS = Number(__ENV.TARGET_SESSION_DURATION_MS || 360000);

// Interval between diagnostic ping events used for latency measurement.
const DIAGNOSTIC_PING_INTERVAL_MS = Number(__ENV.DIAGNOSTIC_PING_INTERVAL_MS || 5000);

// Threshold configuration derived from thresholds.json
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const perfEnv =
  thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const loadTestEnv =
  thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;
const connectionStability = perfEnv.websocket_gameplay.connection_stability;
const websocketLoad = loadTestEnv.websocket;
const trueErrorRateTarget =
  loadTestEnv &&
  loadTestEnv.true_errors &&
  typeof loadTestEnv.true_errors.rate === 'number'
    ? loadTestEnv.true_errors.rate
    : 0.005;

/**
 * Engine.IO packet types (prefix character)
 */
const EIO_OPEN = '0';
const EIO_CLOSE = '1';
const EIO_PING = '2';
const EIO_PONG = '3';
const EIO_MESSAGE = '4';

/**
 * Socket.IO packet types (follows Engine.IO MESSAGE prefix)
 */
const SIO_CONNECT = '0';
const SIO_DISCONNECT = '1';
const SIO_EVENT = '2';
const SIO_ACK = '3';

/**
 * Parse an Engine.IO/Socket.IO message and return structured data.
 * 
 * @param {string} raw - Raw WebSocket message
 * @returns {{ eioType: string, sioType?: string, data?: any, error?: string }}
 */
function parseSocketIOMessage(raw) {
  if (!raw || raw.length === 0) {
    return { error: 'empty message' };
  }
  
  const eioType = raw[0];
  
  // Engine.IO level packets
  if (eioType === EIO_OPEN) {
    // Open packet: "0{...json...}"
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
    // Ping packet: "2" or "2probe"
    return { eioType, data: raw.slice(1) || null };
  }
  
  if (eioType === EIO_PONG) {
    return { eioType, data: raw.slice(1) || null };
  }
  
  if (eioType === EIO_MESSAGE) {
    // Socket.IO message: "4" + sioType + payload
    if (raw.length < 2) {
      return { eioType, error: 'truncated message' };
    }
    
    const sioType = raw[1];
    const payload = raw.slice(2);
    
    if (sioType === SIO_CONNECT) {
      // Connect ACK: "40" or "40{...}"
      try {
        const data = payload ? JSON.parse(payload) : {};
        return { eioType, sioType, data };
      } catch (e) {
        return { eioType, sioType, data: payload };
      }
    }
    
    if (sioType === SIO_EVENT) {
      // Event: "42["eventName", data]"
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
      // ACK: "43" + ackId + JSON payload
      return { eioType, sioType, data: payload };
    }
    
    if (sioType === SIO_DISCONNECT) {
      return { eioType, sioType };
    }
    
    return { eioType, sioType, data: payload };
  }
  
  return { error: `unknown packet type: ${eioType}` };
}

/**
 * Build a Socket.IO EVENT message.
 * 
 * @param {string} eventName - Event name
 * @param {any} data - Event data
 * @returns {string} - Framed Socket.IO message
 */
function buildSocketIOEvent(eventName, data) {
  return EIO_MESSAGE + SIO_EVENT + JSON.stringify([eventName, data]);
}

/**
 * Build a Socket.IO CONNECT message for the default namespace.
 * 
 * @param {Object} [auth] - Optional auth payload
 * @returns {string} - Framed Socket.IO CONNECT message
 */
function buildSocketIOConnect(auth) {
  if (auth) {
    return EIO_MESSAGE + SIO_CONNECT + JSON.stringify({ auth });
  }
  return EIO_MESSAGE + SIO_CONNECT;
}

/**
 * Build an Engine.IO PONG response.
 * 
 * @param {string} [probe] - Optional probe data to echo
 * @returns {string} - Engine.IO PONG packet
 */
function buildEnginePong(probe) {
  return EIO_PONG + (probe || '');
}

/**
 * Build a Socket.IO WebSocket endpoint that matches the production
 * WebSocketServer contract:
 *
 *   - Path: /socket.io/
 *   - Query params:
 *       EIO=4&transport=websocket
 *       token=<JWT> (for auth middleware)
 *       vu=<VU id> (diagnostic only)
 *
 * The server expects proper Engine.IO/Socket.IO framed messages after
 * the WebSocket handshake completes.
 */
function buildSocketIoEndpoint(wsBase, token, vu) {
  let origin = wsBase || '';
  if (origin.startsWith('http://') || origin.startsWith('https://')) {
    origin = origin.replace(/^http/, 'ws');
  }
  origin = origin.replace(/\/$/, '');

  const path = '/socket.io/';
  const params = [];
  // Socket.IO v4 engine and transport selector
  params.push('EIO=4', 'transport=websocket', `vu=${vu}`);
  // WebSocketServer authentication middleware accepts the JWT via
  // either handshake.auth.token (used by the browser client) or
  // handshake.query.token. We use the query-string form here.
  if (token) {
    params.push(`token=${encodeURIComponent(token)}`);
  }

  const query = params.join('&');
  return `${origin}${path}?${query}`;
}

export function setup() {
  console.log('Starting WebSocket connection stress test');
  console.log('Target: 500+ simultaneous WebSocket connections');
  console.log('Duration: 5+ minute sustained connection period');
  console.log('Protocol: Socket.IO v4 / Engine.IO v4');

  // Lightweight health check to match other scenarios and fail fast if
  // the API is not reachable.
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  if (healthCheck.status !== 200) {
    // Treat failed health checks as capacity/infra issues rather than
    // contract failures for this transport-focused scenario.
    capacityFailures.add(1);
  }

  // Use the shared auth helper so the WebSocket stress test reuses the
  // canonical /api/auth/login contract with { email, password } and
  // the same pre-seeded load-test user.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
    metrics: {
      contractFailures,
      capacityFailures,
    },
  });

  return { wsBase: WS_BASE, baseUrl: BASE_URL, token, userId };
}

export default function(data) {
  const wsBase = data.wsBase;
  const baseUrl = data.baseUrl;
  const token = data.token;

  // Each VU maintains a WebSocket connection
  const connectionStart = Date.now();
  let messagesSent = 0;
  let messagesReceived = 0;
  let reconnectAttempts = 0;
  let handshakeComplete = false;
  let serverPingInterval = null;
  let lastDiagnosticPingAt = 0;
  let nextDiagnosticSequence = 0;

  // WebSocket connection using proper Socket.IO/Engine.IO protocol
  const wsEndpoint = buildSocketIoEndpoint(wsBase, token, __VU);

  const res = ws.connect(
    wsEndpoint,
    {
      headers: {
        'User-Agent': 'k6-load-test',
      },
      tags: { vu: __VU.toString() },
    },
    function (socket) {
    
    // Track active connections
    wsConnections.add(1);
    
    socket.on('open', () => {
      console.log(`VU ${__VU}: WebSocket transport connected, awaiting Engine.IO handshake`);
      wsConnectionSuccess.add(1);
      // Don't send anything yet - wait for server's Engine.IO OPEN packet
    });
    
    socket.on('message', (message) => {
      messagesReceived++;
      
      const parsed = parseSocketIOMessage(message);
      
      if (parsed.error) {
        wsProtocolErrors.add(1);
        // Protocol framing issues are treated as contract-level failures
        // between the harness and the WebSocket server.
        contractFailures.add(1);
        console.warn(`VU ${__VU}: Protocol error - ${parsed.error}`);
        return;
      }
      
      // Handle Engine.IO OPEN - server sends session info
      if (parsed.eioType === EIO_OPEN) {
        console.log(`VU ${__VU}: Engine.IO OPEN received, sid=${parsed.data?.sid}`);
        
        // Store server's ping interval for reference
        if (parsed.data && parsed.data.pingInterval) {
          serverPingInterval = parsed.data.pingInterval;
        }
        
        // Now send Socket.IO CONNECT to join the default namespace
        // The auth token is already in the query string, but we can also
        // send it in the CONNECT payload as a belt-and-suspenders approach
        const connectMsg = buildSocketIOConnect({ token });
        socket.send(connectMsg);
        messagesSent++;
        console.log(`VU ${__VU}: Socket.IO CONNECT sent`);
        return;
      }
      
      // Handle Engine.IO PING - respond with PONG
      if (parsed.eioType === EIO_PING) {
        const pong = buildEnginePong(parsed.data);
        socket.send(pong);
        messagesSent++;
        return;
      }
      
      // Handle Engine.IO CLOSE
      if (parsed.eioType === EIO_CLOSE) {
        console.log(`VU ${__VU}: Engine.IO CLOSE received`);
        return;
      }
      
      // Handle Socket.IO CONNECT ACK - handshake complete
      if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_CONNECT) {
        handshakeComplete = true;
        wsHandshakeSuccess.add(1);
        console.log(`VU ${__VU}: Socket.IO handshake complete`);
        
        // Now we can send application-level events
        // Subscribe to lobby updates as a realistic spectator-like action
        const subscribeMsg = buildSocketIOEvent('lobby:subscribe', {
          timestamp: Date.now(),
          vu: __VU
        });
        socket.send(subscribeMsg);
        messagesSent++;
        return;
      }
      
      // Handle Socket.IO EVENTs
      if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_EVENT) {
        const eventName = parsed.event;
        const eventArgs = parsed.data || [];
        const payload = eventArgs[0];

        // Measure round-trip latency for diagnostic ping/pong.
        if (eventName === 'diagnostic:pong' && payload && typeof payload.timestamp === 'number') {
          const rtt = Date.now() - payload.timestamp;
          wsMessageLatency.add(rtt);
          return;
        }
        
        // Log interesting events for basic observability
        if (
          eventName === 'lobby:game_created' ||
          eventName === 'lobby:game_started' ||
          eventName === 'error'
        ) {
          console.log(`VU ${__VU}: Received ${eventName}`);
        }

        // Map structured WebSocket error payloads into classification counters.
        if (eventName === 'error' && payload && typeof payload === 'object') {
          classifyWebSocketErrorPayload(payload);
        }
        return;
      }
      
      // Handle Socket.IO DISCONNECT
      if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_DISCONNECT) {
        console.log(`VU ${__VU}: Socket.IO DISCONNECT received`);
        return;
      }
    });
    
    socket.on('close', (code) => {
      const duration = Date.now() - connectionStart;
      wsConnectionDuration.add(duration);
      wsConnections.add(-1);
      
      console.log(`VU ${__VU}: WebSocket closed after ${duration}ms (code: ${code})`);
      
      // Record handshake failure if we never completed it
      if (!handshakeComplete) {
        wsHandshakeSuccess.add(0);
      }
      
      // Attempt reconnection if unexpected close
      if (code !== 1000 && reconnectAttempts < 3) {
        wsReconnections.add(1);
        reconnectAttempts++;
        console.log(`VU ${__VU}: Attempting reconnection (${reconnectAttempts}/3)`);
      }
    });
    
    socket.on('error', (e) => {
      wsConnectionErrors.add(1);
      // Transport-level errors are treated as capacity / infrastructure issues
      // rather than contract failures.
      capacityFailures.add(1);
      console.error(`VU ${__VU}: WebSocket error - ${e}`);
    });
    
    // Keep connection alive and periodically send realistic events
    // This simulates spectator activity or game state updates and
    // exercises the dedicated diagnostic ping/pong channel used by
    // this scenario to measure WebSocket latency.
    let tickCount = 0;
    socket.setInterval(() => {
      tickCount++;
      
      // Only send application events after handshake is complete
      if (!handshakeComplete) {
        return;
      }

      const now = Date.now();

      // Emit diagnostic ping at a configurable interval. The backend
      // echoes this as diagnostic:pong with the same payload plus a
      // serverTimestamp, which we use to compute round-trip latency.
      if (now - lastDiagnosticPingAt >= DIAGNOSTIC_PING_INTERVAL_MS) {
        const pingPayload = {
          timestamp: now,
          vu: __VU,
          sequence: nextDiagnosticSequence++,
        };
        const pingMsg = buildSocketIOEvent('diagnostic:ping', pingPayload);
        socket.send(pingMsg);
        messagesSent++;
        lastDiagnosticPingAt = now;
      }
      
      // Send periodic lobby subscription refresh every ~60 seconds
      // (This is a lightweight no-op if already subscribed, but exercises
      // the message path)
      if (tickCount % 60 === 0) {
        const refreshMsg = buildSocketIOEvent('lobby:subscribe', {
          timestamp: now,
          vu: __VU,
          refresh: true
        });
        socket.send(refreshMsg);
        messagesSent++;
      }
      
      // Occasionally request game list as a realistic spectator action
      if (tickCount % 30 === 0 && Math.random() > 0.7) {
        const listMsg = buildSocketIOEvent('lobby:list_games', {
          timestamp: now,
          vu: __VU
        });
        socket.send(listMsg);
        messagesSent++;
      }
      
    }, 1000); // Check every second
    
    // Maintain connection for test duration
    // Connection will close when VU iteration ends
    socket.setTimeout(() => {
      const duration = Date.now() - connectionStart;
      console.log(
        `VU ${__VU}: Connection timeout after ${duration}ms. ` +
          `Sent: ${messagesSent}, Received: ${messagesReceived}, Handshake: ${handshakeComplete}`
      );
      socket.close();
    }, TARGET_SESSION_DURATION_MS);
  });
  
  // Check connection establishment
  check(res, {
    'WebSocket connected': (r) => r && r.status === 101
  });
  
  if (!res || res.status !== 101) {
    wsConnectionErrors.add(1);
    capacityFailures.add(1);
    console.error(`VU ${__VU}: Failed to establish WebSocket connection - status ${res?.status}`);
  }
  
  // After WebSocket closes, brief pause before next iteration
  sleep(2 + Math.random() * 3);
}

function classifyWebSocketErrorPayload(payload) {
  const code = payload && payload.code;
  if (!code || typeof code !== 'string') {
    return;
  }

  switch (code) {
    case 'ACCESS_DENIED':
      authTokenExpired.add(1);
      break;
    case 'INVALID_PAYLOAD':
      // Misuse of the WebSocket/Socket.IO protocol or invalid auth token.
      contractFailures.add(1);
      trueErrors.add(1);
      break;
    case 'RATE_LIMITED':
      rateLimitHit.add(1);
      capacityFailures.add(1);
      break;
    case 'INTERNAL_ERROR':
      // Capacity or server-side failures.
      capacityFailures.add(1);
      trueErrors.add(1);
      break;
    default:
      // Other codes (GAME_NOT_FOUND, MOVE_REJECTED, CHOICE_REJECTED,
      // DECISION_PHASE_TIMEOUT, etc.) are behavioural/game-level concerns and
      // are better covered by dedicated game-session load tests rather than
      // this pure transport scenario.
      trueErrors.add(1);
      break;
  }
}

export function teardown(data) {
  console.log('WebSocket stress test complete');
  console.log('Key metrics to review:');
  console.log('  - Peak websocket_connections_active (should reach 500+)');
  console.log('  - websocket_connection_success_rate');
  console.log('  - websocket_handshake_success_rate (Socket.IO level)');
  console.log('  - websocket_protocol_errors (should be near zero)');
  console.log('  - websocket_connection_duration_ms (p50 should be >5 minutes)');
  console.log('  - websocket_reconnections_total');
  console.log('Check Prometheus for:');
  console.log('  - ringrift_websocket_connections gauge');
  console.log('  - High connection alert triggers (>1000 threshold)');
}

export const handleSummary = makeHandleSummary('websocket-stress');
