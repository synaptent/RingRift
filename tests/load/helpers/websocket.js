/**
 * RingRift Load Test - WebSocket/Socket.IO Helpers
 *
 * Provides reusable WebSocket and Socket.IO v4 protocol helpers for k6 load tests.
 * Implements the Engine.IO/Socket.IO v4 framing used by the production WebSocket server.
 *
 * Usage:
 *   import { connectGame, buildSocketIOEvent, parseSocketIOMessage } from '../helpers/websocket.js';
 */

import ws from 'k6/ws';

// ─────────────────────────────────────────────────────────────────────────────
// Engine.IO / Socket.IO Protocol Constants
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Engine.IO packet types (prefix character).
 */
export const EIO_OPEN = '0';
export const EIO_CLOSE = '1';
export const EIO_PING = '2';
export const EIO_PONG = '3';
export const EIO_MESSAGE = '4';

/**
 * Socket.IO packet types (follows Engine.IO MESSAGE prefix).
 */
export const SIO_CONNECT = '0';
export const SIO_DISCONNECT = '1';
export const SIO_EVENT = '2';
export const SIO_ACK = '3';

// ─────────────────────────────────────────────────────────────────────────────
// Message Parsing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parse an Engine.IO/Socket.IO message and return structured data.
 *
 * @param {string} raw - Raw WebSocket message
 * @returns {{
 *   eioType?: string,
 *   sioType?: string,
 *   event?: string,
 *   data?: any,
 *   error?: string
 * }}
 */
export function parseSocketIOMessage(raw) {
  if (!raw || raw.length === 0) {
    return { error: 'empty message' };
  }

  const eioType = raw[0];

  // Engine.IO level packets
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

  return { error: `unknown packet type: ${eioType}` };
}

// ─────────────────────────────────────────────────────────────────────────────
// Message Building
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a Socket.IO EVENT message.
 *
 * @param {string} eventName - Event name
 * @param {any} data - Event data
 * @returns {string} - Framed Socket.IO message
 */
export function buildSocketIOEvent(eventName, data) {
  return EIO_MESSAGE + SIO_EVENT + JSON.stringify([eventName, data]);
}

/**
 * Build a Socket.IO CONNECT message for the default namespace.
 *
 * @param {Object} [auth] - Optional auth payload
 * @returns {string} - Framed Socket.IO CONNECT message
 */
export function buildSocketIOConnect(auth) {
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
export function buildEnginePong(probe) {
  return EIO_PONG + (probe || '');
}

/**
 * Build an Engine.IO PING packet (for keepalive).
 *
 * @param {string} [probe] - Optional probe data
 * @returns {string} - Engine.IO PING packet
 */
export function buildEnginePing(probe) {
  return EIO_PING + (probe || '');
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get the WebSocket base URL from environment or derive from HTTP base.
 *
 * @param {string} [httpBaseUrl] - HTTP base URL to derive from
 * @returns {string} WebSocket base URL
 */
export function getWsBaseUrl(httpBaseUrl) {
  if (__ENV.WS_URL) {
    return __ENV.WS_URL;
  }

  const httpBase = httpBaseUrl || __ENV.BASE_URL || 'http://localhost:3001';
  return httpBase.replace(/^http/, 'ws');
}

/**
 * Build a Socket.IO WebSocket endpoint URL.
 *
 * Constructs the full connection URL with proper Socket.IO v4 query parameters:
 *   - Path: /socket.io/
 *   - Query: EIO=4&transport=websocket&token=<JWT>
 *
 * @param {Object} options - Connection options
 * @param {string} [options.wsBase] - WebSocket base URL
 * @param {string} [options.token] - JWT access token for authentication
 * @param {number|string} [options.vu] - Virtual user ID for debugging
 * @param {Object} [options.extraParams] - Additional query parameters
 * @returns {string} Full Socket.IO WebSocket URL
 */
export function buildSocketIoEndpoint(options = {}) {
  let origin = options.wsBase || getWsBaseUrl();

  // Normalize protocol
  if (origin.startsWith('http://') || origin.startsWith('https://')) {
    origin = origin.replace(/^http/, 'ws');
  }
  origin = origin.replace(/\/$/, '');

  const path = '/socket.io/';
  const params = ['EIO=4', 'transport=websocket'];

  if (options.vu !== undefined) {
    params.push(`vu=${options.vu}`);
  }

  if (options.token) {
    params.push(`token=${encodeURIComponent(options.token)}`);
  }

  // Add any extra parameters
  if (options.extraParams && typeof options.extraParams === 'object') {
    for (const [key, value] of Object.entries(options.extraParams)) {
      params.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    }
  }

  return `${origin}${path}?${params.join('&')}`;
}

/**
 * Connect to a game via Socket.IO WebSocket.
 *
 * @param {Object} options - Connection options
 * @param {string} options.token - JWT access token
 * @param {string} options.gameId - Game ID to join
 * @param {string} [options.wsBase] - WebSocket base URL
 * @param {number} [options.vu] - Virtual user ID
 * @param {Function} options.onOpen - Called when connection opens
 * @param {Function} options.onMessage - Called for each message
 * @param {Function} options.onClose - Called when connection closes
 * @param {Function} options.onError - Called on error
 * @param {number} [options.timeout=300000] - Connection timeout in ms
 * @returns {Object} k6 ws.connect result
 */
export function connectGame(options) {
  const wsEndpoint = buildSocketIoEndpoint({
    wsBase: options.wsBase,
    token: options.token,
    vu: options.vu,
  });

  const timeout = options.timeout || 300000; // 5 minutes default

  return ws.connect(
    wsEndpoint,
    {
      headers: {
        'User-Agent': 'k6-load-test',
      },
      tags: {
        gameId: options.gameId,
        ...(options.tags || {}),
      },
    },
    function (socket) {
      // Track connection state
      let handshakeComplete = false;

      socket.on('open', () => {
        if (options.onOpen) {
          options.onOpen(socket);
        }
      });

      socket.on('message', (message) => {
        const parsed = parseSocketIOMessage(message);

        // Handle Engine.IO OPEN - send Socket.IO CONNECT
        if (parsed.eioType === EIO_OPEN) {
          const connectMsg = buildSocketIOConnect({ token: options.token });
          socket.send(connectMsg);
          return;
        }

        // Handle Engine.IO PING - respond with PONG
        if (parsed.eioType === EIO_PING) {
          socket.send(buildEnginePong(parsed.data));
          return;
        }

        // Handle Socket.IO CONNECT ACK
        if (parsed.eioType === EIO_MESSAGE && parsed.sioType === SIO_CONNECT) {
          handshakeComplete = true;

          // Auto-join game if gameId was provided
          if (options.gameId) {
            const joinMsg = buildSocketIOEvent('join_game', { gameId: options.gameId });
            socket.send(joinMsg);
          }
        }

        // Pass to user handler
        if (options.onMessage) {
          options.onMessage(parsed, socket, { handshakeComplete });
        }
      });

      socket.on('close', (code) => {
        if (options.onClose) {
          options.onClose(code, { handshakeComplete });
        }
      });

      socket.on('error', (e) => {
        if (options.onError) {
          options.onError(e, socket);
        }
      });

      // Set connection timeout
      socket.setTimeout(() => {
        socket.close();
      }, timeout);
    }
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Game Event Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a join_game event message.
 *
 * @param {string} gameId - Game ID to join
 * @returns {string} Framed Socket.IO message
 */
export function buildJoinGame(gameId) {
  return buildSocketIOEvent('join_game', { gameId });
}

/**
 * Build a player_move_by_id event message.
 *
 * @param {string} gameId - Game ID
 * @param {string} moveId - Move ID from validMoves
 * @returns {string} Framed Socket.IO message
 */
export function buildPlayerMoveById(gameId, moveId) {
  return buildSocketIOEvent('player_move_by_id', { gameId, moveId });
}

/**
 * Build a player_move event message (full move payload).
 *
 * @param {string} gameId - Game ID
 * @param {Object} move - Move payload
 * @returns {string} Framed Socket.IO message
 */
export function buildPlayerMove(gameId, move) {
  return buildSocketIOEvent('player_move', { gameId, move });
}

/**
 * Build a lobby:subscribe event message.
 *
 * @param {Object} [data] - Optional subscription data
 * @returns {string} Framed Socket.IO message
 */
export function buildLobbySubscribe(data = {}) {
  return buildSocketIOEvent('lobby:subscribe', {
    timestamp: Date.now(),
    ...data,
  });
}

/**
 * Build a diagnostic:ping event message for latency measurement.
 *
 * @param {Object} [data] - Optional ping data
 * @returns {string} Framed Socket.IO message
 */
export function buildDiagnosticPing(data = {}) {
  return buildSocketIOEvent('diagnostic:ping', {
    timestamp: Date.now(),
    ...data,
  });
}

/**
 * Check if a parsed message is a specific Socket.IO event.
 *
 * @param {Object} parsed - Parsed Socket.IO message
 * @param {string} eventName - Expected event name
 * @returns {boolean}
 */
export function isEvent(parsed, eventName) {
  return (
    parsed &&
    parsed.eioType === EIO_MESSAGE &&
    parsed.sioType === SIO_EVENT &&
    parsed.event === eventName
  );
}

/**
 * Check if a game state indicates a terminal status.
 *
 * @param {Object} gameState - Game state object
 * @returns {boolean}
 */
export function isTerminalGameState(gameState) {
  if (!gameState) return false;
  const status = gameState.gameStatus || gameState.status;
  return ['completed', 'abandoned', 'finished'].includes(status);
}

/**
 * Extract the first event argument from a parsed message.
 *
 * @param {Object} parsed - Parsed Socket.IO message
 * @returns {any} First event argument or null
 */
export function getEventPayload(parsed) {
  if (!parsed || !Array.isArray(parsed.data)) {
    return null;
  }
  return parsed.data[0] || null;
}