/**
 * RingRift Load Test - API Client Helpers
 *
 * Provides reusable HTTP API client functions for k6 load test scenarios.
 * All functions align with the canonical API contract documented in
 * docs/architecture/API_REFERENCE.md.
 *
 * Usage:
 *   import { createGame, getGameState, makeMove } from '../helpers/api.js';
 */

import http from 'k6/http';
import { check } from 'k6';

// Default configuration
const DEFAULT_BASE_URL = 'http://localhost:3001';
const DEFAULT_API_PREFIX = '/api';

/**
 * Build standard request headers with optional auth token.
 *
 * @param {string} [token] - JWT access token
 * @returns {Object} Headers object
 */
export function buildHeaders(token) {
  const headers = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}

/**
 * Get the base URL from environment or default.
 *
 * @returns {string} Base URL
 */
export function getBaseUrl() {
  return __ENV.BASE_URL || DEFAULT_BASE_URL;
}

/**
 * Get the API prefix from environment or default.
 *
 * @returns {string} API prefix
 */
export function getApiPrefix() {
  return __ENV.API_PREFIX || DEFAULT_API_PREFIX;
}

/**
 * Build a full API endpoint URL.
 *
 * @param {string} path - API path (e.g., '/games' or '/games/:id')
 * @param {string} [baseUrl] - Base URL override
 * @returns {string} Full URL
 */
export function buildUrl(path, baseUrl) {
  const base = baseUrl || getBaseUrl();
  const prefix = getApiPrefix();
  return `${base}${prefix}${path}`;
}

/**
 * Parse a standard API response envelope: { success, data, error }.
 *
 * @param {Object} res - k6 HTTP response
 * @returns {{ success: boolean, data: any, error: any, raw: any }}
 */
export function parseResponse(res) {
  try {
    const raw = JSON.parse(res.body);
    return {
      success: raw && raw.success === true,
      data: raw && raw.data ? raw.data : null,
      error: raw && raw.error ? raw.error : null,
      raw,
    };
  } catch (e) {
    return {
      success: false,
      data: null,
      error: { message: 'Failed to parse response', parseError: e.message },
      raw: null,
    };
  }
}

/**
 * Create a new game.
 *
 * @param {string} token - JWT access token
 * @param {Object} options - Game creation options
 * @param {string} [options.boardType='square8'] - Board type
 * @param {number} [options.maxPlayers=2] - Max players
 * @param {boolean} [options.isPrivate=false] - Private game flag
 * @param {Object} [options.timeControl] - Time control settings
 * @param {boolean} [options.isRated=false] - Rated game flag
 * @param {Object} [options.aiOpponents] - AI opponent configuration
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, game: Object|null, success: boolean }}
 */
export function createGame(token, options = {}) {
  const boardType = options.boardType || 'square8';
  const maxPlayers = options.maxPlayers || 2;
  const isPrivate = options.isPrivate || false;
  const isRated = options.isRated !== undefined ? options.isRated : false;

  const timeControl = options.timeControl || {
    type: 'rapid',
    initialTime: 600,
    increment: 0,
  };

  const payload = {
    boardType,
    maxPlayers,
    isPrivate,
    timeControl,
    isRated,
  };

  // Add AI opponents if specified
  if (options.aiOpponents) {
    payload.aiOpponents = options.aiOpponents;
    // AI games must be unrated per backend contract
    payload.isRated = false;
  }

  const res = http.post(
    buildUrl('/games'),
    JSON.stringify(payload),
    {
      headers: buildHeaders(token),
      tags: options.tags || { name: 'create-game' },
    }
  );

  const parsed = parseResponse(res);
  const game = parsed.data && parsed.data.game ? parsed.data.game : null;

  return {
    res,
    game,
    success: res.status === 201 && game !== null,
  };
}

/**
 * Get game state by ID.
 *
 * @param {string} token - JWT access token
 * @param {string} gameId - Game ID
 * @param {Object} [options] - Request options
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, game: Object|null, success: boolean }}
 */
export function getGameState(token, gameId, options = {}) {
  const res = http.get(
    buildUrl(`/games/${gameId}`),
    {
      headers: buildHeaders(token),
      tags: options.tags || { name: 'get-game' },
    }
  );

  const parsed = parseResponse(res);
  const game = parsed.data && parsed.data.game ? parsed.data.game : null;

  return {
    res,
    game,
    success: res.status === 200 && game !== null,
  };
}

/**
 * Submit a move via HTTP (requires backend ENABLE_HTTP_MOVE_HARNESS).
 *
 * @param {string} token - JWT access token
 * @param {string} gameId - Game ID
 * @param {Object} move - Move payload
 * @param {Object} [options] - Request options
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, success: boolean, latency: number }}
 */
export function makeMove(token, gameId, move, options = {}) {
  const startTime = Date.now();

  const res = http.post(
    buildUrl(`/games/${gameId}/moves`),
    JSON.stringify(move),
    {
      headers: buildHeaders(token),
      tags: options.tags || { name: 'submit-move' },
    }
  );

  const latency = Date.now() - startTime;

  return {
    res,
    success: res.status >= 200 && res.status < 300,
    latency,
  };
}

/**
 * List available games.
 *
 * @param {string} token - JWT access token
 * @param {Object} [options] - Request options
 * @param {string} [options.status] - Filter by status
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, games: Array, success: boolean }}
 */
export function listGames(token, options = {}) {
  let url = buildUrl('/games');
  if (options.status) {
    url += `?status=${encodeURIComponent(options.status)}`;
  }

  const res = http.get(url, {
    headers: buildHeaders(token),
    tags: options.tags || { name: 'list-games' },
  });

  const parsed = parseResponse(res);
  const games = parsed.data && Array.isArray(parsed.data.games)
    ? parsed.data.games
    : [];

  return {
    res,
    games,
    success: res.status === 200,
  };
}

/**
 * Join an existing game.
 *
 * @param {string} token - JWT access token
 * @param {string} gameId - Game ID to join
 * @param {Object} [options] - Request options
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, game: Object|null, success: boolean }}
 */
export function joinGame(token, gameId, options = {}) {
  const res = http.post(
    buildUrl(`/games/${gameId}/join`),
    null,
    {
      headers: buildHeaders(token),
      tags: options.tags || { name: 'join-game' },
    }
  );

  const parsed = parseResponse(res);
  const game = parsed.data && parsed.data.game ? parsed.data.game : null;

  return {
    res,
    game,
    success: res.status === 200 && game !== null,
  };
}

/**
 * Leave a game.
 *
 * @param {string} token - JWT access token
 * @param {string} gameId - Game ID
 * @param {Object} [options] - Request options
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, success: boolean }}
 */
export function leaveGame(token, gameId, options = {}) {
  const res = http.post(
    buildUrl(`/games/${gameId}/leave`),
    null,
    {
      headers: buildHeaders(token),
      tags: options.tags || { name: 'leave-game' },
    }
  );

  return {
    res,
    success: res.status === 200,
  };
}

/**
 * Perform a health check.
 *
 * @param {Object} [options] - Request options
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, healthy: boolean }}
 */
export function healthCheck(options = {}) {
  const res = http.get(
    `${getBaseUrl()}/health`,
    {
      tags: options.tags || { name: 'health-check' },
    }
  );

  return {
    res,
    healthy: res.status === 200,
  };
}

/**
 * Create a game with AI opponents for load testing.
 *
 * @param {string} token - JWT access token
 * @param {Object} [options] - Creation options
 * @param {string} [options.boardType='square8'] - Board type
 * @param {number} [options.aiCount=1] - Number of AI opponents
 * @param {number} [options.aiDifficulty=5] - AI difficulty level (1-10)
 * @param {Object} [options.tags] - k6 request tags
 * @returns {{ res: Object, game: Object|null, success: boolean }}
 */
export function createAIGame(token, options = {}) {
  const aiCount = options.aiCount || 1;
  const aiDifficulty = options.aiDifficulty || 5;
  const boardType = options.boardType || 'square8';

  return createGame(token, {
    boardType,
    maxPlayers: 2,
    isPrivate: false,
    isRated: false,
    aiOpponents: {
      count: aiCount,
      difficulty: Array(aiCount).fill(aiDifficulty),
      mode: 'service',
      aiType: 'heuristic',
    },
    tags: options.tags,
  });
}