/**
 * Game Domain Errors - Structured error types for the game domain
 *
 * This module provides consistent error types for game-related errors across
 * all layers (HTTP routes, WebSocket handlers, GameSession, engine).
 *
 * Error Categories:
 * - **Game State Errors**: Invalid game states, game not found, etc.
 * - **Move Errors**: Invalid moves, validation failures
 * - **AI Errors**: Service unavailable, timeout, fallback failures
 * - **Decision Errors**: Choice timeouts, invalid choices
 * - **Player Errors**: Not your turn, unauthorized access
 *
 * Usage:
 * ```typescript
 * import { GameError, GameErrorCode, InvalidMoveError } from './GameDomainErrors';
 *
 * // Throw a specific error
 * throw new InvalidMoveError('Cannot place ring on occupied space', {
 *   moveType: 'place_ring',
 *   position: { x: 3, y: 3 },
 * });
 *
 * // Check error type
 * if (error instanceof GameError) {
 *   console.log(error.code, error.context);
 * }
 * ```
 *
 * @module GameDomainErrors
 */

// ═══════════════════════════════════════════════════════════════════════════
// ERROR CODES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumeration of all game domain error codes.
 *
 * Error codes are prefixed by category:
 * - GAME_*: General game state errors
 * - MOVE_*: Move-related errors
 * - AI_*: AI service errors
 * - DECISION_*: Player decision errors
 * - PLAYER_*: Player-related errors
 */
export enum GameErrorCode {
  // Game State Errors
  GAME_NOT_FOUND = 'GAME_NOT_FOUND',
  GAME_NOT_ACTIVE = 'GAME_NOT_ACTIVE',
  GAME_ALREADY_STARTED = 'GAME_ALREADY_STARTED',
  GAME_ALREADY_COMPLETED = 'GAME_ALREADY_COMPLETED',
  GAME_INVALID_STATE = 'GAME_INVALID_STATE',
  GAME_INVALID_PHASE = 'GAME_INVALID_PHASE',

  // Move Errors
  MOVE_INVALID = 'MOVE_INVALID',
  MOVE_NOT_YOUR_TURN = 'MOVE_NOT_YOUR_TURN',
  MOVE_INVALID_POSITION = 'MOVE_INVALID_POSITION',
  MOVE_INVALID_TYPE = 'MOVE_INVALID_TYPE',
  MOVE_VALIDATION_FAILED = 'MOVE_VALIDATION_FAILED',
  MOVE_APPLICATION_FAILED = 'MOVE_APPLICATION_FAILED',

  // AI Service Errors
  AI_SERVICE_UNAVAILABLE = 'AI_SERVICE_UNAVAILABLE',
  AI_SERVICE_TIMEOUT = 'AI_SERVICE_TIMEOUT',
  AI_SERVICE_ERROR = 'AI_SERVICE_ERROR',
  AI_NO_MOVE_RETURNED = 'AI_NO_MOVE_RETURNED',
  AI_FALLBACK_FAILED = 'AI_FALLBACK_FAILED',
  AI_FATAL_FAILURE = 'AI_FATAL_FAILURE',

  // Decision/Choice Errors
  DECISION_TIMEOUT = 'DECISION_TIMEOUT',
  DECISION_INVALID_CHOICE = 'DECISION_INVALID_CHOICE',
  DECISION_NOT_PENDING = 'DECISION_NOT_PENDING',
  DECISION_WRONG_PLAYER = 'DECISION_WRONG_PLAYER',

  // Player Errors
  PLAYER_NOT_FOUND = 'PLAYER_NOT_FOUND',
  PLAYER_NOT_IN_GAME = 'PLAYER_NOT_IN_GAME',
  PLAYER_ALREADY_IN_GAME = 'PLAYER_ALREADY_IN_GAME',
  PLAYER_UNAUTHORIZED = 'PLAYER_UNAUTHORIZED',
  PLAYER_DISCONNECTED = 'PLAYER_DISCONNECTED',

  // Internal Errors
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  CONFIGURATION_ERROR = 'CONFIGURATION_ERROR',
}

/**
 * HTTP status codes for error types.
 */
export const ERROR_HTTP_STATUS: Record<GameErrorCode, number> = {
  // Game State Errors - 4xx
  [GameErrorCode.GAME_NOT_FOUND]: 404,
  [GameErrorCode.GAME_NOT_ACTIVE]: 409,
  [GameErrorCode.GAME_ALREADY_STARTED]: 409,
  [GameErrorCode.GAME_ALREADY_COMPLETED]: 409,
  [GameErrorCode.GAME_INVALID_STATE]: 400,
  [GameErrorCode.GAME_INVALID_PHASE]: 400,

  // Move Errors - 400
  [GameErrorCode.MOVE_INVALID]: 400,
  [GameErrorCode.MOVE_NOT_YOUR_TURN]: 403,
  [GameErrorCode.MOVE_INVALID_POSITION]: 400,
  [GameErrorCode.MOVE_INVALID_TYPE]: 400,
  [GameErrorCode.MOVE_VALIDATION_FAILED]: 400,
  [GameErrorCode.MOVE_APPLICATION_FAILED]: 500,

  // AI Errors - 5xx
  [GameErrorCode.AI_SERVICE_UNAVAILABLE]: 503,
  [GameErrorCode.AI_SERVICE_TIMEOUT]: 504,
  [GameErrorCode.AI_SERVICE_ERROR]: 502,
  [GameErrorCode.AI_NO_MOVE_RETURNED]: 500,
  [GameErrorCode.AI_FALLBACK_FAILED]: 500,
  [GameErrorCode.AI_FATAL_FAILURE]: 500,

  // Decision Errors - 4xx
  [GameErrorCode.DECISION_TIMEOUT]: 408,
  [GameErrorCode.DECISION_INVALID_CHOICE]: 400,
  [GameErrorCode.DECISION_NOT_PENDING]: 400,
  [GameErrorCode.DECISION_WRONG_PLAYER]: 403,

  // Player Errors - 4xx
  [GameErrorCode.PLAYER_NOT_FOUND]: 404,
  [GameErrorCode.PLAYER_NOT_IN_GAME]: 404,
  [GameErrorCode.PLAYER_ALREADY_IN_GAME]: 409,
  [GameErrorCode.PLAYER_UNAUTHORIZED]: 403,
  [GameErrorCode.PLAYER_DISCONNECTED]: 410,

  // Internal Errors - 500
  [GameErrorCode.INTERNAL_ERROR]: 500,
  [GameErrorCode.CONFIGURATION_ERROR]: 500,
};

// ═══════════════════════════════════════════════════════════════════════════
// BASE ERROR CLASS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Base class for all game domain errors.
 *
 * Provides:
 * - Structured error code
 * - Context for debugging
 * - HTTP status code mapping
 * - Serialization for API responses
 */
export class GameError extends Error {
  /** Error code for programmatic handling */
  readonly code: GameErrorCode;

  /** Additional context for debugging */
  readonly context: Record<string, unknown>;

  /** Whether this error is fatal (game should be aborted) */
  readonly isFatal: boolean;

  /** Timestamp when error occurred */
  readonly timestamp: Date;

  constructor(
    code: GameErrorCode,
    message: string,
    context: Record<string, unknown> = {},
    isFatal: boolean = false
  ) {
    super(message);
    this.name = 'GameError';
    this.code = code;
    this.context = context;
    this.isFatal = isFatal;
    this.timestamp = new Date();

    // Maintain proper prototype chain
    Object.setPrototypeOf(this, GameError.prototype);
  }

  /** Get HTTP status code for this error */
  get httpStatus(): number {
    return ERROR_HTTP_STATUS[this.code] ?? 500;
  }

  /** Serialize to a JSON-safe object for API responses */
  toJSON(): GameErrorJSON {
    return {
      error: true,
      code: this.code,
      message: this.message,
      context: this.context,
      isFatal: this.isFatal,
      timestamp: this.timestamp.toISOString(),
    };
  }

  /** Create from a JSON representation */
  static fromJSON(json: GameErrorJSON): GameError {
    return new GameError(
      json.code as GameErrorCode,
      json.message,
      json.context,
      json.isFatal
    );
  }
}

/**
 * JSON representation of a GameError.
 */
export interface GameErrorJSON {
  error: true;
  code: string;
  message: string;
  context: Record<string, unknown>;
  isFatal: boolean;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECIFIC ERROR CLASSES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Error for invalid moves.
 */
export class InvalidMoveError extends GameError {
  constructor(message: string, context: Record<string, unknown> = {}) {
    super(GameErrorCode.MOVE_INVALID, message, context, false);
    this.name = 'InvalidMoveError';
    Object.setPrototypeOf(this, InvalidMoveError.prototype);
  }
}

/**
 * Error when it's not the player's turn.
 */
export class NotYourTurnError extends GameError {
  constructor(
    expectedPlayer: number,
    actualPlayer: number,
    context: Record<string, unknown> = {}
  ) {
    super(
      GameErrorCode.MOVE_NOT_YOUR_TURN,
      `Not your turn. Expected player ${expectedPlayer}, got player ${actualPlayer}`,
      { expectedPlayer, actualPlayer, ...context },
      false
    );
    this.name = 'NotYourTurnError';
    Object.setPrototypeOf(this, NotYourTurnError.prototype);
  }
}

/**
 * Error when game is not found.
 */
export class GameNotFoundError extends GameError {
  constructor(gameId: string, context: Record<string, unknown> = {}) {
    super(
      GameErrorCode.GAME_NOT_FOUND,
      `Game not found: ${gameId}`,
      { gameId, ...context },
      false
    );
    this.name = 'GameNotFoundError';
    Object.setPrototypeOf(this, GameNotFoundError.prototype);
  }
}

/**
 * Error when game is not active.
 */
export class GameNotActiveError extends GameError {
  constructor(
    gameId: string,
    status: string,
    context: Record<string, unknown> = {}
  ) {
    super(
      GameErrorCode.GAME_NOT_ACTIVE,
      `Game ${gameId} is not active (status: ${status})`,
      { gameId, status, ...context },
      false
    );
    this.name = 'GameNotActiveError';
    Object.setPrototypeOf(this, GameNotActiveError.prototype);
  }
}

/**
 * Error when AI service is unavailable.
 */
export class AIServiceUnavailableError extends GameError {
  constructor(reason: string, context: Record<string, unknown> = {}) {
    super(
      GameErrorCode.AI_SERVICE_UNAVAILABLE,
      `AI service unavailable: ${reason}`,
      { reason, ...context },
      false
    );
    this.name = 'AIServiceUnavailableError';
    Object.setPrototypeOf(this, AIServiceUnavailableError.prototype);
  }
}

/**
 * Error when AI service times out.
 */
export class AIServiceTimeoutError extends GameError {
  constructor(timeoutMs: number, context: Record<string, unknown> = {}) {
    super(
      GameErrorCode.AI_SERVICE_TIMEOUT,
      `AI service timed out after ${timeoutMs}ms`,
      { timeoutMs, ...context },
      false
    );
    this.name = 'AIServiceTimeoutError';
    Object.setPrototypeOf(this, AIServiceTimeoutError.prototype);
  }
}

/**
 * Error when a decision times out.
 */
export class DecisionTimeoutError extends GameError {
  constructor(
    player: number,
    choiceType: string,
    timeoutMs: number,
    context: Record<string, unknown> = {}
  ) {
    super(
      GameErrorCode.DECISION_TIMEOUT,
      `Player ${player} decision timeout for ${choiceType} after ${timeoutMs}ms`,
      { player, choiceType, timeoutMs, ...context },
      false
    );
    this.name = 'DecisionTimeoutError';
    Object.setPrototypeOf(this, DecisionTimeoutError.prototype);
  }
}

/**
 * Error when player is not authorized.
 */
export class PlayerUnauthorizedError extends GameError {
  constructor(
    playerId: string,
    action: string,
    context: Record<string, unknown> = {}
  ) {
    super(
      GameErrorCode.PLAYER_UNAUTHORIZED,
      `Player ${playerId} not authorized for ${action}`,
      { playerId, action, ...context },
      false
    );
    this.name = 'PlayerUnauthorizedError';
    Object.setPrototypeOf(this, PlayerUnauthorizedError.prototype);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if an error is a GameError.
 */
export function isGameError(error: unknown): error is GameError {
  return error instanceof GameError;
}

/**
 * Check if an error is fatal.
 */
export function isFatalError(error: unknown): boolean {
  return isGameError(error) && error.isFatal;
}

/**
 * Get HTTP status code for an error.
 */
export function getHttpStatus(error: unknown): number {
  if (isGameError(error)) {
    return error.httpStatus;
  }
  return 500;
}

/**
 * Wrap an unknown error in a GameError.
 */
export function wrapError(
  error: unknown,
  context: Record<string, unknown> = {}
): GameError {
  if (isGameError(error)) {
    return error;
  }

  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;

  return new GameError(GameErrorCode.INTERNAL_ERROR, message, {
    ...context,
    originalStack: stack,
  });
}
