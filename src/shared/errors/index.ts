/**
 * Shared Errors Module
 *
 * This module exports structured error types for consistent error handling
 * across the RingRift codebase.
 *
 * @module errors
 */

export {
  // Error codes
  GameErrorCode,
  ERROR_HTTP_STATUS,
  // Base class
  GameError,
  type GameErrorJSON,
  // Specific errors
  InvalidMoveError,
  NotYourTurnError,
  GameNotFoundError,
  GameNotActiveError,
  AIServiceUnavailableError,
  AIServiceTimeoutError,
  DecisionTimeoutError,
  PlayerUnauthorizedError,
  // Utilities
  isGameError,
  isFatalError,
  getHttpStatus,
  wrapError,
} from './GameDomainErrors';
