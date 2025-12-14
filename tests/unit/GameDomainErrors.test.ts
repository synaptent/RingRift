/**
 * Tests for GameDomainErrors - Structured error types for the game domain
 * @module tests/unit/GameDomainErrors.test
 */

import {
  GameError,
  GameErrorCode,
  ERROR_HTTP_STATUS,
  InvalidMoveError,
  NotYourTurnError,
  GameNotFoundError,
  GameNotActiveError,
  AIServiceUnavailableError,
  AIServiceTimeoutError,
  DecisionTimeoutError,
  PlayerUnauthorizedError,
  isGameError,
  isFatalError,
  getHttpStatus,
  wrapError,
  type GameErrorJSON,
} from '../../src/shared/errors/GameDomainErrors';

describe('GameDomainErrors', () => {
  describe('GameErrorCode enum', () => {
    it('should have all expected game state error codes', () => {
      expect(GameErrorCode.GAME_NOT_FOUND).toBe('GAME_NOT_FOUND');
      expect(GameErrorCode.GAME_NOT_ACTIVE).toBe('GAME_NOT_ACTIVE');
      expect(GameErrorCode.GAME_ALREADY_STARTED).toBe('GAME_ALREADY_STARTED');
      expect(GameErrorCode.GAME_ALREADY_COMPLETED).toBe('GAME_ALREADY_COMPLETED');
      expect(GameErrorCode.GAME_INVALID_STATE).toBe('GAME_INVALID_STATE');
      expect(GameErrorCode.GAME_INVALID_PHASE).toBe('GAME_INVALID_PHASE');
    });

    it('should have all expected move error codes', () => {
      expect(GameErrorCode.MOVE_INVALID).toBe('MOVE_INVALID');
      expect(GameErrorCode.MOVE_NOT_YOUR_TURN).toBe('MOVE_NOT_YOUR_TURN');
      expect(GameErrorCode.MOVE_INVALID_POSITION).toBe('MOVE_INVALID_POSITION');
      expect(GameErrorCode.MOVE_INVALID_TYPE).toBe('MOVE_INVALID_TYPE');
      expect(GameErrorCode.MOVE_VALIDATION_FAILED).toBe('MOVE_VALIDATION_FAILED');
      expect(GameErrorCode.MOVE_APPLICATION_FAILED).toBe('MOVE_APPLICATION_FAILED');
    });

    it('should have all expected AI error codes', () => {
      expect(GameErrorCode.AI_SERVICE_UNAVAILABLE).toBe('AI_SERVICE_UNAVAILABLE');
      expect(GameErrorCode.AI_SERVICE_TIMEOUT).toBe('AI_SERVICE_TIMEOUT');
      expect(GameErrorCode.AI_SERVICE_ERROR).toBe('AI_SERVICE_ERROR');
      expect(GameErrorCode.AI_NO_MOVE_RETURNED).toBe('AI_NO_MOVE_RETURNED');
      expect(GameErrorCode.AI_FALLBACK_FAILED).toBe('AI_FALLBACK_FAILED');
      expect(GameErrorCode.AI_FATAL_FAILURE).toBe('AI_FATAL_FAILURE');
    });

    it('should have all expected decision error codes', () => {
      expect(GameErrorCode.DECISION_TIMEOUT).toBe('DECISION_TIMEOUT');
      expect(GameErrorCode.DECISION_INVALID_CHOICE).toBe('DECISION_INVALID_CHOICE');
      expect(GameErrorCode.DECISION_NOT_PENDING).toBe('DECISION_NOT_PENDING');
      expect(GameErrorCode.DECISION_WRONG_PLAYER).toBe('DECISION_WRONG_PLAYER');
    });

    it('should have all expected player error codes', () => {
      expect(GameErrorCode.PLAYER_NOT_FOUND).toBe('PLAYER_NOT_FOUND');
      expect(GameErrorCode.PLAYER_NOT_IN_GAME).toBe('PLAYER_NOT_IN_GAME');
      expect(GameErrorCode.PLAYER_ALREADY_IN_GAME).toBe('PLAYER_ALREADY_IN_GAME');
      expect(GameErrorCode.PLAYER_UNAUTHORIZED).toBe('PLAYER_UNAUTHORIZED');
      expect(GameErrorCode.PLAYER_DISCONNECTED).toBe('PLAYER_DISCONNECTED');
    });
  });

  describe('ERROR_HTTP_STATUS mapping', () => {
    it('should map game state errors to correct HTTP status', () => {
      expect(ERROR_HTTP_STATUS[GameErrorCode.GAME_NOT_FOUND]).toBe(404);
      expect(ERROR_HTTP_STATUS[GameErrorCode.GAME_NOT_ACTIVE]).toBe(409);
      expect(ERROR_HTTP_STATUS[GameErrorCode.GAME_ALREADY_STARTED]).toBe(409);
      expect(ERROR_HTTP_STATUS[GameErrorCode.GAME_INVALID_STATE]).toBe(400);
    });

    it('should map move errors to correct HTTP status', () => {
      expect(ERROR_HTTP_STATUS[GameErrorCode.MOVE_INVALID]).toBe(400);
      expect(ERROR_HTTP_STATUS[GameErrorCode.MOVE_NOT_YOUR_TURN]).toBe(403);
      expect(ERROR_HTTP_STATUS[GameErrorCode.MOVE_APPLICATION_FAILED]).toBe(500);
    });

    it('should map AI errors to 5xx status codes', () => {
      expect(ERROR_HTTP_STATUS[GameErrorCode.AI_SERVICE_UNAVAILABLE]).toBe(503);
      expect(ERROR_HTTP_STATUS[GameErrorCode.AI_SERVICE_TIMEOUT]).toBe(504);
      expect(ERROR_HTTP_STATUS[GameErrorCode.AI_SERVICE_ERROR]).toBe(502);
      expect(ERROR_HTTP_STATUS[GameErrorCode.AI_FATAL_FAILURE]).toBe(500);
    });

    it('should map decision errors to correct HTTP status', () => {
      expect(ERROR_HTTP_STATUS[GameErrorCode.DECISION_TIMEOUT]).toBe(408);
      expect(ERROR_HTTP_STATUS[GameErrorCode.DECISION_INVALID_CHOICE]).toBe(400);
      expect(ERROR_HTTP_STATUS[GameErrorCode.DECISION_WRONG_PLAYER]).toBe(403);
    });
  });

  describe('GameError base class', () => {
    it('should create an error with all properties', () => {
      const error = new GameError(
        GameErrorCode.GAME_NOT_FOUND,
        'Test game not found',
        { gameId: 'game-123' },
        false
      );

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(GameError);
      expect(error.name).toBe('GameError');
      expect(error.code).toBe(GameErrorCode.GAME_NOT_FOUND);
      expect(error.message).toBe('Test game not found');
      expect(error.context).toEqual({ gameId: 'game-123' });
      expect(error.isFatal).toBe(false);
      expect(error.timestamp).toBeInstanceOf(Date);
    });

    it('should default context to empty object and isFatal to false', () => {
      const error = new GameError(GameErrorCode.INTERNAL_ERROR, 'Test error');

      expect(error.context).toEqual({});
      expect(error.isFatal).toBe(false);
    });

    it('should support fatal errors', () => {
      const error = new GameError(GameErrorCode.AI_FATAL_FAILURE, 'Critical failure', {}, true);

      expect(error.isFatal).toBe(true);
    });

    it('should return correct HTTP status via httpStatus getter', () => {
      const notFoundError = new GameError(GameErrorCode.GAME_NOT_FOUND, 'Not found');
      expect(notFoundError.httpStatus).toBe(404);

      const timeoutError = new GameError(GameErrorCode.AI_SERVICE_TIMEOUT, 'Timeout');
      expect(timeoutError.httpStatus).toBe(504);
    });

    it('should default to 500 for unknown error codes', () => {
      const error = new GameError('UNKNOWN_CODE' as GameErrorCode, 'Unknown');
      expect(error.httpStatus).toBe(500);
    });

    it('should serialize to JSON correctly', () => {
      const error = new GameError(
        GameErrorCode.MOVE_INVALID,
        'Invalid move',
        { position: { x: 1, y: 2 } },
        false
      );
      const json = error.toJSON();

      expect(json.error).toBe(true);
      expect(json.code).toBe('MOVE_INVALID');
      expect(json.message).toBe('Invalid move');
      expect(json.context).toEqual({ position: { x: 1, y: 2 } });
      expect(json.isFatal).toBe(false);
      expect(typeof json.timestamp).toBe('string');
      expect(new Date(json.timestamp)).toBeInstanceOf(Date);
    });

    it('should create error from JSON', () => {
      const json: GameErrorJSON = {
        error: true,
        code: 'GAME_NOT_ACTIVE',
        message: 'Game is over',
        context: { gameId: 'g-456' },
        isFatal: false,
        timestamp: new Date().toISOString(),
      };

      const error = GameError.fromJSON(json);

      expect(error).toBeInstanceOf(GameError);
      expect(error.code).toBe(GameErrorCode.GAME_NOT_ACTIVE);
      expect(error.message).toBe('Game is over');
      expect(error.context).toEqual({ gameId: 'g-456' });
    });
  });

  describe('InvalidMoveError', () => {
    it('should create with correct code and properties', () => {
      const error = new InvalidMoveError('Cannot place ring here', {
        moveType: 'place_ring',
        position: { x: 3, y: 3 },
      });

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(InvalidMoveError);
      expect(error.name).toBe('InvalidMoveError');
      expect(error.code).toBe(GameErrorCode.MOVE_INVALID);
      expect(error.message).toBe('Cannot place ring here');
      expect(error.context.moveType).toBe('place_ring');
      expect(error.isFatal).toBe(false);
    });
  });

  describe('NotYourTurnError', () => {
    it('should create with correct message and context', () => {
      const error = new NotYourTurnError(1, 2, { phase: 'movement' });

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(NotYourTurnError);
      expect(error.name).toBe('NotYourTurnError');
      expect(error.code).toBe(GameErrorCode.MOVE_NOT_YOUR_TURN);
      expect(error.message).toBe('Not your turn. Expected player 1, got player 2');
      expect(error.context.expectedPlayer).toBe(1);
      expect(error.context.actualPlayer).toBe(2);
      expect(error.context.phase).toBe('movement');
    });
  });

  describe('GameNotFoundError', () => {
    it('should create with gameId in message and context', () => {
      const error = new GameNotFoundError('game-xyz');

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(GameNotFoundError);
      expect(error.name).toBe('GameNotFoundError');
      expect(error.code).toBe(GameErrorCode.GAME_NOT_FOUND);
      expect(error.message).toBe('Game not found: game-xyz');
      expect(error.context.gameId).toBe('game-xyz');
    });
  });

  describe('GameNotActiveError', () => {
    it('should create with gameId and status', () => {
      const error = new GameNotActiveError('game-123', 'COMPLETED');

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(GameNotActiveError);
      expect(error.name).toBe('GameNotActiveError');
      expect(error.code).toBe(GameErrorCode.GAME_NOT_ACTIVE);
      expect(error.message).toBe('Game game-123 is not active (status: COMPLETED)');
      expect(error.context.gameId).toBe('game-123');
      expect(error.context.status).toBe('COMPLETED');
    });
  });

  describe('AIServiceUnavailableError', () => {
    it('should create with reason', () => {
      const error = new AIServiceUnavailableError('Connection refused');

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(AIServiceUnavailableError);
      expect(error.name).toBe('AIServiceUnavailableError');
      expect(error.code).toBe(GameErrorCode.AI_SERVICE_UNAVAILABLE);
      expect(error.message).toBe('AI service unavailable: Connection refused');
      expect(error.context.reason).toBe('Connection refused');
    });
  });

  describe('AIServiceTimeoutError', () => {
    it('should create with timeout value', () => {
      const error = new AIServiceTimeoutError(5000, { requestId: 'req-123' });

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(AIServiceTimeoutError);
      expect(error.name).toBe('AIServiceTimeoutError');
      expect(error.code).toBe(GameErrorCode.AI_SERVICE_TIMEOUT);
      expect(error.message).toBe('AI service timed out after 5000ms');
      expect(error.context.timeoutMs).toBe(5000);
      expect(error.context.requestId).toBe('req-123');
    });
  });

  describe('DecisionTimeoutError', () => {
    it('should create with player, choice type, and timeout', () => {
      const error = new DecisionTimeoutError(2, 'line_reward', 30000);

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(DecisionTimeoutError);
      expect(error.name).toBe('DecisionTimeoutError');
      expect(error.code).toBe(GameErrorCode.DECISION_TIMEOUT);
      expect(error.message).toBe('Player 2 decision timeout for line_reward after 30000ms');
      expect(error.context.player).toBe(2);
      expect(error.context.choiceType).toBe('line_reward');
      expect(error.context.timeoutMs).toBe(30000);
    });
  });

  describe('PlayerUnauthorizedError', () => {
    it('should create with player and action', () => {
      const error = new PlayerUnauthorizedError('player-789', 'make_move');

      expect(error).toBeInstanceOf(GameError);
      expect(error).toBeInstanceOf(PlayerUnauthorizedError);
      expect(error.name).toBe('PlayerUnauthorizedError');
      expect(error.code).toBe(GameErrorCode.PLAYER_UNAUTHORIZED);
      expect(error.message).toBe('Player player-789 not authorized for make_move');
      expect(error.context.playerId).toBe('player-789');
      expect(error.context.action).toBe('make_move');
    });
  });

  describe('isGameError utility', () => {
    it('should return true for GameError instances', () => {
      expect(isGameError(new GameError(GameErrorCode.INTERNAL_ERROR, 'test'))).toBe(true);
      expect(isGameError(new InvalidMoveError('test'))).toBe(true);
      expect(isGameError(new GameNotFoundError('g-1'))).toBe(true);
    });

    it('should return false for non-GameError values', () => {
      expect(isGameError(new Error('test'))).toBe(false);
      expect(isGameError('error')).toBe(false);
      expect(isGameError(null)).toBe(false);
      expect(isGameError(undefined)).toBe(false);
      expect(isGameError({ code: 'ERROR' })).toBe(false);
    });
  });

  describe('isFatalError utility', () => {
    it('should return true for fatal GameErrors', () => {
      const fatalError = new GameError(GameErrorCode.AI_FATAL_FAILURE, 'Fatal', {}, true);
      expect(isFatalError(fatalError)).toBe(true);
    });

    it('should return false for non-fatal GameErrors', () => {
      const nonFatalError = new GameError(GameErrorCode.MOVE_INVALID, 'Invalid', {}, false);
      expect(isFatalError(nonFatalError)).toBe(false);
      expect(isFatalError(new InvalidMoveError('test'))).toBe(false);
    });

    it('should return false for non-GameError values', () => {
      expect(isFatalError(new Error('test'))).toBe(false);
      expect(isFatalError('error')).toBe(false);
      expect(isFatalError(null)).toBe(false);
    });
  });

  describe('getHttpStatus utility', () => {
    it('should return HTTP status for GameErrors', () => {
      expect(getHttpStatus(new GameNotFoundError('g-1'))).toBe(404);
      expect(getHttpStatus(new AIServiceTimeoutError(5000))).toBe(504);
      expect(getHttpStatus(new InvalidMoveError('test'))).toBe(400);
    });

    it('should return 500 for non-GameError values', () => {
      expect(getHttpStatus(new Error('test'))).toBe(500);
      expect(getHttpStatus('error')).toBe(500);
      expect(getHttpStatus(null)).toBe(500);
    });
  });

  describe('wrapError utility', () => {
    it('should return the same GameError if already a GameError', () => {
      const original = new InvalidMoveError('test');
      const wrapped = wrapError(original);

      expect(wrapped).toBe(original);
    });

    it('should wrap regular Error in GameError', () => {
      const original = new Error('Something went wrong');
      const wrapped = wrapError(original, { source: 'test' });

      expect(wrapped).toBeInstanceOf(GameError);
      expect(wrapped.code).toBe(GameErrorCode.INTERNAL_ERROR);
      expect(wrapped.message).toBe('Something went wrong');
      expect(wrapped.context.source).toBe('test');
      expect(wrapped.context.originalStack).toBeDefined();
    });

    it('should wrap string errors', () => {
      const wrapped = wrapError('String error message');

      expect(wrapped).toBeInstanceOf(GameError);
      expect(wrapped.code).toBe(GameErrorCode.INTERNAL_ERROR);
      expect(wrapped.message).toBe('String error message');
    });

    it('should wrap other types', () => {
      const wrapped = wrapError(42);

      expect(wrapped).toBeInstanceOf(GameError);
      expect(wrapped.message).toBe('42');
    });

    it('should handle null/undefined', () => {
      expect(wrapError(null).message).toBe('null');
      expect(wrapError(undefined).message).toBe('undefined');
    });
  });

  describe('Error inheritance chain', () => {
    it('should maintain proper prototype chain for all error types', () => {
      const errors = [
        new GameError(GameErrorCode.INTERNAL_ERROR, 'test'),
        new InvalidMoveError('test'),
        new NotYourTurnError(1, 2),
        new GameNotFoundError('g-1'),
        new GameNotActiveError('g-1', 'COMPLETED'),
        new AIServiceUnavailableError('test'),
        new AIServiceTimeoutError(1000),
        new DecisionTimeoutError(1, 'line_reward', 1000),
        new PlayerUnauthorizedError('p-1', 'action'),
      ];

      for (const error of errors) {
        expect(error instanceof Error).toBe(true);
        expect(error instanceof GameError).toBe(true);
        expect(error.stack).toBeDefined();
      }
    });
  });
});
