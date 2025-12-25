/**
 * Branch coverage tests for CanonicalRecordError
 *
 * Tests all error factory functions and edge cases in the canonical
 * record error system.
 */

import {
  CanonicalRecordError,
  createLegacyCoercionError,
  createMissingBookkeepingError,
  createPhaseMismatchError,
  createPlayerMismatchError,
  isCanonicalRecordError,
  type PhaseValidationError,
  type PhaseValidationErrorType,
} from '../../../src/shared/engine/errors/CanonicalRecordError';

describe('CanonicalRecordError', () => {
  describe('constructor and properties', () => {
    it('should create error with all properties', () => {
      const validationError: PhaseValidationError = {
        type: 'PHASE_MISMATCH',
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
        message: 'Test error message',
        gameId: 'game-123',
        moveNumber: 42,
        expectedPlayer: 1,
        actualPlayer: 2,
      };

      const error = new CanonicalRecordError(validationError);

      expect(error.name).toBe('CanonicalRecordError');
      expect(error.validationError).toEqual(validationError);
      expect(error.type).toBe('PHASE_MISMATCH');
      expect(error.expectedPhase).toBe('ring_placement');
      expect(error.actualPhase).toBe('movement');
      expect(error.moveType).toBe('place_ring');
      expect(error.gameId).toBe('game-123');
      expect(error.moveNumber).toBe(42);
      expect(error).toBeInstanceOf(Error);
    });

    it('should include formatted message with all context', () => {
      const validationError: PhaseValidationError = {
        type: 'PLAYER_MISMATCH',
        expectedPhase: 'movement',
        actualPhase: 'movement',
        moveType: 'move_stack',
        message: 'Player mismatch',
        gameId: 'abc-def',
        moveNumber: 10,
        expectedPlayer: 2,
        actualPlayer: 1,
      };

      const error = new CanonicalRecordError(validationError);

      expect(error.message).toContain('Player mismatch');
      expect(error.message).toContain('Game: abc-def');
      expect(error.message).toContain('Move #10');
      expect(error.message).toContain('Expected phase: movement');
      expect(error.message).toContain('Actual phase: movement');
      expect(error.message).toContain('Move type: move_stack');
      expect(error.message).toContain('Expected player: 2');
      expect(error.message).toContain('Actual player: 1');
      expect(error.message).toContain('RR-CANON-R073');
    });

    it('should handle missing optional fields', () => {
      const validationError: PhaseValidationError = {
        type: 'INVALID_PHASE_TRANSITION',
        expectedPhase: 'capture',
        actualPhase: 'line_processing',
        moveType: 'overtaking_capture',
        message: 'Invalid transition',
      };

      const error = new CanonicalRecordError(validationError);

      expect(error.gameId).toBeUndefined();
      expect(error.moveNumber).toBeUndefined();
      expect(error.message).toContain('Invalid transition');
      expect(error.message).not.toContain('Game:');
    });

    it('should serialize to JSON correctly', () => {
      const validationError: PhaseValidationError = {
        type: 'MISSING_BOOKKEEPING',
        expectedPhase: 'line_processing',
        actualPhase: 'territory_processing',
        moveType: 'no_line_action',
        message: 'Missing bookkeeping',
        gameId: 'test-game',
      };

      const error = new CanonicalRecordError(validationError);
      const json = error.toJSON();

      expect(json.name).toBe('CanonicalRecordError');
      expect(json.message).toBeDefined();
      expect(json.validationError).toEqual(validationError);
    });
  });

  describe('createPhaseMismatchError', () => {
    it('should create phase mismatch error with minimal params', () => {
      const error = createPhaseMismatchError({
        expectedPhase: 'ring_placement',
        actualPhase: 'territory_processing',
        moveType: 'place_ring',
      });

      expect(error).toBeInstanceOf(CanonicalRecordError);
      expect(error.type).toBe('PHASE_MISMATCH');
      expect(error.expectedPhase).toBe('ring_placement');
      expect(error.actualPhase).toBe('territory_processing');
      expect(error.moveType).toBe('place_ring');
      expect(error.message).toContain('Cannot apply place_ring move in ring_placement phase');
    });

    it('should create phase mismatch error with all optional params', () => {
      const error = createPhaseMismatchError({
        expectedPhase: 'movement',
        actualPhase: 'capture',
        moveType: 'move_stack',
        gameId: 'game-xyz',
        moveNumber: 15,
        expectedPlayer: 1,
        actualPlayer: 2,
      });

      expect(error.gameId).toBe('game-xyz');
      expect(error.moveNumber).toBe(15);
      expect(error.validationError.expectedPlayer).toBe(1);
      expect(error.validationError.actualPlayer).toBe(2);
    });
  });

  describe('createPlayerMismatchError', () => {
    it('should create player mismatch error with required params', () => {
      const error = createPlayerMismatchError({
        currentPhase: 'movement',
        moveType: 'move_stack',
        expectedPlayer: 1,
        actualPlayer: 2,
      });

      expect(error).toBeInstanceOf(CanonicalRecordError);
      expect(error.type).toBe('PLAYER_MISMATCH');
      expect(error.validationError.expectedPlayer).toBe(1);
      expect(error.validationError.actualPlayer).toBe(2);
      expect(error.message).toContain('Move move_stack from player 2 but expected player 1');
    });

    it('should include gameId and moveNumber when provided', () => {
      const error = createPlayerMismatchError({
        currentPhase: 'capture',
        moveType: 'overtaking_capture',
        expectedPlayer: 2,
        actualPlayer: 3,
        gameId: 'test-123',
        moveNumber: 50,
      });

      expect(error.gameId).toBe('test-123');
      expect(error.moveNumber).toBe(50);
    });
  });

  describe('createMissingBookkeepingError', () => {
    it('should create missing bookkeeping error with required params', () => {
      const error = createMissingBookkeepingError({
        currentPhase: 'line_processing',
        nextPhase: 'territory_processing',
        moveType: 'choose_territory_option',
      });

      expect(error).toBeInstanceOf(CanonicalRecordError);
      expect(error.type).toBe('MISSING_BOOKKEEPING');
      expect(error.expectedPhase).toBe('line_processing');
      expect(error.actualPhase).toBe('territory_processing');
      expect(error.message).toContain(
        'Transition from line_processing to territory_processing requires explicit bookkeeping move'
      );
    });

    it('should include gameId and moveNumber when provided', () => {
      const error = createMissingBookkeepingError({
        currentPhase: 'ring_placement',
        nextPhase: 'movement',
        moveType: 'move_stack',
        gameId: 'bookkeeping-test',
        moveNumber: 5,
      });

      expect(error.gameId).toBe('bookkeeping-test');
      expect(error.moveNumber).toBe(5);
    });
  });

  describe('createLegacyCoercionError', () => {
    it('should create legacy coercion error with required params', () => {
      const error = createLegacyCoercionError({
        currentPhase: 'capture',
        wouldCoerceTo: 'chain_capture',
        moveType: 'continue_capture_segment',
      });

      expect(error).toBeInstanceOf(CanonicalRecordError);
      expect(error.type).toBe('LEGACY_COERCION_DETECTED');
      expect(error.expectedPhase).toBe('capture');
      expect(error.actualPhase).toBe('chain_capture');
      expect(error.message).toContain('Non-canonical record');
      expect(error.message).toContain('check_canonical_phase_history.py');
      expect(error.message).toContain('replayCompatibility');
    });

    it('should include all optional params when provided', () => {
      const error = createLegacyCoercionError({
        currentPhase: 'movement',
        wouldCoerceTo: 'capture',
        moveType: 'overtaking_capture',
        gameId: 'legacy-game',
        moveNumber: 100,
        currentPlayer: 1,
        movePlayer: 2,
      });

      expect(error.gameId).toBe('legacy-game');
      expect(error.moveNumber).toBe(100);
      expect(error.validationError.expectedPlayer).toBe(1);
      expect(error.validationError.actualPlayer).toBe(2);
    });
  });

  describe('isCanonicalRecordError', () => {
    it('should return true for CanonicalRecordError instances', () => {
      const error = createPhaseMismatchError({
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
      });

      expect(isCanonicalRecordError(error)).toBe(true);
    });

    it('should return false for regular Error', () => {
      expect(isCanonicalRecordError(new Error('test'))).toBe(false);
    });

    it('should return false for non-error values', () => {
      expect(isCanonicalRecordError(null)).toBe(false);
      expect(isCanonicalRecordError(undefined)).toBe(false);
      expect(isCanonicalRecordError('error string')).toBe(false);
      expect(isCanonicalRecordError(42)).toBe(false);
      expect(isCanonicalRecordError({})).toBe(false);
      expect(isCanonicalRecordError({ type: 'PHASE_MISMATCH' })).toBe(false);
    });
  });

  describe('formatErrorMessage branches', () => {
    it('should format message without optional fields', () => {
      const error = new CanonicalRecordError({
        type: 'PHASE_MISMATCH',
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
        message: 'Base message',
      });

      expect(error.message).toContain('Base message');
      expect(error.message).not.toContain('Game:');
      expect(error.message).not.toContain('Move #');
      expect(error.message).not.toContain('Expected player:');
    });

    it('should format message with gameId only', () => {
      const error = new CanonicalRecordError({
        type: 'PHASE_MISMATCH',
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
        message: 'Base message',
        gameId: 'only-game-id',
      });

      expect(error.message).toContain('Game: only-game-id');
      expect(error.message).not.toContain('Move #');
    });

    it('should format message with moveNumber only', () => {
      const error = new CanonicalRecordError({
        type: 'PHASE_MISMATCH',
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
        message: 'Base message',
        moveNumber: 25,
      });

      expect(error.message).toContain('Move #25');
      expect(error.message).not.toContain('Game:');
    });

    it('should format message with player mismatch info', () => {
      const error = new CanonicalRecordError({
        type: 'PLAYER_MISMATCH',
        expectedPhase: 'movement',
        actualPhase: 'movement',
        moveType: 'move_stack',
        message: 'Player mismatch',
        expectedPlayer: 1,
        actualPlayer: 3,
      });

      expect(error.message).toContain('Expected player: 1');
      expect(error.message).toContain('Actual player: 3');
    });

    it('should include canonical rule reference', () => {
      const error = createPhaseMismatchError({
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
      });

      expect(error.message).toContain('RULES_CANONICAL_SPEC.md');
      expect(error.message).toContain('RR-CANON-R073');
      expect(error.message).toContain('RR-CANON-R075');
    });
  });

  describe('PhaseValidationErrorType coverage', () => {
    const errorTypes: PhaseValidationErrorType[] = [
      'PHASE_MISMATCH',
      'PLAYER_MISMATCH',
      'MISSING_BOOKKEEPING',
      'INVALID_PHASE_TRANSITION',
      'LEGACY_COERCION_DETECTED',
    ];

    it.each(errorTypes)('should handle error type: %s', (errorType) => {
      const error = new CanonicalRecordError({
        type: errorType,
        expectedPhase: 'ring_placement',
        actualPhase: 'movement',
        moveType: 'place_ring',
        message: `Test message for ${errorType}`,
      });

      expect(error.type).toBe(errorType);
      expect(error.validationError.type).toBe(errorType);
    });
  });
});
