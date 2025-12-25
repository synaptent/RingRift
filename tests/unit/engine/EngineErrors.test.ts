/**
 * Test suite for src/shared/engine/errors.ts
 *
 * This file provides comprehensive coverage for all Engine error classes,
 * error codes, type guards, and utility functions.
 */

import {
  EngineError,
  EngineErrorCode,
  RulesViolation,
  InvalidState,
  BoardConstraintViolation,
  MoveRequirementError,
  isEngineError,
  isRulesViolation,
  isInvalidState,
  isBoardConstraintViolation,
  isMoveRequirementError,
  wrapEngineError,
  entityNotFound,
  moveMissingField,
  ERROR_CATEGORY_DESCRIPTIONS,
} from '../../../src/shared/engine/errors';

describe('EngineErrors', () => {
  describe('EngineError base class', () => {
    it('should create an EngineError with all fields', () => {
      const error = new EngineError(
        EngineErrorCode.RULES_LINE_REWARD_REQUIRED,
        'Line reward action required',
        { lineLength: 5 },
        'LineMutator',
        'RR-CANON-R042'
      );

      expect(error.code).toBe(EngineErrorCode.RULES_LINE_REWARD_REQUIRED);
      expect(error.message).toBe('Line reward action required');
      expect(error.context).toEqual({ lineLength: 5 });
      expect(error.domain).toBe('LineMutator');
      expect(error.ruleRef).toBe('RR-CANON-R042');
      expect(error.name).toBe('EngineError');
      expect(error.timestamp).toBeInstanceOf(Date);
    });

    it('should use default domain when not specified', () => {
      const error = new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Assertion failed');

      expect(error.domain).toBe('Engine');
      expect(error.context).toEqual({});
      expect(error.ruleRef).toBeUndefined();
    });

    it('should return category description based on error code prefix', () => {
      const rulesError = new EngineError(EngineErrorCode.RULES_LINE_REWARD_REQUIRED, 'Rules error');
      expect(rulesError.category).toBe('Game rule violation (see RULES_CANONICAL_SPEC.md)');

      const stateError = new EngineError(EngineErrorCode.STATE_PLAYER_NOT_FOUND, 'State error');
      expect(stateError.category).toBe('Corrupted or unexpected game state');

      const boardError = new EngineError(EngineErrorCode.BOARD_UNKNOWN_TYPE, 'Board error');
      expect(boardError.category).toBe('Board geometry/topology constraint violation');

      const moveError = new EngineError(EngineErrorCode.MOVE_FROM_REQUIRED, 'Move error');
      expect(moveError.category).toBe('Missing required move field');

      const fsmError = new EngineError(EngineErrorCode.FSM_INVALID_TRANSITION, 'FSM error');
      expect(fsmError.category).toBe('Invalid state machine transition');

      const internalError = new EngineError(
        EngineErrorCode.INTERNAL_ASSERTION_FAILED,
        'Internal error'
      );
      expect(internalError.category).toBe('Internal engine error (bug)');
    });

    it('should return Unknown for unmapped category prefix', () => {
      // Create an error with an unknown prefix by casting
      const error = new EngineError('UNKNOWN_PREFIX_CODE' as EngineErrorCode, 'Unknown error');
      expect(error.category).toBe('Unknown error category');
    });

    it('should serialize to JSON correctly', () => {
      const error = new EngineError(
        EngineErrorCode.RULES_PHASE_MOVE_MISMATCH,
        'Phase mismatch',
        { phase: 'movement', moveType: 'placement' },
        'TurnOrchestrator',
        'RR-CANON-R050'
      );

      const json = error.toJSON();

      expect(json.error).toBe(true);
      expect(json.type).toBe('EngineError');
      expect(json.code).toBe(EngineErrorCode.RULES_PHASE_MOVE_MISMATCH);
      expect(json.message).toBe('Phase mismatch');
      expect(json.domain).toBe('TurnOrchestrator');
      expect(json.context).toEqual({ phase: 'movement', moveType: 'placement' });
      expect(json.ruleRef).toBe('RR-CANON-R050');
      expect(json.category).toBe('Game rule violation (see RULES_CANONICAL_SPEC.md)');
      expect(typeof json.timestamp).toBe('string');
    });

    it('should be an instance of Error', () => {
      const error = new EngineError(EngineErrorCode.INTERNAL_NOT_IMPLEMENTED, 'Not implemented');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(EngineError);
    });
  });

  describe('RulesViolation', () => {
    it('should create a RulesViolation error', () => {
      const error = new RulesViolation(
        EngineErrorCode.RULES_INVALID_RECOVERY_SLIDE,
        'Invalid recovery slide',
        { from: '0,0', to: '1,1' },
        'RecoveryMutator',
        'RR-CANON-R110'
      );

      expect(error.name).toBe('RulesViolation');
      expect(error.code).toBe(EngineErrorCode.RULES_INVALID_RECOVERY_SLIDE);
      expect(error.domain).toBe('RecoveryMutator');
      expect(error.ruleRef).toBe('RR-CANON-R110');
    });

    it('should use default domain', () => {
      const error = new RulesViolation(
        EngineErrorCode.RULES_RECOVERY_LINE_INVALID,
        'Recovery line invalid'
      );

      expect(error.domain).toBe('Rules');
    });

    it('should be an instance of EngineError', () => {
      const error = new RulesViolation(
        EngineErrorCode.RULES_LINE_REWARD_REQUIRED,
        'Line reward required'
      );

      expect(error).toBeInstanceOf(EngineError);
      expect(error).toBeInstanceOf(RulesViolation);
    });
  });

  describe('InvalidState', () => {
    it('should create an InvalidState error', () => {
      const error = new InvalidState(
        EngineErrorCode.STATE_PLAYER_NOT_FOUND,
        'Player not found',
        { playerId: 1 },
        'GameState'
      );

      expect(error.name).toBe('InvalidState');
      expect(error.code).toBe(EngineErrorCode.STATE_PLAYER_NOT_FOUND);
      expect(error.domain).toBe('GameState');
    });

    it('should use default domain', () => {
      const error = new InvalidState(EngineErrorCode.STATE_STACK_NOT_FOUND, 'Stack not found');

      expect(error.domain).toBe('State');
    });

    it('should be an instance of EngineError', () => {
      const error = new InvalidState(EngineErrorCode.STATE_REGION_NOT_FOUND, 'Region not found');

      expect(error).toBeInstanceOf(EngineError);
      expect(error).toBeInstanceOf(InvalidState);
    });
  });

  describe('BoardConstraintViolation', () => {
    it('should create a BoardConstraintViolation error', () => {
      const error = new BoardConstraintViolation(
        EngineErrorCode.BOARD_UNKNOWN_TYPE,
        'Unknown board type',
        { boardType: 'invalid' },
        'BoardManager'
      );

      expect(error.name).toBe('BoardConstraintViolation');
      expect(error.code).toBe(EngineErrorCode.BOARD_UNKNOWN_TYPE);
      expect(error.domain).toBe('BoardManager');
    });

    it('should use default domain', () => {
      const error = new BoardConstraintViolation(
        EngineErrorCode.BOARD_INVALID_POSITION,
        'Invalid position'
      );

      expect(error.domain).toBe('Board');
    });

    it('should be an instance of EngineError', () => {
      const error = new BoardConstraintViolation(
        EngineErrorCode.BOARD_INVALID_POSITION,
        'Invalid position'
      );

      expect(error).toBeInstanceOf(EngineError);
      expect(error).toBeInstanceOf(BoardConstraintViolation);
    });
  });

  describe('MoveRequirementError', () => {
    it('should create a MoveRequirementError error', () => {
      const error = new MoveRequirementError(
        EngineErrorCode.MOVE_FROM_REQUIRED,
        'Move.from is required',
        { moveType: 'movement' },
        'MoveValidator'
      );

      expect(error.name).toBe('MoveRequirementError');
      expect(error.code).toBe(EngineErrorCode.MOVE_FROM_REQUIRED);
      expect(error.domain).toBe('MoveValidator');
    });

    it('should use default domain', () => {
      const error = new MoveRequirementError(
        EngineErrorCode.MOVE_CAPTURE_TARGET_REQUIRED,
        'Capture target required'
      );

      expect(error.domain).toBe('Move');
    });

    it('should be an instance of EngineError', () => {
      const error = new MoveRequirementError(EngineErrorCode.MOVE_WRONG_TYPE, 'Wrong move type');

      expect(error).toBeInstanceOf(EngineError);
      expect(error).toBeInstanceOf(MoveRequirementError);
    });
  });

  describe('Type Guards', () => {
    describe('isEngineError', () => {
      it('should return true for EngineError', () => {
        const error = new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Test');
        expect(isEngineError(error)).toBe(true);
      });

      it('should return true for subclasses', () => {
        expect(
          isEngineError(new RulesViolation(EngineErrorCode.RULES_LINE_REWARD_REQUIRED, 'Test'))
        ).toBe(true);
        expect(
          isEngineError(new InvalidState(EngineErrorCode.STATE_PLAYER_NOT_FOUND, 'Test'))
        ).toBe(true);
        expect(
          isEngineError(new BoardConstraintViolation(EngineErrorCode.BOARD_UNKNOWN_TYPE, 'Test'))
        ).toBe(true);
        expect(
          isEngineError(new MoveRequirementError(EngineErrorCode.MOVE_FROM_REQUIRED, 'Test'))
        ).toBe(true);
      });

      it('should return false for regular Error', () => {
        expect(isEngineError(new Error('Test'))).toBe(false);
      });

      it('should return false for non-errors', () => {
        expect(isEngineError(null)).toBe(false);
        expect(isEngineError(undefined)).toBe(false);
        expect(isEngineError('error string')).toBe(false);
        expect(isEngineError({ message: 'error object' })).toBe(false);
      });
    });

    describe('isRulesViolation', () => {
      it('should return true for RulesViolation', () => {
        const error = new RulesViolation(EngineErrorCode.RULES_LINE_REWARD_REQUIRED, 'Test');
        expect(isRulesViolation(error)).toBe(true);
      });

      it('should return false for other EngineErrors', () => {
        expect(
          isRulesViolation(new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Test'))
        ).toBe(false);
        expect(
          isRulesViolation(new InvalidState(EngineErrorCode.STATE_PLAYER_NOT_FOUND, 'Test'))
        ).toBe(false);
      });

      it('should return false for non-errors', () => {
        expect(isRulesViolation(null)).toBe(false);
        expect(isRulesViolation('error')).toBe(false);
      });
    });

    describe('isInvalidState', () => {
      it('should return true for InvalidState', () => {
        const error = new InvalidState(EngineErrorCode.STATE_PLAYER_NOT_FOUND, 'Test');
        expect(isInvalidState(error)).toBe(true);
      });

      it('should return false for other EngineErrors', () => {
        expect(
          isInvalidState(new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Test'))
        ).toBe(false);
        expect(
          isInvalidState(new RulesViolation(EngineErrorCode.RULES_LINE_REWARD_REQUIRED, 'Test'))
        ).toBe(false);
      });

      it('should return false for non-errors', () => {
        expect(isInvalidState(undefined)).toBe(false);
        expect(isInvalidState(42)).toBe(false);
      });
    });

    describe('isBoardConstraintViolation', () => {
      it('should return true for BoardConstraintViolation', () => {
        const error = new BoardConstraintViolation(EngineErrorCode.BOARD_UNKNOWN_TYPE, 'Test');
        expect(isBoardConstraintViolation(error)).toBe(true);
      });

      it('should return false for other EngineErrors', () => {
        expect(
          isBoardConstraintViolation(
            new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Test')
          )
        ).toBe(false);
        expect(
          isBoardConstraintViolation(
            new InvalidState(EngineErrorCode.STATE_PLAYER_NOT_FOUND, 'Test')
          )
        ).toBe(false);
      });

      it('should return false for non-errors', () => {
        expect(isBoardConstraintViolation({})).toBe(false);
      });
    });

    describe('isMoveRequirementError', () => {
      it('should return true for MoveRequirementError', () => {
        const error = new MoveRequirementError(EngineErrorCode.MOVE_FROM_REQUIRED, 'Test');
        expect(isMoveRequirementError(error)).toBe(true);
      });

      it('should return false for other EngineErrors', () => {
        expect(
          isMoveRequirementError(new EngineError(EngineErrorCode.INTERNAL_ASSERTION_FAILED, 'Test'))
        ).toBe(false);
        expect(
          isMoveRequirementError(
            new RulesViolation(EngineErrorCode.RULES_LINE_REWARD_REQUIRED, 'Test')
          )
        ).toBe(false);
      });

      it('should return false for non-errors', () => {
        expect(isMoveRequirementError([])).toBe(false);
      });
    });
  });

  describe('Utility Functions', () => {
    describe('wrapEngineError', () => {
      it('should return the same EngineError if already an EngineError', () => {
        const original = new EngineError(
          EngineErrorCode.RULES_LINE_REWARD_REQUIRED,
          'Original error'
        );

        const wrapped = wrapEngineError(original, 'TestDomain');

        expect(wrapped).toBe(original);
      });

      it('should wrap a regular Error in EngineError', () => {
        const original = new Error('Regular error message');

        const wrapped = wrapEngineError(original, 'TestDomain', { extra: 'context' });

        expect(wrapped).toBeInstanceOf(EngineError);
        expect(wrapped.code).toBe(EngineErrorCode.INTERNAL_ASSERTION_FAILED);
        expect(wrapped.message).toBe('Regular error message');
        expect(wrapped.domain).toBe('TestDomain');
        expect(wrapped.context.extra).toBe('context');
        expect(wrapped.context.originalStack).toBeDefined();
      });

      it('should wrap a string in EngineError', () => {
        const wrapped = wrapEngineError('String error');

        expect(wrapped).toBeInstanceOf(EngineError);
        expect(wrapped.message).toBe('String error');
        expect(wrapped.domain).toBe('Engine');
        expect(wrapped.context.originalStack).toBeUndefined();
      });

      it('should wrap null in EngineError', () => {
        const wrapped = wrapEngineError(null);

        expect(wrapped).toBeInstanceOf(EngineError);
        expect(wrapped.message).toBe('null');
      });

      it('should wrap undefined in EngineError', () => {
        const wrapped = wrapEngineError(undefined);

        expect(wrapped).toBeInstanceOf(EngineError);
        expect(wrapped.message).toBe('undefined');
      });

      it('should wrap objects in EngineError', () => {
        const wrapped = wrapEngineError({ someKey: 'someValue' });

        expect(wrapped).toBeInstanceOf(EngineError);
        expect(wrapped.message).toBe('[object Object]');
      });
    });

    describe('entityNotFound', () => {
      it('should create InvalidState for player not found', () => {
        const error = entityNotFound('player', { playerId: 2 }, 'GameState');

        expect(error).toBeInstanceOf(InvalidState);
        expect(error.code).toBe(EngineErrorCode.STATE_PLAYER_NOT_FOUND);
        expect(error.message).toBe('Player not found');
        expect(error.context.playerId).toBe(2);
        expect(error.domain).toBe('GameState');
      });

      it('should create InvalidState for stack not found', () => {
        const error = entityNotFound('stack', { position: '3,4' });

        expect(error.code).toBe(EngineErrorCode.STATE_STACK_NOT_FOUND);
        expect(error.message).toBe('Stack not found');
        expect(error.context.position).toBe('3,4');
        expect(error.domain).toBe('State');
      });

      it('should create InvalidState for region not found', () => {
        const error = entityNotFound('region', { regionId: 'region_1' });

        expect(error.code).toBe(EngineErrorCode.STATE_REGION_NOT_FOUND);
        expect(error.message).toBe('Region not found');
        expect(error.context.regionId).toBe('region_1');
      });

      it('should use default empty context', () => {
        const error = entityNotFound('player');

        expect(error.context).toEqual({});
      });
    });

    describe('moveMissingField', () => {
      it('should create MoveRequirementError for missing from field', () => {
        const error = moveMissingField('from', 'movement', { attemptedMove: 'abc' });

        expect(error).toBeInstanceOf(MoveRequirementError);
        expect(error.code).toBe(EngineErrorCode.MOVE_FROM_REQUIRED);
        expect(error.message).toBe('Move.from is required for movement moves');
        expect(error.context.fieldName).toBe('from');
        expect(error.context.moveType).toBe('movement');
        expect(error.context.attemptedMove).toBe('abc');
        expect(error.domain).toBe('Move');
      });

      it('should create MoveRequirementError for missing captureTarget field', () => {
        const error = moveMissingField('captureTarget', 'capture_overtake');

        expect(error.code).toBe(EngineErrorCode.MOVE_CAPTURE_TARGET_REQUIRED);
        expect(error.message).toBe('Move.captureTarget is required for capture_overtake moves');
        expect(error.context.fieldName).toBe('captureTarget');
        expect(error.context.moveType).toBe('capture_overtake');
      });

      it('should handle unknown field gracefully', () => {
        // Casting to test fallback behavior
        const error = moveMissingField('unknownField' as 'from', 'some_move');

        // Falls back to MOVE_FROM_REQUIRED
        expect(error.code).toBe(EngineErrorCode.MOVE_FROM_REQUIRED);
        expect(error.context.fieldName).toBe('unknownField');
      });
    });
  });

  describe('ERROR_CATEGORY_DESCRIPTIONS', () => {
    it('should have descriptions for all known prefixes', () => {
      expect(ERROR_CATEGORY_DESCRIPTIONS['RULES_']).toBeDefined();
      expect(ERROR_CATEGORY_DESCRIPTIONS['STATE_']).toBeDefined();
      expect(ERROR_CATEGORY_DESCRIPTIONS['BOARD_']).toBeDefined();
      expect(ERROR_CATEGORY_DESCRIPTIONS['MOVE_']).toBeDefined();
      expect(ERROR_CATEGORY_DESCRIPTIONS['FSM_']).toBeDefined();
      expect(ERROR_CATEGORY_DESCRIPTIONS['INTERNAL_']).toBeDefined();
    });
  });

  describe('EngineErrorCode enum', () => {
    it('should contain all expected error codes', () => {
      // Rules codes
      expect(EngineErrorCode.RULES_LINE_REWARD_REQUIRED).toBe('RULES_LINE_REWARD_REQUIRED');
      expect(EngineErrorCode.RULES_MISSING_COLLAPSE_POSITIONS).toBe(
        'RULES_MISSING_COLLAPSE_POSITIONS'
      );
      expect(EngineErrorCode.RULES_PHASE_MOVE_MISMATCH).toBe('RULES_PHASE_MOVE_MISMATCH');
      expect(EngineErrorCode.RULES_INVALID_RECOVERY_SLIDE).toBe('RULES_INVALID_RECOVERY_SLIDE');
      expect(EngineErrorCode.RULES_RECOVERY_LINE_INVALID).toBe('RULES_RECOVERY_LINE_INVALID');

      // State codes
      expect(EngineErrorCode.STATE_PLAYER_NOT_FOUND).toBe('STATE_PLAYER_NOT_FOUND');
      expect(EngineErrorCode.STATE_STACK_NOT_FOUND).toBe('STATE_STACK_NOT_FOUND');
      expect(EngineErrorCode.STATE_REGION_NOT_FOUND).toBe('STATE_REGION_NOT_FOUND');
      expect(EngineErrorCode.STATE_CAPTURE_STACKS_MISSING).toBe('STATE_CAPTURE_STACKS_MISSING');

      // Board codes
      expect(EngineErrorCode.BOARD_UNKNOWN_TYPE).toBe('BOARD_UNKNOWN_TYPE');
      expect(EngineErrorCode.BOARD_INVALID_POSITION).toBe('BOARD_INVALID_POSITION');

      // Move codes
      expect(EngineErrorCode.MOVE_FROM_REQUIRED).toBe('MOVE_FROM_REQUIRED');
      expect(EngineErrorCode.MOVE_CAPTURE_TARGET_REQUIRED).toBe('MOVE_CAPTURE_TARGET_REQUIRED');
      expect(EngineErrorCode.MOVE_UNKNOWN_ACTION_TYPE).toBe('MOVE_UNKNOWN_ACTION_TYPE');
      expect(EngineErrorCode.MOVE_WRONG_TYPE).toBe('MOVE_WRONG_TYPE');

      // FSM codes
      expect(EngineErrorCode.FSM_INVALID_TRANSITION).toBe('FSM_INVALID_TRANSITION');

      // Internal codes
      expect(EngineErrorCode.INTERNAL_ASSERTION_FAILED).toBe('INTERNAL_ASSERTION_FAILED');
      expect(EngineErrorCode.INTERNAL_NOT_IMPLEMENTED).toBe('INTERNAL_NOT_IMPLEMENTED');
    });
  });
});
