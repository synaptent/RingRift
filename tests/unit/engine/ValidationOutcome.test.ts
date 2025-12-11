/**
 * Tests for the unified ValidationOutcome type and helpers.
 *
 * These tests verify the new validation infrastructure that provides
 * consistent error handling across validators.
 */

import {
  ValidationErrorCode,
  ValidationOutcome,
  isValidOutcome,
  validOutcome,
  invalidOutcome,
} from '../../../src/shared/engine';

describe('ValidationOutcome', () => {
  describe('validOutcome helper', () => {
    it('creates a successful outcome with data', () => {
      const result = validOutcome({ position: { x: 3, y: 3 }, count: 2 });

      expect(result.valid).toBe(true);
      expect(isValidOutcome(result)).toBe(true);
      if (isValidOutcome(result)) {
        expect(result.data.position).toEqual({ x: 3, y: 3 });
        expect(result.data.count).toBe(2);
      }
    });

    it('works with void data', () => {
      const result = validOutcome(undefined);

      expect(result.valid).toBe(true);
      expect(isValidOutcome(result)).toBe(true);
    });
  });

  describe('invalidOutcome helper', () => {
    it('creates a failed outcome with code and reason', () => {
      const result = invalidOutcome(
        ValidationErrorCode.PLACEMENT_DEAD_PLACEMENT,
        'Placement would leave stack with no legal moves'
      );

      expect(result.valid).toBe(false);
      expect(isValidOutcome(result)).toBe(false);
      if (!isValidOutcome(result)) {
        expect(result.code).toBe(ValidationErrorCode.PLACEMENT_DEAD_PLACEMENT);
        expect(result.reason).toBe('Placement would leave stack with no legal moves');
        expect(result.context).toBeUndefined();
      }
    });

    it('includes optional context', () => {
      const result = invalidOutcome(
        ValidationErrorCode.MOVEMENT_MUST_MOVE_FROM_STACK,
        'Must move from the placed stack',
        { requiredStack: '3,3', attemptedStack: '5,5' }
      );

      expect(result.valid).toBe(false);
      if (!isValidOutcome(result)) {
        expect(result.context).toEqual({
          requiredStack: '3,3',
          attemptedStack: '5,5',
        });
      }
    });
  });

  describe('isValidOutcome type guard', () => {
    it('narrows type for successful outcomes', () => {
      const outcome: ValidationOutcome<{ value: number }> = validOutcome({ value: 42 });

      if (isValidOutcome(outcome)) {
        // TypeScript should know outcome.data exists here
        expect(outcome.data.value).toBe(42);
      } else {
        fail('Expected valid outcome');
      }
    });

    it('narrows type for failed outcomes', () => {
      const outcome: ValidationOutcome<{ value: number }> = invalidOutcome(
        ValidationErrorCode.GENERAL_INVALID_PLAYER,
        'Invalid player'
      );

      if (!isValidOutcome(outcome)) {
        // TypeScript should know outcome.code and outcome.reason exist here
        expect(outcome.code).toBe(ValidationErrorCode.GENERAL_INVALID_PLAYER);
        expect(outcome.reason).toBe('Invalid player');
      } else {
        fail('Expected invalid outcome');
      }
    });
  });

  describe('ValidationErrorCode enum', () => {
    it('has distinct values for each error type', () => {
      // Verify some key codes exist and are distinct
      const codes = [
        ValidationErrorCode.GENERAL_INVALID_PLAYER,
        ValidationErrorCode.PLACEMENT_DEAD_PLACEMENT,
        ValidationErrorCode.MOVEMENT_MUST_MOVE_FROM_STACK,
        ValidationErrorCode.CAPTURE_CHAIN_REQUIRED,
        ValidationErrorCode.PHASE_INVALID_MOVE_TYPE,
      ];

      const uniqueCodes = new Set(codes);
      expect(uniqueCodes.size).toBe(codes.length);
    });

    it('follows naming convention DOMAIN_SPECIFIC_ERROR', () => {
      // All codes should match the pattern
      const allCodes = Object.values(ValidationErrorCode);

      for (const code of allCodes) {
        expect(code).toMatch(/^[A-Z]+_[A-Z_]+$/);
      }
    });
  });

  describe('usage example: validator function', () => {
    // Example of how a new validator would use ValidationOutcome
    interface PlacementData {
      position: { x: number; y: number };
      count: number;
    }

    function validatePlacementExample(
      hasRings: boolean,
      isValidPos: boolean
    ): ValidationOutcome<PlacementData> {
      if (!hasRings) {
        return invalidOutcome(
          ValidationErrorCode.PLACEMENT_NO_RINGS_IN_HAND,
          'Player has no rings in hand'
        );
      }

      if (!isValidPos) {
        return invalidOutcome(
          ValidationErrorCode.PLACEMENT_INVALID_POSITION,
          'Position is not valid for placement',
          { attempted: { x: -1, y: -1 } }
        );
      }

      return validOutcome({ position: { x: 3, y: 3 }, count: 1 });
    }

    it('returns success when validation passes', () => {
      const result = validatePlacementExample(true, true);

      expect(result.valid).toBe(true);
      if (isValidOutcome(result)) {
        expect(result.data.position).toEqual({ x: 3, y: 3 });
      }
    });

    it('returns failure with code when no rings', () => {
      const result = validatePlacementExample(false, true);

      expect(result.valid).toBe(false);
      if (!isValidOutcome(result)) {
        expect(result.code).toBe(ValidationErrorCode.PLACEMENT_NO_RINGS_IN_HAND);
      }
    });

    it('returns failure with context when invalid position', () => {
      const result = validatePlacementExample(true, false);

      expect(result.valid).toBe(false);
      if (!isValidOutcome(result)) {
        expect(result.code).toBe(ValidationErrorCode.PLACEMENT_INVALID_POSITION);
        expect(result.context?.attempted).toEqual({ x: -1, y: -1 });
      }
    });
  });
});
