/**
 * Phase Validation Tests
 *
 * Tests for the declarative phase-move validation matrix.
 * Verifies that move types are correctly mapped to phases.
 */

import {
  VALID_MOVES_BY_PHASE,
  ALWAYS_VALID_MOVES,
  isMoveValidInPhase,
  getValidMoveTypesForPhase,
  getPhasesForMoveType,
  getEliminationContextForPhase,
  canEliminateInPhase,
  type MoveType,
} from '../../../src/shared/engine/phaseValidation';
import type { GamePhase } from '../../../src/shared/types/game';

describe('phaseValidation', () => {
  // ===========================================================================
  // VALID_MOVES_BY_PHASE Matrix Tests
  // ===========================================================================
  describe('VALID_MOVES_BY_PHASE', () => {
    it('includes all GamePhase values', () => {
      const expectedPhases: GamePhase[] = [
        'ring_placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
        'game_over',
      ];

      for (const phase of expectedPhases) {
        expect(VALID_MOVES_BY_PHASE[phase]).toBeDefined();
      }
    });

    it('has empty array for game_over phase', () => {
      expect(VALID_MOVES_BY_PHASE.game_over).toEqual([]);
    });

    it('includes place_ring in ring_placement phase', () => {
      expect(VALID_MOVES_BY_PHASE.ring_placement).toContain('place_ring');
    });

    it('includes move_stack in movement phase', () => {
      expect(VALID_MOVES_BY_PHASE.movement).toContain('move_stack');
    });

    it('includes choose_territory_option in territory_processing phase', () => {
      expect(VALID_MOVES_BY_PHASE.territory_processing).toContain('choose_territory_option');
    });

    it('includes legacy process_territory_region in territory_processing phase', () => {
      expect(VALID_MOVES_BY_PHASE.territory_processing).toContain('process_territory_region');
    });

    it('includes eliminate_rings_from_stack in territory_processing phase', () => {
      expect(VALID_MOVES_BY_PHASE.territory_processing).toContain('eliminate_rings_from_stack');
    });

    it('includes recovery_slide in movement phase', () => {
      expect(VALID_MOVES_BY_PHASE.movement).toContain('recovery_slide');
    });

    it('includes skip_recovery in movement phase', () => {
      expect(VALID_MOVES_BY_PHASE.movement).toContain('skip_recovery');
    });

    it('includes swap_sides in ring_placement phase', () => {
      expect(VALID_MOVES_BY_PHASE.ring_placement).toContain('swap_sides');
    });
  });

  // ===========================================================================
  // isMoveValidInPhase Tests
  // ===========================================================================
  describe('isMoveValidInPhase', () => {
    it('returns true for place_ring in ring_placement', () => {
      expect(isMoveValidInPhase('place_ring', 'ring_placement')).toBe(true);
    });

    it('returns false for place_ring in movement', () => {
      expect(isMoveValidInPhase('place_ring', 'movement')).toBe(false);
    });

    it('returns true for move_stack in movement', () => {
      expect(isMoveValidInPhase('move_stack', 'movement')).toBe(true);
    });

    it('returns false for move_stack in ring_placement', () => {
      expect(isMoveValidInPhase('move_stack', 'ring_placement')).toBe(false);
    });

    it('returns true for resign in any phase (always valid)', () => {
      const phases: GamePhase[] = [
        'ring_placement',
        'movement',
        'capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
      ];

      for (const phase of phases) {
        expect(isMoveValidInPhase('resign', phase)).toBe(true);
      }
    });

    it('returns true for timeout in any phase (always valid)', () => {
      const phases: GamePhase[] = [
        'ring_placement',
        'movement',
        'line_processing',
        'territory_processing',
      ];

      for (const phase of phases) {
        expect(isMoveValidInPhase('timeout', phase)).toBe(true);
      }
    });

    it('returns false for any move in game_over', () => {
      // All moves except meta moves should be invalid in game_over
      expect(isMoveValidInPhase('place_ring', 'game_over')).toBe(false);
      expect(isMoveValidInPhase('move_stack', 'game_over')).toBe(false);
      expect(isMoveValidInPhase('process_territory_region', 'game_over')).toBe(false);
      // But meta moves are still valid
      expect(isMoveValidInPhase('resign', 'game_over')).toBe(true);
    });

    it('returns false for eliminate_rings_from_stack in line_processing', () => {
      expect(isMoveValidInPhase('eliminate_rings_from_stack', 'line_processing')).toBe(false);
    });

    it('returns true for eliminate_rings_from_stack in territory_processing', () => {
      expect(isMoveValidInPhase('eliminate_rings_from_stack', 'territory_processing')).toBe(true);
    });

    it('returns true for forced_elimination in forced_elimination phase', () => {
      expect(isMoveValidInPhase('forced_elimination', 'forced_elimination')).toBe(true);
    });

    it('returns false for forced_elimination in ring_placement', () => {
      expect(isMoveValidInPhase('forced_elimination', 'ring_placement')).toBe(false);
    });

    it('returns true for recovery_slide in movement', () => {
      expect(isMoveValidInPhase('recovery_slide', 'movement')).toBe(true);
    });

    it('returns true for skip_recovery in movement', () => {
      expect(isMoveValidInPhase('skip_recovery', 'movement')).toBe(true);
    });

    it('returns false for recovery_slide in ring_placement', () => {
      expect(isMoveValidInPhase('recovery_slide', 'ring_placement')).toBe(false);
    });

    it('returns true for swap_sides in ring_placement', () => {
      expect(isMoveValidInPhase('swap_sides', 'ring_placement')).toBe(true);
    });
  });

  // ===========================================================================
  // getValidMoveTypesForPhase Tests
  // ===========================================================================
  describe('getValidMoveTypesForPhase', () => {
    it('returns phase moves plus always-valid moves', () => {
      const moves = getValidMoveTypesForPhase('ring_placement');
      expect(moves).toContain('place_ring');
      expect(moves).toContain('resign');
      expect(moves).toContain('timeout');
    });

    it('returns only always-valid moves for game_over', () => {
      const moves = getValidMoveTypesForPhase('game_over');
      expect(moves.length).toBe(ALWAYS_VALID_MOVES.length);
      expect(moves).toContain('resign');
      expect(moves).toContain('timeout');
    });

    it('includes all line processing moves', () => {
      const moves = getValidMoveTypesForPhase('line_processing');
      expect(moves).toContain('process_line');
      expect(moves).toContain('choose_line_reward');
      expect(moves).toContain('no_line_action');
    });

    it('includes all territory processing moves', () => {
      const moves = getValidMoveTypesForPhase('territory_processing');
      expect(moves).toContain('choose_territory_option');
      expect(moves).toContain('process_territory_region');
      expect(moves).toContain('eliminate_rings_from_stack');
      expect(moves).toContain('no_territory_action');
      expect(moves).toContain('skip_territory_processing');
    });
  });

  // ===========================================================================
  // getPhasesForMoveType Tests
  // ===========================================================================
  describe('getPhasesForMoveType', () => {
    it('returns ring_placement for place_ring', () => {
      const phases = getPhasesForMoveType('place_ring');
      expect(phases).toContain('ring_placement');
      expect(phases.length).toBe(1);
    });

    it('returns movement for move_stack', () => {
      const phases = getPhasesForMoveType('move_stack');
      expect(phases).toContain('movement');
    });

    it('returns multiple phases for eliminate_rings_from_stack', () => {
      const phases = getPhasesForMoveType('eliminate_rings_from_stack');
      expect(phases).toContain('territory_processing');
      expect(phases.length).toBe(1);
    });

    it('returns all non-game_over phases for resign (always valid)', () => {
      const phases = getPhasesForMoveType('resign');
      expect(phases.length).toBe(7); // All phases except game_over
      expect(phases).not.toContain('game_over');
    });

    it('returns movement, capture, and chain_capture for continue_capture_segment', () => {
      const phases = getPhasesForMoveType('continue_capture_segment');
      expect(phases).toContain('movement');
      expect(phases).toContain('capture');
      expect(phases).toContain('chain_capture');
    });
  });

  // ===========================================================================
  // getEliminationContextForPhase Tests
  // ===========================================================================
  describe('getEliminationContextForPhase', () => {
    it('returns line for line_processing phase', () => {
      expect(getEliminationContextForPhase('line_processing')).toBe('line');
    });

    it('returns territory for territory_processing phase', () => {
      expect(getEliminationContextForPhase('territory_processing')).toBe('territory');
    });

    it('returns forced for forced_elimination phase', () => {
      expect(getEliminationContextForPhase('forced_elimination')).toBe('forced');
    });

    it('returns recovery for movement phase', () => {
      expect(getEliminationContextForPhase('movement')).toBe('recovery');
    });

    it('returns null for ring_placement phase', () => {
      expect(getEliminationContextForPhase('ring_placement')).toBe(null);
    });

    it('returns null for capture phase', () => {
      expect(getEliminationContextForPhase('capture')).toBe(null);
    });

    it('returns null for game_over phase', () => {
      expect(getEliminationContextForPhase('game_over')).toBe(null);
    });
  });

  // ===========================================================================
  // canEliminateInPhase Tests
  // ===========================================================================
  describe('canEliminateInPhase', () => {
    it('returns true for line_processing', () => {
      expect(canEliminateInPhase('line_processing')).toBe(true);
    });

    it('returns true for territory_processing', () => {
      expect(canEliminateInPhase('territory_processing')).toBe(true);
    });

    it('returns true for forced_elimination', () => {
      expect(canEliminateInPhase('forced_elimination')).toBe(true);
    });

    it('returns false for ring_placement', () => {
      expect(canEliminateInPhase('ring_placement')).toBe(false);
    });

    it('returns false for movement', () => {
      expect(canEliminateInPhase('movement')).toBe(false);
    });

    it('returns false for capture', () => {
      expect(canEliminateInPhase('capture')).toBe(false);
    });

    it('returns false for game_over', () => {
      expect(canEliminateInPhase('game_over')).toBe(false);
    });
  });

  // ===========================================================================
  // Edge Cases and Consistency Tests
  // ===========================================================================
  describe('consistency', () => {
    it('ALWAYS_VALID_MOVES contains only resign and timeout', () => {
      expect(ALWAYS_VALID_MOVES).toContain('resign');
      expect(ALWAYS_VALID_MOVES).toContain('timeout');
      expect(ALWAYS_VALID_MOVES.length).toBe(2);
    });

    it('isMoveValidInPhase is consistent with VALID_MOVES_BY_PHASE', () => {
      // For each phase/move combination in the matrix, isMoveValidInPhase should return true
      for (const [phase, moves] of Object.entries(VALID_MOVES_BY_PHASE)) {
        for (const move of moves) {
          expect(isMoveValidInPhase(move as MoveType, phase as GamePhase)).toBe(true);
        }
      }
    });

    it('getPhasesForMoveType is inverse of VALID_MOVES_BY_PHASE', () => {
      // For a move that appears in a phase, getPhasesForMoveType should include that phase
      for (const [phase, moves] of Object.entries(VALID_MOVES_BY_PHASE)) {
        for (const move of moves) {
          const phases = getPhasesForMoveType(move as MoveType);
          expect(phases).toContain(phase);
        }
      }
    });

    it('every phase has defined move types (even if empty)', () => {
      const allPhases: GamePhase[] = [
        'ring_placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
        'game_over',
      ];

      for (const phase of allPhases) {
        expect(Array.isArray(VALID_MOVES_BY_PHASE[phase])).toBe(true);
      }
    });
  });
});
