/**
 * Test suite for src/shared/engine/notation.ts
 *
 * Tests move notation formatting for debugging, logging, and display.
 */

import {
  formatPosition,
  formatMove,
  formatMoveList,
  type MoveNotationOptions,
} from '../../../src/shared/engine/notation';
import type { Move, Position, BoardType } from '../../../src/shared/types/game';

describe('notation', () => {
  // Helper to create a test Move
  function createTestMove(overrides: Partial<Move> = {}): Move {
    return {
      id: 'test-move-1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: 1,
      ...overrides,
    };
  }

  describe('formatPosition', () => {
    describe('square board positions', () => {
      it('should format (0,0) as a1 for square8', () => {
        expect(formatPosition({ x: 0, y: 0 }, { boardType: 'square8' })).toBe('a1');
      });

      it('should format corner positions correctly', () => {
        expect(formatPosition({ x: 0, y: 7 }, { boardType: 'square8' })).toBe('a8');
        expect(formatPosition({ x: 7, y: 0 }, { boardType: 'square8' })).toBe('h1');
        expect(formatPosition({ x: 7, y: 7 }, { boardType: 'square8' })).toBe('h8');
      });

      it('should format middle positions correctly', () => {
        expect(formatPosition({ x: 3, y: 4 }, { boardType: 'square8' })).toBe('d5');
        expect(formatPosition({ x: 4, y: 3 }, { boardType: 'square8' })).toBe('e4');
      });

      it('should format square19 positions', () => {
        expect(formatPosition({ x: 0, y: 0 }, { boardType: 'square19' })).toBe('a1');
        expect(formatPosition({ x: 18, y: 18 }, { boardType: 'square19' })).toBe('s19');
      });

      it('should default to square8 when no boardType provided', () => {
        expect(formatPosition({ x: 0, y: 0 })).toBe('a1');
        expect(formatPosition({ x: 3, y: 4 })).toBe('d5');
      });

      it('should handle squareRankFromBottom option', () => {
        // With squareRankFromBottom: rank = boardSize - y
        // For square8 (size 8): y=7 → rank 1, y=0 → rank 8
        expect(
          formatPosition(
            { x: 0, y: 0 },
            {
              boardType: 'square8',
              squareRankFromBottom: true,
            }
          )
        ).toBe('a8');

        expect(
          formatPosition(
            { x: 0, y: 7 },
            {
              boardType: 'square8',
              squareRankFromBottom: true,
            }
          )
        ).toBe('a1');

        expect(
          formatPosition(
            { x: 3, y: 4 },
            {
              boardType: 'square8',
              squareRankFromBottom: true,
            }
          )
        ).toBe('d4'); // 8 - 4 = 4
      });

      it('should respect boardSizeOverride with squareRankFromBottom', () => {
        expect(
          formatPosition(
            { x: 0, y: 0 },
            {
              boardType: 'square8',
              squareRankFromBottom: true,
              boardSizeOverride: 10,
            }
          )
        ).toBe('a10'); // 10 - 0 = 10
      });
    });

    describe('hexagonal board positions', () => {
      it('should format hex positions using algebraic notation', () => {
        // Hexagonal board uses algebraic notation for file + rank
        const pos: Position = { x: 0, y: 0 };
        const result = formatPosition(pos, { boardType: 'hexagonal' });
        // Just verify it produces valid algebraic notation
        expect(result).toMatch(/^[a-z]\d+$/);
      });

      it('should format hex center correctly', () => {
        const pos: Position = { x: 0, y: 0, z: 0 };
        const result = formatPosition(pos, { boardType: 'hexagonal' });
        // Actual output for center is m13 based on the hex radius calculation
        expect(result).toBe('m13');
      });

      it('should format negative x hex positions', () => {
        const pos: Position = { x: -3, y: 2, z: 1 };
        const result = formatPosition(pos, { boardType: 'hexagonal' });
        // Hex board uses algebraic-like notation
        // Just verify it matches the expected format (letter + number)
        expect(result).toMatch(/^[a-z]\d+$/);
      });

      it('should format various hex positions with expected format', () => {
        // Just validate the format is consistent (file + rank)
        const result = formatPosition({ x: -3, y: 2 }, { boardType: 'hexagonal' });
        // Larger hex boards use wider alphabet range
        expect(result).toMatch(/^[a-z]\d+$/);
      });
    });

    describe('fallback tuple notation', () => {
      it('should use tuple notation for negative coordinates on square boards', () => {
        const pos: Position = { x: -1, y: 0 };
        const result = formatPosition(pos, { boardType: 'square8' });
        // Negative x triggers fallback
        expect(result).toBe('(-1,0,1)');
      });

      it('should include z coordinate in tuple', () => {
        const pos: Position = { x: -1, y: 2, z: -1 };
        const result = formatPosition(pos, { boardType: 'square8' });
        expect(result).toBe('(-1,2,-1)');
      });

      it('should calculate z from x and y when not provided', () => {
        const pos: Position = { x: -1, y: 2 };
        const result = formatPosition(pos, { boardType: 'square8' });
        // z = -x - y = -(-1) - 2 = -1
        expect(result).toBe('(-1,2,-1)');
      });
    });
  });

  describe('formatMove', () => {
    describe('place_ring moves', () => {
      it('should format basic placement', () => {
        const move = createTestMove({
          type: 'place_ring',
          player: 1,
          to: { x: 2, y: 3 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: R c4');
      });

      it('should format placement with count', () => {
        const move = createTestMove({
          type: 'place_ring',
          player: 2,
          to: { x: 5, y: 6 },
          placementCount: 3,
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: R f7 x3');
      });

      it('should handle missing to position', () => {
        const move = createTestMove({
          type: 'place_ring',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: R ?');
      });
    });

    describe('move_stack moves', () => {
      it('should format stack movement with from and to', () => {
        const move = createTestMove({
          type: 'move_stack',
          player: 1,
          from: { x: 2, y: 2 },
          to: { x: 2, y: 6 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: M c3→c7');
      });

      it('should format stack movement with only to', () => {
        const move = createTestMove({
          type: 'move_stack',
          player: 1,
          to: { x: 3, y: 4 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: M d5');
      });

      it('should handle move_stack without positions', () => {
        const move = createTestMove({
          type: 'move_stack',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: M');
      });
    });

    describe('overtaking_capture moves', () => {
      it('should format capture with from, target, and to', () => {
        const move = createTestMove({
          type: 'overtaking_capture',
          player: 2,
          from: { x: 3, y: 4 },
          captureTarget: { x: 4, y: 5 },
          to: { x: 5, y: 6 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: C d5×e6→f7');
      });

      it('should format capture with from and to (no target)', () => {
        const move = createTestMove({
          type: 'overtaking_capture',
          player: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 2 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: C a1→c3');
      });

      it('should handle capture without positions', () => {
        const move = createTestMove({
          type: 'overtaking_capture',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: C');
      });
    });

    describe('skip_placement moves', () => {
      it('should format skip placement', () => {
        const move = createTestMove({
          type: 'skip_placement',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        // Skip uses fallback path since it's not explicitly handled
        expect(result).toContain('P1:');
        expect(result).toContain('S');
      });
    });

    describe('recovery_slide moves', () => {
      it('should format recovery slide with from and to', () => {
        const move = createTestMove({
          type: 'recovery_slide',
          player: 1,
          from: { x: 1, y: 1 },
          to: { x: 3, y: 3 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: Rv b2→d4');
      });

      it('should format recovery slide with option', () => {
        const move = createTestMove({
          type: 'recovery_slide',
          player: 1,
          from: { x: 1, y: 1 },
          to: { x: 3, y: 3 },
          recoveryOption: 2,
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: Rv b2→d4 [opt2]');
      });

      it('should format recovery slide with only to position', () => {
        const move = createTestMove({
          type: 'recovery_slide',
          player: 1,
          to: { x: 2, y: 2 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: Rv c3');
      });

      it('should handle recovery slide without positions', () => {
        const move = createTestMove({
          type: 'recovery_slide',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: Rv');
      });
    });

    describe('skip_recovery moves', () => {
      it('should format skip recovery', () => {
        const move = createTestMove({
          type: 'skip_recovery',
          player: 1,
          to: undefined,
        });
        const result = formatMove(move);
        // Skip recovery uses fallback path
        expect(result).toContain('P1:');
        expect(result).toContain('RvS');
      });
    });

    describe('other/unknown move types', () => {
      it('should use move type as symbol for unknown types', () => {
        const move = createTestMove({
          type: 'process_line',
          player: 1,
          to: { x: 2, y: 3 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: process_line →c4');
      });

      it('should handle move with from position in fallback', () => {
        const move = createTestMove({
          type: 'choose_line_option',
          player: 2,
          from: { x: 0, y: 0 },
          to: { x: 5, y: 5 },
        });
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: choose_line_option a1 →f6');
      });
    });
  });

  describe('formatMoveList', () => {
    it('should format a list of moves with numbering', () => {
      const moves: Move[] = [
        createTestMove({ type: 'place_ring', player: 1, to: { x: 0, y: 0 }, moveNumber: 1 }),
        createTestMove({ type: 'place_ring', player: 2, to: { x: 7, y: 7 }, moveNumber: 2 }),
        createTestMove({
          type: 'move_stack',
          player: 1,
          from: { x: 0, y: 0 },
          to: { x: 3, y: 3 },
          moveNumber: 3,
        }),
      ];

      const result = formatMoveList(moves, { boardType: 'square8' });

      expect(result).toHaveLength(3);
      expect(result[0]).toBe('1. P1: R a1');
      expect(result[1]).toBe('2. P2: R h8');
      expect(result[2]).toBe('3. P1: M a1→d4');
    });

    it('should handle empty list', () => {
      const result = formatMoveList([]);
      expect(result).toEqual([]);
    });

    it('should pass options through', () => {
      const moves: Move[] = [
        createTestMove({
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
        }),
      ];

      const result = formatMoveList(moves, {
        boardType: 'square8',
        squareRankFromBottom: true,
      });

      expect(result[0]).toBe('1. P1: R a8'); // With squareRankFromBottom
    });
  });
});
