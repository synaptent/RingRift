/**
 * notation.branchCoverage.test.ts
 *
 * Branch coverage tests for notation.ts targeting uncovered branches:
 * - formatPosition: square board, hex board, negative coordinates, z coordinate
 * - formatMoveType: all move types
 * - formatMove: place_ring, move_stack, move_stack, overtaking_capture, fallback
 * - formatMoveList: basic functionality
 */

import {
  formatPosition,
  formatMove,
  formatMoveList,
  type MoveNotationOptions,
} from '../../src/shared/engine/notation';
import type { Position, Move } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number, z?: number): Position =>
  z !== undefined ? { x, y, z } : { x, y };

// Helper to create a basic Move object
function makeMove(overrides: Partial<Move>): Move {
  return {
    id: 'test-move',
    type: 'place_ring',
    player: 1,
    to: pos(0, 0),
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    ...overrides,
  } as Move;
}

describe('notation branch coverage', () => {
  describe('formatPosition', () => {
    describe('square board coordinates', () => {
      it('formats (0,0) as a1', () => {
        const result = formatPosition(pos(0, 0), { boardType: 'square8' });
        expect(result).toBe('a1');
      });

      it('formats (1,0) as b1', () => {
        const result = formatPosition(pos(1, 0), { boardType: 'square8' });
        expect(result).toBe('b1');
      });

      it('formats (0,1) as a2', () => {
        const result = formatPosition(pos(0, 1), { boardType: 'square8' });
        expect(result).toBe('a2');
      });

      it('formats (7,7) as h8', () => {
        const result = formatPosition(pos(7, 7), { boardType: 'square8' });
        expect(result).toBe('h8');
      });

      it('formats position on square19 board', () => {
        const result = formatPosition(pos(10, 10), { boardType: 'square19' });
        expect(result).toBe('k11');
      });

      it('defaults to square8 when no boardType specified', () => {
        const result = formatPosition(pos(2, 3));
        expect(result).toBe('c4');
      });
    });

    describe('hexagonal board coordinates', () => {
      it('formats hex position with algebraic-like notation', () => {
        const result = formatPosition(pos(0, 0, 0), { boardType: 'hexagonal' });
        // Hex board maps q (x) to rank and r (y) to file
        expect(result).toMatch(/^[a-z]+\d+$/);
      });

      it('formats central hex position', () => {
        const result = formatPosition(pos(0, 0, 0), { boardType: 'hexagonal' });
        // For hexagonal board with size 13 (radius 12):
        // rankNum = radius - pos.x + 1 = 12 - 0 + 1 = 13
        // fileCode = 'a' + (pos.y + radius) = 'a' + (0 + 12) = 'm'
        expect(result).toBe('m13');
      });

      it('formats negative hex position', () => {
        const result = formatPosition(pos(-5, 3, 2), { boardType: 'hexagonal' });
        // rankNum = 12 - (-5) + 1 = 18
        // fileCode = 'a' + (3 + 12) = 'p'
        expect(result).toBe('p18');
      });
    });

    describe('negative coordinates fallback', () => {
      it('uses tuple format for negative x on square board', () => {
        const result = formatPosition(pos(-1, 0), { boardType: 'square8' });
        expect(result).toMatch(/^\(-1,0,-?\d+\)$/);
      });

      it('uses tuple format for negative y on square board', () => {
        const result = formatPosition(pos(0, -1), { boardType: 'square8' });
        expect(result).toMatch(/^\(0,-1,-?\d+\)$/);
      });

      it('calculates z when not provided', () => {
        const result = formatPosition(pos(-1, -2), { boardType: 'square8' });
        // z = -x - y = -(-1) - (-2) = 1 + 2 = 3
        expect(result).toBe('(-1,-2,3)');
      });

      it('uses provided z coordinate', () => {
        const result = formatPosition(pos(-1, -2, 99), { boardType: 'square8' });
        expect(result).toBe('(-1,-2,99)');
      });
    });
  });

  describe('formatMoveType (via formatMove)', () => {
    it('formats place_ring as R', () => {
      const move = makeMove({ type: 'place_ring', to: pos(0, 0) });
      const result = formatMove(move);
      expect(result).toContain(' R ');
    });

    it('formats move_stack as M', () => {
      const move = makeMove({ type: 'move_stack', from: pos(0, 0), to: pos(1, 0) });
      const result = formatMove(move);
      expect(result).toContain(' M ');
    });

    it('formats move_stack as M', () => {
      const move = makeMove({ type: 'move_stack', from: pos(0, 0), to: pos(2, 0) });
      const result = formatMove(move);
      expect(result).toContain(' M ');
    });

    it('formats overtaking_capture as C', () => {
      const move = makeMove({
        type: 'overtaking_capture',
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      });
      const result = formatMove(move);
      expect(result).toContain(' C ');
    });

    it('formats line_formation as L', () => {
      const move = makeMove({ type: 'line_formation', to: pos(0, 0) });
      const result = formatMove(move);
      expect(result).toContain(' L ');
    });

    it('formats territory_claim as T', () => {
      const move = makeMove({ type: 'territory_claim', to: pos(0, 0) });
      const result = formatMove(move);
      expect(result).toContain(' T ');
    });

    it('formats skip_placement as S', () => {
      const move = makeMove({ type: 'skip_placement' });
      const result = formatMove(move);
      expect(result).toContain(' S ');
    });

    it('formats unknown type as-is', () => {
      const move = makeMove({ type: 'custom_type' as Move['type'] });
      const result = formatMove(move);
      expect(result).toContain('custom_type');
    });
  });

  describe('formatMove', () => {
    describe('place_ring moves', () => {
      it('formats basic placement', () => {
        const move = makeMove({ type: 'place_ring', player: 1, to: pos(2, 3) });
        const result = formatMove(move);
        expect(result).toBe('P1: R c4');
      });

      it('formats placement with count > 1', () => {
        const move = makeMove({ type: 'place_ring', player: 2, to: pos(0, 0), placementCount: 3 });
        const result = formatMove(move);
        expect(result).toBe('P2: R a1 x3');
      });

      it('formats placement with count = 1 (no suffix)', () => {
        const move = makeMove({ type: 'place_ring', player: 1, to: pos(0, 0), placementCount: 1 });
        const result = formatMove(move);
        expect(result).toBe('P1: R a1');
      });

      it('handles missing to position', () => {
        const move = makeMove({ type: 'place_ring', player: 1, to: undefined });
        const result = formatMove(move);
        expect(result).toBe('P1: R ?');
      });
    });

    describe('move_stack and move_stack moves', () => {
      it('formats move with from and to', () => {
        const move = makeMove({ type: 'move_stack', player: 1, from: pos(0, 0), to: pos(3, 0) });
        const result = formatMove(move);
        expect(result).toBe('P1: M a1→d1');
      });

      it('formats move with only to', () => {
        const move = makeMove({ type: 'move_stack', player: 2, from: undefined, to: pos(5, 5) });
        const result = formatMove(move);
        expect(result).toBe('P2: M f6');
      });

      it('formats move with neither from nor to', () => {
        const move = makeMove({ type: 'move_stack', player: 1, from: undefined, to: undefined });
        const result = formatMove(move);
        expect(result).toBe('P1: M');
      });
    });

    describe('overtaking_capture moves', () => {
      it('formats capture with all positions', () => {
        const move = makeMove({
          type: 'overtaking_capture',
          player: 1,
          from: pos(0, 0),
          captureTarget: pos(1, 0),
          to: pos(2, 0),
        });
        const result = formatMove(move);
        expect(result).toBe('P1: C a1×b1→c1');
      });

      it('formats capture with from and to but no target', () => {
        const move = makeMove({
          type: 'overtaking_capture',
          player: 2,
          from: pos(3, 3),
          captureTarget: undefined,
          to: pos(5, 5),
        });
        const result = formatMove(move);
        expect(result).toBe('P2: C d4→f6');
      });

      it('formats capture with no positions', () => {
        const move = makeMove({
          type: 'overtaking_capture',
          player: 1,
          from: undefined,
          captureTarget: undefined,
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: C');
      });
    });

    describe('fallback formatting', () => {
      it('formats line_formation with positions', () => {
        const move = makeMove({
          type: 'line_formation',
          player: 1,
          from: pos(1, 1),
          to: pos(4, 4),
        });
        const result = formatMove(move);
        expect(result).toBe('P1: L b2 →e5');
      });

      it('formats territory_claim with only to', () => {
        const move = makeMove({
          type: 'territory_claim',
          player: 2,
          from: undefined,
          to: pos(3, 3),
        });
        const result = formatMove(move);
        expect(result).toBe('P2: T →d4');
      });

      it('formats unknown type with from only', () => {
        const move = makeMove({
          type: 'some_type' as Move['type'],
          player: 1,
          from: pos(2, 2),
          to: undefined,
        });
        const result = formatMove(move);
        expect(result).toBe('P1: some_type c3');
      });
    });

    describe('with options', () => {
      it('uses specified boardType for formatting', () => {
        const move = makeMove({ type: 'place_ring', player: 1, to: pos(0, 0, 0) });
        const result = formatMove(move, { boardType: 'hexagonal' });
        expect(result).toContain('m13'); // Central hex position (0,0,0) cube coords
      });
    });
  });

  describe('formatMoveList', () => {
    it('formats empty list', () => {
      const result = formatMoveList([]);
      expect(result).toEqual([]);
    });

    it('formats single move with number', () => {
      const moves = [makeMove({ type: 'place_ring', player: 1, to: pos(0, 0) })];
      const result = formatMoveList(moves);
      expect(result).toEqual(['1. P1: R a1']);
    });

    it('formats multiple moves with numbers', () => {
      const moves = [
        makeMove({ type: 'place_ring', player: 1, to: pos(0, 0) }),
        makeMove({ type: 'place_ring', player: 2, to: pos(1, 1) }),
        makeMove({ type: 'move_stack', player: 1, from: pos(0, 0), to: pos(2, 0) }),
      ];
      const result = formatMoveList(moves);
      expect(result).toEqual(['1. P1: R a1', '2. P2: R b2', '3. P1: M a1→c1']);
    });

    it('passes options through to formatMove', () => {
      const moves = [makeMove({ type: 'place_ring', player: 1, to: pos(0, 0, 0) })];
      const result = formatMoveList(moves, { boardType: 'hexagonal' });
      expect(result[0]).toContain('m13'); // (0,0,0) cube coords -> m13
    });
  });
});
