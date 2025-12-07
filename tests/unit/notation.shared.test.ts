/**
 * Unit tests for notation.ts
 *
 * Tests for move notation formatting helpers.
 */

import { formatPosition, formatMove, formatMoveList } from '../../src/shared/engine/notation';
import { pos } from '../utils/fixtures';
import type { Move, Position } from '../../src/shared/types/game';

describe('notation', () => {
  describe('formatPosition', () => {
    describe('square boards', () => {
      it('formats (0,0) as a1 for square8', () => {
        expect(formatPosition(pos(0, 0), { boardType: 'square8' })).toBe('a1');
      });

      it('formats (1,0) as b1 for square8', () => {
        expect(formatPosition(pos(1, 0), { boardType: 'square8' })).toBe('b1');
      });

      it('formats (0,1) as a2 for square8', () => {
        expect(formatPosition(pos(0, 1), { boardType: 'square8' })).toBe('a2');
      });

      it('formats (7,7) as h8 for square8', () => {
        expect(formatPosition(pos(7, 7), { boardType: 'square8' })).toBe('h8');
      });

      it('formats (3,4) as d5 for square8', () => {
        expect(formatPosition(pos(3, 4), { boardType: 'square8' })).toBe('d5');
      });

      it('uses default boardType square8 when not specified', () => {
        expect(formatPosition(pos(0, 0))).toBe('a1');
      });

      it('formats square19 positions correctly', () => {
        expect(formatPosition(pos(0, 0), { boardType: 'square19' })).toBe('a1');
        expect(formatPosition(pos(18, 18), { boardType: 'square19' })).toBe('s19');
      });
    });

    describe('hexagonal boards', () => {
      it('formats hex center (0,0,0) correctly', () => {
        const result = formatPosition({ x: 0, y: 0, z: 0 }, { boardType: 'hexagonal' });
        // Hex uses q (x) as rank, r (y) as file (radius=12)
        // For center: rank = radius - 0 + 1 = 13, file = 'a' + (0 + 12) = 'm'
        expect(result).toBe('m13');
      });

      it('formats hex position with positive coordinates', () => {
        const result = formatPosition({ x: 1, y: 1, z: -2 }, { boardType: 'hexagonal' });
        // rank = 12 - 1 + 1 = 12, file = 'a' + (1 + 12) = 'n'
        expect(result).toBe('n12');
      });

      it('formats hex position with negative coordinates', () => {
        const result = formatPosition({ x: -1, y: -1, z: 2 }, { boardType: 'hexagonal' });
        // rank = 12 - (-1) + 1 = 14, file = 'a' + (-1 + 12) = 'l'
        expect(result).toBe('l14');
      });
    });

    describe('raw tuple fallback', () => {
      it('uses raw tuple for negative coordinates on square board', () => {
        // Square boards with negative coords fall back to tuple
        const result = formatPosition({ x: -1, y: 2, z: undefined }, { boardType: 'square8' });
        // Since x < 0, falls through to tuple format
        expect(result).toBe('(-1,2,-1)'); // z = -x - y = -(-1) - 2 = -1
      });

      it('calculates z when not provided', () => {
        const result = formatPosition({ x: 1, y: 2 }, { boardType: 'square8' });
        // For square board with positive coords, uses algebraic
        expect(result).toBe('b3');
      });
    });
  });

  describe('formatMove', () => {
    describe('place_ring', () => {
      it('formats simple ring placement', () => {
        const move: Move = {
          type: 'place_ring',
          player: 1,
          to: pos(2, 3),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: R c4');
      });

      it('formats ring placement with count > 1', () => {
        const move: Move = {
          type: 'place_ring',
          player: 2,
          to: pos(4, 5),
          placementCount: 3,
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: R e6 x3');
      });

      it('formats placement without to as ?', () => {
        const move: Move = {
          type: 'place_ring',
          player: 1,
        } as Move;
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: R ?');
      });
    });

    describe('move_ring / move_stack', () => {
      it('formats move_ring with from and to', () => {
        const move: Move = {
          type: 'move_ring',
          player: 1,
          from: pos(1, 1),
          to: pos(3, 1),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: M b2→d2');
      });

      it('formats move_stack with from and to', () => {
        const move: Move = {
          type: 'move_stack',
          player: 2,
          from: pos(0, 0),
          to: pos(0, 4),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: M a1→a5');
      });

      it('formats movement with only to (no from)', () => {
        const move: Move = {
          type: 'move_ring',
          player: 1,
          to: pos(5, 5),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: M f6');
      });

      it('formats movement without from or to', () => {
        const move: Move = {
          type: 'move_ring',
          player: 1,
        } as Move;
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: M');
      });
    });

    describe('overtaking_capture', () => {
      it('formats capture with from, target, and to', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 1,
          from: pos(2, 2),
          captureTarget: pos(3, 2),
          to: pos(4, 2),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: C c3×d3→e3');
      });

      it('formats capture with from and to but no target', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 2,
          from: pos(1, 1),
          to: pos(3, 1),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: C b2→d2');
      });

      it('formats capture without positions', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 1,
        } as Move;
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: C');
      });
    });

    describe('other move types', () => {
      it('formats line_formation', () => {
        const move: Move = {
          type: 'line_formation',
          player: 1,
          from: pos(0, 0),
          to: pos(4, 0),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: L a1 →e1');
      });

      it('formats territory_claim', () => {
        const move: Move = {
          type: 'territory_claim',
          player: 2,
          to: pos(3, 3),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P2: T →d4');
      });

      it('formats skip_placement', () => {
        const move: Move = {
          type: 'skip_placement',
          player: 1,
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: S');
      });

      it('formats unknown move type as-is', () => {
        const move = {
          type: 'unknown_type',
          player: 3,
        } as unknown as Move;
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P3: unknown_type');
      });

      it('formats move with only from position', () => {
        const move: Move = {
          type: 'line_formation',
          player: 1,
          from: pos(2, 2),
        };
        const result = formatMove(move, { boardType: 'square8' });
        expect(result).toBe('P1: L c3');
      });
    });

    describe('default options', () => {
      it('uses default boardType when options not provided', () => {
        const move: Move = {
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
        };
        const result = formatMove(move);
        expect(result).toBe('P1: R a1');
      });
    });
  });

  describe('formatMoveList', () => {
    it('formats empty list', () => {
      const result = formatMoveList([]);
      expect(result).toEqual([]);
    });

    it('formats single move with index', () => {
      const moves: Move[] = [{ type: 'place_ring', player: 1, to: pos(3, 3) }];
      const result = formatMoveList(moves, { boardType: 'square8' });
      expect(result).toEqual(['1. P1: R d4']);
    });

    it('formats multiple moves with sequential indices', () => {
      const moves: Move[] = [
        { type: 'place_ring', player: 1, to: pos(0, 0) },
        { type: 'place_ring', player: 2, to: pos(7, 7) },
        { type: 'move_ring', player: 1, from: pos(0, 0), to: pos(0, 4) },
      ];
      const result = formatMoveList(moves, { boardType: 'square8' });
      expect(result).toEqual(['1. P1: R a1', '2. P2: R h8', '3. P1: M a1→a5']);
    });

    it('uses default options when not provided', () => {
      const moves: Move[] = [{ type: 'place_ring', player: 1, to: pos(1, 1) }];
      const result = formatMoveList(moves);
      expect(result).toEqual(['1. P1: R b2']);
    });
  });
});
