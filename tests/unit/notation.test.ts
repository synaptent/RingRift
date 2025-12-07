import { BoardType, Move, Position } from '../../src/shared/types/game';
import { formatPosition, formatMove, formatMoveList } from '../../src/shared/engine/notation';

function pos(x: number, y: number, z?: number): Position {
  return z !== undefined ? { x, y, z } : { x, y };
}

describe('notation helpers', () => {
  const boardType: BoardType = 'square8';

  describe('formatPosition', () => {
    it('formats square board positions as algebraic coordinates', () => {
      expect(formatPosition(pos(0, 0), { boardType })).toBe('a1');
      expect(formatPosition(pos(1, 0), { boardType })).toBe('b1');
      expect(formatPosition(pos(0, 1), { boardType })).toBe('a2');
      expect(formatPosition(pos(7, 7), { boardType })).toBe('h8');
    });

    it('falls back to raw tuple for negative coordinates on square boards', () => {
      expect(formatPosition(pos(-1, 0), { boardType })).toBe('(-1,0,1)');
    });

    it('formats hex coordinates using algebraic notation', () => {
      // Hex board size 13 -> radius 12
      // q=1, r=-1 -> Rank = 12 - 1 + 1 = 12
      // File = 'a' + (-1 + 12) = 'l'
      // Expected: l12
      expect(formatPosition({ x: 1, y: -1, z: 0 }, { boardType: 'hexagonal' })).toBe('l12');
    });
  });

  describe('formatMove', () => {
    const baseMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'> = {
      type: 'place_ring',
      player: 1,
      to: pos(0, 0),
    } as any;

    it('formats ring placements with player prefix and destination', () => {
      const move: Move = {
        ...(baseMove as any),
        id: '',
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      expect(formatMove(move, { boardType })).toBe('P1: R a1');
    });

    it('includes placementCount when > 1', () => {
      const move: Move = {
        ...(baseMove as any),
        placementCount: 3,
        id: '',
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      expect(formatMove(move, { boardType })).toBe('P1: R a1 x3');
    });

    it('formats simple movements as M from→to', () => {
      const move: Move = {
        id: '',
        type: 'move_stack',
        player: 2,
        from: pos(2, 2),
        to: pos(2, 5),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 5,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P2: M c3→c6');
    });

    it('formats overtaking captures as C from×target→to when all positions are present', () => {
      const move: Move = {
        id: '',
        type: 'overtaking_capture',
        player: 3,
        from: pos(3, 3),
        captureTarget: pos(4, 4),
        to: pos(5, 5),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 7,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P3: C d4×e5→f6');
    });

    it('formats line_formation moves', () => {
      const move: Move = {
        id: '',
        type: 'line_formation',
        player: 1,
        to: pos(3, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 10,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: L →d4');
    });

    it('formats territory_claim moves', () => {
      const move: Move = {
        id: '',
        type: 'territory_claim',
        player: 2,
        to: pos(4, 4),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 11,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P2: T →e5');
    });

    it('formats skip_placement moves', () => {
      const move: Move = {
        id: '',
        type: 'skip_placement',
        player: 1,
        to: pos(0, 0),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 12,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: S →a1');
    });

    it('formats unknown move types using the type name', () => {
      const move: Move = {
        id: '',
        type: 'process_line' as any,
        player: 1,
        to: pos(2, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 13,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: process_line →c3');
    });

    it('formats move_ring without from position', () => {
      const move: Move = {
        id: '',
        type: 'move_ring',
        player: 1,
        to: pos(3, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 14,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: M d4');
    });

    it('formats move_stack without from or to position', () => {
      const move: Move = {
        id: '',
        type: 'move_stack',
        player: 2,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 15,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P2: M');
    });

    it('formats overtaking capture without captureTarget', () => {
      const move: Move = {
        id: '',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        to: pos(4, 4),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 16,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: C c3→e5');
    });

    it('formats overtaking capture without from or target positions', () => {
      const move: Move = {
        id: '',
        type: 'overtaking_capture',
        player: 1,
        to: pos(5, 5),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 17,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: C');
    });

    it('formats fallback move types with from position', () => {
      const move: Move = {
        id: '',
        type: 'continue_capture_segment' as any,
        player: 1,
        from: pos(1, 1),
        to: pos(3, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 18,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P1: continue_capture_segment b2 →d4');
    });

    it('formats fallback move types without from position', () => {
      const move: Move = {
        id: '',
        type: 'eliminate_rings_from_stack' as any,
        player: 2,
        to: pos(5, 5),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 19,
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P2: eliminate_rings_from_stack →f6');
    });
  });

  describe('formatMoveList', () => {
    it('renders a numbered move list', () => {
      const moves: Move[] = [
        {
          id: '',
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        } as Move,
        {
          id: '',
          type: 'move_stack',
          player: 2,
          from: pos(1, 1),
          to: pos(1, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 2,
        } as Move,
      ];

      const lines = formatMoveList(moves, { boardType });
      expect(lines).toEqual(['1. P1: R a1', '2. P2: M b2→b4']);
    });
  });
});
