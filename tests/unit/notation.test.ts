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

    it('falls back to raw tuple for negative or hex coordinates', () => {
      expect(formatPosition(pos(-1, 0), { boardType })).toBe('(-1,0,1)');
      expect(formatPosition({ x: 1, y: -1, z: 0 }, { boardType: 'hexagonal' })).toBe(
        '(1,-1,0)'
      );
    });
  });

  describe('formatMove', () => {
    const baseMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'> = {
      type: 'place_ring',
      player: 1,
      to: pos(0, 0)
    } as any;

    it('formats ring placements with player prefix and destination', () => {
      const move: Move = {
        ...(baseMove as any),
        id: '',
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1
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
        moveNumber: 1
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
        moveNumber: 5
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
        moveNumber: 7
      } as Move;
      expect(formatMove(move, { boardType })).toBe('P3: C d4×e5→f6');
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
          moveNumber: 1
        } as Move,
        {
          id: '',
          type: 'move_stack',
          player: 2,
          from: pos(1, 1),
          to: pos(1, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 2
        } as Move
      ];

      const lines = formatMoveList(moves, { boardType });
      expect(lines).toEqual(['1. P1: R a1', '2. P2: M b2→b4']);
    });
  });
});
