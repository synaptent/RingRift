/**
 * Validator utils branch coverage tests
 * Tests for src/shared/engine/validators/utils.ts
 *
 * Focus: isValidPosition for hexagonal and square boards.
 */

import { isValidPosition } from '../../../src/shared/engine/validators/utils';
import type { Position, BoardType } from '../../../src/shared/types/game';

describe('isValidPosition', () => {
  describe('square board validation', () => {
    const boardType: BoardType = 'square8';
    const boardSize = 8;

    it('returns true for position at origin (0,0)', () => {
      const pos: Position = { x: 0, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for position at max corner (7,7)', () => {
      const pos: Position = { x: 7, y: 7 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for middle position (4,4)', () => {
      const pos: Position = { x: 4, y: 4 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns false for x below bounds (-1,0)', () => {
      const pos: Position = { x: -1, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for y below bounds (0,-1)', () => {
      const pos: Position = { x: 0, y: -1 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for x at board size (8,0)', () => {
      const pos: Position = { x: 8, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for y at board size (0,8)', () => {
      const pos: Position = { x: 0, y: 8 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for both coordinates out of bounds', () => {
      const pos: Position = { x: 10, y: 10 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });
  });

  describe('square19 board validation', () => {
    const boardType: BoardType = 'square19';
    const boardSize = 19;

    it('returns true for position at origin', () => {
      const pos: Position = { x: 0, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for position at max corner (18,18)', () => {
      const pos: Position = { x: 18, y: 18 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns false for position at (19,0)', () => {
      const pos: Position = { x: 19, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });
  });

  describe('hexagonal board validation', () => {
    const boardType: BoardType = 'hexagonal';
    // Size = bounding box = 2*radius + 1. So boardSize=7 means radius=3.
    const boardSize = 7;

    it('returns true for center hex (0,0)', () => {
      const pos: Position = { x: 0, y: 0, z: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for center hex without explicit z', () => {
      const pos: Position = { x: 0, y: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for hex at radius edge (3,0,-3)', () => {
      const pos: Position = { x: 3, y: 0, z: -3 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for hex at negative q edge (-3,3,0)', () => {
      const pos: Position = { x: -3, y: 3, z: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns true for hex at negative r edge (0,-3,3)', () => {
      const pos: Position = { x: 0, y: -3, z: 3 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(true);
    });

    it('returns false for hex beyond q radius (4,0,-4)', () => {
      const pos: Position = { x: 4, y: 0, z: -4 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex beyond negative q radius (-4,4,0)', () => {
      const pos: Position = { x: -4, y: 4, z: 0 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex beyond r radius (0,4,-4)', () => {
      const pos: Position = { x: 0, y: 4, z: -4 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex beyond negative r radius (0,-4,4)', () => {
      const pos: Position = { x: 0, y: -4, z: 4 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex with invalid q+r+s sum', () => {
      // q + r + s should equal 0 for valid hex coordinates
      const pos: Position = { x: 1, y: 1, z: 1 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex with s beyond radius even if q+r are valid', () => {
      // Even if q and r are within radius, s might push it out
      // radius is 3, so |s|=4 > 3, should be false
      const pos: Position = { x: 2, y: 2, z: -4 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('returns false for hex where calculated s exceeds radius', () => {
      // x=1, y=1 → s = -1-1 = -2, within radius 3: valid
      // x=2, y=2 → s = -2-2 = -4, |s|=4 > 3: invalid
      const pos: Position = { x: 2, y: 2 };
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });

    it('uses strict sum check (q + r + s === 0), rejects floating point errors', () => {
      // The function uses strict validation: sum === 0 (not rounded)
      // This matches VictoryAggregate.ts for consistency
      const pos: Position = { x: 1, y: -1, z: 0.0000001 };
      // q + r + s = 1 - 1 + 0.0000001 ≠ 0 exactly, so invalid
      expect(isValidPosition(pos, boardType, boardSize)).toBe(false);
    });
  });

  describe('hexagonal board with different sizes', () => {
    const boardType: BoardType = 'hexagonal';
    // Size = bounding box = 2*radius + 1. radius = (size - 1) / 2.

    it('size 1 has radius 0, only center is valid', () => {
      // radius = (1 - 1) / 2 = 0
      const boardSize = 1;
      expect(isValidPosition({ x: 0, y: 0 }, boardType, boardSize)).toBe(true);
      expect(isValidPosition({ x: 1, y: 0 }, boardType, boardSize)).toBe(false);
      expect(isValidPosition({ x: 0, y: 1 }, boardType, boardSize)).toBe(false);
    });

    it('size 3 allows radius 1 positions', () => {
      // radius = (3 - 1) / 2 = 1
      const boardSize = 3;
      expect(isValidPosition({ x: 0, y: 0 }, boardType, boardSize)).toBe(true);
      expect(isValidPosition({ x: 1, y: 0, z: -1 }, boardType, boardSize)).toBe(true);
      expect(isValidPosition({ x: -1, y: 1, z: 0 }, boardType, boardSize)).toBe(true);
      expect(isValidPosition({ x: 2, y: 0, z: -2 }, boardType, boardSize)).toBe(false);
    });

    it('size 9 allows radius 4 positions', () => {
      // radius = (9 - 1) / 2 = 4
      const boardSize = 9;
      expect(isValidPosition({ x: 4, y: 0, z: -4 }, boardType, boardSize)).toBe(true);
      expect(isValidPosition({ x: 5, y: 0, z: -5 }, boardType, boardSize)).toBe(false);
    });
  });

  describe('edge cases', () => {
    it('handles zero board size for square (all positions invalid)', () => {
      const pos: Position = { x: 0, y: 0 };
      expect(isValidPosition(pos, 'square8', 0)).toBe(false);
    });

    it('handles various square board types', () => {
      // square8 with size 8
      expect(isValidPosition({ x: 7, y: 7 }, 'square8', 8)).toBe(true);
      // square19 with size 19
      expect(isValidPosition({ x: 18, y: 18 }, 'square19', 19)).toBe(true);
    });

    it('hexagonal with z explicitly provided as undefined uses calculated s', () => {
      const pos: Position = { x: 1, y: -1, z: undefined };
      // s = -q - r = -1 - (-1) = 0
      // All within radius (boardSize=3 means radius=1)
      expect(isValidPosition(pos, 'hexagonal', 3)).toBe(true);
    });
  });
});
