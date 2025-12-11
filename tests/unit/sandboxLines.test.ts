/**
 * Unit tests for line geometry functions.
 *
 * Tests the canonical line detection geometry from LineAggregate.
 * The sandbox's findAllLinesOnBoard is tested to ensure it delegates correctly.
 */

import { findAllLinesOnBoard } from '../../src/client/sandbox/sandboxLines';
import { createTestBoard, pos, posStr } from '../utils/fixtures';
import type { BoardState, BoardType, Position } from '../../src/shared/types/game';
import {
  isValidPosition,
  BOARD_CONFIGS,
  // Canonical line geometry from LineAggregate
  getLineDirections,
  findLineInDirection,
} from '../../src/shared/engine';

// Type alias for backward compatibility with existing tests
type LineDirection = Position;

describe('Line Geometry (LineAggregate)', () => {
  describe('getLineDirections', () => {
    it('returns 3 directions for hexagonal board', () => {
      const directions = getLineDirections('hexagonal');

      expect(directions).toHaveLength(3);
      // Verify hex directions have z component
      expect(directions.every((d) => d.z !== undefined)).toBe(true);
      // Check specific directions
      expect(directions).toContainEqual({ x: 1, y: 0, z: -1 }); // East
      expect(directions).toContainEqual({ x: 1, y: -1, z: 0 }); // Northeast
      expect(directions).toContainEqual({ x: 0, y: -1, z: 1 }); // Northwest
    });

    it('returns 4 directions for square8 board', () => {
      const directions = getLineDirections('square8');

      expect(directions).toHaveLength(4);
      // Verify square directions don't have z component
      expect(directions.every((d) => d.z === undefined)).toBe(true);
      // Check specific directions
      expect(directions).toContainEqual({ x: 1, y: 0 }); // East
      expect(directions).toContainEqual({ x: 1, y: 1 }); // Southeast
      expect(directions).toContainEqual({ x: 0, y: 1 }); // South
      expect(directions).toContainEqual({ x: 1, y: -1 }); // Northeast
    });

    it('returns 4 directions for square19 board', () => {
      const directions = getLineDirections('square19');

      expect(directions).toHaveLength(4);
    });
  });

  describe('findLineInDirection', () => {
    const boardType: BoardType = 'square8';

    it('returns single position when no adjacent markers', () => {
      const board = createTestBoard(boardType);
      board.markers.set(posStr(3, 3), {
        position: pos(3, 3),
        player: 1,
        type: 'regular',
      });

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(1);
      expect(line[0]).toEqual(pos(3, 3));
    });

    it('finds horizontal line of markers', () => {
      const board = createTestBoard(boardType);
      // Place markers at (2,3), (3,3), (4,3), (5,3)
      for (let x = 2; x <= 5; x++) {
        board.markers.set(posStr(x, 3), {
          position: pos(x, 3),
          player: 1,
          type: 'regular',
        });
      }

      const direction: LineDirection = { x: 1, y: 0 }; // East

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(4);
      expect(line[0]).toEqual(pos(2, 3));
      expect(line[3]).toEqual(pos(5, 3));
    });

    it('finds diagonal line of markers', () => {
      const board = createTestBoard(boardType);
      // Place markers diagonally at (1,1), (2,2), (3,3), (4,4)
      for (let i = 1; i <= 4; i++) {
        board.markers.set(posStr(i, i), {
          position: pos(i, i),
          player: 2,
          type: 'regular',
        });
      }

      const direction: LineDirection = { x: 1, y: 1 }; // Southeast

      const line = findLineInDirection(pos(2, 2), direction, 2, board);

      expect(line).toHaveLength(4);
      expect(line[0]).toEqual(pos(1, 1));
      expect(line[3]).toEqual(pos(4, 4));
    });

    it('stops at markers owned by different player', () => {
      const board = createTestBoard(boardType);
      board.markers.set(posStr(2, 3), { position: pos(2, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(3, 3), { position: pos(3, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(4, 3), { position: pos(4, 3), player: 2, type: 'regular' }); // Different player

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(2);
      expect(line).toContainEqual(pos(2, 3));
      expect(line).toContainEqual(pos(3, 3));
    });

    it('stops at empty spaces (no marker)', () => {
      const board = createTestBoard(boardType);
      board.markers.set(posStr(2, 3), { position: pos(2, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(3, 3), { position: pos(3, 3), player: 1, type: 'regular' });
      // Gap at (4, 3)
      board.markers.set(posStr(5, 3), { position: pos(5, 3), player: 1, type: 'regular' });

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(2); // Only (2,3) and (3,3)
    });

    it('stops at collapsed spaces', () => {
      const board = createTestBoard(boardType);
      board.markers.set(posStr(2, 3), { position: pos(2, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(3, 3), { position: pos(3, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(4, 3), { position: pos(4, 3), player: 1, type: 'regular' });
      board.collapsedSpaces.set(posStr(4, 3), { position: pos(4, 3), collapsed: true });

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(2); // Collapsed space blocks the line
    });

    it('stops at stacks', () => {
      const board = createTestBoard(boardType);
      board.markers.set(posStr(2, 3), { position: pos(2, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(3, 3), { position: pos(3, 3), player: 1, type: 'regular' });
      board.markers.set(posStr(4, 3), { position: pos(4, 3), player: 1, type: 'regular' });
      // Stack at position blocks the line even with marker present
      board.stacks.set(posStr(4, 3), {
        position: pos(4, 3),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(3, 3), direction, 1, board);

      expect(line).toHaveLength(2); // Stack blocks the line
    });

    it('stops at board boundary', () => {
      const board = createTestBoard(boardType);
      // Place markers at edge of board
      for (let x = 5; x <= 7; x++) {
        board.markers.set(posStr(x, 3), {
          position: pos(x, 3),
          player: 1,
          type: 'regular',
        });
      }

      const direction: LineDirection = { x: 1, y: 0 };

      const line = findLineInDirection(pos(6, 3), direction, 1, board);

      expect(line).toHaveLength(3); // Stops at x=7 (board edge is 0-7)
    });

    describe('hexagonal board', () => {
      it('finds line on hexagonal board', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };

        // Place markers along East direction (x+1, y, z-1)
        board.markers.set('0,0,0', { position: { x: 0, y: 0, z: 0 }, player: 1, type: 'regular' });
        board.markers.set('1,0,-1', {
          position: { x: 1, y: 0, z: -1 },
          player: 1,
          type: 'regular',
        });
        board.markers.set('2,0,-2', {
          position: { x: 2, y: 0, z: -2 },
          player: 1,
          type: 'regular',
        });

        const direction: LineDirection = { x: 1, y: 0, z: -1 }; // East

        const line = findLineInDirection({ x: 1, y: 0, z: -1 }, direction, 1, board);

        expect(line).toHaveLength(3);
      });

      it('handles hex positions with z coordinate correctly', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };

        // Place markers along Northeast direction (x+1, y-1, z)
        board.markers.set('-1,1,0', {
          position: { x: -1, y: 1, z: 0 },
          player: 2,
          type: 'regular',
        });
        board.markers.set('0,0,0', { position: { x: 0, y: 0, z: 0 }, player: 2, type: 'regular' });
        board.markers.set('1,-1,0', {
          position: { x: 1, y: -1, z: 0 },
          player: 2,
          type: 'regular',
        });

        const direction: LineDirection = { x: 1, y: -1, z: 0 }; // Northeast

        const line = findLineInDirection({ x: 0, y: 0, z: 0 }, direction, 2, board);

        expect(line).toHaveLength(3);
        expect(line[0]).toEqual({ x: -1, y: 1, z: 0 });
        expect(line[2]).toEqual({ x: 1, y: -1, z: 0 });
      });
    });
  });

  describe('findAllLinesOnBoard', () => {
    it('delegates to shared findAllLines helper', () => {
      const board = createTestBoard('square8');
      // Add 5 markers in a row to form a line
      for (let x = 0; x < 5; x++) {
        board.markers.set(posStr(x, 0), {
          position: pos(x, 0),
          player: 1,
          type: 'regular',
        });
      }

      const isValid = (p: Position) => true;
      const strToPos = (s: string) => {
        const [x, y] = s.split(',').map(Number);
        return { x, y };
      };

      const lines = findAllLinesOnBoard('square8', board, isValid, strToPos);

      // Should find at least one line (the 5-marker horizontal line)
      expect(lines.length).toBeGreaterThan(0);
      const foundLine = lines.find((l) => l.positions.length >= 5);
      expect(foundLine).toBeDefined();
    });

    it('returns empty array when no lines exist', () => {
      const board = createTestBoard('square8');
      // Add sparse markers that don't form lines
      board.markers.set(posStr(0, 0), { position: pos(0, 0), player: 1, type: 'regular' });
      board.markers.set(posStr(3, 3), { position: pos(3, 3), player: 1, type: 'regular' });

      const isValid = (p: Position) => true;
      const strToPos = (s: string) => {
        const [x, y] = s.split(',').map(Number);
        return { x, y };
      };

      const lines = findAllLinesOnBoard('square8', board, isValid, strToPos);

      expect(lines).toHaveLength(0);
    });
  });
});
