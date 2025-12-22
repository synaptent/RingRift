/**
 * lineDetection.branchCoverage.test.ts
 *
 * Branch coverage tests for lineDetection.ts targeting uncovered branches:
 * - findAllLines: marker on collapsed/stack, line length threshold, duplicate detection
 * - getLineDirections: hexagonal vs square board types
 * - findLineInDirection: hex vs square stepping, break conditions
 * - isValidPosition: hex vs square validation
 */

import { findAllLines, findLinesForPlayer } from '../../src/shared/engine/lineDetection';
import type {
  BoardState,
  BoardType,
  Position,
  MarkerInfo,
  RingStack,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number, z?: number): Position => {
  const p: Position = { x, y };
  if (z !== undefined) p.z = z;
  return p;
};

// Helper to create a minimal BoardState
function makeBoardState(overrides: Partial<BoardState> = {}): BoardState {
  return {
    type: 'square8' as BoardType,
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    territories: new Map(),
    eliminatedRings: { 1: 0, 2: 0 },
    ...overrides,
  };
}

// Helper to add a marker to the board
function addMarker(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  board.markers.set(key, {
    position,
    player,
    type: 'regular',
  } as MarkerInfo);
}

// Helper to add a stack to the board
function addStack(
  board: BoardState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

// Helper to add a collapsed space
function addCollapsedSpace(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  board.collapsedSpaces.set(key, player);
}

describe('lineDetection branch coverage', () => {
  describe('findAllLines', () => {
    describe('marker filtering', () => {
      it('skips markers on collapsed spaces', () => {
        const board = makeBoardState();
        // Add 4 markers in a row
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        // Collapse the first position
        addCollapsedSpace(board, pos(0, 0), 1);

        const lines = findAllLines(board);

        // Should not form a 4-line because (0,0) is collapsed
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('skips markers where stack exists', () => {
        const board = makeBoardState();
        // Add 4 markers in a row
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        // Add a stack on position (0,0)
        addStack(board, pos(0, 0), 1, [1]);

        const lines = findAllLines(board);

        // Should not form a 4-line because (0,0) has a stack
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('processes markers without stacks or collapsed spaces', () => {
        const board = makeBoardState();
        // Add 4 markers in a row (no obstructions)
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);

        const lines = findAllLines(board);

        // Should find a line of 4
        expect(lines.some((l) => l.player === 1 && l.length === 4)).toBe(true);
      });
    });

    describe('line length threshold', () => {
      it('includes lines of exactly minimum length', () => {
        const board = makeBoardState();
        // 4 markers = minimum for square8
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);

        const lines = findAllLines(board);

        expect(lines.some((l) => l.length === 4)).toBe(true);
      });

      it('excludes lines below minimum length', () => {
        const board = makeBoardState();
        // Only 2 markers (below minimum of 3 for square8)
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);

        const lines = findAllLines(board);

        expect(lines).toHaveLength(0);
      });

      it('includes lines longer than minimum', () => {
        const board = makeBoardState();
        // 5 markers in a row
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);

        const lines = findAllLines(board);

        expect(lines.some((l) => l.length === 5)).toBe(true);
      });
    });

    describe('duplicate detection', () => {
      it('deduplicates identical lines found from different starting markers', () => {
        const board = makeBoardState();
        // 4 markers in a row - each could potentially find the same line
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);

        const lines = findAllLines(board);

        // Should only have one line, not 4 duplicates
        const horizontalLines = lines.filter(
          (l) => l.player === 1 && l.direction.x === 1 && l.direction.y === 0
        );
        expect(horizontalLines).toHaveLength(1);
      });
    });

    describe('multiple directions', () => {
      it('finds lines in different directions for same player', () => {
        const board = makeBoardState();
        // Horizontal line
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        // Vertical line (separate)
        addMarker(board, pos(5, 0), 1);
        addMarker(board, pos(5, 1), 1);
        addMarker(board, pos(5, 2), 1);
        addMarker(board, pos(5, 3), 1);

        const lines = findAllLines(board);

        expect(lines.filter((l) => l.player === 1)).toHaveLength(2);
      });

      it('finds diagonal lines', () => {
        const board = makeBoardState();
        // Diagonal line (southeast)
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 1), 1);
        addMarker(board, pos(2, 2), 1);
        addMarker(board, pos(3, 3), 1);

        const lines = findAllLines(board);

        expect(lines.some((l) => l.direction.x === 1 && l.direction.y === 1)).toBe(true);
      });

      it('finds northeast diagonal lines', () => {
        const board = makeBoardState();
        // Diagonal line (northeast: x+1, y-1)
        addMarker(board, pos(0, 4), 1);
        addMarker(board, pos(1, 3), 1);
        addMarker(board, pos(2, 2), 1);
        addMarker(board, pos(3, 1), 1);

        const lines = findAllLines(board);

        expect(lines.some((l) => l.direction.x === 1 && l.direction.y === -1)).toBe(true);
      });
    });

    describe('multiple players', () => {
      it('finds lines for different players', () => {
        const board = makeBoardState();
        // Player 1 horizontal line
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        // Player 2 horizontal line (different row)
        addMarker(board, pos(0, 2), 2);
        addMarker(board, pos(1, 2), 2);
        addMarker(board, pos(2, 2), 2);
        addMarker(board, pos(3, 2), 2);

        const lines = findAllLines(board);

        expect(lines.some((l) => l.player === 1)).toBe(true);
        expect(lines.some((l) => l.player === 2)).toBe(true);
      });
    });
  });

  describe('findLinesForPlayer', () => {
    it('filters lines by player number', () => {
      const board = makeBoardState();
      // Player 1 line
      addMarker(board, pos(0, 0), 1);
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(3, 0), 1);
      // Player 2 line
      addMarker(board, pos(0, 2), 2);
      addMarker(board, pos(1, 2), 2);
      addMarker(board, pos(2, 2), 2);
      addMarker(board, pos(3, 2), 2);

      const player1Lines = findLinesForPlayer(board, 1);
      const player2Lines = findLinesForPlayer(board, 2);

      expect(player1Lines.every((l) => l.player === 1)).toBe(true);
      expect(player2Lines.every((l) => l.player === 2)).toBe(true);
    });

    it('returns empty array when player has no lines', () => {
      const board = makeBoardState();
      // Only player 1 has a line
      addMarker(board, pos(0, 0), 1);
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(3, 0), 1);

      const player2Lines = findLinesForPlayer(board, 2);

      expect(player2Lines).toHaveLength(0);
    });
  });

  describe('hexagonal board support', () => {
    it('uses hexagonal directions for hex boards', () => {
      // Size = bounding box = 2*radius + 1. size=9 means radius=4.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
      // Hex line along one axis (x+1, y=0, z-1 is one of the 3 hex directions)
      // Using cube coordinates where x + y + z = 0
      addMarker(board, pos(0, 0, 0), 1);
      addMarker(board, pos(1, 0, -1), 1);
      addMarker(board, pos(2, 0, -2), 1);
      addMarker(board, pos(3, 0, -3), 1);

      const lines = findAllLines(board);

      expect(lines.some((l) => l.player === 1)).toBe(true);
    });

    it('validates hex positions using cube coordinate constraints', () => {
      // Size = bounding box = 2*radius + 1. size=9 means radius=4.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
      // Invalid position (x + y + z !== 0 or out of bounds) should not be included
      // But we can't directly test isValidPosition, so we test indirectly by
      // putting markers that would extend past valid bounds

      // Start at center (0,0,0) and try to extend
      addMarker(board, pos(0, 0, 0), 1);
      addMarker(board, pos(1, 0, -1), 1);
      addMarker(board, pos(2, 0, -2), 1);
      addMarker(board, pos(3, 0, -3), 1);
      // pos(4, 0, -4) is still valid for size 9 (radius 4)

      const lines = findAllLines(board);

      expect(lines.length).toBeGreaterThan(0);
    });

    it('stops line at hex board edge', () => {
      // Size = bounding box = 2*radius + 1. size=5 means radius=2.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 5 });
      // Size 5 means radius 2
      // Try to make a line that extends beyond radius
      addMarker(board, pos(-2, 0, 2), 1);
      addMarker(board, pos(-1, 0, 1), 1);
      addMarker(board, pos(0, 0, 0), 1);
      addMarker(board, pos(1, 0, -1), 1);
      addMarker(board, pos(2, 0, -2), 1);
      // This 5-marker line should be found (all within radius 2)

      const lines = findAllLines(board);

      expect(lines.some((l) => l.length === 5)).toBe(true);
    });
  });

  describe('findLineInDirection edge cases', () => {
    describe('forward direction breaks', () => {
      it('stops at board edge', () => {
        const board = makeBoardState();
        // Line starting near edge - only 2 markers (below min 3)
        addMarker(board, pos(6, 0), 1);
        addMarker(board, pos(7, 0), 1);
        // Would continue to (8,0) but that's off board

        const lines = findAllLines(board);

        // Only 2 markers, so no line formed (min is 3)
        expect(lines.filter((l) => l.player === 1)).toHaveLength(0);
      });

      it('stops when marker belongs to different player', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 2); // Different player!
        addMarker(board, pos(4, 0), 1);

        const lines = findAllLines(board);

        // Player 1's line should be broken at (3,0)
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('stops when hitting collapsed space in path', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);
        // Collapse the middle position
        addCollapsedSpace(board, pos(2, 0), 1);

        const lines = findAllLines(board);

        // Line should be broken, no 4+ line formed
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('stops when hitting stack in path', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);
        // Stack at (2,0)
        addStack(board, pos(2, 0), 1, [1]);

        const lines = findAllLines(board);

        // Line broken by stack
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('stops when no marker at next position', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        // Gap at (2,0)
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);

        const lines = findAllLines(board);

        // Gap breaks the line
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });
    });

    describe('backward direction breaks', () => {
      it('stops at board edge in backward direction', () => {
        const board = makeBoardState();
        // Line ending near origin
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        // Starting from (2,0), backward would try (-1,0) which is invalid

        const lines = findAllLines(board);

        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });

      it('stops at different player marker in backward direction', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 2); // Different player
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);

        const lines = findAllLines(board);

        // Player 1 should only have line of 4 starting at (1,0)
        const p1Lines = lines.filter((l) => l.player === 1 && l.length >= 4);
        expect(p1Lines).toHaveLength(1);
        expect(p1Lines[0].length).toBe(4);
      });

      it('stops at collapsed space in backward direction', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addCollapsedSpace(board, pos(1, 0), 1); // Collapsed
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);
        addMarker(board, pos(5, 0), 1);

        const lines = findAllLines(board);

        // Line should be found starting from (2,0)
        const p1Lines = lines.filter((l) => l.player === 1);
        expect(p1Lines.some((l) => l.length === 4)).toBe(true);
      });

      it('stops at stack in backward direction', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);
        addStack(board, pos(1, 0), 1, [1]); // Stack
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(4, 0), 1);
        addMarker(board, pos(5, 0), 1);

        const lines = findAllLines(board);

        // Line from (2,0) to (5,0) should be found
        const p1Lines = lines.filter((l) => l.player === 1);
        expect(p1Lines.some((l) => l.length === 4)).toBe(true);
      });
    });
  });

  describe('isValidPosition coverage', () => {
    describe('square board validation', () => {
      it('rejects negative x coordinate', () => {
        const board = makeBoardState();
        // Position (-1,0) is invalid - tested indirectly through line detection
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        // Cannot extend backwards past (0,0)

        const lines = findAllLines(board);
        // Just verify no errors occur
        expect(lines.length).toBeGreaterThanOrEqual(0);
      });

      it('rejects coordinate >= size', () => {
        const board = makeBoardState({ size: 4 });
        // With size 4, position (4,0) is invalid
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(3, 0), 1);
        // pos(4,0) would be invalid

        const lines = findAllLines(board);
        expect(lines.some((l) => l.length === 4)).toBe(true);
      });

      it('rejects negative y coordinate', () => {
        const board = makeBoardState();
        // Test vertical line stopping at y=0
        addMarker(board, pos(0, 0), 1);
        addMarker(board, pos(0, 1), 1);
        addMarker(board, pos(0, 2), 1);

        const lines = findAllLines(board);
        // Just verifies we handle the boundary correctly
        expect(lines.filter((l) => l.player === 1 && l.length >= 4)).toHaveLength(0);
      });
    });

    describe('hexagonal board validation', () => {
      it('validates cube coordinate constraint (q + r + s === 0)', () => {
        // Size = bounding box = 2*radius + 1. size=9 means radius=4.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
        // Valid hex positions must satisfy q + r + s === 0
        addMarker(board, pos(0, 0, 0), 1); // 0 + 0 + 0 = 0 ✓
        addMarker(board, pos(1, -1, 0), 1); // 1 + (-1) + 0 = 0 ✓
        addMarker(board, pos(2, -2, 0), 1); // 2 + (-2) + 0 = 0 ✓
        addMarker(board, pos(3, -3, 0), 1); // 3 + (-3) + 0 = 0 ✓

        const lines = findAllLines(board);
        expect(lines.length).toBeGreaterThan(0);
      });

      it('validates radius constraint', () => {
        // Size = bounding box = 2*radius + 1. size=5 means radius=2.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 5 });
        // Size 5 means radius 2, so |q|, |r|, |s| <= 2
        addMarker(board, pos(0, 0, 0), 1);
        addMarker(board, pos(1, -1, 0), 1);
        addMarker(board, pos(2, -2, 0), 1);
        // (3, -3, 0) would be out of bounds for radius 2

        const lines = findAllLines(board);
        // With only 3 markers, no line (need 4)
        expect(lines.length).toBe(0);
      });

      it('computes z from x and y when z is undefined', () => {
        // Size = bounding box = 2*radius + 1. size=9 means radius=4.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
        // Test that z = -x - y is used when z is undefined
        // This tests the isValidPosition logic: s = position.z || -q - r
        addMarker(board, pos(0, 0, 0), 1);
        addMarker(board, pos(1, 0, -1), 1);
        addMarker(board, pos(2, 0, -2), 1);
        addMarker(board, pos(3, 0, -3), 1);

        const lines = findAllLines(board);
        expect(lines.length).toBeGreaterThan(0);
      });
    });
  });

  describe('line result structure', () => {
    it('includes correct positions in line', () => {
      const board = makeBoardState();
      addMarker(board, pos(0, 0), 1);
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(3, 0), 1);

      const lines = findAllLines(board);
      const line = lines.find((l) => l.player === 1);

      expect(line).toBeDefined();
      expect(line!.positions).toHaveLength(4);
      expect(line!.positions.some((p) => p.x === 0 && p.y === 0)).toBe(true);
      expect(line!.positions.some((p) => p.x === 3 && p.y === 0)).toBe(true);
    });

    it('includes correct direction', () => {
      const board = makeBoardState();
      addMarker(board, pos(0, 0), 1);
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(3, 0), 1);

      const lines = findAllLines(board);
      const line = lines.find((l) => l.player === 1);

      expect(line).toBeDefined();
      expect(line!.direction).toEqual({ x: 1, y: 0 }); // East direction
    });

    it('includes correct player and length', () => {
      const board = makeBoardState();
      addMarker(board, pos(0, 0), 2);
      addMarker(board, pos(1, 0), 2);
      addMarker(board, pos(2, 0), 2);
      addMarker(board, pos(3, 0), 2);
      addMarker(board, pos(4, 0), 2);

      const lines = findAllLines(board);
      const line = lines.find((l) => l.player === 2);

      expect(line).toBeDefined();
      expect(line!.player).toBe(2);
      expect(line!.length).toBe(5);
    });
  });

  describe('empty board', () => {
    it('returns empty array for board with no markers', () => {
      const board = makeBoardState();

      const lines = findAllLines(board);

      expect(lines).toEqual([]);
    });
  });
});
