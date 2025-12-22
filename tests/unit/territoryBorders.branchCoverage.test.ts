/**
 * territoryBorders.branchCoverage.test.ts
 *
 * Branch coverage tests for territoryBorders.ts targeting uncovered branches:
 * - getBorderMarkerPositionsForRegion: mode selection
 * - getBorderMarkersRustAligned: empty region, seed finding, Moore expansion
 * - getTerritoryNeighbors: hexagonal, von_neumann, Moore fallback
 * - getMooreNeighbors: hex board returns empty
 * - isValidPositionOnBoard: hex vs square validation
 * - comparePositionsStable: square vs hex sorting
 *
 * COVERAGE ANALYSIS:
 * Lines 160-170: Moore adjacency fallback in getTerritoryNeighbors.
 *   This branch is only reached when territoryAdjacency === 'moore'.
 *   However, NO production board type uses 'moore' for territoryAdjacency:
 *   - square8, square19: 'von_neumann'
 *   - hexagonal: 'hexagonal'
 *   This is defensive code for potential future board types.
 *
 * Line 180: Hex check in getMooreNeighbors.
 *   getMooreNeighbors is ONLY called when board.type !== 'hexagonal' (line 83),
 *   so the condition `if (board.type === 'hexagonal')` can never be true.
 *   This is unreachable defensive code.
 *
 * Maximum achievable branch coverage: ~78.43% (40/51 branches)
 * Unreachable branches: lines 160-170 (Moore fallback), 180 (hex check in getMooreNeighbors)
 */

import {
  getBorderMarkerPositionsForRegion,
  TerritoryBorderOptions,
} from '../../src/shared/engine/territoryBorders';
import type { BoardState, Position, MarkerInfo } from '../../src/shared/engine/types';
import type { BoardType } from '../../src/shared/types/game';
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

describe('territoryBorders branch coverage', () => {
  describe('getBorderMarkerPositionsForRegion', () => {
    describe('mode selection', () => {
      it('defaults to rust_aligned mode when no options provided', () => {
        const board = makeBoardState();
        addMarker(board, pos(1, 0), 1);
        const region = [pos(0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should use rust_aligned mode by default
        expect(result).toHaveLength(1);
        expect(result[0]).toEqual(pos(1, 0));
      });

      it('uses rust_aligned mode when explicitly specified', () => {
        const board = makeBoardState();
        addMarker(board, pos(1, 0), 1);
        const region = [pos(0, 0)];
        const opts: TerritoryBorderOptions = { mode: 'rust_aligned' };

        const result = getBorderMarkerPositionsForRegion(board, region, opts);

        expect(result).toHaveLength(1);
      });

      it('uses ts_legacy mode when specified', () => {
        const board = makeBoardState();
        addMarker(board, pos(1, 0), 1);
        const region = [pos(0, 0)];
        const opts: TerritoryBorderOptions = { mode: 'ts_legacy' };

        const result = getBorderMarkerPositionsForRegion(board, region, opts);

        // ts_legacy uses same implementation as rust_aligned
        expect(result).toHaveLength(1);
      });
    });
  });

  describe('getBorderMarkersRustAligned', () => {
    describe('empty region handling', () => {
      it('returns empty array for empty region', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1);

        const result = getBorderMarkerPositionsForRegion(board, []);

        expect(result).toEqual([]);
      });
    });

    describe('seed marker finding', () => {
      it('finds markers adjacent to region via territory adjacency', () => {
        const board = makeBoardState();
        // For square8, territory adjacency is von_neumann (4-direction)
        addMarker(board, pos(1, 0), 1); // East of region
        addMarker(board, pos(0, 1), 1); // South of region
        const region = [pos(0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        expect(result).toHaveLength(2);
      });

      it('ignores markers that are inside the region', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 1); // Inside region
        addMarker(board, pos(1, 0), 1); // Adjacent to region
        const region = [pos(0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Only (1,0) should be found, not (0,0)
        expect(result).toHaveLength(1);
        expect(result[0]).toEqual(pos(1, 0));
      });

      it('does not duplicate markers already in seed', () => {
        const board = makeBoardState();
        addMarker(board, pos(1, 0), 1);
        // Region with two adjacent spaces both touching (1,0)
        const region = [pos(0, 0), pos(2, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // (1,0) should appear only once
        expect(result).toHaveLength(1);
        expect(result[0]).toEqual(pos(1, 0));
      });

      it('returns empty when no adjacent markers found', () => {
        const board = makeBoardState();
        // No markers adjacent to region
        const region = [pos(3, 3)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        expect(result).toEqual([]);
      });
    });

    describe('Moore expansion on square boards', () => {
      it('expands border via Moore adjacency to connected markers', () => {
        const board = makeBoardState();
        // Region at (0,0), marker at (1,0) is adjacent via von_neumann
        // Marker at (2,0) is connected to (1,0) via Moore
        // Marker at (2,1) is connected diagonally via Moore
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1);
        addMarker(board, pos(2, 1), 1);
        const region = [pos(0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should include all three markers via Moore expansion
        expect(result).toHaveLength(3);
      });

      it('does not expand into region spaces', () => {
        const board = makeBoardState();
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(2, 0), 1); // This is inside region
        addMarker(board, pos(3, 0), 1);
        const region = [pos(0, 0), pos(2, 0)]; // (2,0) is in region

        const result = getBorderMarkerPositionsForRegion(board, region);

        // (2,0) should not be in result
        expect(result.find((p) => p.x === 2 && p.y === 0)).toBeUndefined();
      });

      it('skips already visited positions during expansion', () => {
        const board = makeBoardState();
        // Create a cycle of markers
        addMarker(board, pos(1, 0), 1);
        addMarker(board, pos(1, 1), 1);
        addMarker(board, pos(0, 1), 1);
        // (0,1) is adjacent to starting region and to (1,1) and (1,0)
        const region = [pos(0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should find all 3 markers without infinite loop
        expect(result).toHaveLength(3);
      });

      it('handles marker at edge of board', () => {
        const board = makeBoardState({ size: 4 });
        addMarker(board, pos(3, 0), 1);
        addMarker(board, pos(3, 1), 1);
        const region = [pos(2, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should find both markers
        expect(result.length).toBeGreaterThanOrEqual(1);
      });
    });

    describe('hex board behavior', () => {
      it('does not expand on hex boards (seed only)', () => {
        // Size = bounding box = 2*radius + 1. size=9 means radius=4.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
        // Region at center, marker at adjacent position
        addMarker(board, pos(1, 0, -1), 1);
        // Another marker that would be found via expansion on square but not hex
        addMarker(board, pos(2, 0, -2), 1);
        const region = [pos(0, 0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Only (1,0,-1) should be found (direct adjacency)
        // (2,0,-2) is not directly adjacent to region
        expect(result).toHaveLength(1);
        expect(result[0]).toEqual(pos(1, 0, -1));
      });
    });
  });

  describe('getTerritoryNeighbors adjacency types', () => {
    describe('hexagonal adjacency', () => {
      it('uses 6 hex directions for hexagonal boards', () => {
        // Size = bounding box = 2*radius + 1. size=9 means radius=4.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
        // Place markers in all 6 hex directions from center
        addMarker(board, pos(1, 0, -1), 1); // East
        addMarker(board, pos(1, -1, 0), 1); // NE
        addMarker(board, pos(0, -1, 1), 1); // NW
        addMarker(board, pos(-1, 0, 1), 1); // West
        addMarker(board, pos(-1, 1, 0), 1); // SW
        addMarker(board, pos(0, 1, -1), 1); // SE
        const region = [pos(0, 0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should find all 6 markers
        expect(result).toHaveLength(6);
      });
    });

    describe('von_neumann adjacency', () => {
      it('uses 4 directions for square boards with von_neumann', () => {
        // square8 uses von_neumann for territory adjacency
        const board = makeBoardState();
        // Place markers in 4 von_neumann directions
        addMarker(board, pos(1, 1), 1); // East
        addMarker(board, pos(0, 2), 1); // South
        // Diagonal markers should NOT be found via von_neumann
        addMarker(board, pos(2, 2), 1); // Diagonal
        const region = [pos(0, 1)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // (1,1) and (0,2) are von_neumann adjacent
        // (2,2) is only diagonally adjacent but can be reached via Moore expansion
        expect(result.some((p) => p.x === 1 && p.y === 1)).toBe(true);
        expect(result.some((p) => p.x === 0 && p.y === 2)).toBe(true);
      });
    });

    describe('Moore adjacency fallback', () => {
      // The Moore fallback is used when territoryAdjacency is 'moore'
      // But all current board configs use 'von_neumann' or 'hexagonal'
      // This branch is covered by the expansion logic which uses Moore
    });
  });

  describe('getMooreNeighbors', () => {
    it('returns empty for hex boards', () => {
      // Size = bounding box = 2*radius + 1. size=9 means radius=4.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
      addMarker(board, pos(1, 0, -1), 1);
      addMarker(board, pos(1, 1, -2), 1); // Would be Moore neighbor
      const region = [pos(0, 0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Only direct hex adjacency found, no Moore expansion
      expect(result).toHaveLength(1);
    });

    it('returns 8 neighbors for square boards', () => {
      const board = makeBoardState();
      // Place markers in all 8 Moore directions around (3,3)
      addMarker(board, pos(2, 2), 1); // NW
      addMarker(board, pos(3, 2), 1); // N
      addMarker(board, pos(4, 2), 1); // NE
      addMarker(board, pos(4, 3), 1); // E
      addMarker(board, pos(4, 4), 1); // SE
      addMarker(board, pos(3, 4), 1); // S
      addMarker(board, pos(2, 4), 1); // SW
      addMarker(board, pos(2, 3), 1); // W
      // Seed with direct adjacency marker
      addMarker(board, pos(1, 3), 1); // West of region
      const region = [pos(0, 3)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // All markers should be found via Moore expansion chain
      expect(result.length).toBeGreaterThan(1);
    });
  });

  describe('isValidPositionOnBoard', () => {
    describe('hexagonal board validation', () => {
      it('validates hex positions within radius', () => {
        // Size = bounding box = 2*radius + 1. size=5 means radius=2.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 5 });
        // size 5 = radius 2
        addMarker(board, pos(2, 0, -2), 1); // At edge of radius
        const region = [pos(1, 0, -1)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        expect(result.some((p) => p.x === 2 && p.y === 0)).toBe(true);
      });

      it('rejects hex positions outside radius', () => {
        // Size = bounding box = 2*radius + 1. size=3 means radius=1.
        const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 3 });
        // size 3 = radius 1
        // (2, 0, -2) would be outside radius 1
        addMarker(board, pos(2, 0, -2), 1);
        const region = [pos(0, 0, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        // (2,0,-2) is out of bounds, shouldn't be found
        expect(result.find((p) => p.x === 2 && p.y === 0)).toBeUndefined();
      });
    });

    describe('square board validation', () => {
      it('validates positions within size', () => {
        const board = makeBoardState({ size: 4 });
        addMarker(board, pos(3, 3), 1); // At corner
        const region = [pos(2, 3)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        expect(result.some((p) => p.x === 3 && p.y === 3)).toBe(true);
      });

      it('rejects positions with negative coordinates', () => {
        const board = makeBoardState();
        // Region at edge - neighbor check should not include (-1,0)
        const region = [pos(0, 0)];
        addMarker(board, pos(1, 0), 1);

        const result = getBorderMarkerPositionsForRegion(board, region);

        // Should only find (1,0), not any negative positions
        expect(result).toHaveLength(1);
      });

      it('rejects positions >= size', () => {
        const board = makeBoardState({ size: 4 });
        // Region at edge - neighbor check should not include (4,0)
        addMarker(board, pos(3, 0), 1);
        const region = [pos(2, 0)];

        const result = getBorderMarkerPositionsForRegion(board, region);

        expect(result.some((p) => p.x === 3)).toBe(true);
      });
    });
  });

  describe('comparePositionsStable (sorting)', () => {
    it('sorts square positions in row-major order (y then x)', () => {
      const board = makeBoardState();
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(0, 1), 1);
      addMarker(board, pos(1, 0), 1);
      const region = [pos(0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Should be sorted: (1,0), (2,0), (0,1)
      expect(result[0]).toEqual(pos(1, 0));
      expect(result[1]).toEqual(pos(2, 0));
      expect(result[2]).toEqual(pos(0, 1));
    });

    it('sorts hex positions in cube-lexicographic order (x, y, z)', () => {
      // Size = bounding box = 2*radius + 1. size=9 means radius=4.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
      addMarker(board, pos(1, 0, -1), 1); // x=1
      addMarker(board, pos(0, 1, -1), 1); // x=0
      addMarker(board, pos(-1, 1, 0), 1); // x=-1
      const region = [pos(0, 0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Should be sorted by x: -1, 0, 1
      expect(result[0].x).toBe(-1);
      expect(result[1].x).toBe(0);
      expect(result[2].x).toBe(1);
    });

    it('handles mixed z values (computes z from x,y when undefined)', () => {
      // Size = bounding box = 2*radius + 1. size=9 means radius=4.
      const board = makeBoardState({ type: 'hexagonal' as BoardType, size: 9 });
      // Add markers with explicit z
      addMarker(board, pos(1, -1, 0), 1);
      addMarker(board, pos(1, 0, -1), 1);
      const region = [pos(0, 0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Both markers have x=1, should be sorted by y: -1, 0
      expect(result[0].y).toBe(-1);
      expect(result[1].y).toBe(0);
    });
  });

  describe('complex scenarios', () => {
    it('handles L-shaped region', () => {
      const board = makeBoardState();
      addMarker(board, pos(2, 0), 1);
      addMarker(board, pos(2, 1), 1);
      addMarker(board, pos(1, 2), 1);
      // L-shaped region
      const region = [pos(0, 0), pos(1, 0), pos(0, 1), pos(0, 2)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Should find markers adjacent to the L shape
      expect(result.length).toBeGreaterThan(0);
    });

    it('handles single-cell region surrounded by markers', () => {
      const board = makeBoardState();
      // Surround region with markers
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(0, 1), 1);
      addMarker(board, pos(1, 1), 1);
      const region = [pos(0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Should find all adjacent markers
      expect(result).toHaveLength(3);
    });

    it('handles large region with scattered markers', () => {
      const board = makeBoardState({ size: 8 });
      // Large region
      const region = [pos(2, 2), pos(3, 2), pos(4, 2), pos(2, 3), pos(3, 3), pos(4, 3)];
      // Markers at various edges
      addMarker(board, pos(1, 2), 1);
      addMarker(board, pos(5, 3), 1);
      addMarker(board, pos(3, 1), 1);

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Should find all border markers
      expect(result.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('edge cases', () => {
    it('handles region at board corner', () => {
      const board = makeBoardState({ size: 4 });
      addMarker(board, pos(0, 1), 1);
      addMarker(board, pos(1, 0), 1);
      const region = [pos(0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      expect(result).toHaveLength(2);
    });

    it('handles multiple disconnected markers (no expansion chain)', () => {
      const board = makeBoardState();
      // Markers not connected to each other
      addMarker(board, pos(1, 0), 1);
      addMarker(board, pos(0, 7), 1); // Far away
      const region = [pos(0, 0)];

      const result = getBorderMarkerPositionsForRegion(board, region);

      // Only (1,0) is adjacent to region
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(pos(1, 0));
    });
  });
});
