/**
 * territoryProcessing.branchCoverage.test.ts
 *
 * Branch coverage tests for territoryProcessing.ts targeting uncovered branches:
 * - canProcessTerritoryRegion: stack outside region check, different player check
 * - filterProcessableTerritoryRegions: empty regions, filtering logic
 * - applyTerritoryRegion: no stack, stack elimination, border marker skip, territory gain
 */

import {
  canProcessTerritoryRegion,
  filterProcessableTerritoryRegions,
  getProcessableTerritoryRegions,
  applyTerritoryRegion,
  TerritoryProcessingContext,
} from '../../src/shared/engine/territoryProcessing';
import type { RingStack, Territory } from '../../src/shared/engine/types';
import type { Position, BoardType, BoardState } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { calculateCapHeight } from '../../src/shared/engine/core';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

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
    capHeight: calculateCapHeight(rings),
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

// Helper to add a marker to the board
function addMarker(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  board.markers.set(key, {
    position,
    player,
    type: 'regular',
  });
}

// Helper to add collapsed space
function addCollapsed(board: BoardState, position: Position, owner: number): void {
  const key = positionToString(position);
  board.collapsedSpaces.set(key, owner);
}

// Helper to create a territory/region
function makeTerritory(spaces: Position[], controllingPlayer: number = 1): Territory {
  return {
    spaces,
    controllingPlayer,
    isDisconnected: true,
  };
}

describe('territoryProcessing branch coverage', () => {
  describe('canProcessTerritoryRegion', () => {
    it('returns true when player has stack outside region', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]); // Outside region
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(true);
    });

    it('returns false when player has no stacks outside region', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 1, [1, 1]); // Inside region
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(false);
    });

    it('returns false when player has no stacks at all', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 2, [2, 2]); // Different player's stack
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(false);
    });

    it('skips stacks controlled by other players', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 2, [2, 2]); // Player 2's stack outside
      addStack(board, pos(1, 0), 1, [1, 1]); // Player 1's stack outside
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      // Should find player 1's stack
      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(true);
    });

    it('ignores stacks inside region even if controlled by player', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 1, [1, 1]); // Inside region
      addStack(board, pos(5, 6), 1, [1]); // Inside region
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(false);
    });

    it('handles empty board', () => {
      const board = makeBoardState();
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(false);
    });

    it('handles empty region', () => {
      const board = makeBoardState();
      // RR-CANON-R082: Must be height >= 2 to be an eligible cap target
      addStack(board, pos(0, 0), 1, [1, 1]);
      const region = makeTerritory([], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      // Stack is outside region (since region is empty) and is eligible (height 2)
      expect(canProcessTerritoryRegion(board, region, ctx)).toBe(true);
    });
  });

  describe('filterProcessableTerritoryRegions', () => {
    it('returns empty array for empty regions input', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]);
      const ctx: TerritoryProcessingContext = { player: 1 };

      expect(filterProcessableTerritoryRegions(board, [], ctx)).toEqual([]);
    });

    it('filters out non-processable regions', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 1, [1]); // Inside region1, no stacks outside
      const region1 = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = filterProcessableTerritoryRegions(board, [region1], ctx);

      expect(result).toHaveLength(0);
    });

    it('keeps processable regions', () => {
      const board = makeBoardState();
      // RR-CANON-R082: Must be height >= 2 to be an eligible cap target
      addStack(board, pos(0, 0), 1, [1, 1]); // Outside both regions, height 2
      const region1 = makeTerritory([pos(5, 5)], 1);
      const region2 = makeTerritory([pos(6, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = filterProcessableTerritoryRegions(board, [region1, region2], ctx);

      expect(result).toHaveLength(2);
    });

    it('filters out region where all player stacks are inside', () => {
      const board = makeBoardState();
      // Player 1 has only one stack, and it's inside region1
      addStack(board, pos(5, 5), 1, [1]); // This is the ONLY stack for player 1
      const region1 = makeTerritory([pos(5, 5)], 1); // Contains the only player 1 stack
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = filterProcessableTerritoryRegions(board, [region1], ctx);

      // Region1 is NOT processable (player 1 has no stacks outside region1)
      expect(result).toHaveLength(0);
    });
  });

  describe('getProcessableTerritoryRegions', () => {
    it('returns empty for board with no disconnected regions', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = getProcessableTerritoryRegions(board, ctx);

      // No disconnected regions detected, so empty result
      expect(result).toEqual([]);
    });
  });

  describe('applyTerritoryRegion', () => {
    it('eliminates stacks inside region', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 2, [2, 2, 2]); // Stack to eliminate
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Stack should be eliminated
      expect(result.board.stacks.has('5,5')).toBe(false);
      // Eliminations credited to player 1
      expect(result.eliminatedRingsByPlayer[1]).toBe(3);
    });

    it('skips spaces without stacks (continue branch)', () => {
      const board = makeBoardState();
      // No stack at 5,5
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Should complete without error
      expect(result.eliminatedRingsByPlayer).toEqual({});
    });

    it('collapses region spaces', () => {
      const board = makeBoardState();
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      expect(result.board.collapsedSpaces.get('5,5')).toBe(1);
      expect(result.board.collapsedSpaces.get('5,6')).toBe(1);
    });

    it('removes markers in region spaces', () => {
      const board = makeBoardState();
      addMarker(board, pos(5, 5), 2);
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      expect(result.board.markers.has('5,5')).toBe(false);
    });

    it('skips border markers that are inside region (defensive check)', () => {
      const board = makeBoardState();
      // Create a region with some spaces
      const region = makeTerritory([pos(3, 3), pos(3, 4), pos(4, 3), pos(4, 4)], 1);
      // Add markers around and some inside (normally shouldn't happen)
      addMarker(board, pos(2, 3), 1);
      addMarker(board, pos(3, 2), 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Region spaces should be collapsed
      expect(result.board.collapsedSpaces.has('3,3')).toBe(true);
      expect(result.board.collapsedSpaces.has('4,4')).toBe(true);
    });

    it('calculates territory gain correctly', () => {
      const board = makeBoardState();
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // At minimum, territory gain includes region spaces (2)
      expect(result.territoryGainedByPlayer[1]).toBeGreaterThanOrEqual(2);
    });

    it('handles zero territory gain (empty region)', () => {
      const board = makeBoardState();
      const region = makeTerritory([], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Empty region = no territory gain
      expect(result.territoryGainedByPlayer).toEqual({});
    });

    it('updates board eliminatedRings when eliminating stacks', () => {
      const board = makeBoardState();
      board.eliminatedRings = { 1: 5, 2: 3 };
      addStack(board, pos(5, 5), 2, [2, 2]);
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Original player 1 had 5 eliminated, now should have 5 + 2 = 7
      expect(result.board.eliminatedRings[1]).toBe(7);
      // Player 2's count unchanged
      expect(result.board.eliminatedRings[2]).toBe(3);
    });

    it('initializes eliminatedRings if not present', () => {
      const board = makeBoardState();
      board.eliminatedRings = {}; // Empty
      addStack(board, pos(5, 5), 2, [2]);
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      expect(result.board.eliminatedRings[1]).toBe(1);
    });

    it('eliminates multiple stacks in region', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 1, [1, 1]); // 2 rings
      addStack(board, pos(5, 6), 2, [2, 2, 2]); // 3 rings
      const region = makeTerritory([pos(5, 5), pos(5, 6)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // All stacks eliminated
      expect(result.board.stacks.has('5,5')).toBe(false);
      expect(result.board.stacks.has('5,6')).toBe(false);
      // Total eliminations: 2 + 3 = 5, all credited to player 1
      expect(result.eliminatedRingsByPlayer[1]).toBe(5);
    });

    it('does not mutate original board', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 2, [2]);
      const originalStackCount = board.stacks.size;
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      applyTerritoryRegion(board, region, ctx);

      // Original board unchanged
      expect(board.stacks.size).toBe(originalStackCount);
      expect(board.stacks.has('5,5')).toBe(true);
    });

    it('returns new board object', () => {
      const board = makeBoardState();
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      expect(result.board).not.toBe(board);
      expect(result.board.stacks).not.toBe(board.stacks);
      expect(result.board.markers).not.toBe(board.markers);
      expect(result.board.collapsedSpaces).not.toBe(board.collapsedSpaces);
    });

    it('returns the processed region', () => {
      const board = makeBoardState();
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      expect(result.region).toBe(region);
    });

    it('processes border markers', () => {
      const board = makeBoardState();
      // Create a small enclosed region with surrounding markers
      // Region at (3,3)
      const region = makeTerritory([pos(3, 3)], 1);
      // Add markers around it that would form borders
      addMarker(board, pos(2, 3), 1);
      addMarker(board, pos(4, 3), 1);
      addMarker(board, pos(3, 2), 1);
      addMarker(board, pos(3, 4), 1);
      addCollapsed(board, pos(2, 2), 1);
      addCollapsed(board, pos(4, 2), 1);
      addCollapsed(board, pos(2, 4), 1);
      addCollapsed(board, pos(4, 4), 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // Border markers are returned in the result
      expect(result.borderMarkers).toBeDefined();
    });
  });

  describe('cloneBoard (via applyTerritoryRegion)', () => {
    it('clones all map properties', () => {
      const board = makeBoardState();
      board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      board.markers.set('1,1', { position: pos(1, 1), player: 1, type: 'regular' });
      board.collapsedSpaces.set('2,2', 1);
      board.territories.set('t1', makeTerritory([pos(3, 3)]));
      board.formedLines.push({ startPos: pos(0, 0), endPos: pos(5, 0), direction: { x: 1, y: 0 } });
      const region = makeTerritory([pos(5, 5)], 1);
      const ctx: TerritoryProcessingContext = { player: 1 };

      const result = applyTerritoryRegion(board, region, ctx);

      // All maps should be new instances
      expect(result.board.stacks).not.toBe(board.stacks);
      expect(result.board.markers).not.toBe(board.markers);
      expect(result.board.collapsedSpaces).not.toBe(board.collapsedSpaces);
      expect(result.board.territories).not.toBe(board.territories);
      expect(result.board.formedLines).not.toBe(board.formedLines);
      expect(result.board.eliminatedRings).not.toBe(board.eliminatedRings);
    });
  });
});
