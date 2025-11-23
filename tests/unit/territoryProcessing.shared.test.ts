import { Territory, positionToString } from '../../src/shared/types/game';
import {
  getProcessableTerritoryRegions,
  filterProcessableTerritoryRegions,
  applyTerritoryRegion,
  TerritoryProcessingContext,
} from '../../src/shared/engine/territoryProcessing';
import { getBorderMarkerPositionsForRegion } from '../../src/shared/engine/territoryBorders';
import { createTestBoard, addStack, addMarker, pos } from '../utils/fixtures';

describe('territoryProcessing.shared', () => {
  describe('single-region collapse on square8', () => {
    const movingPlayer = 1;
    const victimPlayer = 2;

    it('enumerates and applies a basic disconnected region (Q23-style mini)', () => {
      const board = createTestBoard('square8');

      const regionSpaces = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

      const internalStackHeight = 2;
      for (const p of regionSpaces) {
        addStack(board, p, victimPlayer, internalStackHeight);
      }
      const expectedInternalRings = regionSpaces.length * internalStackHeight;

      const borderCoords: Array<[number, number]> = [];
      for (let x = 1; x <= 4; x++) {
        borderCoords.push([x, 1]);
        borderCoords.push([x, 4]);
      }
      for (let y = 2; y <= 3; y++) {
        borderCoords.push([1, y]);
        borderCoords.push([4, y]);
      }
      borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), movingPlayer));

      // Outside stack for the moving player to satisfy self-elimination prerequisite.
      addStack(board, pos(0, 0), movingPlayer, 3);

      const expectedBorderMarkers = getBorderMarkerPositionsForRegion(board, regionSpaces, {
        mode: 'rust_aligned',
      });
      expect(expectedBorderMarkers.length).toBeGreaterThan(0);

      const ctx: TerritoryProcessingContext = { player: movingPlayer };
      const regions = getProcessableTerritoryRegions(board, ctx);
      expect(regions).toHaveLength(1);

      const region = regions[0];
      const regionKeySet = new Set(region.spaces.map((p) => positionToString(p)));
      expect(regionKeySet.size).toBe(regionSpaces.length);
      for (const p of regionSpaces) {
        expect(regionKeySet.has(positionToString(p))).toBe(true);
      }

      const outcome = applyTerritoryRegion(board, region, ctx);

      const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
      expect(territoryGain).toBe(regionSpaces.length + expectedBorderMarkers.length);
      expect(outcome.borderMarkers.map(positionToString)).toEqual(
        expectedBorderMarkers.map(positionToString)
      );

      const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
      expect(internalElims).toBe(expectedInternalRings);
      expect(outcome.board.eliminatedRings[movingPlayer]).toBe(expectedInternalRings);

      for (const p of regionSpaces) {
        const key = positionToString(p);
        expect(outcome.board.stacks.has(key)).toBe(false);
        expect(outcome.board.collapsedSpaces.get(key)).toBe(movingPlayer);
      }
      for (const p of expectedBorderMarkers) {
        const key = positionToString(p);
        expect(outcome.board.stacks.has(key)).toBe(false);
        expect(outcome.board.collapsedSpaces.get(key)).toBe(movingPlayer);
      }
    });
  });

  describe('multi-region gating semantics', () => {
    it('filters processable regions using the self-elimination prerequisite', () => {
      const board = createTestBoard('square8');
      const movingPlayer = 1;
      const otherPlayer = 2;

      const regionANorthWest = [pos(0, 0), pos(0, 1)];
      const regionBSouthEast = [pos(5, 5), pos(5, 6)];

      const regionA: Territory = {
        spaces: regionANorthWest,
        controllingPlayer: movingPlayer,
        isDisconnected: true,
      };
      const regionB: Territory = {
        spaces: regionBSouthEast,
        controllingPlayer: movingPlayer,
        isDisconnected: true,
      };

      // All stacks for the moving player live inside regionA only.
      addStack(board, pos(0, 0), movingPlayer, 2);

      // Other player has a stack elsewhere so the board is non-empty.
      addStack(board, pos(7, 7), otherPlayer, 1);

      const ctx: TerritoryProcessingContext = { player: movingPlayer };
      const regions: Territory[] = [regionA, regionB];

      const eligible = filterProcessableTerritoryRegions(board, regions, ctx);
      expect(eligible).toHaveLength(1);

      const eligibleRegion = eligible[0];
      const eligibleKeys = new Set(eligibleRegion.spaces.map((p) => positionToString(p)));
      // Only regionB should be processable because the moving player has
      // no stack outside regionA but does have a stack outside regionB.
      expect(eligibleKeys.has(positionToString(pos(5, 5)))).toBe(true);
      expect(eligibleKeys.has(positionToString(pos(0, 0)))).toBe(false);
    });
  });

  describe('hex board territory collapse', () => {
    it('collapses hex region spaces and border markers and credits eliminations', () => {
      const board = createTestBoard('hexagonal');
      const movingPlayer = 1;
      const victimPlayer = 2;

      const center = pos(0, 0, 0);
      const neighbor1 = pos(1, -1, 0);
      const neighbor2 = pos(0, -1, 1);

      const regionSpaces = [center];

      addStack(board, center, victimPlayer, 3);
      addMarker(board, neighbor1, movingPlayer);
      addMarker(board, neighbor2, movingPlayer);

      const region: Territory = {
        spaces: regionSpaces,
        controllingPlayer: movingPlayer,
        isDisconnected: true,
      };
      const ctx: TerritoryProcessingContext = { player: movingPlayer };

      const expectedBorder = getBorderMarkerPositionsForRegion(board, regionSpaces, {
        mode: 'rust_aligned',
      });
      expect(expectedBorder.length).toBeGreaterThan(0);

      const outcome = applyTerritoryRegion(board, region, ctx);

      expect(outcome.borderMarkers.map(positionToString)).toEqual(
        expectedBorder.map(positionToString)
      );

      const centerKey = positionToString(center);
      expect(outcome.board.stacks.has(centerKey)).toBe(false);
      expect(outcome.board.collapsedSpaces.get(centerKey)).toBe(movingPlayer);

      for (const p of expectedBorder) {
        const key = positionToString(p);
        expect(outcome.board.stacks.has(key)).toBe(false);
        expect(outcome.board.collapsedSpaces.get(key)).toBe(movingPlayer);
      }

      const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
      expect(internalElims).toBe(3);
      expect(outcome.board.eliminatedRings[movingPlayer]).toBe(3);
    });
  });
});
