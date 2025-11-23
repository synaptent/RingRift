import { BoardManager } from '../../src/server/game/BoardManager';
import * as sandboxTerritory from '../../src/client/sandbox/sandboxTerritory';
import {
  BoardState,
  BoardType,
  Position,
  Territory,
  positionToString,
} from '../../src/shared/types/game';
import { getBorderMarkerPositionsForRegion as getSharedBorderMarkers } from '../../src/shared/engine/territoryBorders';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../../src/shared/engine/territoryDetection';
import { createTestBoard, addMarker, addStack, pos } from '../utils/fixtures';

function normalizePositions(positions: Position[]): string[] {
  return positions.map(positionToString).sort();
}

function normalizeRegions(regions: Territory[]): string[] {
  return regions.map((r) => r.spaces.map(positionToString).sort().join('|')).sort();
}

describe('TerritoryBorders.Backend_vs_Sandbox - border helper parity', () => {
  it('square8: backend, sandbox, and shared helpers agree on border markers for a 2x2 interior region', () => {
    const board: BoardState = createTestBoard('square8');
    const boardType: BoardType = board.type;

    const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    const borderCoords: Array<[number, number]> = [];
    for (let x = 1; x <= 4; x++) {
      borderCoords.push([x, 1]);
      borderCoords.push([x, 4]);
    }
    for (let y = 2; y <= 3; y++) {
      borderCoords.push([1, y]);
      borderCoords.push([4, y]);
    }

    const borderPlayer = 1;
    for (const [x, y] of borderCoords) {
      addMarker(board, pos(x, y), borderPlayer);
    }

    const shared = getSharedBorderMarkers(board, regionSpaces, { mode: 'rust_aligned' });
    const backendManager = new BoardManager(boardType);
    const backend = backendManager.getBorderMarkerPositions(regionSpaces, board);
    const sandbox = sandboxTerritory.getBorderMarkerPositionsForRegion(board, regionSpaces);

    const sharedKeys = normalizePositions(shared);
    const backendKeys = normalizePositions(backend);
    const sandboxKeys = normalizePositions(sandbox);

    expect(backendKeys).toEqual(sharedKeys);
    expect(sandboxKeys).toEqual(sharedKeys);
    expect(new Set(sharedKeys).size).toBe(sharedKeys.length);
  });

  it('hexagonal: backend, sandbox, and shared helpers use seed-only expansion and agree on border markers', () => {
    const board: BoardState = createTestBoard('hexagonal');
    const boardType: BoardType = board.type;

    const regionSpaces: Position[] = [pos(0, 0, 0), pos(1, -1, 0), pos(0, -1, 1)];

    const seedPositions: Position[] = [pos(0, 1, -1), pos(-1, 1, 0), pos(-1, 0, 1)];

    const extraConnectedMarker = pos(-2, 2, 0);

    const borderPlayer = 1;
    for (const p of [...seedPositions, extraConnectedMarker]) {
      addMarker(board, p, borderPlayer);
    }

    const shared = getSharedBorderMarkers(board, regionSpaces, { mode: 'rust_aligned' });
    const backendManager = new BoardManager(boardType);
    const backend = backendManager.getBorderMarkerPositions(regionSpaces, board);
    const sandbox = sandboxTerritory.getBorderMarkerPositionsForRegion(board, regionSpaces);

    const sharedKeys = normalizePositions(shared);
    const backendKeys = normalizePositions(backend);
    const sandboxKeys = normalizePositions(sandbox);

    expect(backendKeys).toEqual(sharedKeys);
    expect(sandboxKeys).toEqual(sharedKeys);

    const extraKey = positionToString(extraConnectedMarker);
    expect(sharedKeys).not.toContain(extraKey);
    expect(backendKeys).not.toContain(extraKey);
    expect(sandboxKeys).not.toContain(extraKey);

    expect(new Set(sharedKeys).size).toBe(sharedKeys.length);
  });
});

describe('TerritoryBorders.Backend_vs_Sandbox - disconnected-region detection wrappers', () => {
  it('square8: shared, sandbox, and backend detectors agree on disconnected-region spaces', () => {
    const board: BoardState = createTestBoard('square8');
    const boardType: BoardType = board.type;
    const movingPlayer = 1;
    const victimPlayer = 2;

    const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    const internalHeight = 2;
    for (const p of regionSpaces) {
      addStack(board, p, victimPlayer, internalHeight);
    }

    const borderCoords: Array<[number, number]> = [];
    for (let x = 1; x <= 4; x++) {
      borderCoords.push([x, 1]);
      borderCoords.push([x, 4]);
    }
    for (let y = 2; y <= 3; y++) {
      borderCoords.push([1, y]);
      borderCoords.push([4, y]);
    }

    for (const [x, y] of borderCoords) {
      addMarker(board, pos(x, y), movingPlayer);
    }

    addStack(board, pos(0, 0), movingPlayer, 3);

    const sharedRegions = findDisconnectedRegionsShared(board);
    const sandboxRegions = sandboxTerritory.findDisconnectedRegionsOnBoard(board);
    const backendManager = new BoardManager(boardType);
    const backendRegions = backendManager.findDisconnectedRegions(board, movingPlayer);

    const sharedNorm = normalizeRegions(sharedRegions);
    const sandboxNorm = normalizeRegions(sandboxRegions);
    const backendNorm = normalizeRegions(backendRegions);

    expect(sharedNorm.length).toBeGreaterThan(0);
    expect(sharedNorm).toEqual(sandboxNorm);
    expect(sharedNorm).toEqual(backendNorm);

    const targetKeySet = new Set(regionSpaces.map(positionToString));
    const hasExpectedRegion = sharedRegions.some((region) => {
      const keys = new Set(region.spaces.map(positionToString));
      if (keys.size !== targetKeySet.size) return false;
      for (const k of targetKeySet) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(hasExpectedRegion).toBe(true);
  });

  it('hexagonal: detectors agree when there are no disconnected regions', () => {
    const board: BoardState = createTestBoard('hexagonal');
    const boardType: BoardType = board.type;

    addStack(board, pos(0, 0, 0), 1, 1);
    addStack(board, pos(1, -1, 0), 2, 1);

    const sharedRegions = findDisconnectedRegionsShared(board);
    const sandboxRegions = sandboxTerritory.findDisconnectedRegionsOnBoard(board);
    const backendManager = new BoardManager(boardType);
    const backendRegions = backendManager.findDisconnectedRegions(board, 1);

    const sharedNorm = normalizeRegions(sharedRegions);
    const sandboxNorm = normalizeRegions(sandboxRegions);
    const backendNorm = normalizeRegions(backendRegions);

    expect(sharedNorm).toEqual([]);
    expect(sandboxNorm).toEqual([]);
    expect(backendNorm).toEqual([]);
  });
});
