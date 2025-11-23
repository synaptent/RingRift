import { getBorderMarkerPositionsForRegion } from '../../src/shared/engine/territoryBorders';
import type { BoardState, Position } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { createTestBoard, addMarker, addStack, pos } from '../utils/fixtures';

function normalizePositions(positions: Position[]): string[] {
  return positions.map(positionToString).sort();
}

describe('territoryBorders.shared - square8 border markers', () => {
  it('computes border markers for a 2x2 interior region with a rectangular marker ring', () => {
    const board: BoardState = createTestBoard('square8');
    const movingPlayer = 1;
    const victimPlayer = 2;

    const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    for (const p of regionSpaces) {
      addStack(board, p, victimPlayer, 1);
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

    const expectedCoords: Array<[number, number]> = [
      [1, 1],
      [2, 1],
      [3, 1],
      [4, 1],
      [1, 2],
      [4, 2],
      [1, 3],
      [4, 3],
      [1, 4],
      [2, 4],
      [3, 4],
      [4, 4],
    ];
    const expectedPositions = expectedCoords.map(([x, y]) => pos(x, y));

    const result = getBorderMarkerPositionsForRegion(board, regionSpaces);
    const actualStrings = normalizePositions(result);
    const expectedStrings = normalizePositions(expectedPositions);

    expect(actualStrings).toEqual(expectedStrings);
    expect(new Set(actualStrings).size).toBe(actualStrings.length);
  });
});

describe('territoryBorders.shared - hexagonal border markers', () => {
  it('uses seed-only expansion for hex boards (no flood beyond territory-adjacent markers)', () => {
    const board: BoardState = createTestBoard('hexagonal');
    const movingPlayer = 1;

    const regionSpaces: Position[] = [pos(0, 0, 0), pos(1, -1, 0), pos(0, -1, 1)];

    const seedPositions: Position[] = [pos(0, 1, -1), pos(-1, 1, 0), pos(-1, 0, 1)];

    const extraConnectedMarker = pos(-2, 2, 0);

    for (const p of [...seedPositions, extraConnectedMarker]) {
      addMarker(board, p, movingPlayer);
    }

    const result = getBorderMarkerPositionsForRegion(board, regionSpaces);
    const actualStrings = normalizePositions(result);
    const expectedStrings = normalizePositions(seedPositions);

    expect(actualStrings).toEqual(expectedStrings);
    expect(actualStrings).not.toContain(positionToString(extraConnectedMarker));
  });
});

describe('territoryBorders.shared - edge-adjacent regions', () => {
  it('handles a region touching the board edge and expands border markers correctly', () => {
    const board: BoardState = createTestBoard('square8');
    const movingPlayer = 1;

    const regionSpaces: Position[] = [pos(0, 1)];

    const markerCoords: Array<[number, number]> = [
      [1, 1],
      [0, 0],
      [0, 2],
      [1, 0],
      [1, 2],
    ];
    for (const [x, y] of markerCoords) {
      addMarker(board, pos(x, y), movingPlayer);
    }

    const result = getBorderMarkerPositionsForRegion(board, regionSpaces);
    const actualStrings = normalizePositions(result);
    const expectedStrings = normalizePositions(markerCoords.map(([x, y]) => pos(x, y)));

    expect(actualStrings).toEqual(expectedStrings);
    for (const p of result) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeGreaterThanOrEqual(0);
    }
  });
});
