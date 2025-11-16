import { BoardManager } from '../../src/server/game/BoardManager';
import {
  createTestBoard,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos
} from '../utils/fixtures';

/**
 * Territory disconnection scenarios derived from ringrift_complete_rules.md
 *
 * These tests exercise the BoardManager.findDisconnectedRegions logic against
 * concrete patterns described in Section 12 (Area Disconnection & Collapse)
 * and the 19×19 examples (e.g. 16.9.6 / 16.9.8), using Von Neumann
 * adjacency for territory and single-color marker borders.
 */

describe('BoardManager territory disconnection (square19, Von Neumann)', () => {
  let boardManager: BoardManager;

  beforeEach(() => {
    boardManager = new BoardManager('square19');
  });

  it('detects a region surrounded by a single-color marker border as disconnected when a color is missing (Section 12.1 / Q15)', () => {
    const board = createTestBoard('square19');

    // Three players: 1 (A), 2 (B), 3 (C)
    // Region of interest: a 3×3 block from (5,5)–(7,7) containing only
    // stacks for player 2 (B). Player 3 (C) has representation elsewhere
    // on the board but not inside this region. Player 1 (A) forms a
    // continuous marker border around the region. By the rules:
    //   * Physical disconnection: border consists solely of A markers
    //     (plus edges/empty beyond)
    //   * Representation: region lacks at least one active color (C)
    //   → Region should be considered disconnected.

    // Interior B stacks (3×3 block)
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        addStack(board, pos(x, y), 2, 1);
      }
    }

    // A marker border around the 3×3 block using Von Neumann adjacency.
    const borderCoords: Array<[number, number]> = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push([x, 4]);
      borderCoords.push([x, 8]);
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push([4, y]);
      borderCoords.push([8, y]);
    }
    borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), 1));

    // Player 3 (C) has a stack elsewhere on the board, so C is an
    // "active" color but not represented inside the region.
    addStack(board, pos(0, 0), 3, 1);

    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);

    // We expect at least one disconnected region whose spaces are exactly
    // the interior 3×3 block.
    const interiorKeys = new Set(
      Array.from({ length: 3 }, (_, dx) => 5 + dx).flatMap(x =>
        Array.from({ length: 3 }, (_, dy) => 5 + dy).map(y => `${x},${y}`)
      )
    );

    const disconnectedInterior = regions.find(region => {
      if (!region.isDisconnected) return false;
      const keys = new Set(region.spaces.map(p => `${p.x},${p.y}`));
      if (keys.size !== interiorKeys.size) return false;
      for (const k of interiorKeys) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(disconnectedInterior).toBeDefined();
  });

  it('detects regions bounded only by collapsed spaces and edges as disconnected when a color is missing (Section 12.2)', () => {
    const board = createTestBoard('square19');

    // Similar pattern as the previous test, but the border is composed of
    // collapsed spaces instead of markers. This exercises the
    // findRegionsWithoutMarkerBorder path and the
    // isRegionBorderedByCollapsedOnly check.

    // Interior B stacks (3×3 block)
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        addStack(board, pos(x, y), 2, 1);
      }
    }

    // Collapsed border around the region
    const borderCoords: Array<[number, number]> = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push([x, 4]);
      borderCoords.push([x, 8]);
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push([4, y]);
      borderCoords.push([8, y]);
    }
    borderCoords.forEach(([x, y]) => addCollapsedSpace(board, pos(x, y), 1));

    // Active players: B (inside) and C (outside). Region lacks C.
    addStack(board, pos(0, 0), 3, 1);

    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);

    const interiorKeys = new Set(
      Array.from({ length: 3 }, (_, dx) => 5 + dx).flatMap(x =>
        Array.from({ length: 3 }, (_, dy) => 5 + dy).map(y => `${x},${y}`)
      )
    );

    const disconnectedInterior = regions.find(region => {
      if (!region.isDisconnected) return false;
      const keys = new Set(region.spaces.map(p => `${p.x},${p.y}`));
      if (keys.size !== interiorKeys.size) return false;
      for (const k of interiorKeys) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(disconnectedInterior).toBeDefined();
  });

  it('does NOT treat a region as disconnected when all active colors are represented inside (FAQ Q15 important exception)', () => {
    const board = createTestBoard('square19');

    // Three players: 1 (A), 2 (B), 3 (C)
    // Same 3×3 interior region, but now we place stacks for all three
    // players inside it. Physically, A still forms a marker border, but
    // representation includes all active colors, so by rules this region
    // must NOT be considered disconnected.

    const interiorPlayers = [1, 2, 3];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const player = interiorPlayers[(x + y) % interiorPlayers.length];
        addStack(board, pos(x, y), player, 1);
      }
    }

    const borderCoords: Array<[number, number]> = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push([x, 4]);
      borderCoords.push([x, 8]);
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push([4, y]);
      borderCoords.push([8, y]);
    }
    borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), 1));

    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);

    const interiorKeys = new Set(
      Array.from({ length: 3 }, (_, dx) => 5 + dx).flatMap(x =>
        Array.from({ length: 3 }, (_, dy) => 5 + dy).map(y => `${x},${y}`)
      )
    );

    const anyMatchingInterior = regions.some(region => {
      if (!region.isDisconnected) return false;
      const keys = new Set(region.spaces.map(p => `${p.x},${p.y}`));
      if (keys.size !== interiorKeys.size) return false;
      for (const k of interiorKeys) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(anyMatchingInterior).toBe(false);
  });

  it('detects multiple disconnected regions under a shared marker border color', () => {
    const board = createTestBoard('square19');

    // Two disjoint 3×3 interior regions of B stacks, each surrounded
    // by an A marker border, with C active elsewhere. Both regions
    // should be detected as disconnected according to the same
    // representation rule as the single-region test.

    const makeInteriorBlock = (x0: number, y0: number) => {
      const coords: Array<[number, number]> = [];
      for (let x = x0; x <= x0 + 2; x++) {
        for (let y = y0; y <= y0 + 2; y++) {
          addStack(board, pos(x, y), 2, 1);
          coords.push([x, y]);
        }
      }
      return coords;
    };

    const block1 = makeInteriorBlock(5, 5);
    const block2 = makeInteriorBlock(11, 5);

    const makeBorder = (x0: number, y0: number) => {
      const border: Array<[number, number]> = [];
      for (let x = x0 - 1; x <= x0 + 3; x++) {
        border.push([x, y0 - 1]);
        border.push([x, y0 + 3]);
      }
      for (let y = y0; y <= y0 + 2; y++) {
        border.push([x0 - 1, y]);
        border.push([x0 + 3, y]);
      }
      border.forEach(([bx, by]) => addMarker(board, pos(bx, by), 1));
    };

    makeBorder(5, 5);
    makeBorder(11, 5);

    // C active elsewhere but not inside either region
    addStack(board, pos(0, 0), 3, 1);

    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);

    const blockKeys = (coords: Array<[number, number]>) =>
      new Set(coords.map(([x, y]) => `${x},${y}`));

    const keys1 = blockKeys(block1);
    const keys2 = blockKeys(block2);

    const matchRegion = (keys: Set<string>) =>
      regions.find(region => {
        if (!region.isDisconnected) return false;
        const rkeys = new Set(region.spaces.map(p => `${p.x},${p.y}`));
        if (rkeys.size !== keys.size) return false;
        for (const k of keys) {
          if (!rkeys.has(k)) return false;
        }
        return true;
      });

    expect(matchRegion(keys1)).toBeDefined();
    expect(matchRegion(keys2)).toBeDefined();
  });

  it('returns no disconnected regions on an empty board', () => {
    const board = createTestBoard('square19');
    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);
    expect(regions).toHaveLength(0);
  });
});
