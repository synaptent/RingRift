import { BoardManager } from '../../src/server/game/BoardManager';
import { createTestBoard, addStack, addMarker, pos } from '../utils/fixtures';

/**
 * BoardManager territory disconnection (square8, Von Neumann adjacency)
 *
 * FAQ Q20 emphasises that territory connectivity on both 8×8 and 19×19 uses
 * Von Neumann (4-direction orthogonal) adjacency, even though movement and
 * lines use Moore (8-direction). This suite mirrors the canonical 3×3
 * disconnection example from the 19×19 rules on the smaller 8×8 board.
 */

describe('BoardManager territory disconnection (square8, Von Neumann)', () => {
  let boardManager: BoardManager;

  beforeEach(() => {
    boardManager = new BoardManager('square8');
  });

  it('Q20_square8_von_neumann_territory_disconnection_backend', () => {
    const board = createTestBoard('square8');

    // Three players: 1 (A), 2 (B), 3 (C).
    // Region: 3×3 block from (2,2)–(4,4) containing only stacks for B.
    // A forms a continuous single-colour marker border around the
    // region. C has a stack elsewhere on the board but not inside the
    // region. Under Von Neumann adjacency, this region is physically
    // disconnected and lacks C representation, so it must be marked
    // disconnected for the moving player.

    // Interior B stacks (3×3 block)
    for (let x = 2; x <= 4; x++) {
      for (let y = 2; y <= 4; y++) {
        addStack(board, pos(x, y), 2, 1);
      }
    }

    // A marker border around the 3×3 block using Von Neumann adjacency.
    const borderCoords: Array<[number, number]> = [];
    for (let x = 1; x <= 5; x++) {
      borderCoords.push([x, 1]);
      borderCoords.push([x, 5]);
    }
    for (let y = 2; y <= 4; y++) {
      borderCoords.push([1, y]);
      borderCoords.push([5, y]);
    }
    borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), 1));

    // Player 3 (C) has a stack elsewhere, so C is an active colour but
    // not represented inside the region.
    addStack(board, pos(0, 0), 3, 1);

    const regions = boardManager.findDisconnectedRegions(board, /*movingPlayer*/ 1);

    const interiorKeys = new Set(
      Array.from({ length: 3 }, (_, dx) => 2 + dx).flatMap((x) =>
        Array.from({ length: 3 }, (_, dy) => 2 + dy).map((y) => `${x},${y}`)
      )
    );

    const disconnectedInterior = regions.find((region) => {
      if (!region.isDisconnected) return false;
      const keys = new Set(region.spaces.map((p) => `${p.x},${p.y}`));
      if (keys.size !== interiorKeys.size) return false;
      for (const k of interiorKeys) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(disconnectedInterior).toBeDefined();
  });
});
