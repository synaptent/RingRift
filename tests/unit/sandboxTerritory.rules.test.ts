import {
  findDisconnectedRegionsOnBoard,
  getBorderMarkerPositionsForRegion,
  processDisconnectedRegionOnBoard,
} from '../../src/client/sandbox/sandboxTerritory';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  BoardType,
  BoardState,
  GameState,
  Player,
  positionToString,
} from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';

/**
 * Focused rules-layer tests for sandbox territory processing invariants.
 *
 * These tests bypass ClientSandboxEngine and drive the pure
 * `processDisconnectedRegionOnBoard` helper directly on handcrafted
 * BoardState/Player structures. They mirror the backend
 * territoryProcessing.rules test geometry and assert that for a small,
 * fully-specified disconnected region:
 *
 * - findDisconnectedRegionsOnBoard discovers a region matching the curated
 *   interior coordinates.
 * - All internal stacks are eliminated and credited to the moving player.
 * - Mandatory self-elimination adds its own cap-height delta via
 *   forceEliminateCapOnBoard.
 * - Territory gain for the moving player is exactly
 *   |region.spaces| + |borderMarkers|.
 * - Elimination accounting (board.eliminatedRings[movingPlayer] and the
 *   moving player’s eliminatedRings) both increase by the same
 *   totalRingsEliminatedDelta reported by the helper.
 * - When wrapped in a minimal GameState, the shared S-invariant helper
 *   (markers + collapsed + eliminated) increases by
 *   |region.spaces| + totalRingsEliminatedDelta.
 */

describe('sandboxTerritory.rules – sandbox invariants (square8)', () => {
  const boardType: BoardType = 'square8';
  const movingPlayer = 1;
  const victimPlayer = 2;

  /**
   * Geometry: same 2×2 interior region as the backend rules test.
   *
   * Region spaces (no markers, only stacks for the victim):
   *   (2,2), (2,3), (3,2), (3,3)
   *
   * Border markers for the moving player (A) using a rectangle around the
   * 2×2 block:
   *   x = 1..4, y = 1 and y = 4  → top/bottom border
   *   x = 1 and x = 4, y = 2..3 → left/right border
   *
   * Internal stacks: victim stacks of height 2 in each region space
   *   → internalRingsEliminated = 4 * 2 = 8
   *
   * Mandatory self-elimination: single stack for the moving player at (0,0)
   *   of height 3 (all rings = player 1), so capHeight = 3 in the
   *   forceEliminateCapOnBoard helper.
   */
  it('processDisconnectedRegionOnBoard preserves sandbox elimination and territory invariants', () => {
    const board: BoardState = createTestBoard(boardType);

    // Players: movingPlayer (1) and victim (2).
    const players: Player[] = [
      createTestPlayer(movingPlayer, {
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      }),
      createTestPlayer(victimPlayer, {
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      }),
    ];

    const regionSpaces = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    // Victim stacks inside the region: height 2 each.
    const internalStackHeight = 2;
    for (const p of regionSpaces) {
      addStack(board, p, victimPlayer, internalStackHeight);
    }
    const expectedInternalRings = regionSpaces.length * internalStackHeight; // 4 * 2 = 8

    // Marker border for moving player around the region.
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

    // Mandatory self-elimination stack for moving player outside the region.
    const selfStackPos = pos(0, 0);
    const selfStackHeight = 3;
    addStack(board, selfStackPos, movingPlayer, selfStackHeight);

    // Sanity: disconnected region detection should discover a region whose
    // spaces match the curated interior set under the sandbox helper.
    const disconnected = findDisconnectedRegionsOnBoard(board);
    expect(disconnected.length).toBeGreaterThan(0);

    const interiorKeySet = new Set(regionSpaces.map((p) => positionToString(p)));
    const matchingRegion = disconnected.find((r) => {
      const keys = new Set(r.spaces.map((p) => positionToString(p)));
      if (keys.size !== interiorKeySet.size) return false;
      for (const k of interiorKeySet) {
        if (!keys.has(k)) return false;
      }
      return true;
    });

    expect(matchingRegion).toBeDefined();

    // Border markers seen by the sandbox helper for this region.
    const borderMarkers = getBorderMarkerPositionsForRegion(board, regionSpaces);
    const expectedTerritoryGain = regionSpaces.length + borderMarkers.length;
    expect(borderMarkers.length).toBeGreaterThan(0);

    // Capture pre-processing metrics.
    const movingBefore = players.find((p) => p.playerNumber === movingPlayer)!;
    const territoryBefore = movingBefore.territorySpaces;
    const playerElimBefore = movingBefore.eliminatedRings;
    const boardElimBefore = board.eliminatedRings[movingPlayer] ?? 0;

    // S-invariant snapshot before, via a thin GameState wrapper.
    const gameStateBefore: GameState = createTestGameState({
      boardType,
      board,
      players,
      totalRingsEliminated: 0,
    });
    const snapshotBefore = computeProgressSnapshot(gameStateBefore);
    const SBefore = snapshotBefore.S;

    // Execute the sandbox helper.
    const result = processDisconnectedRegionOnBoard(board, players, movingPlayer, regionSpaces);

    const { board: boardAfter, players: playersAfter, totalRingsEliminatedDelta } = result;

    const movingAfter = playersAfter.find((p) => p.playerNumber === movingPlayer)!;
    const territoryAfter = movingAfter.territorySpaces;
    const playerElimAfter = movingAfter.eliminatedRings;
    const boardElimAfter = boardAfter.eliminatedRings[movingPlayer] ?? 0;

    // --- Territory invariants ---
    expect(territoryAfter - territoryBefore).toBe(expectedTerritoryGain);

    for (const p of regionSpaces) {
      const key = positionToString(p);
      expect(boardAfter.stacks.has(key)).toBe(false);
      expect(boardAfter.collapsedSpaces.get(key)).toBe(movingPlayer);
    }

    for (const p of borderMarkers) {
      const key = positionToString(p);
      expect(boardAfter.stacks.has(key)).toBe(false);
      expect(boardAfter.collapsedSpaces.get(key)).toBe(movingPlayer);
    }

    // --- Elimination invariants ---
    const expectedSelfCapHeight = selfStackHeight; // all rings are movingPlayer
    const expectedTotalDelta = expectedInternalRings + expectedSelfCapHeight;

    expect(totalRingsEliminatedDelta).toBe(expectedTotalDelta);
    expect(playerElimAfter - playerElimBefore).toBe(expectedTotalDelta);
    expect(boardElimAfter - boardElimBefore).toBe(expectedTotalDelta);

    // Outside self-elimination stack has been reduced or removed.
    const selfKey = positionToString(selfStackPos);
    expect(boardAfter.stacks.has(selfKey)).toBe(false);

    // --- S-invariant ---
    const gameStateAfter: GameState = createTestGameState({
      boardType,
      board: boardAfter,
      players: playersAfter,
      totalRingsEliminated: totalRingsEliminatedDelta,
    });
    const snapshotAfter = computeProgressSnapshot(gameStateAfter);
    const SAfter = snapshotAfter.S;

    const deltaS = SAfter - SBefore;

    // For this curated scenario, S should increase by:
    //   |regionSpaces| (new collapsed spaces)
    // + totalRingsEliminatedDelta (internal + forced self-elimination).
    expect(deltaS).toBe(regionSpaces.length + totalRingsEliminatedDelta);
    expect(SAfter).toBeGreaterThanOrEqual(SBefore);
  });
});
