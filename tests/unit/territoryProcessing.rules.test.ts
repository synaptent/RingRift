import { BoardManager } from '../../src/server/game/BoardManager';
import {
  processDisconnectedRegionsForCurrentPlayer,
  TerritoryProcessingDeps,
} from '../../src/server/game/rules/territoryProcessing';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { GameState, Territory, BoardType, positionToString } from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';

/**
 * Focused rules-layer tests for backend territory processing invariants.
 *
 * These tests bypass GameEngine and drive the pure
 * `processDisconnectedRegionsForCurrentPlayer` helper directly on a
 * handcrafted GameState. They assert that for a small, fully-specified
 * disconnected region:
 *
 * - All internal stacks are eliminated and credited to the moving player.
 * - Mandatory self-elimination adds its own cap-height delta.
 * - Territory gain for the moving player is exactly
 *   |region.spaces| + |borderMarkers|.
 * - Elimination accounting (GameState.totalRingsEliminated,
 *   board.eliminatedRings[movingPlayer], and the moving player’s
 *   eliminatedRings) all increase by the same total.
 * - The shared S-invariant (markers + collapsed + eliminated) increases by
 *   region size + internal eliminations + self-elimination cap height.
 *
 * Compact Q23 mini-region reference:
 * - Rules / FAQ: §12.2, FAQ Q23 (self-elimination prerequisite)
 * - Scenario nickname: `Rules_12_2_Q23_mini_region_square8_numeric_invariant`
 */

describe('territoryProcessing.rules – backend invariants (square8)', () => {
  const boardType: BoardType = 'square8';
  const movingPlayer = 1;
  const victimPlayer = 2;

  /**
   * Positive Q23 case: moving player controls a disconnected region and has
   * an outside stack available for mandatory self-elimination. The region
   * MUST be processed.
   *
   * Geometry: a 2×2 interior region at (2,2)–(3,3) surrounded by a single
   * ring of markers for the moving player, mirroring the Q20-style example
   * but scaled down for easier reasoning.
   *
   * Region spaces (no markers, only stacks for the victim):
   *   (2,2), (2,3), (3,2), (3,3)
   *
   * Border markers for the moving player (A) using Von Neumann-style
   * rectangle around the 2×2 block:
   *   x = 1..4, y = 1 and y = 4  → top/bottom border
   *   x = 1 and x = 4, y = 2..3 → left/right border
   *
   * Internal stacks: victim stacks of height 2 in each region space
   *   → internalRingsEliminated = 4 * 2 = 8
   *
   * Mandatory self-elimination: single stack for the moving player at (0,0)
   *   of height 3 (all rings = player 1), so capHeight = 3.
   */
  it('processDisconnectedRegionsForCurrentPlayer applies region processing and self-elimination when outside stack exists (Q23 positive mini-region)', async () => {
    const board = createTestBoard(boardType);

    // Players: movingPlayer (1) and victim (2).
    const players = [
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

    const boardManager = new BoardManager(boardType);

    // Sanity-check the border marker count and derived territory gain.
    const borderMarkers = boardManager.getBorderMarkerPositions(regionSpaces, board);
    const expectedTerritoryGain = regionSpaces.length + borderMarkers.length;
    expect(borderMarkers.length).toBeGreaterThan(0);

    // Mandatory self-elimination stack for moving player outside the region.
    const selfStackPos = pos(0, 0);
    const selfStackHeight = 3;
    addStack(board, selfStackPos, movingPlayer, selfStackHeight);
    const playerStacksBefore = boardManager.getPlayerStacks(board, movingPlayer);
    expect(playerStacksBefore.length).toBeGreaterThan(0);

    // Prepare a minimal GameState.
    let gameState: GameState = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: movingPlayer,
      currentPhase: 'territory_processing',
      totalRingsEliminated: 0,
    });

    const deps: TerritoryProcessingDeps = { boardManager };

    // Stub disconnection detection to focus purely on processing invariants.
    const regionTerritory: Territory = {
      spaces: regionSpaces,
      controllingPlayer: movingPlayer,
      isDisconnected: true,
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    // Capture pre-processing metrics.
    const snapshotBefore = computeProgressSnapshot(gameState);
    const playerBefore = gameState.players.find((p) => p.playerNumber === movingPlayer)!;
    const initialTerritory = playerBefore.territorySpaces;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialPlayerEliminated = playerBefore.eliminatedRings;
    const initialBoardEliminated = gameState.board.eliminatedRings[movingPlayer] ?? 0;

    // S-invariant before (definition-level, via shared helper).
    const SBefore = snapshotBefore.S;

    // Execute the rules-layer helper.
    gameState = await processDisconnectedRegionsForCurrentPlayer(gameState, deps);

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

    const snapshotAfter = computeProgressSnapshot(gameState);
    const playerAfter = gameState.players.find((p) => p.playerNumber === movingPlayer)!;
    const finalTerritory = playerAfter.territorySpaces;
    const finalTotalEliminated = gameState.totalRingsEliminated;
    const finalPlayerEliminated = playerAfter.eliminatedRings;
    const finalBoardEliminated = gameState.board.eliminatedRings[movingPlayer] ?? 0;

    const SAfter = snapshotAfter.S;

    // --- Territory invariants ---
    expect(finalTerritory - initialTerritory).toBe(expectedTerritoryGain);

    for (const p of regionSpaces) {
      const key = positionToString(p);
      expect(gameState.board.stacks.has(key)).toBe(false);
      expect(gameState.board.collapsedSpaces.get(key)).toBe(movingPlayer);
    }

    for (const p of borderMarkers) {
      const key = positionToString(p);
      expect(gameState.board.stacks.has(key)).toBe(false);
      expect(gameState.board.collapsedSpaces.get(key)).toBe(movingPlayer);
    }

    // --- Elimination invariants ---
    const expectedSelfCapHeight = selfStackHeight; // all rings are movingPlayer
    const expectedTotalDelta = expectedInternalRings + expectedSelfCapHeight;

    expect(finalTotalEliminated - initialTotalEliminated).toBe(expectedTotalDelta);
    expect(finalPlayerEliminated - initialPlayerEliminated).toBe(expectedTotalDelta);
    expect(finalBoardEliminated - initialBoardEliminated).toBe(expectedTotalDelta);

    // Outside self-elimination stack has been reduced or removed.
    const selfKey = positionToString(selfStackPos);
    expect(gameState.board.stacks.has(selfKey)).toBe(false);

    // --- S-invariant ---
    const deltaS = SAfter - SBefore;

    // For this curated scenario, S should increase by:
    //   |regionSpaces| (new collapsed spaces)
    // + internal elimination count
    // + self-elimination cap height.
    expect(deltaS).toBe(regionSpaces.length + expectedInternalRings + expectedSelfCapHeight);
    expect(SAfter).toBeGreaterThanOrEqual(SBefore);
  });

  /**
   * Negative Q23 case: moving player controls a disconnected region but has
   * no stacks outside that region. The self-elimination prerequisite fails,
   * so the region MUST NOT be processed.
   */
  it('processDisconnectedRegionsForCurrentPlayer does not process region when moving player has no outside stack (Q23 negative mini-region)', async () => {
    const board = createTestBoard(boardType);

    const players = [
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

    const boardManager = new BoardManager(boardType);

    // Sanity: moving player has no stacks anywhere on the board.
    const stacksForMover = boardManager.getPlayerStacks(board, movingPlayer);
    expect(stacksForMover.length).toBe(0);

    const regionTerritory: Territory = {
      spaces: regionSpaces,
      controllingPlayer: movingPlayer,
      isDisconnected: true,
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    let gameState: GameState = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: movingPlayer,
      currentPhase: 'territory_processing',
      totalRingsEliminated: 0,
    });

    const deps: TerritoryProcessingDeps = { boardManager };

    const snapshotBefore = computeProgressSnapshot(gameState);
    const playerBefore = gameState.players.find((p) => p.playerNumber === movingPlayer)!;
    const initialTerritory = playerBefore.territorySpaces;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialPlayerEliminated = playerBefore.eliminatedRings;
    const initialBoardEliminated = gameState.board.eliminatedRings[movingPlayer] ?? 0;
    const initialCollapsedCount = gameState.board.collapsedSpaces.size;

    const SBefore = snapshotBefore.S;

    // Execute helper – with no outside stack, the region should NOT be processed.
    gameState = await processDisconnectedRegionsForCurrentPlayer(gameState, deps);

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

    const snapshotAfter = computeProgressSnapshot(gameState);
    const playerAfter = gameState.players.find((p) => p.playerNumber === movingPlayer)!;
    const finalTerritory = playerAfter.territorySpaces;
    const finalTotalEliminated = gameState.totalRingsEliminated;
    const finalPlayerEliminated = playerAfter.eliminatedRings;
    const finalBoardEliminated = gameState.board.eliminatedRings[movingPlayer] ?? 0;
    const finalCollapsedCount = gameState.board.collapsedSpaces.size;

    const SAfter = snapshotAfter.S;

    // Region must remain unprocessed: stacks still present in region, no new collapsed spaces.
    for (const p of regionSpaces) {
      const key = positionToString(p);
      expect(gameState.board.stacks.has(key)).toBe(true);
    }
    expect(finalCollapsedCount).toBe(initialCollapsedCount);

    // No elimination or territory changes for the moving player.
    expect(finalTerritory).toBe(initialTerritory);
    expect(finalTotalEliminated).toBe(initialTotalEliminated);
    expect(finalPlayerEliminated).toBe(initialPlayerEliminated);
    expect(finalBoardEliminated).toBe(initialBoardEliminated);

    // S-invariant remains unchanged for this non-processed scenario.
    expect(SAfter).toBe(SBefore);
  });
});
