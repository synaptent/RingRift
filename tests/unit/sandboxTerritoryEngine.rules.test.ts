import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { BoardType, GameState, Position, positionToString } from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';
import {
  processDisconnectedRegionsForCurrentPlayerEngine,
  TerritoryInteractionHandler,
} from '../../src/client/sandbox/sandboxTerritoryEngine';

/**
 * Focused rules-layer tests for sandbox territory engine invariants.
 *
 * These tests bypass ClientSandboxEngine and drive the pure
 * `processDisconnectedRegionsForCurrentPlayerEngine` helper directly on a
 * handcrafted GameState with a sandbox-specific canProcessRegion predicate.
 *
 * Compact Q23 mini-region reference:
 * - Rules / FAQ: §12.2, FAQ Q23 (self-elimination prerequisite)
 * - Scenario nickname: `Rules_12_2_Q23_mini_region_square8_numeric_invariant`
 */

describe('sandboxTerritoryEngine.rules – Q23 self-elimination prerequisite (square8)', () => {
  const boardType: BoardType = 'square8';
  const movingPlayer = 1;
  const victimPlayer = 2;

  function makeMiniRegionGeometry() {
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

    const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    // Victim stacks inside the region: height 2 each.
    for (const p of regionSpaces) {
      addStack(board, p, victimPlayer, 2);
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

    return { board, players, regionSpaces };
  }

  /**
   * Negative Q23 case at sandbox engine level: moving player controls a
   * disconnected region but has no stacks outside that region. The
   * canProcessRegion predicate should return false and
   * processDisconnectedRegionsForCurrentPlayerEngine must leave the state
   * unchanged (no region collapse, no eliminations, S-invariant constant).
   */
  it('does not process region when moving player has no outside stack (Q23 negative mini-region)', async () => {
    const { board, players, regionSpaces } = makeMiniRegionGeometry();

    // No moving-player stacks anywhere on the board.
    const gameStateBefore: GameState = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: movingPlayer,
      currentPhase: 'territory_processing',
      totalRingsEliminated: 0,
    });

    const regionKeySet = new Set(regionSpaces.map((p) => positionToString(p)));

    const canProcessRegion = (
      spaces: Position[],
      playerNumber: number,
      state: GameState
    ): boolean => {
      // Enforce the same self-elimination prerequisite as backend Q23:
      // player must have at least one stack outside the region.
      const regionKeys = new Set(spaces.map((p) => positionToString(p)));

      for (const [key, stack] of state.board.stacks.entries()) {
        if (stack.controllingPlayer !== playerNumber) continue;
        if (!regionKeys.has(key)) {
          return true;
        }
      }

      return false;
    };

    // No interaction needed for a single-region scenario.
    const interactionHandler: TerritoryInteractionHandler | null = null;

    const snapshotBefore = computeProgressSnapshot(gameStateBefore);
    const SBefore = snapshotBefore.S;

    const moverBefore = gameStateBefore.players.find((p) => p.playerNumber === movingPlayer)!;
    const territoryBefore = moverBefore.territorySpaces;
    const eliminatedBefore = moverBefore.eliminatedRings;
    const boardElimBefore = gameStateBefore.board.eliminatedRings[movingPlayer] ?? 0;
    const collapsedBefore = gameStateBefore.board.collapsedSpaces.size;

    const { state: stateAfter } = await processDisconnectedRegionsForCurrentPlayerEngine(
      gameStateBefore,
      interactionHandler,
      canProcessRegion
    );

    const snapshotAfter = computeProgressSnapshot(stateAfter);
    const SAfter = snapshotAfter.S;

    const moverAfter = stateAfter.players.find((p) => p.playerNumber === movingPlayer)!;
    const territoryAfter = moverAfter.territorySpaces;
    const eliminatedAfter = moverAfter.eliminatedRings;
    const boardElimAfter = stateAfter.board.eliminatedRings[movingPlayer] ?? 0;
    const collapsedAfter = stateAfter.board.collapsedSpaces.size;

    // Region must remain unprocessed: stacks still present in region, no new collapsed spaces.
    for (const p of regionSpaces) {
      const key = positionToString(p);
      expect(stateAfter.board.stacks.has(key)).toBe(true);
    }
    expect(collapsedAfter).toBe(collapsedBefore);

    // No elimination or territory changes for the moving player.
    expect(territoryAfter).toBe(territoryBefore);
    expect(eliminatedAfter).toBe(eliminatedBefore);
    expect(boardElimAfter).toBe(boardElimBefore);

    // S-invariant remains unchanged.
    expect(SAfter).toBe(SBefore);

    // Sanity: region key set is unchanged.
    const regionKeysAfter = new Set(regionSpaces.map((p) => positionToString(p)));
    expect(regionKeysAfter.size).toBe(regionKeySet.size);
  });
});
