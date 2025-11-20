import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  positionToString
} from '../../src/shared/types/game';
import { pos, addStack } from '../utils/fixtures';

/**
 * Territory disconnection scenarios focused on FAQ Q23 (self-elimination prerequisite).
 *
 * These tests isolate GameEngine.processDisconnectedRegions behaviour when a
 * disconnected region exists but the moving player DOES NOT satisfy the
 * self-elimination prerequisite from Section 12.2 / FAQ Q23:
 *
 * - If the moving player has no ring/stack cap outside the region, the region
 *   must not be processed (no collapse, no eliminations).
 */

describe('GameEngine territory disconnection scenarios (Q23)', () => {
  const boardType: BoardType = 'square19';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
    return [
      {
        id: 'p1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'p2',
        username: 'Player2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'p3',
        username: 'Player3',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];
  }

  test('Q23_disconnected_region_illegal_when_no_self_elimination_available_backend', async () => {
    // Rules reference:
    // - Section 12.2 / FAQ Q23: A disconnected region may only be processed if,
    //   hypothetically eliminating all rings in that region, the moving player
    //   would still have at least one ring/stack cap elsewhere to pay the
    //   mandatory self-elimination cost.
    //
    // Scenario:
    // - There is a disconnected region on the board (supplied via stubbed
    //   BoardManager.findDisconnectedRegions).
    // - Player 1 has no stacks outside that region.
    // - Therefore, the region must NOT be processed: no collapse and no
    //   eliminations for player 1.

    const players = createPlayers();
    const engine = new GameEngine('territory-q23', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;
    const boardManager: any = (engineAny as any).boardManager;

    gameState.currentPlayer = 1;

    // Construct a synthetic region: a 3x3 block in the middle of the board.
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
      }
    }

    // Place stacks for another player (2) inside the region.
    for (const p of interiorCoords) {
      addStack(board, p, 2, 1);
    }

    // Crucially, player 1 has NO stacks anywhere on the board. This ensures
    // canProcessDisconnectedRegion will return false for them.
    const p1Stacks = boardManager.getPlayerStacks(board, 1);
    expect(p1Stacks.length).toBe(0);

    // Make sure some other player is active elsewhere so the region is
    // semantically meaningful as a disconnection w.r.t representation.
    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 1);

    // Stub disconnected-region detection so we can focus purely on the
    // self-elimination prerequisite logic inside GameEngine.
    const regionTerritory = {
      spaces: interiorCoords,
      controllingPlayer: 1,
      isDisconnected: true
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    const initialCollapsedCount = board.collapsedSpaces.size;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialP1Eliminated = gameState.players.find(p => p.playerNumber === 1)!.eliminatedRings;

    await (engineAny as any).processDisconnectedRegions();

    // Ensure detection was invoked.
    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

    // Because player 1 has no stacks outside the region (in fact none at all),
    // the region should NOT be processed:
    // - No new collapsed spaces.
    // - Interior stacks remain.
    // - No eliminations credited to player 1.
    expect(board.collapsedSpaces.size).toBe(initialCollapsedCount);

    const stacksInRegion = Array.from(board.stacks.keys()).filter(key => {
      return interiorCoords.some(p => positionToString(p) === key);
    });
    expect(stacksInRegion.length).toBe(interiorCoords.length);

    const finalTotalEliminated = gameState.totalRingsEliminated;
    const finalP1Eliminated = gameState.players.find(p => p.playerNumber === 1)!.eliminatedRings;
    expect(finalTotalEliminated).toBe(initialTotalEliminated);
    expect(finalP1Eliminated).toBe(initialP1Eliminated);
  });

  test('Q23_disconnected_region_processed_when_self_elimination_available_backend', async () => {
    // Complementary scenario for Section 12.2 / FAQ Q23:
    //
    // When the moving player DOES have at least one ring/stack cap outside a
    // disconnected region, the region should be processed:
    // - All spaces in the region collapse to the moving player's colour.
    // - All rings inside the region are eliminated.
    // - Exactly one of the moving player's rings/caps outside the region is
    //   eliminated to pay the self-elimination cost.

    const players = createPlayers();
    const engine = new GameEngine('territory-q23-process', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;
    const boardManager: any = (engineAny as any).boardManager;

    gameState.currentPlayer = 1;

    // Synthetic 3x3 interior region as before.
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        interiorCoords.push(pos(x, y));
      }
    }

    // Place stacks for another player (2) inside the region so that the
    // region is semantically meaningful as a disconnection once surfaced by
    // BoardManager.findDisconnectedRegions.
    for (const p of interiorCoords) {
      addStack(board, p, 2, 1);
    }

    // Give player 1 a single stack OUTSIDE the region so they can satisfy the
    // self-elimination prerequisite.
    const p1Outside = pos(0, 1);
    addStack(board, p1Outside, 1, 2);

    const p1StacksOutside = boardManager.getPlayerStacks(board, 1);
    expect(p1StacksOutside.length).toBeGreaterThan(0);

    // Stub disconnected-region detection exactly as in the previous test so
    // we isolate GameEngine.processDisconnectedRegions behaviour.
    const regionTerritory = {
      spaces: interiorCoords,
      controllingPlayer: 1,
      isDisconnected: true
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    const initialCollapsedCount = board.collapsedSpaces.size;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialP1Eliminated = gameState.players.find(p => p.playerNumber === 1)!.eliminatedRings;

    await (engineAny as any).processDisconnectedRegions();

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

    // Region MUST be processed:
    // - All interior spaces collapsed for player 1.
    // - No stacks remain on those spaces.
    for (const p of interiorCoords) {
      const key = positionToString(p);
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.has(key)).toBe(false);
    }

    const finalTotalEliminated = gameState.totalRingsEliminated;
    const finalP1Eliminated = gameState.players.find(p => p.playerNumber === 1)!.eliminatedRings;

    expect(board.collapsedSpaces.size).toBeGreaterThan(initialCollapsedCount);
    expect(finalTotalEliminated).toBeGreaterThan(initialTotalEliminated);
    expect(finalP1Eliminated).toBeGreaterThan(initialP1Eliminated);
  });

  test('territory_processing_getValidMoves_exposes_process_territory_region_moves', () => {
    // Rules reference:
    // - Section 12.2–12.3 / FAQ Q23: when disconnected regions exist and the
    //   self-elimination prerequisite is satisfied, the engine should surface
    //   explicit territory-processing decisions for the moving player.
    //
    // This test exercises the territory_processing branch of GameEngine.getValidMoves:
    // - We stub BoardManager.findDisconnectedRegions to return two regions for
    //   the moving player.
    // - We ensure the moving player has at least one stack outside those regions
    //   so canProcessDisconnectedRegion returns true.
    // - getValidMoves(currentPlayer) must then return one process_territory_region
    //   Move per eligible region when currentPhase === 'territory_processing'.
    const players = createPlayers();
    const engine = new GameEngine('territory-q23-enumeration', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;
    const boardManager: any = (engineAny as any).boardManager;

    gameState.currentPlayer = 1;
    (gameState as any).currentPhase = 'territory_processing';

    // Clear any existing board state for a clean scenario.
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Create a single outside stack for player 1 so the self-elimination
    // prerequisite is satisfied for any region that does not include this pos.
    const p1Outside: Position = pos(0, 1);
    addStack(board, p1Outside, 1, 2);

    // Define two synthetic disconnected regions for the moving player. We do
    // not rely on BoardManager's geometry here; instead we stub the return
    // value of findDisconnectedRegions directly.
    const region1Spaces: Position[] = [pos(5, 5)];
    const region2Spaces: Position[] = [pos(10, 10)];

    const region1 = {
      spaces: region1Spaces,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const region2 = {
      spaces: region2Spaces,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementation(() => [region1, region2]);

    const moves = engine.getValidMoves(1);

    const territoryMoves = moves.filter((m) => m.type === 'process_territory_region');

    // One process_territory_region per eligible disconnected region.
    expect(territoryMoves).toHaveLength(2);
    expect(territoryMoves.every((m) => m.player === 1)).toBe(true);

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();
  });
});
