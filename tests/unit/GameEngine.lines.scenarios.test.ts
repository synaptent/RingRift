import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString
} from '../../src/shared/types/game';

/**
 * Backend line-formation scenario tests aligned with rules/FAQ.
 *
 * These focus on:
 * - Section 11 (Line Formation & Collapse)
 * - FAQ Q7 (exact-length lines)
 * - FAQ Q22 (graduated line rewards and Option 1 vs Option 2)
 */

describe('GameEngine line formation scenarios (square8)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
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
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  function createEngine(): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: any;
  } {
    const engine = new GameEngine('lines-scenarios', boardType, basePlayers, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;
    return { engine, gameState, boardManager };
  }

  function makeStack(
    boardManager: any,
    gameState: GameState,
    playerNumber: number,
    height: number,
    position: Position
  ) {
    const rings = Array(height).fill(playerNumber);
    const stack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  test('Q7_exact_length_line_collapse_backend', async () => {
    // Rules reference:
    // - Section 11.2: exact-length line → collapse all markers + eliminate ring/cap.
    // - FAQ Q7: exact-length lines always require elimination.
    //
    // This test focuses on GameEngine.processLineFormations semantics and
    // uses a BoardManager.findAllLines spy to supply a canonical line,
    // leaving geometric line detection to BoardManager unit tests.
    const { engine, gameState, boardManager } = createEngine();
    const engineAny: any = engine;

    gameState.currentPlayer = 1;
    const board = gameState.board;

    // Clear any existing markers/stacks/collapsed spaces.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Synthetic exact-length line for player 1 at y=1.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      linePositions.push({ x: i, y: 1 });
    }

    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy
      .mockImplementationOnce(() => [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
          direction: { x: 1, y: 0 }
        }
      ])
      .mockImplementation(() => []);

    // Add a stack for player 1 so there is a cap to eliminate.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(boardManager, gameState, 1, 2, stackPos);

    const player1Before = gameState.players.find(p => p.playerNumber === 1)!;
    const initialTerritory = player1Before.territorySpaces;
    const initialEliminated = player1Before.eliminatedRings;
    const initialTotalEliminated = gameState.totalRingsEliminated;

    // Invoke backend line processing.
    await engineAny.processLineFormations();

    const player1After = gameState.players.find(p => p.playerNumber === 1)!;

    // All line positions should now be collapsed spaces for p1.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.markers.has(key)).toBe(false);
      expect(board.stacks.has(key)).toBe(false);
    }

    // The elimination stack should have been removed, and eliminated ring
    // counts should have increased.
    expect(board.stacks.get(positionToString(stackPos))).toBeUndefined();
    expect(player1After.eliminatedRings).toBeGreaterThan(initialEliminated);
    expect(gameState.totalRingsEliminated).toBeGreaterThan(initialTotalEliminated);

    // Territory spaces should have increased by exactly the line length.
    expect(player1After.territorySpaces).toBe(initialTerritory + requiredLength);
  });

  test('Q22_graduated_rewards_option2_min_collapse_backend_default', async () => {
    // Rules reference:
    // - Section 11.2 / 11.3: lines longer than required may use Option 2
    //   (collapse only the minimum required markers, no elimination).
    // - FAQ Q22: strategic tradeoff for preserving rings by choosing
    //   minimum collapse.
    //
    // This test exercises the default backend behaviour when no
    // PlayerInteractionManager is wired: overlong lines default to
    // Option 2 (min collapse, no elimination).
    const { engine, gameState, boardManager } = createEngine();
    const engineAny: any = engine;

    gameState.currentPlayer = 1;
    const board = gameState.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Synthetic line longer than required: requiredLength + 1 markers.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      linePositions.push({ x: i, y: 2 });
    }

    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy
      .mockImplementationOnce(() => [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
          direction: { x: 1, y: 0 }
        }
      ])
      .mockImplementation(() => []);

    // Add a stack for player 1; for Option 2 we expect no elimination, so the
    // stack should remain unchanged.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(boardManager, gameState, 1, 2, stackPos);

    const player1Before = gameState.players.find(p => p.playerNumber === 1)!;
    const initialEliminated = player1Before.eliminatedRings;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialTerritory = player1Before.territorySpaces;

    await engineAny.processLineFormations();

    const player1After = gameState.players.find(p => p.playerNumber === 1)!;
    const collapsedKeys = new Set<string>();
    for (const [key, owner] of board.collapsedSpaces) {
      if (owner === 1) collapsedKeys.add(key);
    }

    // Exactly requiredLength markers should be collapsed; the remaining
    // marker should still exist and not be collapsed.
    expect(collapsedKeys.size).toBe(requiredLength);

    const remainingPos = linePositions[requiredLength];
    const remainingKey = positionToString(remainingPos);
    expect(board.collapsedSpaces.has(remainingKey)).toBe(false);

    // No elimination should have occurred.
    expect(player1After.eliminatedRings).toBe(initialEliminated);
    expect(gameState.totalRingsEliminated).toBe(initialTotalEliminated);

    // Territory spaces should have increased by exactly requiredLength.
    expect(player1After.territorySpaces).toBe(initialTerritory + requiredLength);

    // The stack used for potential elimination should still exist.
    expect(board.stacks.get(positionToString(stackPos))).toBeDefined();
  });
});
