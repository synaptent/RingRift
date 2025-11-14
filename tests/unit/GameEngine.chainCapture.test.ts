import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, BoardType, TimeControl, RingStack } from '../../src/shared/types/game';

/**
 * Basic behavioural tests for the chain-capture enforcement layer in GameEngine.
 *
 * These tests do NOT attempt to validate full board legality or RuleEngine
 * integration. Instead, they focus narrowly on the new TsChainCaptureState
 * gate that prevents players from:
 *   - moving a different player's piece while a chain is in progress
 *   - playing any non-overtaking or wrong-origin move during an active chain
 */

describe('GameEngine chain capture enforcement (TsChainCaptureState)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

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

  function createEngine(): GameEngine {
    // GameEngine constructor reassigns playerNumber and timeRemaining, so
    // passing these base players is sufficient for our purposes here.
    return new GameEngine('test-game-chain', boardType, basePlayers, timeControl, false);
  }

  test('rejects moves from a different player while a chain capture is in progress', async () => {
    const engine = createEngine();

    const chainStart: Position = { x: 3, y: 3 };
    const chainCurrent: Position = { x: 5, y: 5 };

    // Force an internal chain state as if player 1 had started a capture.
    (engine as any).chainCaptureState = {
      playerNumber: 1,
      startPosition: chainStart,
      currentPosition: chainCurrent,
      segments: [],
      availableMoves: [],
      visitedPositions: new Set<string>(['3,3'])
    };

    const result = await engine.makeMove({
      // Wrong player attempts to move while chain is active
      player: 2,
      type: 'move_ring',
      from: chainCurrent,
      to: { x: 6, y: 6 }
    } as any);

    expect(result.success).toBe(false);
    expect(result.error).toBe('Chain capture in progress: only the capturing player may move');
  });

  test('rejects non-overtaking or wrong-origin moves from the capturing player during an active chain', async () => {
    const engine = createEngine();

    const chainStart: Position = { x: 3, y: 3 };
    const chainCurrent: Position = { x: 5, y: 5 };

    (engine as any).chainCaptureState = {
      playerNumber: 1,
      startPosition: chainStart,
      currentPosition: chainCurrent,
      segments: [],
      availableMoves: [],
      visitedPositions: new Set<string>(['3,3', '5,5'])
    };

    // Case 1: correct player but wrong move type
    const wrongType = await engine.makeMove({
      player: 1,
      type: 'move_ring',
      from: chainCurrent,
      to: { x: 6, y: 6 }
    } as any);

    expect(wrongType.success).toBe(false);
    expect(wrongType.error).toBe('Chain capture in progress: must continue capturing with the same stack');

    // Case 2: correct player and type, but from a different origin than currentPosition
    const wrongOrigin = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: { x: 4, y: 4 },
      captureTarget: { x: 6, y: 6 },
      to: { x: 7, y: 7 }
    } as any);

    expect(wrongOrigin.success).toBe(false);
    expect(wrongOrigin.error).toBe('Chain capture in progress: must continue capturing with the same stack');
  });

  test('performs a full two-step chain capture end-to-end (ported from Rust)', async () => {
    // This scenario mirrors the Rust test_chain_capture setup on an 8x8 board:
    // Red stack at (2,2) height 2, Blue at (2,3) height 1, Green at (2,5) height 1.
    // Red jumps over Blue to land at (2,4), then is forced to continue and
    // jumps over Green to land at (2,7), capturing both Blue and Green.
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const players: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'blue',
        username: 'Blue',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'green',
        username: 'Green',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];

    const engine = new GameEngine('chain-e2e', 'square8', players, timeControl, false);
    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as any;

    // Set current phase/player so that capture validation passes
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    // Helper to build a stack for a given player and height
    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber
      };
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 2, y: 5 };

    makeStack(1, 2, redPos);   // Red height 2 at (2,2)
    makeStack(2, 1, bluePos);  // Blue height 1 at (2,3)
    makeStack(3, 1, greenPos); // Green height 1 at (2,5)

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 2, y: 4 }
    } as any);

    expect(result.success).toBe(true);

    // After the engine-driven chain, the original red stack and both targets
    // should be gone, and the capturing stack should be at (2,7) with height 4.
    const board = gameState.board;
    const stackAtRed = board.stacks.get('2,2');
    const stackAtBlue = board.stacks.get('2,3');
    const stackAtGreen = board.stacks.get('2,5');
    const stackAtFinal = board.stacks.get('2,7');

    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtGreen).toBeUndefined();
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.stackHeight).toBe(4);
    expect(stackAtFinal!.controllingPlayer).toBe(1);

    // Internal chain state should be cleared once no further captures exist.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });
});
