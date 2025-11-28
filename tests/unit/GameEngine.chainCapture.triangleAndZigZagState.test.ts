import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  RingStack,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import { getChainCaptureContinuationInfo } from '../../src/shared/engine/aggregates/CaptureAggregate';
import { BoardManager } from '../../src/server/game/BoardManager';

describe('GameEngine chain-capture state and enumeration (triangle & zig-zag scenarios)', () => {
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
      territorySpaces: 0,
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
      territorySpaces: 0,
    },
  ];

  function createEngine(): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: BoardManager;
  } {
    const engine = new GameEngine('triangle-zigzag', boardType, basePlayers, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: BoardManager = engineAny.boardManager as BoardManager;
    return { engine, gameState, boardManager };
  }

  function setStack(
    boardManager: BoardManager,
    gameState: GameState,
    pos: Position,
    player: number,
    height: number
  ): void {
    const rings = Array(height).fill(player);
    const stack: RingStack = {
      position: pos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: player,
    };
    boardManager.setStack(pos, stack, gameState.board);
  }

  test('triangle loop: chainCaptureState and enumerator after first segment', async () => {
    const { engine, gameState, boardManager } = createEngine();

    // Initial triangle setup (mirrors ComplexChainCaptures + rulesMatrix):
    // P1 at (3,3) H1
    // P2 at (3,4) H1
    // P2 at (4,4) H1
    // P2 at (4,3) H1
    const startPos: Position = { x: 3, y: 3 };
    const target1: Position = { x: 3, y: 4 };
    const target2: Position = { x: 4, y: 4 };
    const target3: Position = { x: 4, y: 3 };

    gameState.board.stacks.clear();
    setStack(boardManager, gameState, startPos, 1, 1);
    setStack(boardManager, gameState, target1, 2, 1);
    setStack(boardManager, gameState, target2, 2, 1);
    setStack(boardManager, gameState, target3, 2, 1);

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'capture';

    // First segment: (3,3) -> (3,4) -> (3,5)
    const firstMove: Move = {
      id: 'triangle-first',
      type: 'overtaking_capture',
      player: 1,
      from: startPos,
      captureTarget: target1,
      to: { x: 3, y: 5 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = await engine.makeMove(firstMove as any);
    expect(result.success).toBe(true);

    const engineAny: any = engine;
    const stateAfter: GameState = engineAny.gameState as GameState;
    const chainState = engineAny.chainCaptureState as
      | {
          playerNumber: number;
          startPosition: Position;
          currentPosition: Position;
          availableMoves: Move[];
        }
      | undefined;

    // Instrument: board and chain state after first capture
    const boardAfter = stateAfter.board;
    const stackKeys = Array.from(boardAfter.stacks.keys()).sort();

    expect(chainState).toBeDefined();
    expect(chainState!.playerNumber).toBe(1);
    expect(positionToString(chainState!.currentPosition)).toBe(positionToString({ x: 3, y: 5 }));

    // Board should contain the expected stacks: attacker at 3,5 and the two
    // remaining triangle vertices at 4,4 and 4,3.
    expect(stackKeys).toEqual(expect.arrayContaining(['3,5', '4,4', '4,3']));
    const stackAt35 = boardAfter.stacks.get('3,5')!;
    const stackAt44 = boardAfter.stacks.get('4,4')!;
    const stackAt43 = boardAfter.stacks.get('4,3')!;

    expect(stackAt35.controllingPlayer).toBe(1);
    expect(stackAt35.stackHeight).toBe(2);

    expect(stackAt44.controllingPlayer).toBe(2);
    expect(stackAt44.stackHeight).toBe(1);

    expect(stackAt43.controllingPlayer).toBe(2);
    expect(stackAt43.stackHeight).toBe(1);

    // Directly query the shared enumerator from the chain position
    const { availableContinuations: followUps } = getChainCaptureContinuationInfo(
      stateAfter,
      1,
      chainState!.currentPosition
    );

    expect(followUps.length).toBeGreaterThan(0);

    // There must be at least the expected segment (3,5) -> (4,4) -> (5,3).
    const hasExpected = followUps.some(
      (m) =>
        m.player === 1 &&
        m.from &&
        m.captureTarget &&
        m.to &&
        m.from.x === 3 &&
        m.from.y === 5 &&
        m.captureTarget.x === 4 &&
        m.captureTarget.y === 4 &&
        m.to.x === 5 &&
        m.to.y === 3
    );
    expect(hasExpected).toBe(true);
  });

  test('zig-zag chain: chainCaptureState and enumerator after first segment', async () => {
    const { engine, gameState, boardManager } = createEngine();

    // Zig-zag setup (mirrors ComplexChainCaptures.Multi_Directional_ZigZag_Chain):
    // P1 at (0,0) H1
    // P2 at (1,1) H1
    // P2 at (3,2) H1
    // P2 at (4,3) H1
    const startPos: Position = { x: 0, y: 0 };
    const target1: Position = { x: 1, y: 1 };
    const target2: Position = { x: 3, y: 2 };
    const target3: Position = { x: 4, y: 3 };

    gameState.board.stacks.clear();
    setStack(boardManager, gameState, startPos, 1, 1);
    setStack(boardManager, gameState, target1, 2, 1);
    setStack(boardManager, gameState, target2, 2, 1);
    setStack(boardManager, gameState, target3, 2, 1);

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'capture';

    // First segment: (0,0) -> (1,1) -> (2,2)
    const firstMove: Move = {
      id: 'zigzag-first',
      type: 'overtaking_capture',
      player: 1,
      from: startPos,
      captureTarget: target1,
      to: { x: 2, y: 2 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = await engine.makeMove(firstMove as any);
    expect(result.success).toBe(true);

    const engineAny: any = engine;
    const stateAfter: GameState = engineAny.gameState as GameState;
    const chainState = engineAny.chainCaptureState as
      | {
          playerNumber: number;
          startPosition: Position;
          currentPosition: Position;
          availableMoves: Move[];
        }
      | undefined;

    const boardAfter = stateAfter.board;
    const stackKeys = Array.from(boardAfter.stacks.keys()).sort();

    expect(chainState).toBeDefined();
    expect(chainState!.playerNumber).toBe(1);
    expect(positionToString(chainState!.currentPosition)).toBe(positionToString({ x: 2, y: 2 }));

    // Board should contain attacker at (2,2), P2 stacks at (3,2) and (4,3).
    expect(stackKeys).toEqual(expect.arrayContaining(['2,2', '3,2', '4,3']));
    const stackAt22 = boardAfter.stacks.get('2,2')!;
    const stackAt32 = boardAfter.stacks.get('3,2')!;
    const stackAt43 = boardAfter.stacks.get('4,3')!;

    expect(stackAt22.controllingPlayer).toBe(1);
    expect(stackAt22.stackHeight).toBe(2);

    expect(stackAt32.controllingPlayer).toBe(2);
    expect(stackAt32.stackHeight).toBe(1);

    expect(stackAt43.controllingPlayer).toBe(2);
    expect(stackAt43.stackHeight).toBe(1);

    // Enumerate follow-up segments from (2,2).
    const { availableContinuations: followUps } = getChainCaptureContinuationInfo(
      stateAfter,
      1,
      chainState!.currentPosition
    );

    expect(followUps.length).toBeGreaterThan(0);

    // There must be at least the expected second segment: (2,2) -> (3,2) -> (4,2).
    const hasExpected = followUps.some(
      (m) =>
        m.player === 1 &&
        m.from &&
        m.captureTarget &&
        m.to &&
        m.from.x === 2 &&
        m.from.y === 2 &&
        m.captureTarget.x === 3 &&
        m.captureTarget.y === 2 &&
        m.to.x === 4 &&
        m.to.y === 2
    );
    expect(hasExpected).toBe(true);
  });
});
