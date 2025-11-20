import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  Move,
  Player,
  Position,
  TimeControl,
} from '../../src/shared/types/game';

/**
 * Tests for the unified chain_capture + continue_capture_segment model.
 *
 * These tests validate that when a chain capture is in progress,
 * GameEngine.getValidMoves exposes continuation segments as
 * `continue_capture_segment` moves for the capturing player, using the
 * internal chainCaptureState and getCaptureOptionsFromPosition helper.
 */

describe('GameEngine chain_capture getValidMoves integration', () => {
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

  function createEngine(): GameEngine {
    return new GameEngine('test-game-chain-capture', boardType, basePlayers, timeControl, false);
  }

  it('exposes continue_capture_segment moves for an active chain from currentPosition', () => {
    const engine = createEngine();
    const engineAny: any = engine;

    // Seed chain state and phase so getValidMoves treats this as an
    // interactive chain_capture decision point for player 1.
    const currentPos: Position = { x: 5, y: 5 };
    engineAny.chainCaptureState = {
      playerNumber: 1,
      startPosition: { x: 3, y: 3 },
      currentPosition: currentPos,
      segments: [],
      availableMoves: [],
      visitedPositions: new Set<string>(['3,3', '5,5']),
    };

    const gameState = engineAny.gameState as any;
    gameState.currentPhase = 'chain_capture';
    gameState.currentPlayer = 1;

    const baseMoveA: Move = {
      id: 'capture-a',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: { x: 6, y: 6 },
      to: { x: 7, y: 7 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const baseMoveB: Move = {
      id: 'capture-b',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: { x: 4, y: 6 },
      to: { x: 3, y: 7 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    // Stub out the internal capture enumeration helper so we can focus
    // this test purely on how GameEngine.getValidMoves re-labels
    // overtaking_capture candidates as continue_capture_segment moves.
    const spy = jest
      .spyOn(engineAny, 'getCaptureOptionsFromPosition')
      .mockReturnValue([baseMoveA, baseMoveB]);

    const moves = engine.getValidMoves(1);

    expect(spy).toHaveBeenCalledWith(currentPos, 1);
    expect(moves).toHaveLength(2);

    // All returned moves should be relabelled as continue_capture_segment
    // for the capturing player, preserving geometry.
    for (const m of moves) {
      expect(m.type).toBe('continue_capture_segment');
      expect(m.player).toBe(1);
      expect(m.from).toEqual(currentPos);
    }

    const segmentTargets = moves.map((m) => `${m.captureTarget!.x},${m.captureTarget!.y}`);
    const segmentLandings = moves.map((m) => `${m.to!.x},${m.to!.y}`);

    expect(segmentTargets).toEqual(
      expect.arrayContaining([
        '6,6',
        '4,6',
      ])
    );
    expect(segmentLandings).toEqual(
      expect.arrayContaining([
        '7,7',
        '3,7',
      ])
    );
  });

  it('returns no moves and clears chain state when no further capture options exist', () => {
    const engine = createEngine();
    const engineAny: any = engine;

    const currentPos: Position = { x: 2, y: 2 };
    engineAny.chainCaptureState = {
      playerNumber: 1,
      startPosition: { x: 2, y: 0 },
      currentPosition: currentPos,
      segments: [],
      availableMoves: [],
      visitedPositions: new Set<string>(['2,0', '2,2']),
    };

    const gameState = engineAny.gameState as any;
    gameState.currentPhase = 'chain_capture';
    gameState.currentPlayer = 1;

    const spy = jest
      .spyOn(engineAny, 'getCaptureOptionsFromPosition')
      .mockReturnValue([]);

    const moves = engine.getValidMoves(1);

    expect(spy).toHaveBeenCalledWith(currentPos, 1);
    expect(moves).toHaveLength(0);
    // When there are no available continuations, the engine clears
    // internal chain state so callers do not see a stuck interactive
    // phase with no legal actions.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });
});
