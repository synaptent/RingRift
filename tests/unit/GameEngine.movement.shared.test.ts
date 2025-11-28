import type {
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  BoardType,
} from '../../src/shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  applySimpleMovement as applySimpleMovementAggregate,
  hashGameState,
} from '../../src/shared/engine';
import { GameEngine } from '../../src/server/game/GameEngine';

describe('GameEngine movement integration with shared MovementAggregate', () => {
  function createEngine(boardType: BoardType = 'square8' as BoardType): GameEngine {
    const players: Player[] = [
      { id: 'P1', username: 'P1', type: 'human' } as Player,
      { id: 'P2', username: 'P2', type: 'human' } as Player,
    ];

    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    const engine = new GameEngine(
      'movement-aggregate-test',
      boardType,
      players,
      timeControl,
      false
    );

    // For this test we want to exercise the legacy GameEngine.applyMove path
    // directly rather than delegating to the shared orchestrator adapter.
    (engine as any).disableOrchestratorAdapter?.();

    return engine;
  }

  it('applies non-capture movement via MovementAggregate with identical board state', () => {
    const engine = createEngine();
    const engineAny = engine as any;

    // Seed the internal GameState with a simple two-ring stack for player 1
    // on an otherwise empty square board.
    const internalState: GameState = engineAny.gameState as GameState;
    const board = internalState.board;

    const origin: Position = { x: 3, y: 3 };
    const dest: Position = { x: 5, y: 3 };

    const originKey = positionToString(origin);
    board.stacks.set(originKey, {
      position: origin,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    });

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    // Sanity: destination is on board and empty.
    const config = BOARD_CONFIGS[internalState.boardType];
    expect(dest.x).toBeGreaterThanOrEqual(0);
    expect(dest.x).toBeLessThan(config.size);
    expect(dest.y).toBeGreaterThanOrEqual(0);
    expect(dest.y).toBeLessThan(config.size);
    expect(board.stacks.has(positionToString(dest))).toBe(false);

    // Take a defensive snapshot for the shared aggregate using the public
    // getter, which deep-clones board/maps.
    const snapshotBefore: GameState = engine.getGameState();

    const coreOutcome = applySimpleMovementAggregate(snapshotBefore, {
      from: origin,
      to: dest,
      player: 1,
    });

    // Apply the same logical move via the GameEngine.applyMove helper, which
    // is wired to MovementAggregate.applySimpleMovement under the hood.
    const move: Move = {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: origin,
      to: dest,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    engineAny.applyMove(move);

    const engineStateAfter: GameState = engine.getGameState();

    const coreHash = hashGameState(coreOutcome.nextState);
    const engineHash = hashGameState(engineStateAfter);

    expect(engineHash).toEqual(coreHash);
  });
});
