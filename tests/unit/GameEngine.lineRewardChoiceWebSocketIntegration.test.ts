import { EventEmitter } from 'events';
import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import {
  BoardType,
  GameState,
  LineRewardChoice,
  Move,
  Player,
  PlayerChoiceResponse,
  Position,
  RingStack,
  TimeControl
} from '../../src/shared/types/game';

// Minimal Socket.IO Server stub for testing end-to-end choice plumbing
class FakeSocketIOServer extends EventEmitter {
  public toCalls: Array<{ target: string; event: string; payload: any }> = [];

  to(target: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ target, event, payload });
        this.emit(event, payload);
      }
    };
  }
}

function makeStack(playerNumber: number, height: number, position: Position): RingStack {
  const rings = Array(height).fill(playerNumber);
  return {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer: playerNumber
  };
}

const boardType: BoardType = 'square8';
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
  }
];

/**
 * This test validates an end-to-end LineRewardChoice flow:
 *   GameEngine → PlayerInteractionManager → WebSocketInteractionHandler →
 *   fake Socket.IO client → WebSocketInteractionHandler.handleChoiceResponse →
 *   GameEngine.processLineFormations.
 *
 * It focuses on the wiring rather than the full geometry of line detection,
 * so the board is prepared directly with an overlong marker line for the
 * current player. The test asserts that:
 *   1. A player_choice_required event is emitted with a LineRewardChoice.
 *   2. Responding with Option 1 collapses all markers and eliminates rings
 *      from a player-controlled stack.
 */

describe('GameEngine + WebSocketInteractionHandler line reward choice integration', () => {
  it('emits LineRewardChoice over WebSocket and applies the selected reward option', async () => {
    const io = new FakeSocketIOServer();

    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');
    const handler = new WebSocketInteractionHandler(
      io as any,
      'line-reward-game',
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'line-reward-game',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    // Prepare game state: current player is Red. We do not rely on the
    // internal implementation of BoardManager.findAllLines here; instead we
    // stub it to return an overlong line for player 1 so that we can exercise
    // the LineRewardChoice plumbing in isolation from geometry details.
    gameState.currentPlayer = 1;

    const markerPositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 }
    ];

    // First call returns an overlong line for player 1; subsequent calls
    // return an empty array so processLineFormations terminates after a
    // single iteration. This keeps the test focused on the choice plumbing
    // rather than on BoardManager geometry.
    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy.mockImplementationOnce(() => [
      {
        player: 1,
        positions: markerPositions
      }
    ]);
    findAllLinesSpy.mockImplementation(() => []);

    const stackPos: Position = { x: 0, y: 1 };
    boardManager.setStack(stackPos, makeStack(1, 3, stackPos), gameState.board);

    const initialEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;

    // Invoke the line processing pipeline directly. This mirrors the internal
    // call from processAutomaticConsequences, but keeps the test focused.
    const processPromise: Promise<void> = engineAny.processLineFormations();

    // A LineRewardChoice should have been emitted via WebSocket.
    expect(getTargetForPlayer).toHaveBeenCalledWith(1);
    expect(io.toCalls).toHaveLength(1);

    const call = io.toCalls[0];
    expect(call.event).toBe('player_choice_required');

    const choice = call.payload as LineRewardChoice;
    expect(choice.type).toBe('line_reward_option');
    expect(choice.playerNumber).toBe(1);
    expect(choice.options).toEqual([
      'option_1_collapse_all_and_eliminate',
      'option_2_min_collapse_no_elimination'
    ]);

    // Simulate the client choosing Option 1: collapse all markers and
    // eliminate a ring/cap from one of Red's stacks.
    const response: PlayerChoiceResponse<(typeof choice.options)[number]> = {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      selectedOption: 'option_1_collapse_all_and_eliminate'
    };

    handler.handleChoiceResponse(response as any);

    // Wait for GameEngine.processLineFormations to complete.
    await processPromise;

    // All marker positions should now be collapsed spaces for player 1.
    for (const pos of markerPositions) {
      expect(boardManager.isCollapsedSpace(pos, gameState.board)).toBe(true);
    }

    // Red's eliminated ring count should have increased.
    const finalEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;
    expect(finalEliminated).toBeGreaterThan(initialEliminated);
  });

  it('emits RingEliminationChoice and eliminates rings from the chosen stack', async () => {
    const io = new FakeSocketIOServer();

    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');
    const handler = new WebSocketInteractionHandler(
      io as any,
      'ring-elimination-game',
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'ring-elimination-game',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    // Two stacks for Red so the elimination choice has multiple options.
    const stackA: Position = { x: 1, y: 1 };
    const stackB: Position = { x: 2, y: 2 };

    boardManager.setStack(stackA, makeStack(1, 2, stackA), gameState.board);
    boardManager.setStack(stackB, makeStack(1, 3, stackB), gameState.board);

    const initialEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;

    // Call the internal helper that issues a RingEliminationChoice. This is
    // intentionally a white-box test focused on the PlayerChoice plumbing.
    const eliminatePromise: Promise<void> = engineAny.eliminatePlayerRingOrCapWithChoice(1);

    // A player_choice_required should have been emitted.
    expect(getTargetForPlayer).toHaveBeenCalledWith(1);
    expect(io.toCalls).toHaveLength(1);

    const call = io.toCalls[0];
    expect(call.event).toBe('player_choice_required');

    const choice = call.payload as any;
    expect(choice.type).toBe('ring_elimination');
    expect(choice.playerNumber).toBe(1);
    expect(Array.isArray(choice.options)).toBe(true);
    expect(choice.options.length).toBe(2);

    // Choose the option corresponding to stackB (position 2,2).
    const selectedOption = choice.options.find((opt: any) =>
      opt.stackPosition.x === stackB.x && opt.stackPosition.y === stackB.y
    );
    expect(selectedOption).toBeDefined();

    const response: PlayerChoiceResponse<any> = {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      selectedOption
    };

    handler.handleChoiceResponse(response);

    // Wait for elimination to complete.
    await eliminatePromise;

    // Rings should have been removed from stackB; total eliminated count increased.
    const finalEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;
    expect(finalEliminated).toBeGreaterThan(initialEliminated);
  });
});
