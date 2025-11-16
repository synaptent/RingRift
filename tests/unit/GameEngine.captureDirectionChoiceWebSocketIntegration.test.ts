import { EventEmitter } from 'events';
import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import {
  BoardType,
  CaptureDirectionChoice,
  Move,
  Player,
  PlayerChoice,
  PlayerChoiceResponse,
  PlayerChoiceResponseFor,
  Position,
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

const boardType: BoardType = 'square8';
const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

const players: Player[] = [
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

/**
 * This test validates a CaptureDirectionChoice flow over WebSockets:
 *
 *   GameEngine.chooseCaptureDirectionFromState → PlayerInteractionManager →
 *   WebSocketInteractionHandler → FakeSocketIOServer ("client") →
 *   WebSocketInteractionHandler.handleChoiceResponse → GameEngine helper.
 *
 * It is the capture-direction analogue to the existing
 * GameEngine.lineRewardChoiceWebSocketIntegration test and complements
 * tests/unit/GameEngine.captureDirectionChoice.test.ts, which exercises
 * the same helper with a direct mock handler rather than a WebSocket
 * transport.
 */

describe('GameEngine + WebSocketInteractionHandler capture direction choice integration', () => {
  it('emits CaptureDirectionChoice and applies the selected follow-up capture segment', async () => {
    const io = new FakeSocketIOServer();

    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');
    const handler = new WebSocketInteractionHandler(
      io as any,
      'capture-direction-game',
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'capture-direction-game',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;

    // Simulate a chain in progress for player 1 from position (5,5), with two
    // possible follow-up capture segments along different rays.
    const currentPos: Position = { x: 5, y: 5 };
    const targetA: Position = { x: 6, y: 6 };
    const landingA: Position = { x: 7, y: 7 };
    const targetB: Position = { x: 4, y: 6 };
    const landingB: Position = { x: 3, y: 7 };

    const optionA: Move = {
      id: 'opt-a',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: targetA,
      to: landingA,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const optionB: Move = {
      id: 'opt-b',
      type: 'overtaking_capture',
      player: 1,
      from: currentPos,
      captureTarget: targetB,
      to: landingB,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    // Seed the internal chainCaptureState so that chooseCaptureDirectionFromState
    // will issue a CaptureDirectionChoice for player 1 with these two options.
    (engineAny as any).chainCaptureState = {
      playerNumber: 1,
      startPosition: { x: 3, y: 3 },
      currentPosition: currentPos,
      segments: [],
      availableMoves: [optionA, optionB],
      visitedPositions: new Set<string>(['3,3', '5,5'])
    };

    // Kick off the helper that goes through PlayerInteractionManager. We
    // intentionally use the internal helper (already covered in
    // GameEngine.captureDirectionChoice.test.ts) to keep this focused on the
    // WebSocket transport wiring.
    const chosenPromise: Promise<Move | undefined> = (engineAny as any).chooseCaptureDirectionFromState();

    // WebSocketInteractionHandler should have looked up the target socket for
    // player 1 and emitted a player_choice_required event.
    expect(getTargetForPlayer).toHaveBeenCalledWith(1);
    expect(io.toCalls).toHaveLength(1);

    const call = io.toCalls[0];
    expect(call.event).toBe('player_choice_required');

    const choice = call.payload as CaptureDirectionChoice;
    expect(choice.type).toBe('capture_direction');
    expect(choice.playerNumber).toBe(1);
    expect(choice.options.length).toBe(2);

    // Simulate the client choosing the option whose landingPosition is the
    // "earlier" one in lexicographic (x,y) order; this mirrors the
    // deterministic selection used in the pure handler tests.
    const selected = choice.options.reduce((prev, cur) =>
      cur.landingPosition.x < prev.landingPosition.x ||
      (cur.landingPosition.x === prev.landingPosition.x &&
        cur.landingPosition.y < prev.landingPosition.y)
        ? cur
        : prev
    );

    const response: PlayerChoiceResponseFor<CaptureDirectionChoice> = {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      choiceType: 'capture_direction',
      selectedOption: selected
    };

    handler.handleChoiceResponse(response as unknown as PlayerChoiceResponse<unknown>);

    const chosen = await chosenPromise;

    // The helper should return the Move corresponding to the selected option.
    expect(chosen).toBeDefined();
    expect(chosen!.type).toBe('overtaking_capture');
    expect(chosen!.player).toBe(1);
    expect(chosen!.captureTarget).toEqual(selected.targetPosition);
    expect(chosen!.to).toEqual(selected.landingPosition);

    // Internal chain state remains owned by the engine; this test only
    // asserts the correctness of the choice plumbing.
    expect(gameState.currentPlayer).toBe(1);
  });
});
