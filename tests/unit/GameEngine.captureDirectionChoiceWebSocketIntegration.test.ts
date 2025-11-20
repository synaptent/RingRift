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

describe.skip('GameEngine + WebSocketInteractionHandler capture direction choice integration (legacy capture_direction flow)', () => {
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

  it('runs a full orthogonal chain via WebSocket capture_direction choices (backend + transport)', async () => {
    // End-to-end version of the orthogonal multi-branch chain scenario used in
    // GameEngine.chainCaptureChoiceIntegration and the sandbox tests. Here we
    // drive the entire chain through:
    //   GameEngine.makeMove → PlayerInteractionManager → WebSocketInteractionHandler
    // and simulate client responses to player_choice_required events.

    const io = new FakeSocketIOServer();
    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');
    const handler = new WebSocketInteractionHandler(
      io as any,
      'capture-direction-game-full-chain',
      getTargetForPlayer,
      30_000
    );
    const interactionManager = new PlayerInteractionManager(handler);

    const timeControlLocal: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const playersForChain: Player[] = [
      {
        id: 'red',
        username: 'Red',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControlLocal.initialTime * 1000,
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
        timeRemaining: timeControlLocal.initialTime * 1000,
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
        timeRemaining: timeControlLocal.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'yellow',
        username: 'Yellow',
        type: 'human',
        playerNumber: 4,
        isReady: true,
        timeRemaining: timeControlLocal.initialTime * 1000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];

    const engine = new GameEngine(
      'capture-direction-game-full-chain',
      boardType,
      playersForChain,
      timeControlLocal,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const gameState = engineAny.gameState as any;
    const boardManager = engineAny.boardManager as any;

    // Ensure capture phase & correct player so RuleEngine allows capture.
    gameState.currentPhase = 'capture';
    gameState.currentPlayer = 1;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber
      } as RingStack;
      boardManager.setStack(position, stack, gameState.board);
    };

    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 3, y: 4 };
    const greenPos: Position = { x: 4, y: 5 };
    const yellowPos: Position = { x: 2, y: 5 };

    makeStack(1, 2, redPos); // Red attacker
    makeStack(2, 1, bluePos);
    makeStack(3, 1, greenPos);
    makeStack(4, 1, yellowPos);

    const choices: CaptureDirectionChoice[] = [];

    // Respond to each player_choice_required event by selecting the
    // lexicographically earliest landing position, mirroring other tests.
    io.on('player_choice_required', (payload: CaptureDirectionChoice) => {
      choices.push(payload);

      const options = payload.options || [];
      expect(options.length).toBeGreaterThan(0);

      const selected = options.reduce((prev, cur) =>
        cur.landingPosition.x < prev.landingPosition.x ||
        (cur.landingPosition.x === prev.landingPosition.x &&
          cur.landingPosition.y < prev.landingPosition.y)
          ? cur
          : prev
      );

      const response: PlayerChoiceResponseFor<CaptureDirectionChoice> = {
        choiceId: payload.id,
        playerNumber: payload.playerNumber,
        choiceType: 'capture_direction',
        selectedOption: selected
      };

      handler.handleChoiceResponse(response as unknown as PlayerChoiceResponse<unknown>);
    });

    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: redPos,
      captureTarget: bluePos,
      to: { x: 3, y: 5 }
    } as any);

    expect(result.success).toBe(true);

    // There should have been at least one capture_direction choice issued
    // over WebSockets during the engine-driven chain.
    expect(choices.length).toBeGreaterThan(0);

    const allPairs = choices.flatMap(ch =>
      (ch.options || []).map(o =>
        `${o.targetPosition.x},${o.targetPosition.y}->${o.landingPosition.x},${o.landingPosition.y}`
      )
    );

    expect(allPairs).toEqual(
      expect.arrayContaining([
        '4,5->6,5',
        '4,5->7,5',
        '2,5->0,5'
      ])
    );

    const board = gameState.board;
    const stackAtStart = board.stacks.get('3,3');
    const stackAtBlue = board.stacks.get('3,4');
    const stackAtIntermediate = board.stacks.get('3,5');

    expect(stackAtStart).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtIntermediate).toBeUndefined();

    // Exactly one Red-controlled stack should remain on the board as the
    // final capturing stack for this turn.
    const redStacks = Array.from(board.stacks.values()).filter(
      (s: any) => s.controllingPlayer === 1
    );
    expect(redStacks.length).toBe(1);
    expect(redStacks[0].stackHeight).toBeGreaterThanOrEqual(3);

    // Chain state must be cleared after the engine-driven chain completes.
    expect(engineAny.chainCaptureState).toBeUndefined();
  });
});
