import { EventEmitter } from 'events';
import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import {
  PlayerChoice,
  PlayerChoiceResponse,
  Position
} from '../../src/shared/types/game';

// Minimal Socket.IO Server stub for testing
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

describe('WebSocketInteractionHandler', () => {
  const gameId = 'game-1';
  const playerNumber = 1;

  const samplePositions: Position[] = [
    { x: 0, y: 0 },
    { x: 1, y: 1 }
  ];

  const baseChoice: PlayerChoice = {
    id: 'choice-1',
    gameId,
    playerNumber,
    type: 'line_order',
    prompt: 'Choose a line',
    options: [
      { lineId: '0', markerPositions: samplePositions },
      { lineId: '1', markerPositions: samplePositions }
    ]
  } as const;

  it('emits player_choice_required and resolves on valid response', async () => {
    const io = new FakeSocketIOServer();
    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');

    const handler = new WebSocketInteractionHandler(
      io as any,
      gameId,
      getTargetForPlayer,
      30_000
    );

    const promise = handler.requestChoice(baseChoice);

    // Verify that the choice was emitted to the correct target
    expect(getTargetForPlayer).toHaveBeenCalledWith(playerNumber);
    expect(io.toCalls).toHaveLength(1);
    expect(io.toCalls[0]).toMatchObject({
      target: 'socket-1',
      event: 'player_choice_required',
      payload: baseChoice
    });

    const response: PlayerChoiceResponse<(typeof baseChoice.options)[number]> = {
      choiceId: baseChoice.id,
      playerNumber,
      selectedOption: baseChoice.options[0]
    };

    handler.handleChoiceResponse(response);

    await expect(promise).resolves.toEqual(response);
  });

  it('rejects when selectedOption is not one of the original options', async () => {
    const io = new FakeSocketIOServer();
    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');

    const handler = new WebSocketInteractionHandler(
      io as any,
      gameId,
      getTargetForPlayer,
      30_000
    );

    const promise = handler.requestChoice(baseChoice);

    const invalidResponse: PlayerChoiceResponse<any> = {
      choiceId: baseChoice.id,
      playerNumber,
      selectedOption: {
        lineId: '999',
        markerPositions: []
      }
    };

    handler.handleChoiceResponse(invalidResponse);

    await expect(promise).rejects.toThrow(/Invalid selectedOption/);
  });

  it('rejects when response playerNumber does not match choice.playerNumber', async () => {
    const io = new FakeSocketIOServer();
    const getTargetForPlayer = jest.fn().mockReturnValue('socket-1');

    const handler = new WebSocketInteractionHandler(
      io as any,
      gameId,
      getTargetForPlayer,
      30_000
    );

    const promise = handler.requestChoice(baseChoice);

    const wrongPlayerResponse: PlayerChoiceResponse<(typeof baseChoice.options)[number]> = {
      choiceId: baseChoice.id,
      playerNumber: playerNumber + 1,
      selectedOption: baseChoice.options[0]
    };

    handler.handleChoiceResponse(wrongPlayerResponse);

    await expect(promise).rejects.toThrow(/playerNumber mismatch/);
  });
});
