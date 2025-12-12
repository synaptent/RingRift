import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import type { LineOrderChoice } from '../../src/shared/types/game';

describe('WebSocketInteractionHandler', () => {
  let handler: WebSocketInteractionHandler;
  let mockIo: any;
  let mockGetTargetForPlayer: jest.Mock<string | undefined, [number]>;
  const gameId = 'test-game-123';

  // Helper to create valid PlayerChoice test data
  function createLineOrderChoice(
    id: string,
    playerNumber: number,
    overrides: Partial<LineOrderChoice> = {}
  ): LineOrderChoice {
    return {
      id,
      gameId,
      playerNumber,
      type: 'line_order' as const,
      prompt: 'Choose which line to process first',
      options: [{ lineId: 'line-1', markerPositions: [{ x: 0, y: 0 }], moveId: 'move-1' }],
      ...overrides,
    };
  }

  beforeEach(() => {
    mockIo = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
    };
    // Mock function to return socket IDs for players
    mockGetTargetForPlayer = jest.fn((playerNumber: number) => `socket-${playerNumber}`);
    // Constructor signature: (io, gameId, getTargetForPlayer, defaultTimeoutMs?)
    handler = new WebSocketInteractionHandler(mockIo, gameId, mockGetTargetForPlayer);
  });

  describe('cancelAllChoicesForPlayer', () => {
    it('should cancel all pending choices for a specific player number', async () => {
      // First, add some pending choices for different players
      const choice1 = createLineOrderChoice('choice-1', 1);
      const choice2 = createLineOrderChoice('choice-2', 1);
      const choice3 = createLineOrderChoice('choice-3', 2);

      // Start choice requests - don't await them as they're meant to wait for responses
      const promise1 = handler.requestChoice(choice1);
      const promise2 = handler.requestChoice(choice2);
      const promise3 = handler.requestChoice(choice3);

      // Verify we have 3 pending choices
      expect(handler.getPendingCount()).toBe(3);

      // Cancel all choices for player 1
      handler.cancelAllChoicesForPlayer(1);

      // Verify only player 2's choice remains
      expect(handler.getPendingCount()).toBe(1);

      // The canceled promises should reject with disconnect message
      await expect(promise1).rejects.toThrow('cancelled due to player disconnect');
      await expect(promise2).rejects.toThrow('cancelled due to player disconnect');

      // Player 2's choice is still pending - cancel it too for cleanup
      handler.cancelAllChoicesForPlayer(2);
      await expect(promise3).rejects.toThrow('cancelled due to player disconnect');

      expect(handler.getPendingCount()).toBe(0);
    });

    it('should emit player_choice_canceled for each canceled choice', () => {
      const choice = createLineOrderChoice('choice-1', 1);

      // Start a choice request
      const choicePromise = handler.requestChoice(choice);

      // Verify emitted player_choice_required event (not player_choice)
      expect(mockIo.to).toHaveBeenCalledWith('socket-1');
      expect(mockIo.emit).toHaveBeenCalledWith(
        'player_choice_required',
        expect.objectContaining({
          id: 'choice-1',
          gameId,
          playerNumber: 1,
        })
      );

      // Clear mocks for the cancel emit
      mockIo.to.mockClear();
      mockIo.emit.mockClear();

      // Cancel the choice
      handler.cancelAllChoicesForPlayer(1);

      // Verify player_choice_canceled was emitted to the game room
      expect(mockIo.to).toHaveBeenCalledWith(gameId);
      expect(mockIo.emit).toHaveBeenCalledWith('player_choice_canceled', 'choice-1');

      // Cleanup - catch the rejection
      choicePromise.catch(() => {
        /* expected */
      });
    });

    it('should not cancel choices for other players in the same game', async () => {
      const choiceP1 = createLineOrderChoice('choice-1', 1);
      const choiceP2 = createLineOrderChoice('choice-2', 2);

      const promise1 = handler.requestChoice(choiceP1);
      const promise2 = handler.requestChoice(choiceP2);

      expect(handler.getPendingCount()).toBe(2);

      // Cancel only player 1
      handler.cancelAllChoicesForPlayer(1);

      // Player 2's choice should still be pending
      expect(handler.getPendingCount()).toBe(1);

      // Clean up
      handler.cancelAllChoicesForPlayer(2);
      await expect(promise1).rejects.toThrow('cancelled due to player disconnect');
      await expect(promise2).rejects.toThrow('cancelled due to player disconnect');
    });

    it('should handle case where player has no pending choices', () => {
      // No choices for player 3
      handler.cancelAllChoicesForPlayer(3);
      expect(handler.getPendingCount()).toBe(0);
      // Should not throw
    });
  });

  describe('cancelAllChoices', () => {
    it('cancels all pending choices for the game', async () => {
      const choiceP1 = createLineOrderChoice('choice-1', 1);
      const choiceP2 = createLineOrderChoice('choice-2', 2);

      const p1 = handler.requestChoice(choiceP1);
      const p2 = handler.requestChoice(choiceP2);

      expect(handler.getPendingCount()).toBe(2);

      handler.cancelAllChoices();

      expect(handler.getPendingCount()).toBe(0);
      await expect(p1).rejects.toThrow('was cancelled by server');
      await expect(p2).rejects.toThrow('was cancelled by server');
    });
  });

  describe('getPendingCount', () => {
    it('should return 0 when no choices are pending', () => {
      expect(handler.getPendingCount()).toBe(0);
    });

    it('should return correct count of pending choices', () => {
      const choice1 = createLineOrderChoice('choice-1', 1);
      const choice2 = createLineOrderChoice('choice-2', 2);

      const p1 = handler.requestChoice(choice1);
      expect(handler.getPendingCount()).toBe(1);

      const p2 = handler.requestChoice(choice2);
      expect(handler.getPendingCount()).toBe(2);

      // Cancel all for cleanup
      handler.cancelAllChoicesForPlayer(1);
      handler.cancelAllChoicesForPlayer(2);
      p1.catch(() => {});
      p2.catch(() => {});
    });
  });

  describe('ChoiceStatus lifecycle', () => {
    it('marks choice as fulfilled on valid response', async () => {
      const choice = createLineOrderChoice('choice-fulfilled', 1);

      const promise = handler.requestChoice(choice);

      const response: import('../../src/shared/types/game').PlayerChoiceResponseFor<LineOrderChoice> =
        {
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
          choiceType: choice.type,
          selectedOption: choice.options[0],
        };

      handler.handleChoiceResponse(response);

      await expect(promise).resolves.toEqual(
        expect.objectContaining({
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
        })
      );

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      expect(status!.kind).toBe('fulfilled');
      expect(status!.gameId).toBe(gameId);
      expect(status!.choiceId).toBe(choice.id);
    });

    it('marks choice as rejected with INVALID_OPTION when response option is invalid', async () => {
      const choice = createLineOrderChoice('choice-invalid-option', 1);

      const promise = handler.requestChoice(choice);

      const invalidOption = {
        lineId: 'non-existent-line',
        markerPositions: [],
        moveId: 'invalid-move-id',
      };

      const response: import('../../src/shared/types/game').PlayerChoiceResponse = {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        selectedOption: invalidOption,
      };

      handler.handleChoiceResponse(response);

      await expect(promise).rejects.toThrow('Invalid selectedOption for choice');

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      expect(status!.kind).toBe('rejected');
      expect(status!.gameId).toBe(gameId);
      expect(status!.choiceId).toBe(choice.id);
      expect(status!.reason).toBe('INVALID_OPTION');
    });

    it('marks choice as rejected with PLAYER_MISMATCH when a different player responds', async () => {
      const choice = createLineOrderChoice('choice-player-mismatch', 1);

      const promise = handler.requestChoice(choice);

      const response: import('../../src/shared/types/game').PlayerChoiceResponseFor<LineOrderChoice> =
        {
          choiceId: choice.id,
          playerNumber: choice.playerNumber + 1,
          choiceType: choice.type,
          selectedOption: choice.options[0],
        };

      handler.handleChoiceResponse(response);

      await expect(promise).rejects.toThrow('playerNumber mismatch for choice');

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      expect(status!.kind).toBe('rejected');
      expect(status!.reason).toBe('PLAYER_MISMATCH');
    });

    it('marks choice as expired after the timeout elapses', async () => {
      jest.useFakeTimers();

      const timeoutMs = 1000;
      const choice = createLineOrderChoice('choice-expired', 1, { timeoutMs });

      const promise = handler.requestChoice(choice);

      // Advance timers past the timeout to trigger the expiration path.
      jest.advanceTimersByTime(timeoutMs + 1);

      await expect(promise).rejects.toThrow('Player choice timed out');

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      expect(status!.kind).toBe('expired');
      expect(status!.gameId).toBe(gameId);
      expect(status!.choiceId).toBe(choice.id);

      jest.useRealTimers();
    });

    it('marks choice as canceled with SERVER_CANCEL when cancelChoice is called', async () => {
      const choice = createLineOrderChoice('choice-server-cancel', 1);

      const promise = handler.requestChoice(choice);

      handler.cancelChoice(gameId, choice.id, choice.playerNumber);

      await expect(promise).rejects.toThrow('was cancelled by server');

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      expect(status!.kind).toBe('canceled');
      expect(status!.reason).toBe('SERVER_CANCEL');
    });

    it('marks choices as canceled with DISCONNECT when cancelAllChoicesForPlayer is called', async () => {
      const choice = createLineOrderChoice('choice-disconnect-cancel', 1);

      const promise = handler.requestChoice(choice);

      handler.cancelAllChoicesForPlayer(1);

      await expect(promise).rejects.toThrow('cancelled due to player disconnect');

      const status = handler.getLastChoiceStatusSnapshotForTesting(choice.id, choice.playerNumber);
      expect(status).toBeDefined();
      if (!status || status.kind !== 'canceled') {
        throw new Error('Expected canceled ChoiceStatus with DISCONNECT reason');
      }
      expect(status.reason).toBe('DISCONNECT');
    });
  });
});
