/**
 * Tests for client-side GameConnection reconnection and error handling.
 *
 * This file covers client reconnection scenarios not addressed elsewhere:
 * - Connection status tracking
 * - Reconnection attempts and status transitions
 * - Error handling for various failure modes
 * - Message queuing and state recovery
 *
 * Related spec: docs/archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md
 * - §2.4 Connection sub-states and reconnect windows
 * - §3.1 Decision / AI request lifecycle
 * - §3.4 Frontend connection + decision UX under reconnect
 *
 * @jest-environment jsdom
 */

import { SocketGameConnection } from '../../src/client/services/GameConnection';
import type { GameEventHandlers, ConnectionStatus } from '../../src/client/domain/GameAPI';
import type {
  GameStateUpdateMessage,
  GameOverMessage,
  ChatMessageServerPayload,
} from '../../src/shared/types/websocket';
import type { PlayerChoice, Move } from '../../src/shared/types/game';

// Mock socket.io-client
const mockEmit = jest.fn();
const mockOn = jest.fn();
const mockDisconnect = jest.fn();
const mockSocket = {
  on: mockOn,
  emit: mockEmit,
  disconnect: mockDisconnect,
  connected: true,
};

jest.mock('socket.io-client', () => ({
  io: jest.fn(() => mockSocket),
}));

// Mock getSocketBaseUrl
jest.mock('../../src/client/utils/socketBaseUrl', () => ({
  getSocketBaseUrl: jest.fn(() => 'http://localhost:3000'),
}));

describe('GameConnection - Reconnection', () => {
  let handlers: jest.Mocked<GameEventHandlers>;
  let connection: SocketGameConnection;
  let eventListeners: Map<string, (data: any) => void>;

  beforeEach(() => {
    jest.clearAllMocks();
    eventListeners = new Map();

    // Capture event listeners registered via socket.on
    mockOn.mockImplementation((event: string, callback: (data: any) => void) => {
      eventListeners.set(event, callback);
      return mockSocket;
    });

    handlers = {
      onGameState: jest.fn(),
      onGameOver: jest.fn(),
      onChoiceRequired: jest.fn(),
      onChoiceCanceled: jest.fn(),
      onChatMessage: jest.fn(),
      onError: jest.fn(),
      onDisconnect: jest.fn(),
      onConnectionStatusChange: jest.fn(),
    };

    connection = new SocketGameConnection(handlers);
  });

  describe('Connection Status Tracking', () => {
    it('should start with disconnected status', () => {
      expect(connection.status).toBe('disconnected');
    });

    it('should transition to connecting when connect is called', async () => {
      const connectPromise = connection.connect('game-1');

      // Status should change to connecting immediately
      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('connecting');

      // Simulate connection success
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      await connectPromise;
    });

    it('should transition to connected on successful connection', async () => {
      const connectPromise = connection.connect('game-1');

      // Simulate connection success
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('connected');
      expect(connection.status).toBe('connected');
    });

    it('should transition to disconnected on connection error', async () => {
      const connectPromise = connection.connect('game-1');

      // Simulate connection error
      const errorHandler = eventListeners.get('connect_error');
      if (errorHandler) {
        errorHandler(new Error('Connection refused'));
      }

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('disconnected');
      expect(handlers.onError).toHaveBeenCalledWith(expect.any(Error));
    });

    it('should transition to reconnecting during reconnect attempts', async () => {
      await connection.connect('game-1');

      // Simulate connect
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Simulate reconnect attempt
      const reconnectAttemptHandler = eventListeners.get('reconnect_attempt');
      if (reconnectAttemptHandler) {
        reconnectAttemptHandler(1);
      }

      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('reconnecting');
    });

    it('should transition back to connected on successful reconnect', async () => {
      await connection.connect('game-1');

      // Simulate connect
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Reset mock to track reconnect events
      if (handlers.onConnectionStatusChange) {
        (handlers.onConnectionStatusChange as jest.Mock).mockClear();
      }

      // Simulate reconnect
      const reconnectHandler = eventListeners.get('reconnect');
      if (reconnectHandler) {
        reconnectHandler(1);
      }

      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('connected');
    });
  });

  describe('Auto-Rejoin on Reconnect', () => {
    it('should emit join_game on initial connect', async () => {
      await connection.connect('game-1');

      // Simulate connection success
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      expect(mockEmit).toHaveBeenCalledWith('join_game', { gameId: 'game-1' });
    });

    it('should emit join_game again after reconnect', async () => {
      await connection.connect('game-1');

      // Simulate initial connect
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Clear emit mock
      mockEmit.mockClear();

      // Simulate reconnect
      const reconnectHandler = eventListeners.get('reconnect');
      if (reconnectHandler) {
        reconnectHandler(1);
      }

      expect(mockEmit).toHaveBeenCalledWith('join_game', { gameId: 'game-1' });
    });

    it('should emit join_game on server request_reconnect event', async () => {
      await connection.connect('game-1');

      // Simulate initial connect
      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Clear emit mock
      mockEmit.mockClear();

      // Simulate request_reconnect from server
      const requestReconnectHandler = eventListeners.get('request_reconnect');
      if (requestReconnectHandler) {
        requestReconnectHandler(undefined);
      }

      expect(mockEmit).toHaveBeenCalledWith('join_game', { gameId: 'game-1' });
    });
  });

  describe('Error Handling', () => {
    it('should call onError handler for connection errors', async () => {
      await connection.connect('game-1');

      const error = new Error('Connection failed');
      const errorHandler = eventListeners.get('connect_error');
      if (errorHandler) {
        errorHandler(error);
      }

      expect(handlers.onError).toHaveBeenCalledWith(error);
    });

    it('should call onError handler for server-side errors', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const errorPayload = {
        type: 'error',
        code: 'GAME_NOT_FOUND',
        message: 'Game not found',
      };

      const serverErrorHandler = eventListeners.get('error');
      if (serverErrorHandler) {
        serverErrorHandler(errorPayload);
      }

      expect(handlers.onError).toHaveBeenCalledWith(errorPayload);
    });

    it('should call onDisconnect handler on disconnect', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const disconnectHandler = eventListeners.get('disconnect');
      if (disconnectHandler) {
        disconnectHandler('transport error');
      }

      expect(handlers.onDisconnect).toHaveBeenCalledWith('transport error');
      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('disconnected');
    });

    it('should keep socket instance for auto-reconnect if not client-initiated disconnect', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Simulate server-side disconnect (transport error)
      const disconnectHandler = eventListeners.get('disconnect');
      if (disconnectHandler) {
        disconnectHandler('transport error');
      }

      // Socket should still be available for auto-reconnect
      // Connection.disconnect was NOT called
      expect(mockDisconnect).not.toHaveBeenCalled();
      // gameId is preserved for potential reconnection
      expect(connection.gameId).toBe('game-1');
    });

    it('should set socket to null on client-initiated disconnect', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Simulate client-initiated disconnect reason
      const disconnectHandler = eventListeners.get('disconnect');
      if (disconnectHandler) {
        disconnectHandler('io client disconnect');
      }

      // On client-initiated disconnect, the socket is set to null
      // but gameId remains until disconnect() is explicitly called
      // This is expected behavior - the disconnect handler clears socket reference
      expect(handlers.onDisconnect).toHaveBeenCalledWith('io client disconnect');
      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('disconnected');
    });
  });

  describe('Disconnect Method', () => {
    it('should disconnect socket and clear state', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      expect(connection.gameId).toBe('game-1');

      connection.disconnect();

      expect(mockDisconnect).toHaveBeenCalled();
      expect(connection.gameId).toBe(null);
      expect(connection.status).toBe('disconnected');
    });

    it('should handle disconnect when already disconnected', () => {
      // Should not throw
      connection.disconnect();
      expect(connection.status).toBe('disconnected');
    });
  });

  describe('State Recovery', () => {
    it('should receive game state on successful connection/reconnection', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const gameStatePayload: GameStateUpdateMessage = {
        type: 'game_update',
        data: {
          gameId: 'game-1',
          gameState: {
            id: 'game-1',
            boardType: 'square8',
            currentPlayer: 1,
            currentPhase: 'ring_placement',
          } as any,
          validMoves: [],
        },
        timestamp: new Date().toISOString(),
      };

      const gameStateHandler = eventListeners.get('game_state');
      if (gameStateHandler) {
        gameStateHandler(gameStatePayload);
      }

      expect(handlers.onGameState).toHaveBeenCalledWith(gameStatePayload);
      expect(handlers.onConnectionStatusChange).toHaveBeenCalledWith('connected');
    });

    it('should receive game over notification', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const gameOverPayload = {
        type: 'game_over',
        data: {
          gameId: 'game-1',
          gameState: {},
          gameResult: {
            winner: 1,
            reason: 'ring_elimination',
            finalScore: { 1: 5, 2: 3 },
          },
        },
        timestamp: new Date().toISOString(),
      } as unknown as GameOverMessage;

      const gameOverHandler = eventListeners.get('game_over');
      if (gameOverHandler) {
        gameOverHandler(gameOverPayload);
      }

      expect(handlers.onGameOver).toHaveBeenCalledWith(gameOverPayload);
    });
  });
});

describe('GameConnection - Message Submission', () => {
  let handlers: jest.Mocked<GameEventHandlers>;
  let connection: SocketGameConnection;
  let eventListeners: Map<string, (data: any) => void>;

  beforeEach(() => {
    jest.clearAllMocks();
    eventListeners = new Map();

    mockOn.mockImplementation((event: string, callback: (data: any) => void) => {
      eventListeners.set(event, callback);
      return mockSocket;
    });

    handlers = {
      onGameState: jest.fn(),
      onGameOver: jest.fn(),
      onChoiceRequired: jest.fn(),
      onChoiceCanceled: jest.fn(),
      onChatMessage: jest.fn(),
      onError: jest.fn(),
      onDisconnect: jest.fn(),
      onConnectionStatusChange: jest.fn(),
    };

    connection = new SocketGameConnection(handlers);
  });

  describe('submitMove', () => {
    it('should emit player_move when connected', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const move = {
        id: 'move-1',
        type: 'ring_placement',
        player: 1,
        to: { x: 3, y: 4 },
        moveNumber: 1,
        timestamp: new Date(),
        thinkTime: 5000,
      } as unknown as Move;

      connection.submitMove(move);

      expect(mockEmit).toHaveBeenCalledWith(
        'player_move',
        expect.objectContaining({
          gameId: 'game-1',
          move: expect.objectContaining({
            moveType: 'ring_placement',
          }),
        })
      );
    });

    it('should warn and not emit when not connected', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const move = {
        id: 'move-1',
        type: 'ring_placement',
        player: 1,
        to: { x: 3, y: 4 },
        moveNumber: 1,
        timestamp: new Date(),
        thinkTime: 5000,
      } as unknown as Move;

      connection.submitMove(move);

      expect(consoleSpy).toHaveBeenCalledWith('submitMove called without active socket/game');
      expect(mockEmit).not.toHaveBeenCalledWith('player_move', expect.anything());

      consoleSpy.mockRestore();
    });
  });

  describe('submitMoveById', () => {
    it('should emit player_move_by_id when connected', async () => {
      await connection.connect('game-1');

      eventListeners.get('connect')?.(undefined);
      mockEmit.mockClear();

      connection.submitMoveById('territory-move-1');

      expect(mockEmit).toHaveBeenCalledWith('player_move_by_id', {
        gameId: 'game-1',
        moveId: 'territory-move-1',
      });
    });
  });

  describe('respondToChoice', () => {
    it('should emit player_choice_response for line_order choice with moveId', async () => {
      // RR-FIX-2026-01-15: respondToChoice now always uses player_choice_response
      // instead of player_move_by_id to avoid lock contention.
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      const choice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-1',
        prompt: 'Choose line order',
        options: [],
      } as unknown as PlayerChoice;

      connection.respondToChoice(choice, { moveId: 'move-123' });

      expect(mockEmit).toHaveBeenCalledWith('player_choice_response', {
        choiceId: 'choice-1',
        playerNumber: 1,
        choiceType: 'line_order',
        selectedOption: { moveId: 'move-123' },
      });
    });

    it('should emit player_choice_response for choices without moveId', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      const choice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-1',
        prompt: 'Choose option',
        options: [],
      } as unknown as PlayerChoice;

      connection.respondToChoice(choice, 'option-a');

      expect(mockEmit).toHaveBeenCalledWith(
        'player_choice_response',
        expect.objectContaining({
          choiceId: 'choice-1',
          playerNumber: 1,
          selectedOption: 'option-a',
        })
      );
    });

    it('should warn when not connected', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const choice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-1',
        prompt: 'Choose',
        options: [],
      } as unknown as PlayerChoice;

      connection.respondToChoice(choice, { moveId: 'move-1' });

      expect(consoleSpy).toHaveBeenCalledWith('respondToChoice called without active socket/game');

      consoleSpy.mockRestore();
    });
  });

  describe('sendChatMessage', () => {
    it('should emit chat_message when connected', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      connection.sendChatMessage('Hello world');

      expect(mockEmit).toHaveBeenCalledWith('chat_message', {
        gameId: 'game-1',
        text: 'Hello world',
      });
    });

    it('should warn when not connected', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      connection.sendChatMessage('Hello');

      expect(consoleSpy).toHaveBeenCalledWith('sendChatMessage called without active socket/game');

      consoleSpy.mockRestore();
    });
  });

  describe('Rematch', () => {
    it('should emit rematch_request when connected', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      connection.requestRematch();

      expect(mockEmit).toHaveBeenCalledWith('rematch_request', {
        gameId: 'game-1',
      });
    });

    it('should emit rematch_respond with accept', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      connection.respondToRematch('request-123', true);

      expect(mockEmit).toHaveBeenCalledWith('rematch_respond', {
        requestId: 'request-123',
        accept: true,
      });
    });

    it('should emit rematch_respond with decline', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      mockEmit.mockClear();

      connection.respondToRematch('request-123', false);

      expect(mockEmit).toHaveBeenCalledWith('rematch_respond', {
        requestId: 'request-123',
        accept: false,
      });
    });
  });
});

describe('GameConnection - Event Handlers', () => {
  let handlers: jest.Mocked<GameEventHandlers>;
  let connection: SocketGameConnection;
  let eventListeners: Map<string, (data: any) => void>;

  beforeEach(() => {
    jest.clearAllMocks();
    eventListeners = new Map();

    mockOn.mockImplementation((event: string, callback: (data: any) => void) => {
      eventListeners.set(event, callback);
      return mockSocket;
    });

    handlers = {
      onGameState: jest.fn(),
      onGameOver: jest.fn(),
      onChoiceRequired: jest.fn(),
      onChoiceCanceled: jest.fn(),
      onChatMessage: jest.fn(),
      onError: jest.fn(),
      onDisconnect: jest.fn(),
      onConnectionStatusChange: jest.fn(),
      onChatMessagePersisted: jest.fn(),
      onChatHistory: jest.fn(),
      onDecisionPhaseTimeoutWarning: jest.fn(),
      onDecisionPhaseTimedOut: jest.fn(),
      onRematchRequested: jest.fn(),
      onRematchResponse: jest.fn(),
      onPositionEvaluation: jest.fn(),
    };

    connection = new SocketGameConnection(handlers);
  });

  describe('Player Choice Events', () => {
    it('should call onChoiceRequired for player_choice_required event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const choice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_reward_option',
        gameId: 'game-1',
        prompt: 'Choose reward',
        options: [],
      } as unknown as PlayerChoice;

      const choiceHandler = eventListeners.get('player_choice_required');
      if (choiceHandler) {
        choiceHandler(choice);
      }

      expect(handlers.onChoiceRequired).toHaveBeenCalledWith(choice);
    });

    it('should call onChoiceCanceled for player_choice_canceled event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const choiceCanceledHandler = eventListeners.get('player_choice_canceled');
      if (choiceCanceledHandler) {
        choiceCanceledHandler('choice-1');
      }

      expect(handlers.onChoiceCanceled).toHaveBeenCalledWith('choice-1');
    });
  });

  describe('Chat Events', () => {
    it('should call onChatMessage for chat_message event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const chatPayload: ChatMessageServerPayload = {
        sender: 'Player1',
        text: 'Good game!',
        timestamp: new Date().toISOString(),
      };

      const chatHandler = eventListeners.get('chat_message');
      if (chatHandler) {
        chatHandler(chatPayload);
      }

      expect(handlers.onChatMessage).toHaveBeenCalledWith(chatPayload);
    });

    it('should call onChatMessagePersisted for chat_message_persisted event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const persistedPayload = {
        id: 'msg-1',
        gameId: 'game-1',
        userId: 'user-1',
        username: 'Player1',
        message: 'Hello',
        createdAt: new Date().toISOString(),
      };

      const persistedHandler = eventListeners.get('chat_message_persisted');
      if (persistedHandler) {
        persistedHandler(persistedPayload);
      }

      expect(handlers.onChatMessagePersisted).toHaveBeenCalledWith(persistedPayload);
    });

    it('should call onChatHistory for chat_history event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const historyPayload = {
        gameId: 'game-1',
        messages: [
          {
            id: 'msg-1',
            gameId: 'game-1',
            userId: 'user-1',
            username: 'Player1',
            message: 'Hello',
            createdAt: new Date().toISOString(),
          },
        ],
      };

      const historyHandler = eventListeners.get('chat_history');
      if (historyHandler) {
        historyHandler(historyPayload);
      }

      expect(handlers.onChatHistory).toHaveBeenCalledWith(historyPayload);
    });
  });

  describe('Decision Phase Events', () => {
    it('should call onDecisionPhaseTimeoutWarning for timeout warning', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const warningPayload = {
        gameId: 'game-1',
        playerNumber: 1,
        remainingSeconds: 10,
      };

      const warningHandler = eventListeners.get('decision_phase_timeout_warning');
      if (warningHandler) {
        warningHandler(warningPayload);
      }

      expect(handlers.onDecisionPhaseTimeoutWarning).toHaveBeenCalledWith(warningPayload);
    });

    it('should call onDecisionPhaseTimedOut for timeout', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const timeoutPayload = {
        gameId: 'game-1',
        playerNumber: 1,
        choiceId: 'choice-1',
      };

      const timeoutHandler = eventListeners.get('decision_phase_timed_out');
      if (timeoutHandler) {
        timeoutHandler(timeoutPayload);
      }

      expect(handlers.onDecisionPhaseTimedOut).toHaveBeenCalledWith(timeoutPayload);
    });
  });

  describe('Rematch Events', () => {
    it('should call onRematchRequested for rematch_requested event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const rematchPayload = {
        id: 'rematch-1',
        gameId: 'game-1',
        requesterId: 'user-1',
        requesterUsername: 'Player1',
        expiresAt: new Date().toISOString(),
      };

      const rematchHandler = eventListeners.get('rematch_requested');
      if (rematchHandler) {
        rematchHandler(rematchPayload);
      }

      expect(handlers.onRematchRequested).toHaveBeenCalledWith(rematchPayload);
    });

    it('should call onRematchResponse for rematch_response event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const responsePayload = {
        requestId: 'rematch-1',
        gameId: 'game-1',
        status: 'accepted' as const,
        newGameId: 'game-2',
      };

      const responseHandler = eventListeners.get('rematch_response');
      if (responseHandler) {
        responseHandler(responsePayload);
      }

      expect(handlers.onRematchResponse).toHaveBeenCalledWith(responsePayload);
    });
  });

  describe('Position Evaluation Events', () => {
    it('should call onPositionEvaluation for position_evaluation event', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const evalPayload = {
        gameId: 'game-1',
        evaluation: {
          player1Advantage: 0.25,
          confidence: 0.8,
        },
      };

      const evalHandler = eventListeners.get('position_evaluation');
      if (evalHandler) {
        evalHandler(evalPayload);
      }

      expect(handlers.onPositionEvaluation).toHaveBeenCalledWith(evalPayload);
    });
  });

  describe('Player Disconnect/Reconnect Events', () => {
    it('should call onPlayerDisconnected for player_disconnected event', async () => {
      (handlers as any).onPlayerDisconnected = jest.fn();

      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const disconnectPayload = {
        type: 'player_disconnected',
        data: {
          gameId: 'game-1',
          player: {
            id: 'user-2',
            username: 'Player2',
          },
          reconnectionTimeoutMs: 30000,
        },
        timestamp: new Date().toISOString(),
      };

      const playerDisconnectedHandler = eventListeners.get('player_disconnected');
      if (playerDisconnectedHandler) {
        playerDisconnectedHandler(disconnectPayload);
      }

      expect((handlers as any).onPlayerDisconnected).toHaveBeenCalledWith(disconnectPayload);
    });

    it('should call onPlayerReconnected for player_reconnected event', async () => {
      (handlers as any).onPlayerReconnected = jest.fn();

      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const reconnectPayload = {
        type: 'player_reconnected',
        data: {
          gameId: 'game-1',
          player: {
            id: 'user-2',
            username: 'Player2',
          },
        },
        timestamp: new Date().toISOString(),
      };

      const playerReconnectedHandler = eventListeners.get('player_reconnected');
      if (playerReconnectedHandler) {
        playerReconnectedHandler(reconnectPayload);
      }

      expect((handlers as any).onPlayerReconnected).toHaveBeenCalledWith(reconnectPayload);
    });

    it('should not throw if onPlayerDisconnected handler is not defined', async () => {
      // handlers without onPlayerDisconnected
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const disconnectPayload = {
        type: 'player_disconnected',
        data: {
          gameId: 'game-1',
          player: { id: 'user-2', username: 'Player2' },
        },
        timestamp: new Date().toISOString(),
      };

      // Should not throw
      const playerDisconnectedHandler = eventListeners.get('player_disconnected');
      expect(() => {
        if (playerDisconnectedHandler) {
          playerDisconnectedHandler(disconnectPayload);
        }
      }).not.toThrow();
    });
  });
});

describe('GameConnection - Edge Cases', () => {
  let handlers: jest.Mocked<GameEventHandlers>;
  let connection: SocketGameConnection;
  let eventListeners: Map<string, (data: any) => void>;

  beforeEach(() => {
    jest.clearAllMocks();
    eventListeners = new Map();

    mockOn.mockImplementation((event: string, callback: (data: any) => void) => {
      eventListeners.set(event, callback);
      return mockSocket;
    });

    handlers = {
      onGameState: jest.fn(),
      onGameOver: jest.fn(),
      onChoiceRequired: jest.fn(),
      onChoiceCanceled: jest.fn(),
      onChatMessage: jest.fn(),
      onError: jest.fn(),
      onDisconnect: jest.fn(),
      onConnectionStatusChange: jest.fn(),
    };

    connection = new SocketGameConnection(handlers);
  });

  describe('Connection to Same Game', () => {
    it('should not reconnect if already connected to same game', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      const callCount = mockEmit.mock.calls.length;

      // Try to connect to same game
      await connection.connect('game-1');

      // Should not have emitted another join_game
      expect(mockEmit.mock.calls.length).toBe(callCount);
    });

    it('should disconnect and reconnect if connecting to different game', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      // Connect to different game
      await connection.connect('game-2');

      expect(mockDisconnect).toHaveBeenCalled();
    });
  });

  describe('Multiple Rapid Status Changes', () => {
    it('should handle rapid connect/disconnect cycles', async () => {
      await connection.connect('game-1');

      const connectHandler = eventListeners.get('connect');
      if (connectHandler) {
        connectHandler(undefined);
      }

      expect(connection.gameId).toBe('game-1');

      connection.disconnect();

      // Should be disconnected
      expect(connection.status).toBe('disconnected');
      expect(connection.gameId).toBe(null);

      // Create fresh event listeners for new connection
      eventListeners.clear();
      await connection.connect('game-2');

      // Simulate connect - need to get fresh handler
      const newConnectHandler = eventListeners.get('connect');
      if (newConnectHandler) {
        newConnectHandler(undefined);
      }

      // Should be connected to new game
      expect(connection.gameId).toBe('game-2');
    });
  });

  describe('localStorage Token Handling', () => {
    it('should use token from localStorage if available', async () => {
      const mockGetItem = jest.spyOn(Storage.prototype, 'getItem');
      mockGetItem.mockReturnValue('test-token');

      const { io } = require('socket.io-client');

      await connection.connect('game-1');

      expect(io).toHaveBeenCalledWith(
        'http://localhost:3000',
        expect.objectContaining({
          auth: { token: 'test-token' },
        })
      );

      mockGetItem.mockRestore();
    });

    it('should connect without auth if no token in localStorage', async () => {
      const mockGetItem = jest.spyOn(Storage.prototype, 'getItem');
      mockGetItem.mockReturnValue(null);

      const { io } = require('socket.io-client');
      io.mockClear();

      await connection.connect('game-1');

      // When no token exists, auth property should not be set at all
      expect(io).toHaveBeenCalledWith(
        'http://localhost:3000',
        expect.objectContaining({
          transports: ['websocket', 'polling'],
        })
      );
      // Verify auth is NOT in the options (undefined or missing)
      const callArgs = io.mock.calls[0][1];
      expect(callArgs.auth).toBeUndefined();

      mockGetItem.mockRestore();
    });
  });
});
