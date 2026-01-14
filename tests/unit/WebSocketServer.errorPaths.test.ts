/**
 * Tests for WebSocket Server error paths and edge cases.
 *
 * This file covers error handling scenarios not addressed in other WebSocket test files:
 * - Malformed message handling
 * - Invalid payload validation
 * - Connection error scenarios
 * - Rate limiting behavior
 * - Server-initiated close scenarios
 * - Session termination
 *
 * Related spec: docs/archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md (§2.4 Connection sub-states)
 */

import { WebSocketServer } from '../../src/server/websocket/server';
import { ZodError } from 'zod';

type AuthenticatedTestSocket = any;

// Mock database connection
const mockGameFindUnique = jest.fn();
const mockUserFindUnique = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockGameFindUnique,
      update: jest.fn(),
    },
    user: {
      findUnique: mockUserFindUnique,
    },
  }),
}));

// Mock GameSessionManager
const mockGetOrCreateSession = jest.fn();
const mockGetSession = jest.fn();

jest.mock('../../src/server/game/GameSessionManager', () => {
  return {
    GameSessionManager: jest.fn().mockImplementation(() => ({
      getOrCreateSession: mockGetOrCreateSession,
      getSession: mockGetSession,
      withGameLock: async (_gameId: string, operation: () => Promise<unknown>) => operation(),
    })),
  };
});

// Mock Redis client (for rate limiting)
const mockRedisIncr = jest.fn();
const mockRedisExpire = jest.fn();

jest.mock('../../src/server/cache/redis', () => ({
  getRedisClient: () => ({
    incr: mockRedisIncr,
    expire: mockRedisExpire,
  }),
}));

// Mock chat persistence service
jest.mock('../../src/server/services/ChatPersistenceService', () => ({
  getChatPersistenceService: () => ({
    saveMessage: jest.fn().mockResolvedValue({
      id: 'msg-1',
      gameId: 'test-game',
      userId: 'user-1',
      username: 'Tester',
      message: 'Hello',
      createdAt: new Date(),
    }),
    getMessagesForGame: jest.fn().mockResolvedValue([]),
  }),
}));

// Mock rematch service with shared instance so tests can
// configure behaviour and inspect calls.
const mockRematchService = {
  createRematchRequest: jest.fn(),
  acceptRematch: jest.fn(),
  declineRematch: jest.fn(),
};

jest.mock('../../src/server/services/RematchService', () => ({
  getRematchService: () => mockRematchService,
}));

// Mock metrics service
jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    incWebSocketConnections: jest.fn(),
    decWebSocketConnections: jest.fn(),
    recordWebsocketReconnection: jest.fn(),
    recordMoveRejected: jest.fn(),
  }),
}));

jest.mock('../../src/server/utils/rulesParityMetrics', () => ({
  webSocketConnectionsGauge: {
    inc: jest.fn(),
    dec: jest.fn(),
  },
}));

// Mock auth middleware
type MiddlewareFn = (socket: any, next: (err?: Error) => void) => void;
let capturedMiddleware: MiddlewareFn | null = null;

jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  return {
    ...actual,
    verifyToken: jest.fn(() => ({
      userId: 'user-1',
      email: 'user1@example.com',
      tokenVersion: 0,
    })),
    validateUser: jest.fn().mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      isActive: true,
      tokenVersion: 0,
    }),
  };
});

// Mock socket.io
let connectionHandler: ((socket: any) => void) | null = null;

jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public use = jest.fn((fn: MiddlewareFn) => {
      capturedMiddleware = fn;
    });

    public on = jest.fn((event: string, handler: (socket: any) => void) => {
      if (event === 'connection') {
        connectionHandler = handler;
      }
    });

    public to = jest.fn(() => ({
      emit: jest.fn(),
    }));

    public sockets = {
      adapter: {
        rooms: new Map<string, Set<string>>(),
      },
      sockets: new Map<string, any>(),
    };

    constructor(..._args: any[]) {}
  }

  return {
    Server: FakeSocketIOServer,
  };
});

describe('WebSocket Server - Error Paths', () => {
  const gameId = 'error-test-game';
  const userId = 'user-1';

  let wsServer: WebSocketServer;
  let mockSession: any;
  let socket: AuthenticatedTestSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    capturedMiddleware = null;
    connectionHandler = null;

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
      handleChoiceResponse: jest.fn(),
    };

    const gameState = {
      id: gameId,
      players: [
        {
          id: userId,
          username: 'Tester',
          type: 'human',
          playerNumber: 1,
        },
      ],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      gameStatus: 'active',
      isRated: false,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession = {
      getGameState: jest.fn(() => gameState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
      handlePlayerMove: jest.fn(),
      handlePlayerMoveById: jest.fn(),
      maybePerformAITurn: jest.fn(),
    };

    mockGetOrCreateSession.mockResolvedValue(mockSession);
    mockGetSession.mockReturnValue(mockSession);

    mockGameFindUnique.mockResolvedValue({
      id: gameId,
      allowSpectators: true,
      player1Id: userId,
      player2Id: null,
      player3Id: null,
      player4Id: null,
    });

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);

    socket = {
      id: 'socket-1',
      userId,
      username: 'Tester',
      gameId: undefined,
      join: jest.fn(),
      leave: jest.fn(),
      emit: jest.fn(),
      to: jest.fn(() => ({ emit: jest.fn() })),
      rooms: new Set<string>(),
      on: jest.fn(),
      disconnect: jest.fn(),
      handshake: {
        auth: { token: 'valid-token' },
        query: {},
      },
    } as any;
  });

  describe('Malformed Message Handling', () => {
    it('should emit INVALID_PAYLOAD error for malformed join_game payload', async () => {
      // Trigger connection to set up event handlers
      if (connectionHandler) {
        connectionHandler(socket);
      }

      // Find the join_game handler from socket.on calls
      const onCalls = socket.on.mock.calls;
      const joinGameHandler = onCalls.find(([event]: [string]) => event === 'join_game')?.[1];

      if (!joinGameHandler) {
        throw new Error('join_game handler not registered');
      }

      // Call with invalid payload (missing gameId)
      await joinGameHandler({});

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
          message: 'Invalid payload',
          event: 'join_game',
        })
      );
    });

    it('should emit INVALID_PAYLOAD error for malformed player_move payload', async () => {
      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const moveHandler = onCalls.find(([event]: [string]) => event === 'player_move')?.[1];

      if (!moveHandler) {
        throw new Error('player_move handler not registered');
      }

      // Call with invalid payload (missing required fields)
      await moveHandler({ gameId: 'test', move: {} });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
        })
      );
    });

    it('should emit INVALID_PAYLOAD error for malformed chat_message payload', async () => {
      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const chatHandler = onCalls.find(([event]: [string]) => event === 'chat_message')?.[1];

      if (!chatHandler) {
        throw new Error('chat_message handler not registered');
      }

      // Call with invalid payload (missing text)
      await chatHandler({ gameId: 'test' });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
        })
      );
    });

    it('should emit CHOICE_REJECTED for malformed player_choice_response', async () => {
      socket.gameId = gameId;

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const choiceHandler = onCalls.find(
        ([event]: [string]) => event === 'player_choice_response'
      )?.[1];

      if (!choiceHandler) {
        throw new Error('player_choice_response handler not registered');
      }

      // Call with invalid payload
      await choiceHandler({});

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
        })
      );
    });

    it('should emit INVALID_PAYLOAD error for malformed rematch_request payload', async () => {
      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const rematchHandler = onCalls.find(([event]: [string]) => event === 'rematch_request')?.[1];

      if (!rematchHandler) {
        throw new Error('rematch_request handler not registered');
      }

      // Call with invalid payload (missing gameId)
      await rematchHandler({});

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
          event: 'rematch_request',
        })
      );
    });

    it('should emit INVALID_PAYLOAD error for malformed rematch_respond payload', async () => {
      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const respondHandler = onCalls.find(([event]: [string]) => event === 'rematch_respond')?.[1];

      if (!respondHandler) {
        throw new Error('rematch_respond handler not registered');
      }

      // Call with invalid payload (missing requestId)
      await respondHandler({});

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INVALID_PAYLOAD',
          event: 'rematch_respond',
        })
      );
    });
  });

  describe('Authentication Error Handling', () => {
    it('should emit ACCESS_DENIED for missing auth token', async () => {
      const socketWithoutToken: AuthenticatedTestSocket = {
        ...socket,
        id: 'socket-no-token',
        handshake: {
          auth: {},
          query: {},
        },
      };

      if (!capturedMiddleware) {
        throw new Error('Auth middleware not registered');
      }

      await new Promise<void>((resolve) => {
        capturedMiddleware!(socketWithoutToken, (err?: Error) => {
          resolve();
        });
      });

      expect(socketWithoutToken.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Authentication token required',
        })
      );
    });

    it('should emit ACCESS_DENIED when token validation fails', async () => {
      const { verifyToken } = require('../../src/server/middleware/auth');
      verifyToken.mockImplementationOnce(() => {
        throw new Error('Invalid token');
      });

      const socketWithBadToken: AuthenticatedTestSocket = {
        ...socket,
        id: 'socket-bad-token',
        handshake: {
          auth: { token: 'invalid-token' },
          query: {},
        },
      };

      if (!capturedMiddleware) {
        throw new Error('Auth middleware not registered');
      }

      await new Promise<void>((resolve) => {
        capturedMiddleware!(socketWithBadToken, (err?: Error) => {
          resolve();
        });
      });

      expect(socketWithBadToken.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Authentication failed',
        })
      );
    });

    it('should emit ACCESS_DENIED when user validation fails', async () => {
      const { validateUser } = require('../../src/server/middleware/auth');
      validateUser.mockRejectedValueOnce(new Error('User not found'));

      if (!capturedMiddleware) {
        throw new Error('Auth middleware not registered');
      }

      await new Promise<void>((resolve) => {
        capturedMiddleware!(socket, (err?: Error) => {
          resolve();
        });
      });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Authentication failed',
        })
      );
    });

    it('should emit ACCESS_DENIED when unauthenticated user sends rematch_request', async () => {
      const unauthSocket: AuthenticatedTestSocket = {
        ...socket,
        id: 'socket-unauth-rematch',
        userId: undefined,
      };

      if (connectionHandler) {
        connectionHandler(unauthSocket);
      }

      const onCalls = unauthSocket.on.mock.calls;
      const rematchHandler = onCalls.find(([event]: [string]) => event === 'rematch_request')?.[1];

      if (!rematchHandler) {
        throw new Error('rematch_request handler not registered');
      }

      await rematchHandler({ gameId });

      expect(unauthSocket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          event: 'rematch_request',
        })
      );
    });

    it('should emit ACCESS_DENIED when unauthenticated user sends rematch_respond', async () => {
      const unauthSocket: AuthenticatedTestSocket = {
        ...socket,
        id: 'socket-unauth-rematch-respond',
        userId: undefined,
      };

      if (connectionHandler) {
        connectionHandler(unauthSocket);
      }

      const onCalls = unauthSocket.on.mock.calls;
      const respondHandler = onCalls.find(([event]: [string]) => event === 'rematch_respond')?.[1];

      if (!respondHandler) {
        throw new Error('rematch_respond handler not registered');
      }

      await respondHandler({ requestId: 'req-1', accept: true });

      expect(unauthSocket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          event: 'rematch_respond',
        })
      );
    });
  });

  describe('Game Access Error Handling', () => {
    it('should emit GAME_NOT_FOUND for non-existent game', async () => {
      mockGameFindUnique.mockResolvedValueOnce(null);

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const joinGameHandler = onCalls.find(([event]: [string]) => event === 'join_game')?.[1];

      if (!joinGameHandler) {
        throw new Error('join_game handler not registered');
      }

      await joinGameHandler({ gameId: 'nonexistent-game' });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'GAME_NOT_FOUND',
          message: 'Game not found',
        })
      );
    });

    it('should emit ACCESS_DENIED for private game when not a player', async () => {
      mockGameFindUnique.mockResolvedValueOnce({
        id: gameId,
        allowSpectators: false,
        player1Id: 'other-user',
        player2Id: 'another-user',
        player3Id: null,
        player4Id: null,
      });

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const joinGameHandler = onCalls.find(([event]: [string]) => event === 'join_game')?.[1];

      if (!joinGameHandler) {
        throw new Error('join_game handler not registered');
      }

      await joinGameHandler({ gameId });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'You are not allowed to join this game',
        })
      );
    });

    it('should emit ACCESS_DENIED for move when not in game room', async () => {
      socket.gameId = undefined; // Not in any game room

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const moveHandler = onCalls.find(([event]: [string]) => event === 'player_move')?.[1];

      if (!moveHandler) {
        throw new Error('player_move handler not registered');
      }

      // Use valid moveType 'place_ring' (not 'ring_placement') to pass schema validation
      await moveHandler({
        gameId,
        move: {
          moveNumber: 1,
          position: JSON.stringify({ to: { x: 0, y: 0 } }),
          moveType: 'place_ring',
        },
      });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
        })
      );
    });
  });

  describe('Rate Limiting', () => {
    it('should emit RATE_LIMITED when chat rate limit is exceeded via Redis', async () => {
      socket.gameId = gameId;
      socket.rooms.add(gameId);

      // Simulate rate limit exceeded - Redis returns high count
      mockRedisIncr.mockResolvedValue(25); // Over the 20 message limit

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const chatHandler = onCalls.find(([event]: [string]) => event === 'chat_message')?.[1];

      if (!chatHandler) {
        throw new Error('chat_message handler not registered');
      }

      await chatHandler({ gameId, text: 'Test message' });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'RATE_LIMITED',
          message: 'Chat rate limit exceeded',
        })
      );
    });

    it('should allow chat when under rate limit', async () => {
      socket.gameId = gameId;
      socket.rooms.add(gameId);

      // Simulate under rate limit
      mockRedisIncr.mockResolvedValue(5); // Under the 20 message limit

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const chatHandler = onCalls.find(([event]: [string]) => event === 'chat_message')?.[1];

      if (!chatHandler) {
        throw new Error('chat_message handler not registered');
      }

      await chatHandler({ gameId, text: 'Test message' });

      // Should not have rate limit error
      const errorCalls = socket.emit.mock.calls.filter(
        ([event, payload]: [string, any]) => event === 'error' && payload.code === 'RATE_LIMITED'
      );
      expect(errorCalls.length).toBe(0);
    });
  });

  describe('Session Termination', () => {
    it('should terminate user sessions and emit ACCESS_DENIED', () => {
      // Setup: Add socket to the internal sockets map
      const io = (wsServer as any).io;
      io.sockets.sockets.set(socket.id, socket);
      (wsServer as any).userSockets.set(userId, socket.id);

      // Terminate user sessions
      const count = wsServer.terminateUserSessions(userId, 'Account deleted');

      expect(count).toBe(1);
      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Account deleted',
        })
      );
      expect(socket.disconnect).toHaveBeenCalledWith(true);
    });

    it('should return 0 when user has no active sessions', () => {
      const count = wsServer.terminateUserSessions('nonexistent-user');
      expect(count).toBe(0);
    });

    it('should clean up stale socket mappings', () => {
      // Setup: Add user to userSockets but not to actual sockets map
      (wsServer as any).userSockets.set('stale-user', 'stale-socket-id');
      // do NOT add to io.sockets.sockets

      const count = wsServer.terminateUserSessions('stale-user');
      expect(count).toBe(0);
      expect((wsServer as any).userSockets.has('stale-user')).toBe(false);
    });

    it('should clear pending reconnection timeouts for terminated user', () => {
      // Setup: Add socket and pending reconnection
      const io = (wsServer as any).io;
      io.sockets.sockets.set(socket.id, socket);
      (wsServer as any).userSockets.set(userId, socket.id);

      const timeout = setTimeout(() => {}, 30000);
      (wsServer as any).pendingReconnections.set(`${gameId}:${userId}`, {
        timeout,
        playerNumber: 1,
        gameId,
        userId,
      });

      const count = wsServer.terminateUserSessions(userId);

      expect(count).toBe(1);
      expect((wsServer as any).pendingReconnections.has(`${gameId}:${userId}`)).toBe(false);

      clearTimeout(timeout);
    });
  });

  describe('Move Rejection Scenarios', () => {
    it('should emit MOVE_REJECTED when engine rejects move', async () => {
      socket.gameId = gameId;
      mockSession.handlePlayerMove.mockRejectedValueOnce(new Error('Invalid move'));

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const moveHandler = onCalls.find(([event]: [string]) => event === 'player_move')?.[1];

      if (!moveHandler) {
        throw new Error('player_move handler not registered');
      }

      // Use valid moveType 'place_ring' (not 'ring_placement') to pass schema validation
      await moveHandler({
        gameId,
        move: {
          moveNumber: 1,
          position: JSON.stringify({ to: { x: 0, y: 0 } }),
          moveType: 'place_ring',
        },
      });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'MOVE_REJECTED',
          message: 'Move was not valid in the current game state',
        })
      );
    });

    it('should emit MOVE_REJECTED when move_by_id fails', async () => {
      socket.gameId = gameId;
      mockSession.handlePlayerMoveById.mockRejectedValueOnce(new Error('Invalid move id'));

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const moveByIdHandler = onCalls.find(
        ([event]: [string]) => event === 'player_move_by_id'
      )?.[1];

      if (!moveByIdHandler) {
        throw new Error('player_move_by_id handler not registered');
      }

      await moveByIdHandler({
        gameId,
        moveId: 'invalid-move-id',
      });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'MOVE_REJECTED',
        })
      );
    });
  });

  describe('Spectator Move Rejection', () => {
    it('should emit ACCESS_DENIED when spectator tries to make move', async () => {
      const spectatorUserId = 'spectator-user';
      const spectatorSocket: AuthenticatedTestSocket = {
        ...socket,
        id: 'spectator-socket',
        userId: spectatorUserId,
        gameId,
      };

      const spectatorGameState = {
        id: gameId,
        players: [{ id: userId, username: 'Player', type: 'human', playerNumber: 1 }],
        spectators: [spectatorUserId],
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      };

      mockSession.getGameState.mockReturnValue(spectatorGameState);
      mockSession.handlePlayerMove.mockRejectedValueOnce(new Error('Spectators cannot make moves'));

      if (connectionHandler) {
        connectionHandler(spectatorSocket);
      }

      const onCalls = spectatorSocket.on.mock.calls;
      const moveHandler = onCalls.find(([event]: [string]) => event === 'player_move')?.[1];

      if (!moveHandler) {
        throw new Error('player_move handler not registered');
      }

      // Use valid moveType 'place_ring' (not 'ring_placement') to pass schema validation
      await moveHandler({
        gameId,
        move: {
          moveNumber: 1,
          position: JSON.stringify({ to: { x: 0, y: 0 } }),
          moveType: 'place_ring',
        },
      });

      expect(spectatorSocket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'You are not allowed to make moves in this game',
        })
      );
    });

    it('should emit ACCESS_DENIED when spectator tries to respond to choice', async () => {
      const spectatorUserId = 'spectator-user';
      const spectatorSocket: AuthenticatedTestSocket = {
        ...socket,
        id: 'spectator-socket',
        userId: spectatorUserId,
        gameId,
      };

      const spectatorGameState = {
        id: gameId,
        players: [{ id: userId, username: 'Player', type: 'human', playerNumber: 1 }],
        spectators: [spectatorUserId],
        currentPhase: 'line_processing',
        currentPlayer: 1,
        gameStatus: 'active',
      };

      mockSession.getGameState.mockReturnValue(spectatorGameState);

      if (connectionHandler) {
        connectionHandler(spectatorSocket);
      }

      const onCalls = spectatorSocket.on.mock.calls;
      const choiceHandler = onCalls.find(
        ([event]: [string]) => event === 'player_choice_response'
      )?.[1];

      if (!choiceHandler) {
        throw new Error('player_choice_response handler not registered');
      }

      await choiceHandler({
        choiceId: 'choice-1',
        playerNumber: 1,
        choiceType: 'line_reward_option',
        selectedOption: 'eliminate_opponent',
      });

      expect(spectatorSocket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'ACCESS_DENIED',
          message: 'Spectators cannot respond to player choices',
        })
      );
    });
  });

  describe('Database Unavailability', () => {
    it('should emit INTERNAL_ERROR when database is unavailable during join', async () => {
      // Simulate a database failure scenario by having game lookup fail
      mockGameFindUnique.mockRejectedValueOnce(new Error('Connection refused'));

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const onCalls = socket.on.mock.calls;
      const joinGameHandler = onCalls.find(([event]: [string]) => event === 'join_game')?.[1];

      if (!joinGameHandler) {
        throw new Error('join_game handler not registered');
      }

      await joinGameHandler({ gameId });

      expect(socket.emit).toHaveBeenCalledWith(
        'error',
        expect.objectContaining({
          type: 'error',
          code: 'INTERNAL_ERROR',
        })
      );
    });
  });

  describe('Rematch Events', () => {
    it('should delegate rematch_request to RematchService and broadcast rematch_requested', async () => {
      const expiresAt = new Date('2025-12-01T00:00:00.000Z');

      mockRematchService.createRematchRequest.mockResolvedValue({
        success: true,
        request: {
          id: 'req-1',
          gameId,
          requesterId: userId,
          requesterUsername: 'Tester',
          expiresAt,
        },
      });

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const io = (wsServer as any).io;
      const roomEmitter = { emit: jest.fn() };
      io.to = jest.fn(() => roomEmitter);

      const onCalls = socket.on.mock.calls;
      const rematchHandler = onCalls.find(([event]: [string]) => event === 'rematch_request')?.[1];

      if (!rematchHandler) {
        throw new Error('rematch_request handler not registered');
      }

      await rematchHandler({ gameId });

      expect(mockRematchService.createRematchRequest).toHaveBeenCalledWith(gameId, userId);
      expect(io.to).toHaveBeenCalledWith(gameId);
      expect(roomEmitter.emit).toHaveBeenCalledWith(
        'rematch_requested',
        expect.objectContaining({
          id: 'req-1',
          gameId,
          requesterId: userId,
          requesterUsername: 'Tester',
          expiresAt: expiresAt.toISOString(),
        })
      );
    });

    it('should broadcast accepted rematch_response when accept=true', async () => {
      mockRematchService.acceptRematch.mockImplementation(
        async (
          _requestId: string,
          _userId: string,
          _createGame: (gameId: string) => Promise<string>
        ) => ({
          success: true,
          request: {
            id: 'req-1',
            gameId,
          },
          newGameId: 'new-game-id',
        })
      );

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const io = (wsServer as any).io;
      const roomEmitter = { emit: jest.fn() };
      io.to = jest.fn(() => roomEmitter);

      const onCalls = socket.on.mock.calls;
      const respondHandler = onCalls.find(([event]: [string]) => event === 'rematch_respond')?.[1];

      if (!respondHandler) {
        throw new Error('rematch_respond handler not registered');
      }

      await respondHandler({ requestId: 'req-1', accept: true });

      expect(mockRematchService.acceptRematch).toHaveBeenCalledWith(
        'req-1',
        userId,
        expect.any(Function)
      );
      expect(io.to).toHaveBeenCalledWith(gameId);
      expect(roomEmitter.emit).toHaveBeenCalledWith(
        'rematch_response',
        expect.objectContaining({
          requestId: 'req-1',
          gameId,
          status: 'accepted',
          newGameId: 'new-game-id',
        })
      );
    });

    it('should broadcast declined rematch_response when accept=false', async () => {
      mockRematchService.declineRematch.mockResolvedValue({
        success: true,
        request: {
          id: 'req-1',
          gameId,
        },
      });

      if (connectionHandler) {
        connectionHandler(socket);
      }

      const io = (wsServer as any).io;
      const roomEmitter = { emit: jest.fn() };
      io.to = jest.fn(() => roomEmitter);

      const onCalls = socket.on.mock.calls;
      const respondHandler = onCalls.find(([event]: [string]) => event === 'rematch_respond')?.[1];

      if (!respondHandler) {
        throw new Error('rematch_respond handler not registered');
      }

      await respondHandler({ requestId: 'req-1', accept: false });

      expect(mockRematchService.declineRematch).toHaveBeenCalledWith('req-1', userId);
      expect(io.to).toHaveBeenCalledWith(gameId);
      expect(roomEmitter.emit).toHaveBeenCalledWith(
        'rematch_response',
        expect.objectContaining({
          requestId: 'req-1',
          gameId,
          status: 'declined',
        })
      );
    });
  });
});

describe('WebSocket Server - Server-Initiated Close Scenarios', () => {
  const gameId = 'close-test-game';
  const userId = 'user-1';

  let wsServer: WebSocketServer;
  let socket: AuthenticatedTestSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    capturedMiddleware = null;
    connectionHandler = null;

    const gameState = {
      id: gameId,
      players: [{ id: userId, username: 'Tester', type: 'human', playerNumber: 1 }],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      gameStatus: 'active',
      isRated: false,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    const mockSession = {
      getGameState: jest.fn(() => gameState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => ({
        cancelAllChoicesForPlayer: jest.fn(),
      })),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
      maybePerformAITurn: jest.fn(),
    };

    mockGetOrCreateSession.mockResolvedValue(mockSession);
    mockGetSession.mockReturnValue(mockSession);

    mockGameFindUnique.mockResolvedValue({
      id: gameId,
      allowSpectators: true,
      player1Id: userId,
      player2Id: null,
      player3Id: null,
      player4Id: null,
    });

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);

    socket = {
      id: 'socket-1',
      userId,
      username: 'Tester',
      gameId: undefined,
      join: jest.fn(),
      leave: jest.fn(),
      emit: jest.fn(),
      to: jest.fn(() => ({ emit: jest.fn() })),
      rooms: new Set<string>(),
      on: jest.fn(),
      disconnect: jest.fn(),
      handshake: {
        auth: { token: 'valid-token' },
        query: {},
      },
    } as any;
  });

  it('should handle forced disconnect with custom reason', () => {
    // Setup socket in server
    const io = (wsServer as any).io;
    io.sockets.sockets.set(socket.id, socket);
    (wsServer as any).userSockets.set(userId, socket.id);

    // Terminate with custom reason
    wsServer.terminateUserSessions(userId, 'Server maintenance');

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        code: 'ACCESS_DENIED',
        message: 'Server maintenance',
      })
    );
    expect(socket.disconnect).toHaveBeenCalledWith(true);
  });

  it('should clear all pending reconnections for user on terminate', () => {
    const io = (wsServer as any).io;
    io.sockets.sockets.set(socket.id, socket);
    (wsServer as any).userSockets.set(userId, socket.id);

    // Add multiple pending reconnections
    const timeout1 = setTimeout(() => {}, 30000);
    const timeout2 = setTimeout(() => {}, 30000);

    (wsServer as any).pendingReconnections.set(`game1:${userId}`, {
      timeout: timeout1,
      playerNumber: 1,
      gameId: 'game1',
      userId,
    });
    (wsServer as any).pendingReconnections.set(`game2:${userId}`, {
      timeout: timeout2,
      playerNumber: 2,
      gameId: 'game2',
      userId,
    });

    wsServer.terminateUserSessions(userId);

    expect((wsServer as any).pendingReconnections.has(`game1:${userId}`)).toBe(false);
    expect((wsServer as any).pendingReconnections.has(`game2:${userId}`)).toBe(false);

    clearTimeout(timeout1);
    clearTimeout(timeout2);
  });
});

describe('WebSocket Server - Concurrent Event Handling', () => {
  const gameId = 'concurrent-test-game';
  const userId = 'user-1';

  let wsServer: WebSocketServer;
  let mockSession: any;

  beforeEach(() => {
    jest.clearAllMocks();
    capturedMiddleware = null;
    connectionHandler = null;

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
    };

    const gameState = {
      id: gameId,
      players: [{ id: userId, username: 'Tester', type: 'human', playerNumber: 1 }],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      gameStatus: 'active',
      isRated: false,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession = {
      getGameState: jest.fn(() => gameState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
      maybePerformAITurn: jest.fn(),
    };

    mockGetOrCreateSession.mockResolvedValue(mockSession);
    mockGetSession.mockReturnValue(mockSession);

    mockGameFindUnique.mockResolvedValue({
      id: gameId,
      allowSpectators: true,
      player1Id: userId,
      player2Id: null,
      player3Id: null,
      player4Id: null,
    });

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);
  });

  it('should handle multiple rapid disconnect/reconnect events without race conditions', async () => {
    const sockets: AuthenticatedTestSocket[] = [];

    for (let i = 0; i < 5; i++) {
      sockets.push({
        id: `socket-${i}`,
        userId,
        username: 'Tester',
        gameId: undefined,
        join: jest.fn(),
        leave: jest.fn(),
        emit: jest.fn(),
        to: jest.fn(() => ({ emit: jest.fn() })),
        rooms: new Set<string>(),
        on: jest.fn(),
      } as any);
    }

    // Simulate rapid connect/disconnect by testing the socket tracking
    // Rather than calling internal methods that require full setup,
    // we verify that the server's tracking structures remain consistent
    for (const socket of sockets) {
      if (connectionHandler) {
        connectionHandler(socket);
      }
      // Set gameId to simulate joining
      socket.gameId = gameId;
      socket.rooms.add(gameId);

      // Call disconnect handler
      (wsServer as any).handleDisconnect(socket);
    }

    // Verify only one pending reconnection exists for the user
    // (each disconnect should clear the previous timeout)
    const pendingReconnections = (wsServer as any).pendingReconnections as Map<string, unknown>;
    const userReconnections = Array.from(pendingReconnections.keys()).filter((key) =>
      key.includes(userId)
    );
    expect(userReconnections.length).toBeLessThanOrEqual(1);
  });

  it('should serialize join/leave operations through game lock', async () => {
    const operations: string[] = [];

    // Override withGameLock to track operation order
    const sessionManager = (wsServer as any).sessionManager;
    const originalWithGameLock = sessionManager.withGameLock;
    sessionManager.withGameLock = async (_gameId: string, operation: () => Promise<unknown>) => {
      operations.push('lock-acquired');
      const result = await operation();
      operations.push('lock-released');
      return result;
    };

    const socket: AuthenticatedTestSocket = {
      id: 'socket-1',
      userId,
      username: 'Tester',
      gameId: undefined,
      join: jest.fn(),
      leave: jest.fn(),
      emit: jest.fn(),
      to: jest.fn(() => ({ emit: jest.fn() })),
      rooms: new Set<string>(),
      on: jest.fn(),
    } as any;

    // Join game
    await (wsServer as any).handleJoinGame(socket, gameId);

    expect(operations).toContain('lock-acquired');
    expect(operations).toContain('lock-released');
    expect(operations.indexOf('lock-acquired')).toBeLessThan(operations.indexOf('lock-released'));

    // Restore original
    sessionManager.withGameLock = originalWithGameLock;
  });
});
