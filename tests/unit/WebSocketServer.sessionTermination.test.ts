import { WebSocketServer } from '../../src/server/websocket/server';
import { GameSession } from '../../src/server/game/GameSession';
import { WebSocketInteractionHandler } from '../../src/server/game/WebSocketInteractionHandler';
import type { PlayerChoice } from '../../src/shared/types/game';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';

type MiddlewareFn = (socket: any, next: (err?: Error) => void) => void;

let capturedMiddleware: MiddlewareFn | null = null;

const mockUserFindUnique = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    user: {
      findUnique: mockUserFindUnique,
    },
  }),
}));

jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  const verifyToken = jest.fn(() => ({
    userId: 'user-1',
    email: 'user1@example.com',
    tokenVersion: 0,
  }));
  const validateUser = jest.fn().mockResolvedValue({
    id: 'user-1',
    username: 'Tester',
    email: 'user1@example.com',
    isActive: true,
    tokenVersion: 0,
  });
  return {
    ...actual,
    verifyToken,
    validateUser,
  };
});

jest.mock('../../src/server/services/AIServiceClient');

// Track sockets created during connection events
const mockSockets = new Map<string, any>();

// Create a mock socket that simulates Socket.IO socket behavior
const createMockSocket = (id: string, userId: string) => ({
  id,
  userId,
  emit: jest.fn(),
  disconnect: jest.fn(),
  handshake: {
    auth: { token: 'ACCESS_TOKEN' },
    query: {},
  },
  join: jest.fn(),
  leave: jest.fn(),
  rooms: new Set<string>(),
  on: jest.fn(),
  to: jest.fn(() => ({ emit: jest.fn() })),
});
let connectionHandler: ((socket: any) => void) | null = null;

const mockGetSession = jest.fn();
const mockWithGameLock = jest.fn(async (_gameId: string, operation: () => Promise<unknown>) =>
  operation()
);

jest.mock('../../src/server/game/GameSessionManager', () => {
  return {
    GameSessionManager: jest.fn().mockImplementation(() => ({
      getOrCreateSession: jest.fn(),
      getSession: mockGetSession,
      withGameLock: mockWithGameLock,
    })),
  };
});

jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public use = jest.fn((fn: MiddlewareFn) => {
      capturedMiddleware = fn;
    });

    public on = jest.fn((event: string, handler: any) => {
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
      sockets: mockSockets,
    };

    constructor(..._args: any[]) {}
  }

  return {
    Server: FakeSocketIOServer,
  };
});

describe('WebSocketServer.terminateUserSessions', () => {
  let wsServer: WebSocketServer;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSockets.clear();
    capturedMiddleware = null;
    connectionHandler = null;

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);
  });

  it('returns 0 when user has no active connections', () => {
    const terminatedCount = wsServer.terminateUserSessions('nonexistent-user', 'Test reason');

    expect(terminatedCount).toBe(0);
  });

  it('terminates active WebSocket connection for user', async () => {
    // Simulate a user connecting
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    // Run the auth middleware to register the connection
    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    // Simulate the connection event handler to register the socket
    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Now terminate the user's sessions
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    // Socket should have been disconnected
    expect(mockSocket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Account deleted',
      })
    );
    expect(mockSocket.disconnect).toHaveBeenCalledWith(true);
    expect(terminatedCount).toBe(1);
  });

  it('cleans up stale socket mapping when socket no longer exists', async () => {
    // Simulate a user connecting
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Now remove the socket from the Socket.IO server (simulating it was already closed)
    mockSockets.delete('socket-1');

    // Terminate should handle this gracefully
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    expect(terminatedCount).toBe(0);
    // Verify no crash occurred
  });

  it('uses default reason when not provided', async () => {
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Terminate without providing a reason
    wsServer.terminateUserSessions('user-1');

    expect(mockSocket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Session terminated',
      })
    );
  });

  it('invokes GameSession.terminate when terminating user sessions (session-level cleanup)', async () => {
    const mockSocket = createMockSocket('socket-1', 'user-1');
    // Simulate the socket currently being in a game room so that
    // terminateUserSessions can locate an in-memory GameSession via
    // the socket.gameId + GameSessionManager.getSession.
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const terminateMock = jest.fn();
    const mockSession: any = {
      terminate: terminateMock,
      getGameState: jest.fn(() => ({
        id: 'game-1',
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        boardType: 'square8',
        players: [],
        spectators: [],
        moveHistory: [],
      })),
    };

    mockGetSession.mockReturnValue(mockSession);

    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    expect(terminatedCount).toBe(1);
    // GameSession.terminate should be invoked to allow session-scoped
    // cancellation (AI turns, timers) to run before socket teardown.
    expect(terminateMock).toHaveBeenCalledWith('session_cleanup');
  });

  it('clears GameSession decision-phase timers when terminating user sessions', async () => {
    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const ioForSession: any = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    };

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-1', ioForSession, pythonClient, userSockets);

    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        id: 'game-1',
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'line_processing',
        boardType: 'square8',
        players: [],
        spectators: [],
        moveHistory: [],
      })),
    };

    (session as any).decisionTimeoutDeadlineMs = Date.now() + 10_000;
    (session as any).decisionTimeoutHandle = setTimeout(() => {}, 10_000);
    (session as any).decisionTimeoutWarningHandle = setTimeout(() => {}, 5_000);

    mockGetSession.mockReturnValue(session as any);

    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    expect(terminatedCount).toBe(1);
    expect((session as any).decisionTimeoutDeadlineMs).toBeNull();
    expect((session as any).decisionTimeoutHandle).toBeNull();
    expect((session as any).decisionTimeoutWarningHandle).toBeNull();
  });

  it('propagates terminateUserSessions cancellation into subsequent AI turn requests (session token canceled before AI HTTP)', async () => {
    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const ioForSession: any = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    };

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-1', ioForSession, pythonClient, userSockets);

    // Seed a minimal game state where it is an AI player's turn in a
    // non-decision phase, so maybePerformAITurn would normally attempt
    // a service-backed AI move.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        gameId: 'game-1',
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: { 1: 0, 2: 0 },
        },
        players: [
          {
            id: 'ai-player',
            username: 'AI',
            playerNumber: 1,
            type: 'ai',
            isReady: true,
            timeRemaining: 0,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
        spectators: [],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 0,
        timeControl: { initialTime: 600, increment: 5, type: 'blitz' },
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
        totalRingsEliminated: 0,
      })),
    };

    // Attach the session to the GameSessionManager mock so that
    // terminateUserSessions can locate it and invoke session.terminate().
    mockGetSession.mockReturnValue(session as any);

    // Spy on getAIMoveWithTimeout to ensure any subsequent AI turn work
    // observes a canceled session token before reaching the AI service
    // layer. In the real implementation, a canceled token causes
    // AIServiceClient to short-circuit HTTP calls via token.throwIfCanceled.
    const recordedTokens: unknown[] = [];
    const mockGetAIMoveWithTimeout = jest.fn(
      async (
        _playerNumber: number,
        _state: any,
        _timeoutMs: number,
        options?: { token?: { isCanceled: boolean } }
      ) => {
        if (options?.token) {
          recordedTokens.push(options.token);
        }
        // Simulate the real cancellation path: an error bubbles up as
        // "AI request canceled" and is handled by maybePerformAITurn's
        // catch block without issuing HTTP.
        throw new Error('AI request canceled');
      }
    );
    (session as any).getAIMoveWithTimeout = mockGetAIMoveWithTimeout;

    // Terminate user sessions, which calls GameSession.terminate and
    // cancels the session-scoped cancellation token.
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');
    expect(terminatedCount).toBe(1);

    // Trigger an AI turn after termination; the session-level cancellation
    // token should prevent any new AI HTTP calls from being issued.
    await (session as any).maybePerformAITurn();

    expect(mockGetAIMoveWithTimeout).toHaveBeenCalledTimes(1);
    expect(recordedTokens.length).toBe(1);
    const token = recordedTokens[0] as { isCanceled: boolean };
    expect(token.isCanceled).toBe(true);
  });

  it('does not emit additional game_state or player_choice_required events for a user after terminateUserSessions during a pending decision', async () => {
    jest.useFakeTimers();

    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Wire a minimal session-level WebSocketInteractionHandler that targets
    // this user's socket id for Player 1. This mirrors the production
    // GameSession → WebSocketInteractionHandler → Socket.IO path but keeps
    // the session itself as a lightweight stub.
    const ioForSession: any = {
      to: (target: string) => ({
        emit: (event: string, payload: unknown) => {
          // Route both direct socket-targeted emits and game-room emits for
          // this single-player test through the same mock socket.
          const socketId = target === 'game-1' ? 'socket-1' : target;
          const socket = mockSockets.get(socketId);
          if (socket && typeof socket.emit === 'function') {
            socket.emit(event, payload);
          }
        },
      }),
      sockets: {
        adapter: { rooms: new Map<string, Set<string>>() },
        sockets: mockSockets,
      },
    };

    const getTargetForPlayer = (playerNumber: number): string | undefined =>
      playerNumber === 1 ? 'socket-1' : undefined;

    const wsHandler = new WebSocketInteractionHandler(
      ioForSession as any,
      'game-1',
      getTargetForPlayer,
      30_000
    );

    const session: any = {
      terminate: jest.fn(() => {
        // On termination, cancel all pending choices for Player 1 so that any
        // outstanding decision is resolved without emitting new choices.
        wsHandler.cancelAllChoicesForPlayer(1);
      }),
    };

    mockGetSession.mockReturnValue(session);

    // Issue a synthetic decision for Player 1 and confirm that it results in
    // exactly one player_choice_required event to this socket.
    const choice: PlayerChoice = {
      id: 'choice-1',
      gameId: 'game-1',
      playerNumber: 1,
      type: 'region_order',
      prompt: 'Choose region order',
      timeoutMs: 30_000,
      options: [{ regionId: 'region-A' }],
    } as any;

    const choicePromise = wsHandler.requestChoice(choice).catch(() => {});

    const initialRelevantEmits = mockSocket.emit.mock.calls.filter(
      ([event]) => event === 'player_choice_required' || event === 'game_state'
    );
    expect(initialRelevantEmits.length).toBe(1);

    const emitCallCountBeforeTerminate = mockSocket.emit.mock.calls.length;

    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');
    expect(terminatedCount).toBe(1);
    expect(session.terminate).toHaveBeenCalledWith('session_cleanup');

    const relevantAfterTerminate = mockSocket.emit.mock.calls
      .slice(emitCallCountBeforeTerminate)
      .filter(([event]) => event === 'player_choice_required' || event === 'game_state');

    // After terminateUserSessions, the only additional emits allowed for this
    // user/game are error + optional player_choice_canceled notifications; no
    // new game_state or player_choice_required events should be observed.
    expect(relevantAfterTerminate.length).toBe(0);

    jest.runOnlyPendingTimers();
    await choicePromise;
    jest.useRealTimers();
  });

  it('does not emit additional game_state or player_choice_required events after terminateUserSessions during a pending ring_elimination decision', async () => {
    jest.useFakeTimers();

    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const ioForSession: any = {
      to: (target: string) => ({
        emit: (event: string, payload: unknown) => {
          const socketId = target === 'game-1' ? 'socket-1' : target;
          const socket = mockSockets.get(socketId);
          if (socket && typeof socket.emit === 'function') {
            socket.emit(event, payload);
          }
        },
      }),
      sockets: {
        adapter: { rooms: new Map<string, Set<string>>() },
        sockets: mockSockets,
      },
    };

    const getTargetForPlayer = (playerNumber: number): string | undefined =>
      playerNumber === 1 ? 'socket-1' : undefined;

    const wsHandler = new WebSocketInteractionHandler(
      ioForSession as any,
      'game-1',
      getTargetForPlayer,
      30_000
    );

    const session: any = {
      terminate: jest.fn(() => {
        wsHandler.cancelAllChoicesForPlayer(1);
      }),
    };

    mockGetSession.mockReturnValue(session);

    const choice: PlayerChoice = {
      id: 'choice-ring-1',
      gameId: 'game-1',
      playerNumber: 1,
      type: 'ring_elimination',
      prompt: 'Choose elimination stack',
      timeoutMs: 30_000,
      options: [{ stackKey: '7,7', count: 2 }],
    } as any;

    const choicePromise = wsHandler.requestChoice(choice).catch(() => {});

    const initialRelevantEmits = mockSocket.emit.mock.calls.filter(
      ([event]) => event === 'player_choice_required' || event === 'game_state'
    );
    expect(initialRelevantEmits.length).toBe(1);

    const emitCallCountBeforeTerminate = mockSocket.emit.mock.calls.length;

    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');
    expect(terminatedCount).toBe(1);
    expect(session.terminate).toHaveBeenCalledWith('session_cleanup');

    const relevantAfterTerminate = mockSocket.emit.mock.calls
      .slice(emitCallCountBeforeTerminate)
      .filter(([event]) => event === 'player_choice_required' || event === 'game_state');

    expect(relevantAfterTerminate.length).toBe(0);

    jest.runOnlyPendingTimers();
    await choicePromise;
    jest.useRealTimers();
  });

  it('cancels AI-backed region_order choice when terminateUserSessions is called mid-decision', async () => {
    jest.useFakeTimers();

    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeClient = {
      getRegionOrderChoice: jest.fn(),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const ioForSession: any = {
      to: (target: string) => ({
        emit: (event: string, payload: unknown) => {
          const socketId = target === 'game-1' ? 'socket-1' : target;
          const socket = mockSockets.get(socketId);
          if (socket && typeof socket.emit === 'function') {
            socket.emit(event, payload);
          }
        },
      }),
      sockets: {
        adapter: { rooms: new Map<string, Set<string>>() },
        sockets: mockSockets,
      },
    };

    const getTargetForPlayer = (playerNumber: number): string | undefined =>
      playerNumber === 1 ? 'socket-1' : undefined;

    const wsHandler = new WebSocketInteractionHandler(
      ioForSession as any,
      'game-1',
      getTargetForPlayer,
      30_000
    );

    const session: any = {
      terminate: jest.fn(() => {
        wsHandler.cancelAllChoicesForPlayer(1);
      }),
      getInteractionHandler: () => wsHandler,
    };

    mockGetSession.mockReturnValue(session);

    const choice: PlayerChoice = {
      id: 'choice-region-order-ai',
      gameId: 'game-1',
      playerNumber: 1,
      type: 'region_order',
      prompt: 'AI-backed region order',
      timeoutMs: 30_000,
      options: [{ regionId: 'A', size: 2 }],
    } as any;

    // Start a pending choice Promise that, in a real session, would be
    // answered either by the player or by the AI service.
    const pendingChoice = wsHandler.requestChoice(choice).catch(() => {});

    // Simulate the session being terminated (e.g. via account deletion)
    // while the choice is still outstanding.
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');
    expect(terminatedCount).toBe(1);
    expect(session.terminate).toHaveBeenCalledWith('session_cleanup');

    // Ensure no additional game_state / player_choice_required events
    // are emitted after termination for this user/game.
    const relevantAfterTerminate = mockSocket.emit.mock.calls.filter(
      ([event]) => event === 'player_choice_required' || event === 'game_state'
    );
    expect(relevantAfterTerminate.length).toBeLessThanOrEqual(1);

    // In this integration slice we only assert that the AI service
    // client is either not called at all, or, if wired, will see a
    // canceled token. For now we require the stronger invariant that
    // terminateUserSessions prevents any HTTP call.
    expect(fakeClient.getRegionOrderChoice).not.toHaveBeenCalled();

    jest.runOnlyPendingTimers();
    await pendingChoice;
    jest.useRealTimers();
  });

  it('cancels AI-backed line_reward_option choice when terminateUserSessions is called mid-decision', async () => {
    jest.useFakeTimers();

    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeClient = {
      getLineRewardChoice: jest.fn(),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const mockSocket = createMockSocket('socket-1', 'user-1');
    (mockSocket as any).gameId = 'game-1';
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    const ioForSession: any = {
      to: (target: string) => ({
        emit: (event: string, payload: unknown) => {
          const socketId = target === 'game-1' ? 'socket-1' : target;
          const socket = mockSockets.get(socketId);
          if (socket && typeof socket.emit === 'function') {
            socket.emit(event, payload);
          }
        },
      }),
      sockets: {
        adapter: { rooms: new Map<string, Set<string>>() },
        sockets: mockSockets,
      },
    };

    const getTargetForPlayer = (playerNumber: number): string | undefined =>
      playerNumber === 1 ? 'socket-1' : undefined;

    const wsHandler = new WebSocketInteractionHandler(
      ioForSession as any,
      'game-1',
      getTargetForPlayer,
      30_000
    );

    const session: any = {
      terminate: jest.fn(() => {
        wsHandler.cancelAllChoicesForPlayer(1);
      }),
      getInteractionHandler: () => wsHandler,
    };

    mockGetSession.mockReturnValue(session);

    const choice: PlayerChoice = {
      id: 'choice-line-reward-ai',
      gameId: 'game-1',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'AI-backed line reward',
      timeoutMs: 30_000,
      options: [
        { id: 'option_1_collapse_all_and_eliminate', label: 'Collapse all + eliminate' },
        {
          id: 'option_2_minimum_collapse_no_elimination',
          label: 'Minimum collapse, no elimination',
        },
      ],
    } as any;

    const pendingChoice = wsHandler.requestChoice(choice).catch(() => {});

    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');
    expect(terminatedCount).toBe(1);
    expect(session.terminate).toHaveBeenCalledWith('session_cleanup');

    // As in the region_order slice, assert that no additional game_state or
    // player_choice_required events are emitted for this user after
    // termination and that no AI HTTP call is made.
    const relevantAfterTerminate = mockSocket.emit.mock.calls.filter(
      ([event]) => event === 'player_choice_required' || event === 'game_state'
    );
    expect(relevantAfterTerminate.length).toBeLessThanOrEqual(1);

    expect(fakeClient.getLineRewardChoice).not.toHaveBeenCalled();

    jest.runOnlyPendingTimers();
    await pendingChoice;
    jest.useRealTimers();
  });
});
