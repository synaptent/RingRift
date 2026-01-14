import { WebSocketServer } from '../../src/server/websocket/server';
import { getMetricsService } from '../../src/server/services/MetricsService';
import { withTimeControl } from '../helpers/TimeController';

type AuthenticatedTestSocket = any;

const mockGameFindUnique = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockGameFindUnique,
    },
  }),
}));

// Provide a lightweight mock for GameSessionManager so we can control
// getOrCreateSession/getSession and the returned GameSession shape.
const mockGetOrCreateSession = jest.fn();
const mockGetSession = jest.fn();

jest.mock('../../src/server/game/GameSessionManager', () => {
  return {
    GameSessionManager: jest.fn().mockImplementation(() => ({
      getOrCreateSession: mockGetOrCreateSession,
      getSession: mockGetSession,
      // For tests, run the operation immediately without a distributed lock.
      withGameLock: async (_gameId: string, operation: () => Promise<unknown>) => operation(),
    })),
  };
});

// Mock MetricsService to observe reconnection metrics without touching the
// real Prometheus registry. This keeps the tests focused on behaviour rather
// than prom-client internals.
jest.mock('../../src/server/services/MetricsService', () => {
  const metrics = {
    incWebSocketConnections: jest.fn(),
    decWebSocketConnections: jest.fn(),
    recordWebsocketReconnection: jest.fn(),
    recordMoveRejected: jest.fn(),
  };

  return {
    __esModule: true,
    getMetricsService: () => metrics,
  };
});

// Reuse the minimal Socket.IO server mock pattern from existing tests so that
// constructing WebSocketServer does not spin up a real server.
jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public use = jest.fn();
    public on = jest.fn();
    public to = jest.fn(() => ({ emit: jest.fn() }));

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

describe('WebSocketServer PlayerConnectionState diagnostics', () => {
  const gameId = 'connection-test-game';
  const userId = 'user-1';
  const playerNumber = 1;

  let wsServer: WebSocketServer;
  let socket: AuthenticatedTestSocket;
  let mockSession: any;

  beforeEach(() => {
    jest.resetModules();

    // Minimal GameState stub – we only care about players[] and spectators[]
    // Include minimal board structure for serializeBoardState() compatibility
    const gameState: any = {
      id: gameId,
      players: [
        {
          id: userId,
          username: 'Tester',
          type: 'human',
          playerNumber,
        },
      ],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: playerNumber,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
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
    } as any;
  });

  it('marks player as connected on successful join', async () => {
    await (wsServer as any).handleJoinGame(socket, gameId);

    const state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state).toBeDefined();
    if (!state || state.kind !== 'connected') {
      throw new Error('Expected connected PlayerConnectionState after join');
    }
    expect(state.gameId).toBe(gameId);
    expect(state.userId).toBe(userId);
    expect(state.playerNumber).toBe(playerNumber);
    expect(typeof state.connectedAt).toBe('number');
    expect(typeof state.lastSeenAt).toBe('number');
  });

  it('marks player as disconnected_pending_reconnect on disconnect with window', async () => {
    await (wsServer as any).handleJoinGame(socket, gameId);

    (wsServer as any).handleDisconnect(socket);

    const state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state).toBeDefined();
    if (!state || state.kind !== 'disconnected_pending_reconnect') {
      throw new Error(
        'Expected disconnected_pending_reconnect PlayerConnectionState after disconnect'
      );
    }
    expect(state.gameId).toBe(gameId);
    expect(state.userId).toBe(userId);
    expect(state.playerNumber).toBe(playerNumber);
    expect(typeof state.disconnectedAt).toBe('number');
    expect(typeof state.deadlineAt).toBe('number');
  });

  it('restores connected state on successful reconnection before timeout', async () => {
    // Initial join marks the player as connected
    await (wsServer as any).handleJoinGame(socket, gameId);

    const initial = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(initial).toBeDefined();
    expect(initial!.kind).toBe('connected');

    // Simulate a disconnect which starts the reconnection window
    (wsServer as any).handleDisconnect(socket);

    const pending = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(pending).toBeDefined();
    expect(pending!.kind).toBe('disconnected_pending_reconnect');

    // Simulate the user reconnecting on a new socket before the window expires
    const roomEmitter = { emit: jest.fn() };
    const toMock = jest.fn(() => roomEmitter);

    const reconnectSocket: AuthenticatedTestSocket = {
      ...socket,
      id: 'socket-2',
      join: jest.fn(),
      to: toMock,
      emit: jest.fn(),
    };

    await (wsServer as any).handleJoinGame(reconnectSocket, gameId);

    const after = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(after).toBeDefined();
    expect(after!.kind).toBe('connected');
    expect(after!.gameId).toBe(gameId);
    expect(after!.userId).toBe(userId);

    // Reconnection path should emit a player_reconnected event to the room
    expect(toMock).toHaveBeenCalledWith(gameId);
    expect(roomEmitter.emit).toHaveBeenCalledWith(
      'player_reconnected',
      expect.objectContaining({
        type: 'player_reconnected',
        data: expect.objectContaining({
          gameId,
          player: expect.objectContaining({
            id: userId,
          }),
        }),
      })
    );

    // Critically, reconnecting within the window must NOT trigger any
    // abandonment semantics or persistence side-effects.
    expect(mockSession.handleAbandonmentForDisconnectedPlayer).not.toHaveBeenCalled();

    // Successful reconnection within the window must be reflected in the
    // reconnection metrics as a `result="success"` increment. This is part
    // of the connection lifecycle observability described in P18.3-1
    // (§2.4 / §4.3).
    const metrics = getMetricsService() as any;
    expect(metrics.recordWebsocketReconnection).toHaveBeenCalledWith('success');
  });

  it('marks player as disconnected_expired when reconnection window expires', async () => {
    await (wsServer as any).handleJoinGame(socket, gameId);

    (wsServer as any).handleDisconnect(socket);

    const before = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(before).toBeDefined();
    expect(before!.kind).toBe('disconnected_pending_reconnect');

    (wsServer as any).handleReconnectionTimeout(gameId, userId, playerNumber);

    const after = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(after).toBeDefined();
    if (!after || after.kind !== 'disconnected_expired') {
      throw new Error('Expected disconnected_expired PlayerConnectionState after timeout expiry');
    }
    expect(after.gameId).toBe(gameId);
    expect(after.userId).toBe(userId);
    expect(after.playerNumber).toBe(playerNumber);
    expect(typeof after.disconnectedAt).toBe('number');
    expect(typeof (after as any).expiredAt).toBe('number');

    // Ensure stale choices would be cleared via interaction handler
    const interactionHandler = mockSession.getInteractionHandler();
    expect(interactionHandler.cancelAllChoicesForPlayer).toHaveBeenCalledWith(playerNumber);
  });

  it('drives reconnection timeout via timers and applies abandonment semantics', async () => {
    await withTimeControl(async (tc) => {
      const httpServerStub: any = {};
      wsServer = new WebSocketServer(httpServerStub as any);

      // Minimal rated 1v1 state so abandonment can award a win.
      const ratedState: any = {
        id: gameId,
        players: [
          {
            id: userId,
            username: 'P1',
            type: 'human',
            playerNumber,
          },
          {
            id: 'user-2',
            username: 'P2',
            type: 'human',
            playerNumber: 2,
          },
        ],
        spectators: [],
        currentPhase: 'ring_placement',
        currentPlayer: playerNumber,
        gameStatus: 'active',
        isRated: true,
        board: {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
        },
      };

      mockSession.getGameState.mockReturnValue(ratedState);

      // Join marks the player as connected and seeds connection state.
      await (wsServer as any).handleJoinGame(socket, gameId);

      const beforeDisconnect = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
      expect(beforeDisconnect).toBeDefined();
      expect(beforeDisconnect!.kind).toBe('connected');

      // Seed connection state for the opponent so that when the window
      // expires, anyOpponentStillAlive resolves to true and a rated win
      // can be awarded.
      const opponentId = 'user-2';
      const now = Date.now();
      (wsServer as any).playerConnectionStates.set(`${gameId}:${opponentId}`, {
        kind: 'connected',
        gameId,
        userId: opponentId,
        playerNumber: 2,
        connectedAt: now,
        lastSeenAt: now,
      });

      // Simulate disconnect; this schedules a reconnection timeout.
      (wsServer as any).handleDisconnect(socket);

      const pending = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
      expect(pending).toBeDefined();
      expect(pending!.kind).toBe('disconnected_pending_reconnect');

      // Advance virtual time past the 30s reconnection window so the
      // internal timeout fires and handleReconnectionTimeout is invoked.
      await tc.advanceTime(30_000);

      const expired = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
      expect(expired).toBeDefined();
      expect(expired!.kind).toBe('disconnected_expired');

      const interactionHandler = mockSession.getInteractionHandler();
      expect(interactionHandler.cancelAllChoicesForPlayer).toHaveBeenCalledWith(playerNumber);

      // Because the game is rated and another human remains "alive" in
      // playerConnectionStates, abandonment should be applied with
      // awardWinToOpponent=true.
      expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(
        playerNumber,
        true
      );

      // When the reconnection window expires via the scheduled timer, the
      // WebSocket server must record a `result="timeout"` reconnection
      // metric so operators can distinguish clean reconnects from
      // abandonment timeouts.
      const metrics = getMetricsService() as any;
      expect(metrics.recordWebsocketReconnection).toHaveBeenCalledWith('timeout');
    });
  });

  it('does not start reconnection window for spectators and clears diagnostics entry on disconnect', async () => {
    // Treat the socket user as a pure spectator: not present in players[],
    // but included in the spectators list.
    const spectatorGameState: any = {
      id: gameId,
      players: [],
      spectators: [userId],
      currentPhase: 'ring_placement',
      currentPlayer: playerNumber,
      gameStatus: 'active',
      isRated: false,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession.getGameState.mockReturnValue(spectatorGameState);

    // At the DB layer the user is not a player but spectators are allowed.
    mockGameFindUnique.mockResolvedValueOnce({
      id: gameId,
      allowSpectators: true,
      player1Id: null,
      player2Id: null,
      player3Id: null,
      player4Id: null,
    });

    await (wsServer as any).handleJoinGame(socket, gameId);

    // Joining as a spectator should still record a connected state, but with
    // no playerNumber.
    const before = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(before).toBeDefined();
    expect(before!.kind).toBe('connected');
    expect(before!.playerNumber).toBeUndefined();

    (wsServer as any).handleDisconnect(socket);

    // Spectator disconnect should not schedule a reconnection timeout entry.
    const pendingReconnections = (wsServer as any).pendingReconnections as Map<string, unknown>;
    expect(pendingReconnections.size).toBe(0);

    // And the diagnostics entry for this spectator should be removed.
    const after = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(after).toBeUndefined();
  });

  it('invokes abandonment handler with awardWinToOpponent=false when no opponents remain or game is unrated', () => {
    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);

    const ratedState: any = {
      id: gameId,
      players: [
        {
          id: userId,
          username: 'P1',
          type: 'human',
          playerNumber,
        },
      ],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: playerNumber,
      gameStatus: 'active',
      isRated: false,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession.getGameState.mockReturnValue(ratedState);

    // No other human players are tracked in playerConnectionStates, so
    // anyOpponentStillAlive resolves to false and awardWinToOpponent=false.
    (wsServer as any).handleReconnectionTimeout(gameId, userId, playerNumber);

    expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(
      playerNumber,
      false
    );
  });

  it('invokes abandonment handler with awardWinToOpponent=true when rated and an opponent remains connected', () => {
    const ratedGameId = 'rated-abandonment-game';
    const userId1 = 'user-1';
    const userId2 = 'user-2';

    const ratedState: any = {
      id: ratedGameId,
      players: [
        {
          id: userId1,
          username: 'P1',
          type: 'human',
          playerNumber: 1,
        },
        {
          id: userId2,
          username: 'P2',
          type: 'human',
          playerNumber: 2,
        },
      ],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      gameStatus: 'active',
      isRated: true,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession.getGameState.mockReturnValue(ratedState);

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);

    // Manually seed connection state to indicate that user-2 is still
    // connected at the time user-1's reconnect window expires.
    (wsServer as any).playerConnectionStates.set(`${ratedGameId}:${userId2}`, {
      kind: 'connected',
      gameId: ratedGameId,
      userId: userId2,
      playerNumber: 2,
      connectedAt: Date.now(),
      lastSeenAt: Date.now(),
    });

    (wsServer as any).handleReconnectionTimeout(ratedGameId, userId1, 1);

    expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(1, true);
  });

  it('invokes abandonment handler with awardWinToOpponent=false when all human players have expired reconnect windows in a rated game', () => {
    const ratedGameId = 'rated-all-expired-game';
    const userId1 = 'user-1';
    const userId2 = 'user-2';

    const ratedState: any = {
      id: ratedGameId,
      players: [
        {
          id: userId1,
          username: 'P1',
          type: 'human',
          playerNumber: 1,
        },
        {
          id: userId2,
          username: 'P2',
          type: 'human',
          playerNumber: 2,
        },
      ],
      spectators: [],
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      gameStatus: 'active',
      isRated: true,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
      },
    };

    mockSession.getGameState.mockReturnValue(ratedState);

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);

    const now = Date.now();

    // Seed connection state to indicate that *both* humans have already had
    // their reconnect windows expire when user-1's timeout handler runs. In
    // this case the server must treat the game as an abandonment draw (no
    // winner), so awardWinToOpponent=false is propagated to the session.
    (wsServer as any).playerConnectionStates.set(`${ratedGameId}:${userId1}`, {
      kind: 'disconnected_expired',
      gameId: ratedGameId,
      userId: userId1,
      playerNumber: 1,
      disconnectedAt: now - 10_000,
      expiredAt: now - 5_000,
    });
    (wsServer as any).playerConnectionStates.set(`${ratedGameId}:${userId2}`, {
      kind: 'disconnected_expired',
      gameId: ratedGameId,
      userId: userId2,
      playerNumber: 2,
      disconnectedAt: now - 10_000,
      expiredAt: now - 5_000,
    });

    (wsServer as any).handleReconnectionTimeout(ratedGameId, userId1, 1);

    expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(1, false);
  });
});
