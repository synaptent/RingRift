import { WebSocketServer } from '../../src/server/websocket/server';

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
    };

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
    };

    mockSession = {
      getGameState: jest.fn(() => gameState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
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
    expect(state!.kind).toBe('connected');
    expect(state!.gameId).toBe(gameId);
    expect(state!.userId).toBe(userId);
    expect(state!.playerNumber).toBe(playerNumber);
    expect(typeof state!.connectedAt).toBe('number');
    expect(typeof state!.lastSeenAt).toBe('number');
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
});
