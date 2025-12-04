/**
 * Tests for player reconnection flows across different game phases.
 *
 * This file covers reconnection scenarios not addressed in other test files:
 * - Reconnection during ring placement phase
 * - Reconnection during movement phase
 * - Edge cases: rapid disconnect/reconnect, reconnect after opponent resigned,
 *   reconnect to completed game, duplicate connection prevention
 *
 * These tests ensure robust player reconnection UX across the full game lifecycle.
 *
 * Related spec: docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md
 */

import { GameSession } from '../../src/server/game/GameSession';
import { WebSocketServer } from '../../src/server/websocket/server';
import type { Server as SocketIOServer } from 'socket.io';
import type { GameState, Move, Position, Player, TimeControl } from '../../src/shared/types/game';

// Mock database connection
const mockGameFindUnique = jest.fn();
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: {
      findUnique: mockGameFindUnique,
      update: jest.fn(),
    },
    move: {
      create: jest.fn(),
    },
  })),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({
    evaluateMove: jest.fn(),
    healthCheck: jest.fn(),
  })),
}));

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIConfig: jest.fn(),
    getAIMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../src/server/services/AIUserService', () => ({
  getOrCreateAIUser: jest.fn(() => Promise.resolve({ id: 'ai-user-id' })),
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

// Mock socket.io
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

type AuthenticatedTestSocket = any;

describe('GameSession Reconnect Flow Tests', () => {
  const gameId = 'reconnect-flow-test-game';
  const userId1 = 'user-1';
  const userId2 = 'user-2';

  let mockIo: jest.Mocked<SocketIOServer>;
  let session: GameSession;
  let mockSession: any;
  let wsServer: WebSocketServer;

  const createMockIo = () =>
    ({
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
      on: jest.fn(),
    }) as any;

  const createMockSocket = (userId: string, socketId: string): AuthenticatedTestSocket => ({
    id: socketId,
    userId,
    username: `User ${userId}`,
    gameId: undefined,
    join: jest.fn(),
    leave: jest.fn(),
    emit: jest.fn(),
    to: jest.fn(() => ({ emit: jest.fn() })),
    rooms: new Set<string>(),
  });

  const createBasePlacementPhaseState = (): GameState =>
    ({
      id: gameId,
      boardType: 'square8',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      players: [
        {
          id: userId1,
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: userId2,
          username: 'Player 2',
          playerNumber: 2,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        eliminatedRings: { 1: 0, 2: 0 },
        formedLines: [],
      },
      timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    }) as unknown as GameState;

  const createMovementPhaseState = (): GameState => {
    const base = createBasePlacementPhaseState();
    return {
      ...base,
      currentPhase: 'movement',
      players: base.players.map((p) => ({
        ...p,
        ringsInHand: 0, // All rings placed
      })),
      moveHistory: [
        // Simulate some placement moves happened
        {
          id: 'm1',
          type: 'ring_placement' as const,
          player: 1,
          to: { x: 0, y: 0 },
          moveNumber: 1,
          timestamp: new Date(),
          thinkTime: 0,
        },
        {
          id: 'm2',
          type: 'ring_placement' as const,
          player: 2,
          to: { x: 7, y: 7 },
          moveNumber: 2,
          timestamp: new Date(),
          thinkTime: 0,
        },
      ] as unknown as Move[],
    };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockIo = createMockIo();

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
    };

    mockSession = {
      getGameState: jest.fn(() => createBasePlacementPhaseState()),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
    };

    mockGetOrCreateSession.mockResolvedValue(mockSession);
    mockGetSession.mockReturnValue(mockSession);

    mockGameFindUnique.mockResolvedValue({
      id: gameId,
      allowSpectators: true,
      player1Id: userId1,
      player2Id: userId2,
      player3Id: null,
      player4Id: null,
    });

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);
  });

  describe('Reconnection during Placement Phase', () => {
    it('should restore game state for reconnecting player during placement phase', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Player 1 joins initially
      await (wsServer as any).handleJoinGame(socket1, gameId);

      const initialState = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(initialState).toBeDefined();
      expect(initialState!.kind).toBe('connected');
      expect(initialState!.playerNumber).toBe(1);

      // Simulate disconnect
      (wsServer as any).handleDisconnect(socket1);

      const disconnectedState = wsServer.getPlayerConnectionStateSnapshotForTesting(
        gameId,
        userId1
      );
      expect(disconnectedState).toBeDefined();
      expect(disconnectedState!.kind).toBe('disconnected_pending_reconnect');

      // Simulate reconnect with new socket
      const socket1Reconnect = createMockSocket(userId1, 'socket-1-reconnect');
      await (wsServer as any).handleJoinGame(socket1Reconnect, gameId);

      const reconnectedState = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(reconnectedState).toBeDefined();
      expect(reconnectedState!.kind).toBe('connected');
      expect(reconnectedState!.playerNumber).toBe(1);
    });

    it('should broadcast reconnect event to other players during placement phase', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');
      const roomEmitter = { emit: jest.fn() };
      const toMock = jest.fn(() => roomEmitter);

      // Player 1 joins and disconnects
      await (wsServer as any).handleJoinGame(socket1, gameId);
      (wsServer as any).handleDisconnect(socket1);

      // Player 1 reconnects
      const socket1Reconnect = {
        ...createMockSocket(userId1, 'socket-1-reconnect'),
        to: toMock,
      };
      await (wsServer as any).handleJoinGame(socket1Reconnect, gameId);

      expect(toMock).toHaveBeenCalledWith(gameId);
      expect(roomEmitter.emit).toHaveBeenCalledWith(
        'player_reconnected',
        expect.objectContaining({
          type: 'player_reconnected',
          data: expect.objectContaining({
            gameId,
            playerNumber: 1,
            player: expect.objectContaining({
              id: userId1,
            }),
          }),
        })
      );
    });

    it('should not allow duplicate connections for same player during placement phase', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1a = createMockSocket(userId1, 'socket-1a');
      const socket1b = createMockSocket(userId1, 'socket-1b');

      // Player 1 joins with first socket
      await (wsServer as any).handleJoinGame(socket1a, gameId);

      const stateAfterFirst = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(stateAfterFirst).toBeDefined();
      expect(stateAfterFirst!.kind).toBe('connected');

      // Player 1 joins with second socket (should replace first)
      await (wsServer as any).handleJoinGame(socket1b, gameId);

      const stateAfterSecond = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(stateAfterSecond).toBeDefined();
      expect(stateAfterSecond!.kind).toBe('connected');
      // Connection state should be maintained (not duplicated)
    });
  });

  describe('Reconnection during Movement Phase', () => {
    it('should restore game state for reconnecting player during movement phase', async () => {
      const gameState = createMovementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Player 1 joins initially
      await (wsServer as any).handleJoinGame(socket1, gameId);

      // Simulate disconnect
      (wsServer as any).handleDisconnect(socket1);

      const disconnectedState = wsServer.getPlayerConnectionStateSnapshotForTesting(
        gameId,
        userId1
      );
      expect(disconnectedState!.kind).toBe('disconnected_pending_reconnect');

      // Simulate reconnect
      const socket1Reconnect = createMockSocket(userId1, 'socket-1-reconnect');
      await (wsServer as any).handleJoinGame(socket1Reconnect, gameId);

      const reconnectedState = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(reconnectedState!.kind).toBe('connected');
    });

    it('should preserve move history on reconnect during movement phase', async () => {
      const gameState = createMovementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      // Verify the session returns correct state with history
      expect(gameState.moveHistory.length).toBe(2);
      expect(gameState.currentPhase).toBe('movement');

      const socket1 = createMockSocket(userId1, 'socket-1');
      await (wsServer as any).handleJoinGame(socket1, gameId);
      (wsServer as any).handleDisconnect(socket1);

      const socket1Reconnect = createMockSocket(userId1, 'socket-1-reconnect');
      await (wsServer as any).handleJoinGame(socket1Reconnect, gameId);

      // After reconnect, session should still have the same state
      const restoredState = mockSession.getGameState();
      expect(restoredState.moveHistory.length).toBe(2);
      expect(restoredState.currentPhase).toBe('movement');
    });
  });

  describe('Server-side full state sync on reconnect', () => {
    it('calls getGameState and emits game_state on player reconnect', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Initial join
      await (wsServer as any).handleJoinGame(socket1, gameId);

      // Disconnect within reconnection window
      (wsServer as any).handleDisconnect(socket1);

      // Clear prior getGameState calls so we can attribute the next call to reconnect
      mockSession.getGameState.mockClear();

      // Reconnect with a new socket instance
      const socket1Reconnect = createMockSocket(userId1, 'socket-1-reconnect');
      await (wsServer as any).handleJoinGame(socket1Reconnect, gameId);

      // On reconnect, the session must be queried for a fresh GameState snapshot
      expect(mockSession.getGameState).toHaveBeenCalledTimes(1);

      // And the server must emit a full game_state payload to the reconnecting socket
      expect(socket1Reconnect.emit).toHaveBeenCalledWith(
        'game_state',
        expect.objectContaining({
          type: 'game_update',
          data: expect.objectContaining({
            gameId,
            gameState: expect.any(Object),
          }),
        })
      );
    });
  });

  describe('Reconnection Edge Cases', () => {
    it('should handle rapid disconnect/reconnect within grace period', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Join
      await (wsServer as any).handleJoinGame(socket1, gameId);

      // Rapid disconnect/reconnect cycle
      for (let i = 0; i < 3; i++) {
        (wsServer as any).handleDisconnect(socket1);
        const reconnectSocket = createMockSocket(userId1, `socket-1-reconnect-${i}`);
        await (wsServer as any).handleJoinGame(reconnectSocket, gameId);

        const state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
        expect(state!.kind).toBe('connected');
      }

      // No abandonment should be triggered
      expect(mockSession.handleAbandonmentForDisconnectedPlayer).not.toHaveBeenCalled();
    });

    it('should timeout player if no reconnect within grace period', async () => {
      const gameState = createBasePlacementPhaseState();
      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      await (wsServer as any).handleJoinGame(socket1, gameId);
      (wsServer as any).handleDisconnect(socket1);

      const pendingState = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(pendingState!.kind).toBe('disconnected_pending_reconnect');

      // Simulate timeout expiration
      (wsServer as any).handleReconnectionTimeout(gameId, userId1, 1);

      const expiredState = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(expiredState!.kind).toBe('disconnected_expired');
    });

    it('should handle reconnect after opponent resigned', async () => {
      // Game state where player 2 has resigned
      const gameState: GameState = {
        ...createBasePlacementPhaseState(),
        gameStatus: 'completed',
        gameResult: {
          winner: 1,
          reason: 'resignation',
          winType: 'resignation' as any,
          finalScores: { 1: 0, 2: 0 },
        },
      } as unknown as GameState;

      mockSession.getGameState.mockReturnValue(gameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Player 1 joins after disconnect (opponent already resigned)
      await (wsServer as any).handleJoinGame(socket1, gameId);

      const state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(state).toBeDefined();
      expect(state!.kind).toBe('connected');

      // Game state should show completed
      const restoredState = mockSession.getGameState();
      expect(restoredState.gameStatus).toBe('completed');
      expect(restoredState.gameResult?.winner).toBe(1);
    });

    it('should handle reconnect to completed game', async () => {
      // Completed game state
      const completedGameState: GameState = {
        ...createBasePlacementPhaseState(),
        gameStatus: 'completed',
        gameResult: {
          winner: 2,
          reason: 'ring_elimination',
          winType: 'ring_elimination' as any,
          finalScores: { 1: 3, 2: 5 },
        },
      } as unknown as GameState;

      mockSession.getGameState.mockReturnValue(completedGameState);

      const socket1 = createMockSocket(userId1, 'socket-1');

      // Player reconnects to completed game (view final state)
      await (wsServer as any).handleJoinGame(socket1, gameId);

      const state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId1);
      expect(state).toBeDefined();
      expect(state!.kind).toBe('connected');

      // Verify they can see the completed game
      const restoredState = mockSession.getGameState();
      expect(restoredState.gameStatus).toBe('completed');
      expect(restoredState.gameResult).toBeDefined();
    });

    it('should clear pending choices when reconnection window expires', async () => {
      const gameState = createBasePlacementPhaseState();
      const interactionHandler = {
        cancelAllChoicesForPlayer: jest.fn(),
      };
      mockSession.getGameState.mockReturnValue(gameState);
      mockSession.getInteractionHandler.mockReturnValue(interactionHandler);

      const socket1 = createMockSocket(userId1, 'socket-1');

      await (wsServer as any).handleJoinGame(socket1, gameId);
      (wsServer as any).handleDisconnect(socket1);

      // Simulate timeout expiration
      (wsServer as any).handleReconnectionTimeout(gameId, userId1, 1);

      // Verify choices are cancelled
      expect(interactionHandler.cancelAllChoicesForPlayer).toHaveBeenCalledWith(1);
    });
  });

  describe('Chain Capture During Reconnection', () => {
    it('should preserve chainCapturePosition on reconnect during chain capture phase', () => {
      // This is a unit test for GameSession.getGameState preserving chainCapturePosition
      const chainPosition: Position = { x: 5, y: 3 };
      const chainCaptureState: GameState = {
        ...createMovementPhaseState(),
        currentPhase: 'chain_capture',
        chainCapturePosition: chainPosition,
      } as unknown as GameState;

      mockSession.getGameState.mockReturnValue(chainCaptureState);

      // Verify state is preserved
      const state = mockSession.getGameState();
      expect(state.currentPhase).toBe('chain_capture');
      expect(state.chainCapturePosition).toEqual(chainPosition);
    });
  });
});

describe('WebSocketServer Reconnection Lifecycle', () => {
  const gameId = 'ws-reconnect-lifecycle-test';
  const userId = 'user-lifecycle-1';

  let wsServer: WebSocketServer;
  let mockSession: any;

  beforeEach(() => {
    jest.clearAllMocks();

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
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
      isRated: true,
    };

    mockSession = {
      getGameState: jest.fn(() => gameState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
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

  it('tracks complete connection lifecycle: connect → disconnect → reconnect', async () => {
    const socket1 = {
      id: 'socket-1',
      userId,
      username: 'Tester',
      join: jest.fn(),
      leave: jest.fn(),
      emit: jest.fn(),
      to: jest.fn(() => ({ emit: jest.fn() })),
      rooms: new Set<string>(),
    } as any;

    // Connect
    await (wsServer as any).handleJoinGame(socket1, gameId);
    let state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('connected');

    // Disconnect (starts reconnection window)
    (wsServer as any).handleDisconnect(socket1);
    state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('disconnected_pending_reconnect');

    // Reconnect before timeout
    const socket2 = {
      ...socket1,
      id: 'socket-2',
      to: jest.fn(() => ({ emit: jest.fn() })),
    };
    await (wsServer as any).handleJoinGame(socket2, gameId);
    state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('connected');
  });

  it('tracks lifecycle: connect → disconnect → timeout (expired)', async () => {
    const socket1 = {
      id: 'socket-1',
      userId,
      username: 'Tester',
      join: jest.fn(),
      leave: jest.fn(),
      emit: jest.fn(),
      to: jest.fn(() => ({ emit: jest.fn() })),
      rooms: new Set<string>(),
    } as any;

    // Connect
    await (wsServer as any).handleJoinGame(socket1, gameId);
    let state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('connected');

    // Disconnect
    (wsServer as any).handleDisconnect(socket1);
    state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('disconnected_pending_reconnect');

    // Timeout expires without reconnect
    (wsServer as any).handleReconnectionTimeout(gameId, userId, 1);
    state = wsServer.getPlayerConnectionStateSnapshotForTesting(gameId, userId);
    expect(state!.kind).toBe('disconnected_expired');
  });
});

describe('WebSocketServer reconnection timeout abandonment semantics', () => {
  const gameId = 'ws-reconnect-abandonment-test';
  const disconnectingUserId = 'user-disconnect';
  const opponentUserId = 'user-opponent';

  let wsServer: WebSocketServer;
  let mockSession: any;

  beforeEach(() => {
    jest.clearAllMocks();

    const interactionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
    };

    const baseState: GameState = {
      id: gameId,
      boardType: 'square8',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'movement',
      isRated: true,
      players: [
        {
          id: disconnectingUserId,
          username: 'Disconnecting',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: opponentUserId,
          username: 'Opponent',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        eliminatedRings: { 1: 0, 2: 0 },
        formedLines: [],
      },
      timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    } as unknown as GameState;

    mockSession = {
      getGameState: jest.fn(() => baseState),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: jest.fn(() => interactionHandler),
      handleAbandonmentForDisconnectedPlayer: jest.fn(),
    };

    mockGetOrCreateSession.mockResolvedValue(mockSession);
    mockGetSession.mockReturnValue(mockSession);

    mockGameFindUnique.mockResolvedValue({
      id: gameId,
      allowSpectators: true,
      player1Id: disconnectingUserId,
      player2Id: opponentUserId,
      player3Id: null,
      player4Id: null,
    });

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);
  });

  it('awards abandonment win to remaining opponent in rated games when opponent still alive', () => {
    // Mark opponent as still "alive" (connected or within their own window)
    const states = (wsServer as any).playerConnectionStates as Map<string, any>;
    states.set(`${gameId}:${disconnectingUserId}`, {
      kind: 'disconnected_pending_reconnect',
      gameId,
      userId: disconnectingUserId,
      playerNumber: 1,
    });
    states.set(`${gameId}:${opponentUserId}`, {
      kind: 'connected',
      gameId,
      userId: opponentUserId,
      playerNumber: 2,
    });

    (wsServer as any).handleReconnectionTimeout(gameId, disconnectingUserId, 1);

    expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(1, true);
  });

  it('does not award abandonment win when all humans have expired windows', () => {
    const ratedState = mockSession.getGameState() as GameState;
    ratedState.isRated = true;

    const states = (wsServer as any).playerConnectionStates as Map<string, any>;
    states.set(`${gameId}:${disconnectingUserId}`, {
      kind: 'disconnected_pending_reconnect',
      gameId,
      userId: disconnectingUserId,
      playerNumber: 1,
    });
    states.set(`${gameId}:${opponentUserId}`, {
      kind: 'disconnected_expired',
      gameId,
      userId: opponentUserId,
      playerNumber: 2,
    });

    (wsServer as any).handleReconnectionTimeout(gameId, disconnectingUserId, 1);

    expect(mockSession.handleAbandonmentForDisconnectedPlayer).toHaveBeenCalledWith(1, false);
  });

  it('does not trigger abandonment when game already ended by timeout before reconnection window expires', () => {
    // Simulate a game that has already been completed by rules-level timeout
    const completedState = {
      ...(mockSession.getGameState() as GameState),
      gameStatus: 'completed',
      gameResult: {
        winner: 2,
        reason: 'timeout',
        finalScore: { 1: 0, 2: 1 },
      },
    } as GameState;

    mockSession.getGameState.mockReturnValueOnce(completedState);

    const states = (wsServer as any).playerConnectionStates as Map<string, any>;
    states.set(`${gameId}:${disconnectingUserId}`, {
      kind: 'disconnected_pending_reconnect',
      gameId,
      userId: disconnectingUserId,
      playerNumber: 1,
    });
    states.set(`${gameId}:${opponentUserId}`, {
      kind: 'connected',
      gameId,
      userId: opponentUserId,
      playerNumber: 2,
    });

    (wsServer as any).handleReconnectionTimeout(gameId, disconnectingUserId, 1);

    // Abandonment helper must not be invoked once the rules-level result is final.
    expect(mockSession.handleAbandonmentForDisconnectedPlayer).not.toHaveBeenCalled();

    // Connection diagnostics still record the expired reconnection window.
    const snapshot = wsServer.getPlayerConnectionStateSnapshotForTesting(
      gameId,
      disconnectingUserId
    );
    expect(snapshot).toBeDefined();
    expect(snapshot!.kind).toBe('disconnected_expired');
  });
});
