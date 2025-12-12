import { WebSocketServer } from '../../src/server/websocket/server';
import { Move } from '../../src/shared/types/game';

// Jest-hoisted mock state for the Prisma client methods used by
// WebSocketServer.handlePlayerMoveById. We keep these mocks at the
// module level so individual tests can configure expectations.
const mockFindUnique = jest.fn();
const mockCreateMove = jest.fn();
const mockUpdateGame = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockFindUnique,
      update: mockUpdateGame,
    },
    move: {
      create: mockCreateMove,
    },
  }),
}));

// Minimal Socket.IO "server" stub that records game_state emissions
// and mirrors the shape used in WebSocketServer.aiTurn.integration.test.
class FakeSocketIOServer {
  public toCalls: Array<{ gameId: string; event: string; payload: any }> = [];

  to(gameId: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ gameId, event, payload });
      },
    };
  }
}

/**
 * Helper to create a fake GameSession that wraps a fake engine.
 * This mirrors the shape that handlePlayerMoveById expects from
 * GameSessionManager.getOrCreateSession().
 */
function createFakeSession(fakeEngine: any) {
  return {
    handlePlayerMoveById: jest.fn(async (socket: any, moveId: string) => {
      const state = fakeEngine.getGameState();
      const player = state.players.find((p: any) => p.id === socket.userId);
      if (!player) throw new Error('Player not found');
      await fakeEngine.makeMoveById(player.playerNumber, moveId);
    }),
    getGameState: fakeEngine.getGameState,
    getEngine: jest.fn(() => fakeEngine),
  };
}

describe('WebSocketServer.handlePlayerMoveById (human decision phases)', () => {
  beforeEach(() => {
    mockFindUnique.mockReset();
    mockCreateMove.mockReset();
    mockUpdateGame.mockReset();
  });

  it('applies a canonical process_line Move for a human in line_processing via player_move_by_id and emits game_state', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const gameId = 'game-line-processing';

    // Lightweight game record: active status so the handler proceeds.
    mockFindUnique.mockResolvedValue({
      id: gameId,
      status: 'active',
      allowSpectators: true,
    } as any);

    const decisionMove: Move = {
      id: 'process-line-0-0,0',
      type: 'process_line',
      player: 1,
      formedLines: [],
      // Decision-phase Moves do not require a real `to`, but Move
      // type demands it; provide a harmless sentinel.
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const state: any = {
      gameStatus: 'active',
      currentPhase: 'line_processing',
      currentPlayer: 1,
      players: [
        { id: 'user-1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'ai-2', username: 'AI', playerNumber: 2, type: 'ai' },
      ],
      moveHistory: [decisionMove],
    };

    const fakeEngine: any = {
      getGameState: jest.fn(() => state),
      makeMoveById: jest.fn(async (playerNumber: number, moveId: string) => {
        // Ensure the handler passes the correct player and Move id through.
        expect(playerNumber).toBe(1);
        expect(moveId).toBe(decisionMove.id);
        return { success: true, gameState: state };
      }),
      // For this integration test we only care that getValidMoves can be
      // called successfully when broadcasting the next player's options.
      getValidMoves: jest.fn(() => []),
    };

    const fakeSession = createFakeSession(fakeEngine);

    // Mock the sessionManager to return our fake session
    serverAny.sessionManager = {
      withGameLock: jest.fn(async (_gameId: string, fn: () => Promise<void>) => {
        await fn();
      }),
      getOrCreateSession: jest.fn().mockResolvedValue(fakeSession),
    };

    // Minimal AuthenticatedSocket stub for a human player in the room.
    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, { gameId, moveId: decisionMove.id });

    // Session manager should have been invoked with a lock.
    expect(serverAny.sessionManager.withGameLock).toHaveBeenCalledWith(
      gameId,
      expect.any(Function)
    );
    expect(serverAny.sessionManager.getOrCreateSession).toHaveBeenCalledWith(gameId);

    // Verify the session's handlePlayerMoveById was called
    expect(fakeSession.handlePlayerMoveById).toHaveBeenCalledWith(fakeSocket, decisionMove.id);

    // Verify the engine's makeMoveById was called through the session
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, decisionMove.id);
  });

  it('applies a canonical choose_territory_option Move for a human in territory_processing via player_move_by_id and emits game_state', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const gameId = 'game-territory-processing';

    mockFindUnique.mockResolvedValue({
      id: gameId,
      status: 'active',
      allowSpectators: true,
    } as any);

    const territoryDecisionMove: Move = {
      id: 'process-region-0-0,1',
      type: 'choose_territory_option',
      player: 1,
      // Territory decisions identify the region via disconnectedRegions[0].
      disconnectedRegions: [],
      to: { x: 0, y: 1 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const state: any = {
      gameStatus: 'active',
      currentPhase: 'territory_processing',
      currentPlayer: 1,
      players: [
        { id: 'user-1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'ai-2', username: 'AI', playerNumber: 2, type: 'ai' },
      ],
      moveHistory: [territoryDecisionMove],
    };

    const fakeEngine: any = {
      getGameState: jest.fn(() => state),
      makeMoveById: jest.fn(async (playerNumber: number, moveId: string) => {
        expect(playerNumber).toBe(1);
        expect(moveId).toBe(territoryDecisionMove.id);
        return { success: true, gameState: state };
      }),
      getValidMoves: jest.fn(() => []),
    };

    const fakeSession = createFakeSession(fakeEngine);

    serverAny.sessionManager = {
      withGameLock: jest.fn(async (_gameId: string, fn: () => Promise<void>) => {
        await fn();
      }),
      getOrCreateSession: jest.fn().mockResolvedValue(fakeSession),
    };

    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, {
      gameId,
      moveId: territoryDecisionMove.id,
    });

    expect(serverAny.sessionManager.getOrCreateSession).toHaveBeenCalledWith(gameId);
    expect(fakeSession.handlePlayerMoveById).toHaveBeenCalledWith(
      fakeSocket,
      territoryDecisionMove.id
    );
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, territoryDecisionMove.id);
  });

  it('applies a canonical eliminate_rings_from_stack Move for a human in territory_processing via player_move_by_id and emits game_state', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const gameId = 'game-territory-elimination';

    mockFindUnique.mockResolvedValue({
      id: gameId,
      status: 'active',
      allowSpectators: true,
    } as any);

    const eliminationDecisionMove: Move = {
      id: 'eliminate-0,1',
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: { x: 0, y: 1 },
      eliminatedRings: [{ player: 1, count: 1 }],
      eliminationFromStack: {
        position: { x: 0, y: 1 },
        capHeight: 1,
        totalHeight: 2,
      },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 2,
    };

    const state: any = {
      gameStatus: 'active',
      currentPhase: 'territory_processing',
      currentPlayer: 1,
      players: [
        { id: 'user-1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'ai-2', username: 'AI', playerNumber: 2, type: 'ai' },
      ],
      moveHistory: [eliminationDecisionMove],
    };

    const fakeEngine: any = {
      getGameState: jest.fn(() => state),
      makeMoveById: jest.fn(async (playerNumber: number, moveId: string) => {
        expect(playerNumber).toBe(1);
        expect(moveId).toBe(eliminationDecisionMove.id);
        return { success: true, gameState: state };
      }),
      getValidMoves: jest.fn(() => []),
    };

    const fakeSession = createFakeSession(fakeEngine);

    serverAny.sessionManager = {
      withGameLock: jest.fn(async (_gameId: string, fn: () => Promise<void>) => {
        await fn();
      }),
      getOrCreateSession: jest.fn().mockResolvedValue(fakeSession),
    };

    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, {
      gameId,
      moveId: eliminationDecisionMove.id,
    });

    expect(serverAny.sessionManager.getOrCreateSession).toHaveBeenCalledWith(gameId);
    expect(fakeSession.handlePlayerMoveById).toHaveBeenCalledWith(
      fakeSocket,
      eliminationDecisionMove.id
    );
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, eliminationDecisionMove.id);
  });
});
