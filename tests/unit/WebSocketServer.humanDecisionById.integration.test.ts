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

    // Bypass DB-backed engine creation and inject our fake engine.
    serverAny.getOrCreateGameEngine = jest.fn().mockResolvedValue(fakeEngine);

    // Minimal AuthenticatedSocket stub for a human player in the room.
    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, { gameId, moveId: decisionMove.id });

    // Engine should have been resolved and invoked as expected.
    expect(serverAny.getOrCreateGameEngine).toHaveBeenCalledWith(gameId);
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, decisionMove.id);

    // The handler should persist the canonical Move that was actually
    // applied, using the last entry in moveHistory.
    expect(mockCreateMove).toHaveBeenCalledTimes(1);
    const moveCreateArgs = mockCreateMove.mock.calls[0][0];
    expect(moveCreateArgs.data.gameId).toBe(gameId);
    expect(moveCreateArgs.data.moveType).toBe('process_line');

    // A game_state event should have been emitted with the updated state
    // and next-player validMoves payload.
    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);

    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe(gameId);
    expect(payload.data.gameState.currentPhase).toBe(state.currentPhase);
    expect(Array.isArray(payload.data.validMoves)).toBe(true);
  });

  it('applies a canonical process_territory_region Move for a human in territory_processing via player_move_by_id and emits game_state', async () => {
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
      type: 'process_territory_region',
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

    serverAny.getOrCreateGameEngine = jest.fn().mockResolvedValue(fakeEngine);

    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, {
      gameId,
      moveId: territoryDecisionMove.id,
    });

    expect(serverAny.getOrCreateGameEngine).toHaveBeenCalledWith(gameId);
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, territoryDecisionMove.id);

    expect(mockCreateMove).toHaveBeenCalledTimes(1);
    const moveCreateArgs = mockCreateMove.mock.calls[0][0];
    expect(moveCreateArgs.data.gameId).toBe(gameId);
    expect(moveCreateArgs.data.moveType).toBe('process_territory_region');

    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);
    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe(gameId);
    expect(payload.data.gameState.currentPhase).toBe(state.currentPhase);
    expect(Array.isArray(payload.data.validMoves)).toBe(true);
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

    serverAny.getOrCreateGameEngine = jest.fn().mockResolvedValue(fakeEngine);

    const fakeSocket: any = {
      userId: 'user-1',
      username: 'Human',
      gameId,
    };

    await serverAny.handlePlayerMoveById(fakeSocket, {
      gameId,
      moveId: eliminationDecisionMove.id,
    });

    expect(serverAny.getOrCreateGameEngine).toHaveBeenCalledWith(gameId);
    expect(fakeEngine.makeMoveById).toHaveBeenCalledWith(1, eliminationDecisionMove.id);

    expect(mockCreateMove).toHaveBeenCalledTimes(1);
    const moveCreateArgs = mockCreateMove.mock.calls[0][0];
    expect(moveCreateArgs.data.gameId).toBe(gameId);
    expect(moveCreateArgs.data.moveType).toBe('eliminate_rings_from_stack');

    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);
    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe(gameId);
    expect(payload.data.gameState.currentPhase).toBe(state.currentPhase);
    expect(Array.isArray(payload.data.validMoves)).toBe(true);
  });
});
