import { WebSocketServer } from '../../src/server/websocket/server';
import { Move, GameState } from '../../src/shared/types/game';

// Jest-hoisted mock state for the Prisma client methods used by
// WebSocketServer.handlePlayerMove. We keep these mocks at the module
// level so individual tests can configure expectations.
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

// Minimal Socket.IO "server" stub that records game_state emissions.
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

describe('WebSocketServer + RulesBackendFacade integration', () => {
  beforeEach(() => {
    mockFindUnique.mockReset();
    mockCreateMove.mockReset();
    mockUpdateGame.mockReset();
  });

  it('handlePlayerMove delegates to RulesBackendFacade.applyMove when a facade is registered', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const gameId = 'game-rules-backend';
    const userId = 'user-1';

    // Lightweight game record: active status so the handler proceeds.
    mockFindUnique.mockResolvedValue({
      id: gameId,
      status: 'active',
      allowSpectators: true,
    } as any);

    const baseState: GameState = {
      id: gameId,
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      },
      players: [
        {
          id: userId,
          username: 'Human',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 18,
      totalRingsEliminated: 0,
      victoryThreshold: 10,
      territoryVictoryThreshold: 32,
    };

    const state = { ...baseState };

    const fakeEngine: any = {
      getGameState: jest.fn(() => state),
      makeMove: jest.fn(),
      makeMoveById: jest.fn(),
      getValidMoves: jest.fn(() => []),
    };

    // Bypass DB-backed engine creation and inject our fake engine.
    serverAny.getOrCreateGameEngine = jest.fn().mockResolvedValue(fakeEngine);

    // Register a fake RulesBackendFacade instance for this game. We only
    // care that applyMove is invoked with the canonical engineMove payload;
    // we do not need a real facade implementation here.
    const fakeRulesBackend = {
      applyMove: jest.fn(
        async (
          engineMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        ): Promise<{ success: boolean; gameState?: GameState }> => {
          // Simulate the engine applying the move by pushing a synthetic
          // entry into moveHistory so handlePlayerMove can fetch it via
          // getGameState().
          state.moveHistory.push({
            ...engineMove,
            id: 'move-1',
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          } as Move);
          return { success: true, gameState: state };
        }
      ),
    };
    serverAny.rulesFacades.set(gameId, fakeRulesBackend);

    // Minimal AuthenticatedSocket stub for a human player in the room.
    const fakeSocket: any = {
      userId,
      username: 'Human',
      gameId,
    };

    // Client payload: geometry-based move. The server parses move.position
    // as JSON and translates it into a canonical Move payload for the
    // rules backend.
    const clientMove = {
      gameId,
      move: {
        moveNumber: 1,
        position: JSON.stringify({ to: { x: 0, y: 0 } }),
        moveType: 'place_ring',
      },
    };

    await serverAny.handlePlayerMove(fakeSocket, clientMove);

    // Engine should have been resolved and the facade used to apply the
    // canonical move.
    expect(serverAny.getOrCreateGameEngine).toHaveBeenCalledWith(gameId);
    expect(fakeRulesBackend.applyMove).toHaveBeenCalledTimes(1);

    const appliedArg = fakeRulesBackend.applyMove.mock.calls[0][0] as Omit<
      Move,
      'id' | 'timestamp' | 'moveNumber'
    >;
    expect(appliedArg.player).toBe(1);
    expect(appliedArg.type).toBe('place_ring');
    expect(appliedArg.to).toEqual({ x: 0, y: 0 });

    // The legacy GameEngine.makeMove path should NOT have been used for
    // geometry-based moves when a RulesBackendFacade is registered.
    expect(fakeEngine.makeMove).not.toHaveBeenCalled();

    // The handler should persist the move and emit a game_state event.
    expect(mockCreateMove).toHaveBeenCalledTimes(1);
    const moveCreateArgs = mockCreateMove.mock.calls[0][0];
    expect(moveCreateArgs.data.gameId).toBe(gameId);
    expect(moveCreateArgs.data.moveType).toBe('place_ring');

    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);
    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe(gameId);
    expect(payload.data.gameState.currentPlayer).toBe(state.currentPlayer);
  });
});
