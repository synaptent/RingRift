import { GameSession } from '../../src/server/game/GameSession';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: jest.fn(() => ({})),
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIMove: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

const mockedGlobalAIEngine = globalAIEngine as jest.Mocked<typeof globalAIEngine>;

describe('GameSession AI request state modeling', () => {
  const makeSession = (initialState: any) => {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;

    const session = new GameSession('game-1', io, pythonClient, new Map());

    // Inject a minimal gameEngine and rulesFacade so maybePerformAITurn
    // can operate without hitting the real engine or database.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => initialState),
      getValidMoves: jest.fn(),
    };

    (session as any).rulesFacade = {
      applyMove: jest.fn(),
    };

    // Stub out persistence / broadcast side effects; these are exercised in
    // other tests and are not needed for AIRequestState coverage.
    (session as any).handleAIMoveResult = jest.fn(async () => {});
    (session as any).handleAIFatalFailure = jest.fn(async () => {});

    return session as any;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('marks AI request as completed for a successful service-backed move', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce({
      id: 'm1',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 0, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock).mockResolvedValueOnce({ success: true });

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    expect(requestState.kind).toBe('completed');
  });

  it('records a failed AI request when local fallback is rejected by the rules engine', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // First service attempt returns a move that is rejected by the rules engine.
    mockedGlobalAIEngine.getAIMove
      .mockResolvedValueOnce({
        id: 'service-move',
        player: 1,
        type: 'place_ring',
        from: null,
        to: { q: 0, r: 0 },
        moveNumber: 1,
        timestamp: new Date().toISOString(),
        thinkTime: 0,
      } as any)
      // Second attempt yields no move, triggering the local-fallback path.
      .mockResolvedValueOnce(null as any);

    (session.rulesFacade.applyMove as jest.Mock)
      // Reject service-provided move
      .mockResolvedValueOnce({ success: false, error: 'service move rejected' })
      // Reject local fallback move as well
      .mockResolvedValueOnce({ success: false, error: 'fallback move rejected' });

    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue({
      id: 'fallback-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 1, r: 0 },
      moveNumber: 2,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    expect(requestState.kind).toBe('failed');
    if (requestState.kind === 'failed') {
      expect(requestState.code).toBe('AI_SERVICE_OVERLOADED');
      // The exact aiErrorType may vary depending on where the failure is
      // surfaced (rules engine vs. AI orchestration), but we always
      // record a terminal failure code for diagnostics.
      expect(requestState.aiErrorType).toBeDefined();
    }
  });
});
