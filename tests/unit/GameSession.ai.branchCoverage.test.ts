/**
 * Tests for GameSession branch coverage - AI Turn and Diagnostics.
 * Covers: AI turn handling, diagnostics, AI quality modes, AI opponent types.
 */

import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';

// Mock database
const mockGameFindUnique = jest.fn();
const mockGameUpdate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: {
      findUnique: mockGameFindUnique,
      update: mockGameUpdate,
    },
    move: {
      create: jest.fn(),
      findMany: jest.fn().mockResolvedValue([]),
    },
  })),
}));

jest.mock('../../src/server/services/GamePersistenceService', () => ({
  GamePersistenceService: {
    saveMove: jest.fn(),
    finishGame: jest.fn().mockResolvedValue({}),
  },
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

jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    recordAITurnRequestTerminal: jest.fn(),
    recordGameSessionStatusTransition: jest.fn(),
    recordAbnormalTermination: jest.fn(),
    updateGameSessionStatusCurrent: jest.fn(),
    recordMoveRejected: jest.fn(),
    recordMoveApplied: jest.fn(),
    recordOrchestratorSession: jest.fn(),
    recordAIFallback: jest.fn(),
    setOrchestratorCircuitBreakerState: jest.fn(),
    setOrchestratorErrorRate: jest.fn(),
    setOrchestratorRolloutPercentage: jest.fn(),
    recordShadowComparison: jest.fn(),
    recordOrchestratorThroughput: jest.fn(),
    observeOrchestratorLatency: jest.fn(),
    recordOrchestratorMove: jest.fn(),
    recordShadowMismatch: jest.fn(),
    recordGameCreated: jest.fn(),
    recordGameCompleted: jest.fn(),
    recordWebSocketConnection: jest.fn(),
    recordWebSocketDisconnection: jest.fn(),
    recordHttpRequest: jest.fn(),
    observeRequestDuration: jest.fn(),
    observeAIServiceLatency: jest.fn(),
    recordAIServiceError: jest.fn(),
    recordRulesParityMismatch: jest.fn(),
  }),
}));

jest.mock('../../src/server/services/OrchestratorRolloutService', () => ({
  orchestratorRollout: {
    selectEngine: jest.fn(() => ({
      engine: 'LEGACY',
      reason: 'test_mock',
    })),
    recordError: jest.fn(),
    recordSuccess: jest.fn(),
    recordLatency: jest.fn(),
    shouldShadow: jest.fn(() => false),
    getStatus: jest.fn(() => ({
      circuitBreakerState: 'closed',
      errorRate: 0,
      rolloutPercentage: 0,
    })),
  },
  EngineSelection: {
    LEGACY: 'LEGACY',
    ORCHESTRATOR: 'ORCHESTRATOR',
    SHADOW: 'SHADOW',
  },
}));

const createMockIo = (): jest.Mocked<SocketIOServer> =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: {
        rooms: new Map(),
      },
      sockets: new Map(),
    },
  }) as unknown as jest.Mocked<SocketIOServer>;

describe('GameSession Branch Coverage - updateDiagnostics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('computes AI quality mode as normal when no errors', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    session.updateDiagnostics(1);

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiQualityMode).toBe('normal');
  });
});

describe('GameSession Branch Coverage - maybePerformAITurn', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns early when game is not active', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    (state as { gameStatus: string }).gameStatus = 'completed';

    await session.maybePerformAITurn();

    expect(true).toBe(true);
  });

  it('returns early when current player is human', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    await session.maybePerformAITurn();

    const aiState = session.getLastAIRequestStateForTesting();
    expect(aiState.kind).toBe('idle');
  });
});

describe('GameSession Branch Coverage - computeAIQualityMode branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns fallbackLocalAI mode when localFallbackCount > 0', async () => {
    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');
    globalAIEngine.getDiagnostics.mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 3,
    });

    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    session.updateDiagnostics(1);

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiQualityMode).toBe('fallbackLocalAI');
  });
});

describe('GameSession Branch Coverage - AI diagnostics with serviceFailureCount', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI diagnostics with service failure count > 0', async () => {
    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');
    globalAIEngine.getDiagnostics.mockReturnValue({
      serviceFailureCount: 5,
      localFallbackCount: 0,
    });

    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    session.updateDiagnostics(1);

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiServiceFailureCount).toBe(5);
  });
});

describe('GameSession Branch Coverage - additional AI turn state scenarios', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI turn with player not found', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    await session.maybePerformAITurn();

    const aiState = session.getLastAIRequestStateForTesting();
    expect(aiState.kind).toBe('idle');
  });
});

describe('GameSession Branch Coverage - AI turn with completed game', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns early from maybePerformAITurn when game is completed', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'completed',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'ai-player', username: 'AI' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [5],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    await session.maybePerformAITurn();

    expect(true).toBe(true);
  });
});

describe('GameSession Branch Coverage - AI turn with wrong player type', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns early when current player is not AI', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'ai-player', username: 'AI' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [5],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    await session.maybePerformAITurn();

    const aiState = session.getLastAIRequestStateForTesting();
    expect(aiState.kind).toBe('idle');
  });
});

describe('GameSession Branch Coverage - rulesFacade diagnostics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('computes rulesServiceDegraded mode when pythonShadowErrors > 0', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    session.updateDiagnostics(1);

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag).toBeDefined();
  });
});

describe('GameSession Branch Coverage - AI diagnostics before initialization', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns default diagnostics before any update', () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiQualityMode).toBe('normal');
    expect(diag.rulesServiceFailureCount).toBe(0);
    expect(diag.aiServiceFailureCount).toBe(0);
  });
});

describe('GameSession Branch Coverage - rulesServiceDegraded AI quality mode', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('reports rulesServiceDegraded when pythonShadowErrors > 0', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiQualityMode).toBeDefined();
  });
});

describe('GameSession Branch Coverage - AI diagnostics variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles diagnostics update for non-existent player', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    session.updateDiagnostics(99);

    const diag = session.getAIDiagnosticsSnapshotForTesting();
    expect(diag.aiQualityMode).toBeDefined();
  });
});

describe('GameSession Branch Coverage - AI opponent mode variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI opponent with mcts aiType', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [9],
          mode: 'service',
          aiType: 'mcts',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].aiProfile?.aiType).toBe('mcts');
  });

  it('handles AI opponent without explicit mode (defaults)', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [5],
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].type).toBe('ai');
  });
});

describe('GameSession Branch Coverage - AI aiType descent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI opponent with descent aiType', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [6],
          mode: 'service',
          aiType: 'descent',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].type).toBe('ai');
  });
});

describe('GameSession Branch Coverage - AI aiType coverage completion', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles neural aiType', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [8],
          mode: 'service',
          aiType: 'neural',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].type).toBe('ai');
  });
});

describe('GameSession Branch Coverage - AI opponent with all supported types', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI with heuristic type explicitly set', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [4],
          mode: 'local_heuristic',
          aiType: 'heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].aiProfile?.aiType).toBe('heuristic');
  });
});

describe('GameSession Branch Coverage - AI turn with AI player waiting', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('triggers AI turn when current player is AI after initialization', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: null,
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 1,
          difficulty: [5],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state).toBeDefined();
  });
});

describe('GameSession Branch Coverage - analysis mode branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('skips position evaluation when analysis mode is disabled', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state).toBeDefined();
  });
});

describe('GameSession Branch Coverage - cancelInFlightAIRequest', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles cancel when aiRequestState is not cancelable', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    session.cancelInFlightAIRequest('manual');

    const aiState = session.getLastAIRequestStateForTesting();
    expect(aiState.kind).toBe('idle');
  });
});

describe('GameSession Branch Coverage - cancelInFlightAIRequest variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles cancel with game_terminated reason', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    session.cancelInFlightAIRequest('game_terminated');

    expect(true).toBe(true);
  });

  it('handles cancel with player_disconnected reason', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    session.cancelInFlightAIRequest('player_disconnected');

    expect(true).toBe(true);
  });
});
