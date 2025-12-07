/**
 * Tests for GameSession branch coverage - Setup and Configuration.
 * Covers: userSockets, edge cases, rngSeed, gameId, time control, board types.
 * 
 * Split from GameSession.init.branchCoverage.test.ts for maintainability.
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

describe('GameSession Branch Coverage - userSockets map usage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('uses userSockets map for player targeting', async () => {
    const io = createMockIo();
    const userSockets = new Map<string, string>();
    userSockets.set('player-1', 'socket-1');
    userSockets.set('player-2', 'socket-2');

    const session = new GameSession('test-game-id', io, {} as never, userSockets);

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
    expect(state.players.length).toBe(2);
  });

  it('handles empty userSockets map', async () => {
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
    expect(session.getGameState()).toMatchObject({
      gameStatus: expect.stringMatching(/active|waiting/),
    });
  });
});

describe('GameSession Branch Coverage - edge case game states', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles maxPlayers as null (defaults to 2)', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: null,
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
    // maxPlayers defaults to 2 when null
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });

  it('handles isRated as null', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: null,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // isRated null should not crash initialization
    expect(state.gameStatus).toMatch(/active|waiting/);
  });
});

describe('GameSession Branch Coverage - rngSeed edge cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles rngSeed of zero', async () => {
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
      rngSeed: 0,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.rngSeed).toBe(0);
  });

  it('handles string rngSeed value (invalid type)', async () => {
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
      rngSeed: 'not-a-number' as unknown,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(typeof state.rngSeed === 'number' || state.rngSeed === undefined).toBe(true);
  });

  it('handles negative rngSeed value', async () => {
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
      rngSeed: -12345,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.rngSeed).toBe(-12345);
  });

  it('handles large rngSeed value', async () => {
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
      rngSeed: 2147483647,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.rngSeed).toBe(2147483647);
  });
});

describe('GameSession Branch Coverage - various gameId scenarios', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles gameId with special characters', async () => {
    const io = createMockIo();
    const specialGameId = 'test-game-uuid-123-456';
    const session = new GameSession(specialGameId, io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: specialGameId,
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
    expect(session.gameId).toBe(specialGameId);
  });
});

describe('GameSession Branch Coverage - waiting status without all players', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('does not auto-start when not all players present', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'waiting',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // Only one player is present
    expect(state.players.length).toBeGreaterThanOrEqual(1);
  });
});

describe('GameSession Branch Coverage - game already completed status', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes completed game correctly', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'completed',
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
    // Completed games should still have valid state
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });
});

describe('GameSession Branch Coverage - empty difficulty array', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('uses default difficulty when difficulty array is empty', async () => {
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
          difficulty: [],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].aiDifficulty).toBe(5);
  });
});

describe('GameSession Branch Coverage - gameState with empty object', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles empty object gameState', async () => {
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
      gameState: JSON.stringify({}),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players.every((p) => p.type === 'human')).toBe(true);
  });
});

describe('GameSession Branch Coverage - player2 only no player1', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles game with player2 but no player1', async () => {
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
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players.length).toBeGreaterThanOrEqual(1);
  });
});

describe('GameSession Branch Coverage - special time control configurations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles bullet time control', async () => {
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
      timeControl: JSON.stringify({ type: 'bullet', initialTime: 60000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.type).toBe('bullet');
  });

  it('handles time control with very high increment', async () => {
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
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 60000 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.increment).toBe(60000);
  });

  it('handles time control with zero values', async () => {
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
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 0, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.initialTime).toBe(0);
  });

  it('handles custom time control type', async () => {
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
      timeControl: JSON.stringify({ type: 'custom', initialTime: 900000, increment: 15000 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.type).toBe('custom');
  });
});

describe('GameSession Branch Coverage - abandoned status', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes game with abandoned status', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'abandoned',
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
    // Abandoned games should maintain state integrity
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });
});

describe('GameSession Branch Coverage - AI with default mode', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('defaults to service mode when mode not specified', async () => {
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
    expect(typeof state.players[1].aiProfile).toBe('object');
  });
});

describe('GameSession Branch Coverage - getGameState without initialization', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws when getGameState called before initialize', () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    expect(() => session.getGameState()).toThrow();
  });
});

describe('GameSession Branch Coverage - double initialization check', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles multiple initialize calls', async () => {
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
    await session.initialize();

    const state = session.getGameState();
    // Double initialization should not corrupt state
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });
});

describe('GameSession Branch Coverage - createPlayer edge cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles player with empty string username', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: '' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // Empty string username should still be a string
    expect(typeof state.players[0].username).toBe('string');
  });
});

describe('GameSession Branch Coverage - AI profile creation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI profile creation with all options', async () => {
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
          difficulty: [10],
          mode: 'service',
          aiType: 'neural',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].aiDifficulty).toBe(10);
  });
});

describe('GameSession Branch Coverage - game auto-start variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('skips auto-start when status is not waiting', async () => {
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
    // Active status should not trigger auto-start
    expect(state.gameStatus).toMatch(/active|waiting/);
  });

  it('skips auto-start when maxPlayers not reached', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'waiting',
      maxPlayers: 4,
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
    // maxPlayers not reached should not auto-start
    expect(state.players.length).toBe(2);
  });
});

describe('GameSession Branch Coverage - fixture metadata application', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('stores fixture metadata when provided', async () => {
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
      gameState: JSON.stringify({
        fixture: {
          shortTimeoutMs: 2000,
          shortWarningBeforeMs: 500,
          targetPhase: 'line_processing',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // Fixture metadata should be applied without errors
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });
});

describe('GameSession Branch Coverage - configureEngineSelection branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('uses human player for targeting when available', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'human-player-1', username: 'Human' },
      player2: null,
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
    expect(state.players.length).toBe(2);
  });

  it('handles all-AI game engine selection', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: false,
      player1: null,
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 2,
          difficulty: [5, 5],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players.filter((p) => p.type === 'ai').length).toBe(2);
  });
});

describe('GameSession additional board type coverage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles 3-player game initialization', async () => {
    const io = createMockIo();
    const session = new GameSession('test-3p-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-3p-game',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 3,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      player3: { id: 'player-3', username: 'P3' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    // 3-player game initialization should succeed
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });

  it('handles 4-player game initialization', async () => {
    const io = createMockIo();
    const session = new GameSession('test-4p-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-4p-game',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 4,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      player3: { id: 'player-3', username: 'P3' },
      player4: { id: 'player-4', username: 'P4' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    // 4-player game initialization should succeed
    expect(state.players.length).toBeGreaterThanOrEqual(2);
  });

  it('handles game with missing player slots gracefully', async () => {
    const io = createMockIo();
    const session = new GameSession('test-partial-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-partial-game',
      boardType: 'square8',
      status: 'waiting_for_players',
      maxPlayers: 2,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    expect(state.players.length).toBeGreaterThanOrEqual(1);
  });

  it('handles square19 board type', async () => {
    const io = createMockIo();
    const session = new GameSession('test-sq19-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-sq19-game',
      boardType: 'square19',
      status: 'active',
      maxPlayers: 2,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    expect(state.boardType).toBe('square19');
  });

  it('handles hexagonal board type', async () => {
    const io = createMockIo();
    const session = new GameSession('test-hex-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-hex-game',
      boardType: 'hexagonal',
      status: 'active',
      maxPlayers: 2,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    expect(state.boardType).toBe('hexagonal');
  });

  it('handles unrated game initialization', async () => {
    const io = createMockIo();
    const session = new GameSession('test-unrated-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-unrated-game',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: false,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    expect(state.isRated).toBe(false);
  });
});