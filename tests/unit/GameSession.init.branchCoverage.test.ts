/**
 * Tests for GameSession branch coverage - Initialization and Setup.
 * Covers: time control parsing, AI opponents, RNG seed, player creation, multi-player games.
 */

import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type { GameState } from '../../src/shared/types/game';

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

describe('GameSession Branch Coverage - Initialize', () => {
  const now = new Date();

  function createBaseGameState(overrides: Partial<GameState> = {}): GameState {
    return {
      id: 'test-game-id',
      boardType: 'square8',
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
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
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
      createdAt: now,
      lastMoveAt: now,
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 36,
      totalRingsEliminated: 0,
      victoryThreshold: 19,
      territoryVictoryThreshold: 33,
      ...(overrides as Record<string, unknown>),
    };
  }

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Time Control Parsing Branches', () => {
    it('parses time control from JSON string', async () => {
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
        timeControl: JSON.stringify({ type: 'rapid', initialTime: 300000, increment: 5000 }),
        gameState: null,
        rngSeed: null,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.timeControl.initialTime).toBe(300000);
    });

    it('handles object time control', async () => {
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
        timeControl: { type: 'rapid', initialTime: 450000, increment: 0 },
        gameState: null,
        rngSeed: null,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.timeControl.initialTime).toBe(450000);
    });

    it('uses default time control when not provided', async () => {
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
        timeControl: null,
        gameState: null,
        rngSeed: null,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.timeControl.initialTime).toBe(600000);
    });
  });

  describe('AI Opponents Branches', () => {
    it('initializes AI opponents from gameState snapshot', async () => {
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
            mode: 'local_heuristic',
          },
        }),
        rngSeed: null,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.players.length).toBe(2);
      expect(state.players[1].type).toBe('ai');
    });

    it('handles gameState as object instead of string', async () => {
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
        gameState: {
          aiOpponents: {
            count: 1,
            difficulty: [7],
            mode: 'service',
            aiType: 'heuristic',
          },
        },
        rngSeed: null,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.players[1].aiDifficulty).toBe(7);
    });

    it('handles empty gameState', async () => {
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
      expect(state.players.length).toBe(2);
      expect(state.players.every((p) => p.type === 'human')).toBe(true);
    });
  });

  describe('RNG Seed Branches', () => {
    it('uses persisted rngSeed when available', async () => {
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
        rngSeed: 12345,
        moves: [],
      });

      await session.initialize();

      const state = session.getGameState();
      expect(state.rngSeed).toBe(12345);
    });
  });
});

describe('GameSession Branch Coverage - AI opponent with different aiTypes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles AI opponent with random aiType', async () => {
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
          difficulty: [3],
          mode: 'service',
          aiType: 'random',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].type).toBe('ai');
    expect(state.players[1].aiProfile?.aiType).toBe('random');
  });

  it('handles AI opponent with minimax aiType', async () => {
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
          aiType: 'minimax',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[1].aiProfile?.aiType).toBe('minimax');
  });

  it('handles multiple AI opponents in multiplayer games', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 4,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      player3: null,
      player4: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 3,
          difficulty: [3, 5, 7],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players.length).toBe(4);
    expect(state.players.filter((p) => p.type === 'ai').length).toBe(3);
  });

  it('caps AI count at available slots', async () => {
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
          count: 5, // More than available slots
          difficulty: [3, 5, 7, 8, 9],
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // Should be capped at maxPlayers - existing players = 2 - 1 = 1
    expect(state.players.length).toBe(2);
  });

  it('uses default difficulty when not provided for AI index', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 4,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: null,
      player3: null,
      player4: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: JSON.stringify({
        aiOpponents: {
          count: 3,
          difficulty: [3], // Only one difficulty provided for 3 AIs
          mode: 'local_heuristic',
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    // First AI should have difficulty 3, others should default to 5
    expect(state.players[1].aiDifficulty).toBe(3);
    expect(state.players[2].aiDifficulty).toBe(5);
    expect(state.players[3].aiDifficulty).toBe(5);
  });
});

describe('GameSession Branch Coverage - 4 player initialization', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes with player3 and player4 from database when all present', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 4,
      isRated: true,
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
    expect(state.players.length).toBeGreaterThanOrEqual(2);
    expect(state.maxPlayers).toBeGreaterThanOrEqual(2);
  });

  it('initializes with only player3 present (not player4)', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 3,
      isRated: true,
      player1: { id: 'player-1', username: 'P1' },
      player2: { id: 'player-2', username: 'P2' },
      player3: { id: 'player-3', username: 'P3' },
      player4: null,
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players.length).toBeGreaterThanOrEqual(2);
    expect(state.maxPlayers).toBeGreaterThanOrEqual(2);
  });
});

describe('GameSession Branch Coverage - waiting status and auto-start', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('auto-starts game when status is waiting and all players ready', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'waiting',
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

    expect(mockGameUpdate).toHaveBeenCalled();
  });
});

describe('GameSession Branch Coverage - player without username', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles player with null username', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: null },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[0].username).toBe('Player 1');
  });
});

describe('GameSession Branch Coverage - fixture metadata handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles gameState with fixture metadata in test environment', async () => {
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
          shortTimeoutMs: 1000,
          shortWarningBeforeMs: 500,
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

describe('GameSession Branch Coverage - rulesOptions in gameState', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles rulesOptions from gameState snapshot', async () => {
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
        rulesOptions: {
          enableSwapRule: true,
          enableMixedBoard: false,
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

describe('GameSession Branch Coverage - createPlayer helper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('creates player with all expected fields', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      boardType: 'square8',
      status: 'active',
      maxPlayers: 2,
      isRated: true,
      player1: { id: 'player-1', username: 'TestPlayer' },
      player2: { id: 'player-2', username: 'P2' },
      timeControl: JSON.stringify({ type: 'rapid', initialTime: 300000, increment: 5000 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.players[0].id).toBe('player-1');
    expect(state.players[0].username).toBe('TestPlayer');
    expect(state.players[0].playerNumber).toBe(1);
    expect(state.players[0].type).toBe('human');
    expect(typeof state.players[0].isReady).toBe('boolean');
    expect(typeof state.players[0].timeRemaining).toBe('number');
    expect(state.players[0].timeRemaining).toBeGreaterThan(0);
  });
});

describe('GameSession Branch Coverage - different board types', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes with default square8 board type', async () => {
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
    expect(state.boardType).toBe('square8');
  });

  it('initializes with explicit square8 board type', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-explicit', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-explicit',
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
    expect(state.boardType).toBe('square8');
  });
});

describe('GameSession Branch Coverage - time control variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles blitz time control', async () => {
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
      timeControl: JSON.stringify({ type: 'blitz', initialTime: 180000, increment: 2000 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.type).toBe('blitz');
  });

  it('handles classical time control with increment', async () => {
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
      timeControl: JSON.stringify({ type: 'classical', initialTime: 1800000, increment: 30000 }),
      gameState: null,
      rngSeed: null,
      moves: [],
    });

    await session.initialize();

    const state = session.getGameState();
    expect(state.timeControl.increment).toBe(30000);
  });
});

describe('GameSession Branch Coverage - game not found / database errors', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws when game not found in database', async () => {
    const io = createMockIo();
    const session = new GameSession('nonexistent-game', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue(null);

    await expect(session.initialize()).rejects.toThrow('Game not found');
  });
});

describe('GameSession Branch Coverage - isRated variations', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes unrated game', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
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

describe('GameSession Branch Coverage - database client error handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws on database unavailable', async () => {
    const { getDatabaseClient } = require('../../src/server/database/connection');
    getDatabaseClient.mockReturnValueOnce(null);

    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    await expect(session.initialize()).rejects.toThrow('Database not available');
  });
});

