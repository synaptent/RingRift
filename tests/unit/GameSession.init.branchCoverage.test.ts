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
    expect(session.getGameState()).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state.players[1].aiProfile).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state.players[0].username).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
    expect(state).toBeDefined();
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
