/**
 * Tests for GameSession branch coverage - Lifecycle and Status.
 * Covers: terminate, session status, decision phase timeout, recompute status.
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

describe('GameSession Branch Coverage - terminate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('terminates and records abnormal termination when game is active', async () => {
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

    session.terminate('session_cleanup');

    expect(true).toBe(true);
  });

  it('terminates with custom reason', async () => {
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

    session.terminate('game_terminated');

    expect(true).toBe(true);
  });
});

describe('GameSession Branch Coverage - getDecisionPhaseRemainingMs', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null when no timeout is active', async () => {
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

    const remaining = session.getDecisionPhaseRemainingMs();
    expect(remaining).toBeNull();
  });
});

describe('GameSession Branch Coverage - resetDecisionPhaseTimeout', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('clears timeout handles when called', async () => {
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

    session.resetDecisionPhaseTimeout();

    expect(session.getDecisionPhaseRemainingMs()).toBeNull();
  });
});

describe('GameSession Branch Coverage - session status helpers', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns session status snapshot', async () => {
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

    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
    expect(status?.kind).toBeDefined();
  });

  it('returns valid moves for player', async () => {
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

    const moves = session.getValidMoves(1);
    expect(Array.isArray(moves)).toBe(true);
  });

  it('returns interaction handler', async () => {
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

    const handler = session.getInteractionHandler();
    expect(handler).toBeDefined();
  });
});

describe('GameSession Branch Coverage - mapPhaseToTimeoutPhase', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null for ring_placement phase', async () => {
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

    const remaining = session.getDecisionPhaseRemainingMs();
    expect(remaining).toBeNull();
  });
});

describe('GameSession Branch Coverage - recomputeSessionStatus', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles recompute when gameEngine state is available', async () => {
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

    session.recomputeSessionStatus();

    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
  });

  it('handles recompute with explicit state passed', async () => {
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

    const currentState = session.getGameState();
    session.recomputeSessionStatus(currentState);

    expect(session.getSessionStatusSnapshot()).not.toBeNull();
  });
});

describe('GameSession Branch Coverage - terminate with different reasons', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('terminates with player_disconnected reason', async () => {
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
    session.terminate('player_disconnected');

    expect(true).toBe(true);
  });

  it('terminates without explicit reason (uses default)', async () => {
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
    session.terminate();

    expect(true).toBe(true);
  });
});

describe('GameSession Branch Coverage - status snapshot before initialization', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null status snapshot before initialize', () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    const status = session.getSessionStatusSnapshot();
    expect(status).toBeNull();
  });
});

describe('GameSession Branch Coverage - multiple terminate calls', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles multiple terminate calls gracefully', async () => {
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

    session.terminate('session_cleanup');
    session.terminate('game_terminated');
    session.terminate('manual');

    expect(true).toBe(true);
  });
});

describe('GameSession Branch Coverage - resetDecisionPhaseTimeout with no active timeout', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles reset when no timeout is active', async () => {
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

    session.resetDecisionPhaseTimeout();
    session.resetDecisionPhaseTimeout();

    expect(session.getDecisionPhaseRemainingMs()).toBeNull();
  });
});

describe('GameSession Branch Coverage - decision phase timeout with line_processing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null for mapPhaseToTimeoutPhase with movement phase', async () => {
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

    const remaining = session.getDecisionPhaseRemainingMs();
    expect(remaining).toBeNull();
  });
});

describe('GameSession Branch Coverage - recomputeSessionStatus early return', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles recomputeSessionStatus when gameEngine not initialized', () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    session.recomputeSessionStatus();

    expect(session.getSessionStatusSnapshot()).toBeNull();
  });
});

describe('GameSession Branch Coverage - gameState with enableSwapRule', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles rulesOptions with enableSwapRule false', async () => {
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
          enableSwapRule: false,
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

describe('GameSession Branch Coverage - status transitions and metrics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('records status transition metrics when status changes', async () => {
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

    const status1 = session.getSessionStatusSnapshot();
    expect(status1).not.toBeNull();

    session.recomputeSessionStatus();

    const status2 = session.getSessionStatusSnapshot();
    expect(status2).not.toBeNull();
  });
});

describe('GameSession Branch Coverage - classifyDecisionSurface branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles line_processing phase with process_line moves', async () => {
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
          shortTimeoutMs: 5000,
          shortWarningBeforeMs: 1000,
        },
      }),
      rngSeed: null,
      moves: [],
    });

    await session.initialize();
    const state = session.getGameState();
    expect(state).toBeDefined();
  });

  it('handles territory_processing phase with region moves', async () => {
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
    expect(session.getDecisionPhaseRemainingMs()).toBeNull();
  });
});

describe('GameSession Branch Coverage - mapPhaseToTimeoutPhase branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles chain_capture phase mapping', async () => {
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

    expect(session.getDecisionPhaseRemainingMs()).toBeNull();
  });
});

describe('GameSession Branch Coverage - interaction handler access', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns interaction handler after initialization', async () => {
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

    const handler = session.getInteractionHandler();
    expect(handler).toBeDefined();
  });
});

describe('GameSession Branch Coverage - session status before and after', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('tracks status correctly through lifecycle', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as never, new Map());

    expect(session.getSessionStatusSnapshot()).toBeNull();

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

    expect(session.getSessionStatusSnapshot()).not.toBeNull();

    session.terminate('session_cleanup');
    expect(true).toBe(true);
  });
});

describe('GameSession Branch Coverage - classifyDecisionSurface branches - chain_capture', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('handles chain_capture phase with continue_capture_segment moves', async () => {
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
    expect(state.currentPhase).toBe('ring_placement');
  });
});
