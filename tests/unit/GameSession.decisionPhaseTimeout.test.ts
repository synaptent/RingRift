import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import { config } from '../../src/server/config';

// Mock dependencies
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => null),
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

describe('GameSession Decision Phase Timeout Guards', () => {
  let mockIo: jest.Mocked<SocketIOServer>;
  let mockPythonRulesClient: any;
  let userSockets: Map<string, string>;

  beforeEach(() => {
    // Create mock Socket.IO server
    mockIo = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: {
          rooms: new Map(),
        },
        sockets: new Map(),
      },
    } as any;

    mockPythonRulesClient = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    userSockets = new Map();

    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  describe('Configuration', () => {
    it('should have decision phase timeout configuration', () => {
      expect(config.decisionPhaseTimeouts).toBeDefined();
      expect(config.decisionPhaseTimeouts.defaultTimeoutMs).toBe(30000);
      expect(config.decisionPhaseTimeouts.warningBeforeTimeoutMs).toBe(5000);
      expect(config.decisionPhaseTimeouts.extensionMs).toBe(15000);
    });
  });

  describe('Decision Phase Timeout Tracking', () => {
    it('should provide remaining time for decision phase', () => {
      // Create a minimal GameSession for testing
      const session = new GameSession('test-game-id', mockIo, mockPythonRulesClient, userSockets);

      // Initially should have no remaining time (no active timeout)
      expect(session.getDecisionPhaseRemainingMs()).toBeNull();
    });

    it('should allow resetting decision phase timeout', () => {
      const session = new GameSession('test-game-id', mockIo, mockPythonRulesClient, userSockets);

      // This should not throw even when no game is initialized
      expect(() => session.resetDecisionPhaseTimeout()).not.toThrow();
    });
  });

  describe('WebSocket Event Types', () => {
    it('should have decision_phase_timeout_warning event type', () => {
      // This tests that the TypeScript types are properly defined
      const warningPayload: import('../../src/shared/types/websocket').DecisionPhaseTimeoutWarningPayload =
        {
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'test-game',
            playerNumber: 1,
            phase: 'line_processing',
            remainingMs: 5000,
          },
          timestamp: new Date().toISOString(),
        };

      expect(warningPayload.type).toBe('decision_phase_timeout_warning');
      expect(warningPayload.data.phase).toBe('line_processing');
    });

    it('should have decision_phase_timed_out event type', () => {
      const timeoutPayload: import('../../src/shared/types/websocket').DecisionPhaseTimedOutPayload =
        {
          type: 'decision_phase_timed_out',
          data: {
            gameId: 'test-game',
            playerNumber: 1,
            phase: 'territory_processing',
            autoSelectedMoveId: 'move-123',
            reason: 'Decision timeout: auto-selected process_territory_region',
          },
          timestamp: new Date().toISOString(),
        };

      expect(timeoutPayload.type).toBe('decision_phase_timed_out');
      expect(timeoutPayload.data.phase).toBe('territory_processing');
    });

    it('should support chain_capture phase in timeout events', () => {
      const warningPayload: import('../../src/shared/types/websocket').DecisionPhaseTimeoutWarningPayload =
        {
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'test-game',
            playerNumber: 2,
            phase: 'chain_capture',
            remainingMs: 3000,
          },
          timestamp: new Date().toISOString(),
        };

      expect(warningPayload.data.phase).toBe('chain_capture');
    });
  });

  describe('Error Code Types', () => {
    it('should have DECISION_PHASE_TIMEOUT error code', () => {
      const errorCode: import('../../src/shared/types/websocket').WebSocketErrorCode =
        'DECISION_PHASE_TIMEOUT';
      expect(errorCode).toBe('DECISION_PHASE_TIMEOUT');
    });
  });
});

describe('GameSession decision phase runtime behaviour', () => {
  let mockIo: jest.Mocked<SocketIOServer>;

  beforeEach(() => {
    mockIo = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: {
          rooms: new Map(),
        },
        sockets: new Map(),
      },
    } as any;

    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('emits a decision_phase_timeout_warning event with remaining time', () => {
    const session = new GameSession('test-game-id', mockIo, {} as any, new Map());

    // Prime the internal timeout state to simulate a pending line-processing decision.
    const now = Date.now();
    (session as any).decisionTimeoutDeadlineMs = now + 10_000;
    (session as any).decisionTimeoutPhase = 'line_processing';
    (session as any).decisionTimeoutPlayer = 1;

    // Invoke the private helper directly.
    (session as any).emitDecisionPhaseTimeoutWarning();

    expect(mockIo.to).toHaveBeenCalledWith('test-game-id');
    expect(mockIo.emit).toHaveBeenCalledTimes(1);
    const [eventName, payload] = mockIo.emit.mock.calls[0];

    expect(eventName).toBe('decision_phase_timeout_warning');
    expect(payload.type).toBe('decision_phase_timeout_warning');
    expect(payload.data.gameId).toBe('test-game-id');
    expect(payload.data.playerNumber).toBe(1);
    expect(payload.data.phase).toBe('line_processing');
    // RemainingMs should be non-negative and at most the configured timeout.
    expect(typeof payload.data.remainingMs).toBe('number');
    expect(payload.data.remainingMs).toBeGreaterThanOrEqual(0);
  });

  it('auto-resolves a decision and emits decision_phase_timed_out with decisionAutoResolved meta', async () => {
    const session = new GameSession('test-game-id', mockIo, {} as any, new Map());

    // Prepare a minimal active GameState in line_processing for a human player.
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [
        {
          playerNumber: 1,
          type: 'human',
          id: 'p1',
        },
      ],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const decisionMove = {
      id: 'move-1',
      type: 'process_line' as const,
      player: 1,
    };

    // Stub out GameSession internals that handle rules and persistence.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [decisionMove]),
    };

    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };

    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    // Seed decision-timeout snapshot fields as if scheduleDecisionPhaseTimeout had run.
    (session as any).decisionTimeoutDeadlineMs = Date.now() + 10_000;
    (session as any).decisionTimeoutPhase = 'line_processing';
    (session as any).decisionTimeoutPlayer = 1;
    (session as any).decisionTimeoutChoiceType = 'line_order';
    (session as any).decisionTimeoutChoiceKind = 'line_order';

    await (session as any).handleDecisionPhaseTimedOut();

    // Rules engine should be invoked with the selected move.
    expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'move-1');

    // A decision_phase_timed_out event must be emitted with the selected move id.
    expect(mockIo.emit).toHaveBeenCalledWith(
      'decision_phase_timed_out',
      expect.objectContaining({
        type: 'decision_phase_timed_out',
        data: expect.objectContaining({
          gameId: 'test-game-id',
          playerNumber: 1,
          phase: 'line_processing',
          autoSelectedMoveId: 'move-1',
        }),
      })
    );

    // broadcastUpdate should be called with a decisionAutoResolved meta payload.
    expect((session as any).broadcastUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ success: true }),
      expect.objectContaining({
        choiceType: 'line_order',
        choiceKind: 'line_order',
        actingPlayerNumber: 1,
        resolvedMoveId: 'move-1',
        reason: 'timeout',
      })
    );
  });
});

describe('Decision Phase Timeout Integration', () => {
  /**
   * These tests verify the timeout behavior at a higher level.
   * They require a fully initialized GameSession with a database connection,
   * so they are marked as integration tests and may be skipped in unit test runs.
   */

  it.skip('should auto-resolve line_processing phase after timeout', async () => {
    // This test requires a full game setup with database
    // Implementation would:
    // 1. Create a game in line_processing phase
    // 2. Wait for timeout to expire
    // 3. Verify auto-resolution occurred
  });

  it.skip('should auto-resolve territory_processing phase after timeout', async () => {
    // This test requires a full game setup with database
  });

  it.skip('should auto-resolve chain_capture phase after timeout', async () => {
    // This test requires a full game setup with database
  });

  it.skip('should emit warning before timeout expires', async () => {
    // This test requires a full game setup with WebSocket mock
  });

  it.skip('should clear timeout when player makes a move', async () => {
    // This test requires a full game setup
  });

  it.skip('should reset timeout on player reconnection', async () => {
    // This test requires a full game setup
  });
});
