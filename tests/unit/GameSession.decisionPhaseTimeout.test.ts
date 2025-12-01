import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import { config } from '../../src/server/config';
import { withTimeControl, waitForConditionWithTimeAdvance } from '../helpers/TimeController';

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
   * These tests verify the timeout behavior using mocked internals.
   * They use the same approach as the unit tests above but exercise
   * more complex scenarios.
   */

  let mockIo: jest.Mocked<SocketIOServer>;
  let session: GameSession;

  beforeEach(() => {
    mockIo = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    session = new GameSession('test-game-id', mockIo, {} as any, new Map());
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('should auto-resolve line_processing phase after timeout', async () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const processLineMove = { id: 'line-1', type: 'process_line' as const, player: 1 };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [processLineMove]),
    };
    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };
    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    // Trigger scheduling (simulates entering line_processing phase)
    (session as any).scheduleDecisionPhaseTimeout(state);

    // Advance time past the timeout (30s default)
    await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

    // Verify auto-resolution
    expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'line-1');
    expect(mockIo.emit).toHaveBeenCalledWith(
      'decision_phase_timed_out',
      expect.objectContaining({
        type: 'decision_phase_timed_out',
        data: expect.objectContaining({
          phase: 'line_processing',
          autoSelectedMoveId: 'line-1',
        }),
      })
    );
  });

  it('should auto-resolve territory_processing phase after timeout', async () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 2,
      currentPhase: 'territory_processing',
      players: [{ playerNumber: 2, type: 'human', id: 'p2' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const regionMove = { id: 'region-1', type: 'process_territory_region' as const, player: 2 };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [regionMove]),
    };
    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };
    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    (session as any).scheduleDecisionPhaseTimeout(state);
    await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

    expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(2, 'region-1');
    expect(mockIo.emit).toHaveBeenCalledWith(
      'decision_phase_timed_out',
      expect.objectContaining({
        data: expect.objectContaining({
          phase: 'territory_processing',
          playerNumber: 2,
        }),
      })
    );
  });

  it('should auto-resolve chain_capture phase after timeout', async () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'chain_capture',
      chainCapturePosition: { x: 3, y: 3 },
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const chainMove = {
      id: 'chain-1',
      type: 'continue_capture_segment' as const,
      player: 1,
      from: { x: 3, y: 3 },
      captureTarget: { x: 4, y: 3 },
      to: { x: 5, y: 3 },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [chainMove]),
    };
    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };
    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    (session as any).scheduleDecisionPhaseTimeout(state);
    await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

    expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'chain-1');
    expect(mockIo.emit).toHaveBeenCalledWith(
      'decision_phase_timed_out',
      expect.objectContaining({
        data: expect.objectContaining({
          phase: 'chain_capture',
        }),
      })
    );
  });

  it('selects deterministic decision candidate ordering on timeout', async () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const regionMoveA = {
      id: 'region-A',
      type: 'process_territory_region' as const,
      player: 1,
      disconnectedRegions: [
        {
          spaces: [
            { x: 5, y: 5 },
            { x: 6, y: 6 },
          ],
        },
      ],
    };

    const regionMoveB = {
      id: 'region-B',
      type: 'process_territory_region' as const,
      player: 1,
      disconnectedRegions: [
        {
          spaces: [
            { x: 1, y: 1 },
            { x: 2, y: 2 },
          ],
        },
      ],
    };

    // Intentionally provide candidates in reverse-geometric order to verify
    // that the timeout handler uses its deterministic geometry-based ordering
    // rather than "first element" behaviour.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [regionMoveA, regionMoveB]),
    };
    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };
    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    (session as any).decisionTimeoutDeadlineMs = Date.now() + 10_000;
    (session as any).decisionTimeoutPhase = 'territory_processing';
    (session as any).decisionTimeoutPlayer = 1;
    (session as any).decisionTimeoutChoiceType = 'region_order';
    (session as any).decisionTimeoutChoiceKind = 'territory_region_order';

    await (session as any).handleDecisionPhaseTimedOut();

    // Geometry-based sort should choose the region anchored at (1,1)
    // (regionMoveB) even though regionMoveA appears first in getValidMoves().
    expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'region-B');
  });

  it('should emit warning before timeout expires', async () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const processLineMove = { id: 'line-1', type: 'process_line' as const, player: 1 };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [processLineMove]),
    };
    (session as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
    };
    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    (session as any).scheduleDecisionPhaseTimeout(state);

    // Warning is emitted at (timeout - warningBefore), i.e. 30s - 5s = 25s
    const warningTime =
      config.decisionPhaseTimeouts.defaultTimeoutMs -
      config.decisionPhaseTimeouts.warningBeforeTimeoutMs;

    // Advance to just before warning time
    await jest.advanceTimersByTimeAsync(warningTime - 100);
    expect(mockIo.emit).not.toHaveBeenCalledWith(
      'decision_phase_timeout_warning',
      expect.anything()
    );

    // Advance to after warning time but before timeout
    await jest.advanceTimersByTimeAsync(200);
    expect(mockIo.emit).toHaveBeenCalledWith(
      'decision_phase_timeout_warning',
      expect.objectContaining({
        type: 'decision_phase_timeout_warning',
        data: expect.objectContaining({
          gameId: 'test-game-id',
          playerNumber: 1,
          phase: 'line_processing',
        }),
      })
    );
  });

  it('should clear timeout when resetDecisionPhaseTimeout is called', () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const processLineMove = { id: 'line-1', type: 'process_line' as const, player: 1 };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [processLineMove]),
    };

    // Schedule timeout
    (session as any).scheduleDecisionPhaseTimeout(state);
    expect((session as any).decisionTimeoutDeadlineMs).not.toBeNull();

    // Clear timeout (simulates player making a move)
    session.resetDecisionPhaseTimeout();
    expect((session as any).decisionTimeoutDeadlineMs).toBeNull();
    expect((session as any).decisionTimeoutHandle).toBeNull();
    expect((session as any).decisionTimeoutWarningHandle).toBeNull();
  });

  it('should not schedule timeout for AI players', () => {
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [{ playerNumber: 1, type: 'ai', id: 'ai-1' }],
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
      },
      moveHistory: [],
    };

    const processLineMove = { id: 'line-1', type: 'process_line' as const, player: 1 };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => [processLineMove]),
    };

    (session as any).scheduleDecisionPhaseTimeout(state);

    // No timeout should be scheduled for AI players
    expect((session as any).decisionTimeoutDeadlineMs).toBeNull();
  });

  it('emits warning then auto-resolves line_processing timeout end-to-end while game remains active', async () => {
    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any as jest.Mocked<SocketIOServer>;

    const timedSession = new GameSession('line-timeout-game', io, {} as any, new Map());

    const baseState: any = {
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'line_processing',
      players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
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
      id: 'line-move-1',
      type: 'process_line' as const,
      player: 1,
    };

    const appliedState: any = {
      ...baseState,
      moveHistory: [
        {
          ...decisionMove,
          moveNumber: 1,
          timestamp: new Date(),
          thinkTime: 0,
        },
      ],
      gameStatus: 'active',
    };

    const getGameStateMock = jest
      .fn()
      // Guards and warning read the pre-timeout state
      .mockReturnValueOnce(baseState)
      // handleDecisionPhaseTimedOut reads the post-move state after applyMoveById
      .mockReturnValue(appliedState);

    (timedSession as any).gameEngine = {
      getGameState: getGameStateMock,
      getValidMoves: jest.fn(() => [decisionMove]),
    };

    (timedSession as any).rulesFacade = {
      applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: appliedState }),
    };

    jest.spyOn(timedSession as any, 'persistMove').mockResolvedValue(undefined);
    const broadcastSpy = jest
      .spyOn(timedSession as any, 'broadcastUpdate')
      .mockResolvedValue(undefined);
    jest.spyOn(timedSession as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    // Schedule timeout as the real session code would when entering line_processing.
    (timedSession as any).scheduleDecisionPhaseTimeout(baseState);

    const timeoutMs = config.decisionPhaseTimeouts.defaultTimeoutMs;
    const warningBeforeMs = config.decisionPhaseTimeouts.warningBeforeTimeoutMs;
    const warningTime = timeoutMs - warningBeforeMs;

    // Advance to just after the warning point.
    await jest.advanceTimersByTimeAsync(warningTime + 50);

    expect(io.emit).toHaveBeenCalledWith(
      'decision_phase_timeout_warning',
      expect.objectContaining({
        type: 'decision_phase_timeout_warning',
        data: expect.objectContaining({
          gameId: 'line-timeout-game',
          playerNumber: 1,
          phase: 'line_processing',
        }),
      })
    );

    // Advance to after the final timeout.
    await jest.advanceTimersByTimeAsync(timeoutMs - warningTime + 50);

    // Engine should have been invoked with the deterministically selected move id.
    expect((timedSession as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'line-move-1');

    // A decision_phase_timed_out event must be emitted with the same move id.
    expect(io.emit).toHaveBeenCalledWith(
      'decision_phase_timed_out',
      expect.objectContaining({
        type: 'decision_phase_timed_out',
        data: expect.objectContaining({
          gameId: 'line-timeout-game',
          playerNumber: 1,
          phase: 'line_processing',
          autoSelectedMoveId: 'line-move-1',
        }),
      })
    );

    // broadcastUpdate should carry the DecisionAutoResolvedMeta payload.
    expect(broadcastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ success: true }),
      expect.objectContaining({
        actingPlayerNumber: 1,
        resolvedMoveId: 'line-move-1',
        reason: 'timeout',
      })
    );

    // Game stays ACTIVE – decision-phase timeout only auto-resolves the move.
    expect(appliedState.gameStatus).toBe('active');
  });

  it('produces the same auto-resolved territory decision across repeated timeouts from the same state', async () => {
    const resolvedMoveIds = new Set<string>();

    const runOnce = async () => {
      const io = {
        to: jest.fn().mockReturnThis(),
        emit: jest.fn(),
        sockets: {
          adapter: { rooms: new Map() },
          sockets: new Map(),
        },
      } as any as jest.Mocked<SocketIOServer>;

      const sessionInstance = new GameSession('territory-timeout-game', io, {} as any, new Map());

      const baseState: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'territory_processing',
        players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
        board: {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
        },
        moveHistory: [],
      };

      const regionMoveA = {
        id: 'region-A',
        type: 'process_territory_region' as const,
        player: 1,
        disconnectedRegions: [
          {
            spaces: [
              { x: 5, y: 5 },
              { x: 6, y: 6 },
            ],
          },
        ],
      };

      const regionMoveB = {
        id: 'region-B',
        type: 'process_territory_region' as const,
        player: 1,
        disconnectedRegions: [
          {
            spaces: [
              { x: 1, y: 1 },
              { x: 2, y: 2 },
            ],
          },
        ],
      };

      const appliedState: any = {
        ...baseState,
        moveHistory: [
          {
            ...regionMoveB,
            moveNumber: 1,
            timestamp: new Date(),
            thinkTime: 0,
          },
        ],
      };

      const getGameStateMock = jest
        .fn()
        .mockReturnValueOnce(baseState)
        .mockReturnValue(appliedState);

      (sessionInstance as any).gameEngine = {
        getGameState: getGameStateMock,
        // Intentionally provide candidates in reverse geometry order – the
        // timeout handler should still deterministically select region-B.
        getValidMoves: jest.fn(() => [regionMoveA, regionMoveB]),
      };

      (sessionInstance as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: appliedState }),
      };

      jest.spyOn(sessionInstance as any, 'persistMove').mockResolvedValue(undefined);
      jest.spyOn(sessionInstance as any, 'broadcastUpdate').mockResolvedValue(undefined);
      jest.spyOn(sessionInstance as any, 'maybePerformAITurn').mockResolvedValue(undefined);

      (sessionInstance as any).scheduleDecisionPhaseTimeout(baseState);
      await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 50);

      // Geometry-based ordering inside handleDecisionPhaseTimedOut should always
      // pick region-B regardless of original getValidMoves ordering.
      expect((sessionInstance as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(
        1,
        'region-B'
      );
      resolvedMoveIds.add('region-B');
    };

    // Run the same timeout scenario multiple times from the same starting
    // state to prove that no hidden randomness affects the auto-selected move.
    for (let i = 0; i < 3; i += 1) {
      await runOnce();
      jest.clearAllTimers();
    }

    expect(resolvedMoveIds.size).toBe(1);
    expect([...resolvedMoveIds][0]).toBe('region-B');
  });
});

describe('Decision Phase Timeout with TimeController helper', () => {
  /**
   * This test demonstrates using the shared TimeController helper to
   * drive a GameSession decision-phase timeout with accelerated time,
   * mirroring the multiplayer decision-phase fixtures used in E2E tests.
   */
  it('auto-resolves a line_processing decision using accelerated virtual time', async () => {
    await withTimeControl(async (timeController) => {
      const io = {
        to: jest.fn().mockReturnThis(),
        emit: jest.fn(),
        sockets: {
          adapter: { rooms: new Map() },
          sockets: new Map(),
        },
      } as any as jest.Mocked<SocketIOServer>;

      const session = new GameSession('line-timeout-with-timecontroller', io, {} as any, new Map());

      const baseState: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'line_processing',
        players: [{ playerNumber: 1, type: 'human', id: 'p1' }],
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
        id: 'line-move-fast',
        type: 'process_line' as const,
        player: 1,
      };

      const appliedState: any = {
        ...baseState,
        moveHistory: [
          {
            ...decisionMove,
            moveNumber: 1,
            timestamp: new Date(),
            thinkTime: 0,
          },
        ],
        gameStatus: 'active',
      };

      const getGameStateMock = jest
        .fn()
        // scheduleDecisionPhaseTimeout / guards read base state
        .mockReturnValueOnce(baseState)
        // handleDecisionPhaseTimedOut sees applied state after rulesFacade.applyMoveById
        .mockReturnValue(appliedState);

      (session as any).gameEngine = {
        getGameState: getGameStateMock,
        getValidMoves: jest.fn(() => [decisionMove]),
      };

      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: appliedState }),
      };

      jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
      jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
      jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

      // Schedule the decision-phase timeout as production code does when entering line_processing.
      (session as any).scheduleDecisionPhaseTimeout(baseState);

      const timeoutMs = config.decisionPhaseTimeouts.defaultTimeoutMs;

      await waitForConditionWithTimeAdvance(
        timeController,
        () =>
          // Rules engine invoked with the deterministic move id
          (session as any).rulesFacade.applyMoveById.mock.calls.some(
            ([playerNumber, moveId]: [number, string]) =>
              playerNumber === 1 && moveId === 'line-move-fast'
          ) &&
          // And a matching decision_phase_timed_out event has been emitted
          io.emit.mock.calls.some(
            ([eventName, payload]: [string, any]) =>
              eventName === 'decision_phase_timed_out' &&
              payload?.data?.phase === 'line_processing' &&
              payload?.data?.autoSelectedMoveId === 'line-move-fast'
          ),
        {
          maxTime: timeoutMs + 1_000,
          stepSize: 5_000,
          description: 'line_processing decision timeout with TimeController',
        }
      );
    });
  });
});
