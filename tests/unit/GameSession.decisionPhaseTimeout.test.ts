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
