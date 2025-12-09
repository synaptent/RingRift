/**
 * AIServiceClient.recovery.test.ts
 *
 * Comprehensive tests for AI service recovery scenarios including:
 * - Service recovery detection via health check
 * - Switching back to remote AI after recovery
 * - Circuit breaker recovery (cooldown and half-open state)
 * - Graceful degradation during outages
 * - Game playability maintenance during outage
 *
 * Created as part of P18.7-3: AI Service Outage and Fallback Flow Coverage
 */

import { AIServiceClient } from '../../../src/server/services/AIServiceClient';
import { AIEngine } from '../../../src/server/game/ai/AIEngine';
import { getAIServiceClient } from '../../../src/server/services/AIServiceClient';
import { GameState, Move, AIProfile } from '../../../src/shared/types/game';
import { logger } from '../../../src/server/utils/logger';
import { getServiceStatusManager } from '../../../src/server/services/ServiceStatusManager';

// Mock axios

var mockPost: jest.Mock;

var mockGet: jest.Mock;

var mockCreate: jest.Mock;

jest.mock('axios', () => {
  mockPost = jest.fn();
  mockGet = jest.fn();
  mockCreate = jest.fn(() => ({
    post: mockPost,
    get: mockGet,
    delete: jest.fn(),
    interceptors: {
      response: {
        use: jest.fn(),
      },
    },
  }));

  return {
    __esModule: true,
    default: {
      create: mockCreate,
    },
    create: mockCreate,
  };
});

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('../../../src/server/services/ServiceStatusManager', () => ({
  getServiceStatusManager: jest.fn(() => ({
    updateServiceStatus: jest.fn(),
  })),
}));

jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: jest.fn(() => ({
    recordAIRequest: jest.fn(),
    recordAIRequestDuration: jest.fn(),
    recordAIRequestLatencyMs: jest.fn(),
    recordAIRequestTimeout: jest.fn(),
    recordAIFallback: jest.fn(),
    recordAIChoiceRequest: jest.fn(),
    recordAIChoiceLatencyMs: jest.fn(),
  })),
}));

// Mock rulesParityMetrics to avoid dependency on MetricsService initialization
jest.mock('../../../src/server/utils/rulesParityMetrics', () => ({
  aiMoveLatencyHistogram: {
    labels: () => ({
      observe: jest.fn(),
    }),
  },
  aiFallbackCounter: {
    labels: () => ({
      inc: jest.fn(),
    }),
  },
  rulesParityMetrics: {
    validMismatch: { inc: jest.fn() },
    hashMismatch: { inc: jest.fn() },
    sMismatch: { inc: jest.fn() },
    gameStatusMismatch: { inc: jest.fn() },
    moveParity: { inc: jest.fn() },
    parityCheckDuration: { observe: jest.fn() },
    parityCheckCounter: { inc: jest.fn() },
  },
}));

describe('AIServiceClient - Recovery Scenarios', () => {
  let client: AIServiceClient;
  let mockGameState: GameState;
  let mockStatusManager: any;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Reset internal counters between tests.
    (AIServiceClient as any).inFlightRequests = 0;
    (AIServiceClient as any).maxConcurrent = 16;

    mockStatusManager = {
      updateServiceStatus: jest.fn(),
    };
    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);

    client = new AIServiceClient('http://ai.test');

    mockGameState = {
      id: 'test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        pendingCaptureEvaluations: [],
        eliminatedRings: {} as any,
        size: 8,
      } as any,
      players: [
        {
          id: 'player1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
        },
      ] as any,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 } as any,
      spectators: [] as any,
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 123,
    };
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Service Recovery Detection', () => {
    it('should detect service recovery via health check', async () => {
      // First, health check fails
      mockGet.mockRejectedValueOnce(new Error('Connection refused'));

      const isHealthy1 = await client.healthCheck();
      expect(isHealthy1).toBe(false);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'unhealthy',
        expect.any(String),
        expect.any(Number)
      );

      mockStatusManager.updateServiceStatus.mockClear();

      // Then health check succeeds (recovery)
      mockGet.mockResolvedValueOnce({ data: { status: 'healthy' } });

      const isHealthy2 = await client.healthCheck();
      expect(isHealthy2).toBe(true);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });

    it('should switch back to remote AI after recovery', async () => {
      // First request fails
      mockPost.mockRejectedValueOnce(new Error('Service down'));

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toThrow();

      // Verify unhealthy status was recorded
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'unhealthy',
        expect.any(String),
        expect.any(Number)
      );

      mockStatusManager.updateServiceStatus.mockClear();

      // Second request succeeds (recovery)
      mockPost.mockResolvedValueOnce({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      const response = await client.getAIMove(mockGameState, 1, 5);
      expect(response.move).toBeDefined();

      // Verify healthy status was recorded (recovery)
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });
  });

  describe('Circuit Breaker Recovery', () => {
    it('should attempt recovery after cooldown period (half-open state)', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit breaker with 5 failures
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.getCircuitBreakerStatus().isOpen).toBe(true);

      // Clear mocks
      mockPost.mockClear();

      // Immediately, circuit should reject
      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toThrow(
        'Circuit breaker is open'
      );
      expect(mockPost).not.toHaveBeenCalled();

      // Advance time past the cooldown (60 seconds)
      jest.advanceTimersByTime(61000);

      // Circuit should now be in half-open state, allowing a retry
      mockPost.mockResolvedValueOnce({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      const response = await client.getAIMove(mockGameState, 1, 5);
      expect(response.move).toBeDefined();
      expect(mockPost).toHaveBeenCalled();

      // Circuit should be closed again after success
      expect(client.getCircuitBreakerStatus().isOpen).toBe(false);
      expect(client.getCircuitBreakerStatus().failureCount).toBe(0);
    });

    it('should log circuit breaker transition to half-open', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit breaker
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      // Clear log mocks
      (logger.info as jest.Mock).mockClear();

      // Advance time past cooldown
      jest.advanceTimersByTime(61000);

      // Try again (half-open state)
      mockPost.mockResolvedValueOnce({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      await client.getAIMove(mockGameState, 1, 5);

      // Should have logged the half-open transition
      expect(logger.info).toHaveBeenCalledWith(expect.stringContaining('half-open'));
    });

    it('should close circuit breaker on successful request after cooldown', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.getCircuitBreakerStatus().isOpen).toBe(true);

      // Advance time
      jest.advanceTimersByTime(61000);

      // Successful request
      mockPost.mockResolvedValueOnce({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      await client.getAIMove(mockGameState, 1, 5);

      // Verify circuit reset logged
      expect(logger.info).toHaveBeenCalledWith(
        expect.stringContaining('Circuit breaker reset'),
        expect.objectContaining({
          wasOpen: true,
        })
      );
    });

    it('should track failure in half-open state if recovery request fails', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.getCircuitBreakerStatus().isOpen).toBe(true);

      // Advance time past cooldown
      jest.advanceTimersByTime(61000);

      // Recovery request also fails
      mockPost.mockRejectedValueOnce(new Error('Still down'));

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toThrow();

      // The circuit will record the failure - after the first failure in half-open
      // the circuit records it, leading to a new failure count
      const status = client.getCircuitBreakerStatus();
      expect(status.failureCount).toBeGreaterThan(0);
    });
  });

  describe('Service Availability Tracking', () => {
    it('should report service unavailable when circuit is open', async () => {
      expect(client.isServiceAvailable()).toBe(true);

      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.isServiceAvailable()).toBe(false);
    });

    it('should report service available after cooldown', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.isServiceAvailable()).toBe(false);

      // Advance time past cooldown
      jest.advanceTimersByTime(61000);

      // Should report as available (half-open allows a try)
      expect(client.isServiceAvailable()).toBe(true);
    });
  });

  describe('Graceful Degradation', () => {
    it('should emit recovery event after health check succeeds', async () => {
      // Initially failing
      mockGet.mockRejectedValueOnce(new Error('Down'));
      await client.healthCheck();

      mockStatusManager.updateServiceStatus.mockClear();

      // Recovered
      mockGet.mockResolvedValueOnce({ data: { status: 'healthy' } });
      const isHealthy = await client.healthCheck();

      expect(isHealthy).toBe(true);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });

    it('should handle rapid recovery cycles', async () => {
      // Simulate rapid up/down cycles
      for (let cycle = 0; cycle < 3; cycle++) {
        // Fail
        mockGet.mockRejectedValueOnce(new Error('Down'));
        await client.healthCheck();
        expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
          'aiService',
          'unhealthy',
          expect.any(String),
          expect.any(Number)
        );

        // Recover
        mockGet.mockResolvedValueOnce({ data: { status: 'healthy' } });
        await client.healthCheck();
        expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
          'aiService',
          'healthy',
          undefined,
          expect.any(Number)
        );
      }
    });
  });
});

// Note: AIEngine recovery integration tests are covered in:
// - AIEngine.fallback.test.ts - comprehensive fallback scenario coverage
// - AIResilience.test.ts - integration tests for complete fallback chain
// We keep the AIServiceClient focused tests above for direct client behavior.
