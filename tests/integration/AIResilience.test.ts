/**
 * Integration tests for AI resilience and error handling
 * Tests the complete fallback chain from service to local to random
 */

import { AIEngine } from '../../src/server/game/ai/AIEngine';
import { GameSession } from '../../src/server/game/GameSession';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import { getDatabaseClient } from '../../src/server/database/connection';
import { logger } from '../../src/server/utils/logger';
import client from 'prom-client';

jest.mock('../../src/server/services/AIServiceClient');
jest.mock('../../src/server/database/connection');
jest.mock('../../src/server/utils/logger');
jest.setTimeout(30000); // 30 second timeout for integration tests

describe('AI Resilience Integration Tests', () => {
  let mockAIServiceClient: any;
  let mockPrisma: any;

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock AI service client
    mockAIServiceClient = {
      getAIMove: jest.fn(),
      healthCheck: jest.fn(),
      clearCache: jest.fn(),
      getLineRewardChoice: jest.fn(),
      getRingEliminationChoice: jest.fn(),
      getRegionOrderChoice: jest.fn(),
      getCircuitBreakerStatus: jest.fn(() => ({ isOpen: false, failureCount: 0 })),
    };

    (getAIServiceClient as jest.Mock).mockReturnValue(mockAIServiceClient);

    // Mock database client
    mockPrisma = {
      game: {
        findUnique: jest.fn(),
        update: jest.fn(),
      },
      move: {
        create: jest.fn(),
      },
      user: {
        findFirst: jest.fn(),
        create: jest.fn(),
      },
    };

    (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
  });

  describe('Service Degradation Scenarios', () => {
    it('should complete game even when AI service is completely down', async () => {
      // Simulate total service failure
      mockAIServiceClient.getAIMove.mockRejectedValue(
        new Error('ECONNREFUSED - AI service unreachable')
      );
      mockAIServiceClient.healthCheck.mockResolvedValue(false);

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      // Make multiple move requests - all should succeed via fallback
      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        board: {
          type: 'square8',
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
          size: 8,
        },
        players: [
          {
            playerNumber: 1,
            type: 'ai',
            ringsInHand: 10,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
        moveHistory: [],
        history: [],
        rngSeed: 12345,
      };

      for (let i = 0; i < 10; i++) {
        const move = await aiEngine.getAIMove(1, stateSnapshot);
        expect(move).toBeDefined();
      }

      // Verify fallback diagnostics
      const diag = aiEngine.getDiagnostics(1);
      expect(diag?.serviceFailureCount).toBe(10);
      expect(diag?.localFallbackCount).toBeGreaterThan(0);

      // Basic smoke check: AI move latency metrics should be registered and
      // visible in the Prometheus registry. We don't assert exact bucket
      // counts, only that the histogram family and a heuristic-labelled
      // bucket are present.
      const metrics = await client.register.metrics();
      expect(metrics).toContain('ai_move_latency_ms');
      // Label ordering in Prometheus output is not guaranteed, so assert that
      // there exists at least one heuristic-labelled bucket regardless of the
      // position of the aiType label within the label set.
      expect(metrics).toMatch(/ai_move_latency_ms_bucket\{[^}]*aiType="heuristic"[^}]*\}/);
    });

    it('should handle intermittent service failures gracefully', async () => {
      let callCount = 0;
      mockAIServiceClient.getAIMove.mockImplementation(() => {
        callCount++;
        // Fail every other call
        if (callCount % 2 === 0) {
          return Promise.reject(new Error('Intermittent failure'));
        }
        return Promise.resolve({
          move: {
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            placementCount: 1,
          },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        });
      });

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        board: {
          type: 'square8',
          stacks: new Map(),
          size: 8,
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
        },
        players: [{ playerNumber: 1, type: 'ai', ringsInHand: 10 }],
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 123,
      };

      // All requests should succeed despite intermittent failures
      for (let i = 0; i < 10; i++) {
        const move = await aiEngine.getAIMove(1, stateSnapshot);
        expect(move).toBeDefined();
      }

      const diag = aiEngine.getDiagnostics(1);
      expect(diag?.serviceFailureCount).toBeGreaterThan(0);
    });
  });

  describe('Circuit Breaker Integration', () => {
    it('should open circuit breaker after repeated failures', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));
      mockAIServiceClient.getCircuitBreakerStatus.mockReturnValue({
        isOpen: false,
        failureCount: 0,
      });

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        board: {
          type: 'square8',
          stacks: new Map(),
          size: 8,
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
        },
        players: [{ playerNumber: 1, type: 'ai', ringsInHand: 10 }],
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 123,
      };

      // Make multiple failing requests
      for (let i = 0; i < 6; i++) {
        await aiEngine.getAIMove(1, stateSnapshot);
      }

      // Circuit breaker should track failures
      // (Actual circuit breaker opening is tested in AIServiceClient tests)
      const diag = aiEngine.getDiagnostics(1);
      expect(diag?.serviceFailureCount).toBe(6);
    });
  });

  describe('Move Validation Integration', () => {
    it('should validate service moves against rule engine', async () => {
      // Service returns a move
      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: {
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          placementCount: 1,
        },
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        board: {
          type: 'square8',
          stacks: new Map(),
          size: 8,
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
        },
        players: [{ playerNumber: 1, type: 'ai', ringsInHand: 10 }],
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 123,
      };

      const move = await aiEngine.getAIMove(1, stateSnapshot);

      // Move should be validated and returned
      expect(move).toBeDefined();
      expect(move?.type).toBe('place_ring');
    });
  });

  describe('Error Recovery Patterns', () => {
    it('should recover from transient network errors', async () => {
      let attemptCount = 0;
      mockAIServiceClient.getAIMove.mockImplementation(() => {
        attemptCount++;
        if (attemptCount === 1) {
          return Promise.reject(new Error('ETIMEDOUT'));
        }
        return Promise.resolve({
          move: {
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            placementCount: 1,
          },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        });
      });

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        board: {
          type: 'square8',
          stacks: new Map(),
          size: 8,
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
        },
        players: [{ playerNumber: 1, type: 'ai', ringsInHand: 10 }],
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 123,
      };

      // First call fails, uses fallback
      const move1 = await aiEngine.getAIMove(1, stateSnapshot);
      expect(move1).toBeDefined();
      expect(attemptCount).toBe(1);

      // Second call should succeed via service
      const move2 = await aiEngine.getAIMove(1, stateSnapshot);
      expect(move2).toBeDefined();
      expect(attemptCount).toBe(2);

      const diag = aiEngine.getDiagnostics(1);
      expect(diag?.serviceFailureCount).toBe(1);
    });
  });

  describe('Performance Under Failure', () => {
    it('should maintain acceptable performance when falling back', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const aiEngine = new AIEngine();
      aiEngine.createAI(1, 5);

      const stateSnapshot: any = {
        id: 'test',
        boardType: 'square8',
        currentPlayer: 1,
        board: {
          type: 'square8',
          stacks: new Map(),
          size: 8,
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          pendingCaptureEvaluations: [],
          eliminatedRings: {},
        },
        players: [{ playerNumber: 1, type: 'ai', ringsInHand: 10 }],
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 123,
      };

      const startTime = Date.now();

      // Make 20 fallback moves
      for (let i = 0; i < 20; i++) {
        await aiEngine.getAIMove(1, stateSnapshot);
      }

      const duration = Date.now() - startTime;

      // Fallback moves should be fast (< 100ms each on average)
      expect(duration).toBeLessThan(2000);
    });
  });
});
