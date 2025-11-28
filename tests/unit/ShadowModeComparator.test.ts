/**
 * Unit tests for ShadowModeComparator
 *
 * Tests the shadow mode comparison logic for orchestrator rollout, including:
 * - Parallel engine execution
 * - Result comparison and difference detection
 * - Orchestrator error handling
 * - Metrics calculation
 * - Comparison storage limits
 *
 * @see docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md
 */

import {
  ShadowModeComparator,
  shadowComparator,
  MoveResult,
  ShadowComparison,
  ShadowMetrics,
} from '../../src/server/services/ShadowModeComparator';
import { logger } from '../../src/server/utils/logger';
import type { GameState, BoardState, Player, RingStack } from '../../src/shared/types/game';

// Mock the logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

/**
 * Helper to create a minimal valid GameState for testing
 */
function createMockGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultBoard: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const defaultPlayers: Player[] = [
    {
      id: 'player-1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      rating: 1500,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'player-2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      rating: 1500,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return {
    id: 'test-game-id',
    boardType: 'square8',
    board: defaultBoard,
    players: defaultPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 10, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: true,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
    ...overrides,
  };
}

/**
 * Helper to create a MoveResult for testing
 */
function createMockMoveResult(overrides: Partial<MoveResult> = {}): MoveResult {
  return {
    success: true,
    gameState: createMockGameState(),
    ...overrides,
  };
}

describe('ShadowModeComparator', () => {
  let comparator: ShadowModeComparator;

  beforeEach(() => {
    jest.clearAllMocks();
    comparator = new ShadowModeComparator();
  });

  describe('compare()', () => {
    describe('Engine Execution', () => {
      it('should run both engines in parallel via Promise.all', async () => {
        const executionOrder: string[] = [];

        const legacyEngine = jest.fn(async () => {
          executionOrder.push('legacy-start');
          await new Promise((resolve) => setTimeout(resolve, 10));
          executionOrder.push('legacy-end');
          return createMockMoveResult();
        });

        const orchestratorEngine = jest.fn(async () => {
          executionOrder.push('orchestrator-start');
          await new Promise((resolve) => setTimeout(resolve, 10));
          executionOrder.push('orchestrator-end');
          return createMockMoveResult();
        });

        await comparator.compare('session-1', 1, legacyEngine, orchestratorEngine);

        expect(legacyEngine).toHaveBeenCalledTimes(1);
        expect(orchestratorEngine).toHaveBeenCalledTimes(1);

        // Both should start before either ends (parallel execution)
        expect(executionOrder.indexOf('legacy-start')).toBeLessThan(
          executionOrder.indexOf('orchestrator-end')
        );
        expect(executionOrder.indexOf('orchestrator-start')).toBeLessThan(
          executionOrder.indexOf('legacy-end')
        );
      });

      it('should always return the legacy result (zero risk)', async () => {
        const legacyResult = createMockMoveResult({ success: true });
        const orchestratorResult = createMockMoveResult({ success: false, error: 'fail' });

        const { result } = await comparator.compare(
          'session-1',
          1,
          async () => legacyResult,
          async () => orchestratorResult
        );

        expect(result).toBe(legacyResult);
        expect(result.success).toBe(true);
      });

      it('should capture latency for both engines', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => {
            await new Promise((resolve) => setTimeout(resolve, 50));
            return createMockMoveResult();
          },
          async () => {
            await new Promise((resolve) => setTimeout(resolve, 30));
            return createMockMoveResult();
          }
        );

        expect(comparison.legacyLatencyMs).toBeGreaterThanOrEqual(50);
        expect(comparison.orchestratorLatencyMs).toBeGreaterThanOrEqual(30);
      });
    });

    describe('Error Handling', () => {
      it('should handle orchestrator errors gracefully', async () => {
        const orchestratorError = new Error('Orchestrator crashed');

        const { result, comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult(),
          async () => {
            throw orchestratorError;
          }
        );

        // Should still return legacy result
        expect(result.success).toBe(true);

        // Should track the error
        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContain('orchestrator_error');
        expect(comparison.orchestratorLatencyMs).toBe(-1);
      });

      it('should log orchestrator errors', async () => {
        const orchestratorError = new Error('Test error');

        await comparator.compare(
          'test-session',
          5,
          async () => createMockMoveResult(),
          async () => {
            throw orchestratorError;
          }
        );

        expect(logger.error).toHaveBeenCalledWith(
          expect.stringContaining('orchestrator engine error'),
          expect.objectContaining({
            sessionId: 'test-session',
            errorMessage: 'Test error',
          })
        );
      });

      it('should track orchestrator error count', async () => {
        // First error
        await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult(),
          async () => {
            throw new Error('Error 1');
          }
        );

        // Second error
        await comparator.compare(
          'session-2',
          1,
          async () => createMockMoveResult(),
          async () => {
            throw new Error('Error 2');
          }
        );

        expect(comparator.getOrchestratorErrorCount()).toBe(2);
      });
    });

    describe('Result Matching', () => {
      it('should detect matching results', async () => {
        const gameState = createMockGameState();
        const result = createMockMoveResult({ gameState });

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => result,
          async () => result
        );

        expect(comparison.isMatch).toBe(true);
        expect(comparison.differences).toHaveLength(0);
      });

      it('should detect success mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ success: true }),
          async () => createMockMoveResult({ success: false })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/success:/));
      });

      it('should detect currentPlayer mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ currentPlayer: 1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ currentPlayer: 2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/currentPlayer:/));
      });

      it('should detect phase mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({
              gameState: createMockGameState({ currentPhase: 'ring_placement' }),
            }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ currentPhase: 'movement' }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/phase:/));
      });

      it('should detect gameStatus mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ gameStatus: 'active' }) }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ gameStatus: 'completed' }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/gameStatus:/));
      });

      it('should detect winner mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ winner: 1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ winner: 2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/winner:/));
      });

      it('should detect totalRingsInPlay mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ totalRingsInPlay: 36 }) }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ totalRingsInPlay: 35 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/totalRingsInPlay:/));
      });

      it('should detect totalRingsEliminated mismatch', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ totalRingsEliminated: 0 }) }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ totalRingsEliminated: 2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(
          expect.stringMatching(/totalRingsEliminated:/)
        );
      });

      it('should detect missing gameState', async () => {
        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState() }),
          async () => createMockMoveResult({ gameState: undefined })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/gameState:/));
      });
    });

    describe('Board State Comparison', () => {
      it('should detect stack count differences', async () => {
        const board1: BoardState = {
          stacks: new Map([
            [
              '0,0',
              {
                position: { x: 0, y: 0 },
                rings: [1],
                stackHeight: 1,
                capHeight: 1,
                controllingPlayer: 1,
              },
            ],
          ]),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const board2: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/stack at 0,0:/));
      });

      it('should detect stack content differences', async () => {
        const stack1: RingStack = {
          position: { x: 0, y: 0 },
          rings: [1, 2],
          stackHeight: 2,
          capHeight: 1,
          controllingPlayer: 1,
        };

        const stack2: RingStack = {
          position: { x: 0, y: 0 },
          rings: [2, 1],
          stackHeight: 2,
          capHeight: 1,
          controllingPlayer: 2,
        };

        const board1: BoardState = {
          stacks: new Map([['0,0', stack1]]),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const board2: BoardState = {
          stacks: new Map([['0,0', stack2]]),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences.some((d) => d.includes('controller:'))).toBe(true);
        expect(comparison.differences.some((d) => d.includes('rings:'))).toBe(true);
      });

      it('should detect marker count differences', async () => {
        const board1: BoardState = {
          stacks: new Map(),
          markers: new Map([['1,1', { player: 1, position: { x: 1, y: 1 }, type: 'regular' }]]),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const board2: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/markerCount:/));
      });

      it('should detect collapsed spaces differences', async () => {
        const board1: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map([['2,2', 1]]),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const board2: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8',
        };

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(
          expect.stringMatching(/collapsedSpacesCount:/)
        );
      });

      it('should detect eliminated rings per player differences', async () => {
        const board1: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: { 1: 5, 2: 3 },
          size: 8,
          type: 'square8',
        };

        const board2: BoardState = {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: { 1: 4, 2: 4 },
          size: 8,
          type: 'square8',
        };

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board1 }) }),
          async () => createMockMoveResult({ gameState: createMockGameState({ board: board2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences.some((d) => d.includes('eliminatedRings['))).toBe(true);
      });
    });

    describe('Player State Comparison', () => {
      it('should detect ringsInHand differences', async () => {
        const players1: Player[] = [
          {
            id: '1',
            username: 'P1',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 15,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: '2',
            username: 'P2',
            type: 'human',
            playerNumber: 2,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ];

        const players2: Player[] = [
          {
            id: '1',
            username: 'P1',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 14,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: '2',
            username: 'P2',
            type: 'human',
            playerNumber: 2,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ];

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ players: players1 }) }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ players: players2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(
          expect.stringMatching(/player\[1\]\.ringsInHand:/)
        );
      });

      it('should detect territorySpaces differences', async () => {
        const players1: Player[] = [
          {
            id: '1',
            username: 'P1',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 5,
          },
        ];

        const players2: Player[] = [
          {
            id: '1',
            username: 'P1',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600000,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 10,
          },
        ];

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ players: players1 }) }),
          async () =>
            createMockMoveResult({ gameState: createMockGameState({ players: players2 }) })
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(
          expect.stringMatching(/player\[1\]\.territorySpaces:/)
        );
      });
    });

    describe('Game Result Comparison', () => {
      it('should detect gameResult winner differences', async () => {
        const result1 = createMockMoveResult({
          gameResult: {
            winner: 1,
            reason: 'ring_elimination',
            finalScore: { ringsEliminated: {}, territorySpaces: {}, ringsRemaining: {} },
          },
        });

        const result2 = createMockMoveResult({
          gameResult: {
            winner: 2,
            reason: 'ring_elimination',
            finalScore: { ringsEliminated: {}, territorySpaces: {}, ringsRemaining: {} },
          },
        });

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => result1,
          async () => result2
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/gameResult\.winner:/));
      });

      it('should detect gameResult reason differences', async () => {
        const result1 = createMockMoveResult({
          gameResult: {
            winner: 1,
            reason: 'ring_elimination',
            finalScore: { ringsEliminated: {}, territorySpaces: {}, ringsRemaining: {} },
          },
        });

        const result2 = createMockMoveResult({
          gameResult: {
            winner: 1,
            reason: 'territory_control',
            finalScore: { ringsEliminated: {}, territorySpaces: {}, ringsRemaining: {} },
          },
        });

        const { comparison } = await comparator.compare(
          'session-1',
          1,
          async () => result1,
          async () => result2
        );

        expect(comparison.isMatch).toBe(false);
        expect(comparison.differences).toContainEqual(expect.stringMatching(/gameResult\.reason:/));
      });
    });

    describe('Logging', () => {
      it('should log debug message for matching results', async () => {
        const result = createMockMoveResult();

        await comparator.compare(
          'session-123',
          5,
          async () => result,
          async () => result
        );

        expect(logger.debug).toHaveBeenCalledWith(
          expect.stringContaining('engines match'),
          expect.objectContaining({
            sessionId: 'session-123',
            moveNumber: 5,
          })
        );
      });

      it('should log warn message for mismatched results', async () => {
        await comparator.compare(
          'session-456',
          10,
          async () => createMockMoveResult({ success: true }),
          async () => createMockMoveResult({ success: false })
        );

        expect(logger.warn).toHaveBeenCalledWith(
          expect.stringContaining('ENGINE MISMATCH'),
          expect.objectContaining({
            sessionId: 'session-456',
            moveNumber: 10,
            differenceCount: expect.any(Number),
          })
        );
      });
    });
  });

  describe('Comparison Storage', () => {
    it('should store comparisons', async () => {
      await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      const allComparisons = comparator.getAllComparisons();
      expect(allComparisons).toHaveLength(1);
    });

    it('should respect maximum storage limit', async () => {
      const smallComparator = new ShadowModeComparator(5);

      // Add more comparisons than limit
      for (let i = 0; i < 10; i++) {
        await smallComparator.compare(
          `session-${i}`,
          i,
          async () => createMockMoveResult(),
          async () => createMockMoveResult()
        );
      }

      const allComparisons = smallComparator.getAllComparisons();
      expect(allComparisons).toHaveLength(5);

      // Should have removed oldest comparisons (0-4) and kept newest (5-9)
      expect(allComparisons[0].sessionId).toBe('session-5');
      expect(allComparisons[4].sessionId).toBe('session-9');
    });

    it('should clear comparisons', async () => {
      await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      comparator.clearComparisons();

      expect(comparator.getAllComparisons()).toHaveLength(0);
      expect(comparator.getOrchestratorErrorCount()).toBe(0);
    });
  });

  describe('getMetrics()', () => {
    it('should calculate metrics correctly', async () => {
      // Add matching comparison
      await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      // Add mismatching comparison
      await comparator.compare(
        'session-2',
        1,
        async () => createMockMoveResult({ success: true }),
        async () => createMockMoveResult({ success: false })
      );

      // Add orchestrator error
      await comparator.compare(
        'session-3',
        1,
        async () => createMockMoveResult(),
        async () => {
          throw new Error('Test error');
        }
      );

      const metrics = comparator.getMetrics();

      expect(metrics.totalComparisons).toBe(3);
      expect(metrics.matches).toBe(1);
      expect(metrics.mismatches).toBe(2); // 1 mismatch + 1 error
      expect(metrics.mismatchRate).toBeCloseTo(2 / 3, 2);
      expect(metrics.orchestratorErrors).toBe(1);
      expect(metrics.orchestratorErrorRate).toBeCloseTo(1 / 3, 2);
    });

    it('should calculate average latencies correctly', async () => {
      // Mock deterministic latencies
      await comparator.compare(
        'session-1',
        1,
        async () => {
          await new Promise((r) => setTimeout(r, 20));
          return createMockMoveResult();
        },
        async () => {
          await new Promise((r) => setTimeout(r, 10));
          return createMockMoveResult();
        }
      );

      const metrics = comparator.getMetrics();

      // Latencies should be non-zero
      expect(metrics.avgLegacyLatencyMs).toBeGreaterThan(0);
      expect(metrics.avgOrchestratorLatencyMs).toBeGreaterThan(0);
    });

    it('should handle empty comparisons', () => {
      const metrics = comparator.getMetrics();

      expect(metrics.totalComparisons).toBe(0);
      expect(metrics.matches).toBe(0);
      expect(metrics.mismatches).toBe(0);
      expect(metrics.mismatchRate).toBe(0);
      expect(metrics.orchestratorErrors).toBe(0);
      expect(metrics.orchestratorErrorRate).toBe(0);
      expect(metrics.avgLegacyLatencyMs).toBe(0);
      expect(metrics.avgOrchestratorLatencyMs).toBe(0);
    });

    it('should exclude invalid orchestrator latencies from average', async () => {
      // Add successful comparison
      await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      // Add orchestrator error (latency = -1)
      await comparator.compare(
        'session-2',
        1,
        async () => createMockMoveResult(),
        async () => {
          throw new Error('Test error');
        }
      );

      const metrics = comparator.getMetrics();

      // Average should only include the one valid orchestrator latency
      expect(metrics.avgOrchestratorLatencyMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe('getRecentMismatches()', () => {
    it('should return only mismatches', async () => {
      // Add match
      await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      // Add mismatch
      await comparator.compare(
        'session-2',
        2,
        async () => createMockMoveResult({ success: true }),
        async () => createMockMoveResult({ success: false })
      );

      const mismatches = comparator.getRecentMismatches();

      expect(mismatches).toHaveLength(1);
      expect(mismatches[0].sessionId).toBe('session-2');
    });

    it('should respect limit parameter', async () => {
      // Add multiple mismatches
      for (let i = 0; i < 20; i++) {
        await comparator.compare(
          `session-${i}`,
          i,
          async () => createMockMoveResult({ success: true }),
          async () => createMockMoveResult({ success: false })
        );
      }

      const limitedMismatches = comparator.getRecentMismatches(5);

      expect(limitedMismatches).toHaveLength(5);
    });

    it('should return most recent mismatches', async () => {
      // Add mismatches
      for (let i = 0; i < 10; i++) {
        await comparator.compare(
          `session-${i}`,
          i,
          async () => createMockMoveResult({ success: true }),
          async () => createMockMoveResult({ success: false })
        );
      }

      const mismatches = comparator.getRecentMismatches(3);

      // Should be the last 3
      expect(mismatches[0].sessionId).toBe('session-7');
      expect(mismatches[1].sessionId).toBe('session-8');
      expect(mismatches[2].sessionId).toBe('session-9');
    });

    it('should default to 10 if no limit provided', async () => {
      // Add many mismatches
      for (let i = 0; i < 15; i++) {
        await comparator.compare(
          `session-${i}`,
          i,
          async () => createMockMoveResult({ success: true }),
          async () => createMockMoveResult({ success: false })
        );
      }

      const mismatches = comparator.getRecentMismatches();

      expect(mismatches).toHaveLength(10);
    });
  });

  describe('Interface Types', () => {
    it('should return ShadowComparison with all required fields', async () => {
      const { comparison } = await comparator.compare(
        'session-test',
        42,
        async () => createMockMoveResult(),
        async () => createMockMoveResult()
      );

      expect(comparison).toHaveProperty('sessionId');
      expect(comparison).toHaveProperty('moveNumber');
      expect(comparison).toHaveProperty('legacyResult');
      expect(comparison).toHaveProperty('orchestratorResult');
      expect(comparison).toHaveProperty('isMatch');
      expect(comparison).toHaveProperty('differences');
      expect(comparison).toHaveProperty('legacyLatencyMs');
      expect(comparison).toHaveProperty('orchestratorLatencyMs');

      expect(comparison.sessionId).toBe('session-test');
      expect(comparison.moveNumber).toBe(42);
      expect(typeof comparison.isMatch).toBe('boolean');
      expect(Array.isArray(comparison.differences)).toBe(true);
      expect(typeof comparison.legacyLatencyMs).toBe('number');
      expect(typeof comparison.orchestratorLatencyMs).toBe('number');
    });

    it('should return ShadowMetrics with all required fields', () => {
      const metrics = comparator.getMetrics();

      expect(metrics).toHaveProperty('totalComparisons');
      expect(metrics).toHaveProperty('matches');
      expect(metrics).toHaveProperty('mismatches');
      expect(metrics).toHaveProperty('mismatchRate');
      expect(metrics).toHaveProperty('orchestratorErrors');
      expect(metrics).toHaveProperty('orchestratorErrorRate');
      expect(metrics).toHaveProperty('avgLegacyLatencyMs');
      expect(metrics).toHaveProperty('avgOrchestratorLatencyMs');

      expect(typeof metrics.totalComparisons).toBe('number');
      expect(typeof metrics.matches).toBe('number');
      expect(typeof metrics.mismatches).toBe('number');
      expect(typeof metrics.mismatchRate).toBe('number');
      expect(typeof metrics.orchestratorErrors).toBe('number');
      expect(typeof metrics.orchestratorErrorRate).toBe('number');
      expect(typeof metrics.avgLegacyLatencyMs).toBe('number');
      expect(typeof metrics.avgOrchestratorLatencyMs).toBe('number');
    });
  });

  describe('Edge Cases', () => {
    it('should handle both results having no gameState', async () => {
      const { comparison } = await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult({ gameState: undefined }),
        async () => createMockMoveResult({ gameState: undefined })
      );

      // No gameState in either = match (on that field)
      expect(comparison.differences).not.toContainEqual(expect.stringMatching(/gameState:/));
    });

    it('should handle null orchestrator result from error', async () => {
      const { comparison } = await comparator.compare(
        'session-1',
        1,
        async () => createMockMoveResult(),
        async () => {
          throw new Error('Crash');
        }
      );

      expect(comparison.orchestratorResult).toBeNull();
      expect(comparison.differences).toContain('orchestrator_error');
    });

    it('should handle concurrent comparisons', async () => {
      const promises = [];
      for (let i = 0; i < 50; i++) {
        promises.push(
          comparator.compare(
            `session-${i}`,
            i,
            async () => createMockMoveResult(),
            async () => createMockMoveResult()
          )
        );
      }

      await Promise.all(promises);

      const allComparisons = comparator.getAllComparisons();
      expect(allComparisons).toHaveLength(50);
    });
  });
});

describe('Singleton Export', () => {
  it('should export a singleton instance', () => {
    expect(shadowComparator).toBeDefined();
    expect(shadowComparator).toBeInstanceOf(ShadowModeComparator);
  });

  it('should be the same instance on multiple imports', async () => {
    const { shadowComparator: instance1 } = await import(
      '../../src/server/services/ShadowModeComparator'
    );
    const { shadowComparator: instance2 } = await import(
      '../../src/server/services/ShadowModeComparator'
    );

    expect(instance1).toBe(instance2);
  });
});
