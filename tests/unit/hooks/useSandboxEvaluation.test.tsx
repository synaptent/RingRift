import { renderHook, act, waitFor } from '@testing-library/react';
import { useSandboxEvaluation } from '../../../src/client/hooks/useSandboxEvaluation';
import type { ClientSandboxEngine } from '../../../src/client/services/ClientSandboxEngine';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('useSandboxEvaluation', () => {
  // Create mock engine
  const createMockEngine = (overrides: Partial<ClientSandboxEngine> = {}): ClientSandboxEngine =>
    ({
      getGameState: jest.fn().mockReturnValue({
        gameId: 'test-game-id',
        currentPlayer: 1,
        moveHistory: [{ type: 'place_ring' }, { type: 'move' }],
        gameStatus: 'active',
      }),
      getSerializedState: jest.fn().mockReturnValue({
        gameId: 'test-game-id',
        boardType: 'square8',
        players: [],
        board: { type: 'square8', cells: {} },
      }),
      ...overrides,
    }) as unknown as ClientSandboxEngine;

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('initial state', () => {
    it('returns correct default state with no engine', () => {
      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: null,
          developerToolsEnabled: false,
        })
      );

      expect(result.current.evaluationHistory).toEqual([]);
      expect(result.current.evaluationError).toBeNull();
      expect(result.current.isEvaluating).toBe(false);
      expect(typeof result.current.requestEvaluation).toBe('function');
      expect(typeof result.current.clearHistory).toBe('function');
    });

    it('returns correct default state with engine', () => {
      const mockEngine = createMockEngine();

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      expect(result.current.evaluationHistory).toEqual([]);
      expect(result.current.evaluationError).toBeNull();
      expect(result.current.isEvaluating).toBe(false);
    });
  });

  describe('requestEvaluation', () => {
    it('does nothing when engine is null', async () => {
      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: null,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(mockFetch).not.toHaveBeenCalled();
      expect(result.current.isEvaluating).toBe(false);
    });

    it('makes API call with serialized state', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            gameId: 'test-game-id',
            moveNumber: 2,
            playerNumber: 1,
            evaluation: 0.5,
            breakdown: { material: 0.3, position: 0.2 },
            timestamp: Date.now(),
          }),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/games/sandbox/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: expect.any(String),
      });

      // Verify serialized state was sent
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.state).toBeDefined();
      expect(callBody.state.gameId).toBe('test-game-id');
    });

    it('adds successful evaluation to history', async () => {
      const mockEngine = createMockEngine();
      const evalData = {
        gameId: 'test-game-id',
        moveNumber: 2,
        playerNumber: 1,
        evaluation: 0.5,
        breakdown: { material: 0.3 },
        timestamp: Date.now(),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(evalData),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      expect(result.current.evaluationHistory).toHaveLength(0);

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationHistory).toHaveLength(1);
      expect(result.current.evaluationHistory[0]).toEqual(evalData);
    });

    it('sets isEvaluating during request', async () => {
      const mockEngine = createMockEngine();
      let resolvePromise: (value: unknown) => void;
      const fetchPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockFetch.mockReturnValueOnce(fetchPromise);

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      expect(result.current.isEvaluating).toBe(false);

      // Start evaluation
      let evalPromise: Promise<void>;
      act(() => {
        evalPromise = result.current.requestEvaluation();
      });

      // Should be evaluating now
      expect(result.current.isEvaluating).toBe(true);

      // Resolve the fetch
      await act(async () => {
        resolvePromise!({
          ok: true,
          json: () => Promise.resolve({ evaluation: 0.5 }),
        });
        await evalPromise;
      });

      expect(result.current.isEvaluating).toBe(false);
    });
  });

  describe('error handling', () => {
    it('sets error message on 404 response', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Not found' }),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationError).toContain('AI evaluation is not enabled');
      expect(result.current.evaluationHistory).toHaveLength(0);
    });

    it('sets error message on 503 response', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: () => Promise.resolve({ error: 'Service unavailable' }),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationError).toContain('AI evaluation service is unavailable');
    });

    it('sets error message on network failure', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationError).toContain('Sandbox evaluation failed');
      expect(result.current.evaluationError).toContain('Network error');
    });

    it('handles non-JSON error response', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Not JSON')),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationError).toContain('HTTP 500');
    });
  });

  describe('clearHistory', () => {
    it('clears evaluation history', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            gameId: 'test',
            moveNumber: 1,
            playerNumber: 1,
            evaluation: 0.5,
            breakdown: {},
            timestamp: Date.now(),
          }),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      // Add some evaluations
      await act(async () => {
        await result.current.requestEvaluation();
      });
      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationHistory).toHaveLength(2);

      // Clear history
      act(() => {
        result.current.clearHistory();
      });

      expect(result.current.evaluationHistory).toHaveLength(0);
    });

    it('clears error when clearing history', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockRejectedValueOnce(new Error('Failed'));

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      // Trigger an error
      await act(async () => {
        await result.current.requestEvaluation();
      });

      expect(result.current.evaluationError).not.toBeNull();

      // Clear history should also clear error
      act(() => {
        result.current.clearHistory();
      });

      expect(result.current.evaluationError).toBeNull();
    });
  });

  describe('engine change', () => {
    it('clears history when engine changes', async () => {
      const mockEngine1 = createMockEngine();
      const mockEngine2 = createMockEngine();

      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            gameId: 'test',
            moveNumber: 1,
            playerNumber: 1,
            evaluation: 0.5,
            breakdown: {},
            timestamp: Date.now(),
          }),
      });

      const { result, rerender } = renderHook(
        ({ engine }) =>
          useSandboxEvaluation({
            engine,
            developerToolsEnabled: false,
          }),
        { initialProps: { engine: mockEngine1 } }
      );

      // Add an evaluation with first engine
      await act(async () => {
        await result.current.requestEvaluation();
      });

      // Verify evaluation was added
      expect(result.current.evaluationHistory.length).toBeGreaterThan(0);

      // Change engine
      rerender({ engine: mockEngine2 });

      // History should be cleared by the engine change effect
      expect(result.current.evaluationHistory).toHaveLength(0);
    });
  });

  describe('auto-evaluation', () => {
    it('does not auto-evaluate when developerToolsEnabled is false', async () => {
      const mockEngine = createMockEngine();

      renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: false,
        })
      );

      // Wait a bit to ensure no auto-evaluation happens
      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('auto-evaluates when developerToolsEnabled is true and moves exist', async () => {
      const mockEngine = createMockEngine();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            gameId: 'test',
            moveNumber: 2,
            playerNumber: 1,
            evaluation: 0.5,
            breakdown: {},
            timestamp: Date.now(),
          }),
      });

      const { result } = renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: true,
        })
      );

      // Wait for auto-evaluation effect to complete
      await waitFor(() => {
        expect(result.current.evaluationHistory.length).toBeGreaterThan(0);
      });

      expect(mockFetch).toHaveBeenCalled();
    });

    it('skips auto-evaluation in replay mode', async () => {
      const mockEngine = createMockEngine();

      renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: true,
          isInReplayMode: true,
        })
      );

      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('skips auto-evaluation in history viewing mode', async () => {
      const mockEngine = createMockEngine();

      renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: true,
          isViewingHistory: true,
        })
      );

      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('skips auto-evaluation when no moves in history', async () => {
      const mockEngine = createMockEngine({
        getGameState: jest.fn().mockReturnValue({
          gameId: 'test',
          currentPlayer: 1,
          moveHistory: [], // Empty move history
          gameStatus: 'active',
        }),
      });

      renderHook(() =>
        useSandboxEvaluation({
          engine: mockEngine,
          developerToolsEnabled: true,
        })
      );

      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('callback stability', () => {
    it('requestEvaluation and clearHistory have stable references', () => {
      // Use null engine to avoid auto-evaluation effect and engine change issues
      const { result, rerender } = renderHook(() =>
        useSandboxEvaluation({
          engine: null,
          developerToolsEnabled: false,
        })
      );

      // Ensure result.current is valid
      expect(result.current).not.toBeNull();

      const requestEvaluation1 = result.current.requestEvaluation;
      const clearHistory1 = result.current.clearHistory;

      rerender();

      // Callbacks should maintain reference stability
      expect(result.current.requestEvaluation).toBe(requestEvaluation1);
      expect(result.current.clearHistory).toBe(clearHistory1);
    });
  });
});
