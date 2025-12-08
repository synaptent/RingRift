import { renderHook, act } from '@testing-library/react';
import {
  useSandboxScenarios,
  type ScenarioData,
  type LoadedScenario,
  extractChainCapturePath,
} from '../../../src/client/hooks/useSandboxScenarios';
import type { ClientSandboxEngine } from '../../../src/client/services/ClientSandboxEngine';
import type { GameState } from '../../../src/shared/types/game';

// Mock toast
jest.mock('react-hot-toast', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

describe('useSandboxScenarios', () => {
  // Create mock engine factory
  const createMockEngine = (): ClientSandboxEngine =>
    ({
      getGameState: jest.fn().mockReturnValue({
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [{ number: 1 }, { number: 2 }],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [],
      }),
    }) as unknown as ClientSandboxEngine;

  // Create mock scenario factory
  const createMockScenario = (overrides: Partial<ScenarioData> = {}): ScenarioData => ({
    id: 'test-scenario-id',
    name: 'Test Scenario',
    description: 'A test scenario',
    gameState: {
      gameId: 'scenario-game-id',
      board: { type: 'square8', cells: {} },
      players: [{ number: 1 }, { number: 2 }],
      currentPlayer: 1,
      currentPhase: 'placement',
      moveHistory: [],
    } as GameState,
    onboarding: false,
    source: 'builtin',
    ...overrides,
  });

  const mockInitSandboxWithScenario = jest.fn();
  const mockOnScenarioLoaded = jest.fn();
  const mockOnStateVersionChange = jest.fn();
  const mockOnUIStateReset = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockInitSandboxWithScenario.mockReset();
    mockOnScenarioLoaded.mockReset();
    mockOnStateVersionChange.mockReset();
    mockOnUIStateReset.mockReset();
  });

  describe('initial state', () => {
    it('returns correct default state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.lastLoadedScenario).toBeNull();
      expect(result.current.showScenarioPicker).toBe(false);
      expect(result.current.showSelfPlayBrowser).toBe(false);
      expect(result.current.isInReplayMode).toBe(false);
      expect(result.current.replayState).toBeNull();
      expect(result.current.replayAnimation).toBeNull();
      expect(result.current.isViewingHistory).toBe(false);
      expect(result.current.historyViewIndex).toBe(0);
      expect(result.current.hasHistorySnapshots).toBe(true);
    });

    it('provides all expected functions', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(typeof result.current.setShowScenarioPicker).toBe('function');
      expect(typeof result.current.setShowSelfPlayBrowser).toBe('function');
      expect(typeof result.current.setIsInReplayMode).toBe('function');
      expect(typeof result.current.setReplayState).toBe('function');
      expect(typeof result.current.setReplayAnimation).toBe('function');
      expect(typeof result.current.setIsViewingHistory).toBe('function');
      expect(typeof result.current.setHistoryViewIndex).toBe('function');
      expect(typeof result.current.setHasHistorySnapshots).toBe('function');
      expect(typeof result.current.handleLoadScenario).toBe('function');
      expect(typeof result.current.handleForkFromReplay).toBe('function');
      expect(typeof result.current.handleResetScenario).toBe('function');
      expect(typeof result.current.clearScenarioContext).toBe('function');
    });
  });

  describe('setters', () => {
    it('setShowScenarioPicker updates state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.showScenarioPicker).toBe(false);

      act(() => {
        result.current.setShowScenarioPicker(true);
      });

      expect(result.current.showScenarioPicker).toBe(true);
    });

    it('setShowSelfPlayBrowser updates state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.showSelfPlayBrowser).toBe(false);

      act(() => {
        result.current.setShowSelfPlayBrowser(true);
      });

      expect(result.current.showSelfPlayBrowser).toBe(true);
    });

    it('setIsInReplayMode updates state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.isInReplayMode).toBe(false);

      act(() => {
        result.current.setIsInReplayMode(true);
      });

      expect(result.current.isInReplayMode).toBe(true);
    });

    it('setHistoryViewIndex updates state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.historyViewIndex).toBe(0);

      act(() => {
        result.current.setHistoryViewIndex(5);
      });

      expect(result.current.historyViewIndex).toBe(5);
    });

    it('setIsViewingHistory updates state', () => {
      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.isViewingHistory).toBe(false);

      act(() => {
        result.current.setIsViewingHistory(true);
      });

      expect(result.current.isViewingHistory).toBe(true);
    });
  });

  describe('handleLoadScenario', () => {
    it('calls initSandboxWithScenario', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      const scenario = createMockScenario();

      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(mockInitSandboxWithScenario).toHaveBeenCalledWith(scenario);
    });

    it('calls onUIStateReset when provided', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onUIStateReset: mockOnUIStateReset,
        })
      );

      const scenario = createMockScenario();

      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(mockOnUIStateReset).toHaveBeenCalled();
    });

    it('calls onScenarioLoaded with scenario metadata', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onScenarioLoaded: mockOnScenarioLoaded,
        })
      );

      const scenario = createMockScenario({
        id: 'scenario-123',
        name: 'Test Scenario',
        description: 'Description',
        onboarding: true,
        rulesConcept: 'capture',
        source: 'builtin',
      });

      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(mockOnScenarioLoaded).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'scenario-123',
          name: 'Test Scenario',
          description: 'Description',
          onboarding: true,
          rulesConcept: 'capture',
          source: 'builtin',
        })
      );
    });

    it('calls onStateVersionChange', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onStateVersionChange: mockOnStateVersionChange,
        })
      );

      act(() => {
        result.current.handleLoadScenario(createMockScenario());
      });

      expect(mockOnStateVersionChange).toHaveBeenCalled();
    });

    it('closes scenario picker and self-play browser', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // Open both pickers
      act(() => {
        result.current.setShowScenarioPicker(true);
        result.current.setShowSelfPlayBrowser(true);
      });

      expect(result.current.showScenarioPicker).toBe(true);
      expect(result.current.showSelfPlayBrowser).toBe(true);

      // Load scenario
      act(() => {
        result.current.handleLoadScenario(createMockScenario());
      });

      expect(result.current.showScenarioPicker).toBe(false);
      expect(result.current.showSelfPlayBrowser).toBe(false);
    });

    it('resets history state', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // Set some history state
      act(() => {
        result.current.setIsViewingHistory(true);
        result.current.setHistoryViewIndex(10);
        result.current.setHasHistorySnapshots(false);
        result.current.setIsInReplayMode(true);
      });

      // Load scenario
      act(() => {
        result.current.handleLoadScenario(createMockScenario());
      });

      expect(result.current.isViewingHistory).toBe(false);
      expect(result.current.historyViewIndex).toBe(0);
      expect(result.current.hasHistorySnapshots).toBe(true);
      expect(result.current.isInReplayMode).toBe(false);
      expect(result.current.replayState).toBeNull();
    });

    it('sets lastLoadedScenario', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      expect(result.current.lastLoadedScenario).toBeNull();

      const scenario = createMockScenario({
        id: 'my-scenario',
        name: 'My Scenario',
      });

      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(result.current.lastLoadedScenario).toEqual(
        expect.objectContaining({
          id: 'my-scenario',
          name: 'My Scenario',
        })
      );
    });

    it('does not proceed if initSandboxWithScenario returns null', () => {
      mockInitSandboxWithScenario.mockReturnValue(null);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onScenarioLoaded: mockOnScenarioLoaded,
        })
      );

      act(() => {
        result.current.handleLoadScenario(createMockScenario());
      });

      expect(mockOnScenarioLoaded).not.toHaveBeenCalled();
      expect(result.current.lastLoadedScenario).toBeNull();
    });

    it('stores scenario in originalScenarioRef for reset', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      const scenario = createMockScenario({ id: 'stored-scenario' });

      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(result.current.originalScenarioRef.current).toEqual(scenario);
    });
  });

  describe('handleForkFromReplay', () => {
    it('calls initSandboxWithScenario with fork scenario', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      const forkState = {
        gameId: 'forked-game',
        board: { type: 'square8', cells: {} },
        players: [{ number: 1 }, { number: 2 }],
        currentPlayer: 1,
        currentPhase: 'movement',
        moveHistory: [{ type: 'place_ring' }],
      } as GameState;

      act(() => {
        result.current.handleForkFromReplay(forkState, 5);
      });

      expect(mockInitSandboxWithScenario).toHaveBeenCalledWith(
        expect.objectContaining({
          id: expect.stringContaining('fork-'),
          name: 'Fork from move 5',
          source: 'fork',
        })
      );
    });

    it('calls onUIStateReset when provided', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onUIStateReset: mockOnUIStateReset,
        })
      );

      const forkState = {
        gameId: 'forked-game',
      } as GameState;

      act(() => {
        result.current.handleForkFromReplay(forkState, 3);
      });

      expect(mockOnUIStateReset).toHaveBeenCalled();
    });

    it('clears scenario context (lastLoadedScenario and originalScenarioRef)', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // First load a scenario
      act(() => {
        result.current.handleLoadScenario(createMockScenario());
      });

      expect(result.current.lastLoadedScenario).not.toBeNull();
      expect(result.current.originalScenarioRef.current).not.toBeNull();

      // Then fork
      act(() => {
        result.current.handleForkFromReplay({ gameId: 'fork' } as GameState, 1);
      });

      expect(result.current.lastLoadedScenario).toBeNull();
      expect(result.current.originalScenarioRef.current).toBeNull();
    });

    it('exits replay mode', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // Enter replay mode
      act(() => {
        result.current.setIsInReplayMode(true);
        result.current.setReplayState({ gameId: 'replay' } as GameState);
      });

      expect(result.current.isInReplayMode).toBe(true);

      // Fork
      act(() => {
        result.current.handleForkFromReplay({ gameId: 'fork' } as GameState, 1);
      });

      expect(result.current.isInReplayMode).toBe(false);
      expect(result.current.replayState).toBeNull();
    });

    it('calls onStateVersionChange', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
          onStateVersionChange: mockOnStateVersionChange,
        })
      );

      act(() => {
        result.current.handleForkFromReplay({ gameId: 'fork' } as GameState, 1);
      });

      expect(mockOnStateVersionChange).toHaveBeenCalled();
    });
  });

  describe('handleResetScenario', () => {
    it('re-loads the original scenario', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      const scenario = createMockScenario({ id: 'original-scenario' });

      // Load initial scenario
      act(() => {
        result.current.handleLoadScenario(scenario);
      });

      expect(mockInitSandboxWithScenario).toHaveBeenCalledTimes(1);

      // Reset scenario
      act(() => {
        result.current.handleResetScenario();
      });

      // Should call initSandboxWithScenario again with same scenario
      expect(mockInitSandboxWithScenario).toHaveBeenCalledTimes(2);
      expect(mockInitSandboxWithScenario).toHaveBeenLastCalledWith(scenario);
    });

    it('shows error toast if no scenario to reset', () => {
      const { toast } = jest.requireMock('react-hot-toast') as {
        toast: { error: jest.Mock };
      };

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // No scenario loaded, try reset
      act(() => {
        result.current.handleResetScenario();
      });

      expect(toast.error).toHaveBeenCalledWith('No scenario to reset');
    });
  });

  describe('clearScenarioContext', () => {
    it('clears all scenario-related state', () => {
      const mockEngine = createMockEngine();
      mockInitSandboxWithScenario.mockReturnValue(mockEngine);

      const { result } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      // Set up various state
      act(() => {
        result.current.handleLoadScenario(createMockScenario());
        result.current.setIsViewingHistory(true);
        result.current.setHistoryViewIndex(5);
        result.current.setHasHistorySnapshots(false);
        result.current.setIsInReplayMode(true);
        result.current.setReplayState({ gameId: 'replay' } as GameState);
      });

      // Clear context
      act(() => {
        result.current.clearScenarioContext();
      });

      expect(result.current.lastLoadedScenario).toBeNull();
      expect(result.current.originalScenarioRef.current).toBeNull();
      expect(result.current.isViewingHistory).toBe(false);
      expect(result.current.historyViewIndex).toBe(0);
      expect(result.current.hasHistorySnapshots).toBe(true);
      expect(result.current.isInReplayMode).toBe(false);
      expect(result.current.replayState).toBeNull();
    });
  });

  describe('callback stability', () => {
    it('handlers have stable references', () => {
      const { result, rerender } = renderHook(() =>
        useSandboxScenarios({
          initSandboxWithScenario: mockInitSandboxWithScenario,
        })
      );

      const handleLoadScenario1 = result.current.handleLoadScenario;
      const handleForkFromReplay1 = result.current.handleForkFromReplay;
      const handleResetScenario1 = result.current.handleResetScenario;
      const clearScenarioContext1 = result.current.clearScenarioContext;

      rerender();

      // These should have stable references due to useCallback
      expect(result.current.handleLoadScenario).toBe(handleLoadScenario1);
      expect(result.current.handleForkFromReplay).toBe(handleForkFromReplay1);
      expect(result.current.handleResetScenario).toBe(handleResetScenario1);
      expect(result.current.clearScenarioContext).toBe(clearScenarioContext1);
    });
  });
});

describe('extractChainCapturePath', () => {
  it('returns undefined when gameState is null', () => {
    expect(extractChainCapturePath(null)).toBeUndefined();
  });

  it('returns undefined when not in chain_capture phase', () => {
    const gameState = {
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
    } as GameState;

    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });

  it('returns undefined when moveHistory is empty', () => {
    const gameState = {
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [],
    } as GameState;

    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });

  it('extracts chain capture path from move history', () => {
    const gameState = {
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        {
          type: 'overtaking_capture',
          player: 1,
          from: { q: 0, r: 0 },
          to: { q: 2, r: 0 },
        },
        {
          type: 'continue_capture_segment',
          player: 1,
          to: { q: 4, r: 0 },
        },
      ],
    } as unknown as GameState;

    const path = extractChainCapturePath(gameState);

    expect(path).toBeDefined();
    expect(path).toHaveLength(3);
    expect(path![0]).toEqual({ q: 0, r: 0 });
    expect(path![1]).toEqual({ q: 2, r: 0 });
    expect(path![2]).toEqual({ q: 4, r: 0 });
  });

  it('stops at moves by different player', () => {
    const gameState = {
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        {
          type: 'move',
          player: 2,
          from: { q: -2, r: 0 },
          to: { q: -1, r: 0 },
        },
        {
          type: 'overtaking_capture',
          player: 1,
          from: { q: 0, r: 0 },
          to: { q: 2, r: 0 },
        },
      ],
    } as unknown as GameState;

    const path = extractChainCapturePath(gameState);

    expect(path).toBeDefined();
    expect(path).toHaveLength(2);
    expect(path![0]).toEqual({ q: 0, r: 0 });
    expect(path![1]).toEqual({ q: 2, r: 0 });
  });

  it('returns undefined when path has less than 2 positions', () => {
    const gameState = {
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        {
          type: 'move',
          player: 1,
          from: { q: 0, r: 0 },
          to: { q: 1, r: 0 },
        },
      ],
    } as unknown as GameState;

    // move type stops the walk, so path will be empty
    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });
});
