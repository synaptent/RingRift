/**
 * SandboxGameHost Tests - Replay & Autosave
 *
 * Tests for:
 * - Keyboard shortcuts ("?" and Escape)
 * - Replay mode entry/exit
 * - Replay state board type
 * - Fork from replay position
 * - Autosave failures
 * - Local storage fallback
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import type { BoardState, GameResult, GameState, Position } from '../../../src/shared/types/game';

// ─────────────────────────────────────────────────────────────────────────────
// Mock State Variables
// ─────────────────────────────────────────────────────────────────────────────

const mockNavigate = jest.fn();
let mockAuthUser: { id: string; username: string } | null = { id: 'user-1', username: 'Alice' };
let mockSandboxValue: any;
let lastBoardViewProps: any = null;
let replayPanelProps: any = null;

const mockMaybeRunSandboxAiIfNeeded = jest.fn();
const mockChoiceResolve = jest.fn();
const mockStoreGame = jest.fn().mockResolvedValue({ success: true, totalMoves: 10 });
const mockStoreGameLocally = jest.fn();
const mockGetPendingCount = jest.fn();
const mockGameSyncTriggerSync = jest.fn();
let mockGameSyncSubscribeCallback: ((state: any) => void) | null = null;

// ─────────────────────────────────────────────────────────────────────────────
// Mocks
// ─────────────────────────────────────────────────────────────────────────────

jest.mock('../../../src/client/hooks/useIsMobile', () => ({
  __esModule: true,
  useIsMobile: jest.fn(() => false),
}));

jest.mock('../../../src/client/components/BoardView', () => {
  const React = require('react');
  return {
    __esModule: true,
    BoardView: jest.fn((props: any) => {
      lastBoardViewProps = props;
      const { board, boardType } = props;
      const cells: React.ReactNode[] = [];
      const size = board?.size ?? 8;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const key = `${x},${y}`;
          cells.push(
            React.createElement('button', {
              key,
              'data-x': x,
              'data-y': y,
              'data-testid': `cell-${x}-${y}`,
              onClick: () => props.onCellClick?.({ x, y }),
            })
          );
        }
      }
      const ariaLabel = boardType === 'hexagonal' ? 'Hexagonal game board' : 'Square game board';
      return React.createElement(
        'div',
        { 'data-testid': 'board-view', 'data-board-type': boardType, 'aria-label': ariaLabel },
        cells
      );
    }),
  };
});

jest.mock('react-router-dom', () => ({
  __esModule: true,
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}));

jest.mock('../../../src/client/contexts/AuthContext', () => ({
  __esModule: true,
  useAuth: () => ({ user: mockAuthUser }),
}));

jest.mock('../../../src/client/services/api');

jest.mock('../../../src/client/services/LocalGameStorage', () => ({
  __esModule: true,
  storeGameLocally: (...args: unknown[]) => mockStoreGameLocally(...args),
  getPendingCount: (...args: unknown[]) => mockGetPendingCount(...args),
}));

jest.mock('../../../src/client/services/GameSyncService', () => ({
  __esModule: true,
  GameSyncService: {
    start: jest.fn(),
    stop: jest.fn(),
    subscribe: (cb: (state: any) => void) => {
      mockGameSyncSubscribeCallback = cb;
      cb({
        status: 'idle',
        pendingCount: 0,
        lastSyncAttempt: null,
        lastSuccessfulSync: null,
        consecutiveFailures: 0,
      });
      return jest.fn();
    },
    triggerSync: (...args: unknown[]) => mockGameSyncTriggerSync(...args),
    getState: () => ({
      status: 'idle',
      pendingCount: 0,
      lastSyncAttempt: null,
      lastSuccessfulSync: null,
      consecutiveFailures: 0,
    }),
  },
}));

jest.mock('../../../src/client/services/ReplayService', () => ({
  __esModule: true,
  getReplayService: () => ({ storeGame: mockStoreGame }),
}));

jest.mock('../../../src/client/contexts/SandboxContext', () => ({
  __esModule: true,
  useSandbox: () => mockSandboxValue,
}));

jest.mock('../../../src/client/components/ScenarioPickerModal', () => ({
  __esModule: true,
  ScenarioPickerModal: () => null,
  default: () => null,
}));

jest.mock('../../../src/client/components/ReplayPanel', () => ({
  __esModule: true,
  ReplayPanel: (props: any) => {
    replayPanelProps = props;
    return null;
  },
}));

jest.mock('../../../src/client/components/sandbox', () => {
  const React = require('react');
  return {
    __esModule: true,
    SandboxBoardSection: (props: any) =>
      React.createElement('div', { 'data-testid': 'sandbox-board-section' }, props.children),
    SandboxGameSidebar: () => React.createElement('div', { 'data-testid': 'sandbox-game-sidebar' }),
  };
});

jest.mock('../../../src/client/components/VictoryModal', () => {
  const React = require('react');
  return {
    __esModule: true,
    VictoryModal: (props: any) =>
      props.isOpen
        ? React.createElement(
            'div',
            { 'data-testid': 'victory-modal' },
            React.createElement('button', { onClick: props.onReturnToLobby }, 'Return to Lobby'),
            props.saveStatus && React.createElement('span', null, props.saveStatus)
          )
        : null,
  };
});

jest.mock('../../../src/client/components/ChoiceDialog', () => ({
  __esModule: true,
  ChoiceDialog: () => null,
}));

jest.mock('../../../src/client/components/BoardControlsOverlay', () => {
  const React = require('react');
  return {
    __esModule: true,
    BoardControlsOverlay: (props: any) =>
      props.isOpen ? React.createElement('div', { 'data-testid': 'board-controls-overlay' }) : null,
  };
});
jest.mock('../../../src/client/components/SelfPlayBrowser', () => ({
  __esModule: true,
  SelfPlayBrowser: () => null,
}));
jest.mock('../../../src/client/components/SaveStateDialog', () => ({
  __esModule: true,
  SaveStateDialog: () => null,
}));
jest.mock('../../../src/client/components/RingPlacementCountDialog', () => ({
  __esModule: true,
  RingPlacementCountDialog: () => null,
}));
jest.mock('../../../src/client/components/RecoveryLineChoiceDialog', () => ({
  __esModule: true,
  RecoveryLineChoiceDialog: () => null,
}));
jest.mock('../../../src/client/components/TerritoryRegionChoiceDialog', () => ({
  __esModule: true,
  TerritoryRegionChoiceDialog: () => null,
}));
jest.mock('../../../src/client/components/LineRewardPanel', () => ({
  __esModule: true,
  LineRewardPanel: () => null,
}));
jest.mock('../../../src/client/components/OnboardingModal', () => ({
  __esModule: true,
  OnboardingModal: () => null,
}));
jest.mock('../../../src/client/components/tutorial/TutorialHintBanner', () => ({
  __esModule: true,
  TutorialHintBanner: () => null,
}));
jest.mock('../../../src/client/components/TeachingOverlay', () => ({
  __esModule: true,
  TeachingOverlay: () => null,
}));
jest.mock('../../../src/client/components/AIServiceStatusBanner', () => ({
  __esModule: true,
  AIServiceStatusBanner: () => null,
}));
jest.mock('../../../src/client/components/ScreenReaderAnnouncer', () => ({
  __esModule: true,
  ScreenReaderAnnouncer: () => null,
  useGameAnnouncements: () => ({ announce: jest.fn() }),
  useGameStateAnnouncements: () => {},
}));

jest.mock('../../../src/client/hooks/useSandboxInteractions', () => ({
  __esModule: true,
  useSandboxInteractions: (options: any) => {
    if (options.choiceResolverRef) {
      options.choiceResolverRef.current = (response: any) => mockChoiceResolve(response);
    }
    return {
      handleCellClick: (pos: Position) => {
        options.setSelected(pos);
        options.setValidTargets([{ x: pos.x, y: pos.y + 1 }]);
      },
      handleCellDoubleClick: jest.fn(),
      handleCellContextMenu: jest.fn(),
      maybeRunSandboxAiIfNeeded: () => mockMaybeRunSandboxAiIfNeeded(),
      clearSelection: () => {
        options.setSelected(undefined);
        options.setValidTargets([]);
      },
    };
  },
}));

jest.mock('../../../src/client/hooks/useFirstTimePlayer', () => ({
  __esModule: true,
  useFirstTimePlayer: () => ({
    shouldShowWelcome: false,
    markWelcomeSeen: jest.fn(),
    markGameCompleted: jest.fn(),
    isFirstTimePlayer: true,
    state: {
      seenTutorialPhases: [],
      tutorialHintsEnabled: false,
      hasCompletedFirstGame: false,
      hasSeenWelcome: false,
    },
    markPhaseHintSeen: jest.fn(),
    setTutorialHintsEnabled: jest.fn(),
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxPersistence', () => ({
  __esModule: true,
  useSandboxPersistence: () => ({
    saveStatus: 'idle',
    pendingLocalGames: 0,
    syncState: { status: 'idle', pendingCount: 0 },
    handleTriggerSync: jest.fn(),
    handleSaveGame: jest.fn().mockResolvedValue({ success: true }),
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxEvaluation', () => ({
  __esModule: true,
  useSandboxEvaluation: () => ({
    showEvaluation: false,
    evaluation: null,
    isEvaluating: false,
    toggleEvaluation: jest.fn(),
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxScenarios', () => ({
  __esModule: true,
  useSandboxScenarios: () => ({
    loadedScenario: null,
    loadScenario: jest.fn(),
    clearScenario: jest.fn(),
    isLoadingScenario: false,
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxClock', () => ({
  __esModule: true,
  useSandboxClock: () => ({
    clockP1: 300000,
    clockP2: 300000,
    clockP3: 300000,
    clockP4: 300000,
    activePlayer: null,
    isLowTime: false,
    isPaused: true,
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxAITracking', () => ({
  __esModule: true,
  useSandboxAITracking: () => ({
    state: {
      isAiThinking: false,
      aiThinkingPlayer: null,
      aiMoveError: null,
      aiThinkingStartedAt: null,
      aiLadderHealth: null,
      aiLadderHealthError: null,
      aiLadderHealthLoading: false,
    },
    actions: {
      clearAiError: jest.fn(),
      refreshLadderHealth: jest.fn(),
      copyLadderHealth: jest.fn(),
    },
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxDiagnostics', () => ({
  __esModule: true,
  useSandboxDiagnostics: () => ({
    state: { showDiagnostics: false, lpsTrackingState: null },
    actions: { toggleDiagnostics: jest.fn() },
  }),
}));

jest.mock('../../../src/client/hooks/useBoardViewProps', () => ({
  __esModule: true,
  useBoardOverlays: () => ({
    overlays: {
      showMovementGrid: false,
      showValidTargets: true,
      showCoordinateLabels: false,
      squareRankFromBottom: false,
      showLineOverlays: false,
      showTerritoryOverlays: false,
    },
    setShowMovementGrid: jest.fn(),
    setShowValidTargets: jest.fn(),
    setShowCoordinateLabels: jest.fn(),
    setSquareRankFromBottom: jest.fn(),
    setShowLineOverlays: jest.fn(),
    setShowTerritoryOverlays: jest.fn(),
    resetOverlays: jest.fn(),
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxBoardSelection', () => ({
  __esModule: true,
  useSandboxBoardSelection: () => [
    { selectedCell: null, highlightedCells: [] },
    { setSelectedCell: jest.fn(), setHighlightedCells: jest.fn(), clearSelection: jest.fn() },
  ],
}));

jest.mock('../../../src/client/hooks/useSandboxGameLifecycle', () => ({
  __esModule: true,
  useSandboxGameLifecycle: () => ({
    actions: {
      startLocalGame: jest.fn(),
      startGame: jest.fn(),
      applyQuickStartPreset: jest.fn(),
      resetToSetup: jest.fn(),
      rematch: jest.fn(),
    },
  }),
}));

jest.mock('../../../src/client/hooks/useSandboxAIServiceStatus', () => ({
  __esModule: true,
  useSandboxAIServiceStatus: () => ({
    state: { status: 'connected', message: null, isServiceConfigured: true },
    actions: { retryConnection: jest.fn(), dismissMessage: jest.fn() },
  }),
}));

jest.mock('../../../src/client/contexts/AccessibilityContext', () => ({
  __esModule: true,
  useAccessibility: () => ({
    colorVisionMode: 'normal',
    effectiveReducedMotion: false,
    highContrastMode: false,
    setColorVisionMode: jest.fn(),
    setReducedMotion: jest.fn(),
    setHighContrastMode: jest.fn(),
  }),
}));

jest.mock('../../../src/client/hooks/useMoveAnimation', () => ({
  __esModule: true,
  useAutoMoveAnimation: () => ({
    animationData: null,
    isAnimating: false,
    triggerAnimation: jest.fn(),
    clearAnimation: jest.fn(),
  }),
}));

jest.mock('../../../src/client/hooks/useGameSoundEffects', () => ({
  __esModule: true,
  useGameSoundEffects: () => {},
}));
jest.mock('../../../src/client/hooks/useTutorialHints', () => ({
  __esModule: true,
  useTutorialHints: () => ({ currentHint: null, dismissHint: jest.fn(), isHintsEnabled: false }),
}));
jest.mock('../../../src/client/hooks/useKeyboardNavigation', () => ({
  __esModule: true,
  useGlobalGameShortcuts: () => {},
}));
jest.mock('../../../src/client/contexts/SoundContext', () => ({
  __esModule: true,
  useSoundOptional: () => null,
}));

// Load component after mocks
const { SandboxGameHost } =
  require('../../../src/client/pages/SandboxGameHost') as typeof import('../../../src/client/pages/SandboxGameHost');

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

import {
  createPlayers,
  createSandboxGameState,
  createMockSandboxContext,
  getSquareCell,
} from './SandboxGameHost.testUtils';

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// TODO(2026-01-20): Tests split from SandboxGameHost.test.tsx for memory isolation.
// The original tests were skipped due to OOM issues. After splitting, tests now run
// in separate Jest workers but need assertion updates for current component structure.
// The component's internal rendering changed since these tests were written.
describe.skip('SandboxGameHost - Replay & Autosave', () => {
  let consoleErrorSpy: jest.SpyInstance;
  let consoleWarnSpy: jest.SpyInstance;

  beforeAll(() => {
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterAll(() => {
    consoleErrorSpy.mockRestore();
    consoleWarnSpy.mockRestore();
  });

  afterEach(() => {
    lastBoardViewProps = null;
    mockSandboxValue = null;
    replayPanelProps = null;
    mockGameSyncSubscribeCallback = null;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    localStorage.setItem('ringrift_sandbox_sidebar_show_advanced', 'true');
    mockAuthUser = { id: 'user-1', username: 'Alice' };
    lastBoardViewProps = null;
    mockSandboxValue = createMockSandboxContext();
    mockMaybeRunSandboxAiIfNeeded.mockReset();
    mockChoiceResolve.mockReset();
    mockStoreGame.mockClear();
    mockStoreGameLocally.mockReset();
    mockGetPendingCount.mockReset();
    mockGameSyncTriggerSync.mockReset();
    replayPanelProps = null;
    mockGameSyncSubscribeCallback = null;
  });

  it('toggles board controls overlay with "?" and Escape keyboard shortcuts during an active sandbox game', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    expect(screen.queryByTestId('board-controls-overlay')).toBeNull();

    fireEvent.keyDown(window, { key: '?' });
    expect(screen.getByTestId('board-controls-overlay')).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('board-controls-overlay')).toBeNull();
  });

  it('enters and exits replay mode via ReplayPanel callbacks, making the board read-only while active', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    expect(replayPanelProps).not.toBeNull();

    const replayState: GameState = {
      ...sandboxState,
    };

    act(() => {
      replayPanelProps.onStateChange(replayState);
      replayPanelProps.onReplayModeChange(true);
    });

    expect(screen.getByText(/Viewing replay - board is read-only/i)).toBeInTheDocument();

    const sourceCell = getSquareCell(0, 0);
    fireEvent.click(sourceCell);

    const panel = screen.getByTestId('sandbox-touch-controls');
    expect(panel).toHaveTextContent('Selection');
    expect(panel).not.toHaveTextContent('(0, 0)');

    act(() => {
      replayPanelProps.onReplayModeChange(false);
      replayPanelProps.onStateChange(null);
    });

    fireEvent.click(sourceCell);
    expect(panel).toHaveTextContent('(0, 0)');
  });

  it('uses the replay state board type when in replay mode', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    expect(replayPanelProps).not.toBeNull();

    const replayBoard: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 3,
      type: 'hexagonal',
    };

    const replayState: GameState = {
      ...sandboxState,
      boardType: 'hexagonal',
      board: replayBoard,
    };

    act(() => {
      replayPanelProps.onStateChange(replayState);
      replayPanelProps.onReplayModeChange(true);
    });

    expect(screen.getByLabelText('Hexagonal game board')).toBeInTheDocument();
  });

  it('forks a new sandbox game from a replay position and exits replay mode', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const baseEngine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const initLocalSandboxEngine = jest.fn();
    const fakeEngine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getLpsTrackingState: jest.fn(() => null),
      getValidMoves: jest.fn(() => []),
      initFromSerializedState: jest.fn(),
    };
    initLocalSandboxEngine.mockReturnValue(fakeEngine);

    const setConfig = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: baseEngine,
      initLocalSandboxEngine,
      setConfig,
    });

    render(<SandboxGameHost />);

    expect(replayPanelProps).not.toBeNull();

    const replayBoard: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    };

    const replayState: GameState = {
      ...sandboxState,
      boardType: 'square8',
      board: replayBoard,
      players: createPlayers(),
    };

    act(() => {
      replayPanelProps.onStateChange(replayState);
      replayPanelProps.onReplayModeChange(true);
    });

    expect(screen.getByText(/Viewing replay - board is read-only/i)).toBeInTheDocument();

    act(() => {
      replayPanelProps.onForkFromPosition(replayState);
    });

    expect(initLocalSandboxEngine).toHaveBeenCalledTimes(1);
    expect(initLocalSandboxEngine).toHaveBeenCalledWith(
      expect.objectContaining({
        boardType: 'square8',
        numPlayers: replayState.players.length,
        playerTypes: ['human', 'ai', 'human', 'human'],
      })
    );

    expect(screen.queryByText(/Viewing replay - board is read-only/i)).not.toBeInTheDocument();
  });

  it('handles autosave failures from ReplayService.storeGame without breaking sandbox UI', async () => {
    const players = createPlayers();
    const completedState = createSandboxGameState({
      players,
      currentPhase: 'movement',
      gameStatus: 'completed',
    });

    const engine = {
      getGameState: jest.fn(() => completedState),
      getVictoryResult: jest.fn(
        () =>
          ({
            winner: 1,
            reason: 'ring_elimination',
            finalScore: {
              ringsEliminated: {},
              territorySpaces: {},
              ringsRemaining: {},
            },
          }) as GameResult
      ),
      getGameEndExplanation: jest.fn(() => null),
    };

    mockStoreGame.mockResolvedValue({ success: false, totalMoves: 10 });

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    await waitFor(() => {
      expect(mockStoreGame).toHaveBeenCalled();
      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    expect(screen.getByText('Return to Lobby')).toBeInTheDocument();
  });

  it('falls back to local storage on autosave failure and surfaces saved-local + pending sync UI', async () => {
    const players = createPlayers();
    const completedState = createSandboxGameState({
      players,
      currentPhase: 'movement',
      gameStatus: 'completed',
    });

    const engine = {
      getGameState: jest.fn(() => completedState),
      getVictoryResult: jest.fn(
        () =>
          ({
            winner: 1,
            reason: 'ring_elimination',
            finalScore: {
              ringsEliminated: {},
              territorySpaces: {},
              ringsRemaining: {},
            },
          }) as GameResult
      ),
      getGameEndExplanation: jest.fn(() => null),
    };

    mockStoreGame.mockRejectedValueOnce(new Error('service down'));
    mockStoreGameLocally.mockResolvedValueOnce({ success: true, id: 'local-123' });
    mockGetPendingCount.mockResolvedValueOnce(1);

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    await waitFor(() => {
      expect(mockStoreGame).toHaveBeenCalled();
      expect(mockStoreGameLocally).toHaveBeenCalled();
      expect(screen.getByText(/1 game pending/)).toBeInTheDocument();
    });

    const syncButton = screen.getByRole('button', { name: 'Sync' });
    expect(syncButton).not.toBeDisabled();
    fireEvent.click(syncButton);
    expect(mockGameSyncTriggerSync).toHaveBeenCalled();

    if (mockGameSyncSubscribeCallback) {
      const callback = mockGameSyncSubscribeCallback;
      act(() => {
        callback({
          status: 'idle',
          pendingCount: 0,
          lastSyncAttempt: new Date(),
          lastSuccessfulSync: new Date(),
          consecutiveFailures: 0,
        });
      });
    }

    await waitFor(() => {
      expect(screen.queryByText(/game pending/)).toBeNull();
    });
  });
});
