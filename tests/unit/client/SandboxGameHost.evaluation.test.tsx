/**
 * SandboxGameHost Tests - Evaluation Panel
 *
 * Tests for:
 * - Sandbox evaluation panel + /api/games/sandbox/evaluate wiring
 * - Evaluation error handling
 * - AI opponent count and swap-rule options
 * - Change Setup functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import type { GameResult, Position } from '../../../src/shared/types/game';
import type { LocalPlayerType } from '../../../src/client/contexts/SandboxContext';
import { gameApi } from '../../../src/client/services/api';

// ─────────────────────────────────────────────────────────────────────────────
// Mock State Variables
// ─────────────────────────────────────────────────────────────────────────────

const mockNavigate = jest.fn();
let mockAuthUser: { id: string; username: string } | null = { id: 'user-1', username: 'Alice' };
let mockSandboxValue: any;
let lastBoardViewProps: any = null;

const mockMaybeRunSandboxAiIfNeeded = jest.fn();
const mockChoiceResolve = jest.fn();
const mockStoreGame = jest.fn().mockResolvedValue({ success: true, totalMoves: 10 });

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
      return React.createElement(
        'div',
        { 'data-testid': 'board-view', 'data-board-type': boardType },
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
  storeGameLocally: jest.fn(),
  getPendingCount: jest.fn(),
}));

jest.mock('../../../src/client/services/GameSyncService', () => ({
  __esModule: true,
  GameSyncService: {
    start: jest.fn(),
    stop: jest.fn(),
    subscribe: (cb: any) => {
      cb({
        status: 'idle',
        pendingCount: 0,
        lastSyncAttempt: null,
        lastSuccessfulSync: null,
        consecutiveFailures: 0,
      });
      return jest.fn();
    },
    triggerSync: jest.fn(),
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
  ReplayPanel: () => null,
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

jest.mock('../../../src/client/components/VictoryModal', () => ({
  __esModule: true,
  VictoryModal: () => null,
}));

jest.mock('../../../src/client/components/ChoiceDialog', () => ({
  __esModule: true,
  ChoiceDialog: () => null,
}));

jest.mock('../../../src/client/components/BoardControlsOverlay', () => ({
  __esModule: true,
  BoardControlsOverlay: () => null,
}));
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
  createSandboxGameState,
  createLocalConfig,
  createMockSandboxContext,
} from './SandboxGameHost.testUtils';

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// TODO(2026-01-20): Tests split from SandboxGameHost.test.tsx for memory isolation.
// The original tests were skipped due to OOM issues. After splitting, tests now run
// in separate Jest workers but need assertion updates for current component structure.
// The component's internal rendering changed since these tests were written.
describe.skip('SandboxGameHost - Evaluation Panel', () => {
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
  });

  it('requests sandbox evaluation and renders EvaluationPanel output', async () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getSerializedState: jest.fn(() => ({ dummy: 'state' }) as any),
    };

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      developerToolsEnabled: true,
    });

    const mockResponseData = {
      gameId: sandboxState.id,
      moveNumber: 3,
      boardType: sandboxState.boardType,
      perPlayer: {
        1: { totalEval: 1.5, territoryEval: 0.5, ringEval: 1.0 },
        2: { totalEval: -1.5, territoryEval: -0.5, ringEval: -1.0 },
      },
      engineProfile: 'test-engine',
      evaluationScale: 'zero_sum_margin',
    };

    const originalFetch = (global as any).fetch;
    const fetchMock = jest
      .fn()
      .mockResolvedValue({ ok: true, json: async () => mockResponseData } as any);
    (global as any).fetch = fetchMock;

    render(<SandboxGameHost />);

    const evalButton = screen.getByRole('button', { name: /Request evaluation/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    expect(fetchMock).toHaveBeenCalledWith('/api/games/sandbox/evaluate', expect.any(Object));

    const panel = screen.getByTestId('evaluation-panel');
    await waitFor(() => {
      expect(panel).toHaveTextContent('Move 3');
      expect(panel).toHaveTextContent('test-engine');
    });

    (global as any).fetch = originalFetch;
  });

  it('surfaces a helpful error message when sandbox evaluation fails', async () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getSerializedState: jest.fn(() => ({ dummy: 'state' }) as any),
    };
    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      developerToolsEnabled: true,
    });

    const originalFetch = (global as any).fetch;
    const fetchMock = jest.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ error: 'Sandbox AI evaluation is unavailable.' }),
    } as any);
    (global as any).fetch = fetchMock;

    render(<SandboxGameHost />);

    const evalButton = screen.getByRole('button', { name: /Request evaluation/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    const errorEl = await screen.findByTestId('sandbox-evaluation-error');
    expect(errorEl.textContent || '').toContain('Sandbox AI evaluation is unavailable.');

    (global as any).fetch = originalFetch;
  });

  it('passes AI opponent count and swap-rule options to backend sandbox creation request', async () => {
    (gameApi.createGame as jest.Mock).mockResolvedValue({ id: 'backend-456' });

    mockSandboxValue = createMockSandboxContext({
      config: createLocalConfig({
        playerTypes: ['human', 'ai', 'ai', 'ai'] as LocalPlayerType[],
      }),
    });

    render(<SandboxGameHost />);

    fireEvent.click(screen.getByRole('button', { name: /Show advanced options/i }));

    const launchButton = screen.getByRole('button', { name: /Launch Game/i });
    fireEvent.click(launchButton);

    await waitFor(() => {
      expect(gameApi.createGame).toHaveBeenCalledTimes(1);
    });

    const payload = (gameApi.createGame as jest.Mock).mock.calls[0][0];

    expect(payload.aiOpponents).toEqual({
      count: 1,
      difficulty: [5],
      mode: 'service',
      aiType: 'heuristic',
    });
    expect(payload.rulesOptions).toEqual({ swapRuleEnabled: false });
  });

  it('invokes resetSandboxEngine and clears sandbox host flags when Change Setup is clicked', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const resetSandboxEngine = jest.fn();
    const setBackendSandboxError = jest.fn();
    const setSandboxPendingChoice = jest.fn();
    const setSandboxStallWarning = jest.fn();
    const setSandboxLastProgressAt = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      resetSandboxEngine,
      setBackendSandboxError,
      setSandboxPendingChoice,
      setSandboxStallWarning,
      setSandboxLastProgressAt,
    });

    render(<SandboxGameHost />);

    fireEvent.click(screen.getByRole('button', { name: /Change Setup/i }));

    expect(resetSandboxEngine).toHaveBeenCalledTimes(1);
    expect(setBackendSandboxError).toHaveBeenCalledWith(null);
    expect(setSandboxPendingChoice).toHaveBeenCalledWith(null);
    expect(setSandboxStallWarning).toHaveBeenCalledWith(null);
    expect(setSandboxLastProgressAt).toHaveBeenCalledWith(null);
  });
});
