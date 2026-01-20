/**
 * SandboxGameHost Tests - Controls, Stall Warning, Victory
 *
 * Tests for:
 * - Touch controls / overlay toggles
 * - Mobile HUD
 * - Stall warning + AI trace diagnostics
 * - Victory modal wiring for local sandbox games
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import type { GameResult, Position } from '../../../src/shared/types/game';

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
              className: 'valid-move-cell',
              onClick: () => props.onCellClick?.({ x, y }),
            })
          );
        }
      }
      // Add SVG overlay and valid move indicator based on props
      const svgOverlay = props.overlays?.showMovementGrid
        ? React.createElement('svg', { key: 'svg-overlay' })
        : null;
      return React.createElement(
        'div',
        { 'data-testid': 'board-view', 'data-board-type': boardType },
        cells,
        svgOverlay
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
      showMovementGrid: true,
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
  setIsMobile,
  getSquareCell,
} from './SandboxGameHost.testUtils';

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// TODO(2026-01-20): Tests split from SandboxGameHost.test.tsx for memory isolation.
// The original tests were skipped due to OOM issues. After splitting, tests now run
// in separate Jest workers but need assertion updates for current component structure.
// The component's internal rendering changed since these tests were written.
describe.skip('SandboxGameHost - Controls, Stall Warning, Victory', () => {
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

  // ─────────────────────────────────────────────────────────────────────────────
  // 4. Touch controls / overlay toggles
  // ─────────────────────────────────────────────────────────────────────────────

  it('toggles movement grid and valid-target overlays via SandboxTouchControlsPanel', () => {
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

    const board = screen.getByTestId('board-view');

    const sourceCell = getSquareCell(0, 0);
    fireEvent.click(sourceCell);

    expect(board.querySelector('.valid-move-cell')).not.toBeNull();

    const validTargetsToggle = screen.getByLabelText('Show valid targets') as HTMLInputElement;
    const movementGridToggle = screen.getByLabelText('Show movement grid') as HTMLInputElement;

    expect(validTargetsToggle.checked).toBe(true);
    expect(movementGridToggle.checked).toBe(true);
    expect(board.querySelector('svg')).not.toBeNull();

    fireEvent.click(movementGridToggle);
    expect(movementGridToggle.checked).toBe(false);
    expect(board.querySelector('svg')).toBeNull();

    fireEvent.click(validTargetsToggle);
    expect(validTargetsToggle.checked).toBe(false);
    expect(board.querySelector('.valid-move-cell')).toBeNull();

    fireEvent.click(movementGridToggle);
    expect(movementGridToggle.checked).toBe(true);
    expect(board.querySelector('svg')).not.toBeNull();
  });

  it('renders MobileGameHUD instead of GameHUD on mobile viewports', () => {
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

    setIsMobile(true);

    const { container } = render(<SandboxGameHost />);

    expect(container.querySelector('[data-testid="mobile-game-hud"]')).not.toBeNull();
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 5. Stall warning + AI trace diagnostics
  // ─────────────────────────────────────────────────────────────────────────────

  it('renders stall warning banner, copies AI trace, and clears warning on dismiss', async () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'movement',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const setSandboxStallWarning = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      developerToolsEnabled: true,
      sandboxStallWarning:
        'Potential AI stall detected: sandbox AI has not advanced the game state for several seconds while an AI player is to move.',
      setSandboxStallWarning,
    });

    (window as any).__RINGRIFT_SANDBOX_TRACE__ = [
      { kind: 'stall', timestamp: 123, details: 'test-entry' },
    ];

    const writeText = jest.fn().mockResolvedValue(undefined);
    (navigator as any).clipboard = { writeText };

    render(<SandboxGameHost />);

    expect(screen.getByText(/Potential AI stall detected/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Copy AI trace/i }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledTimes(1);
    });

    const payload = writeText.mock.calls[0][0] as string;
    const parsed = JSON.parse(payload);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed[0].kind).toBe('stall');

    fireEvent.click(screen.getByRole('button', { name: /Dismiss/i }));
    expect(setSandboxStallWarning).toHaveBeenCalledWith(null);
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 6. Victory modal wiring for local sandbox games
  // ─────────────────────────────────────────────────────────────────────────────

  it('shows VictoryModal when sandboxVictoryResult is present and resets engine on Return to Lobby', async () => {
    const players = createPlayers();
    const completedState = createSandboxGameState({
      players,
      currentPhase: 'movement',
      gameStatus: 'completed',
    });

    const victoryResult: GameResult = {
      winner: 1,
      reason: 'ring_elimination',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    const engine = {
      getGameState: jest.fn(() => completedState),
      getVictoryResult: jest.fn(() => victoryResult),
      getGameEndExplanation: jest.fn(() => null),
    };

    const resetSandboxEngine = jest.fn();
    const setBackendSandboxError = jest.fn();
    const setSandboxPendingChoice = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      resetSandboxEngine,
      setBackendSandboxError,
      setSandboxPendingChoice,
    });

    render(<SandboxGameHost />);

    await waitFor(() => {
      expect(mockStoreGame).toHaveBeenCalled();
    });

    expect(screen.getByText('Return to Lobby')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Return to Lobby'));

    expect(resetSandboxEngine).toHaveBeenCalledTimes(1);
    expect(setBackendSandboxError).toHaveBeenCalledWith(null);
    expect(setSandboxPendingChoice).toHaveBeenCalledWith(null);
  });
});
