import React from 'react';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import type {
  BoardState,
  GameResult,
  GameState,
  Move,
  Player,
  PlayerChoice,
  Position,
} from '../../../src/shared/types/game';
import type { LocalConfig, LocalPlayerType } from '../../../src/client/contexts/SandboxContext';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';
import { gameApi } from '../../../src/client/services/api';
import { serializeGameState } from '../../../src/shared/engine/contracts/serialization';

jest.mock('../../../src/client/hooks/useIsMobile', () => ({
  __esModule: true,
  useIsMobile: jest.fn(() => false),
}));

// ─────────────────────────────────────────────────────────────────────────────
// Mocks
// ─────────────────────────────────────────────────────────────────────────────

const mockNavigate = jest.fn();
let mockAuthUser: { id: string; username: string } | null = { id: 'user-1', username: 'Alice' };
const mockStoreGameLocally = jest.fn();
const mockGetPendingCount = jest.fn();

type MockSyncState = {
  status: 'idle' | 'syncing' | 'error' | 'offline';
  pendingCount: number;
  lastSyncAttempt: Date | null;
  lastSuccessfulSync: Date | null;
  consecutiveFailures: number;
};

const mockGameSyncStart = jest.fn();
const mockGameSyncStop = jest.fn();
const mockGameSyncTriggerSync = jest.fn();
let mockGameSyncSubscribeCallback: ((state: MockSyncState) => void) | null = null;

// Capture the most recent BoardView props so tests can assert that
// SandboxGameHost wires chainCapturePath and decision highlights as expected.
let lastBoardViewProps: any = null;
jest.mock('../../../src/client/components/BoardView', () => {
  const actual = jest.requireActual('../../../src/client/components/BoardView');
  return {
    __esModule: true,
    ...actual,
    BoardView: jest.fn((props: any) => {
      lastBoardViewProps = props;
      return actual.BoardView(props);
    }),
  };
});

jest.mock('react-router-dom', () => ({
  // Reuse actual exports where possible, but override useNavigate
  __esModule: true,
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}));

jest.mock('../../../src/client/contexts/AuthContext', () => ({
  __esModule: true,
  useAuth: () => ({
    user: mockAuthUser,
  }),
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
    start: (...args: unknown[]) => mockGameSyncStart(...args),
    stop: (...args: unknown[]) => mockGameSyncStop(...args),
    subscribe: (cb: (state: MockSyncState) => void) => {
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

// Mock ReplayService to prevent unhandled async state updates after tests complete
const mockStoreGame = jest.fn().mockResolvedValue({ success: true, totalMoves: 10 });
jest.mock('../../../src/client/services/ReplayService', () => ({
  __esModule: true,
  getReplayService: () => ({
    storeGame: mockStoreGame,
  }),
}));

let mockSandboxValue: any;
let replayPanelProps: any | null = null;
let scenarioPickerProps: any | null = null;

// Choice + AI helpers wired via useSandboxInteractions mock
const mockMaybeRunSandboxAiIfNeeded = jest.fn();
const mockChoiceResolve = jest.fn();

jest.mock('../../../src/client/contexts/SandboxContext', () => ({
  __esModule: true,
  useSandbox: () => mockSandboxValue,
}));

jest.mock('../../../src/client/components/ScenarioPickerModal', () => ({
  __esModule: true,
  ScenarioPickerModal: (props: any) => {
    scenarioPickerProps = props;
    return null;
  },
  default: (props: any) => {
    scenarioPickerProps = props;
    return null;
  },
}));

jest.mock('../../../src/client/components/ReplayPanel', () => ({
  __esModule: true,
  // Capture props so tests can drive ReplayPanel callbacks (state change,
  // replay mode toggles, and fork-from-position) without rendering the full
  // panel UI.
  ReplayPanel: (props: any) => {
    replayPanelProps = props;
    return null;
  },
}));

jest.mock('../../../src/client/hooks/useSandboxInteractions', () => ({
  __esModule: true,
  useSandboxInteractions: (options: any) => {
    // Ensure the choice resolver ref is always wired so that the host's
    // ChoiceDialog onSelectOption can call back into our test spy.
    if (options.choiceResolverRef) {
      options.choiceResolverRef.current = (response: any) => {
        mockChoiceResolve(response);
      };
    }

    return {
      handleCellClick: (pos: Position) => {
        // Minimal behaviour: selecting a cell updates host-local selection
        // and exposes a single valid target so overlays can be asserted.
        options.setSelected(pos);
        options.setValidTargets([{ x: pos.x, y: pos.y + 1 }]);
      },
      handleCellDoubleClick: jest.fn(),
      handleCellContextMenu: jest.fn(),
      maybeRunSandboxAiIfNeeded: () => {
        mockMaybeRunSandboxAiIfNeeded();
      },
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
    // isFirstTimePlayer: true ensures advanced options start hidden
    // so the "Show advanced options" button is visible
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

// Load the component under test only after all jest.mock() declarations above
// have executed, so the module sees consistent mocked dependencies even when
// Jest reuses a single Node process across multiple test files.
const { SandboxGameHost } =
  require('../../../src/client/pages/SandboxGameHost') as typeof import('../../../src/client/pages/SandboxGameHost');

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function createEmptySquareBoard(size = 8): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size,
    type: 'square8',
  };
}

function createPlayers(overrides: Partial<Player>[] = []): Player[] {
  const base: Player[] = [
    {
      id: 'p1',
      username: 'P1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'P2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
      aiDifficulty: 5,
      aiProfile: { difficulty: 5, aiType: 'heuristic' },
    },
  ];

  return base.map((p, idx) => ({ ...p, ...(overrides[idx] ?? {}) }));
}

function createSandboxGameState(overrides: Partial<GameState> = {}): GameState {
  const board = createEmptySquareBoard(8);
  // Single stack at (0,0) for selection tests
  const key = '0,0';
  (board.stacks as Map<string, any>).set(key, {
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
    position: { x: 0, y: 0 },
  });

  const players = createPlayers(overrides.players as Player[] | undefined);

  const base: GameState = {
    id: 'sandbox-1',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };

  return { ...base, ...overrides, players: overrides.players ?? players };
}

function createLocalConfig(overrides: Partial<LocalConfig> = {}): LocalConfig {
  const base: LocalConfig = {
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai'],
    aiDifficulties: [5, 5, 5, 5],
  };
  return { ...base, ...overrides };
}

function setIsMobile(value: boolean) {
  const mod = require('../../../src/client/hooks/useIsMobile') as {
    useIsMobile: jest.Mock;
  };
  mod.useIsMobile.mockReturnValue(value);
}

function createMockSandboxContext(overrides: Partial<any> = {}): any {
  const config = createLocalConfig();
  const base = {
    config,
    setConfig: jest.fn(),
    isConfigured: false,
    setIsConfigured: jest.fn(),
    backendSandboxError: null,
    setBackendSandboxError: jest.fn(),
    sandboxEngine: null,
    sandboxPendingChoice: null,
    setSandboxPendingChoice: jest.fn(),
    sandboxCaptureChoice: null,
    setSandboxCaptureChoice: jest.fn(),
    sandboxCaptureTargets: [] as Position[],
    setSandboxCaptureTargets: jest.fn(),
    sandboxLastProgressAt: null as number | null,
    setSandboxLastProgressAt: jest.fn(),
    sandboxStallWarning: null as string | null,
    setSandboxStallWarning: jest.fn(),
    sandboxStateVersion: 0,
    setSandboxStateVersion: jest.fn(),
    sandboxDiagnosticsEnabled: true,
    developerToolsEnabled: false,
    setDeveloperToolsEnabled: jest.fn(),
    initLocalSandboxEngine: jest.fn(),
    getSandboxGameState: jest.fn(() => null),
    resetSandboxEngine: jest.fn(),
  };

  const merged = { ...base, ...overrides };

  if (merged.sandboxEngine) {
    const engine = merged.sandboxEngine;
    merged.sandboxEngine = {
      ...engine,
      getLpsTrackingState:
        typeof engine.getLpsTrackingState === 'function'
          ? engine.getLpsTrackingState
          : jest.fn(() => null),
      getValidMoves:
        typeof engine.getValidMoves === 'function' ? engine.getValidMoves : jest.fn(() => []),
      getChainCaptureContextForCurrentPlayer:
        typeof engine.getChainCaptureContextForCurrentPlayer === 'function'
          ? engine.getChainCaptureContextForCurrentPlayer
          : jest.fn(() => null),
    };
  }

  return merged;
}

// Helper: find a square-board cell for sandbox BoardView
function getSquareCell(x: number, y: number): HTMLButtonElement {
  const board = screen.getByTestId('board-view');
  const cell = board.querySelector<HTMLButtonElement>(`button[data-x="${x}"][data-y="${y}"]`);
  if (!cell) {
    throw new Error(`Failed to find sandbox board cell at (${x}, ${y})`);
  }
  return cell;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// TODO(2026-01-20): This test suite has a severe memory leak causing OOM even with 8GB heap.
// The tests pass individually but accumulate memory across runs, hitting ~4GB and crashing.
// Skipped until the leak can be investigated - likely an issue with React component cleanup,
// mock accumulation, or async timers not being properly cleared between tests.
// See: jest --detectOpenHandles --testPathPattern="SandboxGameHost.test"
describe.skip('SandboxGameHost (React host behaviour)', () => {
  // Silence expected console noise from sandbox fallback/error paths so
  // test output remains focused on assertions.
  const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

  afterAll(() => {
    consoleErrorSpy.mockRestore();
    consoleWarnSpy.mockRestore();
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
    mockGameSyncStart.mockReset();
    mockGameSyncStop.mockReset();
    mockGameSyncTriggerSync.mockReset();
    mockGameSyncSubscribeCallback = null;
    replayPanelProps = null;
    scenarioPickerProps = null;
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 1. Local sandbox start & backend game creation
  // ───────────────────────────────────────────────────────────────────────────

  it('attempts backend sandbox creation and navigates to /game/:id on success', async () => {
    (gameApi.createGame as jest.Mock).mockResolvedValue({ id: 'backend-123' });

    render(<SandboxGameHost />);

    // Use the primary quick-start preset ("Learn the Basics") to kick off
    // backend sandbox creation, matching the main UX entry point.
    // Multiple buttons may match (mobile/desktop views) - use first one
    const quickStartButtons = screen.getAllByRole('button', { name: /Learn the Basics/i });
    fireEvent.click(quickStartButtons[0]);

    await waitFor(() => {
      expect(gameApi.createGame).toHaveBeenCalledTimes(1);
      expect(mockNavigate).toHaveBeenCalledWith('/game/backend-123');
    });

    // When backend creation succeeds we should not fall back to local sandbox.
    expect(mockSandboxValue.initLocalSandboxEngine).not.toHaveBeenCalled();
  });

  it('falls back to local sandbox when backend creation fails and triggers AI loop for AI first player', async () => {
    (gameApi.createGame as jest.Mock).mockRejectedValue(new Error('backend error'));

    const aiFirstConfig: LocalConfig = {
      numPlayers: 2,
      boardType: 'square8',
      playerTypes: ['ai', 'human', 'ai', 'ai'] as LocalPlayerType[],
      aiDifficulties: [5, 5, 5, 5],
    };

    const aiPlayers: Player[] = createPlayers([
      { type: 'ai' } as Player,
      { type: 'human' } as Player,
    ]);

    const sandboxState = createSandboxGameState({
      players: aiPlayers,
      currentPlayer: 1,
      gameStatus: 'active',
    });

    const fakeEngine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getLpsTrackingState: jest.fn(() => null),
      getValidMoves: jest.fn(() => []),
    };

    mockSandboxValue = createMockSandboxContext({
      config: aiFirstConfig,
      initLocalSandboxEngine: jest.fn(() => fakeEngine),
    });

    render(<SandboxGameHost />);

    // Multiple buttons may match (mobile/desktop views) - use first one
    const quickStartButtons = screen.getAllByRole('button', { name: /Learn the Basics/i });
    fireEvent.click(quickStartButtons[0]);

    await waitFor(() => {
      expect(mockSandboxValue.initLocalSandboxEngine).toHaveBeenCalledTimes(1);
    });

    // Host should surface a backend error banner and fall back to local-only sandbox.
    expect(mockSandboxValue.setBackendSandboxError).toHaveBeenCalledWith(
      'Backend sandbox game could not be created; falling back to local-only board only.'
    );

    // Because the first player is an AI in the returned GameState, the host
    // should ask the interactions hook to start the sandbox AI loop.
    expect(mockMaybeRunSandboxAiIfNeeded).toHaveBeenCalled();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 2. Active sandbox game: selection + touch controls wiring
  // ───────────────────────────────────────────────────────────────────────────

  it('wires BoardView clicks into sandbox selection and touch controls panel', () => {
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

    // Before any interaction, the touch controls panel should show no selection
    // and the default helper copy.
    const panel = screen.getByTestId('sandbox-touch-controls');
    expect(panel).toHaveTextContent('Selection');
    expect(panel).toHaveTextContent('Targets: 0');
    expect(panel).toHaveTextContent('Tap any stack or empty cell to begin.');

    // Click a cell on the board; our useSandboxInteractions mock will update
    // host-local selection + a single valid target.
    const sourceCell = getSquareCell(0, 0);
    fireEvent.click(sourceCell);

    // The touch controls panel should now reflect the selected position and
    // the number of valid targets exposed by the interactions hook.
    expect(panel).toHaveTextContent('(0, 0)');
    expect(panel).toHaveTextContent('Targets: 1');
  });

  it('shows the chain-capture continuation chip when chain_capture has legal continuation segments', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'chain_capture',
      gameStatus: 'active',
    });

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      // Minimal getValidMoves implementation that exposes a single
      // continue_capture_segment move from the current player so the
      // host can detect an active chain-capture continuation step.
      getValidMoves: jest.fn(() => [
        {
          id: 'm1',
          type: 'continue_capture_segment',
          player: sandboxState.currentPlayer,
        } as Move,
      ]),
      // Chain capture context for current player
      getChainCaptureContextForCurrentPlayer: jest.fn(() => ({
        from: { x: 0, y: 0 },
        landings: [{ x: 1, y: 1 }],
      })),
    };

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
    });

    render(<SandboxGameHost />);

    // The primary subtitle chip under the board should switch from the
    // default board subtitle to an explicit "Continue Chain Capture"
    // prompt when isChainCaptureContinuationStep is true.
    expect(screen.getByText('Continue Chain Capture')).toBeInTheDocument();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 3. Pending sandbox decisions (ChoiceDialog + capture details)
  // ───────────────────────────────────────────────────────────────────────────

  it('renders ChoiceDialog for sandbox pending choice and resolves via stored resolver', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'line_processing',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const pendingChoice: PlayerChoice = {
      id: 'choice-1',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'How do you want to process this line?',
      timeoutMs: undefined,
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    } as any;

    const setSandboxPendingChoice = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxPendingChoice: pendingChoice,
      setSandboxPendingChoice,
    });

    render(<SandboxGameHost />);

    // Options from ChoiceDialog should be present (using actual button text).
    expect(screen.getByText('Full Collapse + Elimination Bonus')).toBeInTheDocument();
    expect(screen.getByText('Minimum Collapse')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Full Collapse + Elimination Bonus'));

    // The resolver provided via useSandboxInteractions should be invoked with
    // a PlayerChoiceResponse payload.
    expect(mockChoiceResolve).toHaveBeenCalledTimes(1);
    const response = mockChoiceResolve.mock.calls[0][0];
    expect(response.choiceId).toBe('choice-1');
    expect(response.playerNumber).toBe(1);
    expect(response.choiceType).toBe('line_reward_option');
    expect(response.selectedOption).toBe('option_1_collapse_all_and_eliminate');

    // The host should also clear the sandboxPendingChoice flag in context.
    expect(setSandboxPendingChoice).toHaveBeenCalledWith(null);
  });

  it('surfaces capture-direction targets in the touch controls panel and pulses landing cells', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'capture',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getValidMoves: jest.fn(() => []),
    };

    const captureChoice: PlayerChoice = {
      id: 'cap-1',
      playerNumber: 1,
      type: 'capture_direction',
      prompt: 'Continue capture?',
      timeoutMs: undefined,
      options: [
        {
          targetPosition: { x: 0, y: 1 },
          landingPosition: { x: 0, y: 2 },
          capturedCapHeight: 1,
        },
      ],
    } as any;

    const captureTargets: Position[] = [
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxCaptureChoice: captureChoice,
      sandboxCaptureTargets: captureTargets,
    });

    render(<SandboxGameHost />);

    const panel = screen.getByTestId('sandbox-touch-controls');
    expect(panel).toHaveTextContent('Capture segments');
    // All capture targets should be listed in the panel.
    expect(panel).toHaveTextContent('(0, 1)');
    expect(panel).toHaveTextContent('(1, 1)');

    // Landing cell should be highlighted as a primary decision target with a
    // pulsing capture overlay so the next chain-capture segment is obvious.
    const landingCell = getSquareCell(0, 2);
    expect(landingCell).toHaveAttribute('data-decision-highlight', 'primary');
    expect(landingCell.className).toContain('decision-pulse-capture');
  });

  it('wires capture_direction choices during chain_capture into both BoardView decision highlights and chainCapturePath', () => {
    // Build a sandbox GameState in the chain_capture phase with a short
    // capture sequence in moveHistory so SandboxGameHost derives a concrete
    // chainCapturePath while a capture_direction choice is pending.
    const sandboxState = createSandboxGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      gameStatus: 'active',
    });

    sandboxState.moveHistory = [
      {
        id: 'm1',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 0, y: 0 },
        captureTarget: { x: 0, y: 1 },
        to: { x: 0, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      } as any,
      {
        id: 'm2',
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 0, y: 2 },
        captureTarget: { x: 0, y: 3 },
        to: { x: 0, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      } as any,
    ];

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const captureChoice: PlayerChoice = {
      id: 'cap-2',
      playerNumber: 1,
      type: 'capture_direction',
      prompt: 'Continue capture?',
      timeoutMs: undefined,
      options: [
        {
          targetPosition: { x: 0, y: 3 },
          landingPosition: { x: 0, y: 4 },
          capturedCapHeight: 1,
        },
      ],
    } as any;

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxCaptureChoice: captureChoice,
      sandboxCaptureTargets: [{ x: 0, y: 3 }],
    });

    render(<SandboxGameHost />);

    // BoardView should receive a non-empty chainCapturePath derived from the
    // chain-capture moves, while also seeing capture_direction decision
    // highlights in its viewModel.
    expect(lastBoardViewProps).not.toBeNull();
    const { chainCapturePath, viewModel } = lastBoardViewProps;

    expect(Array.isArray(chainCapturePath)).toBe(true);
    expect(chainCapturePath.length).toBeGreaterThanOrEqual(2);

    expect(viewModel?.decisionHighlights?.choiceKind).toBe('capture_direction');
  });

  it('surfaces optional capture opportunities in capture phase via pulsing highlights and a HUD chip', () => {
    // Sandbox state in capture phase with no explicit PlayerChoice but with
    // canonical overtaking_capture + skip_capture moves available for the
    // current player. This should surface stronger capture-direction
    // highlights and a bright HUD chip so the opportunity is obvious.
    const sandboxState = createSandboxGameState({
      currentPhase: 'capture',
      gameStatus: 'active',
    });

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
      getValidMoves: jest.fn(() => {
        const from: Position = { x: 0, y: 0 };
        const captureTarget: Position = { x: 0, y: 1 };
        const landing: Position = { x: 0, y: 2 };
        const baseMove: Partial<Move> = {
          id: 'm1',
          player: 1,
          from,
          captureTarget,
          to: landing,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };
        return [
          { ...baseMove, type: 'overtaking_capture' } as Move,
          { ...baseMove, type: 'skip_capture' } as Move,
        ];
      }),
    };

    // Capture targets array is used purely for touch-panel listing; landing
    // cells are still derived from canonical moves above.
    const captureTargets: Position[] = [{ x: 0, y: 1 }];

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxCaptureTargets: captureTargets,
    });

    render(<SandboxGameHost />);

    // Both the capture target stack and landing cell should receive capture
    // pulse styling via decisionHighlights with choiceKind=capture_direction.
    const targetCell = getSquareCell(0, 1);
    expect(targetCell).toHaveAttribute('data-decision-highlight', 'secondary');
    expect(targetCell.className).toContain('capture-target-pulse');

    const landingCell = getSquareCell(0, 2);
    expect(landingCell).toHaveAttribute('data-decision-highlight', 'primary');
    expect(landingCell.className).toContain('decision-pulse-capture');

    // HUD should render a bright attention-style status chip advertising the
    // optional capture opportunity.
    const chip = screen.getByTestId('hud-decision-status-chip');
    expect(chip.textContent).toMatch(/Capture available/i);
  });

  it('applies decision-phase highlights to the sandbox BoardView when a pending choice is active', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'territory_processing',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const pendingChoice: PlayerChoice = {
      id: 'elim-1',
      playerNumber: 1,
      type: 'ring_elimination',
      prompt: 'Choose elimination stack',
      timeoutMs: undefined,
      options: [
        {
          stackPosition: { x: 0, y: 0 },
          eliminatedFromStack: 1,
        },
      ],
    } as any;

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxPendingChoice: pendingChoice,
    });

    render(<SandboxGameHost />);

    const highlightedCell = getSquareCell(0, 0);
    expect(highlightedCell).toHaveAttribute('data-decision-highlight', 'primary');
    expect(highlightedCell.className).toContain('decision-pulse-elimination');

    const stackPulse = highlightedCell.querySelector('.decision-elimination-stack-pulse');
    expect(stackPulse).toBeInTheDocument();
  });

  it('surfaces territory region_order chip and highlights the full region in the sandbox BoardView and touch controls', () => {
    // Board with a small disconnected territory region for Player 1.
    const board = createEmptySquareBoard(8);
    const regionSpaces: Position[] = [
      { x: 3, y: 3 },
      { x: 3, y: 4 },
      { x: 4, y: 3 },
    ];

    const territoryRegion = {
      spaces: regionSpaces,
      controllingPlayer: 1,
      isDisconnected: true,
    } as any;

    (board.territories as Map<string, any>).set('region-0', territoryRegion);

    const sandboxState = createSandboxGameState({
      board,
      currentPhase: 'territory_processing',
      gameStatus: 'active',
    });

    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const pendingChoice: PlayerChoice = {
      id: 'territory-choice-1',
      playerNumber: 1,
      type: 'region_order',
      prompt: 'Choose which region to process',
      timeoutMs: undefined,
      options: [
        {
          regionId: '0',
          size: regionSpaces.length,
          representativePosition: regionSpaces[0],
          moveId: 'process-region-0-3,3',
        },
      ],
    } as any;

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
      sandboxPendingChoice: pendingChoice,
    });

    render(<SandboxGameHost />);

    const decisionHintText = 'Territory claimed \u2013 choose region to process or skip';

    // The summary chip under the board AND the touch controls panel should
    // both surface the decision prompt for consistent UX across interaction
    // modes. Use getAllByText since the hint now appears in both locations.
    const hintElements = screen.getAllByText(decisionHintText);
    expect(hintElements.length).toBeGreaterThanOrEqual(2);

    // Verify the touch controls panel specifically contains the hint.
    const touchControls = screen.getByTestId('sandbox-touch-controls');
    expect(touchControls).toHaveTextContent(decisionHintText);

    // All cells in the disconnected region should be highlighted as primary
    // decision targets with a pulsing green overlay so the territory geometry
    // is clearly visible.
    for (const p of regionSpaces) {
      const cell = getSquareCell(p.x, p.y);
      expect(cell).toHaveAttribute('data-decision-highlight', 'primary');
      expect(cell.className).toContain('decision-pulse-territory');
    }
  });

  it('replays canonical self-play moves into the sandbox engine for scenarios with selfPlayMeta.moves', async () => {
    const applyCanonicalMoveForReplay = jest.fn().mockResolvedValue(undefined);
    const initFromSerializedState = jest.fn();

    const engine = {
      initFromSerializedState,
      applyCanonicalMoveForReplay,
      getGameState: jest.fn(() => createSandboxGameState()),
      getVictoryResult: jest.fn(() => null as GameResult | null),
      getGameEndExplanation: jest.fn(() => null),
    };

    const setSandboxStateVersion = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      initLocalSandboxEngine: jest.fn(() => engine),
      setSandboxStateVersion,
    });

    render(<SandboxGameHost />);

    expect(scenarioPickerProps).not.toBeNull();
    if (!scenarioPickerProps) {
      throw new Error('ScenarioPickerModal props were not captured');
    }

    const initialState = createSandboxGameState({
      moveHistory: [],
      history: [],
    });
    const serializedState = serializeGameState(initialState);

    const now = new Date();

    const moves: Move[] = [
      {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        from: undefined,
        to: { x: 0, y: 0 },
        placedOnStack: false,
        placementCount: 1,
        timestamp: now,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'eliminate_rings_from_stack',
        player: 1,
        from: undefined,
        to: { x: 0, y: 0 },
        timestamp: now,
        thinkTime: 0,
        moveNumber: 2,
      },
    ];

    const scenario: LoadableScenario = {
      id: 'selfplay-fixture',
      name: 'Self-Play Fixture',
      description: 'Test self-play scenario',
      category: 'custom',
      tags: [],
      boardType: 'square8',
      playerCount: 2,
      createdAt: now.toISOString(),
      source: 'custom',
      state: serializedState,
      selfPlayMeta: {
        dbPath: '/tmp/test-selfplay.db',
        gameId: 'game-1',
        totalMoves: moves.length,
        moves,
      },
    };

    await act(async () => {
      scenarioPickerProps.onSelectScenario(scenario);
    });

    await waitFor(() => {
      expect(applyCanonicalMoveForReplay).toHaveBeenCalledTimes(moves.length);
    });

    expect(setSandboxStateVersion).toHaveBeenCalled();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 4. Touch controls / overlay toggles
  // ───────────────────────────────────────────────────────────────────────────

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

    // Click a cell to create a selection + one valid target via our interactions mock.
    const sourceCell = getSquareCell(0, 0);
    fireEvent.click(sourceCell);

    // When valid targets are shown, a highlighted cell should be present.
    expect(board.querySelector('.valid-move-cell')).not.toBeNull();

    const validTargetsToggle = screen.getByLabelText('Show valid targets') as HTMLInputElement;
    const movementGridToggle = screen.getByLabelText('Show movement grid') as HTMLInputElement;

    // Both overlays are enabled by default (see SandboxGameHost line 153-156).
    // Movement grid helps players understand valid moves and adjacency patterns.
    expect(validTargetsToggle.checked).toBe(true);
    expect(movementGridToggle.checked).toBe(true);
    expect(board.querySelector('svg')).not.toBeNull();

    // Hide movement grid: the SVG overlay should disappear.
    fireEvent.click(movementGridToggle);
    expect(movementGridToggle.checked).toBe(false);
    expect(board.querySelector('svg')).toBeNull();

    // Hide valid targets: highlights should disappear but selection remains.
    fireEvent.click(validTargetsToggle);
    expect(validTargetsToggle.checked).toBe(false);
    expect(board.querySelector('.valid-move-cell')).toBeNull();

    // Re-enable movement grid: the overlay SVG should appear again.
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

  // ───────────────────────────────────────────────────────────────────────────
  // 5. Stall warning + AI trace diagnostics
  // ───────────────────────────────────────────────────────────────────────────

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

    // Seed a fake sandbox AI trace buffer on the window object.
    (window as any).__RINGRIFT_SANDBOX_TRACE__ = [
      { kind: 'stall', timestamp: 123, details: 'test-entry' },
    ];

    const writeText = jest.fn().mockResolvedValue(undefined);
    (navigator as any).clipboard = { writeText };

    render(<SandboxGameHost />);

    expect(screen.getByText(/Potential AI stall detected/i)).toBeInTheDocument();

    // Copy AI trace should marshal the trace into JSON and write it to clipboard.
    fireEvent.click(screen.getByRole('button', { name: /Copy AI trace/i }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledTimes(1);
    });

    const payload = writeText.mock.calls[0][0] as string;
    const parsed = JSON.parse(payload);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed[0].kind).toBe('stall');

    // Dismiss should clear the stall warning via context setter.
    fireEvent.click(screen.getByRole('button', { name: /Dismiss/i }));
    expect(setSandboxStallWarning).toHaveBeenCalledWith(null);
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 6. Victory modal wiring for local sandbox games
  // ───────────────────────────────────────────────────────────────────────────

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

    // Wait for the auto-save effect to complete to avoid act() warnings
    await waitFor(() => {
      expect(mockStoreGame).toHaveBeenCalled();
    });

    // When a victory result is returned by the sandbox engine, the VictoryModal
    // should be open and show the Return to Lobby button.
    expect(screen.getByText('Return to Lobby')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Return to Lobby'));

    expect(resetSandboxEngine).toHaveBeenCalledTimes(1);
    expect(setBackendSandboxError).toHaveBeenCalledWith(null);
    expect(setSandboxPendingChoice).toHaveBeenCalledWith(null);
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 7. Sandbox evaluation panel + /api/games/sandbox/evaluate wiring
  // ───────────────────────────────────────────────────────────────────────────

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

    // Developer tools are enabled via the mocked sandbox context so the
    // AI evaluation panel is visible by default.
    const evalButton = screen.getByRole('button', { name: /Request evaluation/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    expect(fetchMock).toHaveBeenCalledWith('/api/games/sandbox/evaluate', expect.any(Object));

    // EvaluationPanel should now render the latest move / engine profile header.
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

    // Developer tools are enabled via the mocked sandbox context so the
    // AI evaluation panel is visible by default.
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

    // Configure the sandbox so that Player 2 is an AI seat. This exercises the
    // branch where the host includes aiOpponents in the CreateGameRequest
    // payload while still using the same LocalConfig/Context wiring as the
    // real UI.
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

    // With 2 players and exactly one AI seat, the host should request one AI
    // opponent via the service-backed AI with heuristic profile. The swap rule
    // is disabled by default for sandbox games.
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

    // Overlay hidden initially
    expect(screen.queryByTestId('board-controls-overlay')).toBeNull();

    // Press "?" to open overlay
    fireEvent.keyDown(window, { key: '?' });
    expect(screen.getByTestId('board-controls-overlay')).toBeInTheDocument();

    // Press Escape to close overlay
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

    // Enter replay mode with a non-null state
    act(() => {
      replayPanelProps.onStateChange(replayState);
      replayPanelProps.onReplayModeChange(true);
    });

    expect(screen.getByText(/Viewing replay - board is read-only/i)).toBeInTheDocument();

    // While in replay mode, BoardView should ignore clicks (no new selection
    // coordinates appear in the touch controls panel).
    const sourceCell = getSquareCell(0, 0);
    fireEvent.click(sourceCell);

    const panel = screen.getByTestId('sandbox-touch-controls');
    expect(panel).toHaveTextContent('Selection');
    expect(panel).not.toHaveTextContent('(0, 0)');

    // Exit replay mode; board becomes interactive again.
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
        // For 2-player games, initSandboxWithScenario uses the default 2P config:
        // ['human', 'ai', 'human', 'human'] (all 4 seats populated for max players).
        playerTypes: ['human', 'ai', 'human', 'human'],
      })
    );

    // After forking, host should leave replay mode.
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

    // Force storeGame to report a non-fatal failure so the host transitions
    // into an "error" save status while keeping the sandbox UI usable.
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

    // First-tier ReplayService save fails, forcing SandboxGameHost to use
    // LocalGameStorage as a fallback for recording.
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
      // Pending-local-games badge should show exactly one pending game.
      expect(screen.getByText(/1 game pending/)).toBeInTheDocument();
    });

    // The Sync button should delegate to GameSyncService.triggerSync.
    const syncButton = screen.getByRole('button', { name: 'Sync' });
    expect(syncButton).not.toBeDisabled();
    fireEvent.click(syncButton);
    expect(mockGameSyncTriggerSync).toHaveBeenCalled();

    // Simulate a later successful sync that clears all pending games by
    // invoking the subscribed callback with a zero-pending state.
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
