import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
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
import { SandboxGameHost } from '../../../src/client/pages/SandboxGameHost';
import { gameApi } from '../../../src/client/services/api';

// ─────────────────────────────────────────────────────────────────────────────
// Mocks
// ─────────────────────────────────────────────────────────────────────────────

const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  // Reuse actual exports where possible, but override useNavigate
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

jest.mock('@/client/contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1', username: 'Alice' },
  }),
}));

jest.mock('../../../src/client/services/api');

let mockSandboxValue: any;

// Choice + AI helpers wired via useSandboxInteractions mock
const mockMaybeRunSandboxAiIfNeeded = jest.fn();
const mockChoiceResolve = jest.fn();

jest.mock('../../../src/client/contexts/SandboxContext', () => ({
  __esModule: true,
  useSandbox: () => mockSandboxValue,
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
  };
  return { ...base, ...overrides };
}

function createMockSandboxContext(overrides: Partial<any> = {}): any {
  const config = createLocalConfig();
  return {
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
    initLocalSandboxEngine: jest.fn(),
    getSandboxGameState: jest.fn(() => null),
    resetSandboxEngine: jest.fn(),
    ...overrides,
  };
}

// Helper: find a square-board cell for sandbox BoardView
function getSquareCell(x: number, y: number): HTMLButtonElement {
  const board = screen.getByTestId('board-view');
  const cell = board.querySelector<HTMLButtonElement>(
    `button[data-x="${x}"][data-y="${y}"]`
  );
  if (!cell) {
    throw new Error(`Failed to find sandbox board cell at (${x}, ${y})`);
  }
  return cell;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('SandboxGameHost (React host behaviour)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSandboxValue = createMockSandboxContext();
    mockMaybeRunSandboxAiIfNeeded.mockReset();
    mockChoiceResolve.mockReset();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 1. Local sandbox start & backend game creation
  // ───────────────────────────────────────────────────────────────────────────

  it('attempts backend sandbox creation and navigates to /game/:id on success', async () => {
    (gameApi.createGame as jest.Mock).mockResolvedValue({ id: 'backend-123' });

    render(<SandboxGameHost />);

    const launchButton = screen.getByRole('button', { name: /Launch Game/i });
    fireEvent.click(launchButton);

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
    };

    mockSandboxValue = createMockSandboxContext({
      config: aiFirstConfig,
      initLocalSandboxEngine: jest.fn(() => fakeEngine),
    });

    render(<SandboxGameHost />);

    const launchButton = screen.getByRole('button', { name: /Launch Game/i });
    fireEvent.click(launchButton);

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

  it('surfaces capture-direction targets in the touch controls panel', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'capture',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
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
  });

  it('applies decision-phase highlights to the sandbox BoardView when a pending choice is active', () => {
    const sandboxState = createSandboxGameState({
      currentPhase: 'territory_processing',
      gameStatus: 'active',
    });
    const engine = {
      getGameState: jest.fn(() => sandboxState),
      getVictoryResult: jest.fn(() => null as GameResult | null),
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
    };

    const setSandboxStallWarning = jest.fn();

    mockSandboxValue = createMockSandboxContext({
      isConfigured: true,
      sandboxEngine: engine,
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

    expect(
      screen.getByText(/Potential AI stall detected/i)
    ).toBeInTheDocument();

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

  it('shows VictoryModal when sandboxVictoryResult is present and resets engine on Return to Lobby', () => {
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

    // When a victory result is returned by the sandbox engine, the VictoryModal
    // should be open and show the Return to Lobby button.
    expect(screen.getByText('Return to Lobby')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Return to Lobby'));

    expect(resetSandboxEngine).toHaveBeenCalledTimes(1);
    expect(setBackendSandboxError).toHaveBeenCalledWith(null);
    expect(setSandboxPendingChoice).toHaveBeenCalledWith(null);
  });
});
