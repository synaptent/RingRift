import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import type {
  BoardState,
  GamePhase,
  GameResult,
  GameState,
  Move,
  Player,
  PlayerChoice,
  Position,
} from '../../../src/shared/types/game';
import { BackendGameHost } from '../../../src/client/pages/BackendGameHost';

// ─────────────────────────────────────────────────────────────────────────────
// Mocks
// ─────────────────────────────────────────────────────────────────────────────

const mockNavigate = jest.fn();

// Keep toast output from polluting test logs
jest.mock('react-hot-toast', () => {
  const base = jest.fn();
  (base as any).success = jest.fn();
  (base as any).error = jest.fn();
  return {
    __esModule: true,
    toast: base,
  };
});

jest.mock('react-router-dom', () => ({
  // Only mock what we actually use here
  useNavigate: () => mockNavigate,
}));

jest.mock('@/client/contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1', username: 'Alice' },
  }),
}));

// Dynamic state backing the hook mocks so tests can control host behaviour.
let mockGameState: GameState | null = null;
let mockValidMoves: Move[] | null = null;
let mockVictoryState: GameResult | null = null;
let mockPlayers: Player[] = [];
let mockCurrentPlayer: Player | undefined;
// Most recent auto-resolved decision metadata surfaced via useGameState
let mockDecisionAutoResolved: any = null;
// Latest decision-phase timeout warning metadata surfaced via useGameState
let mockDecisionPhaseTimeoutWarning: any = null;

let mockConnectionStatus: 'connected' | 'connecting' | 'reconnecting' | 'disconnected' =
  'connected';
let mockIsConnecting = false;
let mockConnectionError: string | null = null;
let mockLastHeartbeatAt: number | null = Date.now();

let mockSubmitMove: jest.Mock = jest.fn();
let mockSendChatMessage: jest.Mock = jest.fn();
let mockChatMessages: { sender: string; text: string }[] = [];

let mockPendingChoice: PlayerChoice | null = null;
let mockChoiceDeadline: number | null = null;
let mockRespondToChoice: jest.Mock = jest.fn();
let mockEvaluationHistory: any[] = [];

jest.mock('@/client/hooks/useGameState', () => ({
  __esModule: true,
  useGameState: () => ({
    gameId: mockGameState ? mockGameState.id : null,
    gameState: mockGameState,
    validMoves: mockValidMoves,
    victoryState: mockVictoryState,
    players: mockPlayers,
    currentPlayer: mockCurrentPlayer,
    decisionAutoResolved: mockDecisionAutoResolved,
    decisionPhaseTimeoutWarning: mockDecisionPhaseTimeoutWarning,
    evaluationHistory: mockEvaluationHistory,
  }),
}));

jest.mock('@/client/hooks/useGameConnection', () => ({
  __esModule: true,
  useGameConnection: () => ({
    gameId: mockGameState ? mockGameState.id : null,
    status: mockConnectionStatus,
    isConnecting: mockIsConnecting,
    isHealthy: mockConnectionStatus === 'connected',
    isStale: false,
    isDisconnected: mockConnectionStatus === 'disconnected',
    statusLabel: 'Connected',
    statusColorClass: 'text-emerald-300',
    timeSinceHeartbeat: mockLastHeartbeatAt ? Date.now() - mockLastHeartbeatAt : null,
    connectToGame: jest.fn(),
    disconnect: jest.fn(),
    error: mockConnectionError,
    lastHeartbeatAt: mockLastHeartbeatAt,
  }),
}));

jest.mock('@/client/hooks/useGameActions', () => ({
  __esModule: true,
  useGameActions: () => ({
    submitMove: (...args: unknown[]) => mockSubmitMove(...args),
    submitPlacement: jest.fn(),
    submitMovement: jest.fn(),
    respondToChoice: jest.fn(),
    sendChat: (...args: unknown[]) => mockSendChatMessage(...args),
    pendingChoice: {
      choice: mockPendingChoice,
      deadline: mockChoiceDeadline,
      hasPendingChoice: !!mockPendingChoice,
    },
    capabilities: {
      canSubmitMove: true,
      canRespondToChoice: !!mockPendingChoice,
      canSendChat: true,
    },
  }),
  usePendingChoice: () => ({
    choice: mockPendingChoice,
    deadline: mockChoiceDeadline,
    hasChoice: !!mockPendingChoice,
    respond: (selectedOption: unknown) => {
      mockRespondToChoice(selectedOption);
    },
    timeRemaining: mockChoiceDeadline ? Math.max(0, mockChoiceDeadline - Date.now()) : null,
    choiceType: mockPendingChoice?.type ?? null,
  }),
  useChatMessages: () => ({
    messages: mockChatMessages,
    sendMessage: (...args: unknown[]) => mockSendChatMessage(...args),
    messageCount: mockChatMessages.length,
  }),
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

function createPlayers(): Player[] {
  return [
    {
      id: 'user-1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'user-2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

function createGameState(phase: GamePhase = 'movement'): GameState {
  const board = createEmptySquareBoard(8);
  const players = createPlayers();

  // Single stack for P1 at (0,0)
  const key = '0,0';
  (board.stacks as Map<string, any>).set(key, {
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 1,
  });

  return {
    id: 'game-123',
    boardType: 'square8',
    board,
    players,
    currentPhase: phase,
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };
}

function setBackendHostState(options: {
  phase?: GamePhase;
  validMoves?: Move[] | null;
  victoryState?: GameResult | null;
}) {
  mockGameState = createGameState(options.phase ?? 'movement');
  mockPlayers = mockGameState.players;
  mockCurrentPlayer = mockPlayers.find((p) => p.playerNumber === mockGameState!.currentPlayer);
  mockValidMoves = options.validMoves ?? null;
  mockVictoryState = options.victoryState ?? null;
  mockConnectionStatus = 'connected';
  mockIsConnecting = false;
  mockConnectionError = null;
  mockLastHeartbeatAt = Date.now();
}

// Utility: find a square-board cell by coordinates
function getSquareCell(x: number, y: number): HTMLButtonElement {
  const board = screen.getByTestId('board-view');
  const cell = board.querySelector<HTMLButtonElement>(`button[data-x="${x}"][data-y="${y}"]`);
  if (!cell) {
    throw new Error(`Failed to find board cell at (${x}, ${y})`);
  }
  return cell;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('BackendGameHost (React host behaviour)', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockGameState = null;
    mockValidMoves = null;
    mockVictoryState = null;
    mockPlayers = [];
    mockCurrentPlayer = undefined;

    mockConnectionStatus = 'connected';
    mockIsConnecting = false;
    mockConnectionError = null;
    mockLastHeartbeatAt = Date.now();

    mockSubmitMove = jest.fn();
    mockSendChatMessage = jest.fn();
    mockChatMessages = [];

    mockPendingChoice = null;
    mockChoiceDeadline = null;
    mockRespondToChoice = jest.fn();

    mockDecisionAutoResolved = null;
    mockDecisionPhaseTimeoutWarning = null;
    mockEvaluationHistory = [];
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 1. Happy-path move flow
  // ───────────────────────────────────────────────────────────────────────────

  it('submits a backend move when clicking source then target with a matching valid move', () => {
    const source: Position = { x: 0, y: 0 };
    const target: Position = { x: 0, y: 1 };

    const gameState = createGameState('movement');
    mockGameState = gameState;
    mockPlayers = gameState.players;
    mockCurrentPlayer = mockPlayers[0];

    const move: Move = {
      id: 'm1',
      type: 'move_stack',
      player: 1,
      from: source,
      to: target,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };
    mockValidMoves = [move];

    render(<BackendGameHost gameId="game-123" />);

    // Click source stack cell then target cell
    fireEvent.click(getSquareCell(source.x, source.y));
    fireEvent.click(getSquareCell(target.x, target.y));

    expect(mockSubmitMove).toHaveBeenCalledTimes(1);
    expect(mockSubmitMove).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'move_stack',
        from: source,
        to: target,
      })
    );

    // Instruction text should reflect movement phase
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();
  });

  it('treats non-player connections as spectators, blocks move submission, and shows spectator UX and evaluation panel', () => {
    const source: Position = { x: 0, y: 0 };
    const target: Position = { x: 0, y: 1 };

    const state = createGameState('movement');
    // Make both players non-matching to the authenticated user so BackendGameHost
    // treats this viewer as a spectator.
    const spectatorPlayers: Player[] = state.players.map((p, idx) => ({
      ...p,
      id: `other-${idx + 1}`,
    }));

    mockGameState = { ...state, players: spectatorPlayers };
    mockPlayers = spectatorPlayers;
    mockCurrentPlayer = spectatorPlayers[0];

    const move: Move = {
      id: 'm1',
      type: 'move_stack',
      player: 1,
      from: source,
      to: target,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };
    mockValidMoves = [move];

    // Seed a single evaluation snapshot so the panel has concrete content.
    mockEvaluationHistory = [
      {
        gameId: 'game-123',
        moveNumber: 5,
        boardType: 'square8',
        engineProfile: 'heuristic_v1_d5',
        evaluationScale: 'zero_sum_margin',
        perPlayer: {
          1: { totalEval: 1.5, territoryEval: 1.0, ringEval: 0.5 },
          2: { totalEval: -1.5, territoryEval: -1.0, ringEval: -0.5 },
        },
      },
    ];

    render(<BackendGameHost gameId="game-123" />);

    // Attempting to click a legal move as a spectator must not submit moves.
    fireEvent.click(getSquareCell(source.x, source.y));
    fireEvent.click(getSquareCell(target.x, target.y));

    expect(mockSubmitMove).not.toHaveBeenCalled();

    // Selection panel should communicate that moves are disabled while spectating.
    expect(screen.getByText('Moves disabled while spectating.')).toBeInTheDocument();

    // HUD should show a prominent spectator banner.
    expect(screen.getByText(/Spectator Mode/i)).toBeInTheDocument();

    // Evaluation panel should be visible for spectators when evaluation history is present.
    const evalPanel = screen.getByTestId('evaluation-panel');
    expect(evalPanel).toBeInTheDocument();
    expect(evalPanel).toHaveTextContent('AI Evaluation');
    expect(evalPanel).toHaveTextContent('Move 5');
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 2. Multi-phase flow + victory modal
  // ───────────────────────────────────────────────────────────────────────────

  it('updates instruction copy across phases and shows victory modal then banner', () => {
    const baseState = createGameState('ring_placement');

    // First render: ring placement, with some valid placement moves
    mockGameState = baseState;
    mockPlayers = baseState.players;
    mockCurrentPlayer = mockPlayers[0];
    mockValidMoves = [
      {
        id: 'p1',
        type: 'place_ring',
        player: 1,
        to: { x: 1, y: 1 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
    ];

    const { rerender } = render(<BackendGameHost gameId="game-123" />);

    // Ring placement instruction
    expect(
      screen.getByText('Place rings on an empty cell or on top of an existing stack.')
    ).toBeInTheDocument();

    // Auto-highlighted placement targets should be visible
    const board = screen.getByTestId('board-view');
    const highlighted = board.querySelector('.valid-move-cell');
    expect(highlighted).not.toBeNull();

    // Second render: line_processing phase
    const lineState = { ...baseState, currentPhase: 'line_processing' as GamePhase };
    mockGameState = lineState;
    rerender(<BackendGameHost gameId="game-123" />);

    expect(
      screen.getByText('Line processing – choose how to resolve your completed line.')
    ).toBeInTheDocument();

    // Third render: territory_processing phase with a victory result
    const victory: GameResult = {
      winner: 1,
      reason: 'territory_control',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };
    const territoryState = {
      ...baseState,
      currentPhase: 'territory_processing' as GamePhase,
      gameStatus: 'completed' as const,
    };
    mockGameState = territoryState;
    mockVictoryState = victory;

    rerender(<BackendGameHost gameId="game-123" />);

    // Territory instruction
    expect(
      screen.getByText('Territory processing – resolve disconnected regions.')
    ).toBeInTheDocument();

    // Victory modal should be open (Return to Lobby button visible)
    expect(screen.getByText('Return to Lobby')).toBeInTheDocument();

    // Close modal; host should keep a game-over banner instead
    fireEvent.click(screen.getByText('Close'));

    expect(screen.queryByText('Return to Lobby')).not.toBeInTheDocument();
    // Text comes from getGameOverBannerText; assert the specific banner text for clarity.
    expect(screen.getByText(/Game over – victory by territory control\./i)).toBeInTheDocument();
  });

  it('announces last-player-standing victories clearly to screen readers', async () => {
    const lpsResult: GameResult = {
      winner: 1,
      reason: 'last_player_standing',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    setBackendHostState({ phase: 'movement', victoryState: lpsResult });

    render(<BackendGameHost gameId="game-123" />);

    await act(async () => {
      // Wait for the screen reader announcer to receive the victory message.
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');
    const combinedText = `${region1.textContent ?? ''} ${region2.textContent ?? ''}`;

    expect(combinedText).toMatch(/last player standing/i);
  });

  it('logs phase changes for line_processing → chain_capture → territory_processing and updates instructions', () => {
    // Initial state: line_processing with no pending choice
    const stateLine = createGameState('line_processing');
    mockGameState = stateLine;
    mockPlayers = stateLine.players;
    mockCurrentPlayer = mockPlayers[0];
    mockValidMoves = [];
    mockPendingChoice = null;

    const { rerender } = render(<BackendGameHost gameId="game-123" />);

    // Instruction for line_processing
    expect(
      screen.getByText('Line processing – choose how to resolve your completed line.')
    ).toBeInTheDocument();

    // Transition to chain_capture with a pending capture_direction choice
    const stateChain: GameState = { ...stateLine, currentPhase: 'chain_capture' as GamePhase };
    mockGameState = stateChain;
    mockPendingChoice = {
      id: 'capdir-1',
      playerNumber: 1,
      type: 'capture_direction',
      prompt: 'Select capture direction',
      options: [],
    } as any;

    rerender(<BackendGameHost gameId="game-123" />);

    // Instruction for chain_capture
    expect(
      screen.getByText('Chain capture in progress – select next capture target.')
    ).toBeInTheDocument();

    // Transition to territory_processing and clear choice
    const stateTerritory: GameState = {
      ...stateLine,
      currentPhase: 'territory_processing' as GamePhase,
    };
    mockGameState = stateTerritory;
    mockPendingChoice = null;

    rerender(<BackendGameHost gameId="game-123" />);

    // Instruction for territory_processing
    expect(
      screen.getByText('Territory processing – resolve disconnected regions.')
    ).toBeInTheDocument();

    // Event log should reflect the phase progression in order.
    const log = screen.getByTestId('game-event-log');

    expect(log).toHaveTextContent(/Phase: line_processing/);
    expect(log).toHaveTextContent(/Phase changed: line_processing → chain_capture/);
    expect(log).toHaveTextContent(/Phase changed: chain_capture → territory_processing/);
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 3. Decision dialog behaviour
  // ───────────────────────────────────────────────────────────────────────────

  it('renders ChoiceDialog for pending choice and responds when an option is selected', () => {
    setBackendHostState({ phase: 'line_processing', validMoves: [] });

    const pending: PlayerChoice = {
      id: 'choice-1',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'How do you want to process this line?',
      timeoutMs: 10_000,
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    } as any;

    mockPendingChoice = pending;
    mockChoiceDeadline = Date.now() + 10_000;

    jest.useFakeTimers();

    render(<BackendGameHost gameId="game-123" />);

    // Ensure the line reward options are visible (using actual button text)
    expect(screen.getByText('Full Collapse + Elimination Bonus')).toBeInTheDocument();
    expect(screen.getByText('Minimum Collapse')).toBeInTheDocument();

    // Click the first option
    fireEvent.click(screen.getByText('Full Collapse + Elimination Bonus'));

    // Respond hook should be called with the selected option
    expect(mockRespondToChoice).toHaveBeenCalledTimes(1);
    expect(mockRespondToChoice).toHaveBeenCalledWith('option_1_collapse_all_and_eliminate');

    // Countdown copy should be present while choice is pending
    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(screen.getByText(/Respond within/i)).toBeInTheDocument();

    jest.useRealTimers();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 4. Connection and error banners
  // ───────────────────────────────────────────────────────────────────────────

  it('shows a connecting shell before initial game state is available', () => {
    mockGameState = null;
    mockValidMoves = null;
    mockVictoryState = null;
    mockIsConnecting = true;
    mockConnectionStatus = 'connecting';

    render(<BackendGameHost gameId="game-connecting" />);

    expect(screen.getByText('Connecting to game…')).toBeInTheDocument();
    expect(screen.getByText(/Game ID: game-connecting/)).toBeInTheDocument();
  });

  it('shows reconnection banner when connection is interrupted during an active game', () => {
    setBackendHostState({ phase: 'movement', validMoves: [] });
    mockConnectionStatus = 'reconnecting';
    mockIsConnecting = true;

    render(<BackendGameHost gameId="game-123" />);

    expect(screen.getByText(/Connection lost. Attempting to reconnect…/)).toBeInTheDocument();
  });

  it('shows an error view when the backend reports a connection error before any game state', () => {
    mockGameState = null;
    mockConnectionStatus = 'disconnected';
    mockIsConnecting = false;
    mockConnectionError = 'Game not found';

    render(<BackendGameHost gameId="missing-game" />);

    expect(screen.getByText('Unable to load game')).toBeInTheDocument();
    expect(screen.getByText('Game not found')).toBeInTheDocument();
    expect(screen.getByText(/Game ID: missing-game/)).toBeInTheDocument();
  });

  it('surfaces a stale connection hint when last heartbeat is old but status is still connected', () => {
    const state = createGameState('movement');
    mockGameState = state;
    mockPlayers = state.players;
    mockCurrentPlayer = mockPlayers[0];
    mockValidMoves = [];
    mockConnectionStatus = 'connected';
    mockLastHeartbeatAt = Date.now() - 10_000; // beyond HEARTBEAT_STALE_THRESHOLD

    render(<BackendGameHost gameId="game-123" />);

    // GameHUD uses an "awaiting update…" hint when connection is stale
    expect(screen.getByText(/awaiting update…/i)).toBeInTheDocument();
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 5. Diagnostics event log and chat
  // ───────────────────────────────────────────────────────────────────────────

  it('feeds diagnostics events into GameEventLog (phase and choice transitions)', () => {
    // Initial state: ring placement, no pending choice
    const state1 = createGameState('ring_placement');
    mockGameState = state1;
    mockPlayers = state1.players;
    mockCurrentPlayer = mockPlayers[0];
    mockValidMoves = [];
    mockPendingChoice = null;

    const { rerender } = render(<BackendGameHost gameId="game-123" />);

    // Next render: movement phase with a pending choice
    const state2 = { ...state1, currentPhase: 'movement' as GamePhase };
    mockGameState = state2;
    mockPendingChoice = {
      id: 'choice-1',
      playerNumber: 1,
      type: 'region_order',
      prompt: 'Pick a region to process',
      options: [],
    } as any;

    rerender(<BackendGameHost gameId="game-123" />);

    const log = screen.getByTestId('game-event-log');

    expect(log).toHaveTextContent(/Phase: ring_placement/);
    expect(log).toHaveTextContent(/Phase changed: ring_placement → movement/);
    expect(log).toHaveTextContent(/Choice requested: region_order for P1/);
  });

  it('renders chat log and calls sendChatMessage when submitting a message', () => {
    setBackendHostState({ phase: 'movement', validMoves: [] });

    mockChatMessages = [{ sender: 'Alice', text: 'Hello world' }];

    render(<BackendGameHost gameId="game-123" />);

    // Existing message rendered
    expect(screen.getByText(/Alice:/)).toBeInTheDocument();
    expect(screen.getByText('Hello world')).toBeInTheDocument();

    const input = screen.getByPlaceholderText('Type a message...') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'New chat message' } });

    fireEvent.submit(input.closest('form') as HTMLFormElement);

    expect(mockSendChatMessage).toHaveBeenCalledTimes(1);
    expect(mockSendChatMessage).toHaveBeenCalledWith('New chat message');
  });

  it('logs and surfaces auto-resolved decisions when metadata is present', () => {
    // Initial state: line processing with no pending choice, then an auto-resolve
    setBackendHostState({ phase: 'line_processing', validMoves: [] });

    mockDecisionAutoResolved = {
      choiceType: 'line_reward_option',
      choiceKind: 'line_reward',
      actingPlayerNumber: 1,
      resolvedMoveId: 'auto-move-1',
      reason: 'timeout',
    } as any;

    render(<BackendGameHost gameId="game-123" />);

    // HUD cue and system log should both render a human-readable auto-resolve summary
    const autoResolveTexts = screen.getAllByText(
      /Decision auto-resolved for P1: line reward \(reason: timeout\)/i
    );
    expect(autoResolveTexts.length).toBeGreaterThanOrEqual(1);

    // At least one instance should be present in the HUD/banner area, and the
    // Game Event Log should also contain the same summary text.
    expect(autoResolveTexts[0]).toBeInTheDocument();

    // System event log should also contain the same summary text
    const log = screen.getByTestId('game-event-log');
    expect(log).toHaveTextContent(
      /Decision auto-resolved for P1: line reward \(reason: timeout\)/i
    );
  });

  it('highlights capture_direction options and routes choice responses', () => {
    // Backend chain-capture decisions use capture_direction choices; the
    // host should highlight both target and landing cells and route the
    // selected option back through respondToChoice.
    setBackendHostState({ phase: 'capture', validMoves: [] });

    mockPendingChoice = {
      id: 'capdir-1',
      playerNumber: 1,
      type: 'capture_direction',
      prompt: 'Select capture direction',
      timeoutMs: 5000,
      options: [
        {
          targetPosition: { x: 1, y: 1 },
          landingPosition: { x: 2, y: 2 },
          capturedCapHeight: 2,
        },
      ],
    } as any;
    mockChoiceDeadline = Date.now() + 5000;

    render(<BackendGameHost gameId="game-123" />);

    // Landing cell should be highlighted as a primary decision target.
    const landingCell = getSquareCell(2, 2);
    expect(landingCell).toHaveAttribute('data-decision-highlight', 'primary');

    // Capture target cell should be highlighted as a secondary decision target.
    const targetCell = getSquareCell(1, 1);
    expect(targetCell).toHaveAttribute('data-decision-highlight', 'secondary');

    // ChoiceDialog should render a direction option for this capture.
    const optionButton = screen.getByText(/Direction 1: target \(1, 1\).*cap 2/i);
    expect(optionButton).toBeInTheDocument();

    fireEvent.click(optionButton);

    expect(mockRespondToChoice).toHaveBeenCalledTimes(1);
    const selected = mockRespondToChoice.mock.calls[0][0];
    expect(selected).toMatchObject({
      targetPosition: { x: 1, y: 1 },
      landingPosition: { x: 2, y: 2 },
      capturedCapHeight: 2,
    });
  });

  it('highlights ring_elimination (forced-elimination) stacks and routes choice responses', () => {
    // Backend forced-elimination and self-elimination decisions are surfaced
    // as ring_elimination PlayerChoices. BackendGameHost should highlight
    // the eligible stacks and route the selected option back through
    // respondToChoice.
    setBackendHostState({ phase: 'movement', validMoves: [] });

    mockPendingChoice = {
      id: 'elim-1',
      playerNumber: 1,
      type: 'ring_elimination',
      prompt: 'Choose which stack to eliminate from',
      timeoutMs: 5000,
      options: [
        {
          stackPosition: { x: 0, y: 0 },
          capHeight: 2,
          totalHeight: 3,
          moveId: 'eliminate-0-0',
        },
      ],
    } as any;
    mockChoiceDeadline = Date.now() + 5000;

    render(<BackendGameHost gameId="game-123" />);

    // The elimination target stack should be highlighted on the board.
    const highlightedCell = getSquareCell(0, 0);
    expect(highlightedCell).toHaveAttribute('data-decision-highlight', 'primary');

    // ChoiceDialog should render a button for the elimination stack.
    const optionButton = screen.getByText(/Stack at \(0, 0\).*cap 2, total 3/);
    expect(optionButton).toBeInTheDocument();

    // Selecting the option should route through the pending-choice responder.
    fireEvent.click(optionButton);

    expect(mockRespondToChoice).toHaveBeenCalledTimes(1);
    const selected = mockRespondToChoice.mock.calls[0][0];
    expect(selected).toMatchObject({
      stackPosition: { x: 0, y: 0 },
      capHeight: 2,
      totalHeight: 3,
      moveId: 'eliminate-0-0',
    });
  });

  it('surfaces decision-phase timeout warnings when metadata is present', () => {
    // Initial state: line processing with a pending choice for P1
    setBackendHostState({ phase: 'line_processing', validMoves: [] });

    mockPendingChoice = {
      id: 'choice-1',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'How do you want to process this line?',
      timeoutMs: 5000,
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    } as any;

    mockDecisionPhaseTimeoutWarning = {
      type: 'decision_phase_timeout_warning',
      data: {
        gameId: 'game-123',
        playerNumber: 1,
        phase: 'line_processing',
        remainingMs: 5000,
        choiceId: 'choice-1',
      },
      timestamp: new Date().toISOString(),
    } as any;

    render(<BackendGameHost gameId="game-123" />);

    // HUD instruction should prioritise the impending timeout warning for the
    // current player whose decision is at risk of auto-resolution.
    expect(
      screen.getByText(/This decision will be auto-resolved in about 5 seconds/i)
    ).toBeInTheDocument();

    // System event log should include a structured timeout warning entry.
    const log = screen.getByTestId('game-event-log');
    expect(log).toHaveTextContent(/Decision timeout warning: P1 in line_processing/i);
  });

  it('caps client-side decision countdown using server timeout warning metadata', () => {
    jest.useFakeTimers();

    const now = Date.now();
    jest.setSystemTime(now);

    // Initial state: line processing with a pending choice for P1
    setBackendHostState({ phase: 'line_processing', validMoves: [] });

    mockPendingChoice = {
      id: 'choice-1',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'How do you want to process this line?',
      timeoutMs: 10000,
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    } as any;

    // Baseline client countdown (from choiceDeadline) would be ~10s, but the
    // server warning reports a shorter remainingMs, which should cap the
    // effective countdown used by the HUD and ChoiceDialog.
    mockChoiceDeadline = now + 10_000;

    mockDecisionPhaseTimeoutWarning = {
      type: 'decision_phase_timeout_warning',
      data: {
        gameId: 'game-123',
        playerNumber: 1,
        phase: 'line_processing',
        remainingMs: 3000,
        choiceId: 'choice-1',
      },
      timestamp: new Date().toISOString(),
    } as any;

    render(<BackendGameHost gameId="game-123" />);

    // The ChoiceDialog countdown should reflect the *capped* effective
    // remaining time derived from the server warning (~3s), not the original
    // ~10s client baseline. The dialog should also surface server-capped
    // semantics in its copy and data attributes.
    const countdown = screen.getByTestId('choice-countdown');
    expect(countdown).toHaveAttribute('data-server-capped', 'true');
    expect(countdown).toHaveAttribute('data-severity', 'critical');

    expect(screen.getByText('Server deadline – respond within')).toBeInTheDocument();
    expect(screen.getByText('3s')).toBeInTheDocument();

    jest.useRealTimers();
  });

  it('does not treat decision as server-capped when server remainingMs exceeds local baseline', () => {
    jest.useFakeTimers();

    const now = Date.now();
    jest.setSystemTime(now);

    // Initial state: line processing with a pending choice for P1
    setBackendHostState({ phase: 'line_processing', validMoves: [] });

    mockPendingChoice = {
      id: 'choice-2',
      playerNumber: 1,
      type: 'line_reward_option',
      prompt: 'How do you want to process this line?',
      timeoutMs: 3000,
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    } as any;

    // Baseline client countdown is ~3s, while the server warning reports a
    // longer remainingMs. In this case the effective countdown should still be
    // ~3s but the decision must *not* be treated as server-capped.
    mockChoiceDeadline = now + 3_000;

    mockDecisionPhaseTimeoutWarning = {
      type: 'decision_phase_timeout_warning',
      data: {
        gameId: 'game-123',
        playerNumber: 1,
        phase: 'line_processing',
        remainingMs: 10_000,
        choiceId: 'choice-2',
      },
      timestamp: new Date().toISOString(),
    } as any;

    render(<BackendGameHost gameId="game-123" />);

    const countdown = screen.getByTestId('choice-countdown');
    expect(countdown).not.toHaveAttribute('data-server-capped');
    expect(countdown).toHaveAttribute('data-severity', 'critical');

    // Default (non-server-capped) label should be used.
    expect(screen.getByText('Respond within')).toBeInTheDocument();
    expect(screen.getByText('3s')).toBeInTheDocument();

    jest.useRealTimers();
  });
});
