/**
 * Unit tests for useGameState hook
 *
 * Tests read-only access to game state and view model transformations.
 */

import { renderHook } from '@testing-library/react';
import {
  useGameState,
  useHUDViewModel,
  useBoardViewModel,
  useEventLogViewModel,
  useVictoryViewModel,
  useGamePhase,
  useGameStatus,
} from '@/client/hooks/useGameState';
import type { GameState, GameResult, Move, Player, BoardState } from '@/shared/types/game';

// ─────────────────────────────────────────────────────────────────────────────
// Mock Setup
// ─────────────────────────────────────────────────────────────────────────────

const createMockBoard = (): BoardState => ({
  stacks: new Map(),
  markers: new Map(),
  collapsedSpaces: new Map(),
  territories: new Map(),
  formedLines: [],
  eliminatedRings: {},
  size: 8,
  type: 'square8',
});

const createMockPlayer = (playerNumber: number): Player => ({
  playerNumber,
  userId: `user-${playerNumber}`,
  rings: 3,
  score: 0,
  stacks: 0,
  markers: 0,
  isEliminated: false,
  ownedCells: 0,
  color: playerNumber === 1 ? 'red' : 'blue',
  username: `Player ${playerNumber}`,
});

const createMockGameState = (overrides: Partial<GameState> = {}): GameState => ({
  gameId: 'game-123',
  gameStatus: 'active',
  currentPhase: 'movement',
  currentPlayer: 1,
  turnNumber: 5,
  players: [createMockPlayer(1), createMockPlayer(2)],
  board: createMockBoard(),
  moveHistory: [],
  history: [],
  config: {
    boardType: 'square8',
    maxPlayers: 2,
    timeControl: null,
    ringsPerPlayer: 3,
    linesForWin: 3,
    variant: 'standard',
  },
  ...overrides,
});

const createMockGameContext = (overrides: Record<string, unknown> = {}) => ({
  gameId: 'game-123',
  gameState: createMockGameState(),
  validMoves: null,
  victoryState: null,
  // No auto-resolve metadata by default; tests can override as needed.
  decisionAutoResolved: null,
   // No timeout warning metadata by default; tests can override as needed.
  decisionPhaseTimeoutWarning: null,
  connectionStatus: 'connected' as const,
  lastHeartbeatAt: Date.now(),
  error: null,
  isConnecting: false,
  connectToGame: jest.fn(),
  disconnect: jest.fn(),
  pendingChoice: null,
  choiceDeadline: null,
  respondToChoice: jest.fn(),
  submitMove: jest.fn(),
  sendChatMessage: jest.fn(),
  chatMessages: [],
  ...overrides,
});

let mockContextValue = createMockGameContext();

jest.mock('@/client/contexts/GameContext', () => ({
  useGame: () => mockContextValue,
}));

// Mock the view model adapters
jest.mock('@/client/adapters/gameViewModels', () => ({
  toHUDViewModel: jest.fn((gameState, options) => ({
    currentPlayer: gameState?.currentPlayer ?? 1,
    phase: gameState?.currentPhase ?? 'movement',
    instruction: options?.instruction ?? 'Make your move',
    connectionStatus: options?.connectionStatus ?? 'connected',
  })),
  toBoardViewModel: jest.fn((board, options) => ({
    cells: [],
    selectedPosition: options?.selectedPosition ?? null,
    validTargets: options?.validTargets ?? [],
    boardType: board?.type ?? 'square8',
  })),
  toEventLogViewModel: jest.fn((history, systemEvents, victoryState, options) => ({
    entries: [],
    hasVictory: !!victoryState,
  })),
  toVictoryViewModel: jest.fn((victoryResult, players, gameState, options) => ({
    winner: victoryResult?.winner ?? null,
    reason: victoryResult?.reason ?? 'unknown',
    isDismissed: options?.isDismissed ?? false,
  })),
}));

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useGameState
// ─────────────────────────────────────────────────────────────────────────────

describe('useGameState', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns all expected state properties', () => {
    const { result } = renderHook(() => useGameState());

    expect(result.current.gameId).toBeDefined();
    expect(result.current.gameState).toBeDefined();
    expect(result.current.validMoves).toBeDefined();
    expect(result.current.victoryState).toBeDefined();
    expect(result.current.players).toBeDefined();
    expect(result.current.currentPlayer).toBeDefined();
  });

  it('returns gameId from context', () => {
    mockContextValue = createMockGameContext({ gameId: 'my-game-abc' });
    const { result } = renderHook(() => useGameState());

    expect(result.current.gameId).toBe('my-game-abc');
  });

  it('returns null gameId when not connected', () => {
    mockContextValue = createMockGameContext({ gameId: null, gameState: null });
    const { result } = renderHook(() => useGameState());

    expect(result.current.gameId).toBeNull();
    expect(result.current.gameState).toBeNull();
  });

  it('returns players array from game state', () => {
    const gameState = createMockGameState();
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameState());

    expect(result.current.players).toHaveLength(2);
    expect(result.current.players[0].playerNumber).toBe(1);
    expect(result.current.players[1].playerNumber).toBe(2);
  });

  it('returns empty players array when no game state', () => {
    mockContextValue = createMockGameContext({ gameState: null });
    const { result } = renderHook(() => useGameState());

    expect(result.current.players).toEqual([]);
  });

  it('returns currentPlayer matching current turn', () => {
    const gameState = createMockGameState({ currentPlayer: 2 });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameState());

    expect(result.current.currentPlayer?.playerNumber).toBe(2);
  });

  it('returns undefined currentPlayer when no match', () => {
    const gameState = createMockGameState({ currentPlayer: 3 }); // No player 3
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameState());

    expect(result.current.currentPlayer).toBeUndefined();
  });

  it('returns valid moves from context', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });
    const { result } = renderHook(() => useGameState());

    expect(result.current.validMoves).toEqual(validMoves);
  });

  it('returns victoryState when game is over', () => {
    const victoryState: GameResult = {
      winner: 1,
      reason: 'lines_completed',
      finalScores: { 1: 3, 2: 1 },
    };
    mockContextValue = createMockGameContext({ victoryState });
    const { result } = renderHook(() => useGameState());

    expect(result.current.victoryState).toEqual(victoryState);
  });

  it('exposes decisionAutoResolved metadata from context when present', () => {
    const decisionAutoResolved = {
      choiceType: 'line_reward_option',
      choiceKind: 'line_reward',
      actingPlayerNumber: 1,
      resolvedMoveId: 'move-123',
      reason: 'timeout',
    } as any;

    mockContextValue = createMockGameContext({ decisionAutoResolved });
    const { result } = renderHook(() => useGameState());

    expect(result.current.decisionAutoResolved).toEqual(decisionAutoResolved);
  });

  it('exposes decisionPhaseTimeoutWarning metadata from context when present', () => {
    const decisionPhaseTimeoutWarning = {
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

    mockContextValue = createMockGameContext({ decisionPhaseTimeoutWarning });
    const { result } = renderHook(() => useGameState());

    expect(result.current.decisionPhaseTimeoutWarning).toEqual(decisionPhaseTimeoutWarning);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useHUDViewModel
// ─────────────────────────────────────────────────────────────────────────────

describe('useHUDViewModel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns null when no game state', () => {
    mockContextValue = createMockGameContext({ gameState: null });
    const { result } = renderHook(() => useHUDViewModel());

    expect(result.current).toBeNull();
  });

  it('returns view model when game state exists', () => {
    const { result } = renderHook(() => useHUDViewModel());

    expect(result.current).not.toBeNull();
    expect(result.current?.currentPlayer).toBe(1);
    expect(result.current?.phase).toBe('movement');
  });

  it('passes instruction option to view model', () => {
    const { result } = renderHook(() => useHUDViewModel({ instruction: 'Select a ring to move' }));

    expect(result.current?.instruction).toBe('Select a ring to move');
  });

  it('includes connection status in view model options', () => {
    mockContextValue = createMockGameContext({ connectionStatus: 'reconnecting' });
    const { result } = renderHook(() => useHUDViewModel());

    expect(result.current?.connectionStatus).toBe('reconnecting');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useBoardViewModel
// ─────────────────────────────────────────────────────────────────────────────

describe('useBoardViewModel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns null when no board state', () => {
    mockContextValue = createMockGameContext({
      gameState: { ...createMockGameState(), board: null },
    });
    const { result } = renderHook(() => useBoardViewModel());

    expect(result.current).toBeNull();
  });

  it('returns board view model with default options', () => {
    const { result } = renderHook(() => useBoardViewModel());

    expect(result.current).not.toBeNull();
    expect(result.current?.boardType).toBe('square8');
  });

  it('passes selectedPosition to view model', () => {
    const selectedPosition = { x: 3, y: 4 };
    const { result } = renderHook(() => useBoardViewModel({ selectedPosition }));

    expect(result.current?.selectedPosition).toEqual(selectedPosition);
  });

  it('passes validTargets to view model', () => {
    const validTargets = [
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const { result } = renderHook(() => useBoardViewModel({ validTargets }));

    expect(result.current?.validTargets).toEqual(validTargets);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useEventLogViewModel
// ─────────────────────────────────────────────────────────────────────────────

describe('useEventLogViewModel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns event log view model', () => {
    const { result } = renderHook(() => useEventLogViewModel());

    expect(result.current).toBeDefined();
    expect(result.current.entries).toBeDefined();
  });

  it('includes victory state in event log', () => {
    const victoryState: GameResult = {
      winner: 2,
      reason: 'elimination',
    };
    mockContextValue = createMockGameContext({ victoryState });
    const { result } = renderHook(() => useEventLogViewModel());

    expect(result.current.hasVictory).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useVictoryViewModel
// ─────────────────────────────────────────────────────────────────────────────

describe('useVictoryViewModel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns null when no victory state', () => {
    mockContextValue = createMockGameContext({ victoryState: null });
    const { result } = renderHook(() => useVictoryViewModel());

    expect(result.current).toBeNull();
  });

  it('returns victory view model when game is over', () => {
    const victoryState: GameResult = {
      winner: 1,
      reason: 'lines_completed',
    };
    mockContextValue = createMockGameContext({ victoryState });
    const { result } = renderHook(() => useVictoryViewModel());

    expect(result.current).not.toBeNull();
    expect(result.current?.winner).toBe(1);
    expect(result.current?.reason).toBe('lines_completed');
  });

  it('passes isDismissed option to view model', () => {
    const victoryState: GameResult = {
      winner: 2,
      reason: 'elimination',
    };
    mockContextValue = createMockGameContext({ victoryState });
    const { result } = renderHook(() => useVictoryViewModel({ isDismissed: true }));

    expect(result.current?.isDismissed).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useGamePhase
// ─────────────────────────────────────────────────────────────────────────────

describe('useGamePhase', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns current phase', () => {
    const gameState = createMockGameState({ currentPhase: 'capture' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.phase).toBe('capture');
  });

  it('returns null phase when no game state', () => {
    mockContextValue = createMockGameContext({ gameState: null });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.phase).toBeNull();
  });

  it('returns isPlacementPhase true during ring_placement', () => {
    const gameState = createMockGameState({ currentPhase: 'ring_placement' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.isPlacementPhase).toBe(true);
    expect(result.current.isMovementPhase).toBe(false);
  });

  it('returns isMovementPhase true during movement', () => {
    const gameState = createMockGameState({ currentPhase: 'movement' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.isMovementPhase).toBe(true);
    expect(result.current.isPlacementPhase).toBe(false);
  });

  it('returns isCapturePhase true during capture', () => {
    const gameState = createMockGameState({ currentPhase: 'capture' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.isCapturePhase).toBe(true);
  });

  it('returns isChainCapturePhase true during chain_capture', () => {
    const gameState = createMockGameState({ currentPhase: 'chain_capture' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGamePhase());

    expect(result.current.isChainCapturePhase).toBe(true);
  });

  it('returns isInProcessingPhase true during processing phases', () => {
    let gameState = createMockGameState({ currentPhase: 'line_processing' });
    mockContextValue = createMockGameContext({ gameState });
    let { result } = renderHook(() => useGamePhase());

    expect(result.current.isLineProcessingPhase).toBe(true);
    expect(result.current.isInProcessingPhase).toBe(true);

    gameState = createMockGameState({ currentPhase: 'territory_processing' });
    mockContextValue = createMockGameContext({ gameState });
    ({ result } = renderHook(() => useGamePhase()));

    expect(result.current.isTerritoryProcessingPhase).toBe(true);
    expect(result.current.isInProcessingPhase).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useGameStatus
// ─────────────────────────────────────────────────────────────────────────────

describe('useGameStatus', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns current status', () => {
    const gameState = createMockGameState({ gameStatus: 'active' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.status).toBe('active');
  });

  it('returns waiting status when no game state', () => {
    mockContextValue = createMockGameContext({ gameState: null });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.status).toBe('waiting');
    expect(result.current.isWaiting).toBe(true);
  });

  it('returns isActive true during active game', () => {
    const gameState = createMockGameState({ gameStatus: 'active' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.isActive).toBe(true);
    expect(result.current.isFinished).toBe(false);
  });

  it('returns isFinished true when game is finished', () => {
    const gameState = createMockGameState({ gameStatus: 'finished' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.isFinished).toBe(true);
    expect(result.current.isActive).toBe(false);
  });

  it('returns isFinished true when game is completed', () => {
    const gameState = createMockGameState({ gameStatus: 'completed' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.isFinished).toBe(true);
  });

  it('returns isPaused true when game is paused', () => {
    const gameState = createMockGameState({ gameStatus: 'paused' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.isPaused).toBe(true);
  });

  it('returns isAbandoned true when game is abandoned', () => {
    const gameState = createMockGameState({ gameStatus: 'abandoned' });
    mockContextValue = createMockGameContext({ gameState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.isAbandoned).toBe(true);
  });

  it('returns hasVictory true when victoryState exists', () => {
    const victoryState: GameResult = {
      winner: 1,
      reason: 'lines_completed',
    };
    mockContextValue = createMockGameContext({ victoryState });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.hasVictory).toBe(true);
    expect(result.current.winner).toBe(1);
  });

  it('returns hasVictory false when no victoryState', () => {
    mockContextValue = createMockGameContext({ victoryState: null });
    const { result } = renderHook(() => useGameStatus());

    expect(result.current.hasVictory).toBe(false);
    expect(result.current.winner).toBeUndefined();
  });
});
