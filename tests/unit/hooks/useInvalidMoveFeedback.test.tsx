/**
 * Unit tests for useInvalidMoveFeedback hook
 *
 * Verifies:
 * - shakingCellKey is set and auto-cleared after the configured duration
 * - toast notifications are emitted with contextual messages when enabled
 * - clearShake cancels any pending timeout and clears state
 * - analyzeInvalidMove returns appropriate InvalidMoveReason values
 */

import { renderHook, act } from '@testing-library/react';
import { toast } from 'react-hot-toast';
import {
  useInvalidMoveFeedback,
  analyzeInvalidMove,
  getReasonExplanation,
  type InvalidMoveReason,
} from '@/client/hooks/useInvalidMoveFeedback';
import type { GameState, BoardState, Position } from '@/shared/types/game';
import { positionToString } from '@/shared/types/game';

// ─────────────────────────────────────────────────────────────────────────────
// Mock Setup
// ─────────────────────────────────────────────────────────────────────────────

jest.mock('react-hot-toast', () => ({
  toast: {
    error: jest.fn(),
  },
}));

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

const createMockGameState = (overrides: Partial<GameState> = {}): GameState => ({
  gameId: 'game-123',
  gameStatus: 'active',
  currentPhase: 'movement',
  currentPlayer: 1,
  turnNumber: 1,
  players: [
    {
      playerNumber: 1,
      userId: 'user-1',
      rings: 3,
      score: 0,
      stacks: 0,
      markers: 0,
      isEliminated: false,
      ownedCells: 0,
      color: 'red',
      username: 'Player 1',
    },
    {
      playerNumber: 2,
      userId: 'user-2',
      rings: 3,
      score: 0,
      stacks: 0,
      markers: 0,
      isEliminated: false,
      ownedCells: 0,
      color: 'blue',
      username: 'Player 2',
    },
  ],
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

const createPosition = (x: number, y: number): Position => ({ x, y });

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: getReasonExplanation
// ─────────────────────────────────────────────────────────────────────────────

describe('getReasonExplanation', () => {
  it('returns human-readable messages for known reasons', () => {
    const cases: Array<[InvalidMoveReason, string]> = [
      ['not_your_turn', "It's not your turn yet"],
      ['spectator', 'Spectators cannot make moves'],
      ['disconnected', 'Reconnecting to server...'],
      ['empty_cell_in_movement', 'Select a stack to move, not an empty cell'],
      ['opponent_stack', "You cannot move your opponent's pieces"],
      ['invalid_placement_position', 'Rings cannot be placed here'],
      ['stack_on_stack_not_allowed', 'Cannot place rings on existing stacks right now'],
      ['must_move_forced_stack', 'You must move the highlighted stack'],
      ['chain_capture_must_continue', 'You must continue the chain capture'],
    ];

    for (const [reason, expected] of cases) {
      expect(getReasonExplanation(reason)).toBe(expected);
    }

    expect(getReasonExplanation('unknown')).toBe('Invalid move');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useInvalidMoveFeedback (hook behaviour)
// ─────────────────────────────────────────────────────────────────────────────

describe('useInvalidMoveFeedback', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
  });

  it('sets shakingCellKey when triggerInvalidMove is called and auto-clears after duration', () => {
    jest.useFakeTimers();

    const { result } = renderHook(() =>
      useInvalidMoveFeedback({ shakeDurationMs: 500, showToast: false })
    );

    const position = createPosition(2, 3);
    const expectedKey = positionToString(position);

    act(() => {
      result.current.triggerInvalidMove(position, 'out_of_range');
    });

    expect(result.current.shakingCellKey).toBe(expectedKey);

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current.shakingCellKey).toBeNull();
  });

  it('emits a toast with the expected message when showToast is true', () => {
    jest.useFakeTimers();

    const { result } = renderHook(() => useInvalidMoveFeedback({ showToast: true }));

    const position = createPosition(1, 1);
    const reason: InvalidMoveReason = 'not_your_turn';
    const expectedMessage = getReasonExplanation(reason);

    act(() => {
      result.current.triggerInvalidMove(position, reason);
    });

    const errorMock = toast.error as jest.Mock;
    expect(errorMock).toHaveBeenCalledTimes(1);
    expect(errorMock).toHaveBeenCalledWith(
      expectedMessage,
      expect.objectContaining({ id: 'invalid-move' })
    );
  });

  it('does not emit a toast when showToast is false', () => {
    jest.useFakeTimers();

    const { result } = renderHook(() =>
      useInvalidMoveFeedback({ showToast: false, shakeDurationMs: 400 })
    );

    const position = createPosition(0, 0);

    act(() => {
      result.current.triggerInvalidMove(position, 'invalid_placement_position');
    });

    const errorMock = toast.error as jest.Mock;
    expect(errorMock).not.toHaveBeenCalled();
  });

  it('clearShake immediately clears shakingCellKey and cancels pending timeout', () => {
    jest.useFakeTimers();

    const { result } = renderHook(() =>
      useInvalidMoveFeedback({ shakeDurationMs: 1000, showToast: false })
    );

    const position = createPosition(4, 5);

    act(() => {
      result.current.triggerInvalidMove(position, 'out_of_range');
    });

    expect(result.current.shakingCellKey).toBe(positionToString(position));

    act(() => {
      result.current.clearShake();
    });

    expect(result.current.shakingCellKey).toBeNull();

    act(() => {
      jest.advanceTimersByTime(2000);
    });

    // Should remain null because the timeout was cleared
    expect(result.current.shakingCellKey).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: analyzeInvalidMove
// ─────────────────────────────────────────────────────────────────────────────

describe('analyzeInvalidMove', () => {
  const basePosition = createPosition(1, 1);

  it('returns spectator when user is not a player', () => {
    const gameState = createMockGameState();

    const reason = analyzeInvalidMove(gameState, basePosition, {
      isPlayer: false,
    });

    expect(reason).toBe('spectator');
  });

  it('returns disconnected when connection is not active', () => {
    const gameState = createMockGameState();

    const reason = analyzeInvalidMove(gameState, basePosition, {
      isConnected: false,
    });

    expect(reason).toBe('disconnected');
  });

  it('returns game_not_active when game is finished', () => {
    const gameState = createMockGameState({ gameStatus: 'finished' });

    const reason = analyzeInvalidMove(gameState, basePosition, {});

    expect(reason).toBe('game_not_active');
  });

  it('returns not_your_turn when isMyTurn is false', () => {
    const gameState = createMockGameState();

    const reason = analyzeInvalidMove(gameState, basePosition, {
      isMyTurn: false,
    });

    expect(reason).toBe('not_your_turn');
  });

  it('returns empty_cell_in_movement when selecting an empty cell in movement phase', () => {
    const gameState = createMockGameState({ currentPhase: 'movement' });

    const reason = analyzeInvalidMove(gameState, basePosition, {
      validMoves: [],
      selectedPosition: null,
    });

    expect(reason).toBe('empty_cell_in_movement');
  });

  it('distinguishes opponent_stack vs no_valid_moves_from_here based on controlling player', () => {
    const posKey = positionToString(basePosition);

    // Opponent stack with no moves
    const opponentBoard: BoardState = {
      ...createMockBoard(),
      stacks: new Map([[posKey, { controllingPlayer: 2 } as any]]),
    };

    const opponentGameState = createMockGameState({
      board: opponentBoard,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const opponentReason = analyzeInvalidMove(opponentGameState, basePosition, {
      selectedPosition: null,
      validMoves: [],
    });

    expect(opponentReason).toBe('opponent_stack');

    // Own stack with no moves
    const ownBoard: BoardState = {
      ...createMockBoard(),
      stacks: new Map([[posKey, { controllingPlayer: 1 } as any]]),
    };

    const ownGameState = createMockGameState({
      board: ownBoard,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const ownReason = analyzeInvalidMove(ownGameState, basePosition, {
      selectedPosition: null,
      validMoves: [],
    });

    expect(ownReason).toBe('no_valid_moves_from_here');
  });

  it('enforces must_move_forced_stack when a different stack is selected', () => {
    const gameState = createMockGameState({ currentPhase: 'movement' });

    const forcedFrom = createPosition(0, 0);
    const selected = createPosition(2, 2);

    const reason = analyzeInvalidMove(gameState, basePosition, {
      mustMoveFrom: forcedFrom,
      selectedPosition: selected,
      validMoves: [],
    });

    expect(reason).toBe('must_move_forced_stack');
  });

  it('returns stack_on_stack_not_allowed vs invalid_placement_position in ring_placement phase', () => {
    const posKey = positionToString(basePosition);

    const emptyBoard = createMockBoard();

    const stackedBoard: BoardState = {
      ...createMockBoard(),
      stacks: new Map([[posKey, { controllingPlayer: 1 } as any]]),
    };

    const basePlacementState = createMockGameState({
      currentPhase: 'ring_placement',
    });

    // Invalid placement on empty cell
    const reasonEmpty = analyzeInvalidMove(
      { ...basePlacementState, board: emptyBoard },
      basePosition,
      {
        validMoves: [],
      }
    );

    expect(reasonEmpty).toBe('invalid_placement_position');

    // Invalid placement on existing stack
    const reasonStacked = analyzeInvalidMove(
      { ...basePlacementState, board: stackedBoard },
      basePosition,
      {
        validMoves: [],
      }
    );

    expect(reasonStacked).toBe('stack_on_stack_not_allowed');
  });

  it('returns chain_capture_must_continue when target is not in valid chain-capture moves', () => {
    const gameState = createMockGameState({ currentPhase: 'chain_capture' });

    const otherPos = createPosition(5, 5);

    const reason = analyzeInvalidMove(gameState, basePosition, {
      validMoves: [
        {
          to: otherPos,
        },
      ],
    });

    expect(reason).toBe('chain_capture_must_continue');
  });

  it('returns out_of_range when target is not a valid destination from the selected stack', () => {
    const gameState = createMockGameState({ currentPhase: 'movement' });

    const selected = createPosition(0, 0);
    const validTarget = createPosition(0, 1);

    const reason = analyzeInvalidMove(gameState, basePosition, {
      selectedPosition: selected,
      validMoves: [
        {
          from: selected,
          to: validTarget,
        },
      ],
    });

    expect(reason).toBe('out_of_range');
  });
});
