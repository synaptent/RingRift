/**
 * Unit tests for useBackendBoardHandlers hook
 *
 * Tests cover:
 * - Cell click handling across different phases
 * - Double-click for 2-ring placement
 * - Context menu for ring placement count
 * - Ring placement prompt state management
 * - Spectator and connection restrictions
 *
 * @jest-environment jsdom
 */

import { renderHook, act } from '@testing-library/react';
import { useBackendBoardHandlers } from '../../src/client/hooks/useBackendBoardHandlers';
import type { UseBackendBoardHandlersDeps } from '../../src/client/hooks/useBackendBoardHandlers';
import type { GameState, Move, Position, BoardState, MoveType } from '../../src/shared/types/game';

// Mock react-hot-toast
jest.mock('react-hot-toast', () => {
  const toast = jest.fn() as jest.Mock & {
    success: jest.Mock;
    error: jest.Mock;
  };
  toast.success = jest.fn();
  toast.error = jest.fn();
  return { toast };
});

import { toast as mockToast } from 'react-hot-toast';

// Mock useInvalidMoveFeedback
jest.mock('../../src/client/hooks/useInvalidMoveFeedback', () => ({
  analyzeInvalidMove: jest.fn(() => 'no_valid_move_here'),
}));

// Helper to create minimal valid Move objects
const createMove = (
  type: MoveType,
  player: number,
  to: Position,
  from?: Position,
  extra?: Partial<Move>
): Move =>
  ({
    id: `move-${Math.random().toString(36).substring(7)}`,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    type,
    player,
    to,
    from,
    ...extra,
  }) as Move;

// Helper to create minimal game state
const createGameState = (overrides: Partial<GameState> = {}): GameState =>
  ({
    id: 'test-game',
    boardType: 'square8',
    currentPlayer: 1,
    currentPhase: 'movement',
    players: [
      { playerNumber: 1, type: 'human' as const },
      { playerNumber: 2, type: 'human' as const },
    ],
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8' as const,
    },
    moveHistory: [],
    gameStatus: 'active' as const,
    ...overrides,
  }) as GameState;

// Helper to create minimal board state
const createBoardState = (stackPositions: Position[] = []): BoardState => {
  const stacks = new Map();
  stackPositions.forEach((pos) => {
    stacks.set(`${pos.x},${pos.y}`, {
      position: pos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 1,
      controllingPlayer: 1,
    });
  });

  return {
    stacks,
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8' as const,
  } as BoardState;
};

// Helper to create default deps
const createDeps = (
  overrides: Partial<UseBackendBoardHandlersDeps> = {}
): UseBackendBoardHandlersDeps => ({
  gameState: createGameState(),
  validMoves: [],
  selected: undefined,
  validTargets: [],
  mustMoveFrom: undefined,
  setSelected: jest.fn(),
  setValidTargets: jest.fn(),
  submitMove: jest.fn(),
  isPlayer: true,
  isConnectionActive: true,
  isMyTurn: true,
  triggerInvalidMove: jest.fn(),
  ...overrides,
});

describe('useBackendBoardHandlers', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial state', () => {
    it('should initialize with null ring placement prompt', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendBoardHandlers(deps));

      expect(result.current.ringPlacementCountPrompt).toBeNull();
    });

    it('should provide all expected handlers', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendBoardHandlers(deps));

      expect(result.current.handleCellClick).toBeInstanceOf(Function);
      expect(result.current.handleCellDoubleClick).toBeInstanceOf(Function);
      expect(result.current.handleCellContextMenu).toBeInstanceOf(Function);
      expect(result.current.handleConfirmRingPlacementCount).toBeInstanceOf(Function);
      expect(result.current.closeRingPlacementPrompt).toBeInstanceOf(Function);
    });
  });

  describe('Spectator restrictions', () => {
    it('should show error toast when spectator clicks', () => {
      const deps = createDeps({ isPlayer: false });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(mockToast.error).toHaveBeenCalledWith(
        'Spectators cannot submit moves',
        expect.any(Object)
      );
    });

    it('should not submit move when spectator clicks', () => {
      const submitMove = jest.fn();
      const deps = createDeps({ isPlayer: false, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });
  });

  describe('Connection restrictions', () => {
    it('should show error toast when disconnected', () => {
      const deps = createDeps({ isConnectionActive: false });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(mockToast.error).toHaveBeenCalledWith(
        'Moves paused while disconnected',
        expect.any(Object)
      );
    });
  });

  describe('Ring placement phase handling', () => {
    it('should submit place_ring move on empty cell click', () => {
      const submitMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
      ];
      const deps = createDeps({ gameState, validMoves, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState(); // No stacks

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          to: { x: 3, y: 4 },
        })
      );
    });

    it('should trigger invalid move feedback when no valid moves at position', () => {
      const triggerInvalidMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 0, y: 0 }, undefined, { placementCount: 1 }),
      ];
      const deps = createDeps({ gameState, validMoves, triggerInvalidMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(triggerInvalidMove).toHaveBeenCalledWith({ x: 3, y: 4 }, expect.any(String));
    });

    it('should select stack when clicking on existing stack in placement phase', () => {
      const setSelected = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [createMove('place_ring', 1, { x: 0, y: 0 })];
      const deps = createDeps({ gameState, validMoves, setSelected });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState([{ x: 3, y: 4 }]);

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(setSelected).toHaveBeenCalledWith({ x: 3, y: 4 });
    });
  });

  describe('Movement/Capture phase handling', () => {
    it('should select cell with valid moves when no prior selection', () => {
      const setSelected = jest.fn();
      const setValidTargets = jest.fn();
      const origin: Position = { x: 3, y: 3 };
      const gameState = createGameState({ currentPhase: 'movement' });
      const validMoves = [
        createMove('move_stack', 1, { x: 3, y: 4 }, origin),
        createMove('move_stack', 1, { x: 3, y: 5 }, origin),
      ];
      const deps = createDeps({ gameState, validMoves, setSelected, setValidTargets });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState([origin]);

      act(() => {
        result.current.handleCellClick(origin, board);
      });

      expect(setSelected).toHaveBeenCalledWith(origin);
      expect(setValidTargets).toHaveBeenCalledWith([
        { x: 3, y: 4 },
        { x: 3, y: 5 },
      ]);
    });

    it('should submit move when clicking valid target with selection', () => {
      const submitMove = jest.fn();
      const origin: Position = { x: 3, y: 3 };
      const target: Position = { x: 3, y: 4 };
      const gameState = createGameState({ currentPhase: 'movement' });
      const validMoves = [createMove('move_stack', 1, target, origin)];
      const deps = createDeps({
        gameState,
        validMoves,
        selected: origin,
        submitMove,
      });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState([origin]);

      act(() => {
        result.current.handleCellClick(target, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'move_stack',
          from: origin,
          to: target,
        })
      );
    });

    it('should clear selection when clicking same cell', () => {
      const setSelected = jest.fn();
      const setValidTargets = jest.fn();
      const origin: Position = { x: 3, y: 3 };
      const gameState = createGameState({ currentPhase: 'movement' });
      const deps = createDeps({
        gameState,
        validMoves: [],
        selected: origin,
        setSelected,
        setValidTargets,
      });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState([origin]);

      act(() => {
        result.current.handleCellClick(origin, board);
      });

      expect(setSelected).toHaveBeenCalledWith(undefined);
      expect(setValidTargets).toHaveBeenCalledWith([]);
    });
  });

  describe('Double-click handling', () => {
    it('should submit 2-ring placement when double-clicking empty cell', () => {
      const submitMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 2 }),
      ];
      const deps = createDeps({ gameState, validMoves, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellDoubleClick({ x: 3, y: 4 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          to: { x: 3, y: 4 },
          placementCount: 2,
        })
      );
    });

    it('should do nothing when not in ring_placement phase', () => {
      const submitMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'movement' });
      const deps = createDeps({ gameState, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellDoubleClick({ x: 3, y: 4 }, board);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });

    it('should show error when spectator double-clicks', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const deps = createDeps({ gameState, isPlayer: false });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellDoubleClick({ x: 3, y: 4 }, board);
      });

      expect(mockToast.error).toHaveBeenCalled();
    });
  });

  describe('Context menu handling', () => {
    it('should open ring placement prompt when multiple counts available', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 2 }),
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 3 }),
      ];
      const deps = createDeps({ gameState, validMoves });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellContextMenu({ x: 3, y: 4 }, board);
      });

      expect(result.current.ringPlacementCountPrompt).not.toBeNull();
      expect(result.current.ringPlacementCountPrompt?.maxCount).toBe(3);
    });

    it('should submit directly when only 1-ring placement available', () => {
      const submitMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
      ];
      const deps = createDeps({ gameState, validMoves, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellContextMenu({ x: 3, y: 4 }, board);
      });

      expect(submitMove).toHaveBeenCalled();
      expect(result.current.ringPlacementCountPrompt).toBeNull();
    });
  });

  describe('Ring placement count confirmation', () => {
    it('should submit move with specified count', () => {
      const submitMove = jest.fn();
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 2 }),
      ];
      const deps = createDeps({ gameState, validMoves, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      // First open the prompt
      act(() => {
        result.current.handleCellContextMenu({ x: 3, y: 4 }, board);
      });

      expect(result.current.ringPlacementCountPrompt).not.toBeNull();

      // Then confirm with count 2
      act(() => {
        result.current.handleConfirmRingPlacementCount(2);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          placementCount: 2,
        })
      );
      expect(result.current.ringPlacementCountPrompt).toBeNull();
    });

    it('should do nothing when no prompt is open', () => {
      const submitMove = jest.fn();
      const deps = createDeps({ submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));

      act(() => {
        result.current.handleConfirmRingPlacementCount(2);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });
  });

  describe('closeRingPlacementPrompt', () => {
    it('should close the ring placement prompt', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves = [
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 1 }),
        createMove('place_ring', 1, { x: 3, y: 4 }, undefined, { placementCount: 2 }),
      ];
      const deps = createDeps({ gameState, validMoves });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      // Open prompt
      act(() => {
        result.current.handleCellContextMenu({ x: 3, y: 4 }, board);
      });

      expect(result.current.ringPlacementCountPrompt).not.toBeNull();

      // Close it
      act(() => {
        result.current.closeRingPlacementPrompt();
      });

      expect(result.current.ringPlacementCountPrompt).toBeNull();
    });
  });

  describe('No game state handling', () => {
    it('should do nothing when gameState is null', () => {
      const submitMove = jest.fn();
      const deps = createDeps({ gameState: null, submitMove });
      const { result } = renderHook(() => useBackendBoardHandlers(deps));
      const board = createBoardState();

      act(() => {
        result.current.handleCellClick({ x: 3, y: 4 }, board);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });
  });
});
