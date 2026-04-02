/**
 * Unit tests for useCellInteractions hook
 *
 * Tests the cell interaction hook that manages:
 * - Cell click handling
 * - Selection state
 * - Valid targets computation
 * - Move construction from clicks
 *
 * @module tests/unit/useCellInteractions.test
 */

import { renderHook, act } from '@testing-library/react';
import type { BoardState, Move, Position } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import {
  useCellInteractions,
  type CellInteractionOptions,
} from '../../src/client/facades/useCellInteractions';
import type {
  GameFacade,
  FacadeConnectionStatus,
  GameFacadeMode,
} from '../../src/client/facades/GameFacade';
import { inferTotalRingsInPlay } from '../utils/fixtures';

// ═══════════════════════════════════════════════════════════════════════════
// TEST FIXTURES
// ═══════════════════════════════════════════════════════════════════════════

function createMinimalBoardState(): BoardState {
  return {
    type: 'square8',
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
  };
}

function createMove(overrides: Partial<Move> = {}): Move {
  return {
    id: 'move-1',
    type: 'move_stack',
    from: { x: 0, y: 0 },
    to: { x: 1, y: 1 },
    player: 1,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    ...overrides,
  } as Move;
}

function createMockFacade(overrides: Partial<GameFacade> = {}): GameFacade {
  const board = createMinimalBoardState();
  const players = [
    {
      id: 'p1',
      username: 'Player 1',
      playerNumber: 1,
      type: 'human',
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    } as any,
    {
      id: 'p2',
      username: 'Player 2',
      playerNumber: 2,
      type: 'ai',
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    } as any,
  ];

  return {
    gameState: {
      id: 'test-game',
      boardType: 'square8',
      board,
      players,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { initialTime: 300, increment: 5, type: 'rapid' },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: inferTotalRingsInPlay(players, board),
      totalRingsEliminated: 0,
      victoryThreshold: 12,
      territoryVictoryThreshold: 33,
    } as any,
    validMoves: [],
    victoryState: null,
    gameEndExplanation: null,
    mode: 'backend' as GameFacadeMode,
    connectionStatus: 'connected' as FacadeConnectionStatus,
    isPlayer: true,
    isMyTurn: true,
    boardType: 'square8',
    currentUserId: 'player-1',
    decisionState: {
      pendingChoice: null,
      choiceDeadline: null,
      choiceTimeRemainingMs: null,
    },
    chainCaptureState: null,
    mustMoveFrom: undefined,
    players: [],
    submitMove: jest.fn(),
    respondToChoice: jest.fn(),
    ...overrides,
  };
}

function createBoardWithStack(position: Position, controllingPlayer: number = 1): BoardState {
  const board = createMinimalBoardState();
  const key = positionToString(position);
  board.stacks.set(key, {
    position,
    rings: [controllingPlayer],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer,
  });
  return board;
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

describe('useCellInteractions', () => {
  describe('initialization', () => {
    it('returns initial state with no selection', () => {
      const facade = createMockFacade();
      const { result } = renderHook(() => useCellInteractions(facade));

      expect(result.current.selected).toBeUndefined();
      expect(result.current.validTargets).toEqual([]);
      expect(result.current.effectiveSelected).toBeUndefined();
    });

    it('returns effectiveSelected from mustMoveFrom when no selection', () => {
      const mustMoveFrom: Position = { x: 2, y: 2 };
      const facade = createMockFacade({ mustMoveFrom });
      const { result } = renderHook(() => useCellInteractions(facade));

      expect(result.current.selected).toBeUndefined();
      expect(result.current.effectiveSelected).toEqual(mustMoveFrom);
    });

    it('handles null facade gracefully', () => {
      const { result } = renderHook(() => useCellInteractions(null));

      expect(result.current.selected).toBeUndefined();
      expect(result.current.validTargets).toEqual([]);
    });
  });

  describe('selection management', () => {
    it('clearSelection clears selection and valid targets', () => {
      const facade = createMockFacade();
      const { result } = renderHook(() => useCellInteractions(facade));

      // First set a selection
      act(() => {
        result.current.setSelected({ x: 1, y: 1 });
      });

      expect(result.current.selected).toEqual({ x: 1, y: 1 });

      // Then clear it
      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.selected).toBeUndefined();
      expect(result.current.validTargets).toEqual([]);
    });

    it('setSelected updates selection', () => {
      const facade = createMockFacade();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.setSelected({ x: 3, y: 4 });
      });

      expect(result.current.selected).toEqual({ x: 3, y: 4 });
    });
  });

  describe('handleCellClick - ring placement phase', () => {
    it('submits place_ring move when clicking empty cell with valid placement', () => {
      const submitMove = jest.fn();
      const validPlaceMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 1,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [validPlaceMove],
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellClick({ x: 2, y: 2 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 1,
        placedOnStack: undefined,
      });
    });

    it('calls onInvalidMove when clicking position without valid placement', () => {
      const onInvalidMove = jest.fn();
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [], // No valid moves
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade, { onInvalidMove }));

      act(() => {
        result.current.handleCellClick({ x: 2, y: 2 }, board);
      });

      // With no valid moves, nothing happens (early return)
      expect(onInvalidMove).not.toHaveBeenCalled();
    });

    it('selects stack when clicking on existing stack in placement phase', () => {
      const validPlaceMove = createMove({
        type: 'place_ring',
        to: { x: 0, y: 0 },
        placementCount: 1,
        placedOnStack: true,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [validPlaceMove],
      });

      const board = createBoardWithStack({ x: 0, y: 0 });
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellClick({ x: 0, y: 0 }, board);
      });

      expect(result.current.selected).toEqual({ x: 0, y: 0 });
    });
  });

  describe('handleCellClick - movement/capture phases', () => {
    it('selects stack with valid moves from it', () => {
      const validMove = createMove({
        type: 'move_stack',
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [validMove],
      });

      const board = createBoardWithStack({ x: 1, y: 1 });
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(result.current.selected).toEqual({ x: 1, y: 1 });
      expect(result.current.validTargets).toEqual([{ x: 2, y: 2 }]);
    });

    it('submits move when clicking valid target from selection', () => {
      const submitMove = jest.fn();
      const validMove = createMove({
        type: 'move_stack',
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [validMove],
        submitMove,
      });

      const board = createBoardWithStack({ x: 1, y: 1 });
      const { result } = renderHook(() => useCellInteractions(facade));

      // First select the stack
      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      // Then click the target
      act(() => {
        result.current.handleCellClick({ x: 2, y: 2 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith({
        type: 'move_stack',
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
    });

    it('clears selection when clicking the same cell', () => {
      const validMove = createMove({
        type: 'move_stack',
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [validMove],
      });

      const board = createBoardWithStack({ x: 1, y: 1 });
      const { result } = renderHook(() => useCellInteractions(facade));

      // Select the stack
      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(result.current.selected).toBeDefined();

      // Click same cell to deselect
      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(result.current.selected).toBeUndefined();
    });

    it('switches selection to new stack with valid moves', () => {
      const move1 = createMove({
        type: 'move_stack',
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      const move2 = createMove({
        type: 'move_stack',
        from: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [move1, move2],
      });

      const board = createBoardWithStack({ x: 1, y: 1 });
      board.stacks.set(positionToString({ x: 3, y: 3 }), {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const { result } = renderHook(() => useCellInteractions(facade));

      // Select first stack
      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(result.current.selected).toEqual({ x: 1, y: 1 });

      // Click second stack
      act(() => {
        result.current.handleCellClick({ x: 3, y: 3 }, board);
      });

      expect(result.current.selected).toEqual({ x: 3, y: 3 });
      expect(result.current.validTargets).toEqual([{ x: 4, y: 4 }]);
    });
  });

  describe('handleCellClick - interaction blocked', () => {
    it('blocks interaction when disconnected in backend mode', () => {
      const onInteractionBlocked = jest.fn();
      const facade = createMockFacade({
        mode: 'backend',
        connectionStatus: 'disconnected',
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade, { onInteractionBlocked }));

      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(onInteractionBlocked).toHaveBeenCalledWith('Moves paused while disconnected');
    });

    it('blocks interaction for spectators', () => {
      const onInteractionBlocked = jest.fn();
      const facade = createMockFacade({ mode: 'spectator' });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade, { onInteractionBlocked }));

      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(onInteractionBlocked).toHaveBeenCalledWith('Spectators cannot submit moves');
    });
  });

  describe('handleCellDoubleClick', () => {
    it('prefers 2-ring placement on empty cell', () => {
      const submitMove = jest.fn();
      const oneRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 1,
      });
      const twoRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 2,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [oneRingMove, twoRingMove],
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellDoubleClick({ x: 2, y: 2 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          to: { x: 2, y: 2 },
          placementCount: 2,
        })
      );
    });

    it('does nothing in non-placement phases', () => {
      const submitMove = jest.fn();
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellDoubleClick({ x: 2, y: 2 }, board);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });

    it('blocks double-click when disconnected', () => {
      const onInteractionBlocked = jest.fn();
      const facade = createMockFacade({
        mode: 'backend',
        connectionStatus: 'disconnected',
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade, { onInteractionBlocked }));

      act(() => {
        result.current.handleCellDoubleClick({ x: 2, y: 2 }, board);
      });

      expect(onInteractionBlocked).toHaveBeenCalledWith(
        'Cannot modify placements while disconnected'
      );
    });
  });

  describe('handleCellContextMenu', () => {
    it('calls requestRingPlacementCount when provided', async () => {
      const submitMove = jest.fn();
      const requestRingPlacementCount = jest.fn().mockResolvedValue(3);
      const oneRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 1,
      });
      const threeRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 3,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [oneRingMove, threeRingMove],
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() =>
        useCellInteractions(facade, { requestRingPlacementCount })
      );

      await act(async () => {
        result.current.handleCellContextMenu({ x: 2, y: 2 }, board);
        // Wait for promise resolution
        await Promise.resolve();
      });

      expect(requestRingPlacementCount).toHaveBeenCalledWith({
        maxCount: 3,
        hasStack: false,
        defaultCount: 2,
      });
    });

    it('submits default count when requestRingPlacementCount not provided', () => {
      const submitMove = jest.fn();
      const oneRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 1,
      });
      const twoRingMove = createMove({
        type: 'place_ring',
        to: { x: 2, y: 2 },
        placementCount: 2,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [oneRingMove, twoRingMove],
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellContextMenu({ x: 2, y: 2 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          to: { x: 2, y: 2 },
          placementCount: 2,
        })
      );
    });

    it('uses 1 ring as default on existing stack', () => {
      const submitMove = jest.fn();
      const oneRingMove = createMove({
        type: 'place_ring',
        to: { x: 0, y: 0 },
        placementCount: 1,
        placedOnStack: true,
      });
      const twoRingMove = createMove({
        type: 'place_ring',
        to: { x: 0, y: 0 },
        placementCount: 2,
        placedOnStack: true,
      });
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'ring_placement',
        } as any,
        validMoves: [oneRingMove, twoRingMove],
        submitMove,
      });

      const board = createBoardWithStack({ x: 0, y: 0 });
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellContextMenu({ x: 0, y: 0 }, board);
      });

      expect(submitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          to: { x: 0, y: 0 },
          placementCount: 1,
        })
      );
    });

    it('does nothing in non-placement phases', () => {
      const submitMove = jest.fn();
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        submitMove,
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellContextMenu({ x: 2, y: 2 }, board);
      });

      expect(submitMove).not.toHaveBeenCalled();
    });
  });

  describe('edge cases', () => {
    it('handles null gameState in facade', () => {
      const facade = createMockFacade({ gameState: null });
      const { result } = renderHook(() => useCellInteractions(facade));
      const board = createMinimalBoardState();

      // Should not throw
      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      expect(result.current.selected).toBeUndefined();
    });

    it('handles empty validMoves array', () => {
      const onInvalidMove = jest.fn();
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [],
      });

      const board = createMinimalBoardState();
      const { result } = renderHook(() => useCellInteractions(facade, { onInvalidMove }));

      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      // Should select position even with no valid moves
      expect(result.current.selected).toEqual({ x: 1, y: 1 });
    });

    it('returns undefined validTargets when clicking position with no moves from it', () => {
      const facade = createMockFacade({
        gameState: {
          ...createMockFacade().gameState!,
          currentPhase: 'movement',
        } as any,
        validMoves: [
          createMove({
            type: 'move_stack',
            from: { x: 5, y: 5 },
            to: { x: 6, y: 6 },
          }),
        ],
      });

      const board = createBoardWithStack({ x: 1, y: 1 });
      const { result } = renderHook(() => useCellInteractions(facade));

      act(() => {
        result.current.handleCellClick({ x: 1, y: 1 }, board);
      });

      // No moves from this position, so no valid targets
      expect(result.current.selected).toBeUndefined();
      expect(result.current.validTargets).toEqual([]);
    });
  });
});
