/**
 * Unit tests for useBackendBoardSelection hook
 *
 * Tests cover:
 * - Initial state values
 * - Selection state management
 * - Valid targets management
 * - MustMoveFrom derivation
 * - Chain capture path derivation
 * - Auto-highlighting for ring placement phase
 *
 * @jest-environment jsdom
 */

import { renderHook, act } from '@testing-library/react';
import { useBackendBoardSelection } from '../../src/client/hooks/useBackendBoardSelection';
import type { GameState, Move, Position, MoveType } from '../../src/shared/types/game';

// Helper to create minimal valid Move objects
const createMove = (type: MoveType, player: number, to: Position, from?: Position): Move =>
  ({
    id: `move-${Math.random().toString(36).substring(7)}`,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    type,
    player,
    to,
    from,
  }) as Move;

describe('useBackendBoardSelection', () => {
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

  describe('Initial state', () => {
    it('should initialize with undefined selected position', () => {
      const { result } = renderHook(() => useBackendBoardSelection(null, null));

      expect(result.current.selected).toBeUndefined();
    });

    it('should initialize with empty valid targets', () => {
      const { result } = renderHook(() => useBackendBoardSelection(null, null));

      expect(result.current.validTargets).toEqual([]);
    });

    it('should initialize with undefined mustMoveFrom', () => {
      const { result } = renderHook(() => useBackendBoardSelection(null, null));

      expect(result.current.mustMoveFrom).toBeUndefined();
    });

    it('should initialize with undefined chainCapturePath', () => {
      const { result } = renderHook(() => useBackendBoardSelection(null, null));

      expect(result.current.chainCapturePath).toBeUndefined();
    });
  });

  describe('Selection management', () => {
    it('should update selected position', () => {
      const gameState = createGameState();
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      const pos: Position = { x: 3, y: 4 };
      act(() => {
        result.current.setSelected(pos);
      });

      expect(result.current.selected).toEqual(pos);
    });

    it('should clear selected position when set to undefined', () => {
      const gameState = createGameState();
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      act(() => {
        result.current.setSelected({ x: 3, y: 4 });
      });
      expect(result.current.selected).toBeDefined();

      act(() => {
        result.current.setSelected(undefined);
      });

      expect(result.current.selected).toBeUndefined();
    });
  });

  describe('Valid targets management', () => {
    it('should update valid targets', () => {
      const gameState = createGameState();
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      const targets: Position[] = [
        { x: 1, y: 1 },
        { x: 2, y: 2 },
      ];

      act(() => {
        result.current.setValidTargets(targets);
      });

      expect(result.current.validTargets).toEqual(targets);
    });

    it('should clear valid targets with empty array', () => {
      const gameState = createGameState();
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      act(() => {
        result.current.setValidTargets([{ x: 1, y: 1 }]);
      });
      expect(result.current.validTargets).toHaveLength(1);

      act(() => {
        result.current.setValidTargets([]);
      });

      expect(result.current.validTargets).toEqual([]);
    });
  });

  describe('clearSelection', () => {
    it('should clear both selected and valid targets', () => {
      const gameState = createGameState();
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      act(() => {
        result.current.setSelected({ x: 3, y: 4 });
        result.current.setValidTargets([{ x: 1, y: 1 }]);
      });

      expect(result.current.selected).toBeDefined();
      expect(result.current.validTargets).toHaveLength(1);

      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.selected).toBeUndefined();
      expect(result.current.validTargets).toEqual([]);
    });
  });

  describe('mustMoveFrom derivation', () => {
    it('should return undefined when no valid moves', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.mustMoveFrom).toBeUndefined();
    });

    it('should return undefined when not in movement or capture phase', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const moves: Move[] = [createMove('place_ring', 1, { x: 3, y: 4 })];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, moves));

      expect(result.current.mustMoveFrom).toBeUndefined();
    });

    it('should return the common origin when all moves from same stack', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const origin: Position = { x: 3, y: 3 };
      const moves: Move[] = [
        createMove('move_stack', 1, { x: 3, y: 4 }, origin),
        createMove('move_stack', 1, { x: 3, y: 5 }, origin),
        createMove('move_stack', 1, { x: 4, y: 3 }, origin),
      ];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, moves));

      expect(result.current.mustMoveFrom).toEqual(origin);
    });

    it('should return undefined when moves from different stacks', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const moves: Move[] = [
        createMove('move_stack', 1, { x: 3, y: 4 }, { x: 3, y: 3 }),
        createMove('move_stack', 1, { x: 5, y: 6 }, { x: 5, y: 5 }),
      ];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, moves));

      expect(result.current.mustMoveFrom).toBeUndefined();
    });

    it('should work for capture phase with overtaking_capture moves', () => {
      const gameState = createGameState({ currentPhase: 'capture' });
      const origin: Position = { x: 2, y: 2 };
      const moves: Move[] = [
        createMove('overtaking_capture', 1, { x: 2, y: 4 }, origin),
        createMove('overtaking_capture', 1, { x: 4, y: 2 }, origin),
      ];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, moves));

      expect(result.current.mustMoveFrom).toEqual(origin);
    });
  });

  describe('chainCapturePath derivation', () => {
    it('should return undefined when not in chain_capture phase', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.chainCapturePath).toBeUndefined();
    });

    it('should return undefined when no move history', () => {
      const gameState = createGameState({
        currentPhase: 'chain_capture',
        moveHistory: [],
      });

      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.chainCapturePath).toBeUndefined();
    });

    it('should extract chain capture path from move history', () => {
      const gameState = createGameState({
        currentPhase: 'chain_capture',
        currentPlayer: 1,
        moveHistory: [
          createMove('overtaking_capture', 1, { x: 1, y: 3 }, { x: 1, y: 1 }),
          createMove('continue_capture_segment', 1, { x: 3, y: 3 }, { x: 1, y: 3 }),
        ],
      });

      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.chainCapturePath).toEqual([
        { x: 1, y: 1 },
        { x: 1, y: 3 },
        { x: 3, y: 3 },
      ]);
    });

    it('should stop at moves from other players', () => {
      const gameState = createGameState({
        currentPhase: 'chain_capture',
        currentPlayer: 1,
        moveHistory: [
          createMove('move_stack', 2, { x: 5, y: 6 }, { x: 5, y: 5 }),
          createMove('overtaking_capture', 1, { x: 1, y: 3 }, { x: 1, y: 1 }),
        ],
      });

      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.chainCapturePath).toEqual([
        { x: 1, y: 1 },
        { x: 1, y: 3 },
      ]);
    });
  });

  describe('Auto-highlighting for ring placement', () => {
    it('should auto-highlight placement targets during ring_placement phase', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const validMoves: Move[] = [
        createMove('place_ring', 1, { x: 0, y: 0 }),
        createMove('place_ring', 1, { x: 1, y: 1 }),
        createMove('place_ring', 1, { x: 2, y: 2 }),
      ];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, validMoves));

      expect(result.current.validTargets).toEqual([
        { x: 0, y: 0 },
        { x: 1, y: 1 },
        { x: 2, y: 2 },
      ]);
    });

    it('should clear valid targets when no placement moves available', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });

      const { result } = renderHook(() => useBackendBoardSelection(gameState, []));

      expect(result.current.validTargets).toEqual([]);
    });

    it('should not auto-highlight during movement phase', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const validMoves: Move[] = [createMove('move_stack', 1, { x: 0, y: 1 }, { x: 0, y: 0 })];

      const { result } = renderHook(() => useBackendBoardSelection(gameState, validMoves));

      expect(result.current.validTargets).toEqual([]);
    });
  });

  describe('State updates on game state changes', () => {
    it('should recalculate mustMoveFrom when valid moves change', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const origin: Position = { x: 3, y: 3 };
      const initialMoves: Move[] = [createMove('move_stack', 1, { x: 3, y: 4 }, origin)];

      const { result, rerender } = renderHook(
        ({ moves }) => useBackendBoardSelection(gameState, moves),
        { initialProps: { moves: initialMoves } }
      );

      expect(result.current.mustMoveFrom).toEqual(origin);

      // Change to moves from different origins
      const newMoves: Move[] = [
        createMove('move_stack', 1, { x: 1, y: 2 }, { x: 1, y: 1 }),
        createMove('move_stack', 1, { x: 5, y: 6 }, { x: 5, y: 5 }),
      ];

      rerender({ moves: newMoves });

      expect(result.current.mustMoveFrom).toBeUndefined();
    });

    it('should recalculate chainCapturePath when game state changes', () => {
      const initialGameState = createGameState({
        currentPhase: 'chain_capture',
        currentPlayer: 1,
        moveHistory: [createMove('overtaking_capture', 1, { x: 1, y: 3 }, { x: 1, y: 1 })],
      });

      const { result, rerender } = renderHook(
        ({ gameState }) => useBackendBoardSelection(gameState, []),
        { initialProps: { gameState: initialGameState } }
      );

      expect(result.current.chainCapturePath).toEqual([
        { x: 1, y: 1 },
        { x: 1, y: 3 },
      ]);

      // Move to different phase
      const newGameState = createGameState({
        currentPhase: 'movement',
        moveHistory: initialGameState.moveHistory,
      });

      rerender({ gameState: newGameState });

      expect(result.current.chainCapturePath).toBeUndefined();
    });
  });
});
