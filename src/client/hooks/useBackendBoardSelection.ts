/**
 * @fileoverview useBackendBoardSelection Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game UI state.
 * It manages board selection state, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Movement logic: `src/shared/engine/aggregates/Movement.ts`
 *
 * This adapter:
 * - Tracks currently selected cell (e.g., clicked stack for movement)
 * - Tracks highlighted cells (valid move targets)
 * - Computes mustMoveFrom position (when all moves originate from same stack)
 * - Computes chainCapturePath for visualisation during chain_capture phase
 * - Auto-highlights placement targets during ring_placement phase
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useMemo, useEffect, useCallback } from 'react';
import type { Position, GameState, Move } from '../../shared/types/game';
import { positionsEqual } from '../../shared/types/game';

/**
 * Board selection state managed by the hook.
 */
export interface BackendBoardSelectionState {
  /** Currently selected cell position (e.g., clicked stack) */
  selected: Position | undefined;
  /** Cells highlighted as valid targets for the current selection */
  validTargets: Position[];
  /** When all moves originate from a single stack, this is that position */
  mustMoveFrom: Position | undefined;
  /** Chain capture path for visualisation during chain_capture phase */
  chainCapturePath: Position[] | undefined;
}

/**
 * Actions for managing board selection.
 */
export interface BackendBoardSelectionActions {
  /** Set the currently selected cell */
  setSelected: (cell: Position | undefined) => void;
  /** Set the highlighted target cells */
  setValidTargets: (cells: Position[]) => void;
  /** Clear all selection state */
  clearSelection: () => void;
}

/**
 * Return type for useBackendBoardSelection hook.
 */
export interface UseBackendBoardSelectionReturn extends BackendBoardSelectionState {
  setSelected: (cell: Position | undefined) => void;
  setValidTargets: (cells: Position[]) => void;
  clearSelection: () => void;
}

/**
 * Custom hook for managing backend game board selection state.
 *
 * Handles:
 * - Currently selected cell (e.g., clicked stack for movement)
 * - Highlighted cells (valid move targets)
 * - MustMoveFrom derivation (when all moves from same source)
 * - Chain capture path for visualisation
 * - Auto-highlighting placement targets during ring_placement
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param gameState - Current game state from backend
 * @param validMoves - Array of valid moves from backend (can be null)
 * @returns Object with selection state and actions
 */
export function useBackendBoardSelection(
  gameState: GameState | null,
  validMoves: Move[] | null
): UseBackendBoardSelectionReturn {
  // Currently selected cell on the board
  const [selected, setSelected] = useState<Position | undefined>(undefined);

  // Valid target positions for the current selection
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  // Clear all selection state
  const clearSelection = useCallback(() => {
    setSelected(undefined);
    setValidTargets([]);
  }, []);

  // Approximate must-move stack highlighting: if all movement/capture moves
  // originate from the same stack, treat that as the forced origin.
  const mustMoveFrom: Position | undefined = useMemo(() => {
    if (!Array.isArray(validMoves) || !gameState) return undefined;
    if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
      return undefined;
    }

    const origins = validMoves
      .filter((m) => m.from && (m.type === 'move_stack' || m.type === 'overtaking_capture'))
      .map((m) => m.from as Position);

    if (origins.length === 0) return undefined;
    const first = origins[0];
    const allSame = origins.every((p) => positionsEqual(p, first));
    return allSame ? first : undefined;
  }, [validMoves, gameState]);

  // Extract chain capture path for visualisation during chain_capture phase.
  // The path includes the starting position and all landing positions visited
  // in order, mirroring the sandbox host semantics so overlays remain
  // consistent between backend and local games.
  const chainCapturePath: Position[] | undefined = useMemo(() => {
    if (!gameState || gameState.currentPhase !== 'chain_capture') {
      return undefined;
    }

    const moveHistory = gameState.moveHistory;
    if (!moveHistory || moveHistory.length === 0) {
      return undefined;
    }

    const currentPlayerNumber = gameState.currentPlayer;
    const path: Position[] = [];

    for (let i = moveHistory.length - 1; i >= 0; i--) {
      const move = moveHistory[i];
      if (!move) continue;

      if (
        move.player !== currentPlayerNumber ||
        (move.type !== 'overtaking_capture' && move.type !== 'continue_capture_segment')
      ) {
        break;
      }

      if (move.to) {
        path.unshift(move.to);
      }

      if (move.type === 'overtaking_capture' && move.from) {
        path.unshift(move.from);
      }
    }

    return path.length >= 2 ? path : undefined;
  }, [gameState]);

  // Auto-highlight valid placement targets during ring_placement
  useEffect(() => {
    if (!gameState) return;

    if (gameState.currentPhase === 'ring_placement') {
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const placementTargets = validMoves.filter((m) => m.type === 'place_ring').map((m) => m.to);

        setValidTargets((prev) => {
          if (prev.length !== placementTargets.length) return placementTargets;
          const allMatch = prev.every((p) => placementTargets.some((pt) => positionsEqual(p, pt)));
          return allMatch ? prev : placementTargets;
        });
      } else {
        // Only clear if not already empty, avoiding unnecessary re-renders
        setValidTargets((prev) => (prev.length === 0 ? prev : []));
      }
    }
  }, [gameState?.currentPhase, validMoves, gameState]);

  return {
    selected,
    validTargets,
    mustMoveFrom,
    chainCapturePath,
    setSelected,
    setValidTargets,
    clearSelection,
  };
}

export default useBackendBoardSelection;
