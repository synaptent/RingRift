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

  // Must-move stack highlighting: prioritize server-provided mustMoveFromStackKey
  // (set after ring placement), otherwise infer from validMoves when all moves
  // originate from the same stack.
  const mustMoveFrom: Position | undefined = useMemo(() => {
    if (!gameState) return undefined;

    // Priority 1: Use server-provided mustMoveFromStackKey (set after ring placement)
    if (gameState.mustMoveFromStackKey) {
      const [x, y] = gameState.mustMoveFromStackKey.split(',').map(Number);
      if (!isNaN(x) && !isNaN(y)) {
        return { x, y };
      }
    }

    // Priority 2: Infer from validMoves (all moves from same origin)
    if (!Array.isArray(validMoves)) return undefined;
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

  // Auto-populate selection when entering chain_capture phase.
  // This ensures the player sees the next capture source and valid targets
  // highlighted automatically, matching sandbox behavior.
  useEffect(() => {
    if (!gameState || gameState.currentPhase !== 'chain_capture') return;
    if (!Array.isArray(validMoves) || validMoves.length === 0) return;

    // Derive chain capture context from validMoves
    const chainMoves = validMoves.filter(
      (m) => m.type === 'continue_capture_segment' || m.type === 'overtaking_capture'
    );
    if (chainMoves.length === 0) return;

    // Set the capture source as selected
    const from = chainMoves[0]?.from;
    if (from) {
      setSelected((prev) => {
        if (prev && positionsEqual(prev, from)) return prev;
        return from;
      });
    }

    // Set valid landings as targets
    const landings = chainMoves.map((m) => m.to).filter((t): t is Position => t !== undefined);
    setValidTargets((prev) => {
      if (
        prev.length === landings.length &&
        prev.every((p) => landings.some((l) => positionsEqual(p, l)))
      ) {
        return prev;
      }
      return landings;
    });
  }, [gameState?.currentPhase, validMoves, gameState]);

  // Auto-select mustMoveFrom stack during movement phase (e.g., after ring placement).
  // This enables click-to-move: user places ring, stack is auto-selected, user clicks target.
  useEffect(() => {
    if (!gameState || gameState.currentPhase !== 'movement') return;
    if (!mustMoveFrom) return;
    if (!Array.isArray(validMoves) || validMoves.length === 0) return;

    // Auto-select the mustMoveFrom position
    setSelected((prev) => {
      if (prev && positionsEqual(prev, mustMoveFrom)) return prev;
      return mustMoveFrom;
    });

    // Highlight valid landing positions for this stack
    const landings = validMoves
      .filter(
        (m) =>
          (m.type === 'move_stack' || m.type === 'overtaking_capture') &&
          m.from &&
          positionsEqual(m.from, mustMoveFrom)
      )
      .map((m) => m.to)
      .filter((t): t is Position => t !== undefined);

    setValidTargets((prev) => {
      if (
        prev.length === landings.length &&
        prev.every((p) => landings.some((l) => positionsEqual(p, l)))
      ) {
        return prev;
      }
      return landings;
    });
  }, [gameState?.currentPhase, validMoves, mustMoveFrom, gameState]);

  // Auto-highlight elimination targets during elimination phases.
  // This ensures players see which stacks they can click to eliminate rings from.
  useEffect(() => {
    if (!gameState) return;

    const eliminationPhases = ['forced_elimination', 'line_processing', 'territory_processing'];
    if (!eliminationPhases.includes(gameState.currentPhase)) return;
    if (!Array.isArray(validMoves) || validMoves.length === 0) return;

    // Find elimination moves and highlight their target positions
    const elimMoves = validMoves.filter((m) => m.type === 'eliminate_rings_from_stack' && m.to);
    if (elimMoves.length === 0) return;

    const targets = elimMoves.map((m) => m.to).filter((t): t is Position => t !== undefined);
    setValidTargets((prev) => {
      if (
        prev.length === targets.length &&
        prev.every((p) => targets.some((t) => positionsEqual(p, t)))
      ) {
        return prev;
      }
      return targets;
    });

    // Clear selection since elimination is target-only (no source selection needed)
    setSelected(undefined);
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
