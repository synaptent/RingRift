/**
 * @fileoverview useAutoAdvancePhases Hook
 *
 * Automatically advances through game phases when there's only one valid move
 * and it's a "no-action" move type. This prevents the game from getting stuck
 * when a player has no meaningful choices to make.
 *
 * This matches the sandbox behavior where no-action moves are auto-applied
 * without requiring explicit user interaction.
 *
 * @see src/client/hooks/useSandboxInteractions.ts - Source of this pattern
 */

import { useEffect, useRef } from 'react';
import type { GameState, Move, Position } from '../../shared/types/game';

/**
 * Move types that should be auto-submitted when they're the only valid option.
 * These represent situations where the player has no meaningful choice.
 */
const NO_ACTION_MOVE_TYPES = [
  'no_placement_action',
  'no_movement_action',
  'no_line_action',
  'no_territory_action',
  'skip_capture',
  'skip_recovery',
] as const;

/**
 * PartialMove type for move submission (matches useGameActions signature).
 */
export interface PartialMove {
  type: string;
  from?: Position;
  to?: Position;
  placementCount?: number;
  placedOnStack?: boolean;
  captureDirection?: string;
  regionId?: string;
  segmentId?: string;
  lineId?: string;
  eliminationCount?: number;
  targetPlayer?: number;
}

/**
 * Hook that automatically submits no-action moves when they're the only option.
 *
 * This prevents the game from getting stuck in phases where the player has
 * no valid moves except a pass/skip action. The move is submitted after a
 * brief delay to allow the UI to update and give visual feedback.
 *
 * @param gameState - Current game state from backend
 * @param validMoves - Array of valid moves from backend (can be null during loading)
 * @param isMyTurn - Whether it's currently this player's turn
 * @param submitMove - Function to submit a move to the server
 */
export function useAutoAdvancePhases(
  gameState: GameState | null,
  validMoves: Move[] | null,
  isMyTurn: boolean,
  submitMove: (move: PartialMove) => void
): void {
  // Track whether we've already submitted for this phase to prevent double-submission
  const submittedRef = useRef<string | null>(null);

  useEffect(() => {
    // Guard: Need active game state
    if (!gameState || gameState.gameStatus !== 'active') {
      submittedRef.current = null;
      return;
    }

    // Guard: Must be our turn
    if (!isMyTurn) {
      submittedRef.current = null;
      return;
    }

    // Guard: Need exactly one valid move
    if (!Array.isArray(validMoves) || validMoves.length !== 1) {
      submittedRef.current = null;
      return;
    }

    const onlyMove = validMoves[0];

    // Guard: Must be a no-action move type
    if (!NO_ACTION_MOVE_TYPES.includes(onlyMove.type as (typeof NO_ACTION_MOVE_TYPES)[number])) {
      submittedRef.current = null;
      return;
    }

    // Create a unique key for this phase to prevent double-submission
    const phaseKey = `${gameState.currentPhase}-${gameState.moveHistory.length}-${onlyMove.type}`;
    if (submittedRef.current === phaseKey) {
      return; // Already submitted for this phase
    }

    // Auto-submit after a brief delay for UI feedback
    const timer = setTimeout(() => {
      submittedRef.current = phaseKey;

      // Convert Move to PartialMove for submission
      const partialMove: PartialMove = {
        type: onlyMove.type,
      };

      // Copy optional fields if present
      if (onlyMove.from) partialMove.from = onlyMove.from;
      if (onlyMove.to) partialMove.to = onlyMove.to;

      submitMove(partialMove);
    }, 150);

    return () => clearTimeout(timer);
  }, [gameState, validMoves, isMyTurn, submitMove]);
}

export default useAutoAdvancePhases;
