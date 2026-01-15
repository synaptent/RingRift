/**
 * @fileoverview usePendingMovement Hook - Pending Movement Handler
 *
 * Handles the skip_placement + movement shortcut pattern where:
 * 1. User clicks a valid landing during placement phase
 * 2. We submit skip_placement and store the intended movement
 * 3. After phase changes to movement, we auto-submit the movement
 *
 * This hook encapsulates the retry logic for when validMoves might be stale
 * after a phase change.
 *
 * @see useBackendBoardHandlers.ts
 */

import { useEffect, useRef, MutableRefObject } from 'react';
import type { Position, Move } from '../../shared/types/game';
import { positionsEqual } from '../../shared/types/game';
import type { PartialMove } from './useGameActions';

/**
 * Pending movement state stored in ref for cross-effect access.
 */
export interface PendingMovement {
  from: Position;
  to: Position;
  timestamp: number;
}

/**
 * Dependencies for usePendingMovement hook.
 */
export interface UsePendingMovementDeps {
  /** Current game phase */
  currentPhase: string | undefined;
  /** Valid moves from the server */
  validMoves: Move[] | null;
  /** Function to submit a move to the backend */
  submitMove: (move: PartialMove) => void;
  /** Function to clear the selected position */
  setSelected: (pos: Position | undefined) => void;
  /** Function to clear valid targets */
  setValidTargets: (targets: Position[]) => void;
}

/**
 * Configuration options for pending movement behavior.
 */
export interface UsePendingMovementOptions {
  /** Timeout in ms before clearing stale pending movements (default: 10000) */
  staleTimeout?: number;
  /** Maximum retry attempts before giving up (default: 5) */
  maxRetries?: number;
}

/**
 * Extract captureTarget from a move object if it's a capture move.
 */
function extractCaptureTarget(move: Move | PartialMove): Position | undefined {
  if ('captureTarget' in move && move.captureTarget) {
    return move.captureTarget as Position;
  }
  return undefined;
}

/**
 * Hook for managing pending movement state after skip_placement.
 *
 * When user clicks a valid landing during placement phase, the parent hook
 * can set the pendingMovementRef to store the intended movement. This hook
 * then watches for phase changes to 'movement' and auto-submits the move.
 *
 * @param deps - Dependencies including game phase, valid moves, and actions
 * @param options - Optional configuration for timeout and retry behavior
 * @returns Ref for setting pending movement from parent hook
 */
export function usePendingMovement(
  deps: UsePendingMovementDeps,
  options: UsePendingMovementOptions = {}
): MutableRefObject<PendingMovement | null> {
  const { currentPhase, validMoves, submitMove, setSelected, setValidTargets } = deps;
  const { staleTimeout = 10000, maxRetries = 5 } = options;

  // Pending movement target for skip_placement + movement shortcut
  const pendingMovementRef = useRef<PendingMovement | null>(null);
  const prevPhaseRef = useRef<string | null>(null);
  const retryCountRef = useRef<number>(0);

  // Effect to handle pending movement after skip_placement
  // Includes retry logic for when validMoves might be stale after phase change
  useEffect(() => {
    const _prevPhase = prevPhaseRef.current;
    prevPhaseRef.current = currentPhase ?? null;

    // Clear stale pending movements (older than configured timeout)
    if (pendingMovementRef.current) {
      const age = Date.now() - pendingMovementRef.current.timestamp;
      if (age > staleTimeout) {
        console.warn(
          `[PendingMovement] Clearing stale pending movement after ${staleTimeout}ms timeout`
        );
        pendingMovementRef.current = null;
        retryCountRef.current = 0;
        return;
      }
    }

    // If phase changed to movement and we have a pending target
    const shouldTryPendingMovement =
      currentPhase === 'movement' && pendingMovementRef.current && validMoves;

    if (!shouldTryPendingMovement) {
      // Reset retry count if we're no longer in movement phase or no pending movement
      if (currentPhase !== 'movement' || !pendingMovementRef.current) {
        retryCountRef.current = 0;
      }
      return;
    }

    const pending = pendingMovementRef.current;
    if (!pending) {
      return;
    }

    // Find the matching move_stack or overtaking_capture move
    // (capture targets use overtaking_capture type, not move_stack)
    const pendingMove = validMoves.find(
      (m) =>
        (m.type === 'move_stack' || m.type === 'overtaking_capture') &&
        m.from &&
        positionsEqual(m.from, pending.from) &&
        m.to &&
        positionsEqual(m.to, pending.to)
    );

    if (pendingMove) {
      // Success! Submit the move and clear pending state
      // For capture moves, include captureTarget from the server's move
      const captureTarget = extractCaptureTarget(pendingMove);
      pendingMovementRef.current = null;
      retryCountRef.current = 0;
      submitMove({
        type: pendingMove.type,
        from: pendingMove.from,
        to: pendingMove.to,
        captureTarget,
      } as PartialMove);
      setSelected(undefined);
      setValidTargets([]);
    } else if (retryCountRef.current < maxRetries) {
      // Move not found in validMoves - this may be because validMoves is stale
      // Schedule a retry after a brief delay to allow state to settle
      retryCountRef.current += 1;
      // The effect will re-run when validMoves updates
    } else {
      // Max retries reached, clear the pending movement
      console.warn('[PendingMovement] Max retries reached, clearing pending movement');
      pendingMovementRef.current = null;
      retryCountRef.current = 0;
    }
  }, [
    currentPhase,
    validMoves,
    submitMove,
    setSelected,
    setValidTargets,
    staleTimeout,
    maxRetries,
  ]);

  return pendingMovementRef;
}
