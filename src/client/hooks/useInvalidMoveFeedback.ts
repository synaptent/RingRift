/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useInvalidMoveFeedback Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides enhanced feedback for invalid move attempts, including:
 * - Visual shake animation targeting a specific cell position
 * - Explanatory toast messages explaining WHY a move is invalid
 * - Screen reader announcements for accessibility
 *
 * Usage:
 * ```tsx
 * const { triggerInvalidMove, shakingCellKey, getInvalidMoveReason } = useInvalidMoveFeedback();
 *
 * const handleCellClick = (pos: Position) => {
 *   if (!isValidMove(pos)) {
 *     const reason = getInvalidMoveReason(gameState, pos);
 *     triggerInvalidMove(pos, reason);
 *     return;
 *   }
 *   // ... handle valid move
 * };
 * ```
 */

import { useState, useCallback, useRef } from 'react';
import { toast } from 'react-hot-toast';
import type { Position, GameState, BoardState } from '../../shared/types/game';
import { positionToString } from '../../shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Reason why a move is invalid, used to generate contextual feedback
 */
export type InvalidMoveReason =
  | 'not_your_turn'
  | 'game_not_active'
  | 'spectator'
  | 'disconnected'
  | 'empty_cell_in_movement'
  | 'opponent_stack'
  | 'out_of_range'
  | 'blocked_path'
  | 'invalid_placement_position'
  | 'stack_on_stack_not_allowed'
  | 'no_valid_moves_from_here'
  | 'must_move_forced_stack'
  | 'chain_capture_must_continue'
  | 'unknown';

/**
 * Configuration for invalid move feedback
 */
export interface InvalidMoveFeedbackConfig {
  /** Duration of shake animation in ms (default: 400) */
  shakeDurationMs?: number;
  /** Whether to show toast notifications (default: true) */
  showToast?: boolean;
  /** Whether to announce to screen readers (default: true) */
  announceToScreenReader?: boolean;
}

/**
 * Result from the useInvalidMoveFeedback hook
 */
export interface InvalidMoveFeedbackResult {
  /** The position key that should be shaking (null if none) */
  shakingCellKey: string | null;
  /** Trigger invalid move feedback for a position with a reason */
  triggerInvalidMove: (position: Position, reason: InvalidMoveReason) => void;
  /** Clear any active shake animation */
  clearShake: () => void;
  /** Get a user-friendly explanation for an invalid move reason */
  getReasonExplanation: (reason: InvalidMoveReason) => string;
  /** Analyze game state to determine why a move to a position would be invalid */
  analyzeInvalidMove: (
    gameState: GameState | null,
    position: Position,
    options?: AnalyzeInvalidMoveOptions
  ) => InvalidMoveReason;
}

export interface AnalyzeInvalidMoveOptions {
  isPlayer?: boolean;
  isMyTurn?: boolean;
  isConnected?: boolean;
  selectedPosition?: Position | null;
  validMoves?: Array<{ from?: Position; to: Position }>;
  mustMoveFrom?: Position;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get a user-friendly explanation for why a move is invalid
 */
export function getReasonExplanation(reason: InvalidMoveReason): string {
  switch (reason) {
    case 'not_your_turn':
      return "It's not your turn yet";
    case 'game_not_active':
      return 'The game has ended';
    case 'spectator':
      return 'Spectators cannot make moves';
    case 'disconnected':
      return 'Reconnecting to server...';
    case 'empty_cell_in_movement':
      return 'Select a stack to move, not an empty cell';
    case 'opponent_stack':
      return "You cannot move your opponent's pieces";
    case 'out_of_range':
      return 'This cell is too far away';
    case 'blocked_path':
      return 'The path to this cell is blocked';
    case 'invalid_placement_position':
      return 'Rings cannot be placed here';
    case 'stack_on_stack_not_allowed':
      return 'Cannot place rings on existing stacks right now';
    case 'no_valid_moves_from_here':
      return 'No valid moves available from this position';
    case 'must_move_forced_stack':
      return 'You must move the highlighted stack';
    case 'chain_capture_must_continue':
      return 'You must continue the chain capture';
    case 'unknown':
    default:
      return 'Invalid move';
  }
}

/**
 * Analyze game state to determine why a position would be an invalid move
 */
export function analyzeInvalidMove(
  gameState: GameState | null,
  position: Position,
  options: AnalyzeInvalidMoveOptions = {}
): InvalidMoveReason {
  const {
    isPlayer = true,
    isMyTurn = true,
    isConnected = true,
    selectedPosition,
    validMoves = [],
    mustMoveFrom,
  } = options;

  // Connection/role checks
  if (!isPlayer) return 'spectator';
  if (!isConnected) return 'disconnected';
  if (!gameState) return 'game_not_active';
  if (gameState.gameStatus !== 'active') return 'game_not_active';
  if (!isMyTurn) return 'not_your_turn';

  const board = gameState.board;
  const phase = gameState.currentPhase;
  const posKey = positionToString(position);
  const hasStack = !!board.stacks.get(posKey);

  // Chain capture phase: must continue the chain
  if (phase === 'chain_capture') {
    const isValidChainTarget = validMoves.some((m) => positionToString(m.to) === posKey);
    if (!isValidChainTarget) {
      return 'chain_capture_must_continue';
    }
  }

  // Forced stack must-move check
  if (mustMoveFrom) {
    const mustMoveKey = positionToString(mustMoveFrom);
    if (selectedPosition && positionToString(selectedPosition) !== mustMoveKey) {
      return 'must_move_forced_stack';
    }
  }

  // Ring placement phase
  if (phase === 'ring_placement') {
    const isValidPlacement = validMoves.some((m) => positionToString(m.to) === posKey);
    if (!isValidPlacement) {
      if (hasStack) {
        return 'stack_on_stack_not_allowed';
      }
      return 'invalid_placement_position';
    }
  }

  // Movement/capture phases
  if (phase === 'movement' || phase === 'capture') {
    // Selecting a source
    if (!selectedPosition) {
      if (!hasStack) {
        return 'empty_cell_in_movement';
      }

      // Check if this stack has any valid moves
      const hasMovesFromHere = validMoves.some(
        (m) => m.from && positionToString(m.from) === posKey
      );
      if (!hasMovesFromHere) {
        // Check if it's an opponent's stack
        const stack = board.stacks.get(posKey);
        if (stack && stack.controllingPlayer !== gameState.currentPlayer) {
          return 'opponent_stack';
        }
        return 'no_valid_moves_from_here';
      }
    } else {
      // Already have a selection, clicking a target
      const selectedKey = positionToString(selectedPosition);
      const isValidTarget = validMoves.some(
        (m) =>
          m.from && positionToString(m.from) === selectedKey && positionToString(m.to) === posKey
      );
      if (!isValidTarget) {
        // Determine more specific reason
        // Check if target is simply out of range or blocked
        return 'out_of_range';
      }
    }
  }

  return 'unknown';
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useInvalidMoveFeedback
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for providing enhanced invalid move feedback with animations and toasts
 */
export function useInvalidMoveFeedback(
  config: InvalidMoveFeedbackConfig = {}
): InvalidMoveFeedbackResult {
  const { shakeDurationMs = 400, showToast = true, announceToScreenReader = true } = config;

  const [shakingCellKey, setShakingCellKey] = useState<string | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear any existing timeout
  const clearExistingTimeout = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  // Clear shake state
  const clearShake = useCallback(() => {
    clearExistingTimeout();
    setShakingCellKey(null);
  }, [clearExistingTimeout]);

  // Trigger invalid move feedback
  const triggerInvalidMove = useCallback(
    (position: Position, reason: InvalidMoveReason) => {
      const posKey = positionToString(position);
      const explanation = getReasonExplanation(reason);

      // Set shake animation
      clearExistingTimeout();
      setShakingCellKey(posKey);

      // Clear after animation completes
      timeoutRef.current = setTimeout(() => {
        setShakingCellKey(null);
      }, shakeDurationMs);

      // Show toast notification
      if (showToast) {
        // Use a consistent toast ID to prevent duplicate toasts
        toast.error(explanation, {
          id: 'invalid-move',
          duration: 2000,
          position: 'bottom-center',
          style: {
            background: '#1e293b',
            color: '#f8fafc',
            border: '1px solid rgba(239, 68, 68, 0.5)',
          },
          icon: '⚠️',
        });
      }

      // Screen reader announcement is handled by the toast library's aria-live region
      // and the sr-only announcement in the component if needed
    },
    [shakeDurationMs, showToast, clearExistingTimeout]
  );

  return {
    shakingCellKey,
    triggerInvalidMove,
    clearShake,
    getReasonExplanation,
    analyzeInvalidMove,
  };
}

export default useInvalidMoveFeedback;
