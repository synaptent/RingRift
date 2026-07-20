/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useGameActions Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides action submission functions for game interactions including
 * move submission, decision handling, and chat. This hook wraps GameContext
 * to expose only action-related functions.
 *
 * Benefits:
 * - Clear separation of actions from state
 * - Type-safe move and choice submission
 * - Easy to mock for testing interaction scenarios
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useGame } from '../contexts/GameContext';
import { useAuth } from '../contexts/AuthContext';
import type { Move, Position, PlayerChoice, GameState } from '../../shared/types/game';
import { getChoiceViewModel } from '../adapters/choiceViewModels';
import type { ChoiceViewModel } from '../adapters/choiceViewModels';

/** Type guard to check if a player matches the authenticated user */
function isUserPlayer(gameState: GameState | null, userId: string | undefined): boolean {
  if (!gameState || !userId) return false;
  const players = gameState.players;
  if (!Array.isArray(players)) return false;
  return players.some((p) => p.id === userId || (p as { userId?: string }).userId === userId);
}

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Partial move data for submission (fields auto-populated by context)
 */
export type PartialMove = Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>;

/**
 * Simplified placement action
 */
export interface PlacementAction {
  type: 'place_ring';
  to: Position;
  placementCount?: number;
  placedOnStack?: boolean;
}

/**
 * Simplified movement action
 */
export interface MovementAction {
  type: 'move_stack';
  from: Position;
  to: Position;
}

/**
 * Pending choice state (lightweight)
 */
export interface PendingChoiceState {
  /** Current choice awaiting response (null if none) */
  choice: PlayerChoice | null;
  /** Deadline timestamp in ms (null if no timeout) */
  deadline: number | null;
  /** Whether a choice is currently pending */
  hasPendingChoice: boolean;
}

/**
 * Rich decision-phase view used by decision UIs (ChoiceDialog, HUD, etc.).
 * This is intentionally derived from PlayerChoice via choiceViewModels so
 * that all decision semantics (labels, timeouts, future highlights) share a
 * single source of truth.
 */
export interface PendingChoiceView {
  /** Underlying domain choice currently awaiting a response. */
  choice: PlayerChoice;
  /**
   * Mapped UX metadata for this choice, including titles, labels, and
   * timeout semantics.
   */
  viewModel: ChoiceViewModel;
  /**
   * Client-side estimate of remaining time (ms) if a deadline was provided
   * by the server, or null when no explicit timeout exists.
   */
  timeRemainingMs: number | null;
  /**
   * Convenience flag indicating that, according to the local clock, the
   * decision deadline has elapsed (timeRemainingMs has reached 0).
   */
  isExpiredClientSide: boolean;
}

/**
 * Choice submission result
 */
export interface ChoiceResponse<T = unknown> {
  choiceId: string;
  playerNumber: number;
  selectedOption: T;
}

/**
 * Action capabilities based on context
 */
export interface ActionCapabilities {
  /** Whether move submission is available */
  canSubmitMove: boolean;
  /** Whether choice responses can be made */
  canRespondToChoice: boolean;
  /** Whether chat is available */
  canSendChat: boolean;
  /** Reason if actions are disabled */
  disabledReason?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useGameActions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for submitting game actions (moves, choices, chat)
 *
 * Usage:
 * ```tsx
 * const { submitMove, respondToChoice, sendChat, capabilities } = useGameActions();
 *
 * const handlePlacement = (pos: Position) => {
 *   if (capabilities.canSubmitMove) {
 *     submitMove({
 *       type: 'place_ring',
 *       player: currentPlayer.playerNumber,
 *       to: pos,
 *       placementCount: 1,
 *     });
 *   }
 * };
 *
 * const handleChoiceSelection = (option: any) => {
 *   if (pendingChoice.hasPendingChoice && capabilities.canRespondToChoice) {
 *     respondToChoice(pendingChoice.choice, option);
 *   }
 * };
 * ```
 */
export function useGameActions() {
  const {
    gameId,
    gameState,
    submitMove: contextSubmitMove,
    submitMoveById: contextSubmitMoveById,
    respondToChoice: contextRespondToChoice,
    sendChatMessage,
    pendingChoice,
    choiceDeadline,
  } = useGame();
  const { user } = useAuth();

  // Stable move submission with type narrowing. Spectators are never allowed to
  // submit backend moves; this guard enforces read-only semantics even if
  // callers bypass the exported capabilities flags.
  const submitMove = useCallback(
    (partialMove: PartialMove) => {
      if (!gameId || !gameState) {
        // Preserve defensive behaviour when no active game is connected.
        // GameContext.submitMove will also warn, but we short-circuit here.
        // This keeps the hook safe to call even during transient reconnects.

        console.warn('submitMove called without active game');
        return;
      }

      if (!isUserPlayer(gameState, user?.id)) {
        console.warn('submitMove called by spectator – ignoring');
        return;
      }

      contextSubmitMove(partialMove);
    },
    [contextSubmitMove, gameId, gameState, user]
  );

  // Canonical decision moves supplied in game_state should be submitted by
  // their server-issued id. This keeps decision-only move types out of the
  // generic player_move transport and avoids re-serializing their payloads.
  const submitMoveById = useCallback(
    (moveId: string) => {
      if (!gameId || !gameState) {
        console.warn('submitMoveById called without active game');
        return;
      }

      if (!isUserPlayer(gameState, user?.id)) {
        console.warn('submitMoveById called by spectator – ignoring');
        return;
      }

      contextSubmitMoveById(moveId);
    },
    [contextSubmitMoveById, gameId, gameState, user]
  );

  // Convenience method for placement
  const submitPlacement = useCallback(
    (action: PlacementAction & { player: number }) => {
      submitMove(action as PartialMove);
    },
    [submitMove]
  );

  // Convenience method for movement
  const submitMovement = useCallback(
    (action: MovementAction & { player: number }) => {
      submitMove(action as PartialMove);
    },
    [submitMove]
  );

  // Choice response with proper typing. Spectators must never be able to send
  // decision responses; this guard mirrors the move-submission restrictions.
  const respondToChoice = useCallback(
    <T>(choice: PlayerChoice, selectedOption: T) => {
      if (!gameId || !gameState) {
        console.warn('respondToChoice called without active game');
        return;
      }

      if (!isUserPlayer(gameState, user?.id)) {
        console.warn('respondToChoice called by spectator – ignoring');
        return;
      }

      contextRespondToChoice(choice, selectedOption);
    },
    [contextRespondToChoice, gameId, gameState, user]
  );

  // Chat message
  const sendChat = useCallback(
    (text: string) => {
      sendChatMessage(text);
    },
    [sendChatMessage]
  );

  // Pending choice state
  const pendingChoiceState = useMemo(
    (): PendingChoiceState => ({
      choice: pendingChoice,
      deadline: choiceDeadline,
      hasPendingChoice: !!pendingChoice,
    }),
    [pendingChoice, choiceDeadline]
  );

  // Action capabilities
  const capabilities = useMemo((): ActionCapabilities => {
    if (!gameId || !gameState) {
      return {
        canSubmitMove: false,
        canRespondToChoice: false,
        canSendChat: false,
        disabledReason: 'Not connected to a game',
      };
    }

    if (gameState.gameStatus !== 'active') {
      return {
        canSubmitMove: false,
        canRespondToChoice: false,
        canSendChat: true, // Chat may still work in finished games
        disabledReason: `Game is ${gameState.gameStatus}`,
      };
    }

    // Spectators are always read-only: they can observe clocks, phases, and
    // chat, but must never submit moves or decision responses.
    if (!isUserPlayer(gameState, user?.id)) {
      return {
        canSubmitMove: false,
        canRespondToChoice: false,
        canSendChat: true,
        disabledReason: 'Spectators cannot submit moves or decisions',
      };
    }

    return {
      canSubmitMove: true,
      canRespondToChoice: !!pendingChoice,
      canSendChat: true,
    };
  }, [gameId, gameState, pendingChoice, user]);

  return {
    // Core actions
    submitMove,
    submitMoveById,
    submitPlacement,
    submitMovement,
    respondToChoice,
    sendChat,
    // State
    pendingChoice: pendingChoiceState,
    capabilities,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: usePendingChoice
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook focused on pending choice state and handling
 *
 * Usage:
 * ```tsx
 * const { choice, deadline, respond, timeRemaining } = usePendingChoice();
 *
 * if (choice) {
 *   return (
 *     <ChoiceDialog
 *       choice={choice}
 *       timeRemaining={timeRemaining}
 *       onSelect={(option) => respond(option)}
 *     />
 *   );
 * }
 * ```
 */
export function usePendingChoice() {
  const { pendingChoice, choiceDeadline, respondToChoice } = useGame();

  // Centralised client-side countdown for the active decision phase. This
  // is the single timer used by both ChoiceDialog and HUD/diagnostics via
  // hooks like useBackendDecisionUI.
  const [timeRemainingMs, setTimeRemainingMs] = useState<number | null>(null);

  useEffect(() => {
    if (!pendingChoice || !choiceDeadline) {
      setTimeRemainingMs(null);
      return;
    }

    const update = () => {
      const remaining = choiceDeadline - Date.now();
      setTimeRemainingMs(remaining > 0 ? remaining : 0);
    };

    // Initial tick then regular updates.
    update();
    const intervalId = setInterval(update, 250);

    return () => {
      clearInterval(intervalId);
    };
  }, [pendingChoice, choiceDeadline]);

  const respond = useCallback(
    <T>(selectedOption: T) => {
      if (pendingChoice) {
        respondToChoice(pendingChoice, selectedOption);
      }
    },
    [pendingChoice, respondToChoice]
  );

  const viewModel = useMemo(() => {
    if (!pendingChoice) return null;
    return getChoiceViewModel(pendingChoice);
  }, [pendingChoice]);

  const isExpiredClientSide = useMemo(() => {
    if (!choiceDeadline) return false;
    if (timeRemainingMs === null) return false;
    return timeRemainingMs === 0;
  }, [choiceDeadline, timeRemainingMs]);

  const activeView: PendingChoiceView | null =
    pendingChoice && viewModel
      ? {
          choice: pendingChoice,
          viewModel,
          timeRemainingMs,
          isExpiredClientSide,
        }
      : null;

  return {
    // Legacy-style fields (kept for backward compatibility)
    choice: pendingChoice,
    deadline: choiceDeadline,
    hasChoice: !!pendingChoice,
    respond,
    timeRemaining: timeRemainingMs,
    choiceType: pendingChoice?.type ?? null,
    // New SSOT-derived view for decision UIs
    viewModel,
    isExpiredClientSide,
    view: activeView,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useChatMessages
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for chat functionality
 *
 * Usage:
 * ```tsx
 * const { messages, sendMessage } = useChatMessages();
 *
 * return (
 *   <>
 *     {messages.map((msg, i) => (
 *       <ChatMessage key={i} sender={msg.sender} text={msg.text} />
 *     ))}
 *     <ChatInput onSend={sendMessage} />
 *   </>
 * );
 * ```
 */
export function useChatMessages() {
  const { chatMessages, sendChatMessage } = useGame();

  const sendMessage = useCallback(
    (text: string) => {
      if (text.trim()) {
        sendChatMessage(text.trim());
      }
    },
    [sendChatMessage]
  );

  return {
    messages: chatMessages,
    sendMessage,
    messageCount: chatMessages.length,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useValidMoves
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for accessing valid moves from the server
 *
 * Usage:
 * ```tsx
 * const { moves, hasValidMoves, findMoveFor, getTargetsFrom } = useValidMoves();
 *
 * const handleCellClick = (pos: Position) => {
 *   if (selectedPos) {
 *     const move = findMoveFor(selectedPos, pos);
 *     if (move) submitMove(move);
 *   } else {
 *     const targets = getTargetsFrom(pos);
 *     setValidTargets(targets);
 *   }
 * };
 * ```
 */
export function useValidMoves() {
  const { validMoves } = useGame();

  const positionsEqual = useCallback((pos1?: Position, pos2?: Position): boolean => {
    if (!pos1 || !pos2) return false;
    return pos1.x === pos2.x && pos1.y === pos2.y && (pos1.z || 0) === (pos2.z || 0);
  }, []);

  // Find specific move by from/to positions
  const findMoveFor = useCallback(
    (from: Position, to: Position): Move | undefined => {
      if (!validMoves) return undefined;
      return validMoves.find(
        (m) => m.from && positionsEqual(m.from, from) && positionsEqual(m.to, to)
      );
    },
    [validMoves, positionsEqual]
  );

  // Get all valid target positions from a source
  const getTargetsFrom = useCallback(
    (from: Position): Position[] => {
      if (!validMoves) return [];
      return validMoves.filter((m) => m.from && positionsEqual(m.from, from)).map((m) => m.to);
    },
    [validMoves, positionsEqual]
  );

  // Get all valid placement positions
  const getPlacementPositions = useCallback((): Position[] => {
    if (!validMoves) return [];
    return validMoves.filter((m) => m.type === 'place_ring').map((m) => m.to);
  }, [validMoves]);

  // Check if a position is a valid target
  const isValidTarget = useCallback(
    (from: Position | undefined, to: Position): boolean => {
      if (!validMoves) return false;
      if (!from) {
        // For placements, check if 'to' is a valid placement position
        return validMoves.some((m) => m.type === 'place_ring' && positionsEqual(m.to, to));
      }
      return validMoves.some(
        (m) => m.from && positionsEqual(m.from, from) && positionsEqual(m.to, to)
      );
    },
    [validMoves, positionsEqual]
  );

  return {
    moves: validMoves ?? [],
    hasValidMoves: !!validMoves && validMoves.length > 0,
    findMoveFor,
    getTargetsFrom,
    getPlacementPositions,
    isValidTarget,
  };
}
