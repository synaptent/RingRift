/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useGameState Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides read-only access to current game state with optional view model
 * transformation. This hook wraps GameContext to provide a focused interface
 * for components that only need to read game state.
 *
 * Benefits:
 * - Clear separation: components that read state vs. those that modify it
 * - View model integration: get presentation-ready data in one call
 * - Memoized transformations: prevent unnecessary re-renders
 */

import { useMemo } from 'react';
import { useGame } from '../contexts/GameContext';
import { useAccessibility } from '../contexts/AccessibilityContext';
import type { GameState, GameResult, Player, Move } from '../../shared/types/game';
import type {
  DecisionAutoResolvedMeta,
  DecisionPhaseTimeoutWarningPayload,
} from '../../shared/types/websocket';
import {
  toHUDViewModel,
  toEventLogViewModel,
  toBoardViewModel,
  toVictoryViewModel,
  type HUDViewModel,
  type EventLogViewModel,
  type BoardViewModel,
  type VictoryViewModel,
  type ToHUDViewModelOptions,
  type ToBoardViewModelOptions,
} from '../adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Raw game state from the context
 */
export interface RawGameState {
  /** Current game ID (null if not connected) */
  gameId: string | null;
  /** Full game state from server/sandbox */
  gameState: GameState | null;
  /** Valid moves for current player (if provided by server) */
  validMoves: Move[] | null;
  /** Terminal game result (null if game ongoing) */
  victoryState: GameResult | null;
  /** Summary of the most recently auto-resolved decision, if any, on the latest update */
  decisionAutoResolved: DecisionAutoResolvedMeta | null;
  /**
   * Optional payload for an in-progress decision-phase timeout warning, if the
   * server has indicated that a pending decision is approaching auto-resolution.
   */
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null;
  /** Players array (convenience accessor) */
  players: Player[];
  /** Current player object (convenience accessor) */
  currentPlayer: Player | undefined;
  /** Streaming AI evaluation history for the current game (analysis mode). */
  evaluationHistory: import('../../shared/types/websocket').PositionEvaluationPayload['data'][];
}

/**
 * Options for HUD view model generation
 */
export interface UseHUDViewModelOptions {
  instruction?: string;
  currentUserId?: string;
}

/**
 * Options for victory view model generation
 */
export interface UseVictoryViewModelOptions {
  currentUserId?: string;
  isDismissed?: boolean;
}

/**
 * Options for event log view model generation
 */
export interface UseEventLogViewModelOptions {
  systemEvents?: string[];
  maxEntries?: number;
  /** Override board type for position formatting (defaults to gameState.boardType) */
  boardType?: import('../../shared/types/game').BoardType;
  /** When true, square board ranks are computed from the bottom (chess style) */
  squareRankFromBottom?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useGameState
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for read-only access to raw game state
 *
 * Usage:
 * ```tsx
 * const { gameState, players, currentPlayer, victoryState } = useGameState();
 *
 * if (!gameState) return <Loading />;
 * return <div>Current phase: {gameState.currentPhase}</div>;
 * ```
 */
export function useGameState(): RawGameState {
  const {
    gameId,
    gameState,
    validMoves,
    victoryState,
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
    evaluationHistory,
  } = useGame();

  return useMemo(() => {
    const players = gameState?.players ?? [];
    const currentPlayer = players.find((p) => p.playerNumber === gameState?.currentPlayer);

    return {
      gameId,
      gameState,
      validMoves,
      victoryState,
      players,
      currentPlayer,
      decisionAutoResolved,
      decisionPhaseTimeoutWarning,
      evaluationHistory,
    };
  }, [
    gameId,
    gameState,
    validMoves,
    victoryState,
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
    evaluationHistory,
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useHUDViewModel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook that returns a HUDViewModel for the current game state
 *
 * Usage:
 * ```tsx
 * const hudVM = useHUDViewModel({ instruction: 'Place a ring', currentUserId: user.id });
 *
 * if (!hudVM) return null;
 * return <GameHUD viewModel={hudVM} />;
 * ```
 */
export function useHUDViewModel(options: UseHUDViewModelOptions = {}): HUDViewModel | null {
  const { gameState, victoryState, connectionStatus, lastHeartbeatAt } = useGame();
  const { colorVisionMode } = useAccessibility();
  const { instruction, currentUserId } = options;

  return useMemo(() => {
    if (!gameState) return null;

    // Derive isSpectator (simplified - actual logic may involve user lookup)
    const isSpectator = false; // This should be passed or derived from auth context

    const viewModelOptions: ToHUDViewModelOptions = {
      instruction,
      connectionStatus,
      lastHeartbeatAt,
      isSpectator,
      currentUserId,
      colorVisionMode,
      victoryState,
    };

    return toHUDViewModel(gameState, viewModelOptions);
  }, [
    gameState,
    victoryState,
    connectionStatus,
    lastHeartbeatAt,
    instruction,
    currentUserId,
    colorVisionMode,
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useBoardViewModel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook that returns a BoardViewModel for rendering
 *
 * Usage:
 * ```tsx
 * const boardVM = useBoardViewModel({ selectedPosition, validTargets });
 *
 * if (!boardVM) return null;
 * return <BoardRenderer viewModel={boardVM} />;
 * ```
 */
export function useBoardViewModel(options: ToBoardViewModelOptions = {}): BoardViewModel | null {
  const { gameState } = useGame();
  const { colorVisionMode } = useAccessibility();
  const { selectedPosition, validTargets } = options;

  return useMemo(() => {
    if (!gameState?.board) return null;

    return toBoardViewModel(gameState.board, { selectedPosition, validTargets, colorVisionMode });
  }, [gameState?.board, selectedPosition, validTargets, colorVisionMode]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useEventLogViewModel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook that returns an EventLogViewModel for the game history
 *
 * Usage:
 * ```tsx
 * const logVM = useEventLogViewModel({ systemEvents: eventLog, maxEntries: 30 });
 *
 * return <EventLogDisplay viewModel={logVM} />;
 * ```
 */
export function useEventLogViewModel(options: UseEventLogViewModelOptions = {}): EventLogViewModel {
  const { gameState, victoryState } = useGame();
  const { systemEvents = [], maxEntries = 40, boardType, squareRankFromBottom } = options;

  // Default to gameState.boardType if not explicitly provided
  const effectiveBoardType = boardType ?? gameState?.boardType ?? 'square8';
  // Default squareRankFromBottom based on board type if not explicitly provided
  const effectiveSquareRankFromBottom =
    squareRankFromBottom ?? (effectiveBoardType === 'square8' || effectiveBoardType === 'square19');

  return useMemo(() => {
    const history = gameState?.history ?? [];
    return toEventLogViewModel(history, systemEvents, victoryState, {
      maxEntries,
      boardType: effectiveBoardType,
      squareRankFromBottom: effectiveSquareRankFromBottom,
    });
  }, [
    gameState?.history,
    victoryState,
    systemEvents,
    maxEntries,
    effectiveBoardType,
    effectiveSquareRankFromBottom,
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useVictoryViewModel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook that returns a VictoryViewModel when the game is over
 *
 * Usage:
 * ```tsx
 * const victoryVM = useVictoryViewModel({ currentUserId: user.id, isDismissed });
 *
 * if (!victoryVM) return null;
 * return <VictoryModal viewModel={victoryVM} />;
 * ```
 */
export function useVictoryViewModel(
  options: UseVictoryViewModelOptions = {}
): VictoryViewModel | null {
  const { gameState, victoryState } = useGame();
  const { colorVisionMode } = useAccessibility();
  const { currentUserId, isDismissed = false } = options;

  return useMemo(() => {
    if (!victoryState) return null;

    const players = gameState?.players ?? [];
    return toVictoryViewModel(victoryState, players, gameState ?? undefined, {
      currentUserId,
      isDismissed,
      colorVisionMode,
    });
  }, [gameState, victoryState, currentUserId, isDismissed, colorVisionMode]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useGamePhase
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for phase-specific logic
 *
 * Usage:
 * ```tsx
 * const { phase, isPlacementPhase, isMovementPhase, isCapturePhase } = useGamePhase();
 *
 * if (isPlacementPhase) {
 *   // Show placement hints
 * }
 * ```
 */
export function useGamePhase() {
  const { gameState } = useGame();

  return useMemo(() => {
    const phase = gameState?.currentPhase ?? null;

    return {
      phase,
      isPlacementPhase: phase === 'ring_placement',
      isMovementPhase: phase === 'movement',
      isCapturePhase: phase === 'capture',
      isChainCapturePhase: phase === 'chain_capture',
      isLineProcessingPhase: phase === 'line_processing',
      isTerritoryProcessingPhase: phase === 'territory_processing',
      isInProcessingPhase: phase === 'line_processing' || phase === 'territory_processing',
    };
  }, [gameState?.currentPhase]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useGameStatus
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for game status queries
 *
 * Usage:
 * ```tsx
 * const { isActive, isFinished, isWaiting, status } = useGameStatus();
 *
 * if (isFinished) {
 *   // Show final results
 * }
 * ```
 */
export function useGameStatus() {
  const { gameState, victoryState } = useGame();

  return useMemo(() => {
    const status = gameState?.gameStatus ?? 'waiting';

    return {
      status,
      isActive: status === 'active',
      isFinished: status === 'finished' || status === 'completed',
      isWaiting: status === 'waiting',
      isPaused: status === 'paused',
      isAbandoned: status === 'abandoned',
      hasVictory: !!victoryState,
      winner: victoryState?.winner,
    };
  }, [gameState?.gameStatus, victoryState]);
}
