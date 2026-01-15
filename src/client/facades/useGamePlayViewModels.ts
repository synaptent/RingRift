/**
 * useGamePlayViewModels - Derives common view models from a GameFacade
 *
 * This hook takes a GameFacade and derives all the view models needed for
 * rendering the game play area. Both BackendGameHost and SandboxGameHost can
 * use this hook to get consistent view model derivation.
 *
 * View models derived:
 * - boardViewModel: For BoardView component
 * - hudViewModel: For GameHUD/MobileGameHUD components
 * - victoryViewModel: For VictoryModal component
 * - eventLogViewModel: For GameEventLog component
 *
 * @module facades/useGamePlayViewModels
 */

import { useMemo } from 'react';
import type { Position } from '../../shared/types/game';
import { positionToString } from '../../shared/types/game';
import type { GameFacade } from './GameFacade';
import { useAccessibility } from '../contexts/AccessibilityContext';
import {
  toBoardViewModel,
  toHUDViewModel,
  toVictoryViewModel,
  toEventLogViewModel,
  deriveBoardDecisionHighlights,
  type BoardViewModel,
  type HUDViewModel,
  type VictoryViewModel,
  type EventLogViewModel,
} from '../adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for deriving view models.
 */
export interface GamePlayViewModelOptions {
  /** Currently selected position on the board */
  selectedPosition?: Position;
  /** Valid target positions for the selected piece */
  validTargets?: Position[];
  /** Additional event log entries (system events, etc.) */
  additionalEventLogEntries?: string[];
  /** Whether victory modal has been dismissed */
  isVictoryModalDismissed?: boolean;
  /** Whether to show optional capture highlights */
  showCaptureHighlights?: boolean;
}

/**
 * All view models derived from the facade.
 */
export interface GamePlayViewModels {
  /** Board view model for BoardView */
  boardViewModel: BoardViewModel | null;
  /** HUD view model for GameHUD/MobileGameHUD */
  hudViewModel: HUDViewModel | null;
  /** Victory view model for VictoryModal */
  victoryViewModel: VictoryViewModel | null;
  /** Event log view model for GameEventLog */
  eventLogViewModel: EventLogViewModel;
  /** Decision highlights for board overlays */
  decisionHighlights: ReturnType<typeof deriveBoardDecisionHighlights> | undefined;
  /** Whether optional capture is available */
  hasOptionalCapture: boolean;
  /** Whether skip capture is available */
  hasSkipCapture: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Derive all game play view models from a facade.
 *
 * This centralizes the view model derivation logic that was previously
 * duplicated between BackendGameHost and SandboxGameHost.
 */
export function useGamePlayViewModels(
  facade: GameFacade | null,
  options: GamePlayViewModelOptions = {}
): GamePlayViewModels {
  const { colorVisionMode } = useAccessibility();
  const {
    selectedPosition,
    validTargets = [],
    additionalEventLogEntries = [],
    isVictoryModalDismissed = false,
    showCaptureHighlights = true,
  } = options;

  // Derive decision highlights from pending choice
  const decisionHighlights = useMemo(() => {
    if (!facade?.gameState || !facade.decisionState.pendingChoice) {
      return undefined;
    }
    return deriveBoardDecisionHighlights(facade.gameState, facade.decisionState.pendingChoice);
  }, [facade?.gameState, facade?.decisionState.pendingChoice]);

  // Derive optional capture highlights when in capture phase with no explicit choice
  const captureHighlights = useMemo(() => {
    if (!showCaptureHighlights) {
      return undefined;
    }

    if (decisionHighlights) {
      // Already have decision highlights, don't override
      return undefined;
    }

    if (!facade?.gameState || facade.gameState.gameStatus !== 'active') {
      return undefined;
    }

    if (facade.gameState.currentPhase !== 'capture') {
      return undefined;
    }

    const captureMoves = facade.validMoves.filter((m) => m.type === 'overtaking_capture');
    if (captureMoves.length === 0) {
      return undefined;
    }

    type CaptureHighlight = { positionKey: string; intensity: 'primary' | 'secondary' };
    const highlights: CaptureHighlight[] = [];
    const seenPrimary = new Set<string>();
    const seenAny = new Set<string>();

    for (const move of captureMoves) {
      const target = move.captureTarget as Position | undefined;
      const landing = move.to as Position | undefined;

      if (landing) {
        const key = positionToString(landing);
        if (!seenPrimary.has(key)) {
          seenPrimary.add(key);
          seenAny.add(key);
          highlights.push({ positionKey: key, intensity: 'primary' });
        }
      }

      if (target) {
        const key = positionToString(target);
        if (!seenAny.has(key)) {
          seenAny.add(key);
          highlights.push({ positionKey: key, intensity: 'secondary' });
        }
      }
    }

    if (highlights.length === 0) {
      return undefined;
    }

    return {
      choiceKind: 'capture_direction' as const,
      highlights,
    };
  }, [facade?.gameState, facade?.validMoves, decisionHighlights, showCaptureHighlights]);

  // Merge decision highlights with capture highlights
  const mergedHighlights = decisionHighlights ?? captureHighlights;

  // Check for optional capture availability
  const hasOptionalCapture = useMemo(() => {
    if (!facade?.gameState || facade.gameState.currentPhase !== 'capture') {
      return false;
    }
    return facade.validMoves.some((m) => m.type === 'overtaking_capture');
  }, [facade?.gameState, facade?.validMoves]);

  const hasSkipCapture = useMemo(() => {
    if (!facade?.gameState || facade.gameState.currentPhase !== 'capture') {
      return false;
    }
    return facade.validMoves.some((m) => m.type === 'skip_capture');
  }, [facade?.gameState, facade?.validMoves]);

  // Board view model
  const boardViewModel = useMemo<BoardViewModel | null>(() => {
    if (!facade?.gameState) {
      return null;
    }

    const effectiveSelected = selectedPosition ?? facade.mustMoveFrom;

    return toBoardViewModel(facade.gameState.board, {
      selectedPosition: effectiveSelected,
      validTargets,
      decisionHighlights: mergedHighlights,
      colorVisionMode,
    });
  }, [
    facade?.gameState,
    facade?.mustMoveFrom,
    selectedPosition,
    validTargets,
    mergedHighlights,
    colorVisionMode,
  ]);

  // HUD view model
  const hudViewModel = useMemo<HUDViewModel | null>(() => {
    if (!facade?.gameState) {
      return null;
    }

    const baseHudVM = toHUDViewModel(facade.gameState, {
      connectionStatus:
        facade.connectionStatus === 'local-only' ? 'connected' : facade.connectionStatus,
      lastHeartbeatAt: null,
      isSpectator: !facade.isPlayer,
      currentUserId: facade.currentUserId,
      colorVisionMode,
      pendingChoice: facade.decisionState.pendingChoice,
      choiceDeadline: facade.decisionState.choiceDeadline,
      choiceTimeRemainingMs: facade.decisionState.choiceTimeRemainingMs,
      decisionIsServerCapped: facade.decisionState.isServerCapped,
      victoryState: facade.victoryState,
      gameEndExplanation: facade.gameEndExplanation,
    });

    // Augment with optional capture decision phase if applicable
    if (baseHudVM && !baseHudVM.decisionPhase && hasOptionalCapture) {
      const { currentPlayer, players } = facade.gameState;
      const actingPlayer = players.find((p) => p.playerNumber === currentPlayer);
      const actingPlayerName =
        actingPlayer?.username || `Player ${actingPlayer?.playerNumber ?? currentPlayer}`;

      return {
        ...baseHudVM,
        decisionPhase: {
          isActive: true,
          actingPlayerNumber: currentPlayer,
          actingPlayerName,
          isLocalActor: facade.isMyTurn,
          label: hasSkipCapture
            ? 'Optional capture available'
            : 'Capture available from this stack',
          description: hasSkipCapture
            ? 'You may jump over a neighbouring stack for an overtaking capture, or skip capture to continue this turn.'
            : 'You may jump over a neighbouring stack for an overtaking capture from your last move.',
          shortLabel: 'Capture opportunity',
          timeRemainingMs: null,
          showCountdown: false,
          warningThresholdMs: undefined,
          isServerCapped: undefined,
          spectatorLabel: hasSkipCapture
            ? `${actingPlayerName} may choose an overtaking capture or skip.`
            : `${actingPlayerName} may choose an overtaking capture.`,
          statusChip: {
            text: hasSkipCapture
              ? 'Capture available – click a landing or skip'
              : 'Capture available – click a landing',
            tone: 'attention' as const,
          },
          canSkip: hasSkipCapture,
        },
      };
    }

    return baseHudVM;
  }, [
    facade?.gameState,
    facade?.connectionStatus,
    facade?.isPlayer,
    facade?.currentUserId,
    facade?.decisionState,
    facade?.victoryState,
    facade?.gameEndExplanation,
    facade?.isMyTurn,
    hasOptionalCapture,
    hasSkipCapture,
    colorVisionMode,
  ]);

  // Victory view model
  const victoryViewModel = useMemo<VictoryViewModel | null>(() => {
    if (!facade?.gameState || !facade.victoryState) {
      return null;
    }

    return toVictoryViewModel(facade.victoryState, facade.gameState.players, facade.gameState, {
      currentUserId: facade.currentUserId,
      isDismissed: isVictoryModalDismissed,
      colorVisionMode,
      gameEndExplanation: facade.gameEndExplanation,
    });
  }, [
    facade?.gameState,
    facade?.victoryState,
    facade?.currentUserId,
    facade?.gameEndExplanation,
    isVictoryModalDismissed,
    colorVisionMode,
  ]);

  // Event log view model
  const eventLogBoardType = facade?.gameState?.boardType ?? 'square8';
  const eventLogSquareRankFromBottom =
    eventLogBoardType === 'square8' || eventLogBoardType === 'square19';
  const eventLogViewModel = useMemo<EventLogViewModel>(() => {
    return toEventLogViewModel(
      facade?.gameState?.history ?? [],
      additionalEventLogEntries,
      facade?.victoryState ?? null,
      {
        maxEntries: 40,
        boardType: eventLogBoardType,
        squareRankFromBottom: eventLogSquareRankFromBottom,
      }
    );
  }, [
    facade?.gameState?.history,
    additionalEventLogEntries,
    facade?.victoryState,
    eventLogBoardType,
    eventLogSquareRankFromBottom,
  ]);

  return {
    boardViewModel,
    hudViewModel,
    victoryViewModel,
    eventLogViewModel,
    decisionHighlights: mergedHighlights,
    hasOptionalCapture,
    hasSkipCapture,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL HOOKS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Derive instruction text for the current game state.
 */
export function useInstructionText(facade: GameFacade | null): string | undefined {
  return useMemo(() => {
    if (!facade?.gameState) {
      return undefined;
    }

    const { gameState, isPlayer, isMyTurn } = facade;
    const currentPlayer = gameState.players.find((p) => p.playerNumber === gameState.currentPlayer);

    if (!isPlayer) {
      return `Spectating: ${currentPlayer?.username || `Player ${currentPlayer?.playerNumber}`}'s turn`;
    }

    if (!isMyTurn) {
      return `Waiting for ${currentPlayer?.username || `Player ${currentPlayer?.playerNumber}`}...`;
    }

    switch (gameState.currentPhase) {
      case 'ring_placement':
        return 'Place rings on an empty cell or on top of an existing stack.';
      case 'movement':
        return 'Select a stack to move.';
      case 'capture':
        return 'Select a stack to capture with.';
      case 'chain_capture':
        return 'Chain capture in progress – select next capture target.';
      case 'line_processing':
        return 'Line processing – choose how to resolve your completed line.';
      case 'territory_processing':
        return 'Territory processing – resolve disconnected regions.';
      case 'forced_elimination':
        return 'No legal moves available – select a stack to eliminate from.';
      default:
        return 'Make your move.';
    }
  }, [facade?.gameState, facade?.isPlayer, facade?.isMyTurn]);
}
