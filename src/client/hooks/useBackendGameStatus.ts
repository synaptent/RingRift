/**
 * @fileoverview useBackendGameStatus Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game status state.
 * It manages UI state for game status, modals, and resignation, not rules logic.
 *
 * Canonical SSoT:
 * - Victory conditions: `src/shared/engine/aggregates/Victory.ts`
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 *
 * This adapter:
 * - Tracks fatal game error state
 * - Tracks victory modal dismissal state
 * - Manages resignation confirmation state
 * - Handles resignation API calls with telemetry
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { toast } from 'react-hot-toast';
import { gameApi } from '../services/api';
import type { GameState, GameResult } from '../../shared/types/game';
import type { RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import { sendRulesUxEvent } from '../utils/rulesUxTelemetry';

/**
 * Fatal game error state.
 */
export interface FatalGameError {
  message: string;
  technical?: string;
}

/**
 * Dependencies required by the game status hook.
 */
export interface UseBackendGameStatusDeps {
  /** Game ID from the backend */
  gameId: string | null;
  /** Current game state */
  gameState: GameState | null;
  /** Victory/end state */
  victoryState: GameResult | null;
  /** Route game ID (from URL params) */
  routeGameId: string;
  /** Current weird state type (for telemetry) */
  weirdStateType: RulesUxWeirdStateType | 'none';
  /** Timestamp when weird state was first seen (for telemetry) */
  weirdStateFirstSeenAt: number | null;
  /** Set of weird state types already reported for resignation */
  weirdStateResignReported: Set<string>;
  /** Mark a weird state type as reported */
  markWeirdStateResignReported: (type: string) => void;
}

/**
 * Return type for useBackendGameStatus hook.
 */
export interface UseBackendGameStatusReturn {
  /** Fatal error state */
  fatalGameError: FatalGameError | null;
  /** Set fatal error */
  setFatalGameError: (error: FatalGameError | null) => void;
  /** Whether the victory modal has been dismissed */
  isVictoryModalDismissed: boolean;
  /** Dismiss the victory modal */
  dismissVictoryModal: () => void;
  /** Whether a resignation is in progress */
  isResigning: boolean;
  /** Whether the resign confirmation dialog is open */
  isResignConfirmOpen: boolean;
  /** Set whether the resign confirmation dialog is open */
  setIsResignConfirmOpen: (open: boolean) => void;
  /** Handle resignation */
  handleResign: () => Promise<void>;
}

/**
 * Custom hook for managing backend game status state.
 *
 * Handles:
 * - Fatal game error state
 * - Victory modal dismissal
 * - Resignation flow with telemetry
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param deps - Dependencies including game state and telemetry callbacks
 * @returns Object with status state and actions
 */
export function useBackendGameStatus(deps: UseBackendGameStatusDeps): UseBackendGameStatusReturn {
  const {
    gameId,
    gameState,
    victoryState,
    routeGameId,
    weirdStateType,
    weirdStateFirstSeenAt,
    weirdStateResignReported,
    markWeirdStateResignReported,
  } = deps;

  // Fatal game error state
  const [fatalGameError, setFatalGameError] = useState<FatalGameError | null>(null);

  // Victory modal dismissal state
  const [isVictoryModalDismissed, setIsVictoryModalDismissed] = useState(false);

  // Resignation state
  const [isResigning, setIsResigning] = useState(false);
  const [isResignConfirmOpen, setIsResignConfirmOpen] = useState(false);

  // Reset victory modal dismissal whenever the active game or victory state changes
  useEffect(() => {
    setIsVictoryModalDismissed(false);
  }, [routeGameId, victoryState]);

  // Dismiss victory modal
  const dismissVictoryModal = useCallback(() => {
    setIsVictoryModalDismissed(true);
  }, []);

  // Handle resignation with telemetry
  const handleResign = useCallback(async () => {
    if (!gameId || isResigning) return;

    setIsResigning(true);
    try {
      // Derive coarse board / difficulty context from the current GameState.
      let boardTypeForTelemetry: GameState['boardType'] | undefined;
      let numPlayersForTelemetry: number | undefined;
      let aiDifficultyForTelemetry: number | undefined;

      if (gameState) {
        boardTypeForTelemetry = gameState.boardType;
        numPlayersForTelemetry = gameState.players.length;

        const aiPlayers = gameState.players.filter((p) => p.type === 'ai');
        let maxDifficulty = 0;
        for (const p of aiPlayers) {
          const d = p.aiProfile?.difficulty ?? p.aiDifficulty;
          if (typeof d === 'number' && Number.isFinite(d) && d > maxDifficulty) {
            maxDifficulty = d;
          }
        }
        if (maxDifficulty > 0) {
          aiDifficultyForTelemetry = maxDifficulty;
        }
      }

      if (
        boardTypeForTelemetry &&
        typeof numPlayersForTelemetry === 'number' &&
        numPlayersForTelemetry > 0 &&
        weirdStateType &&
        weirdStateType !== 'none' &&
        !weirdStateResignReported.has(weirdStateType)
      ) {
        const secondsSinceWeirdState =
          typeof weirdStateFirstSeenAt === 'number'
            ? Math.max(0, Math.round((Date.now() - weirdStateFirstSeenAt) / 1000))
            : undefined;

        markWeirdStateResignReported(weirdStateType);

        void sendRulesUxEvent({
          type: 'rules_weird_state_resign',
          boardType: boardTypeForTelemetry,
          numPlayers: numPlayersForTelemetry,
          aiDifficulty: aiDifficultyForTelemetry,
          weirdStateType: weirdStateType as RulesUxWeirdStateType,
          secondsSinceWeirdState,
        });
      }

      await gameApi.leaveGame(gameId);
      toast.success('You have resigned from the game.');
      // The server will broadcast victory/game over state via WebSocket
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to resign';
      toast.error(message);
      setIsResigning(false);
    }
  }, [
    gameId,
    isResigning,
    gameState,
    weirdStateType,
    weirdStateFirstSeenAt,
    weirdStateResignReported,
    markWeirdStateResignReported,
  ]);

  return {
    fatalGameError,
    setFatalGameError,
    isVictoryModalDismissed,
    dismissVictoryModal,
    isResigning,
    isResignConfirmOpen,
    setIsResignConfirmOpen,
    handleResign,
  };
}

export default useBackendGameStatus;
