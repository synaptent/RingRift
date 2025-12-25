/**
 * @fileoverview useBackendTelemetry Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game telemetry.
 * It manages weird state tracking and calibration event reporting, not game rules.
 *
 * This adapter:
 * - Tracks weird state (ANM, FE, structural stalemate, LPS) for telemetry
 * - Manages calibration game completion events
 * - Provides context for resignation telemetry
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/ux/UX_RULES_TELEMETRY_SPEC.md
 */

import { useRef, useEffect, useCallback } from 'react';
import type { GameState, GameResult } from '../../shared/types/game';
import type { RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import { getWeirdStateBanner } from '../utils/gameStateWeirdness';
import { isSurfaceableWeirdStateType } from '../../shared/engine/weirdStateReasons';
import {
  sendDifficultyCalibrationEvent,
  getDifficultyCalibrationSession,
  clearDifficultyCalibrationSession,
} from '../utils/difficultyCalibrationTelemetry';

/**
 * Weird state context for telemetry.
 */
export interface WeirdStateContext {
  type: RulesUxWeirdStateType;
  durationSeconds: number;
}

/**
 * Return type for useBackendTelemetry hook.
 */
export interface UseBackendTelemetryReturn {
  /** Current weird state type */
  weirdStateType: RulesUxWeirdStateType | 'none';
  /** Timestamp when weird state was first seen */
  weirdStateFirstSeenAt: number | null;
  /** Set of weird state types already reported for resignation */
  weirdStateResignReported: Set<string>;
  /** Mark a weird state type as reported for resignation */
  markWeirdStateResignReported: (type: string) => void;
  /** Get the current weird state context for telemetry */
  getWeirdStateContext: () => WeirdStateContext | null;
  /** Whether the calibration event has been reported for this game */
  isCalibrationEventReported: boolean;
}

/**
 * Custom hook for managing backend game telemetry state.
 *
 * Handles:
 * - Weird state tracking for resign telemetry
 * - Calibration game completion events
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param gameState - Current game state from backend
 * @param victoryState - Victory/end state
 * @param routeGameId - Route game ID (from URL params)
 * @returns Object with telemetry state and actions
 */
export function useBackendTelemetry(
  gameState: GameState | null,
  victoryState: GameResult | null,
  routeGameId: string
): UseBackendTelemetryReturn {
  // Weird-state tracking refs
  const weirdStateFirstSeenAtRef = useRef<number | null>(null);
  const weirdStateTypeRef = useRef<RulesUxWeirdStateType | 'none'>('none');
  const weirdStateResignReportedRef = useRef<Set<string>>(new Set());

  // Calibration tracking
  const calibrationEventReportedRef = useRef(false);

  // Mark weird state as reported for resignation
  const markWeirdStateResignReported = useCallback((type: string) => {
    weirdStateResignReportedRef.current.add(type);
  }, []);

  // Get current weird state context for telemetry
  const getWeirdStateContext = useCallback((): WeirdStateContext | null => {
    const type = weirdStateTypeRef.current;
    const firstSeenAt = weirdStateFirstSeenAtRef.current;

    if (type === 'none' || firstSeenAt === null) {
      return null;
    }

    return {
      type: type as RulesUxWeirdStateType,
      durationSeconds: Math.max(0, Math.round((Date.now() - firstSeenAt) / 1000)),
    };
  }, []);

  // Track the first time each weird-state type appears so that we can emit
  // coarse-grained rules-UX telemetry when the local player resigns while a
  // weird state is active.
  useEffect(() => {
    if (!gameState) {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    const weird = getWeirdStateBanner(gameState, { victoryState });
    const nextType = weird.type;
    const currentType = weirdStateTypeRef.current;

    if (nextType === 'none') {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    if (!isSurfaceableWeirdStateType(nextType as RulesUxWeirdStateType)) {
      weirdStateTypeRef.current = 'none';
      weirdStateFirstSeenAtRef.current = null;
      return;
    }

    if (currentType !== nextType) {
      weirdStateTypeRef.current = nextType as RulesUxWeirdStateType;
      weirdStateFirstSeenAtRef.current = Date.now();
    }
  }, [gameState, victoryState]);

  // Emit a single difficulty_calibration_game_completed event for calibration games
  // that have a stored session (created from the Lobby). This remains purely
  // client-driven for now and does not depend on server-side calibration flags.
  useEffect(() => {
    if (!victoryState || calibrationEventReportedRef.current) {
      return;
    }

    const session = getDifficultyCalibrationSession(routeGameId);
    if (!session || !session.isCalibrationOptIn) {
      return;
    }

    if (!gameState) {
      return;
    }

    let result: 'win' | 'loss' | 'draw' | 'abandoned' = 'abandoned';

    if (victoryState.reason === 'draw') {
      result = 'draw';
    } else if (victoryState.reason === 'abandonment') {
      result = 'abandoned';
    } else if (typeof victoryState.winner === 'number') {
      const winnerPlayer = gameState.players.find((p) => p.playerNumber === victoryState.winner);
      if (winnerPlayer?.type === 'human') {
        result = 'win';
      } else if (winnerPlayer?.type === 'ai') {
        result = 'loss';
      }
    }

    const movesPlayed =
      Array.isArray(gameState.moveHistory) && gameState.moveHistory.length > 0
        ? gameState.moveHistory.length
        : undefined;

    calibrationEventReportedRef.current = true;
    clearDifficultyCalibrationSession(routeGameId);

    void sendDifficultyCalibrationEvent({
      type: 'difficulty_calibration_game_completed',
      boardType: session.boardType,
      numPlayers: session.numPlayers,
      difficulty: session.difficulty,
      isCalibrationOptIn: true,
      result,
      movesPlayed,
    });
  }, [victoryState, gameState, routeGameId]);

  return {
    weirdStateType: weirdStateTypeRef.current,
    weirdStateFirstSeenAt: weirdStateFirstSeenAtRef.current,
    weirdStateResignReported: weirdStateResignReportedRef.current,
    markWeirdStateResignReported,
    getWeirdStateContext,
    isCalibrationEventReported: calibrationEventReportedRef.current,
  };
}

export default useBackendTelemetry;
