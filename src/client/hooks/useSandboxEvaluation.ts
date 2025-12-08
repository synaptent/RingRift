/**
 * useSandboxEvaluation - AI evaluation management for sandbox mode
 *
 * This hook extracts AI evaluation functionality from SandboxGameHost:
 * - Requesting AI position evaluation
 * - Auto-evaluation when developer tools are enabled
 * - Tracking evaluation history for visualization
 * - Managing evaluation loading state
 *
 * @module hooks/useSandboxEvaluation
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { ClientSandboxEngine } from '../services/ClientSandboxEngine';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evaluation data from AI service.
 */
export type EvaluationData = PositionEvaluationPayload['data'];

/**
 * Options for the evaluation hook.
 */
export interface SandboxEvaluationOptions {
  /** Sandbox engine instance */
  engine: ClientSandboxEngine | null;
  /** Whether developer tools are enabled */
  developerToolsEnabled: boolean;
  /** Whether in replay mode (skip auto-evaluation) */
  isInReplayMode?: boolean;
  /** Whether viewing history (skip auto-evaluation) */
  isViewingHistory?: boolean;
  /** API endpoint for evaluation */
  evaluationEndpoint?: string;
}

/**
 * Return type for useSandboxEvaluation.
 */
export interface SandboxEvaluationState {
  /** Evaluation history for visualization */
  evaluationHistory: EvaluationData[];
  /** Current evaluation error message */
  evaluationError: string | null;
  /** Whether evaluation is in progress */
  isEvaluating: boolean;
  /** Request a new evaluation */
  requestEvaluation: () => Promise<void>;
  /** Clear evaluation history */
  clearHistory: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing AI evaluation in sandbox mode.
 */
export function useSandboxEvaluation(
  options: SandboxEvaluationOptions
): SandboxEvaluationState {
  const {
    engine,
    developerToolsEnabled,
    isInReplayMode = false,
    isViewingHistory = false,
    evaluationEndpoint = '/api/games/sandbox/evaluate',
  } = options;

  // Derive game state from engine
  const gameState = engine?.getGameState() ?? null;

  // State
  const [evaluationHistory, setEvaluationHistory] = useState<EvaluationData[]>([]);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Ref to track last evaluated move count for auto-evaluation
  const lastEvaluatedMoveRef = useRef<number>(-1);

  // Request evaluation
  const requestEvaluation = useCallback(async () => {
    // Get game state from engine directly to avoid forward reference issues
    const currentState = engine?.getGameState();
    if (!engine || !currentState) {
      return;
    }

    setIsEvaluating(true);
    setEvaluationError(null);

    try {
      const serialized = engine.getSerializedState();

      const response = await fetch(evaluationEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ state: serialized }),
      });

      if (!response.ok) {
        let message = 'Sandbox evaluation request failed.';
        try {
          const errorBody = (await response.json()) as { error?: string } | null;
          if (errorBody && typeof errorBody.error === 'string') {
            message = errorBody.error;
          }
        } catch {
          // Ignore JSON parse errors (HTML or empty responses)
        }

        const statusHint =
          response.status === 404
            ? 'AI evaluation is not enabled for this environment.'
            : response.status === 503
              ? 'Sandbox AI evaluation service is unavailable. Ensure the AI service is running.'
              : `HTTP ${response.status}`;

        setEvaluationError(`${message} ${statusHint}`.trim());
        return;
      }

      const data: EvaluationData = await response.json();
      setEvaluationHistory((prev) => [...prev, data]);
    } catch (err) {
      console.warn('[useSandboxEvaluation] Evaluation request threw', err);
      const message =
        err instanceof Error ? err.message : 'Unknown error during sandbox evaluation request';
      setEvaluationError(`Sandbox evaluation failed: ${message}`);
    } finally {
      setIsEvaluating(false);
    }
  }, [engine, evaluationEndpoint]);

  // Clear history
  const clearHistory = useCallback(() => {
    setEvaluationHistory([]);
    setEvaluationError(null);
    lastEvaluatedMoveRef.current = -1;
  }, []);

  // Reset on engine change
  useEffect(() => {
    clearHistory();
  }, [engine, clearHistory]);

  // Auto-evaluation when developer tools are enabled
  // When developer tools are enabled, automatically request a sandbox AI
  // evaluation after each new move so the EvaluationPanel can render a
  // lightweight sparkline over the turn history.
  useEffect(() => {
    if (!developerToolsEnabled || !engine || !gameState) {
      return;
    }

    // Skip when viewing historical states via replay/fixtures.
    if (isInReplayMode || isViewingHistory) {
      return;
    }

    const moveNumber = gameState.moveHistory?.length ?? 0;
    if (moveNumber <= 0) {
      return;
    }

    if (lastEvaluatedMoveRef.current === moveNumber) {
      return;
    }

    lastEvaluatedMoveRef.current = moveNumber;
    // Fire and forget; requestEvaluation manages its own loading state.
    requestEvaluation();
  }, [
    developerToolsEnabled,
    engine,
    gameState,
    isInReplayMode,
    isViewingHistory,
    requestEvaluation,
  ]);

  return {
    evaluationHistory,
    evaluationError,
    isEvaluating,
    requestEvaluation,
    clearHistory,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format evaluation score for display.
 */
export function formatEvaluationScore(score: number): string {
  if (score > 0) {
    return `+${score.toFixed(2)}`;
  }
  return score.toFixed(2);
}

/**
 * Get evaluation trend from history.
 */
export function getEvaluationTrend(
  history: EvaluationData[],
  playerNumber: number
): 'improving' | 'declining' | 'stable' | 'unknown' {
  if (history.length < 2) {
    return 'unknown';
  }

  // Get last few evaluations for the player
  const playerEvals = history
    .filter((e) => e.playerNumber === playerNumber)
    .slice(-3);

  if (playerEvals.length < 2) {
    return 'unknown';
  }

  const first = playerEvals[0].evaluation;
  const last = playerEvals[playerEvals.length - 1].evaluation;
  const diff = last - first;

  if (Math.abs(diff) < 0.1) {
    return 'stable';
  }

  return diff > 0 ? 'improving' : 'declining';
}

/**
 * Get key features from evaluation breakdown.
 */
export function getKeyFeatures(
  breakdown: Record<string, number>,
  limit: number = 5
): Array<{ name: string; value: number }> {
  return Object.entries(breakdown)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}
