/**
 * @fileoverview useSandboxAITracking Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It manages UI state for AI timing, diagnostics, and auto-trigger behavior.
 *
 * Canonical SSoT:
 * - AI turn logic: `src/client/sandbox/sandboxAI.ts`
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 *
 * This adapter:
 * - Tracks when AI starts/stops thinking for progress display
 * - Auto-triggers AI turns when game state indicates an AI player is active
 * - Manages AI ladder health diagnostics for debugging
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useEffect, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import type { GameState } from '../../shared/types/game';
import type { ClientSandboxEngine } from '../sandbox/ClientSandboxEngine';

/**
 * State for AI timing tracking.
 */
export interface AITrackingState {
  /** Timestamp when the current AI started thinking (null if not AI turn) */
  aiThinkingStartedAt: number | null;
  /** Cached AI ladder health data from AI service */
  aiLadderHealth: Record<string, unknown> | null;
  /** Error message from ladder health fetch */
  aiLadderHealthError: string | null;
  /** Whether ladder health is currently loading */
  aiLadderHealthLoading: boolean;
}

/**
 * Actions for AI tracking.
 */
export interface AITrackingActions {
  /** Manually set AI thinking start timestamp */
  setAiThinkingStartedAt: (timestamp: number | null) => void;
  /** Fetch AI ladder health from the service */
  refreshLadderHealth: () => Promise<void>;
  /** Copy AI ladder health to clipboard */
  copyLadderHealth: () => Promise<void>;
}

/**
 * Return type for useSandboxAITracking hook.
 */
export interface UseSandboxAITrackingReturn {
  /** Current AI tracking state */
  state: AITrackingState;
  /** Available AI tracking actions */
  actions: AITrackingActions;
}

/**
 * Custom hook for managing AI timing, diagnostics, and auto-trigger behavior in sandbox mode.
 *
 * Extracts AI-related state and effects from SandboxGameHost to reduce component complexity.
 *
 * Features:
 * - Tracks when AI starts/stops thinking for progress display
 * - Auto-triggers AI turns when the game state indicates an AI player is active
 * - Manages AI ladder health diagnostics for debugging
 *
 * @param engine - The sandbox engine instance (or null if not initialized)
 * @param gameState - The current game state (or null if no game active)
 * @param maybeRunSandboxAiIfNeeded - Callback to trigger AI turn execution
 * @returns AI tracking state and action functions
 *
 * @example
 * ```tsx
 * const { state, actions } = useSandboxAITracking(
 *   sandboxEngine,
 *   sandboxGameState,
 *   maybeRunSandboxAiIfNeeded
 * );
 *
 * // In JSX:
 * <AIThinkTimeProgress
 *   isAiThinking={state.aiThinkingStartedAt !== null}
 *   thinkingStartedAt={state.aiThinkingStartedAt}
 * />
 * ```
 */
export function useSandboxAITracking(
  engine: ClientSandboxEngine | null,
  gameState: GameState | null,
  maybeRunSandboxAiIfNeeded: () => void
): UseSandboxAITrackingReturn {
  // AI thinking state: timestamp when AI started thinking on current turn
  const [aiThinkingStartedAt, setAiThinkingStartedAt] = useState<number | null>(null);

  // AI ladder health (AI service internal) – loaded on-demand from the devtools panel
  const [aiLadderHealth, setAiLadderHealth] = useState<Record<string, unknown> | null>(null);
  const [aiLadderHealthError, setAiLadderHealthError] = useState<string | null>(null);
  const [aiLadderHealthLoading, setAiLadderHealthLoading] = useState(false);

  /**
   * Track AI thinking state for progress bar.
   * Sets start time when AI turn begins, clears when turn ends or game is not active.
   */
  useEffect(() => {
    if (!engine || !gameState) {
      setAiThinkingStartedAt(null);
      return;
    }

    const current = gameState.players.find((p) => p.playerNumber === gameState.currentPlayer);

    const isAiTurn = gameState.gameStatus === 'active' && current && current.type === 'ai';

    if (isAiTurn) {
      // Only set start time if not already set (avoid resetting mid-turn)
      setAiThinkingStartedAt((prev) => prev ?? Date.now());
    } else {
      setAiThinkingStartedAt(null);
    }
  }, [engine, gameState]);

  /**
   * Auto-trigger sandbox AI loop when the sandbox state reflects an active AI turn.
   * This keeps AI progression in sync with orchestrator-driven state changes
   * (including line/territory processing and elimination decisions) without
   * requiring an extra board click from the user.
   */
  useEffect(() => {
    if (!engine || !gameState) {
      return;
    }

    const current = gameState.players.find((p) => p.playerNumber === gameState.currentPlayer);

    if (gameState.gameStatus !== 'active' || !current || current.type !== 'ai') {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      maybeRunSandboxAiIfNeeded();
    }, 60);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [engine, gameState, maybeRunSandboxAiIfNeeded]);

  /**
   * Fetch AI ladder health from the AI service.
   * Requires an active sandbox game with a valid boardType and player count.
   */
  const refreshLadderHealth = useCallback(async () => {
    if (!gameState) {
      toast.error('No sandbox game is currently active.');
      return;
    }
    if (typeof fetch !== 'function') {
      toast.error('Fetch API unavailable.');
      return;
    }

    setAiLadderHealthLoading(true);
    setAiLadderHealthError(null);
    try {
      const params = new URLSearchParams({
        boardType: gameState.boardType,
        numPlayers: String(gameState.players.length),
      });
      const response = await fetch(`/api/games/sandbox/ai/ladder/health?${params.toString()}`);
      if (!response.ok) {
        const details = await response.text().catch(() => '');
        throw new Error(details || `HTTP ${response.status}`);
      }

      const data = (await response.json()) as Record<string, unknown>;
      setAiLadderHealth(data);
      toast.success('AI ladder health loaded');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load AI ladder health';
      setAiLadderHealthError(message);
      toast.error('Failed to load AI ladder health; see console for details.');
      console.error('Failed to load AI ladder health', err);
    } finally {
      setAiLadderHealthLoading(false);
    }
  }, [gameState]);

  /**
   * Copy AI ladder health to clipboard.
   */
  const copyLadderHealth = useCallback(async () => {
    if (!aiLadderHealth) {
      toast.error('AI ladder health has not been loaded yet.');
      return;
    }

    try {
      const payload = JSON.stringify(aiLadderHealth, null, 2);
      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('AI ladder health copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('AI ladder health', aiLadderHealth);
        toast.success('AI ladder health logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to copy AI ladder health', err);
      toast.error('Failed to copy AI ladder health; see console for details.');
    }
  }, [aiLadderHealth]);

  return {
    state: {
      aiThinkingStartedAt,
      aiLadderHealth,
      aiLadderHealthError,
      aiLadderHealthLoading,
    },
    actions: {
      setAiThinkingStartedAt,
      refreshLadderHealth,
      copyLadderHealth,
    },
  };
}

export default useSandboxAITracking;
