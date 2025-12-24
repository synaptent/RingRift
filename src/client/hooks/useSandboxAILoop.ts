/**
 * useSandboxAILoop Hook
 *
 * Manages AI turn loop execution in sandbox mode. Extracted from
 * useSandboxInteractions to reduce complexity.
 *
 * Responsibilities:
 * - Running AI turns in an async loop (with safety bounds)
 * - Triggering AI turn checks when human actions complete
 * - Scheduling continuation batches for AI-vs-AI games
 */

import { useState, useCallback } from 'react';
import type { Position } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';

export interface UseSandboxAILoopOptions {
  setSelected: React.Dispatch<React.SetStateAction<Position | undefined>>;
  setValidTargets: React.Dispatch<React.SetStateAction<Position[]>>;
}

export interface UseSandboxAILoopReturn {
  /** Trigger a re-render for AI-vs-AI games */
  bumpSandboxTurn: () => void;
  /** Check if AI should run and start the loop if needed */
  maybeRunSandboxAiIfNeeded: () => void;
  /** Run the AI turn loop (internal, exposed for testing) */
  runSandboxAiTurnLoop: () => Promise<void>;
}

/**
 * Hook for managing AI turn execution in sandbox mode.
 *
 * Drives AI-vs-AI game progression with bounded batches and
 * inter-move delays for visual continuity.
 */
export function useSandboxAILoop({
  setSelected,
  setValidTargets,
}: UseSandboxAILoopOptions): UseSandboxAILoopReturn {
  const { sandboxEngine, setSandboxLastProgressAt, setSandboxStallWarning } = useSandbox();

  // Local render tick used to force re-renders for AI-vs-AI games even when
  // React state derived from GameState hasn't otherwise changed.
  const [, setSandboxTurn] = useState(0);

  const bumpSandboxTurn = useCallback(() => {
    setSandboxTurn((t) => t + 1);
  }, []);

  const runSandboxAiTurnLoop = useCallback(async () => {
    const engine = sandboxEngine;
    if (!engine) return;

    let safetyCounter = 0;
    // Allow a bounded number of consecutive AI turns per batch to avoid
    // accidental infinite loops, but drive progression one visible move at a
    // time so AI-vs-AI games feel continuous rather than "bursty".
    while (safetyCounter < 32) {
      const state = engine.getGameState();
      if (state.gameStatus !== 'active') break;
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!current || current.type !== 'ai') break;

      await engine.maybeRunAITurn();

      // After each AI move, clear any stale selection/highlights and bump the
      // sandboxTurn counter so BoardView re-renders with the latest state.
      setSelected(undefined);
      setValidTargets([]);
      setSandboxTurn((t) => t + 1);
      setSandboxLastProgressAt(Date.now());
      setSandboxStallWarning(null);

      safetyCounter += 1;

      // Small delay between moves so AI-only games progress in a smooth
      // sequence rather than a single visual burst of many moves.
      await new Promise((resolve) => window.setTimeout(resolve, 120));
    }

    // If the game is still active and the next player is an AI, schedule
    // another batch so AI-vs-AI games continue advancing without manual
    // clicks. The safety counter above still bounds each batch.
    const finalState = engine.getGameState();
    const next = finalState.players.find((p) => p.playerNumber === finalState.currentPlayer);
    if (finalState.gameStatus === 'active' && next && next.type === 'ai') {
      window.setTimeout(() => {
        void runSandboxAiTurnLoop();
      }, 200);
    }
  }, [
    sandboxEngine,
    setSelected,
    setValidTargets,
    setSandboxLastProgressAt,
    setSandboxStallWarning,
  ]);

  const maybeRunSandboxAiIfNeeded = useCallback(() => {
    const engine = sandboxEngine;
    if (!engine) return;

    const state = engine.getGameState();
    const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (state.gameStatus === 'active' && current && current.type === 'ai') {
      void runSandboxAiTurnLoop();
    }
  }, [sandboxEngine, runSandboxAiTurnLoop]);

  return {
    bumpSandboxTurn,
    maybeRunSandboxAiIfNeeded,
    runSandboxAiTurnLoop,
  };
}

export default useSandboxAILoop;
