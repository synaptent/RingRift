/**
 * @fileoverview useSandboxAILoop Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It manages UI state for AI turn execution, not rules logic.
 *
 * Canonical SSoT:
 * - AI turn logic: `src/client/sandbox/sandboxAI.ts`
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 *
 * This adapter:
 * - Runs AI turns in an async loop (with safety bounds)
 * - Triggers AI turn checks when human actions complete
 * - Schedules continuation batches for AI-vs-AI games
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useCallback, useRef } from 'react';
import type { Position } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';
import { FreezeDebugger } from '../utils/freezeDebugger';

export interface UseSandboxAILoopOptions {
  setSelected: (pos: Position | undefined) => void;
  setValidTargets: (cells: Position[]) => void;
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

  // RR-FIX-2026-01-11: Guard against concurrent AI loop executions.
  // Multiple calls to maybeRunSandboxAiIfNeeded can race and cause phase
  // desynchronization errors like "place_ring is not valid for phase movement".
  const aiLoopRunningRef = useRef(false);

  const bumpSandboxTurn = useCallback(() => {
    setSandboxTurn((t) => t + 1);
  }, []);

  const runSandboxAiTurnLoop = useCallback(async () => {
    const engine = sandboxEngine;
    if (!engine) return;

    // RR-FIX-2026-01-11: Prevent concurrent AI loop executions.
    // If a loop is already running, skip this call to avoid race conditions.
    if (aiLoopRunningRef.current) {
      return;
    }
    aiLoopRunningRef.current = true;

    try {
      let safetyCounter = 0;
      // Allow a bounded number of consecutive AI turns per batch to avoid
      // accidental infinite loops, but drive progression one visible move at a
      // time so AI-vs-AI games feel continuous rather than "bursty".
      while (safetyCounter < 32) {
        const state = engine.getGameState();
        if (state.gameStatus !== 'active') break;
        const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
        if (!current || current.type !== 'ai') break;

        // RR-FIX-2026-01-12: Yield to browser BEFORE heavy AI computation to prevent
        // browser freeze on large boards. The move enumeration can take 500ms-2s+ on
        // large boards (square19, hexagonal) with many stacks, causing the UI to become
        // completely unresponsive. By yielding here, the browser can process pending
        // events (scroll, clicks, DevTools) before the next heavy computation.
        await new Promise((resolve) => window.setTimeout(resolve, 0));

        // RR-FIX-2026-01-12: Capture state BEFORE AI turn for freeze debugging.
        // If the browser freezes, the last saved state is the problematic one.
        // Enable via: window.__FREEZE_DEBUGGER__.enable() in browser console.
        FreezeDebugger.beforeAITurn(state, safetyCounter);

        await engine.maybeRunAITurn();

        // Mark turn complete for freeze debugger watchdog
        FreezeDebugger.afterAITurn();

        // After each AI move, clear any stale selection/highlights and bump the
        // sandboxTurn counter so BoardView re-renders with the latest state.
        setSelected(undefined);
        setValidTargets([]);
        setSandboxTurn((t) => t + 1);
        setSandboxLastProgressAt(Date.now());
        setSandboxStallWarning(null);

        safetyCounter += 1;

        // RR-FIX-2026-01-12: Adaptive delay based on board size to prevent browser
        // freeze on large boards. Large boards (square19, hexagonal) have quadratic
        // complexity in move enumeration due to O(stacks × directions × board_size).
        // Use longer delays to keep the browser responsive during AI-vs-AI games.
        const boardCellCount = state.board.stacks.size + state.board.markers.size;
        const isLargeBoard = state.boardType === 'square19' || state.boardType === 'hexagonal';
        const hasManyPieces = boardCellCount > 30;
        const baseDelay = isLargeBoard || hasManyPieces ? 250 : 120;

        // Small delay between moves so AI-only games progress in a smooth
        // sequence rather than a single visual burst of many moves.
        await new Promise((resolve) => window.setTimeout(resolve, baseDelay));
      }

      // If the game is still active and the next player is an AI, schedule
      // another batch so AI-vs-AI games continue advancing without manual
      // clicks. The safety counter above still bounds each batch.
      const finalState = engine.getGameState();
      const next = finalState.players.find((p) => p.playerNumber === finalState.currentPlayer);
      if (finalState.gameStatus === 'active' && next && next.type === 'ai') {
        // Release the lock before scheduling next batch to allow re-entry
        aiLoopRunningRef.current = false;
        window.setTimeout(() => {
          void runSandboxAiTurnLoop();
        }, 200);
        return; // Don't release lock again in finally
      }
    } finally {
      aiLoopRunningRef.current = false;
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
