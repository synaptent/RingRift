/**
 * @fileoverview useSandboxDiagnostics Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for sandbox diagnostics UI.
 * It manages export and debug functionality, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Diagnostics: `src/client/sandbox/sandboxAiDiagnostics.ts`
 *
 * This adapter:
 * - Save state dialog visibility
 * - Copy test fixture to clipboard
 * - Export scenario JSON
 * - Copy AI trace/metadata for debugging
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import type { GameState } from '../../shared/types/game';
import { buildTestFixtureFromGameState, exportGameStateToFile } from '../sandbox/statePersistence';
import { getSandboxAiDiagnostics } from '../sandbox/sandboxAiDiagnostics';

/**
 * State managed by the diagnostics hook.
 */
export interface SandboxDiagnosticsState {
  /** Whether the save state dialog is visible */
  showSaveStateDialog: boolean;
}

/**
 * Actions provided by the diagnostics hook.
 */
export interface SandboxDiagnosticsActions {
  /** Open or close the save state dialog */
  setShowSaveStateDialog: (show: boolean) => void;
  /** Copy current game state as a test fixture to clipboard */
  copyTestFixture: () => Promise<void>;
  /** Export current game state as a downloadable scenario JSON file */
  exportScenarioJson: () => void;
  /** Copy sandbox AI trace logs to clipboard (for debugging) */
  copyAiTrace: () => Promise<void>;
  /** Copy sandbox AI metadata/diagnostics to clipboard */
  copyAiMeta: () => Promise<void>;
}

/**
 * Return type for useSandboxDiagnostics hook.
 */
export interface UseSandboxDiagnosticsReturn {
  /** Current diagnostics state */
  state: SandboxDiagnosticsState;
  /** Available diagnostics actions */
  actions: SandboxDiagnosticsActions;
}

/**
 * Custom hook for managing sandbox diagnostics, export, and debug functionality.
 *
 * Extracts diagnostics-related state and handlers from SandboxGameHost to reduce
 * component complexity and improve testability.
 *
 * @param gameState - The current game state (or null if no game active)
 * @returns Diagnostics state and action functions
 *
 * @example
 * ```tsx
 * const { state, actions } = useSandboxDiagnostics(sandboxGameState);
 *
 * // In JSX:
 * <button onClick={actions.copyTestFixture}>Copy Test Fixture</button>
 * <button onClick={() => actions.setShowSaveStateDialog(true)}>Save State</button>
 *
 * <SaveStateDialog
 *   isOpen={state.showSaveStateDialog}
 *   onClose={() => actions.setShowSaveStateDialog(false)}
 *   gameState={gameState}
 * />
 * ```
 */
export function useSandboxDiagnostics(gameState: GameState | null): UseSandboxDiagnosticsReturn {
  // Save state dialog visibility
  const [showSaveStateDialog, setShowSaveStateDialog] = useState(false);

  /**
   * Copy the sandbox AI trace to clipboard for debugging.
   * The trace is captured in a global window variable by the sandbox engine.
   */
  const copyAiTrace = useCallback(async () => {
    try {
      if (typeof window === 'undefined') {
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing debug global
      const anyWindow = window as any;
      const trace = anyWindow.__RINGRIFT_SANDBOX_TRACE__ ?? [];
      const payload = JSON.stringify(trace, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox AI trace copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox AI trace', trace);
        toast.success('Sandbox AI trace logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox AI trace', err);
      toast.error('Failed to export sandbox AI trace; see console for details.');
    }
  }, []);

  /**
   * Copy sandbox AI metadata/diagnostics to clipboard.
   * Includes information about AI type, difficulty, neural network usage, etc.
   */
  const copyAiMeta = useCallback(async () => {
    try {
      const meta = getSandboxAiDiagnostics();
      const payload = JSON.stringify(meta, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox AI metadata copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox AI metadata', meta);
        toast.success('Sandbox AI metadata logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox AI metadata', err);
      toast.error('Failed to export sandbox AI metadata; see console for details.');
    }
  }, []);

  /**
   * Copy the current game state as a test fixture to clipboard.
   * Useful for creating reproducible test cases.
   */
  const copyTestFixture = useCallback(async () => {
    try {
      if (!gameState) {
        toast.error('No sandbox game is currently active.');
        return;
      }

      const fixture = buildTestFixtureFromGameState(gameState);
      const payload = JSON.stringify(fixture, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox test fixture copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox test fixture', fixture);
        toast.success('Sandbox test fixture logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox test fixture', err);
      toast.error('Failed to export sandbox test fixture; see console for details.');
    }
  }, [gameState]);

  /**
   * Export the current game state as a downloadable scenario JSON file.
   * Creates a file named based on the current turn number.
   */
  const exportScenarioJson = useCallback(() => {
    try {
      if (!gameState) {
        toast.error('No sandbox game is currently active.');
        return;
      }

      const turnLabel = gameState.moveHistory.length + 1;
      const name = `Sandbox Scenario - Turn ${turnLabel}`;
      exportGameStateToFile(gameState, name);
      toast.success('Sandbox scenario JSON downloaded');
    } catch (err) {
      console.error('Failed to export sandbox scenario JSON', err);
      toast.error('Failed to export sandbox scenario; see console for details.');
    }
  }, [gameState]);

  return {
    state: {
      showSaveStateDialog,
    },
    actions: {
      setShowSaveStateDialog,
      copyTestFixture,
      exportScenarioJson,
      copyAiTrace,
      copyAiMeta,
    },
  };
}

export default useSandboxDiagnostics;
