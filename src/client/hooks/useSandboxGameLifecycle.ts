/**
 * @fileoverview useSandboxGameLifecycle Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for sandbox game lifecycle.
 * It manages game start, reset, rematch, and preset operations.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 *
 * This adapter:
 * - Starting games (backend + local fallback)
 * - Quick-start presets
 * - Resetting to setup screen
 * - Rematch with same configuration
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import type { BoardType, CreateGameRequest, Position, PlayerChoice } from '../../shared/types/game';
import type { LocalConfig, LocalPlayerType } from '../contexts/SandboxContext';
import type {
  ClientSandboxEngine,
  SandboxInteractionHandler,
} from '../sandbox/ClientSandboxEngine';
import { gameApi } from '../services/api';

/**
 * Quick-start preset configuration for sandbox games.
 */
export interface QuickStartPreset {
  id: string;
  label: string;
  description: string;
  learnMoreText?: string;
  icon: string;
  badge?: string;
  config: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
  };
}

/**
 * Dependencies required by the useSandboxGameLifecycle hook.
 */
export interface GameLifecycleDeps {
  /** Current sandbox configuration */
  config: LocalConfig;
  /** Update sandbox configuration */
  setConfig: React.Dispatch<React.SetStateAction<LocalConfig>>;
  /** Currently authenticated user (null if not logged in) */
  user: { id: string } | null;
  /** Initialize a local sandbox engine with the given options */
  initLocalSandboxEngine: (options: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
    aiDifficulties: number[];
    interactionHandler: SandboxInteractionHandler;
  }) => ClientSandboxEngine;
  /** Reset the sandbox engine to unconfigured state */
  resetSandboxEngine: () => void;
  /** Factory to create interaction handlers for a given set of player types */
  createSandboxInteractionHandler: (playerTypes: LocalPlayerType[]) => SandboxInteractionHandler;
  /** Trigger AI turn execution if needed */
  maybeRunSandboxAiIfNeeded: () => void;
  /** Clear scenario context (for non-scenario games) */
  clearScenarioContext: () => void;
  /** Reset game UI state (selection, pending choices, etc.) */
  resetGameUIState: () => void;
  /** Mark onboarding welcome as seen */
  markWelcomeSeen: () => void;
  /** Set backend sandbox error message */
  setBackendSandboxError: (error: string | null) => void;
  /** Set selected cell position */
  setSelected: (pos: Position | undefined) => void;
  /** Set valid target cells */
  setValidTargets: (targets: Position[]) => void;
  /** Set pending player choice */
  setSandboxPendingChoice: (choice: PlayerChoice | null) => void;
  /** Set capture choice */
  setSandboxCaptureChoice: (choice: PlayerChoice | null) => void;
  /** Set capture targets */
  setSandboxCaptureTargets: (targets: Position[]) => void;
  /** Set stall warning */
  setSandboxStallWarning: (warning: string | null) => void;
  /** Set last progress timestamp */
  setSandboxLastProgressAt: (timestamp: number | null) => void;
  /** Bump state version counter */
  setSandboxStateVersion: React.Dispatch<React.SetStateAction<number>>;
  /** Set last loaded scenario */
  setLastLoadedScenario: (scenario: unknown) => void;
  /** Set victory modal dismissed state */
  setIsSandboxVictoryModalDismissed: (dismissed: boolean) => void;
}

/**
 * Actions for managing game lifecycle.
 */
export interface GameLifecycleActions {
  /**
   * Start a new game with the given configuration.
   * Attempts to create a backend game first, falls back to local-only sandbox.
   */
  startGame: (config: LocalConfig) => Promise<void>;

  /**
   * Start a game using only the local sandbox engine (no backend).
   */
  startLocalGame: (config: LocalConfig) => void;

  /**
   * Apply a quick-start preset and immediately start a game.
   */
  applyQuickStartPreset: (preset: QuickStartPreset) => void;

  /**
   * Reset the sandbox back to the setup/configuration screen.
   */
  resetToSetup: () => void;

  /**
   * Start a rematch with the same configuration.
   */
  rematch: () => void;
}

/**
 * Return type for useSandboxGameLifecycle hook.
 */
export interface UseSandboxGameLifecycleReturn {
  /** Available game lifecycle actions */
  actions: GameLifecycleActions;
}

/**
 * Custom hook for managing sandbox game lifecycle operations.
 *
 * Extracts game start, reset, rematch, and preset handlers from SandboxGameHost
 * to reduce component complexity and improve testability.
 *
 * @param deps - Dependencies required for lifecycle operations
 * @returns Game lifecycle actions
 *
 * @example
 * ```tsx
 * const { actions } = useSandboxGameLifecycle({
 *   config,
 *   setConfig,
 *   user,
 *   initLocalSandboxEngine,
 *   resetSandboxEngine,
 *   // ... other deps
 * });
 *
 * // Start a game
 * await actions.startGame(config);
 *
 * // Apply a preset
 * actions.applyQuickStartPreset(learnBasicsPreset);
 *
 * // Reset to setup screen
 * actions.resetToSetup();
 *
 * // Start a rematch
 * actions.rematch();
 * ```
 */
export function useSandboxGameLifecycle(deps: GameLifecycleDeps): UseSandboxGameLifecycleReturn {
  const navigate = useNavigate();

  const {
    config,
    setConfig,
    user,
    initLocalSandboxEngine,
    resetSandboxEngine,
    createSandboxInteractionHandler,
    maybeRunSandboxAiIfNeeded,
    clearScenarioContext,
    resetGameUIState,
    markWelcomeSeen,
    setBackendSandboxError,
    setSelected,
    setValidTargets,
    setSandboxPendingChoice,
    setSandboxCaptureChoice,
    setSandboxCaptureTargets,
    setSandboxStallWarning,
    setSandboxLastProgressAt,
    setSandboxStateVersion: _setSandboxStateVersion,
    setLastLoadedScenario,
    setIsSandboxVictoryModalDismissed,
  } = deps;

  /**
   * Start a game using only the local sandbox engine (no backend attempt).
   */
  const startLocalGame = useCallback(
    (snapshot: LocalConfig) => {
      const interactionHandler = createSandboxInteractionHandler(
        snapshot.playerTypes.slice(0, snapshot.numPlayers)
      );
      const engine = initLocalSandboxEngine({
        boardType: snapshot.boardType,
        numPlayers: snapshot.numPlayers,
        playerTypes: snapshot.playerTypes.slice(0, snapshot.numPlayers) as LocalPlayerType[],
        aiDifficulties: snapshot.aiDifficulties.slice(0, snapshot.numPlayers),
        interactionHandler,
      });

      setSelected(undefined);
      setValidTargets([]);
      setSandboxPendingChoice(null);

      // If the first player is an AI, immediately start the sandbox AI turn loop.
      if (engine) {
        const state = engine.getGameState();
        const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
        if (current && current.type === 'ai') {
          maybeRunSandboxAiIfNeeded();
        }
      }
    },
    [
      createSandboxInteractionHandler,
      initLocalSandboxEngine,
      maybeRunSandboxAiIfNeeded,
      setSelected,
      setValidTargets,
      setSandboxPendingChoice,
    ]
  );

  /**
   * Start a new game with the given configuration.
   * First attempts to create a backend game, falls back to local-only sandbox on failure.
   */
  const startGame = useCallback(
    async (snapshot: LocalConfig) => {
      // Starting a non-scenario sandbox game; clear any prior scenario context so
      // scenario-specific telemetry does not attribute future victories here.
      setLastLoadedScenario(null);

      // When not authenticated, skip backend game creation entirely and go
      // straight to the local sandbox engine to avoid expected 401 noise.
      if (!user) {
        startLocalGame(snapshot);
        return;
      }

      // First, attempt to create a real backend game using the same CreateGameRequest
      // shape as the lobby. On success, navigate into the real backend game route.
      try {
        const payload: CreateGameRequest = {
          boardType: snapshot.boardType,
          maxPlayers: snapshot.numPlayers,
          isRated: false,
          isPrivate: true,
          timeControl: {
            type: 'rapid',
            initialTime: 600,
            increment: 0,
          },
          aiOpponents: (() => {
            const clampDifficulty = (value: number) => Math.max(1, Math.min(10, Math.round(value)));
            const seatTypes = snapshot.playerTypes.slice(0, snapshot.numPlayers);
            const aiDifficulties = seatTypes
              .map((t, idx) =>
                t === 'ai' ? clampDifficulty(snapshot.aiDifficulties[idx] ?? 5) : null
              )
              .filter((d): d is number => d !== null);
            const aiSeats = aiDifficulties.length;
            if (aiSeats <= 0) return undefined;
            return {
              count: aiSeats,
              difficulty: aiDifficulties,
              mode: 'service',
              aiType: 'heuristic',
            };
          })(),
          // Pie rule (swap sides) is opt-in for 2-player games.
          // Data shows P2 wins >55% with pie rule enabled by default.
          rulesOptions: snapshot.numPlayers === 2 ? { swapRuleEnabled: false } : undefined,
        };

        const game = await gameApi.createGame(payload);
        navigate(`/game/${game.id}`);
        return;
      } catch (err) {
        console.error(
          'Failed to create backend sandbox game, falling back to local-only board',
          err
        );
        setBackendSandboxError(
          'Backend sandbox game could not be created; falling back to local-only board only.'
        );
      }

      startLocalGame(snapshot);
    },
    [user, startLocalGame, setLastLoadedScenario, setBackendSandboxError, navigate]
  );

  /**
   * Apply a quick-start preset and immediately start a game.
   */
  const applyQuickStartPreset = useCallback(
    (preset: QuickStartPreset) => {
      if (preset.id === 'learn-basics') {
        markWelcomeSeen();
      }
      clearScenarioContext();
      resetGameUIState();

      // Build an explicit snapshot so we can both update config and launch a
      // game immediately without relying on async state updates.
      const baseTypes = [...config.playerTypes];
      const updatedTypes = baseTypes.map((t, idx) =>
        idx < preset.config.playerTypes.length ? preset.config.playerTypes[idx] : t
      );

      const snapshot: LocalConfig = {
        boardType: preset.config.boardType,
        numPlayers: preset.config.numPlayers,
        playerTypes: updatedTypes,
        aiDifficulties: [...config.aiDifficulties],
      };

      setConfig(snapshot);
      void startGame(snapshot);
    },
    [
      config.playerTypes,
      config.aiDifficulties,
      setConfig,
      markWelcomeSeen,
      clearScenarioContext,
      resetGameUIState,
      startGame,
    ]
  );

  /**
   * Reset the sandbox back to the setup/configuration screen.
   */
  const resetToSetup = useCallback(() => {
    resetSandboxEngine();
    setSelected(undefined);
    setValidTargets([]);
    setBackendSandboxError(null);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setLastLoadedScenario(null);
  }, [
    resetSandboxEngine,
    setSelected,
    setValidTargets,
    setBackendSandboxError,
    setSandboxPendingChoice,
    setIsSandboxVictoryModalDismissed,
    setLastLoadedScenario,
  ]);

  /**
   * Start a rematch with the same configuration.
   */
  const rematch = useCallback(() => {
    // Reset state and start a new game with the same configuration.
    // Rematches are treated as generic sandbox games rather than
    // curated teaching scenarios, so we clear any scenario context.
    setIsSandboxVictoryModalDismissed(false);
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setLastLoadedScenario(null);

    // Re-initialize with the same config
    const interactionHandler = createSandboxInteractionHandler(
      config.playerTypes.slice(0, config.numPlayers)
    );
    const engine = initLocalSandboxEngine({
      boardType: config.boardType,
      numPlayers: config.numPlayers,
      playerTypes: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
      aiDifficulties: config.aiDifficulties.slice(0, config.numPlayers),
      interactionHandler,
    });

    // If the first player is AI, start the AI turn loop
    if (engine) {
      const state = engine.getGameState();
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (current && current.type === 'ai') {
        maybeRunSandboxAiIfNeeded();
      }
    }

    toast.success('New game started with the same settings!');
  }, [
    config,
    createSandboxInteractionHandler,
    initLocalSandboxEngine,
    maybeRunSandboxAiIfNeeded,
    setIsSandboxVictoryModalDismissed,
    setSelected,
    setValidTargets,
    setSandboxPendingChoice,
    setSandboxCaptureChoice,
    setSandboxCaptureTargets,
    setSandboxStallWarning,
    setSandboxLastProgressAt,
    setLastLoadedScenario,
  ]);

  return {
    actions: {
      startGame,
      startLocalGame,
      applyQuickStartPreset,
      resetToSetup,
      rematch,
    },
  };
}

export default useSandboxGameLifecycle;
