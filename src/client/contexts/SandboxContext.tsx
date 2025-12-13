import React, { createContext, useContext, useEffect, useState } from 'react';
import type { BoardType, GameState, PlayerChoice, Position } from '../../shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../sandbox/ClientSandboxEngine';
import { isSandboxAiStallDiagnosticsEnabled } from '../../shared/utils/envFlags';

export type LocalPlayerType = 'human' | 'ai';

/** Default AI difficulty level for sandbox (D4 = Intermediate) */
export const DEFAULT_AI_DIFFICULTY = 4;

export interface LocalConfig {
  numPlayers: number;
  boardType: BoardType;
  playerTypes: LocalPlayerType[]; // indexed 0..3 for players 1..4
  /** AI difficulty levels per player (1-10), indexed 0..3 for players 1..4 */
  aiDifficulties: number[];
}

interface SandboxContextValue {
  config: LocalConfig;
  setConfig: React.Dispatch<React.SetStateAction<LocalConfig>>;
  isConfigured: boolean;
  setIsConfigured: (value: boolean) => void;
  backendSandboxError: string | null;
  setBackendSandboxError: (value: string | null) => void;
  sandboxEngine: ClientSandboxEngine | null;
  sandboxPendingChoice: PlayerChoice | null;
  setSandboxPendingChoice: React.Dispatch<React.SetStateAction<PlayerChoice | null>>;
  sandboxCaptureChoice: PlayerChoice | null;
  setSandboxCaptureChoice: React.Dispatch<React.SetStateAction<PlayerChoice | null>>;
  sandboxCaptureTargets: Position[];
  setSandboxCaptureTargets: React.Dispatch<React.SetStateAction<Position[]>>;
  sandboxLastProgressAt: number | null;
  setSandboxLastProgressAt: React.Dispatch<React.SetStateAction<number | null>>;
  sandboxStallWarning: string | null;
  setSandboxStallWarning: React.Dispatch<React.SetStateAction<string | null>>;
  sandboxStateVersion: number;
  setSandboxStateVersion: React.Dispatch<React.SetStateAction<number>>;
  sandboxDiagnosticsEnabled: boolean;
  developerToolsEnabled: boolean;
  setDeveloperToolsEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  /**
   * Create or replace the client-local sandbox engine using the provided
   * config + interaction handler. This owns the lifecycle of the engine
   * instance but does not perform any navigation or UI updates.
   */
  initLocalSandboxEngine: (options: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
    interactionHandler: SandboxInteractionHandler;
  }) => ClientSandboxEngine;
  /**
   * Convenience accessor for the current sandbox GameState, or null when no
   * engine is active.
   */
  getSandboxGameState: () => GameState | null;
  /**
   * Reset the sandbox engine and configuration flags back to an
   * unconfigured state. Does not modify UI-local selection state.
   */
  resetSandboxEngine: () => void;
}

const SandboxContext = createContext<SandboxContextValue | undefined>(undefined);

export function SandboxProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<LocalConfig>({
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai'],
    aiDifficulties: [
      DEFAULT_AI_DIFFICULTY,
      DEFAULT_AI_DIFFICULTY,
      DEFAULT_AI_DIFFICULTY,
      DEFAULT_AI_DIFFICULTY,
    ],
  });
  const [isConfigured, setIsConfigured] = useState(false);
  const [backendSandboxError, setBackendSandboxError] = useState<string | null>(null);
  const [sandboxEngine, setSandboxEngine] = useState<ClientSandboxEngine | null>(null);
  const [sandboxPendingChoice, setSandboxPendingChoice] = useState<PlayerChoice | null>(null);
  const [sandboxCaptureChoice, setSandboxCaptureChoice] = useState<PlayerChoice | null>(null);
  const [sandboxCaptureTargets, setSandboxCaptureTargets] = useState<Position[]>([]);
  const [sandboxLastProgressAt, setSandboxLastProgressAt] = useState<number | null>(null);
  const [sandboxStallWarning, setSandboxStallWarning] = useState<string | null>(null);
  const [sandboxStateVersion, setSandboxStateVersion] = useState(0);
  const [developerToolsEnabled, setDeveloperToolsEnabled] = useState(true);
  const sandboxDiagnosticsEnabled = isSandboxAiStallDiagnosticsEnabled();

  const initLocalSandboxEngine = (options: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
    aiDifficulties?: number[];
    interactionHandler: SandboxInteractionHandler;
  }): ClientSandboxEngine => {
    const sandboxConfig: SandboxConfig = {
      boardType: options.boardType,
      numPlayers: options.numPlayers,
      playerKinds: options.playerTypes,
      aiDifficulties: options.aiDifficulties,
    };

    const engine = new ClientSandboxEngine({
      config: sandboxConfig,
      interactionHandler: options.interactionHandler,
    });

    setSandboxEngine(engine);
    setIsConfigured(true);
    return engine;
  };

  const getSandboxGameState = (): GameState | null => {
    if (!sandboxEngine) {
      return null;
    }
    return sandboxEngine.getGameState();
  };

  const resetSandboxEngine = (): void => {
    setSandboxEngine(null);
    setIsConfigured(false);
  };

  // Local sandbox AI-only stall watchdog. This runs independently of the
  // internal sandbox AI diagnostics and focuses on scheduler-level stalls
  // (situations where an AI player is to move but the local game state has
  // not advanced for an extended period).
  useEffect(() => {
    if (!isConfigured) {
      return;
    }

    const STALL_TIMEOUT_MS = 8000;
    const POLL_INTERVAL_MS = 1000;

    const id = window.setInterval(() => {
      setSandboxStallWarning((prevWarning) => {
        const last = sandboxLastProgressAt;
        if (last === null) {
          return prevWarning;
        }

        const engine = sandboxEngine;
        if (!engine) {
          return null;
        }

        const state = engine.getGameState();
        const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
        const now = Date.now();

        if (state.gameStatus !== 'active' || !current || current.type !== 'ai') {
          // Clear any previous warning when there is no active AI turn pending.
          return null;
        }

        if (now - last > STALL_TIMEOUT_MS) {
          return (
            prevWarning ??
            'Potential AI stall detected: sandbox AI has not advanced the game state for several seconds while an AI player is to move.'
          );
        }

        // Below threshold but still in AI turn: preserve any existing warning
        // (it may have been set by the diagnostics watcher or a previous poll).
        return prevWarning;
      });
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(id);
    };
  }, [isConfigured, sandboxLastProgressAt, sandboxEngine]);

  // Structural AI stall diagnostics watcher: when enabled via the
  // RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS flag, poll the sandbox AI
  // trace buffer and surface any "stall" entries as a UI banner so that
  // AI-vs-AI stalls are visible and debuggable from the /sandbox route.
  useEffect(() => {
    if (!sandboxDiagnosticsEnabled) {
      return;
    }

    if (typeof window === 'undefined') {
      return;
    }

    const POLL_INTERVAL_MS = 1000;
    // Type-safe access to debug globals on window
    const windowWithDebug = window as Window & { __RINGRIFT_SANDBOX_TRACE__?: unknown[] };
    let lastSeenStallTimestamp = 0;

    const id = window.setInterval(() => {
      const trace = windowWithDebug.__RINGRIFT_SANDBOX_TRACE__ ?? [];
      if (!Array.isArray(trace) || trace.length === 0) {
        return;
      }

      const latestStall = [...trace].reverse().find((entry) => {
        const entryRecord = entry as unknown as Record<string, unknown> | null | undefined;
        return entryRecord && entryRecord.kind === 'stall';
      });
      if (!latestStall) {
        return;
      }

      const stall = latestStall as unknown as Record<string, unknown>;
      const ts = typeof stall.timestamp === 'number' ? stall.timestamp : Date.now();
      if (ts <= lastSeenStallTimestamp) {
        return;
      }

      lastSeenStallTimestamp = ts;

      setSandboxStallWarning(
        (prev) =>
          prev ??
          'Sandbox AI stall detected by diagnostics: consecutive AI turns are not changing the game state. Use "Copy AI trace" for detailed debugging.'
      );
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(id);
    };
  }, [sandboxDiagnosticsEnabled]);

  // Test-only hook: in non-production builds, expose a minimal E2E helper on
  // window so that Playwright tests can seed a sandbox stall warning and AI
  // trace without relying on timing-sensitive AI behaviour. This helper is
  // not attached in production bundles.
  useEffect(() => {
    if (process.env.NODE_ENV === 'production') {
      return;
    }

    if (typeof window === 'undefined') {
      return;
    }

    // Type-safe interface for E2E test globals. We only declare the E2E setter
    // here since __RINGRIFT_SANDBOX_TRACE__ is already declared globally in
    // sandboxAI.ts via `declare global { interface Window { ... } }`.
    type E2ESetterFn = (message: string, trace: unknown) => void;
    interface WindowWithE2EDebug extends Window {
      __RINGRIFT_E2E_SET_SANDBOX_STALL__?: E2ESetterFn;
    }

    const windowWithE2E = window as WindowWithE2EDebug;
    const setter: E2ESetterFn = (message: string, trace: unknown) => {
      setSandboxStallWarning(message);
      // Delegate to the sandbox AI trace buffer without over-constraining types
      // here. sandboxAI.ts owns the precise SandboxAITurnTraceEntry[] shape.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- test-only E2E helper
      (window as any).__RINGRIFT_SANDBOX_TRACE__ = trace;
    };

    windowWithE2E.__RINGRIFT_E2E_SET_SANDBOX_STALL__ = setter;

    return () => {
      if (windowWithE2E.__RINGRIFT_E2E_SET_SANDBOX_STALL__ === setter) {
        delete windowWithE2E.__RINGRIFT_E2E_SET_SANDBOX_STALL__;
      }
    };
  }, []);

  const value: SandboxContextValue = {
    config,
    setConfig,
    isConfigured,
    setIsConfigured,
    backendSandboxError,
    setBackendSandboxError,
    sandboxEngine,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    sandboxCaptureTargets,
    setSandboxCaptureTargets,
    sandboxLastProgressAt,
    setSandboxLastProgressAt,
    sandboxStallWarning,
    setSandboxStallWarning,
    sandboxStateVersion,
    setSandboxStateVersion,
    sandboxDiagnosticsEnabled,
    developerToolsEnabled,
    setDeveloperToolsEnabled,
    initLocalSandboxEngine,
    getSandboxGameState,
    resetSandboxEngine,
  };

  return <SandboxContext.Provider value={value}>{children}</SandboxContext.Provider>;
}

export function useSandbox(): SandboxContextValue {
  const ctx = useContext(SandboxContext);
  if (!ctx) {
    throw new Error('useSandbox must be used within a SandboxProvider');
  }
  return ctx;
}
