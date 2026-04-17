/**
 * @fileoverview useSandboxPersistence Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for sandbox persistence.
 * It manages game save/sync functionality, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Replay service: `src/client/services/ReplayService.ts`
 *
 * This adapter:
 * - Captures initial game state for storage
 * - Auto-saves completed games to server or local storage
 * - Syncs local games to server when online
 * - Tracks save status for UI feedback
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 * @module hooks/useSandboxPersistence
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';
import type { GameState } from '../../shared/types/game';
import type { ClientSandboxEngine } from '../sandbox/ClientSandboxEngine';
import { getReplayService } from '../services/ReplayService';
import { storeGameLocally, getPendingCount } from '../services/LocalGameStorage';
import { GameSyncService, type SyncState } from '../services/GameSyncService';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Save status for UI feedback.
 */
export type GameSaveStatus = 'idle' | 'saving' | 'saved' | 'saved-local' | 'error';

/**
 * Re-export SyncState for consumers.
 */
export type { SyncState };

/**
 * Player type configuration for sandbox games.
 */
export type LocalPlayerType = 'human' | 'ai';

/**
 * Options for the persistence hook.
 */
export interface SandboxPersistenceOptions {
  /** Sandbox engine instance */
  engine: ClientSandboxEngine | null;
  /** Player types configuration */
  playerTypes: LocalPlayerType[];
  /** Number of players */
  numPlayers: number;
  /** Whether auto-save is enabled by default */
  defaultAutoSave?: boolean;
  /** Sandbox state version (to trigger re-evaluation when state changes) */
  stateVersion?: number;
  /** Whether autosave should write to the replay service instead of local-only storage. */
  serverPersistenceEnabled?: boolean;
}

/**
 * Return type for useSandboxPersistence.
 */
export interface SandboxPersistenceState {
  /** Whether auto-save is enabled */
  autoSaveGames: boolean;
  /** Toggle auto-save */
  setAutoSaveGames: (enabled: boolean) => void;
  /** Current save status */
  gameSaveStatus: GameSaveStatus;
  /** Number of pending local games */
  pendingLocalGames: number;
  /** Sync service state */
  syncState: SyncState | null;
  /** Initial game state ref (for external access if needed) */
  initialGameStateRef: React.RefObject<GameState | null>;
  /** Game saved ref (for external access if needed) */
  gameSavedRef: React.RefObject<boolean>;
  /** Clone initial game state utility */
  cloneInitialGameState: (state: GameState) => GameState;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing sandbox game persistence and sync.
 */
export function useSandboxPersistence(options: SandboxPersistenceOptions): SandboxPersistenceState {
  const {
    engine,
    playerTypes,
    numPlayers,
    defaultAutoSave = true,
    stateVersion = 0,
    serverPersistenceEnabled = true,
  } = options;

  // Derive victory result from engine
  const victoryResult = engine?.getVictoryResult() ?? null;

  // State
  const [autoSaveGames, setAutoSaveGames] = useState(defaultAutoSave);
  const [gameSaveStatus, setGameSaveStatus] = useState<GameSaveStatus>('idle');
  const [pendingLocalGames, setPendingLocalGames] = useState(0);
  const [syncState, setSyncState] = useState<SyncState | null>(null);

  // Refs for tracking saves
  const initialGameStateRef = useRef<GameState | null>(null);
  const gameSavedRef = useRef(false);

  // Clone initial game state utility - Safari-safe deep clone
  const cloneInitialGameState = useCallback((state: GameState): GameState => {
    // structuredClone is available in modern browsers but may not be in older Safari
    if (typeof structuredClone === 'function') {
      return structuredClone(state) as GameState;
    }
    return JSON.parse(JSON.stringify(state)) as GameState;
  }, []);

  // Capture initial game state when engine is created for game storage
  useEffect(() => {
    if (!engine) {
      // Reset refs when engine is destroyed
      initialGameStateRef.current = null;
      gameSavedRef.current = false;
      setGameSaveStatus('idle');
      return;
    }
    // Capture initial state only once per game (when moveHistory is empty)
    const currentState = engine.getGameState();
    if (currentState.moveHistory.length === 0 && !initialGameStateRef.current) {
      initialGameStateRef.current = cloneInitialGameState(currentState);
      gameSavedRef.current = false;
      setGameSaveStatus('idle');
    }
  }, [engine, cloneInitialGameState]);

  // Start game sync service and subscribe to state updates
  useEffect(() => {
    if (!serverPersistenceEnabled) {
      setSyncState(null);
      setPendingLocalGames(0);
      return;
    }

    GameSyncService.start();
    const unsubscribe = GameSyncService.subscribe((state) => {
      setSyncState(state);
      setPendingLocalGames(state.pendingCount);
    });
    return () => {
      unsubscribe();
      GameSyncService.stop();
    };
  }, []);

  // Auto-save completed games to replay database when victory is detected
  useEffect(() => {
    if (!autoSaveGames || !victoryResult || gameSavedRef.current) {
      return;
    }

    const saveCompletedGame = async () => {
      const finalState = engine?.getGameState();
      const initialState = initialGameStateRef.current;

      if (!finalState || !initialState) {
        console.warn('[useSandboxPersistence] Cannot save game: missing state');
        return;
      }

      const hasHuman = playerTypes.slice(0, numPlayers).includes('human');
      const hasAI = playerTypes.slice(0, numPlayers).includes('ai');
      const metadata = {
        source: hasHuman && hasAI ? 'human_vs_ai' : 'sandbox',
        submissionIntent: 'autosave',
        gameId: finalState.id,
        boardType: finalState.board.type,
        numPlayers: finalState.players.length,
        playerTypes: playerTypes.slice(0, numPlayers),
        players: finalState.players.map((player) => ({
          playerNumber: player.playerNumber,
          playerType: player.type,
          aiType: player.aiProfile?.aiType,
          aiDifficulty: player.aiProfile?.difficulty ?? player.aiDifficulty,
        })),
        victoryReason: victoryResult.reason,
        winnerPlayerNumber: victoryResult.winner,
      };

      const saveLocally = async (localOnly: boolean): Promise<void> => {
        // Fallback to IndexedDB local storage
        try {
          const localResult = await storeGameLocally(
            initialState,
            finalState,
            finalState.moveHistory as unknown[],
            !localOnly
              ? metadata
              : {
                  ...metadata,
                  localOnly: true,
                  skipServerSync: true,
                  syncPolicy: 'local_only',
                }
          );

          if (localResult.success) {
            gameSavedRef.current = true;
            setGameSaveStatus('saved-local');
            const newCount = await getPendingCount();
            setPendingLocalGames(newCount);
            toast.success(
              !localOnly
                ? 'Game saved locally (will sync when server available)'
                : 'Game saved locally',
              {
                icon: '💾',
              }
            );
          } else {
            setGameSaveStatus('error');
            toast.error('Failed to save game');
          }
        } catch (localError) {
          console.error('[useSandboxPersistence] Local storage also failed:', localError);
          setGameSaveStatus('error');
          toast.error('Failed to save game (storage unavailable)');
        }
      };

      setGameSaveStatus('saving');

      if (!serverPersistenceEnabled) {
        await saveLocally(true);
        return;
      }

      try {
        const replayService = getReplayService();
        const result = await replayService.storeGame({
          gameId: finalState.id,
          initialState,
          finalState,
          moves: finalState.moveHistory as unknown as Record<string, unknown>[],
          metadata,
        });

        if (result.success) {
          gameSavedRef.current = true;
          setGameSaveStatus('saved');
          toast.success(`Game saved (${result.totalMoves ?? 0} moves)`);
        } else {
          // Server not configured or unavailable - use local fallback
          // This is expected in production when RINGRIFT_AI_SERVICE_URL is not set
          throw new Error(result.message || 'Server storage unavailable');
        }
      } catch (error) {
        // Note: This is expected in production when AI service URL is not configured.
        // Games are saved to IndexedDB and can sync later if server becomes available.
        console.debug(
          '[useSandboxPersistence] Server storage unavailable, using local storage:',
          error
        );

        await saveLocally(false);
      }
    };

    saveCompletedGame();
  }, [
    autoSaveGames,
    victoryResult,
    engine,
    playerTypes,
    numPlayers,
    stateVersion,
    serverPersistenceEnabled,
  ]);

  return {
    autoSaveGames,
    setAutoSaveGames,
    gameSaveStatus,
    pendingLocalGames,
    syncState,
    initialGameStateRef,
    gameSavedRef,
    cloneInitialGameState,
  };
}
