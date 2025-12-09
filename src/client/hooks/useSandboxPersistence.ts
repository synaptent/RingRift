/**
 * useSandboxPersistence - Game persistence and sync logic for sandbox mode
 *
 * This hook extracts all game auto-save and sync functionality that was
 * previously embedded in SandboxGameHost. It handles:
 * - Capturing initial game state for storage
 * - Auto-saving completed games to server or local storage
 * - Syncing local games to server when online
 * - Tracking save status for UI feedback
 *
 * @module hooks/useSandboxPersistence
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';
import type { GameState } from '../../shared/types/game';
import type { ClientSandboxEngine } from '../services/ClientSandboxEngine';
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
  const { engine, playerTypes, numPlayers, defaultAutoSave = true, stateVersion = 0 } = options;

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

      const metadata = {
        source: 'sandbox',
        boardType: finalState.board.type,
        numPlayers: finalState.players.length,
        playerTypes: playerTypes.slice(0, numPlayers),
        victoryReason: victoryResult.reason,
        winnerPlayerNumber: victoryResult.winner,
      };

      try {
        setGameSaveStatus('saving');
        const replayService = getReplayService();
        const result = await replayService.storeGame({
          initialState,
          finalState,
          moves: finalState.moveHistory as unknown as Record<string, unknown>[],
          metadata,
        });

        if (result.success) {
          gameSavedRef.current = true;
          setGameSaveStatus('saved');
          toast.success(`Game saved (${result.totalMoves} moves)`);
        } else {
          // Server rejected - try local fallback
          throw new Error('Server rejected game storage');
        }
      } catch (error) {
        console.warn('[useSandboxPersistence] Server save failed, trying local storage:', error);

        // Fallback to IndexedDB local storage
        try {
          const localResult = await storeGameLocally(
            initialState,
            finalState,
            finalState.moveHistory as unknown[],
            metadata
          );

          if (localResult.success) {
            gameSavedRef.current = true;
            setGameSaveStatus('saved-local');
            const newCount = await getPendingCount();
            setPendingLocalGames(newCount);
            toast.success('Game saved locally (will sync when server available)', {
              icon: '💾',
            });
          } else {
            setGameSaveStatus('error');
            toast.error('Failed to save game');
          }
        } catch (localError) {
          console.error('[useSandboxPersistence] Local storage also failed:', localError);
          setGameSaveStatus('error');
          toast.error('Failed to save game (storage unavailable)');
        }
      }
    };

    saveCompletedGame();
  }, [autoSaveGames, victoryResult, engine, playerTypes, numPlayers, stateVersion]);

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
