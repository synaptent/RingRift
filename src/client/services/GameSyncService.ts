/**
 * GameSyncService - Automatically syncs locally-stored games to server when available.
 *
 * Features:
 * - Periodic sync attempts (every 30 seconds when pending games exist)
 * - Syncs immediately when online event fires
 * - Exponential backoff on repeated failures
 * - Cleans up old synced games
 */

import {
  getUnsyncedGames,
  markGameSynced,
  deleteGame,
  getPendingCount,
  cleanupSyncedGames,
  type LocalGameRecord,
} from './LocalGameStorage';
import { getReplayService } from './ReplayService';

const SYNC_INTERVAL_MS = 30_000; // 30 seconds
const MAX_BACKOFF_MS = 300_000; // 5 minutes max
const CLEANUP_INTERVAL_MS = 3600_000; // 1 hour

export type SyncStatus = 'idle' | 'syncing' | 'error' | 'offline';

export interface SyncState {
  status: SyncStatus;
  pendingCount: number;
  lastSyncAttempt: Date | null;
  lastSuccessfulSync: Date | null;
  consecutiveFailures: number;
}

type SyncListener = (state: SyncState) => void;

class GameSyncServiceImpl {
  private state: SyncState = {
    status: 'idle',
    pendingCount: 0,
    lastSyncAttempt: null,
    lastSuccessfulSync: null,
    consecutiveFailures: 0,
  };

  private listeners: Set<SyncListener> = new Set();
  private syncIntervalId: ReturnType<typeof setInterval> | null = null;
  private cleanupIntervalId: ReturnType<typeof setInterval> | null = null;
  private isSyncing = false;

  /**
   * Start the sync service.
   */
  start(): void {
    // Initial pending count check
    this.updatePendingCount();

    // Listen for online/offline events
    window.addEventListener('online', this.handleOnline);
    window.addEventListener('offline', this.handleOffline);

    // Set initial online status
    if (!navigator.onLine) {
      this.updateState({ status: 'offline' });
    }

    // Start periodic sync
    this.syncIntervalId = setInterval(() => {
      this.attemptSync();
    }, SYNC_INTERVAL_MS);

    // Start periodic cleanup
    this.cleanupIntervalId = setInterval(() => {
      this.cleanupOldGames();
    }, CLEANUP_INTERVAL_MS);

    // Try initial sync
    this.attemptSync();
  }

  /**
   * Stop the sync service.
   */
  stop(): void {
    window.removeEventListener('online', this.handleOnline);
    window.removeEventListener('offline', this.handleOffline);

    if (this.syncIntervalId) {
      clearInterval(this.syncIntervalId);
      this.syncIntervalId = null;
    }

    if (this.cleanupIntervalId) {
      clearInterval(this.cleanupIntervalId);
      this.cleanupIntervalId = null;
    }
  }

  /**
   * Subscribe to state changes.
   */
  subscribe(listener: SyncListener): () => void {
    this.listeners.add(listener);
    // Immediately notify with current state
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  /**
   * Get current state.
   */
  getState(): SyncState {
    return { ...this.state };
  }

  /**
   * Manually trigger a sync attempt.
   */
  async triggerSync(): Promise<void> {
    await this.attemptSync();
  }

  private handleOnline = (): void => {
    // Network came back online - reset failure count and trigger sync
    this.updateState({ status: 'idle', consecutiveFailures: 0 });
    this.attemptSync();
  };

  private handleOffline = (): void => {
    // Network went offline - update status to reflect unavailability
    this.updateState({ status: 'offline' });
  };

  private async updatePendingCount(): Promise<void> {
    try {
      const count = await getPendingCount();
      this.updateState({ pendingCount: count });
    } catch (err) {
      console.error('[GameSyncService] Failed to get pending count:', err);
    }
  }

  private async attemptSync(): Promise<void> {
    // Skip if offline, already syncing, or no pending games
    if (!navigator.onLine || this.isSyncing) {
      return;
    }

    // Check if we should back off
    if (this.state.consecutiveFailures > 0) {
      const backoffMs = Math.min(
        SYNC_INTERVAL_MS * Math.pow(2, this.state.consecutiveFailures - 1),
        MAX_BACKOFF_MS
      );
      const timeSinceLastAttempt = this.state.lastSyncAttempt
        ? Date.now() - this.state.lastSyncAttempt.getTime()
        : Infinity;

      if (timeSinceLastAttempt < backoffMs) {
        return; // Still in backoff period
      }
    }

    // Get replay service instance
    const replayService = getReplayService();

    // Get pending games
    let pendingGames: LocalGameRecord[];
    try {
      pendingGames = await getUnsyncedGames();
    } catch (err) {
      console.error('[GameSyncService] Failed to get unsynced games:', err);
      return;
    }

    if (pendingGames.length === 0) {
      this.updateState({ pendingCount: 0, status: 'idle' });
      return;
    }

    this.isSyncing = true;
    this.updateState({ status: 'syncing', lastSyncAttempt: new Date() });

    let syncedCount = 0;
    let failedCount = 0;

    for (const game of pendingGames) {
      try {
        // Convert LocalGameRecord to StoreGameRequest for ReplayService
        const result = await replayService.storeGame({
          initialState: game.initialState,
          finalState: game.finalState,
          moves: game.moves as Record<string, unknown>[],
          metadata: {
            source: game.metadata.source,
            boardType: game.metadata.boardType,
            numPlayers: game.metadata.numPlayers,
            playerTypes: game.metadata.playerTypes,
            victoryReason: game.metadata.victoryReason,
            winnerPlayerNumber: game.metadata.winnerPlayerNumber,
          },
        });

        if (result.success) {
          // Mark as synced and optionally delete
          await markGameSynced(game.id);
          await deleteGame(game.id); // Delete after successful sync
          syncedCount++;
        } else {
          failedCount++;
        }
      } catch (err) {
        console.error('[GameSyncService] Failed to sync game:', game.id, err);
        failedCount++;
      }
    }

    this.isSyncing = false;

    // Update state based on results
    const newPendingCount = await getPendingCount().catch(() => pendingGames.length - syncedCount);

    if (failedCount === 0 && syncedCount > 0) {
      // All games synced successfully
      this.updateState({
        status: 'idle',
        pendingCount: newPendingCount,
        lastSuccessfulSync: new Date(),
        consecutiveFailures: 0,
      });
    } else if (syncedCount > 0) {
      // Partial success - some games synced, some failed
      console.warn(`[GameSyncService] Partial sync: ${syncedCount} synced, ${failedCount} failed`);
      this.updateState({
        status: 'idle',
        pendingCount: newPendingCount,
        lastSuccessfulSync: new Date(),
        consecutiveFailures: 0, // Reset on partial success
      });
    } else {
      console.warn(`[GameSyncService] Sync failed: ${failedCount} games could not be synced`);
      this.updateState({
        status: 'error',
        pendingCount: newPendingCount,
        consecutiveFailures: this.state.consecutiveFailures + 1,
      });
    }
  }

  private async cleanupOldGames(): Promise<void> {
    try {
      // Clean games synced more than 7 days ago to save local storage
      await cleanupSyncedGames(7);
    } catch (err) {
      console.error('[GameSyncService] Failed to cleanup old games:', err);
    }
  }

  private updateState(partial: Partial<SyncState>): void {
    this.state = { ...this.state, ...partial };
    this.notifyListeners();
  }

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.error('[GameSyncService] Listener error:', err);
      }
    }
  }
}

// Singleton instance
export const GameSyncService = new GameSyncServiceImpl();

/**
 * React hook for using the sync service.
 */
export function useGameSync(): SyncState & { triggerSync: () => Promise<void> } {
  // This would need React imports, so we'll provide a simple subscription-based API
  // The actual hook will be in the component
  return {
    ...GameSyncService.getState(),
    triggerSync: () => GameSyncService.triggerSync(),
  };
}
