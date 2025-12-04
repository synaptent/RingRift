/**
 * LocalGameStorage - IndexedDB fallback for game recording when AI service is unavailable.
 *
 * Stores completed games locally in the browser, with the ability to sync
 * to the server when the AI service becomes available again.
 */

import type { GameState } from '../../shared/types/game';

const DB_NAME = 'RingRiftGameStorage';
const DB_VERSION = 1;
const STORE_NAME = 'pendingGames';

export interface LocalGameRecord {
  id: string;
  initialState: GameState;
  finalState: GameState;
  moves: unknown[];
  metadata: {
    source: string;
    boardType: string;
    numPlayers: number;
    playerTypes: string[];
    victoryReason?: string;
    winnerPlayerNumber?: number | null;
  };
  createdAt: string;
  synced: boolean;
}

/**
 * Open the IndexedDB database.
 */
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error('Failed to open IndexedDB'));
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Create the pending games store
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('synced', 'synced', { unique: false });
        store.createIndex('createdAt', 'createdAt', { unique: false });
      }
    };
  });
}

/**
 * Generate a unique ID for a local game record.
 */
function generateId(): string {
  return `local-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Store a game locally in IndexedDB.
 */
export async function storeGameLocally(
  initialState: GameState,
  finalState: GameState,
  moves: unknown[],
  metadata: LocalGameRecord['metadata']
): Promise<{ success: boolean; id: string }> {
  try {
    const db = await openDatabase();
    const record: LocalGameRecord = {
      id: generateId(),
      initialState,
      finalState,
      moves,
      metadata,
      createdAt: new Date().toISOString(),
      synced: false,
    };

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.add(record);

      request.onsuccess = () => {
        resolve({ success: true, id: record.id });
      };

      request.onerror = () => {
        reject(new Error('Failed to store game locally'));
      };

      transaction.oncomplete = () => {
        db.close();
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to store game:', error);
    return { success: false, id: '' };
  }
}

/**
 * Get all unsynced games from local storage.
 */
export async function getUnsyncedGames(): Promise<LocalGameRecord[]> {
  try {
    const db = await openDatabase();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const allRecords = request.result as LocalGameRecord[];
        resolve(allRecords.filter((record) => record.synced === false));
      };

      request.onerror = () => {
        reject(new Error('Failed to get unsynced games'));
      };

      transaction.oncomplete = () => {
        db.close();
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to get unsynced games:', error);
    return [];
  }
}

/**
 * Mark a game as synced (after successfully uploading to server).
 */
export async function markGameSynced(id: string): Promise<boolean> {
  try {
    const db = await openDatabase();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const getRequest = store.get(id);

      getRequest.onsuccess = () => {
        const record = getRequest.result as LocalGameRecord | undefined;
        if (record) {
          record.synced = true;
          const putRequest = store.put(record);
          putRequest.onsuccess = () => resolve(true);
          putRequest.onerror = () => reject(new Error('Failed to mark game synced'));
        } else {
          resolve(false);
        }
      };

      getRequest.onerror = () => {
        reject(new Error('Failed to get game for syncing'));
      };

      transaction.oncomplete = () => {
        db.close();
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to mark game synced:', error);
    return false;
  }
}

/**
 * Delete a synced game from local storage.
 */
export async function deleteGame(id: string): Promise<boolean> {
  try {
    const db = await openDatabase();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onsuccess = () => {
        resolve(true);
      };

      request.onerror = () => {
        reject(new Error('Failed to delete game'));
      };

      transaction.oncomplete = () => {
        db.close();
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to delete game:', error);
    return false;
  }
}

/**
 * Get the count of pending (unsynced) games.
 */
export async function getPendingCount(): Promise<number> {
  try {
    const db = await openDatabase();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const allRecords = request.result as LocalGameRecord[];
        const pendingCount = allRecords.filter((record) => record.synced === false).length;
        resolve(pendingCount);
      };

      request.onerror = () => {
        reject(new Error('Failed to count pending games'));
      };

      transaction.oncomplete = () => {
        db.close();
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to count pending games:', error);
    return 0;
  }
}

/**
 * Clear all synced games older than the specified number of days.
 */
export async function cleanupSyncedGames(daysOld = 7): Promise<number> {
  try {
    const db = await openDatabase();
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysOld);
    const cutoffIso = cutoffDate.toISOString();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.openCursor();

      let deletedCount = 0;

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
        if (cursor) {
          const record = cursor.value as LocalGameRecord;
          if (record.synced === true && record.createdAt < cutoffIso) {
            cursor.delete();
            deletedCount++;
          }
          cursor.continue();
        }
      };

      request.onerror = () => {
        reject(new Error('Failed to cleanup synced games'));
      };

      transaction.oncomplete = () => {
        db.close();
        resolve(deletedCount);
      };
    });
  } catch (error) {
    console.error('[LocalGameStorage] Failed to cleanup synced games:', error);
    return 0;
  }
}
