/**
 * LocalGameStorage tests
 *
 * These tests install a lightweight in-memory IndexedDB stub so that
 * LocalGameStorage can be exercised in a jsdom environment without
 * external polyfills.
 */

import type { LocalGameRecord } from '../../../src/client/services/LocalGameStorage';

// ─────────────────────────────────────────────────────────────────────────────
// Minimal IndexedDB stub
// ─────────────────────────────────────────────────────────────────────────────

type AnyRequest = {
  result?: unknown;
  onsuccess: ((ev: { target: AnyRequest }) => void) | null;
  onerror: (() => void) | null;
};

const storeData = new Map<string, LocalGameRecord>();

function createRequest(initialResult?: unknown): AnyRequest {
  return {
    result: initialResult,
    onsuccess: null,
    onerror: null,
  };
}

// IDBKeyRange.only stub used by LocalGameStorage
(globalThis as any).IDBKeyRange = {
  only(value: unknown) {
    return { value };
  },
};

// Basic IndexedDB stub sufficient for LocalGameStorage use-cases.
(globalThis as any).indexedDB = {
  open(_name: string, _version: number) {
    const db = {
      objectStoreNames: {
        contains(storeName: string) {
          // We create the store lazily; from the LocalGameStorage perspective
          // this simply needs to be false on first open so createObjectStore
          // is invoked.
          return storeName === 'pendingGames' && storeData.size > 0;
        },
      },
      createObjectStore(_storeName: string, _options: { keyPath: string }) {
        return {
          // Index creation is a no-op in this stub; indices are inferred at
          // query time from the stored records.
          createIndex: () => {},
        };
      },
      transaction(_stores: string[], _mode: 'readonly' | 'readwrite') {
        const transaction: {
          objectStore: () => {
            add: (record: LocalGameRecord) => AnyRequest;
            get: (id: string) => AnyRequest;
            put: (record: LocalGameRecord) => AnyRequest;
            delete: (id: string) => AnyRequest;
            getAll: () => AnyRequest;
            openCursor: () => AnyRequest;
            index: (name: string) => {
              getAll: (range: { value: unknown }) => AnyRequest;
              count: (range: { value: unknown }) => AnyRequest;
              openCursor: (range: { value: unknown }) => AnyRequest;
            };
          };
          oncomplete: ((ev: unknown) => void) | null;
        } = {
          objectStore() {
            return {
              add(record: LocalGameRecord) {
                storeData.set(record.id, record);
                const req = createRequest();
                setTimeout(() => {
                  req.onsuccess?.({ target: req });
                }, 0);
                return req;
              },
              get(id: string) {
                const record = storeData.get(id);
                const req = createRequest(record);
                setTimeout(() => {
                  req.onsuccess?.({ target: req });
                }, 0);
                return req;
              },
              put(record: LocalGameRecord) {
                storeData.set(record.id, record);
                const req = createRequest();
                setTimeout(() => {
                  req.onsuccess?.({ target: req });
                }, 0);
                return req;
              },
              delete(id: string) {
                storeData.delete(id);
                const req = createRequest();
                setTimeout(() => {
                  req.onsuccess?.({ target: req });
                }, 0);
                return req;
              },
              getAll() {
                const req = createRequest();
                setTimeout(() => {
                  req.result = [...storeData.values()];
                  req.onsuccess?.({ target: req });
                }, 0);
                return req;
              },
              openCursor() {
                const req = createRequest();
                const allRecords = [...storeData.values()];

                setTimeout(() => {
                  let index = 0;
                  const advance = () => {
                    if (index >= allRecords.length) {
                      req.result = null;
                      req.onsuccess?.({ target: req });
                      transaction.oncomplete?.({});
                      return;
                    }
                    const record = allRecords[index];
                    const cursor = {
                      value: record,
                      delete() {
                        storeData.delete(record.id);
                      },
                      continue() {
                        index += 1;
                        advance();
                      },
                    };
                    req.result = cursor;
                    req.onsuccess?.({ target: req });
                  };

                  advance();
                }, 0);

                return req;
              },
              index(name: string) {
                return {
                  getAll(range: { value: unknown }) {
                    const req = createRequest();
                    setTimeout(() => {
                      let results: LocalGameRecord[] = [];
                      if (name === 'synced') {
                        results = [...storeData.values()].filter(
                          (r) => r.synced === (range.value as boolean)
                        );
                      } else if (name === 'createdAt') {
                        results = [...storeData.values()].filter(
                          (r) => r.createdAt >= (range.value as string)
                        );
                      }
                      req.result = results;
                      req.onsuccess?.({ target: req });
                    }, 0);
                    return req;
                  },
                  count(range: { value: unknown }) {
                    const req = createRequest();
                    setTimeout(() => {
                      const count = [...storeData.values()].filter(
                        (r) => r.synced === (range.value as boolean)
                      ).length;
                      req.result = count;
                      req.onsuccess?.({ target: req });
                    }, 0);
                    return req;
                  },
                  openCursor(range: { value: unknown }) {
                    const req = createRequest();
                    const matching = [...storeData.values()].filter(
                      (r) => r.synced === (range.value as boolean)
                    );

                    setTimeout(() => {
                      let index = 0;
                      const advance = () => {
                        if (index >= matching.length) {
                          // Signal completion with null cursor
                          req.result = null;
                          req.onsuccess?.({ target: req });
                          // Notify transaction completion so callers that
                          // await transaction.oncomplete (e.g. cleanup) can
                          // resolve their Promises.
                          transaction.oncomplete?.({});
                          return;
                        }
                        const record = matching[index];
                        const cursor = {
                          value: record,
                          delete() {
                            storeData.delete(record.id);
                          },
                          continue() {
                            index += 1;
                            advance();
                          },
                        };
                        req.result = cursor;
                        req.onsuccess?.({ target: req });
                      };

                      advance();
                    }, 0);

                    return req;
                  },
                };
              },
            };
          },
          oncomplete: null,
        };
        return transaction;
      },
      close() {
        // No-op for in-memory implementation
      },
    };

    const openRequest: AnyRequest & {
      onupgradeneeded: ((ev: { target: AnyRequest }) => void) | null;
    } = {
      ...createRequest(db),
      onupgradeneeded: null,
    };

    // Simulate upgrade + success in the next tick.
    setTimeout(() => {
      openRequest.onupgradeneeded?.({ target: openRequest });
      openRequest.onsuccess?.({ target: openRequest });
    }, 0);

    return openRequest;
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic import of LocalGameStorage after installing the stub
// ─────────────────────────────────────────────────────────────────────────────

let storeGameLocally: typeof import('../../../src/client/services/LocalGameStorage').storeGameLocally;
let getUnsyncedGames: typeof import('../../../src/client/services/LocalGameStorage').getUnsyncedGames;
let markGameSynced: typeof import('../../../src/client/services/LocalGameStorage').markGameSynced;
let deleteGame: typeof import('../../../src/client/services/LocalGameStorage').deleteGame;
let getPendingCount: typeof import('../../../src/client/services/LocalGameStorage').getPendingCount;
let cleanupSyncedGames: typeof import('../../../src/client/services/LocalGameStorage').cleanupSyncedGames;

beforeAll(async () => {
  const mod = await import('../../../src/client/services/LocalGameStorage');
  storeGameLocally = mod.storeGameLocally;
  getUnsyncedGames = mod.getUnsyncedGames;
  markGameSynced = mod.markGameSynced;
  deleteGame = mod.deleteGame;
  getPendingCount = mod.getPendingCount;
  cleanupSyncedGames = mod.cleanupSyncedGames;
});

beforeEach(() => {
  storeData.clear();
});

describe('LocalGameStorage with in-memory IndexedDB stub', () => {
  it('stores games locally and lists them as unsynced', async () => {
    const initialState: any = { id: 'game-1', moveHistory: [] };
    const finalState: any = { id: 'game-1', moveHistory: [] };
    const metadata: LocalGameRecord['metadata'] = {
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai'],
    };

    const result = await storeGameLocally(initialState, finalState, [], metadata);
    expect(result.success).toBe(true);
    expect(result.id).toMatch(/^local-/);

    const unsynced = await getUnsyncedGames();
    expect(unsynced).toHaveLength(1);
    expect(unsynced[0].id).toBe(result.id);

    const pendingCount = await getPendingCount();
    expect(pendingCount).toBe(1);
  });

  it('marks a game as synced and removes it from unsynced queries', async () => {
    const initialState: any = { id: 'game-2', moveHistory: [] };
    const finalState: any = { id: 'game-2', moveHistory: [] };
    const metadata: LocalGameRecord['metadata'] = {
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai'],
    };

    const { id } = await storeGameLocally(initialState, finalState, [], metadata);
    const marked = await markGameSynced(id);
    expect(marked).toBe(true);

    const unsynced = await getUnsyncedGames();
    expect(unsynced).toHaveLength(0);

    const pendingCount = await getPendingCount();
    expect(pendingCount).toBe(0);
  });

  it('deletes a game and returns true when deletion succeeds', async () => {
    const initialState: any = { id: 'game-3', moveHistory: [] };
    const finalState: any = { id: 'game-3', moveHistory: [] };
    const metadata: LocalGameRecord['metadata'] = {
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai'],
    };

    const { id } = await storeGameLocally(initialState, finalState, [], metadata);
    expect(storeData.has(id)).toBe(true);

    const deleted = await deleteGame(id);
    expect(deleted).toBe(true);
    expect(storeData.has(id)).toBe(false);
  });

  it('cleans up synced games older than the cutoff and returns deleted count', async () => {
    // Create two records and mark both as synced; adjust createdAt so that
    // only one falls before the cutoff date.
    const initialState: any = { id: 'game-4', moveHistory: [] };
    const finalState: any = { id: 'game-4', moveHistory: [] };
    const metadata: LocalGameRecord['metadata'] = {
      source: 'sandbox',
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai'],
    };

    const rec1 = await storeGameLocally(initialState, finalState, [], metadata);
    const rec2 = await storeGameLocally(initialState, finalState, [], metadata);

    // Mark both as synced.
    await markGameSynced(rec1.id);
    await markGameSynced(rec2.id);

    // Manually adjust createdAt so that rec1 is far in the past and rec2 is
    // in the future relative to the cutoff.
    const oldDate = new Date('2020-01-01T00:00:00.000Z').toISOString();
    const futureDate = new Date('2030-01-01T00:00:00.000Z').toISOString();

    const record1 = storeData.get(rec1.id);
    const record2 = storeData.get(rec2.id);
    if (!record1 || !record2) {
      throw new Error('Expected test records to exist in storeData');
    }
    record1.createdAt = oldDate;
    record2.createdAt = futureDate;
    storeData.set(rec1.id, record1);
    storeData.set(rec2.id, record2);

    // Use daysOld = 0 so the cutoff is "now"; only the old record should be
    // eligible for deletion.
    const deletedCount = await cleanupSyncedGames(0);
    expect(typeof deletedCount).toBe('number');
    expect(deletedCount).toBe(1);

    // The older record should have been removed; the newer should remain.
    expect(storeData.has(rec1.id)).toBe(false);
    expect(storeData.has(rec2.id)).toBe(true);
  });
});
