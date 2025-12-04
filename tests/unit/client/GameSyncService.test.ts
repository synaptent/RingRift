/**
 * GameSyncService tests
 *
 * Focused unit coverage for:
 * - Basic status transitions (idle/offline/error)
 * - Backoff behaviour on repeated failures
 * - Successful, partial, and failed sync attempts
 *
 * These tests stub LocalGameStorage and ReplayService so we can exercise the
 * sync state machine deterministically without touching IndexedDB or the
 * real network.
 */

import { GameSyncService } from '../../../src/client/services/GameSyncService';

// LocalGameStorage is stubbed to avoid IndexedDB usage in these tests.
jest.mock('../../../src/client/services/LocalGameStorage', () => ({
  getPendingCount: jest.fn(),
  getUnsyncedGames: jest.fn(),
  markGameSynced: jest.fn(),
  deleteGame: jest.fn(),
  cleanupSyncedGames: jest.fn(),
}));

// ReplayService is stubbed so we can control storeGame outcomes.
const mockStoreGame = jest.fn();
jest.mock('../../../src/client/services/ReplayService', () => ({
  __esModule: true,
  getReplayService: () => ({
    storeGame: mockStoreGame,
  }),
}));

type LocalGameStorageModule = typeof import('../../../src/client/services/LocalGameStorage');
const localGameStorage = jest.requireMock(
  '../../../src/client/services/LocalGameStorage'
) as jest.Mocked<LocalGameStorageModule>;

describe('GameSyncService', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Reset singleton internal state between tests where needed.
    const serviceAny = GameSyncService as any;
    serviceAny.state = {
      status: 'idle',
      pendingCount: 0,
      lastSyncAttempt: null,
      lastSuccessfulSync: null,
      consecutiveFailures: 0,
    };
    serviceAny.isSyncing = false;
  });

  afterEach(() => {
    GameSyncService.stop();
  });

  it('starts in idle status with zero pending games', () => {
    const state = GameSyncService.getState();
    expect(state.status).toBe('idle');
    expect(state.pendingCount).toBe(0);
  });

  it('marks service offline when navigator reports offline and start() is called', () => {
    const onlineSpy = jest.spyOn(window.navigator, 'onLine', 'get').mockReturnValue(false);

    GameSyncService.start();

    const state = GameSyncService.getState();
    expect(state.status).toBe('offline');

    onlineSpy.mockRestore();
  });

  it('does not attempt sync when offline', async () => {
    const onlineSpy = jest.spyOn(window.navigator, 'onLine', 'get').mockReturnValue(false);
    localGameStorage.getPendingCount.mockResolvedValue(3);

    const serviceAny = GameSyncService as any;
    await serviceAny.attemptSync();

    expect(localGameStorage.getUnsyncedGames).not.toHaveBeenCalled();
    const state = GameSyncService.getState();
    expect(state.status === 'offline' || state.status === 'idle').toBe(true);

    onlineSpy.mockRestore();
  });

  it('skips sync while in backoff window after previous failures', async () => {
    const onlineSpy = jest.spyOn(window.navigator, 'onLine', 'get').mockReturnValue(true);

    const serviceAny = GameSyncService as any;
    // Simulate a prior failed sync very recently.
    serviceAny.state = {
      status: 'error',
      pendingCount: 5,
      lastSyncAttempt: new Date(),
      lastSuccessfulSync: null,
      consecutiveFailures: 2,
    };

    await serviceAny.attemptSync();

    // Backoff logic should prevent a new fetch of unsynced games.
    expect(localGameStorage.getUnsyncedGames).not.toHaveBeenCalled();

    onlineSpy.mockRestore();
  });

  it('performs a no-op sync when there are no pending games', async () => {
    const onlineSpy = jest.spyOn(window.navigator, 'onLine', 'get').mockReturnValue(true);
    localGameStorage.getUnsyncedGames.mockResolvedValue([]);
    localGameStorage.getPendingCount.mockResolvedValue(0);

    const serviceAny = GameSyncService as any;
    await serviceAny.attemptSync();

    expect(localGameStorage.getUnsyncedGames).toHaveBeenCalledTimes(1);
    expect(mockStoreGame).not.toHaveBeenCalled();

    const state = GameSyncService.getState();
    expect(state.status).toBe('idle');
    expect(state.pendingCount).toBe(0);

    onlineSpy.mockRestore();
  });

  it('marks games as synced and clears them on successful sync', async () => {
    const pendingGames = [
      {
        id: 'local-1',
        initialState: {} as any,
        finalState: { moveHistory: [] } as any,
        moves: [],
        metadata: {
          source: 'sandbox',
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
          victoryReason: 'elimination',
          winnerPlayerNumber: 1,
        },
      },
      {
        id: 'local-2',
        initialState: {} as any,
        finalState: { moveHistory: [] } as any,
        moves: [],
        metadata: {
          source: 'sandbox',
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
          victoryReason: 'timeout',
          winnerPlayerNumber: 2,
        },
      },
    ];

    localGameStorage.getUnsyncedGames.mockResolvedValue(pendingGames as any);
    // After syncing both, there should be no remaining pending games.
    localGameStorage.getPendingCount.mockResolvedValue(0);
    mockStoreGame.mockResolvedValue({ success: true, totalMoves: 10 });

    const serviceAny = GameSyncService as any;
    await serviceAny.attemptSync();

    expect(localGameStorage.getUnsyncedGames).toHaveBeenCalledTimes(1);
    expect(mockStoreGame).toHaveBeenCalledTimes(2);

    // Each successfully synced game should be marked synced and deleted.
    expect(localGameStorage.markGameSynced).toHaveBeenCalledWith('local-1');
    expect(localGameStorage.markGameSynced).toHaveBeenCalledWith('local-2');
    expect(localGameStorage.deleteGame).toHaveBeenCalledWith('local-1');
    expect(localGameStorage.deleteGame).toHaveBeenCalledWith('local-2');

    const state = GameSyncService.getState();
    expect(state.status).toBe('idle');
    expect(state.pendingCount).toBe(0);
    expect(state.consecutiveFailures).toBe(0);
    expect(state.lastSuccessfulSync).not.toBeNull();
  });

  it('treats partial success as success and resets failure backoff', async () => {
    const pendingGames = [
      {
        id: 'local-ok',
        initialState: {} as any,
        finalState: { moveHistory: [] } as any,
        moves: [],
        metadata: {
          source: 'sandbox',
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
        },
      },
      {
        id: 'local-fail',
        initialState: {} as any,
        finalState: { moveHistory: [] } as any,
        moves: [],
        metadata: {
          source: 'sandbox',
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
        },
      },
    ];

    localGameStorage.getUnsyncedGames.mockResolvedValue(pendingGames as any);
    // After partial success we still expect at least one pending game.
    localGameStorage.getPendingCount.mockResolvedValue(1);

    mockStoreGame
      .mockResolvedValueOnce({ success: true, totalMoves: 5 })
      .mockResolvedValueOnce({ success: false, totalMoves: 0 });

    const serviceAny = GameSyncService as any;
    // Seed prior failure state to ensure backoff is reset on partial success.
    serviceAny.state.consecutiveFailures = 2;

    await serviceAny.attemptSync();

    expect(mockStoreGame).toHaveBeenCalledTimes(2);
    expect(localGameStorage.markGameSynced).toHaveBeenCalledWith('local-ok');
    expect(localGameStorage.deleteGame).toHaveBeenCalledWith('local-ok');

    const state = GameSyncService.getState();
    expect(state.status).toBe('idle');
    expect(state.pendingCount).toBe(1);
    expect(state.consecutiveFailures).toBe(0);
    expect(state.lastSuccessfulSync).not.toBeNull();
  });

  it('marks status as error and increments consecutiveFailures when all syncs fail', async () => {
    const pendingGames = [
      {
        id: 'local-err-1',
        initialState: {} as any,
        finalState: { moveHistory: [] } as any,
        moves: [],
        metadata: { source: 'sandbox', boardType: 'square8', numPlayers: 2, playerTypes: [] },
      },
    ];

    localGameStorage.getUnsyncedGames.mockResolvedValue(pendingGames as any);
    localGameStorage.getPendingCount.mockResolvedValue(1);

    // Simulate a hard failure from ReplayService.
    mockStoreGame.mockRejectedValue(new Error('Service unavailable'));

    const serviceAny = GameSyncService as any;
    expect(GameSyncService.getState().consecutiveFailures).toBe(0);

    await serviceAny.attemptSync();

    expect(mockStoreGame).toHaveBeenCalledTimes(1);
    expect(localGameStorage.markGameSynced).not.toHaveBeenCalled();
    expect(localGameStorage.deleteGame).not.toHaveBeenCalled();

    const state = GameSyncService.getState();
    expect(state.status).toBe('error');
    expect(state.pendingCount).toBe(1);
    expect(state.consecutiveFailures).toBe(1);
  });
});
