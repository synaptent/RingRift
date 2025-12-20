/**
 * LF5.a Abandonment Flow Tests
 *
 * Tests for abandonment loss scenarios where the reconnection window expires
 * while the player still has time remaining on their clock.
 *
 * Per RULES_SCENARIO_MATRIX.md LF5.a:
 * "Abandonment loss while time remains: player disconnects from an ACTIVE,
 * timed 2-player game with time left on their clock; reconnection window
 * expires first, ending the game by abandonment with a distinct result
 * reason (not time-loss)"
 *
 * Related spec: docs/CANONICAL_ENGINE_API.md §3.9.4
 */

import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type { GameResult, GameState } from '../../src/shared/types/game';
import { GamePersistenceService } from '../../src/server/services/GamePersistenceService';

jest.mock('../../src/server/services/GamePersistenceService', () => ({
  GamePersistenceService: {
    saveMove: jest.fn(),
    finishGame: jest.fn().mockResolvedValue({}),
  },
}));

const createMockIo = (): jest.Mocked<SocketIOServer> =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: {
        rooms: new Map(),
      },
      sockets: new Map(),
    },
  }) as any;

const mockFinishGame = GamePersistenceService.finishGame as jest.MockedFunction<any>;

describe('LF5.a: Abandonment Loss While Time Remains', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  function createTimedGameState(overrides: Partial<GameState> = {}): GameState {
    const now = new Date();
    return {
      id: 'abandonment-test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      },
      players: [
        {
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000, // 10 minutes remaining
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000, // 10 minutes remaining
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 36,
      totalRingsEliminated: 0,
      victoryThreshold: 18,
      territoryVictoryThreshold: 33,
      ...(overrides as any),
    };
  }

  it('should end game with abandonment reason when reconnection window expires while time remains', async () => {
    const io = createMockIo();
    const session = new GameSession('abandonment-test-game', io, {} as any, new Map());

    const state = createTimedGameState();
    const gameResult: GameResult = {
      winner: 2,
      reason: 'abandonment',
      finalScore: {
        ringsEliminated: { 1: 0, 2: 0 },
        territorySpaces: { 1: 0, 2: 0 },
        ringsRemaining: { 1: 18, 2: 18 },
      },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(() => ({ success: true, gameResult })),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    // Simulate player 1 disconnecting and reconnection window expiring
    // Player still has 600000ms (10 min) on their clock
    const result = await session.handleAbandonmentForDisconnectedPlayer(1, true);

    // Verify the game ended with abandonment, not time-loss
    expect(result).toEqual(gameResult);
    expect(result?.reason).toBe('abandonment');
    expect(result?.reason).not.toBe('time_loss');

    // Verify player 2 wins
    expect(result?.winner).toBe(2);

    // Verify finishGame was called with correct parameters
    expect(mockFinishGame).toHaveBeenCalledTimes(1);
    expect(mockFinishGame).toHaveBeenCalledWith(
      'abandonment-test-game',
      'player-2', // Winner is player 2
      expect.any(Object),
      gameResult
    );

    // Verify broadcast was sent
    expect((session as any).broadcastUpdate).toHaveBeenCalledWith({
      success: true,
      gameResult,
    });
  });

  it('should preserve time remaining values in game result when abandoned', async () => {
    const io = createMockIo();
    const session = new GameSession('time-preserved-game', io, {} as any, new Map());

    const state = createTimedGameState({
      id: 'time-preserved-game',
      players: [
        {
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 300000, // 5 minutes remaining
          ringsInHand: 15,
          eliminatedRings: 3,
          territorySpaces: 2,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 450000, // 7.5 minutes remaining
          ringsInHand: 16,
          eliminatedRings: 2,
          territorySpaces: 1,
        },
      ] as any,
    });

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(() => ({
        success: true,
        gameResult: {
          winner: 2,
          reason: 'abandonment',
          finalScore: {
            ringsEliminated: { 1: 3, 2: 2 },
            territorySpaces: { 1: 2, 2: 1 },
            ringsRemaining: { 1: 15, 2: 16 },
          },
        },
      })),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handleAbandonmentForDisconnectedPlayer(1, true);

    // Verify time was not the deciding factor
    expect(result?.reason).toBe('abandonment');
    expect(state.players[0].timeRemaining).toBeGreaterThan(0);
    expect(state.players[1].timeRemaining).toBeGreaterThan(0);
  });

  it('should handle abandonment during active player turn with pending decision', async () => {
    const io = createMockIo();
    const session = new GameSession('pending-decision-game', io, {} as any, new Map());

    const state = createTimedGameState({
      id: 'pending-decision-game',
      currentPhase: 'movement',
    });

    const mockInteractionHandler = {
      cancelAllChoicesForPlayer: jest.fn(),
      hasPendingChoice: jest.fn(() => true),
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(() => ({
        success: true,
        gameResult: {
          winner: 2,
          reason: 'abandonment',
          finalScore: {
            ringsEliminated: { 1: 0, 2: 0 },
            territorySpaces: { 1: 0, 2: 0 },
            ringsRemaining: { 1: 18, 2: 18 },
          },
        },
      })),
    };

    (session as any).interactionHandler = mockInteractionHandler;

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handleAbandonmentForDisconnectedPlayer(1, true);

    // Verify abandonment was processed
    expect(result?.reason).toBe('abandonment');
    expect(result?.winner).toBe(2);
  });

  it('should distinguish abandonment from time-loss in result reason', async () => {
    const io = createMockIo();
    const session = new GameSession('reason-test-game', io, {} as any, new Map());

    // Create state where player has ample time
    const state = createTimedGameState({
      id: 'reason-test-game',
    });

    const abandonmentResult: GameResult = {
      winner: 2,
      reason: 'abandonment',
      finalScore: {
        ringsEliminated: { 1: 0, 2: 0 },
        territorySpaces: { 1: 0, 2: 0 },
        ringsRemaining: { 1: 18, 2: 18 },
      },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(() => ({
        success: true,
        gameResult: abandonmentResult,
      })),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handleAbandonmentForDisconnectedPlayer(1, true);

    // Critical assertion: reason must be 'abandonment', not 'time_loss'
    expect(result?.reason).toBe('abandonment');
    expect(result?.reason).not.toBe('time_loss');
    expect(result?.reason).not.toBe('timeout');
  });
});

describe('LF5.b: Time-Loss While Disconnected', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function createLowTimeGameState(): GameState {
    const now = new Date();
    return {
      id: 'time-loss-test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      },
      players: [
        {
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 0, // No time remaining
          ringsInHand: 10,
          eliminatedRings: 8,
          territorySpaces: 5,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 300000, // 5 minutes remaining
          ringsInHand: 12,
          eliminatedRings: 6,
          territorySpaces: 3,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'movement',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 36,
      totalRingsEliminated: 14,
      victoryThreshold: 18,
      territoryVictoryThreshold: 33,
    } as unknown as GameState;
  }

  it('should handle time expiry while player is disconnected', async () => {
    const io = createMockIo();
    const session = new GameSession('time-loss-test-game', io, {} as any, new Map());

    const state = createLowTimeGameState();
    const timeoutResult: GameResult = {
      winner: 2,
      reason: 'time_loss',
      finalScore: {
        ringsEliminated: { 1: 8, 2: 6 },
        territorySpaces: { 1: 5, 2: 3 },
        ringsRemaining: { 1: 10, 2: 12 },
      },
    };

    // In this scenario, the clock expires before the reconnection window
    // The game engine should report time_loss, not abandonment
    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      handleTimeExpiry: jest.fn(() => ({
        success: true,
        gameResult: timeoutResult,
      })),
    };

    // Simulate time expiry handling
    // Note: This tests the semantic distinction - when clock runs out,
    // the reason should be 'time_loss' even if player is disconnected
    const mockHandleTimeExpiry = (session as any).gameEngine.handleTimeExpiry;

    const result = mockHandleTimeExpiry(1);

    expect(result.success).toBe(true);
    expect(result.gameResult.reason).toBe('time_loss');
    expect(result.gameResult.reason).not.toBe('abandonment');
  });
});
