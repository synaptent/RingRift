import { GameSession } from '../../src/server/game/GameSession';

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => null),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({})),
}));

/**
 * Focused unit tests for the derived GameSessionStatus projection used by
 * GameSession. These tests avoid touching the real database or GameEngine
 * by stubbing getGameState and calling the internal recomputeSessionStatus
 * helper via `any`.
 */

describe('GameSessionStatus projection', () => {
  const makeSession = () => {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;
    const userSockets = new Map<string, string>();

    return new GameSession('game-1', io, pythonClient, userSockets) as any;
  };

  it('projects waiting GameState to waiting_for_players session status', () => {
    const session = makeSession();

    const waitingState: any = {
      id: 'game-1',
      gameStatus: 'waiting',
      boardType: 'square8' as any,
      currentPlayer: 1,
      currentPhase: 'ring_placement' as any,
      players: [],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 1234,
      board: {} as any,
    };

    session.gameEngine = {
      getGameState: jest.fn(() => waitingState),
    };

    // Let recomputeSessionStatus pull from gameEngine.getGameState
    session.recomputeSessionStatus();

    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
    if (status) {
      expect(status.kind).toBe('waiting_for_players');
      expect(status.gameId).toBe('game-1');
    }
  });

  it('projects active GameState to active_turn session status', () => {
    const session = makeSession();

    const activeState: any = {
      id: 'game-1',
      gameStatus: 'active',
      boardType: 'square8' as any,
      currentPlayer: 2,
      currentPhase: 'ring_placement' as any,
      players: [
        {
          id: 'p1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as any,
        {
          id: 'p2',
          username: 'Player 2',
          playerNumber: 2,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as any,
      ],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 5678,
      board: {} as any,
    };

    // Exercise the overload that takes an explicit GameState argument
    session.recomputeSessionStatus(activeState);

    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
    if (status && status.kind === 'active_turn') {
      expect(status.gameId).toBe('game-1');
      expect(status.currentPlayer).toBe(2);
      expect(status.phase).toBe(activeState.currentPhase);
    }
  });

  it('projects abandoned GameState + GameResult to abandoned session status with result snapshot', () => {
    const session = makeSession();

    const abandonedState: any = {
      id: 'game-1',
      gameStatus: 'abandoned',
      boardType: 'square8' as any,
      currentPlayer: 1,
      currentPhase: 'ring_placement' as any,
      players: [],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 9999,
      board: {} as any,
    };

    const result: any = {
      reason: 'abandonment',
      finalScore: {
        ringsEliminated: { 1: 0 },
        territorySpaces: { 1: 0 },
        ringsRemaining: { 1: 0 },
      },
    };

    session.recomputeSessionStatus(abandonedState, result);

    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
    if (status && status.kind === 'abandoned') {
      expect(status.gameId).toBe('game-1');
      expect(status.status).toBe('abandoned');
      expect(status.result).toEqual(result);
    }
  });
});
