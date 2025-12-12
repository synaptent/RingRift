/**
 * InitialState Unit Tests
 *
 * Tests for createInitialGameState function that creates pristine game states.
 */

import { createInitialGameState } from '../../src/shared/engine/initialState';
import type { Player, TimeControl, BoardType, RulesOptions } from '../../src/shared/types/game';

describe('initialState', () => {
  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 0, // Will be overwritten
      isReady: false,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 0,
      isReady: false,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const timeControl: TimeControl = {
    initialTime: 600, // 10 minutes
    increment: 5,
    type: 'rapid',
  };

  describe('createInitialGameState', () => {
    it('should create a game state with correct gameId', () => {
      const state = createInitialGameState('game-123', 'square8', createPlayers(), timeControl);

      expect(state.id).toBe('game-123');
    });

    it('should set boardType correctly', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.boardType).toBe('square8');
      expect(state.board.type).toBe('square8');
    });

    it('should initialize board size from config', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.board.size).toBe(8);
    });

    it('should initialize empty board structures', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.board.stacks.size).toBe(0);
      expect(state.board.markers.size).toBe(0);
      expect(state.board.collapsedSpaces.size).toBe(0);
      expect(state.board.territories.size).toBe(0);
      expect(state.board.formedLines).toHaveLength(0);
    });

    it('should initialize eliminated rings to 0 for each player', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.board.eliminatedRings[1]).toBe(0);
      expect(state.board.eliminatedRings[2]).toBe(0);
    });

    it('should assign player numbers starting from 1', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.players[0].playerNumber).toBe(1);
      expect(state.players[1].playerNumber).toBe(2);
    });

    it('should initialize player time remaining from timeControl', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      // 600 seconds * 1000 = 600000 milliseconds
      expect(state.players[0].timeRemaining).toBe(600000);
      expect(state.players[1].timeRemaining).toBe(600000);
    });

    it('should set human players as not ready', () => {
      const players = createPlayers();
      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players[0].isReady).toBe(false);
      expect(state.players[1].isReady).toBe(false);
    });

    it('should set AI players as ready', () => {
      const players: Player[] = [
        { ...createPlayers()[0], type: 'ai' },
        { ...createPlayers()[1], type: 'ai' },
      ];
      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players[0].isReady).toBe(true);
      expect(state.players[1].isReady).toBe(true);
    });

    it('should initialize ringsInHand from config', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      // square8 has 18 rings per player
      expect(state.players[0].ringsInHand).toBe(18);
      expect(state.players[1].ringsInHand).toBe(18);
    });

    it('should set initial player state values', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.players[0].eliminatedRings).toBe(0);
      expect(state.players[0].territorySpaces).toBe(0);
    });

    it('should set game to ring_placement phase', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.currentPhase).toBe('ring_placement');
    });

    it('should set player 1 as current player', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.currentPlayer).toBe(1);
    });

    it('should initialize empty move history', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.moveHistory).toEqual([]);
      expect(state.history).toEqual([]);
    });

    it('should set game status to waiting', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.gameStatus).toBe('waiting');
    });

    it('should initialize empty spectators array', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.spectators).toEqual([]);
    });

    it('should set timeControl correctly', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.timeControl).toEqual(timeControl);
    });

    it('should set createdAt and lastMoveAt to current time', () => {
      const before = new Date();
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);
      const after = new Date();

      expect(state.createdAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
      expect(state.createdAt.getTime()).toBeLessThanOrEqual(after.getTime());
      expect(state.lastMoveAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
    });

    it('should set isRated to true by default', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.isRated).toBe(true);
    });

    it('should set isRated to false when specified', () => {
      const state = createInitialGameState(
        'game-1',
        'square8',
        createPlayers(),
        timeControl,
        false
      );

      expect(state.isRated).toBe(false);
    });

    it('should generate rngSeed when not provided', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.rngSeed).toBeDefined();
      expect(typeof state.rngSeed).toBe('number');
    });

    it('should use provided rngSeed', () => {
      const seed = 12345;
      const state = createInitialGameState(
        'game-1',
        'square8',
        createPlayers(),
        timeControl,
        true,
        seed
      );

      expect(state.rngSeed).toBe(12345);
    });

    it('should not include rulesOptions when not provided', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.rulesOptions).toBeUndefined();
    });

    it('should include rulesOptions when provided', () => {
      const rulesOptions: RulesOptions = { swapRuleEnabled: true };
      const state = createInitialGameState(
        'game-1',
        'square8',
        createPlayers(),
        timeControl,
        true,
        undefined,
        rulesOptions
      );

      expect(state.rulesOptions).toEqual({ swapRuleEnabled: true });
    });

    it('should set maxPlayers to number of players', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.maxPlayers).toBe(2);
    });

    it('should initialize totalRingsInPlay to 0', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.totalRingsInPlay).toBe(0);
    });

    it('should initialize totalRingsEliminated to 0', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      expect(state.totalRingsEliminated).toBe(0);
    });

    it('should calculate victoryThreshold correctly', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      // Canonical rule RR-CANON-R061: for 2-player games, victoryThreshold = ringsPerPlayer = 18
      expect(state.victoryThreshold).toBe(18);
    });

    it('should calculate territoryVictoryThreshold correctly', () => {
      const state = createInitialGameState('game-1', 'square8', createPlayers(), timeControl);

      // 64 total spaces, floor(64/2) + 1 = 33
      expect(state.territoryVictoryThreshold).toBe(33);
    });

    it('should handle 3 players', () => {
      const players: Player[] = [
        ...createPlayers(),
        {
          id: 'p3',
          username: 'Player3',
          type: 'human',
          playerNumber: 0,
          isReady: false,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players).toHaveLength(3);
      expect(state.players[2].playerNumber).toBe(3);
      expect(state.maxPlayers).toBe(3);
      // Per RR-CANON-R061: round(18 * (2/3 + 1/3 * 2)) = round(18 * 4/3) = 24
      expect(state.victoryThreshold).toBe(24);
    });

    it('should handle 4 players', () => {
      const players: Player[] = [
        ...createPlayers(),
        {
          id: 'p3',
          username: 'Player3',
          type: 'human',
          playerNumber: 0,
          isReady: false,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p4',
          username: 'Player4',
          type: 'human',
          playerNumber: 0,
          isReady: false,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players).toHaveLength(4);
      expect(state.players[3].playerNumber).toBe(4);
      expect(state.maxPlayers).toBe(4);
      // Per RR-CANON-R061: round(18 * (2/3 + 1/3 * 3)) = round(18 * 5/3) = 30
      expect(state.victoryThreshold).toBe(30);
    });

    it('should handle mixed human/AI players', () => {
      const players: Player[] = [
        { ...createPlayers()[0], type: 'human' },
        { ...createPlayers()[1], type: 'ai' },
      ];
      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players[0].isReady).toBe(false); // Human
      expect(state.players[1].isReady).toBe(true); // AI
    });

    it('should preserve player properties from input', () => {
      const players = createPlayers();
      players[0].rating = 1500;
      players[1].aiDifficulty = 3;

      const state = createInitialGameState('game-1', 'square8', players, timeControl);

      expect(state.players[0].rating).toBe(1500);
      expect(state.players[1].aiDifficulty).toBe(3);
    });
  });
});
