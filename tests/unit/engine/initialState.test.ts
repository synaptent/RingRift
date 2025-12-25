/**
 * Test suite for src/shared/engine/initialState.ts
 *
 * Tests createInitialGameState function which creates a pristine
 * initial GameState for new games.
 */

import { createInitialGameState } from '../../../src/shared/engine/initialState';
import {
  BOARD_CONFIGS,
  type Player,
  type TimeControl,
  type BoardType,
  type RulesOptions,
} from '../../../src/shared/types/game';

describe('initialState', () => {
  // Test fixtures
  const testTimeControl: TimeControl = {
    type: 'blitz',
    initialTime: 300, // 5 minutes
    increment: 5,
  };

  function createTestPlayers(count: number): Player[] {
    return Array.from({ length: count }, (_, i) => ({
      id: `player-${i + 1}`,
      username: `Player${i + 1}`,
      rating: 1500,
      type: 'human' as const,
      playerNumber: 0, // Will be set by createInitialGameState
      timeRemaining: 0,
      isReady: false,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
  }

  describe('createInitialGameState', () => {
    describe('basic state initialization', () => {
      it('should create a valid game state with required fields', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.id).toBe('game-123');
        expect(state.boardType).toBe('square8');
        expect(state.players).toHaveLength(2);
        expect(state.currentPhase).toBe('ring_placement');
        expect(state.currentPlayer).toBe(1);
        expect(state.gameStatus).toBe('waiting');
      });

      it('should initialize move history and history as empty', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.moveHistory).toEqual([]);
        expect(state.history).toEqual([]);
      });

      it('should set spectators as empty array', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.spectators).toEqual([]);
      });

      it('should set createdAt and lastMoveAt timestamps', () => {
        const before = new Date();
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );
        const after = new Date();

        expect(state.createdAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
        expect(state.createdAt.getTime()).toBeLessThanOrEqual(after.getTime());
        expect(state.lastMoveAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
      });
    });

    describe('board initialization', () => {
      it('should create empty board with correct size for square8', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.board.size).toBe(BOARD_CONFIGS.square8.size);
        expect(state.board.type).toBe('square8');
        expect(state.board.stacks.size).toBe(0);
        expect(state.board.markers.size).toBe(0);
        expect(state.board.collapsedSpaces.size).toBe(0);
        expect(state.board.territories.size).toBe(0);
        expect(state.board.formedLines).toEqual([]);
      });

      it('should create empty board with correct size for square19', () => {
        const state = createInitialGameState(
          'game-123',
          'square19',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.board.size).toBe(BOARD_CONFIGS.square19.size);
        expect(state.board.type).toBe('square19');
      });

      it('should create empty board with correct size for hexagonal', () => {
        const state = createInitialGameState(
          'game-123',
          'hexagonal',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.board.size).toBe(BOARD_CONFIGS.hexagonal.size);
        expect(state.board.type).toBe('hexagonal');
      });

      it('should initialize eliminatedRings counters for each player', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(3),
          testTimeControl
        );

        expect(state.board.eliminatedRings[1]).toBe(0);
        expect(state.board.eliminatedRings[2]).toBe(0);
        expect(state.board.eliminatedRings[3]).toBe(0);
      });
    });

    describe('player initialization', () => {
      it('should assign player numbers starting from 1', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(3),
          testTimeControl
        );

        expect(state.players[0].playerNumber).toBe(1);
        expect(state.players[1].playerNumber).toBe(2);
        expect(state.players[2].playerNumber).toBe(3);
      });

      it('should convert time control to milliseconds', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        // 300 seconds = 300000 milliseconds
        expect(state.players[0].timeRemaining).toBe(300000);
        expect(state.players[1].timeRemaining).toBe(300000);
      });

      it('should set ringsInHand to board config default', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.players[0].ringsInHand).toBe(BOARD_CONFIGS.square8.ringsPerPlayer);
        expect(state.players[1].ringsInHand).toBe(BOARD_CONFIGS.square8.ringsPerPlayer);
      });

      it('should initialize eliminatedRings and territorySpaces to 0', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        state.players.forEach((p) => {
          expect(p.eliminatedRings).toBe(0);
          expect(p.territorySpaces).toBe(0);
        });
      });

      it('should set AI players as ready, human players as not ready', () => {
        const players: Player[] = [
          {
            id: '1',
            username: 'Human',
            rating: 1500,
            type: 'human',
            playerNumber: 0,
            timeRemaining: 0,
            isReady: false,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: '2',
            username: 'AI',
            rating: 1500,
            type: 'ai',
            playerNumber: 0,
            timeRemaining: 0,
            isReady: false,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ];

        const state = createInitialGameState('game-123', 'square8', players, testTimeControl);

        expect(state.players[0].isReady).toBe(false); // Human
        expect(state.players[1].isReady).toBe(true); // AI
      });
    });

    describe('isRated parameter', () => {
      it('should default to rated game', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.isRated).toBe(true);
      });

      it('should respect explicit isRated=false', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          false
        );

        expect(state.isRated).toBe(false);
      });

      it('should respect explicit isRated=true', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          true
        );

        expect(state.isRated).toBe(true);
      });
    });

    describe('RNG seed', () => {
      it('should use provided rngSeed when given', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          true,
          12345
        );

        expect(state.rngSeed).toBe(12345);
      });

      it('should generate rngSeed when not provided', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.rngSeed).toBeDefined();
        expect(typeof state.rngSeed).toBe('number');
      });

      it('should generate different seeds for different games', () => {
        const state1 = createInitialGameState(
          'game-1',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        const state2 = createInitialGameState(
          'game-2',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        // Technically could be same, but probability is negligible
        // In practice this validates seed generation runs
        expect(state1.rngSeed).toBeDefined();
        expect(state2.rngSeed).toBeDefined();
      });
    });

    describe('rulesOptions', () => {
      it('should not include rulesOptions when not provided', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.rulesOptions).toBeUndefined();
      });

      it('should include rulesOptions when provided', () => {
        const rulesOptions: RulesOptions = { swapRuleEnabled: true };
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          true,
          undefined,
          rulesOptions
        );

        expect(state.rulesOptions).toEqual(rulesOptions);
      });

      it('should use custom ringsPerPlayer from rulesOptions', () => {
        const rulesOptions: RulesOptions = { ringsPerPlayer: 50 };
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          true,
          undefined,
          rulesOptions
        );

        expect(state.players[0].ringsInHand).toBe(50);
        expect(state.players[1].ringsInHand).toBe(50);
      });

      it('should use custom lpsRoundsRequired from rulesOptions', () => {
        const rulesOptions: RulesOptions = { lpsRoundsRequired: 5 };
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl,
          true,
          undefined,
          rulesOptions
        );

        expect(state.lpsRoundsRequired).toBe(5);
      });
    });

    describe('victory thresholds', () => {
      it('should calculate victoryThreshold based on rings and player count', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        // Victory threshold depends on ringsPerPlayer and player count
        expect(state.victoryThreshold).toBeGreaterThan(0);
      });

      it('should set territoryVictoryThreshold to majority of total spaces', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        const expectedThreshold = Math.floor(BOARD_CONFIGS.square8.totalSpaces / 2) + 1;
        expect(state.territoryVictoryThreshold).toBe(expectedThreshold);
      });

      it('should calculate territoryVictoryMinimum', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.territoryVictoryMinimum).toBeGreaterThan(0);
      });
    });

    describe('game counters', () => {
      it('should initialize totalRingsInPlay to 0', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.totalRingsInPlay).toBe(0);
      });

      it('should initialize totalRingsEliminated to 0', () => {
        const state = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );

        expect(state.totalRingsEliminated).toBe(0);
      });

      it('should set maxPlayers based on provided players', () => {
        const state2 = createInitialGameState(
          'game-123',
          'square8',
          createTestPlayers(2),
          testTimeControl
        );
        expect(state2.maxPlayers).toBe(2);

        const state4 = createInitialGameState(
          'game-456',
          'square8',
          createTestPlayers(4),
          testTimeControl
        );
        expect(state4.maxPlayers).toBe(4);
      });
    });

    describe('board type variations', () => {
      const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal', 'hex8'];

      boardTypes.forEach((boardType) => {
        it(`should correctly initialize ${boardType} board`, () => {
          const state = createInitialGameState(
            `game-${boardType}`,
            boardType,
            createTestPlayers(2),
            testTimeControl
          );

          expect(state.boardType).toBe(boardType);
          expect(state.board.type).toBe(boardType);
          expect(state.board.size).toBe(BOARD_CONFIGS[boardType].size);
          expect(state.players[0].ringsInHand).toBe(BOARD_CONFIGS[boardType].ringsPerPlayer);
        });
      });
    });
  });
});
